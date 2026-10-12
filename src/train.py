"""Bucle propio de entrenamiento, checkpoints y evaluación reproducible."""

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torchvision
from sklearn.metrics import classification_report, f1_score
from torch import nn
from tqdm.auto import tqdm

from src.utils import CLASSES, find_dataset, make_loaders
from src.models import build_model, parameter_counts
from src.utils import export_results, plot_confusion, plot_history, seed_everything, write_json


# Hiperparámetros fijos de la práctica; se pueden editar aquí.
DEFAULT_SETTINGS = {'seed': 42,
 'validation_fraction': 0.2,
 'batch_size': 64,
 'num_workers': 2,
 'scratch_epochs': 10,
 'transfer_epochs': 5,
 'scratch_lr': 0.01,
 'high_lr': 0.1,
 'momentum': 0.9,
 'weight_decay': 0.0001,
 'transfer_lrs': {'feature': 0.001, 'partial': 0.0001, 'full': 1e-05},
 'scratch_size': 64,
 'transfer_size': 224,
 'augmentation': True}


def experiment_plan(settings, group="all"):
    """Cuatro modelos base, dos controles LR×10 de LeNet y tres estrategias de transferencia."""
    if group not in {"all", "scratch", "high_lr", "transfer"}:
        raise ValueError(f"Grupo desconocido: {group}")
    common = {k: settings[k] for k in ("seed", "batch_size", "num_workers", "weight_decay", "augmentation")}
    plan = []
    for experiment_group, lr in (("scratch", settings["scratch_lr"]), ("high_lr", settings["high_lr"])):
        architectures = ("lenet", "vgg11") if experiment_group == "scratch" else ("lenet",)
        for architecture in architectures:
            for bn in (False, True):
                plan.append({**common, "name": f"{architecture}{'_bn' if bn else ''}_{experiment_group}_seed{settings['seed']}",
                             "architecture": architecture, "batch_norm": bn, "strategy": None,
                             "group": experiment_group, "lr": lr, "optimizer": "sgd",
                             "momentum": settings["momentum"], "epochs": settings["scratch_epochs"],
                             "size": settings["scratch_size"]})
    for strategy, lr in settings["transfer_lrs"].items():
        plan.append({**common, "name": f"resnet18_{strategy}_seed{settings['seed']}",
                     "architecture": "resnet18", "batch_norm": False, "strategy": strategy,
                     "group": "transfer", "lr": lr, "optimizer": "adam", "momentum": 0,
                     "epochs": settings["transfer_epochs"], "size": settings["transfer_size"]})
    return [item for item in plan if group == "all" or item["group"] == group]


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0
    truths, predictions = [], []
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Pérdida no finita (NaN/Inf)")
            if training:
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError("Gradientes no finitos (NaN/Inf)")
                optimizer.step()
            predicted = logits.argmax(1)
            total_loss += loss.item() * len(labels)
            correct += int((predicted == labels).sum().item())
            total += len(labels)
            if not training:
                truths.extend(labels.cpu().tolist())
                predictions.extend(predicted.cpu().tolist())
    if not total:
        raise ValueError("Partición vacía")
    return {"loss": total_loss / total, "accuracy": correct / total,
            "y_true": truths, "y_pred": predictions}


def source_digest():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def train_experiment(config, data_root="data/raw", manifest="data/splits/manifest.csv", output="results", device="auto"):
    if config["epochs"] < 1 or config["batch_size"] < 2 or config["lr"] <= 0:
        raise ValueError("Se requieren epochs>=1, batch_size>=2 y lr>0")
    root = find_dataset(data_root)
    manifest = Path(manifest)
    if not manifest.is_file():
        raise FileNotFoundError("Falta manifest.csv. Ejecuta primero el notebook 01 o src.utils prepare.")
    output = Path(output)
    run = output / "runs" / config["name"]
    run.mkdir(parents=True, exist_ok=True)
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    identity = {"config": config, "manifest_sha256": manifest_hash, "source_sha256": source_digest()}
    metadata = run / "identity.json"
    if metadata.exists():
        previous = json.loads(metadata.read_text(encoding="utf-8"))
        if previous != identity:
            raise ValueError(f"{run} pertenece a otro código/configuración/split. Usa otro --output.")
        if (run / "metrics.json").exists():
            metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
            if metrics["status"] == "completed":
                print(f"Ya completado: {config['name']}; se conservan sus resultados.")
                return metrics
    write_json(metadata, identity)
    seed_everything(config["seed"])
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device)
    loaders = make_loaders(root, manifest, config)
    # Los experimentos reales de transferencia siempre cargan pesos ImageNet.
    model = build_model(config["architecture"], config["batch_norm"], config["strategy"], pretrained=True).to(device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(parameters, lr=config["lr"], momentum=config["momentum"], weight_decay=config["weight_decay"])
    else:
        optimizer = torch.optim.Adam(parameters, lr=config["lr"], weight_decay=config["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    environment = {"python": platform.python_version(), "torch": torch.__version__,
                   "torchvision": torchvision.__version__, "device": str(device),
                   "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
                   "classes": CLASSES, "data_root": str(root)}
    write_json(run / "environment.json", environment)
    frozen = subprocess.run([sys.executable, "-m", "pip", "freeze"], text=True, capture_output=True)
    if frozen.returncode == 0:
        (run / "environment.txt").write_text(frozen.stdout, encoding="utf-8")
    history, best_accuracy, best_loss, best_epoch = [], -1.0, math.inf, None
    status, error, failed_epoch = "completed", None, None
    for epoch in tqdm(range(1, config["epochs"] + 1), desc=config["name"]):
        try:
            synchronize(device)
            started = time.perf_counter()
            train = run_epoch(model, loaders["train"], criterion, device, optimizer)
            synchronize(device)
            train_seconds = time.perf_counter() - started
            started = time.perf_counter()
            validation = run_epoch(model, loaders["val"], criterion, device)
            synchronize(device)
            val_seconds = time.perf_counter() - started
        except FloatingPointError as failure:
            status, error, failed_epoch = "unstable", str(failure), epoch
            break
        history.append({"epoch": epoch, "train_loss": train["loss"], "train_accuracy": train["accuracy"],
                        "val_loss": validation["loss"], "val_accuracy": validation["accuracy"],
                        "train_seconds": train_seconds, "val_seconds": val_seconds, "lr": config["lr"]})
        pd.DataFrame(history).to_csv(run / "history.csv", index=False)
        if (validation["accuracy"], -validation["loss"]) > (best_accuracy, -best_loss):
            best_accuracy, best_loss, best_epoch = validation["accuracy"], validation["loss"], epoch
            torch.save({"model_state": model.state_dict(), "config": config, "classes": CLASSES,
                        "epoch": epoch, "val_accuracy": best_accuracy, "manifest_sha256": manifest_hash}, run / "best.pt")
        torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                    "config": config, "epoch": epoch}, run / "last.pt")
    metrics = {**config, **parameter_counts(model), "status": status, "error": error,
               "failed_epoch": failed_epoch, "completed_epochs": len(history), "best_epoch": best_epoch,
               "best_val_accuracy": best_accuracy if history else None,
               "best_val_loss": best_loss if history else None,
               "mean_train_seconds": sum(row["train_seconds"] for row in history) / len(history) if history else None,
               "epoch_to_80_val": next((row["epoch"] for row in history if row["val_accuracy"] >= 0.8), None),
               "test_accuracy": None, "test_loss": None, "test_f1_macro": None}
    if status == "completed":
        checkpoint = torch.load(run / "best.pt", map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state"])
        test = run_epoch(model, loaders["test"], criterion, device)
        metrics.update(test_accuracy=test["accuracy"], test_loss=test["loss"],
                       test_f1_macro=f1_score(test["y_true"], test["y_pred"], labels=list(range(4)), average="macro", zero_division=0))
        test_paths = pd.read_csv(manifest).query("split == 'test'").path.tolist()
        pd.DataFrame({"path": test_paths, "y_true": test["y_true"], "y_pred": test["y_pred"]}).to_csv(run / "predictions.csv", index=False)
        report = classification_report(test["y_true"], test["y_pred"], labels=list(range(4)),
                                       target_names=CLASSES, output_dict=True, zero_division=0)
        write_json(run / "classification_report.json", report)
        plot_confusion(test["y_true"], test["y_pred"], output / "figures" / f"{config['name']}_confusion.png", config["name"])
    if history:
        plot_history(pd.DataFrame(history), config["name"], output / "figures" / f"{config['name']}_curves.png")
    write_json(run / "metrics.json", metrics)
    print(f"{config['name']}: {status}; mejor val={metrics['best_val_accuracy']}; test={metrics['test_accuracy']}")
    return metrics


def run_suite(settings, group="all", data_root="data/raw", manifest="data/splits/manifest.csv", output="results", device="auto"):
    results = []
    for config in experiment_plan(settings, group):
        results.append(train_experiment(config, data_root, manifest, output, device))
    export_results(output)
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=["all", "scratch", "high_lr", "transfer"], default="all")
    parser.add_argument("--data", default="data/raw")
    parser.add_argument("--manifest", default="data/splits/manifest.csv")
    parser.add_argument("--output", default="results")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--update-readme", action="store_true")
    args = parser.parse_args()
    if not args.export_only:
        settings = DEFAULT_SETTINGS
        run_suite(settings, args.group, args.data, args.manifest, args.output, args.device)
    print(export_results(args.output, "README.md" if args.update_readme else None).to_string(index=False))


if __name__ == "__main__":
    main()

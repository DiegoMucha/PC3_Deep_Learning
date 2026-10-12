"""Datos, reproducibilidad, gráficas y exportación de resultados."""

import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay

import argparse
import hashlib
import shutil
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


CLASSES = ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_dataset(root):
    """Localiza TRAIN/TEST sin confundir TEST_SIMPLE ni duplicar árboles anidados."""
    root = Path(root).expanduser().resolve()
    candidates = []
    for train in sorted(root.rglob("TRAIN")):
        if train.is_dir() and (train.parent / "TEST").is_dir():
            if all((train / c).is_dir() and (train.parent / "TEST" / c).is_dir() for c in CLASSES):
                candidates.append(train.parent)
    if len(candidates) != 1:
        raise ValueError(
            f"Se esperaba un único árbol con TRAIN/TEST y las cuatro clases en {root}; "
            f"se encontraron {len(candidates)}. Indica la carpeta images que corresponda."
        )
    return candidates[0]


def download_data(destination="data/raw"):
    destination = Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()):
        found = find_dataset(destination)
        print(f"Dataset ya disponible: {found}")
        return found
    import kagglehub
    cached = Path(kagglehub.dataset_download("paultimothymooney/blood-cells"))
    source = find_dataset(cached)
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination / "images", dirs_exist_ok=False)
    print(f"Dataset descargado: {destination / 'images'}")
    return destination / "images"


def inventory(root, hash_images=True):
    root = find_dataset(root)
    rows = []
    for partition in ("TRAIN", "TEST"):
        for label, name in enumerate(CLASSES):
            paths = sorted(p for p in (root / partition / name).rglob("*") if p.suffix.lower() in EXTENSIONS)
            if not paths:
                raise ValueError(f"No hay imágenes en {partition}/{name}")
            for path in paths:
                try:
                    with Image.open(path) as original:
                        original.load()
                        rgb = original.convert("RGB")
                        digest = hashlib.sha256(
                            f"{rgb.size}".encode() + rgb.tobytes()
                        ).hexdigest() if hash_images else ""
                        width, height = rgb.size
                except Exception as error:
                    raise ValueError(f"Imagen inválida: {path}") from error
                rows.append({"path": path.relative_to(root).as_posix(), "class": name,
                             "label": label, "source_split": partition, "width": width,
                             "height": height, "sha256": digest})
    return pd.DataFrame(rows)


def make_manifest(root, output="data/splits", seed=42, validation_fraction=0.2):
    """Agrupa duplicados exactos para que no crucen train/val; conserva TEST oficial."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction debe estar entre 0 y 1")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame = inventory(root)
    if frame.groupby("sha256").label.nunique().gt(1).any():
        raise ValueError("Hay imágenes idénticas con etiquetas contradictorias; revisar el dataset.")
    training = frame.loc[frame.source_split == "TRAIN"]
    groups = training.drop_duplicates("sha256")
    _, validation = train_test_split(
        groups, test_size=validation_fraction, stratify=groups.label, random_state=seed,
    )
    frame["split"] = "train"
    frame.loc[frame.sha256.isin(validation.sha256), "split"] = "val"
    frame.loc[frame.source_split == "TEST", "split"] = "test"
    train_hashes = set(training.sha256)
    test_hashes = set(frame.loc[frame.source_split == "TEST", "sha256"])
    overlap = train_hashes & test_hashes
    audit = {
        "seed": seed, "validation_fraction": validation_fraction,
        "classes": CLASSES, "rows": len(frame),
        "duplicate_images": int(frame.duplicated("sha256").sum()),
        "train_test_shared_hashes": len(overlap),
        "limitation": "No hay agrupación por paciente ni detección de aumentos de una misma célula.",
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    frame.to_csv(output / "manifest.csv", index=False)
    if overlap:
        print("AVISO: hay duplicados exactos entre TRAIN y TEST. Consulta audit.json; "
              "no interpretes las métricas como generalización independiente.")
    return frame


class BloodCells(Dataset):
    def __init__(self, root, frame, transform):
        self.root = Path(root)
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(self.root / row.path) as image:
            x = self.transform(image.convert("RGB"))
        return x, int(row.label)


def get_transforms(size=64, training=False, transfer=False, augmentation=True):
    operations = [transforms.Resize((size, size))]
    if training and augmentation:
        operations += [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                       transforms.RandomRotation(15)]
    operations.append(transforms.ToTensor())
    mean, std = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) if transfer else ((0.5,) * 3, (0.5,) * 3)
    operations.append(transforms.Normalize(mean, std))
    return transforms.Compose(operations)


def seed_worker(_worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def make_loaders(root, manifest, config):
    frame = pd.read_csv(manifest)
    if set(frame["class"]) != set(CLASSES) or set(frame.split) != {"train", "val", "test"}:
        raise ValueError("Manifest incompleto: se requieren cuatro clases y train/val/test")
    loaders = {}
    for split in ("train", "val", "test"):
        subset = frame.loc[frame.split == split]
        dataset = BloodCells(root, subset, get_transforms(
            size=config["size"], training=split == "train",
            transfer=config["architecture"] == "resnet18", augmentation=config["augmentation"],
        ))
        loaders[split] = DataLoader(
            dataset, batch_size=config["batch_size"], shuffle=split == "train",
            num_workers=config["num_workers"], worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(config["seed"]),
            pin_memory=torch.cuda.is_available(),
        )
    return loaders



def seed_everything(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def save_figure(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_history(history, title, output):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for axis, metric, label in zip(axes, ("loss", "accuracy"), ("Entropía cruzada", "Accuracy")):
        for split, name in (("train", "Entrenamiento"), ("val", "Validación")):
            axis.plot(history.epoch, history[f"{split}_{metric}"], label=name)
        axis.set(xlabel="Época", ylabel=label, title=f"{title}: {label}")
        axis.legend()
        axis.grid(alpha=0.25)
    axes[1].axhline(0.8, color="gray", linestyle="--", label="Umbral 80%")
    axes[1].set_ylim(0, 1.02)
    axes[1].legend()
    save_figure(fig, output)


def plot_confusion(y_true, y_pred, output, title="Matriz de confusión — test"):
    fig, axis = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=list(range(4)), display_labels=CLASSES,
        ax=axis, cmap="Blues", colorbar=False, xticks_rotation=25,
    )
    axis.set(title=title, xlabel="Clase predicha", ylabel="Clase real")
    save_figure(fig, output)


def markdown_table(frame):
    """Evita una dependencia adicional de tabulate."""
    def format_value(value):
        if pd.isna(value):
            return "—"
        if isinstance(value, (float, np.floating)):
            return f"{value:.4f}"
        return str(value)
    rows = ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    rows += ["| " + " | ".join(format_value(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join(rows)


def collect_results(output="results"):
    paths = sorted((Path(output) / "runs").glob("*/metrics.json"))
    if not paths:
        raise FileNotFoundError("No hay resultados. Ejecuta los experimentos en Colab primero.")
    return pd.DataFrame([json.loads(path.read_text(encoding="utf-8")) for path in paths])


def plot_bn_comparison(output, architecture, group, seed):
    output = Path(output)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    found = False
    for bn in (False, True):
        name = f"{architecture}{'_bn' if bn else ''}_{group}_seed{seed}"
        history_path = output / "runs" / name / "history.csv"
        if not history_path.exists():
            continue
        found = True
        history = pd.read_csv(history_path)
        for axis, split, label in zip(axes, ("train", "val"), ("Entrenamiento", "Validación")):
            axis.plot(history.epoch, history[f"{split}_loss"], label="Con BN" if bn else "Sin BN")
            axis.set(title=f"{architecture} / {group}: {label}", xlabel="Época", ylabel="Entropía cruzada")
            axis.legend()
            axis.grid(alpha=0.25)
    if found:
        save_figure(fig, output / "figures" / f"bn_{architecture}_{group}_seed{seed}.png")
    else:
        plt.close(fig)


def export_results(output="results", readme=None):
    """Selecciona por validación; TEST se usa solo para reportar rendimiento final."""
    output = Path(output)
    frame = collect_results(output)
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    frame.to_csv(tables / "summary.csv", index=False)
    columns = ["name", "parameters", "trainable_parameters", "mean_train_seconds",
               "best_val_accuracy", "test_accuracy", "test_f1_macro", "epoch_to_80_val", "status"]
    for group in ("scratch", "high_lr", "transfer"):
        frame.loc[frame.group == group, columns].to_csv(tables / f"{group}.csv", index=False)
    candidates = frame.loc[(frame.status == "completed") & frame.group.isin(["scratch", "transfer"])]
    comparisons = []
    # No mezclar modelos de semillas distintas para declarar el ganador de una repetición.
    for seed in sorted(frame.seed.unique()):
        seed_frame = candidates.loc[candidates.seed == seed]
        scratch = seed_frame.loc[seed_frame.group == "scratch"]
        if not scratch.empty:
            best_scratch = scratch.sort_values(["best_val_accuracy", "best_val_loss"], ascending=[False, True]).iloc[0]
            comparisons.append(best_scratch.to_dict())
        comparisons.extend(seed_frame.loc[seed_frame.group == "transfer"].to_dict("records"))
        if not seed_frame.empty:
            best = seed_frame.sort_values(["best_val_accuracy", "best_val_loss"], ascending=[False, True]).iloc[0]
            predictions = pd.read_csv(output / "runs" / best["name"] / "predictions.csv")
            plot_confusion(predictions.y_true, predictions.y_pred,
                           output / "figures" / f"best_confusion_seed{seed}.png",
                           f"Mejor por validación: {best['name']}")
            write_json(tables / f"best_model_seed{seed}.json", {
                "name": best["name"], "selection": "Mayor accuracy de validación; desempate por menor pérdida",
                "all_seven_completed": len(seed_frame) == 7,
            })
        for architecture in ("lenet", "vgg11"):
            for group in ("scratch", "high_lr"):
                plot_bn_comparison(output, architecture, group, seed)
    if comparisons:
        pd.DataFrame(comparisons)[columns].to_csv(tables / "transfer_vs_best_scratch.csv", index=False)
    summary = markdown_table(frame[columns])
    (tables / "summary.md").write_text(summary + "\n", encoding="utf-8")
    if readme:
        readme = Path(readme)
        text = readme.read_text(encoding="utf-8")
        start, end = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
        if start not in text or end not in text:
            raise ValueError("README sin marcadores de resultados")
        completed = int((frame.status == "completed").sum())
        note = (f"Experimentos completados: {completed}/{len(frame)} registrados. "
                "Accuracy en escala 0–1; — en epoch_to_80_val significa que no llegó al 80%. "
                "La selección usa validación; los hallazgos se redactan tras revisar las curvas.")
        readme.write_text(text.split(start)[0] + start + "\n\n" + note + "\n\n" + summary
                          + "\n\n" + end + text.split(end)[1], encoding="utf-8")
    return frame[columns]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download")
    download.add_argument("--destination", default="data/raw")
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--data", default="data/raw")
    prepare.add_argument("--output", default="data/splits")
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--validation-fraction", type=float, default=0.2)
    args = parser.parse_args()
    if args.command == "download":
        download_data(args.destination)
    else:
        result = make_manifest(args.data, args.output, args.seed, args.validation_fraction)
        print(result.groupby(["split", "class"]).size().unstack())


if __name__ == "__main__":
    main()

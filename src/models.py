"""Arquitecturas propias y adaptación de ResNet-18 preentrenada."""

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class LeNet5(nn.Module):
    """RGB 64×64: dos conv-pool y tres FC (adaptación de LeNet-5)."""

    def __init__(self, batch_norm=False, num_classes=4):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1 = nn.BatchNorm2d(6) if batch_norm else nn.Identity()
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16) if batch_norm else nn.Identity()
        self.pool = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(16 * 13 * 13, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(torch.tanh(self.bn1(self.conv1(x))))
        x = self.pool(torch.tanh(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)  # Logits: CrossEntropyLoss ya aplica log-softmax.


class VGG11Small(nn.Module):
    """VGG-A: 8 convoluciones + 3 FC; canales a la mitad y FC reducidas."""

    def __init__(self, batch_norm=False, num_classes=4):
        super().__init__()
        configuration = [32, "M", 64, "M", 128, 128, "M", 256, 256, "M", 256, 256, "M"]
        layers, channels = [], 3
        for width in configuration:
            if width == "M":
                layers.append(nn.MaxPool2d(2))
            else:
                # Se mantiene bias en ambas variantes: solo cambia la presencia de BN.
                layers.append(nn.Conv2d(channels, width, 3, padding=1))
                if batch_norm:
                    layers.append(nn.BatchNorm2d(width))
                layers.append(nn.ReLU(inplace=True))
                channels = width
        self.features = nn.Sequential(*layers)
        # Con entrada 64×64, los cinco pools producen exactamente 2×2.
        self.pool = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(256 * 2 * 2, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(torch.flatten(self.pool(self.features(x)), 1))


class TransferResNet18(nn.Module):
    """Congela también las estadísticas de BN de las regiones congeladas."""

    def __init__(self, strategy="feature", pretrained=True, num_classes=4):
        super().__init__()
        if strategy not in {"feature", "partial", "full"}:
            raise ValueError(f"Estrategia desconocida: {strategy}")
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.network = resnet18(weights=weights)
        self.network.fc = nn.Linear(self.network.fc.in_features, num_classes)
        self.strategy = strategy
        for name, parameter in self.network.named_parameters():
            parameter.requires_grad = (
                strategy == "full" or name.startswith("fc.")
                or (strategy == "partial" and name.startswith(("layer3.", "layer4.")))
            )

    def train(self, mode=True):
        super().train(mode)
        if mode and self.strategy != "full":
            for name, module in self.network.named_children():
                if name != "fc" and not (
                    self.strategy == "partial" and name in {"layer3", "layer4"}
                ):
                    module.eval()
        return self

    def forward(self, x):
        return self.network(x)


def build_model(architecture, batch_norm=False, strategy=None, pretrained=True):
    if architecture == "lenet":
        return LeNet5(batch_norm)
    if architecture == "vgg11":
        return VGG11Small(batch_norm)
    if architecture == "resnet18":
        return TransferResNet18(strategy or "feature", pretrained)
    raise ValueError(f"Arquitectura desconocida: {architecture}")


def parameter_counts(model):
    return {
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }

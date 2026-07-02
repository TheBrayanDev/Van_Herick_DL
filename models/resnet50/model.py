import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNet50VHG(nn.Module):
    """
    ResNet50 pre-entrenado para clasificación VHG.
    Basado en Cassanelli et al. (2022) — Paper 3.

    Entrada: RGB (3, 224, 224)
    Salida: 4 clases VHG (grados 1-4)
    """

    def __init__(self, num_classes=4, dropout=0.5, pretrained=True):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = resnet50(weights=weights)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)

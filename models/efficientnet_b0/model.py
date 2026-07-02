import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class EfficientNetB0VHG(nn.Module):
    """
    EfficientNet-B0 pre-entrenado para clasificación VHG.
    EfficientNet es una arquitectura eficiente SOTA.

    Entrada: RGB (3, 224, 224)
    Salida: 4 clases VHG (grados 1-4)
    """

    def __init__(self, num_classes=4, dropout=0.5, pretrained=True):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)

import torch.nn as nn
from torchvision.models import vit_s_16, ViT_S_16_Weights


class ViTS16VHG(nn.Module):
    """
    Vision Transformer Small (ViT-S/16) pre-entrenado para clasificación VHG.
    Dosovitskiy et al. (2020) — An Image is Worth 16x16 Words.

    Entrada: RGB (3, 224, 224)
    Salida: 4 clases VHG (grados 1-4)
    """

    def __init__(self, num_classes=4, dropout=0.5, pretrained=True):
        super().__init__()
        weights = ViT_S_16_Weights.DEFAULT if pretrained else None
        self.backbone = vit_s_16(weights=weights)

        in_features = self.backbone.heads.head.in_features
        self.backbone.heads = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)

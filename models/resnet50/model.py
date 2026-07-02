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

        self.embedding_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.dropout = nn.Dropout(p=dropout)
        self.classifier_final = nn.Linear(self.embedding_dim, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        x = self.classifier_final(x)
        return x

    def forward_embedding(self, x):
        x = self.backbone(x)
        embedding = self.dropout(x)
        logits = self.classifier_final(embedding)
        return logits, embedding

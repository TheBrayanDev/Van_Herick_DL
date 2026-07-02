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

        self.embedding_dim = self.backbone.heads.head.in_features
        self.backbone.heads = nn.Identity()

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

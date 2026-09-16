"""EfficientNet-B4 model definition for binary deepfake classification.

Architecture:
- Pretrained EfficientNet-B4 backbone (torchvision weights)
- Custom binary classification head (2 output logits: 0=Real, 1=Fake)
- Dropout rate configurable for Monte Carlo Dropout uncertainty estimation
"""

import torch
import torch.nn as nn
import torchvision.models as models


class EfficientNetB4Deepfake(nn.Module):
    def __init__(self, dropout_rate: float = 0.4, pretrained: bool = True):
        super().__init__()
        weights = models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b4(weights=weights)
        in_features = self.backbone.classifier[1].in_features  # 1792
        self.dropout_rate = dropout_rate
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.features(x)

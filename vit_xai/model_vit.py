import torch
import torch.nn as nn
from torchvision.models import ViT_B_16_Weights, vit_b_16


class ViTBaseline(nn.Module):
    """ViT-B/16 frame detector with video-level logit averaging."""

    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = vit_b_16(weights=weights)
        self.backbone.heads.head = nn.Linear(self.backbone.hidden_dim, num_classes)

    def forward_frame(self, x):
        return self.backbone(x)

    def forward(self, x):
        b, t, c, h, w = x.shape
        logits = self.forward_frame(x.reshape(b * t, c, h, w))
        return logits.reshape(b, t, -1).mean(dim=1)


"""
models/efficientnet_model.py — EfficientNet-B0 transfer learning model.

Architecture
------------
Backbone : EfficientNet-B0 (ImageNet pretrained), torchvision
Head     : Dropout(p) → Linear(1280 → num_classes)
Output   : Raw logits — no softmax. Apply externally (loss fn / predictor).

GradCAM target : model.backbone[-1]  (last MBConv block, index 8 of features)
"""

import torch
import torch.nn as nn
from typing import Dict, List
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from utils.config import DEVICE, NUM_CLASSES, IMAGE_SIZE

__all__ = ["PlantDiseaseEfficientNet", "build_model"]

# EfficientNet-B0 classifier head in_features — defined by the architecture,
# not by num_classes, so it is a constant here rather than computed at runtime.
_EFFICIENTNET_B0_IN_FEATURES: int = 1280


class PlantDiseaseEfficientNet(nn.Module):
    
    def __init__(
        self,
        num_classes:     int   = NUM_CLASSES,
        dropout_rate:    float = 0.4,
        freeze_backbone: bool  = True,
    ) -> None:
        super().__init__()

        if num_classes <= 1:
            raise ValueError("num_classes must be greater than 1.")
        
        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError("dropout_rate must be between 0 and 1.")

        
        _base = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

        self.backbone: nn.Sequential = _base.features
        self.pool: nn.AdaptiveAvgPool2d = _base.avgpool

        self.classifier: nn.Sequential = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=False),
            nn.Linear(_EFFICIENTNET_B0_IN_FEATURES, num_classes),
        )

        # Initialise the new head with a known-good scheme.
        nn.init.xavier_uniform_(self.classifier[1].weight)
        nn.init.zeros_(self.classifier[1].bias)

        if freeze_backbone:
            self._freeze_backbone()

    # ── Freeze / Unfreeze API ─────────────────────────────────────────

    def _freeze_backbone(self) -> None:
        """Freeze all backbone weights (Stage 1 training — head only)."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, num_blocks: int = 3) -> None:
        
        blocks = list(self.backbone.children())
        num_blocks = min(num_blocks, len(blocks))
        for block in blocks[-num_blocks:]:
            for param in block.parameters():
                param.requires_grad = True

    def freeze_all(self) -> None:
        """Freeze everything — useful for feature-extraction / eval benchmarks."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters — full fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True

    # ── Forward ───────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        if x.dim() != 4:
            raise ValueError(
                f"Expected 4D input (N, C, H, W), got {x.dim()}D tensor."
            )
        
        if x.dtype != torch.float32:
            raise ValueError("Input tensor must be float32.")

        
        if x.size(1) != 3:
            raise ValueError("Expected 3-channel RGB input.")
        
        if x.size(2) != IMAGE_SIZE or x.size(3) != IMAGE_SIZE:
            raise ValueError("Invalid input image size.")


        features = self.backbone(x)          # (N, 1280, 7, 7)
        pooled   = self.pool(features)       # (N, 1280, 1, 1)
        flat     = torch.flatten(pooled, 1)  # (N, 1280)
        return self.classifier(flat)         # (N, num_classes)

    # ── GradCAM ───────────────────────────────────────────────────────

    def get_gradcam_target_layer(self) -> nn.Module:
        
        return self.backbone[-1][0]

    # ── Diagnostics ───────────────────────────────────────────────────

    def count_parameters(self) -> Dict[str, int]:
        """Return trainable, frozen, and total parameter counts."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen    = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {"trainable": trainable, "frozen": frozen, "total": trainable + frozen}

    def trainable_layer_names(self) -> List[str]:
        """Return names of all layers with at least one trainable parameter."""
        return [
            name for name, param in self.named_parameters()
            if param.requires_grad
        ]


# ── Convenience factory ───────────────────────────────────────────────────────

def build_model(
    num_classes:     int   = NUM_CLASSES,
    dropout_rate:    float = 0.4,
    freeze_backbone: bool  = True,
    device: str = DEVICE,
) -> PlantDiseaseEfficientNet:
    
    model = PlantDiseaseEfficientNet(
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        freeze_backbone=freeze_backbone,
    )

    model.eval()
    
    return model.to(device)

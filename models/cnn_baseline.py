"""
CNN Image Classifier — PyTorch
Input  : 3 × 224 × 224 RGB images
Output : 9-class probability distribution
Architecture:
    5 convolutional blocks  (Conv2d → ReLU → MaxPool)
    Channels: 3 → 32 → 64 → 128 → 256 → 512
    2 fully-connected layers with Dropout
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helper: single convolutional block
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Conv2d → BatchNorm2d → ReLU → MaxPool2d"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        pool_size: int = 2,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,          # bias redundant when BatchNorm follows
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_size, stride=pool_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class CNNModel(nn.Module):
    """
    5-block CNN for 9-class image classification.

    Spatial resolution after each MaxPool(2×2):
        Input  :  3 × 224 × 224
        Block 1: 32 × 112 × 112
        Block 2: 64 ×  56 ×  56
        Block 3: 128 × 28 ×  28
        Block 4: 256 × 14 ×  14
        Block 5: 512 ×  7 ×   7
        Flatten: 512 × 7 × 7 = 25 088
    """

    NUM_CLASSES: int = 9

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.5) -> None:
        super().__init__()

        # ------------------------------------------------------------------
        # Feature extractor — 5 convolutional blocks
        # ------------------------------------------------------------------
        self.features = nn.Sequential(
            ConvBlock(in_channels=3,   out_channels=32),   # 224 → 112
            ConvBlock(in_channels=32,  out_channels=64),   # 112 →  56
            ConvBlock(in_channels=64,  out_channels=128),  #  56 →  28
            ConvBlock(in_channels=128, out_channels=256),  #  28 →  14
            ConvBlock(in_channels=256, out_channels=512),  #  14 →   7
        )

        # Adaptive pool → guarantees a fixed 7×7 output regardless of input size
        self.adaptive_pool = nn.AdaptiveAvgPool2d(output_size=(7, 7))

        # ------------------------------------------------------------------
        # Classifier head
        # ------------------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Flatten(),

            # FC-1
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),

            # FC-2
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),

            # Output — raw logits (use CrossEntropyLoss during training)
            nn.Linear(256, num_classes),
        )

        # Weight initialisation
        self._init_weights()

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, 3, 224, 224)
        Returns:
            logits: Tensor of shape (B, num_classes)
        """
        x = self.features(x)          # (B, 512, 7, 7)
        x = self.adaptive_pool(x)     # (B, 512, 7, 7)  — no-op for 224×224 input
        x = self.classifier(x)        # (B, num_classes)
        return x

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        """Returns the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Quick sanity check (run this file directly: python cnn_classifier.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNNModel(num_classes=9, dropout=0.5).to(device)
    model.eval()

    dummy_input = torch.randn(4, 3, 224, 224, device=device)   # batch of 4
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model        : CNNModel")
    print(f"Parameters   : {model.count_parameters():,}")
    print(f"Input shape  : {tuple(dummy_input.shape)}")
    print(f"Output shape : {tuple(output.shape)}")   # expected → (4, 9)
    assert output.shape == (4, 9), "Shape mismatch — check architecture!"
    print("Sanity check passed ✓")
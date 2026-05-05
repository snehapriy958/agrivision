import torch
import torch.nn as nn
from torchvision import models


class PlantDiseaseResNet(nn.Module):

    def __init__(self, num_classes: int = 9) -> None:
        super().__init__()

        weights = models.ResNet50_Weights.DEFAULT
        backbone = models.resnet50(weights=weights)

        for param in backbone.parameters():
            param.requires_grad = False

        for param in backbone.layer4.parameters():
            param.requires_grad = True

        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes),
        )

        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PlantDiseaseResNet(num_classes=9).to(device)
    model.eval()

    dummy = torch.randn(1, 3, 224, 224, device=device)
    with torch.no_grad():
        out = model(dummy)

    assert out.shape == (1, 9), f"Unexpected output shape: {out.shape}"
    print(f"Model        : ResNet50 (transfer learning)")
    print(f"Parameters   : {model.count_parameters():,}")
    print(f"Output shape : {tuple(out.shape)}")
    print("Sanity check passed ✓")
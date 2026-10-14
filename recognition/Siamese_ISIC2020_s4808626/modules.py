"""
Model components for the ISIC 2020 melanoma project.

- BaselineClassifier: ResNet-18 that outputs one melanoma score per image.
- SiameseNetwork: shared ResNet-18 encoder that maps each image to a
  128-number embedding, trained on image pairs with contrastive loss.

Uses PyTorch only (no NumPy), as the course specification requires.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

BACKBONE_FEATURES = 512  # Size of ResNet-18's output before its final layer
EMBEDDING_DIM = 128


def make_backbone(pretrained=True):
    """ResNet-18 with its final layer removed: image in, 512 features out."""
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Identity()
    return model


class BaselineClassifier(nn.Module):
    """
    Standard CNN classifier used as the baseline.

    Outputs one raw score (a logit) per image. Higher means more likely
    melanoma. A sigmoid turns it into a probability between 0 and 1.
    """

    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = make_backbone(pretrained)
        self.head = nn.Linear(BACKBONE_FEATURES, 1)

    def forward(self, images):
        return self.head(self.backbone(images)).squeeze(1)


def count_parameters(model):
    """Total number of learnable numbers in a model."""
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    # Quick checks with fake data. Run: python modules.py
    torch.manual_seed(0)
    images = torch.randn(4, 3, 224, 224)

    with torch.no_grad():
        baseline = BaselineClassifier()
        print(f"Baseline output shape: {tuple(baseline(images).shape)}")
        print(f"Baseline parameters: {count_parameters(baseline):,}")
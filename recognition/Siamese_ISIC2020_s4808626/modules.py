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

class EmbeddingNet(nn.Module):
    """
    Maps each image to an embedding of length 1.
    """

    def __init__(self, embedding_dim=EMBEDDING_DIM, pretrained=True):
        super().__init__()
        self.backbone = make_backbone(pretrained)
        self.projection = nn.Linear(BACKBONE_FEATURES, embedding_dim)

    def forward(self, images):
        return F.normalize(self.projection(self.backbone(images)), dim=1)


class SiameseNetwork(nn.Module):
    """
    Both images in a pair go through the same encoder, so the two branches
    share every weight. Similar lesions should end up with nearby embeddings.
    """

    def __init__(self, embedding_dim=EMBEDDING_DIM, pretrained=True):
        super().__init__()
        self.encoder = EmbeddingNet(embedding_dim, pretrained)

    def forward(self, images_a, images_b):
        return self.encoder(images_a), self.encoder(images_b)

    def embed(self, images):
        """Embed single images, used when comparing to reference images."""
        return self.encoder(images)

class ContrastiveLoss(nn.Module):
    """
    For each pair with embedding distance d:
    - same class (same_class = 1): loss = d^2, pulling the pair together
    - different class (same_class = 0): loss = max(0, margin - d)^2, pushing
      the pair apart until they are at least `margin` apart
    """

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb_a, emb_b, same_class):
        distance = F.pairwise_distance(emb_a, emb_b)
        pull = same_class * distance.pow(2)
        push = (1 - same_class) * F.relu(self.margin - distance).pow(2)
        return (pull + push).mean()


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

        siamese = SiameseNetwork()
        emb_a, emb_b = siamese(images, images.flip(0))
        print(f"Siamese embedding shape: {tuple(emb_a.shape)}")
        print(f"Embedding lengths (should all be 1): {emb_a.norm(dim=1)}")
        print(f"Siamese parameters: {count_parameters(siamese):,}")

    # Loss checks on hand-made 2D embeddings with known distances.
    loss_fn = ContrastiveLoss(margin=1.0)
    point = torch.tensor([[1.0, 0.0]])
    opposite = torch.tensor([[-1.0, 0.0]])  # distance 2 from point
    same, different = torch.tensor([1.0]), torch.tensor([0.0])
    print("Contrastive loss checks:")
    print(f"same class, same place (expect 0): {loss_fn(point, point, same):.4f}")
    print(f"same class, far apart (expect 4):  {loss_fn(point, opposite, same):.4f}")
    print(f"diff class, same place (expect 1): {loss_fn(point, point, different):.4f}")
    print(f"diff class, far apart (expect 0):  {loss_fn(point, opposite, different):.4f}")
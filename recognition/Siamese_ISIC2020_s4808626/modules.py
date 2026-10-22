"""
Model components for the ISIC 2020 melanoma project.

- BaselineClassifier: ResNet-18 that outputs one melanoma score per image.
- SiameseNetwork: shared ResNet-18 encoder that maps each image to a
  128-number embedding, trained on image pairs with contrastive loss.

Standard methods used here (full references in README.md):
- ResNet-18 [9], pretrained on ImageNet [11]
- Siamese network with a shared encoder, Koch et al. [7]
- Contrastive loss, Hadsell et al. [8]
- Classifying by nearest reference images, as in Tschandl et al. [10]
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
    Standard Siamese design from Koch et al. [7].
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
    Standard contrastive loss from Hadsell et al. [8].
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


def class_distances(query_emb, ref_emb, ref_labels, k=10):
    """
    Average distance from each query embedding to its k nearest melanoma
    and k nearest benign reference embeddings. Nearest-reference
    classification follows Tschandl et al. [10].

    Each class is searched separately so the far larger benign group cannot
    outvote melanoma. Returns (melanoma_distance, benign_distance), each with
    one value per query.
    """
    distances = torch.cdist(query_emb, ref_emb)  # (num_queries, num_refs)
    per_class = []
    for label in (1, 0):
        class_dist = distances[:, ref_labels == label]
        k_class = min(k, class_dist.shape[1])  # small test subsets may have fewer than k
        nearest = class_dist.topk(k_class, dim=1, largest=False).values
        per_class.append(nearest.mean(dim=1))
    return per_class[0], per_class[1]


def melanoma_probability(mel_dist, ben_dist, scale=1.0):
    """
    Turn the two class distances into a melanoma probability.

    An image closer to melanoma references than benign ones gets a probability
    above 0.5, and one equally close to both gets 0.5, meaning unsure. `scale`
    controls how fast confidence grows with the gap and is tuned on the
    validation split so confidence matches real accuracy.
    """
    return torch.sigmoid(scale * (ben_dist - mel_dist))


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

    # Reference comparison checks on hand-made 2D embeddings.
    refs = torch.tensor([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, -0.1]])
    ref_labels = torch.tensor([1, 1, 0, 0])  # two melanoma, two benign
    queries = torch.tensor([[1.0, 0.05], [-1.0, 0.05], [0.0, 1.0]])
    mel_d, ben_d = class_distances(queries, refs, ref_labels, k=2)
    probs = melanoma_probability(mel_d, ben_d, scale=5.0)
    print("Reference comparison (expect near 1, near 0, about 0.5):")
    print(f"  {[round(p, 3) for p in probs.tolist()]}")

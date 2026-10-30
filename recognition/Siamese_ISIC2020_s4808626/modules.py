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
    """
    Build a ResNet-18 [9] with its final classification layer removed.

    Args:
        pretrained: If True, start from torchvision's ImageNet weights [11].

    Returns:
        A network that maps a batch of 224x224 images to 512 features each.
    """
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Identity()
    return model


class BaselineClassifier(nn.Module):
    """
    Standard CNN classifier used as the baseline.

    A ResNet-18 backbone followed by one linear layer. It outputs one raw score
    (a logit) per image, where higher means more likely melanoma. A sigmoid
    turns the score into a probability between 0 and 1.
    """

    def __init__(self, pretrained=True):
        """
        Create the backbone and the single-output head.

        Args:
            pretrained: If True, the backbone starts from ImageNet weights.
        """
        super().__init__()
        self.backbone = make_backbone(pretrained)
        self.head = nn.Linear(BACKBONE_FEATURES, 1)

    def forward(self, images):
        """
        Score a batch of images.

        Args:
            images: Tensor of shape (batch, 3, 224, 224).

        Returns:
            Tensor of shape (batch,) with one logit per image.
        """
        return self.head(self.backbone(images)).squeeze(1)

class EmbeddingNet(nn.Module):
    """
    Encoder that maps each image to a 128-number embedding of length 1.

    A ResNet-18 backbone, a linear layer to 128 outputs, and scaling to unit
    length, so the distance between two embeddings always lies between 0 and 2.
    """

    def __init__(self, embedding_dim=EMBEDDING_DIM, pretrained=True):
        """
        Create the backbone and the projection to the embedding size.

        Args:
            embedding_dim: Number of values in each embedding.
            pretrained: If True, the backbone starts from ImageNet weights.
        """
        super().__init__()
        self.backbone = make_backbone(pretrained)
        self.projection = nn.Linear(BACKBONE_FEATURES, embedding_dim)

    def forward(self, images):
        """
        Embed a batch of images.

        Args:
            images: Tensor of shape (batch, 3, 224, 224).

        Returns:
            Tensor of shape (batch, embedding_dim), each row scaled to length 1.
        """
        return F.normalize(self.projection(self.backbone(images)), dim=1)


class SiameseNetwork(nn.Module):
    """
    Siamese network with one shared encoder, the standard design from Koch et al. [7].

    Both images in a pair go through the same encoder, so the two branches
    share every weight. Trained with contrastive loss, similar lesions end up
    with nearby embeddings.
    """

    def __init__(self, embedding_dim=EMBEDDING_DIM, pretrained=True):
        """
        Create the shared encoder.

        Args:
            embedding_dim: Number of values in each embedding.
            pretrained: If True, the encoder's backbone starts from ImageNet weights.
        """
        super().__init__()
        self.encoder = EmbeddingNet(embedding_dim, pretrained)

    def forward(self, images_a, images_b):
        """
        Embed both images of each pair with the shared encoder.

        Args:
            images_a: First images of the pairs, shape (batch, 3, 224, 224).
            images_b: Second images of the pairs, same shape.

        Returns:
            A tuple (emb_a, emb_b) of embeddings, each of shape (batch, embedding_dim).
        """
        return self.encoder(images_a), self.encoder(images_b)

    def embed(self, images):
        """
        Embed single images, used when comparing them with reference images.

        Args:
            images: Tensor of shape (batch, 3, 224, 224).

        Returns:
            Tensor of shape (batch, embedding_dim).
        """
        return self.encoder(images)


class ContrastiveLoss(nn.Module):
    """
    Standard contrastive loss from Hadsell et al. [8].

    For each pair with embedding distance d:
    - same class (same_class = 1): loss = d^2, pulling the pair together
    - different class (same_class = 0): loss = max(0, margin - d)^2, pushing
      the pair apart until they are at least `margin` apart
    The loss is averaged over the batch.
    """

    def __init__(self, margin=1.0):
        """
        Store the margin.

        Args:
            margin: Distance that different-class pairs are pushed beyond.
        """
        super().__init__()
        self.margin = margin

    def forward(self, emb_a, emb_b, same_class):
        """
        Compute the average contrastive loss over a batch of pairs.

        Args:
            emb_a: Embeddings of the first images, shape (batch, embedding_dim).
            emb_b: Embeddings of the second images, same shape.
            same_class: Tensor of shape (batch,), 1.0 for same-class pairs and
                0.0 otherwise.

        Returns:
            A scalar loss tensor.
        """
        distance = F.pairwise_distance(emb_a, emb_b)
        pull = same_class * distance.pow(2)
        push = (1 - same_class) * F.relu(self.margin - distance).pow(2)
        return (pull + push).mean()


def class_distances(query_emb, ref_emb, ref_labels, k=10):
    """
    Average distance from each query to its k nearest melanoma and k nearest benign references.

    Nearest-reference classification follows Tschandl et al. [10]. Each class is
    searched separately, so the larger benign group cannot outvote melanoma.

    Args:
        query_emb: Embeddings of the images to classify, shape (queries, dim).
        ref_emb: Embeddings of the labelled reference images, shape (refs, dim).
        ref_labels: Labels of the references (1 melanoma, 0 benign), shape (refs,).
        k: Number of nearest references used per class. Small subsets with
            fewer references than k use all of them.

    Returns:
        A tuple (melanoma_distance, benign_distance), each of shape (queries,).
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
    Turn the two class distances into a rough melanoma probability.

    An image closer to melanoma references than benign ones gets a probability
    above 0.5, and one equally close to both gets 0.5, meaning unsure. It is only
    used during training, with scale=1.0, to score the validation split after
    each epoch. The final calibrated probabilities come from Platt scaling in
    evaluate.py.

    Args:
        mel_dist: Average distance to the nearest melanoma references.
        ben_dist: Average distance to the nearest benign references.
        scale: How fast the probability changes with the distance gap.

    Returns:
        Tensor of probabilities with the same shape as the inputs.
    """
    return torch.sigmoid(scale * (ben_dist - mel_dist))


def count_parameters(model):
    """
    Count the learnable numbers (weights and biases) in a model.

    Args:
        model: Any PyTorch module.

    Returns:
        The total number of parameters as an int.
    """
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

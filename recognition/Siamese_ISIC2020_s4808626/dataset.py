"""
Data loading and patient-level splitting for the ISIC 2020 melanoma dataset.

Run directly to create the train, validation and test splits:
    python dataset.py --data-dir ~/data/isic2020
"""
import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Share of patients in each split. Must add up to 1.
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
# ImageNet colour statistics, expected by the pretrained ResNet-18.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224


def load_metadata(data_dir):
    """Read train.csv and remove rows flagged as duplicate images."""
    df = pd.read_csv(Path(data_dir) / "train.csv")
    before = len(df)
    # In this copy of the dataset, tfrecord == -1 marks a duplicate image.
    df = df[df["tfrecord"] != -1].reset_index(drop=True)
    print(f"Removed {before - len(df)} duplicate rows, {len(df)} images remain")
    return df


def split_patients(df, seed=42):
    """
    Assign every patient to exactly one split (train, val or test).

    Patients with at least one melanoma and patients with none are split
    separately, so each split gets a similar share of the rare class.
    """
    # One row per patient: 1 if any of their images is melanoma, else 0.
    patients = df.groupby("patient_id")["target"].max().reset_index()
    patients.columns = ["patient_id", "has_melanoma"]

    assignments = {}
    for _, group in patients.groupby("has_melanoma"):
        # Shuffle this group's patients. The fixed seed makes it repeatable.
        ids = group.sample(frac=1, random_state=seed)["patient_id"].tolist()
        n_train = round(len(ids) * SPLIT_FRACTIONS["train"])
        n_val = round(len(ids) * SPLIT_FRACTIONS["val"])

        for i, patient_id in enumerate(ids):
            if i < n_train:
                assignments[patient_id] = "train"
            elif i < n_train + n_val:
                assignments[patient_id] = "val"
            else:
                assignments[patient_id] = "test"

    df = df.copy()
    df["split"] = df["patient_id"].map(assignments)
    return df


def check_no_leakage(df):
    """Stop with an error if any patient appears in more than one split."""
    patients = {name: set(group["patient_id"]) for name, group in df.groupby("split")}
    assert not patients["train"] & patients["val"], "Patient in both train and val"
    assert not patients["train"] & patients["test"], "Patient in both train and test"
    assert not patients["val"] & patients["test"], "Patient in both val and test"
    print("Leakage check passed: no patient appears in more than one split")


def summarise(df):
    """Print image, patient and melanoma counts for each split."""
    for name in ["train", "val", "test"]:
        part = df[df["split"] == name]
        n_mel = int(part["target"].sum())
        print(
            f"{name:>5}: {len(part):>6} images, {part['patient_id'].nunique():>5} patients, "
            f"{n_mel:>4} melanoma ({100 * n_mel / len(part):.2f}%)"
        )


def save_splits(df, out_dir):
    """Save one CSV per split so every run uses exactly the same patients."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    columns = ["image_name", "patient_id", "target"]
    for name in ["train", "val", "test"]:
        df[df["split"] == name][columns].to_csv(out_dir / f"{name}.csv", index=False)
    print(f"Saved splits to {out_dir.resolve()}")

def get_transforms(train):
    """
    Image preparation steps.

    Training images get random flips, rotations, crops and colour changes so
    the model sees varied versions of each lesion. Lesion photos have no fixed
    "up" direction, and colour varies between clinics and cameras.
    Validation and test images are only centre-cropped, so results are repeatable.
    """
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.RandomCrop(IMAGE_SIZE),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class LesionDataset(Dataset):
    """
    Single lesion images with their labels, read from one split CSV.

    Returns (image tensor, label, image name). The label is 1.0 for melanoma
    and 0.0 for benign. The image name is kept for later error analysis.
    """

    def __init__(self, split_csv, image_dir, transform=None):
        self.df = pd.read_csv(split_csv)
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(self.image_dir / f"{row['image_name']}.jpg").convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = torch.tensor(row["target"], dtype=torch.float32)
        return image, label, row["image_name"]


def save_preview(data_dir, splits_dir, out_path, n_per_class=8, seed=0):
    """Save a grid of augmented training images, half melanoma and half benign."""
    dataset = LesionDataset(
        Path(splits_dir) / "train.csv", Path(data_dir) / "train", get_transforms(train=True)
    )

    # Check that a batch loads with the expected shapes.
    images, labels, _ = next(iter(DataLoader(dataset, batch_size=32, shuffle=True)))
    print(f"Batch shapes: images {tuple(images.shape)}, labels {tuple(labels.shape)}")

    # Pick equal numbers of each class so melanoma actually appears in the grid.
    rng = random.Random(seed)
    melanoma = dataset.df.index[dataset.df["target"] == 1].tolist()
    benign = dataset.df.index[dataset.df["target"] == 0].tolist()
    chosen = rng.sample(melanoma, n_per_class) + rng.sample(benign, n_per_class)

    # Undo the normalisation so colours display correctly.
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for ax, idx in zip(axes.flat, chosen):
        image, label, name = dataset[idx]
        image = (image * std + mean).clamp(0, 1).permute(1, 2, 0)
        ax.imshow(image)
        ax.set_title(f"{'Melanoma' if label == 1 else 'Benign'}\n{name}", fontsize=8)
        ax.axis("off")
    fig.suptitle("Augmented training images (top two rows melanoma, bottom two benign)")
    fig.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    print(f"Saved preview to {Path(out_path).resolve()}")

class PairDataset(Dataset):
    """
    Random image pairs for training the Siamese network.

    Each pair is chosen with two coin flips:
    1. The first image is melanoma or benign with equal chance.
    2. The second image is the same class or the other class with equal chance.
    All four pair types are equally common, so melanoma appears in 75% of
    pairs even though it is under 2% of images.

    Returns (image_a, image_b, same_class, label_a, label_b), where same_class
    is 1.0 if both images have the same label and 0.0 otherwise.
    """

    def __init__(self, split_csv, image_dir, transform, pairs_per_epoch=None):
        self.images = LesionDataset(split_csv, image_dir, transform)
        df = self.images.df
        self.by_class = {
            1: df.index[df["target"] == 1].tolist(),
            0: df.index[df["target"] == 0].tolist(),
        }
        # Half as many pairs as images, so one epoch processes the same
        # number of images as one baseline epoch.
        self.pairs_per_epoch = pairs_per_epoch or len(df) // 2

    def __len__(self):
        return self.pairs_per_epoch

    def sample_pair_indices(self):
        """Pick two row indices and whether they share a class."""
        class_a = random.choice([0, 1])
        same = random.random() < 0.5
        class_b = class_a if same else 1 - class_a

        idx_a = random.choice(self.by_class[class_a])
        idx_b = random.choice(self.by_class[class_b])
        # Never pair an image with itself.
        while idx_b == idx_a:
            idx_b = random.choice(self.by_class[class_b])
        return idx_a, idx_b, same

    def __getitem__(self, idx):
        # idx is ignored: a fresh random pair is drawn every time, so each
        # epoch sees new combinations of the same images.
        idx_a, idx_b, same = self.sample_pair_indices()
        image_a, label_a, _ = self.images[idx_a]
        image_b, label_b, _ = self.images[idx_b]
        same_class = torch.tensor(1.0 if same else 0.0)
        return image_a, image_b, same_class, label_a, label_b


def check_pairs(data_dir, splits_dir, n=10000, seed=0):
    """Count pair types over many samples to confirm the 50/50 coin flips work."""
    random.seed(seed)
    pairs = PairDataset(Path(splits_dir) / "train.csv", Path(data_dir) / "train", transform=None)
    labels = pairs.images.df["target"]

    counts = {"melanoma-melanoma": 0, "melanoma-benign": 0, "benign-benign": 0, "benign-melanoma": 0}
    melanoma_seen = 0
    for _ in range(n):
        idx_a, idx_b, _ = pairs.sample_pair_indices()
        a = "melanoma" if labels[idx_a] == 1 else "benign"
        b = "melanoma" if labels[idx_b] == 1 else "benign"
        counts[f"{a}-{b}"] += 1
        melanoma_seen += (a == "melanoma") + (b == "melanoma")

    print(f"Pairs per epoch: {len(pairs)}")
    for name, count in counts.items():
        print(f"{name:>18}: {100 * count / n:.1f}%")
    print(f"Share of images in pairs that are melanoma: {100 * melanoma_seen / (2 * n):.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create patient-level ISIC 2020 splits")
    parser.add_argument("--data-dir", required=True, help="Folder containing train.csv and train/")
    parser.add_argument("--out-dir", default="splits", help="Where to save the split CSVs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preview", action="store_true", help="Also save a grid of sample images")
    parser.add_argument("--check-pairs", action="store_true", help="Check the pair type balance")
    args = parser.parse_args()

    data = load_metadata(args.data_dir)
    data = split_patients(data, seed=args.seed)
    check_no_leakage(data)
    summarise(data)
    save_splits(data, args.out_dir)

    if args.preview:
        save_preview(args.data_dir, args.out_dir, "figures/sample_training_images.png")

    if args.check_pairs:
        check_pairs(args.data_dir, args.out_dir)
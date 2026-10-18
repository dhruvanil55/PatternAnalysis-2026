"""
Train the baseline CNN classifier on ISIC 2020.
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataset import LesionDataset, get_transforms
from modules import BaselineClassifier
from utils import get_device, loader_settings, set_seed

def parse_args():
    parser = argparse.ArgumentParser(description="Train the ISIC 2020 melanoma models")
    parser.add_argument("--data-dir", required=True, help="Folder containing train/ images")
    parser.add_argument("--splits-dir", default="splits", help="Folder with the split CSVs")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--workers", type=int, default=4, help="Parallel image loaders")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None,
                        help="Use a random subset of this many images, for quick tests")
    return parser.parse_args()

def make_loader(split_csv, image_dir, train, batch_size, workers, limit=None):
    """Build a DataLoader for one split, optionally on a small random subset."""
    dataset = LesionDataset(split_csv, image_dir, get_transforms(train))
    if limit is not None and limit < len(dataset):
        # Random subset, so it still contains a few melanoma images.
        indices = torch.randperm(len(dataset))[:limit].tolist()
        dataset = Subset(dataset, indices)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, **loader_settings(workers))


def melanoma_weight(split_csv):
    """How many times more each melanoma image counts in the loss (benign / melanoma)."""
    labels = pd.read_csv(split_csv)["target"]
    n_melanoma = int(labels.sum())
    return (len(labels) - n_melanoma) / n_melanoma


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    """Run one pass over the training data and return the average loss."""
    model.train()  # turns on training behaviour such as batch norm updates
    total_loss, total_images = 0.0, 0

    for step, (images, labels, _) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()              # clear the previous batch's gradients
        loss = loss_fn(model(images), labels)
        loss.backward()                    # work out how each weight affected the loss
        optimizer.step()                   # adjust the weights to reduce the loss

        total_loss += loss.item() * len(labels)
        total_images += len(labels)
        if step % 50 == 0:
            print(f"  step {step:>4}/{len(loader)}  loss {loss.item():.4f}")

    return total_loss / total_images

def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    image_dir = Path(args.data_dir).expanduser() / "train"
    train_csv = Path(args.splits_dir) / "train.csv"

    train_loader = make_loader(train_csv, image_dir, train=True, batch_size=args.batch_size,
                               workers=args.workers, limit=args.limit)
    print(f"Training images: {len(train_loader.dataset)}")

    model = BaselineClassifier().to(device)
    weight = melanoma_weight(train_csv)
    print(f"Melanoma loss weight: {weight:.1f}")
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        print(f"Epoch {epoch}/{args.epochs}  train loss {train_loss:.4f}  "
              f"time {time.time() - start:.0f}s")

   
if __name__ == "__main__":
    main()
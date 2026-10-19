"""
Train the baseline CNN classifier on ISIC 2020.
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             recall_score, roc_auc_score)
from torch.utils.data import DataLoader, Subset

from dataset import LesionDataset, get_transforms
from modules import BaselineClassifier
from utils import get_device, loader_settings, plot_history, set_seed


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
    parser.add_argument("--out-dir", default="outputs", help="Where to save models and logs")
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
            print(f"  step {step:>4}/{len(loader)}  loss {loss.item():.4f}", flush=True)

    return total_loss / total_images


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    """Score every image in a split and return the loss and key metrics."""
    model.eval()  # turns off training behaviour such as batch norm updates
    total_loss, all_probs, all_labels = 0.0, [], []

    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += loss_fn(logits, labels).item() * len(labels)
        all_probs.append(torch.sigmoid(logits).cpu())  # raw score to probability
        all_labels.append(labels.cpu())

    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy().astype(int)
    preds = (probs >= 0.5).astype(int)  # provisional threshold, tuned later

    metrics = {
        "loss": total_loss / len(labels),
        "balanced_acc": balanced_accuracy_score(labels, preds),
        "melanoma_recall": recall_score(labels, preds, zero_division=0),
    }
    if len(set(labels)) == 2:
        metrics["auroc"] = roc_auc_score(labels, probs)
        metrics["pr_auc"] = average_precision_score(labels, probs)
    else:  # a tiny test subset may contain no melanoma
        metrics["auroc"] = metrics["pr_auc"] = float("nan")
    return metrics


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

    val_csv = Path(args.splits_dir) / "val.csv"
    val_loader = make_loader(val_csv, image_dir, train=False, batch_size=args.batch_size,
                             workers=args.workers, limit=args.limit)
    print(f"Validation images: {len(val_loader.dataset)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = BaselineClassifier().to(device)
    weight = melanoma_weight(train_csv)
    print(f"Melanoma loss weight: {weight:.1f}")
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_auroc = -1.0
    history = []  # one row of results per epoch
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val = evaluate(model, val_loader, loss_fn, device)

        print(f"Epoch {epoch}/{args.epochs}  train loss {train_loss:.4f}  "
              f"val loss {val['loss']:.4f}  bal acc {val['balanced_acc']:.3f}  "
              f"AUROC {val['auroc']:.3f}  PR-AUC {val['pr_auc']:.3f}  "
              f"mel recall {val['melanoma_recall']:.3f}  time {time.time() - start:.0f}s",
              flush=True)

        # Save the results so far and redraw the training curves.
        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val["loss"],
            "val_balanced_acc": val["balanced_acc"], "val_auroc": val["auroc"],
            "val_pr_auc": val["pr_auc"], "val_melanoma_recall": val["melanoma_recall"],
            "seconds": round(time.time() - start),
        })
        history_csv = out_dir / "baseline_history.csv"
        pd.DataFrame(history).to_csv(history_csv, index=False)
        plot_history(history_csv, Path("figures") / "baseline_curves.png", "Baseline CNN")

        # Keep the weights from the epoch with the best validation AUROC.
        if val["auroc"] > best_auroc:
            best_auroc = val["auroc"]
            torch.save(model.state_dict(), out_dir / "baseline_best.pt")
            print(f"  Saved new best model (val AUROC {best_auroc:.3f})", flush=True)


if __name__ == "__main__":
    main()

"""
Train the baseline CNN classifier or the Siamese network on ISIC 2020.

Quick tests on a small subset:
    python train.py --data-dir ~/isic2020 --model baseline --epochs 2 --limit 512
    python train.py --data-dir ~/isic2020 --model siamese --epochs 2 --limit 512
Quick tests save to outputs/<model>_quicktest_*, so they never overwrite a full run.
Full training (saves the best model to outputs/<model>_best.pt):
    python train.py --data-dir ~/isic2020 --model baseline --epochs 10
    python train.py --data-dir ~/isic2020 --model siamese --epochs 10
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

from dataset import LesionDataset, PairDataset, get_transforms
from modules import (BaselineClassifier, ContrastiveLoss, SiameseNetwork,
                     class_distances, melanoma_probability)
from utils import get_device, loader_settings, plot_history, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train the ISIC 2020 melanoma models")
    parser.add_argument("--data-dir", required=True, help="Folder containing train/ images")
    parser.add_argument("--model", choices=["baseline", "siamese"], default="baseline",
                        help="Which model to train")
    parser.add_argument("--splits-dir", default="splits", help="Folder with the split CSVs")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Images per batch (baseline) or pairs per batch (Siamese)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--margin", type=float, default=1.0,
                        help="Contrastive loss margin (Siamese only)")
    parser.add_argument("--k", type=int, default=10,
                        help="Nearest references per class used to predict (Siamese only)")
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


def classification_metrics(probs, labels, loss):
    """
    Validation metrics shared by both models.

    probs are melanoma probabilities and labels are 0 or 1. The 0.5 threshold
    is provisional and is tuned on the validation split later.
    """
    preds = (probs >= 0.5).astype(int)
    metrics = {
        "loss": loss,
        "balanced_acc": balanced_accuracy_score(labels, preds),
        "melanoma_recall": recall_score(labels, preds, zero_division=0),
    }
    if len(set(labels)) == 2:
        metrics["auroc"] = roc_auc_score(labels, probs)
        metrics["pr_auc"] = average_precision_score(labels, probs)
    else:  # a tiny test subset may contain no melanoma
        metrics["auroc"] = metrics["pr_auc"] = float("nan")
    return metrics


# ---------------------------------------------------------------------------
# Baseline CNN: one image in, one melanoma score out, weighted BCE loss
# ---------------------------------------------------------------------------

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
    return classification_metrics(probs, labels, total_loss / len(labels))


# ---------------------------------------------------------------------------
# Siamese network: image pairs in, contrastive loss, predict by comparing
# each image to labelled reference images
# ---------------------------------------------------------------------------

def train_siamese_epoch(model, loader, loss_fn, optimizer, device):
    """Run one pass over freshly sampled training pairs and return the average loss."""
    model.train()
    total_loss, total_pairs = 0.0, 0

    for step, (images_a, images_b, same_class, _, _) in enumerate(loader):
        images_a, images_b = images_a.to(device), images_b.to(device)
        same_class = same_class.to(device)

        optimizer.zero_grad()
        emb_a, emb_b = model(images_a, images_b)  # both go through the shared encoder
        loss = loss_fn(emb_a, emb_b, same_class)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(same_class)
        total_pairs += len(same_class)
        if step % 50 == 0:
            print(f"  step {step:>4}/{len(loader)}  loss {loss.item():.4f}", flush=True)

    return total_loss / total_pairs


@torch.no_grad()
def embed_split(model, loader, device):
    """Embed every image in a split. Returns (embeddings, labels)."""
    model.eval()
    embeddings, labels = [], []
    for images, batch_labels, _ in loader:
        embeddings.append(model.embed(images.to(device)))
        labels.append(batch_labels)
    return torch.cat(embeddings), torch.cat(labels)


@torch.no_grad()
def validation_pair_loss(embeddings, labels, loss_fn, n_pairs=2000, seed=0):
    """
    Contrastive loss on a fixed set of validation pairs.

    Pairs follow the same two 50/50 coin flips as training, and the fixed seed
    means every epoch is scored on the same pairs, so the values are comparable.
    """
    generator = torch.Generator().manual_seed(seed)
    labels = labels.cpu()
    by_class = {c: torch.nonzero(labels == c).squeeze(1) for c in (0, 1)}
    if len(by_class[1]) == 0:
        return float("nan")

    class_a = torch.randint(0, 2, (n_pairs,), generator=generator)
    same = torch.rand(n_pairs, generator=generator) < 0.5
    class_b = torch.where(same, class_a, 1 - class_a)

    def pick(classes):
        chosen = torch.empty(n_pairs, dtype=torch.long)
        for c in (0, 1):
            mask = classes == c
            pool = by_class[c]
            chosen[mask] = pool[torch.randint(0, len(pool), (int(mask.sum()),),
                                              generator=generator)]
        return chosen.to(embeddings.device)

    emb_a, emb_b = embeddings[pick(class_a)], embeddings[pick(class_b)]
    return loss_fn(emb_a, emb_b, same.float().to(embeddings.device)).item()


@torch.no_grad()
def evaluate_siamese(model, ref_loader, val_loader, loss_fn, device, k):
    """
    Predict each validation image from its k nearest melanoma and benign
    training images, then return the loss and key metrics.

    The training images are re-embedded every epoch because the encoder changes.
    """
    ref_emb, ref_labels = embed_split(model, ref_loader, device)
    val_emb, val_labels = embed_split(model, val_loader, device)

    mel_dist, ben_dist = class_distances(val_emb, ref_emb, ref_labels.to(device), k)
    probs = melanoma_probability(mel_dist, ben_dist).cpu().numpy()
    loss = validation_pair_loss(val_emb, val_labels, loss_fn)
    return classification_metrics(probs, val_labels.numpy().astype(int), loss)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}  Model: {args.model}")

    image_dir = Path(args.data_dir).expanduser() / "train"
    train_csv = Path(args.splits_dir) / "train.csv"
    val_csv = Path(args.splits_dir) / "val.csv"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Quick tests (--limit) save under a separate name and keep their curves in
    # outputs/, so they can never overwrite a full run's model, history or figure.
    run_name = f"{args.model}_quicktest" if args.limit else args.model
    figure_dir = out_dir if args.limit else Path("figures")

    val_loader = make_loader(val_csv, image_dir, train=False, batch_size=args.batch_size,
                             workers=args.workers, limit=args.limit)
    print(f"Validation images: {len(val_loader.dataset)}")

    if args.model == "baseline":
        train_loader = make_loader(train_csv, image_dir, train=True, batch_size=args.batch_size,
                                   workers=args.workers, limit=args.limit)
        print(f"Training images: {len(train_loader.dataset)}")
        model = BaselineClassifier().to(device)
        weight = melanoma_weight(train_csv)
        print(f"Melanoma loss weight: {weight:.1f}")
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight, device=device))
        title, loss_label = "Baseline CNN", "Weighted BCE loss"
    else:
        # Fresh random pairs every epoch. With --limit, use limit // 2 pairs.
        pairs = PairDataset(train_csv, image_dir, get_transforms(train=True),
                            pairs_per_epoch=args.limit // 2 if args.limit else None)
        train_loader = DataLoader(pairs, batch_size=args.batch_size,
                                  **loader_settings(args.workers))
        print(f"Training pairs per epoch: {len(pairs)}")
        # Reference images for prediction: training images without augmentation.
        ref_loader = make_loader(train_csv, image_dir, train=False, batch_size=args.batch_size,
                                 workers=args.workers, limit=args.limit)
        print(f"Reference images: {len(ref_loader.dataset)}")
        model = SiameseNetwork().to(device)
        loss_fn = ContrastiveLoss(margin=args.margin)
        title, loss_label = "Siamese network", "Contrastive loss"

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_auroc = -1.0
    history = []  # one row of results per epoch
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        if args.model == "baseline":
            train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
            val = evaluate(model, val_loader, loss_fn, device)
        else:
            train_loss = train_siamese_epoch(model, train_loader, loss_fn, optimizer, device)
            val = evaluate_siamese(model, ref_loader, val_loader, loss_fn, device, args.k)

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
        history_csv = out_dir / f"{run_name}_history.csv"
        pd.DataFrame(history).to_csv(history_csv, index=False)
        plot_history(history_csv, figure_dir / f"{run_name}_curves.png", title, loss_label)

        # Keep the weights from the epoch with the best validation AUROC.
        if val["auroc"] > best_auroc:
            best_auroc = val["auroc"]
            torch.save(model.state_dict(), out_dir / f"{run_name}_best.pt")
            print(f"  Saved new best model (val AUROC {best_auroc:.3f})", flush=True)


if __name__ == "__main__":
    main()

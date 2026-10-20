"""
Evaluate the trained baseline CNN and Siamese network on ISIC 2020.

Everything that is tuned (calibration and thresholds) is fitted on the
validation split only. The test split is then scored once.

Run after both models are trained:
    python evaluate.py --data-dir ~/isic2020
Quick check that it runs, on small subsets (the numbers are not meaningful):
    python evaluate.py --data-dir ~/isic2020 --limit 512
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from dataset import LesionDataset, get_transforms
from modules import BaselineClassifier, SiameseNetwork, class_distances
from utils import get_device, loader_settings, set_seed
MAX_MISSED_MELANOMA = 0.05  # AC4: automatically cleared cases may miss at most 5% of melanomas


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate both ISIC 2020 models")
    parser.add_argument("--data-dir", required=True, help="Folder containing train/ images")
    parser.add_argument("--splits-dir", default="splits", help="Folder with the split CSVs")
    parser.add_argument("--out-dir", default="outputs", help="Where the trained models are")
    parser.add_argument("--k", type=int, default=10,
                        help="Nearest references per class for the Siamese network")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None,
                        help="Use a random subset of each split, for quick tests")
    return parser.parse_args()

def load_split(splits_dir, name, limit, out_dir):
    """
    Return a split's rows and a CSV path LesionDataset can read.

    With --limit, a fixed random subset is written to outputs/ so quick tests
    never touch the real split files.
    """
    path = Path(splits_dir) / f"{name}.csv"
    df = pd.read_csv(path)
    if limit and limit < len(df):
        df = df.sample(n=limit, random_state=0).reset_index(drop=True)
        path = Path(out_dir) / f"quicktest_{name}.csv"
        df.to_csv(path, index=False)
    return df, path

def make_eval_loader(split_csv, image_dir, batch_size, workers):
    """Images in file order with no augmentation, so scores line up with the CSV rows."""
    dataset = LesionDataset(split_csv, image_dir, get_transforms(train=False))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, **loader_settings(workers))

def load_models(out_dir, device):
    """Load the best saved weights of both models."""
    baseline = BaselineClassifier(pretrained=False)
    baseline.load_state_dict(torch.load(Path(out_dir) / "baseline_best.pt", map_location="cpu"))
    siamese = SiameseNetwork(pretrained=False)
    siamese.load_state_dict(torch.load(Path(out_dir) / "siamese_best.pt", map_location="cpu"))
    return baseline.to(device).eval(), siamese.to(device).eval()

@torch.no_grad()
def baseline_scores(model, loader, device):
    """The baseline's raw output (logit) for every image."""
    return torch.cat([model(images.to(device)).cpu() for images, _, _ in loader]).numpy()


@torch.no_grad()
def embed_images(model, loader, device):
    """Siamese embeddings for every image, kept on the device."""
    return torch.cat([model.embed(images.to(device)) for images, _, _ in loader])


@torch.no_grad()
def siamese_scores(query_emb, ref_emb, ref_labels, k, chunk=512):
    """
    Distance gap (benign distance minus melanoma distance) for every query.

    Positive means the image sits closer to melanoma references. Queries are
    processed in chunks to keep memory use low.
    """
    gaps = []
    for start in range(0, len(query_emb), chunk):
        mel_dist, ben_dist = class_distances(query_emb[start:start + chunk], ref_emb, ref_labels, k)
        gaps.append((ben_dist - mel_dist).cpu())
    return torch.cat(gaps).numpy()

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def fit_calibration(scores, labels):
    """
    Platt scaling: fit probability = sigmoid(a * score + b) on validation.

    The same method is used for both models, so their calibrated
    probabilities are directly comparable.
    """
    fit = LogisticRegression(C=1e6, max_iter=1000)  # very weak regularisation
    fit.fit(scores.reshape(-1, 1), labels)
    return float(fit.coef_[0, 0]), float(fit.intercept_[0])


def balanced_threshold(probs, labels):
    """Threshold that maximises balanced accuracy (sensitivity + specificity)."""
    fpr, tpr, thresholds = roc_curve(labels, probs)
    return float(thresholds[np.argmax(tpr - fpr)])


def clear_threshold(probs, labels, max_missed=MAX_MISSED_MELANOMA):
    """
    Highest threshold t such that automatically clearing every image with
    probability below t misses at most `max_missed` of the melanomas (AC4).
    """
    melanoma_probs = np.sort(probs[labels == 1])
    allowed = int(np.floor(max_missed * len(melanoma_probs)))  # melanomas we may clear
    return float(melanoma_probs[allowed]) if allowed < len(melanoma_probs) else 1.0


def route(probs, t_clear, t_urgent):
    """
    Three-tier decision for each image:
    auto-clear (below t_clear), urgent review (at or above t_urgent),
    otherwise routine specialist review.
    """
    decisions = np.full(len(probs), "specialist review", dtype=object)
    decisions[probs < t_clear] = "auto-clear"
    decisions[probs >= t_urgent] = "urgent review"
    return decisions


def routing_summary(probs, labels, t_clear, t_urgent):
    """Share of images in each tier and how many melanomas were automatically cleared."""
    decisions = route(probs, t_clear, t_urgent)
    n_mel = int(labels.sum())
    missed = int(((decisions == "auto-clear") & (labels == 1)).sum())
    urgent = decisions == "urgent review"
    return {
        "auto_clear_share": float((decisions == "auto-clear").mean()),
        "specialist_review_share": float((decisions == "specialist review").mean()),
        "urgent_review_share": float(urgent.mean()),
        "melanomas_auto_cleared": missed,
        "melanoma_miss_rate": missed / n_mel if n_mel else float("nan"),
        "melanoma_share_of_urgent": float(labels[urgent].mean()) if urgent.any() else float("nan"),
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    out_dir = Path(args.out_dir)
    image_dir = Path(args.data_dir).expanduser() / "train"
    print(f"Using device: {device}", flush=True)

    splits = {name: load_split(args.splits_dir, name, args.limit, out_dir)
              for name in ("train", "val", "test")}
    loaders = {name: make_eval_loader(path, image_dir, args.batch_size, args.workers)
               for name, (_, path) in splits.items()}
    labels = {name: df["target"].to_numpy().astype(int) for name, (df, _) in splits.items()}
    for name, (df, _) in splits.items():
        print(f"{name}: {len(df)} images, {labels[name].sum()} melanoma", flush=True)

    baseline, siamese = load_models(out_dir, device)

    # Raw scores on validation and test (higher = more likely melanoma).
    print("Scoring with the baseline...", flush=True)
    scores = {"baseline": {s: baseline_scores(baseline, loaders[s], device) for s in ("val", "test")}}
    print("Embedding reference and query images with the Siamese network...", flush=True)
    ref_emb = embed_images(siamese, loaders["train"], device)
    ref_labels = torch.tensor(labels["train"], device=device)
    scores["siamese"] = {s: siamese_scores(embed_images(siamese, loaders[s], device),
                                           ref_emb, ref_labels, args.k) for s in ("val", "test")}

    # Check: validation AUROC should match the best epoch in each training log.
    print("\nAUROC from raw scores")
    print(f"{'':12}{'Validation':>12}{'Test':>12}")
    for model in ("baseline", "siamese"):
        print(f"{model:12}"
              f"{roc_auc_score(labels['val'], scores[model]['val']):>12.3f}"
              f"{roc_auc_score(labels['test'], scores[model]['test']):>12.3f}")


    # Fit calibration and thresholds on validation only.
    print("\nCalibration and thresholds (fitted on validation)")
    for model in ("baseline", "siamese"):
        a, b = fit_calibration(scores[model]["val"], labels["val"])
        val_probs = sigmoid(a * scores[model]["val"] + b)
        t_balanced = balanced_threshold(val_probs, labels["val"])
        t_clear = clear_threshold(val_probs, labels["val"])
        t_urgent = max(t_balanced, t_clear)
        routing = routing_summary(val_probs, labels["val"], t_clear, t_urgent)
        print(f"{model}: a = {a:.3f}, b = {b:.3f}  |  balanced threshold {t_balanced:.4f}, "
              f"auto-clear threshold {t_clear:.4f}")
        print(f"  validation: auto-clear {routing['auto_clear_share']:.1%}, "
              f"specialist review {routing['specialist_review_share']:.1%}, "
              f"urgent review {routing['urgent_review_share']:.1%}, "
              f"melanomas auto-cleared {routing['melanomas_auto_cleared']} "
              f"({routing['melanoma_miss_rate']:.1%}, limit {MAX_MISSED_MELANOMA:.0%})")


if __name__ == "__main__":
    main()

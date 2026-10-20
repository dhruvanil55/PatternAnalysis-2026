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

import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from dataset import LesionDataset, get_transforms
from modules import BaselineClassifier, SiameseNetwork, class_distances
from utils import get_device, loader_settings, set_seed


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


if __name__ == "__main__":
    main()

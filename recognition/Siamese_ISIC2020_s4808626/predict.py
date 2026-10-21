"""
Example predictions from the trained models. Run evaluate.py first, since it
chooses the calibration and thresholds used here.

For each image it prints both models' calibrated melanoma probability,
routing decision and distance to the nearest melanoma and benign reference.

    python predict.py --data-dir ~/isic2020                       # 4 random test images
    python predict.py --data-dir ~/isic2020 --images ISIC_0149568 ISIC_0188432
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from dataset import get_transforms
from evaluate import load_models, route, sigmoid
from modules import class_distances
from utils import get_device

N_SHOWN = 3  # nearest reference images shown per class


def parse_args():
    parser = argparse.ArgumentParser(description="Show example predictions")
    parser.add_argument("--data-dir", required=True, help="Folder containing train/ images")
    parser.add_argument("--splits-dir", default="splits", help="Folder with the split CSVs")
    parser.add_argument("--out-dir", default="outputs", help="Where the trained models are")
    parser.add_argument("--images", nargs="+", help="Image names to predict, e.g. ISIC_0149568")
    parser.add_argument("--n", type=int, default=4,
                        help="Random test images to predict, half melanoma and half benign")
    parser.add_argument("--quicktest", action="store_true",
                        help="Use the files from evaluate.py --limit")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()

def pick_random(test_predictions, n, seed):
    """Half melanoma and half benign test images, chosen at random."""
    melanoma = test_predictions[test_predictions["target"] == 1]
    benign = test_predictions[test_predictions["target"] == 0]
    chosen = pd.concat([melanoma.sample(n=min(n // 2, len(melanoma)), random_state=seed),
                        benign.sample(n=n - min(n // 2, len(melanoma)), random_state=seed)])
    return chosen["image_name"].tolist(), ["" for _ in range(len(chosen))]


def true_labels(names, splits_dir):
    """Look up each image's label in whichever split it belongs to."""
    all_rows = pd.concat([pd.read_csv(Path(splits_dir) / f"{s}.csv") for s in ("train", "val", "test")])
    lookup = all_rows.set_index("image_name")["target"]
    missing = [n for n in names if n not in lookup.index]
    if missing:
        raise SystemExit(f"Not found in the splits: {', '.join(missing)}")
    return [int(lookup[n]) for n in names]


@torch.no_grad()
def predict(names, image_dir, baseline, siamese, bank, results, device):
    """
    Calibrated melanoma probability from both models, plus each image's
    nearest melanoma and benign reference images (name and distance).
    """
    preprocess = get_transforms(train=False)
    images = torch.stack([preprocess(Image.open(image_dir / f"{n}.jpg").convert("RGB"))
                          for n in names]).to(device)

    cal = results["baseline"]["calibration"]
    baseline_prob = sigmoid(cal["a"] * baseline(images).cpu().numpy() + cal["b"])

    embeddings = siamese.embed(images)
    ref_emb, ref_labels = bank["embeddings"].to(device), bank["labels"].to(device)
    mel_dist, ben_dist = class_distances(embeddings, ref_emb, ref_labels, bank["k"])
    cal = results["siamese"]["calibration"]
    siamese_prob = sigmoid(cal["a"] * (ben_dist - mel_dist).cpu().numpy() + cal["b"])

    distances = torch.cdist(embeddings, ref_emb).cpu()
    labels = bank["labels"]
    neighbours = []
    for row in distances:
        entry = {}
        for cls, key in ((1, "melanoma"), (0, "benign")):
            idx = torch.nonzero(labels == cls).squeeze(1)
            nearest = row[idx].topk(N_SHOWN, largest=False)
            entry[key] = [(bank["image_names"][int(idx[i])], float(d))
                          for i, d in zip(nearest.indices.tolist(), nearest.values.tolist())]
        neighbours.append(entry)
    return baseline_prob, siamese_prob, neighbours


def decisions(probs, model_results):
    """Routing decision for each probability, using the thresholds chosen on validation."""
    t = model_results["thresholds"]
    return route(np.asarray(probs), t["clear"], t["urgent"])

def print_predictions(names, labels, reasons, probs, decided, neighbours):
    print(f"\n{'Image':14}{'Truth':10}{'Baseline':>10}  {'decision':18}{'Siamese':>9}  "
          f"{'decision':18}{'nearest mel':>12}{'nearest ben':>12}")
    for i, name in enumerate(names):
        truth = "melanoma" if labels[i] else "benign"
        print(f"{name:14}{truth:10}{probs['baseline'][i]:>10.1%}  {decided['baseline'][i]:18}"
              f"{probs['siamese'][i]:>9.1%}  {decided['siamese'][i]:18}"
              f"{neighbours[i]['melanoma'][0][1]:>12.3f}{neighbours[i]['benign'][0][1]:>12.3f}")
        if reasons[i]:
            print(f"{'':14}({reasons[i]})")


def main():
    args = parse_args()
    device = get_device()
    out_dir = Path(args.out_dir)
    prefix = "quicktest_" if args.quicktest else ""
    image_dir = Path(args.data_dir).expanduser() / "train"

    with open(out_dir / f"{prefix}evaluation_results.json") as f:
        results = json.load(f)
    test_predictions = pd.read_csv(out_dir / f"{prefix}test_predictions.csv")
    bank = torch.load(out_dir / f"{prefix}siamese_reference_bank.pt")
    baseline, siamese = load_models(out_dir, device)

    if args.images:
        names, reasons = args.images, ["" for _ in args.images]
    else:
        names, reasons = pick_random(test_predictions, args.n, args.seed)
    labels = true_labels(names, args.splits_dir)

    baseline_prob, siamese_prob, neighbours = predict(
        names, image_dir, baseline, siamese, bank, results, device)
    probs = {"baseline": baseline_prob, "siamese": siamese_prob}
    decided = {m: decisions(probs[m], results[m]) for m in probs}

    print_predictions(names, labels, reasons, probs, decided, neighbours)


if __name__ == "__main__":
    main()

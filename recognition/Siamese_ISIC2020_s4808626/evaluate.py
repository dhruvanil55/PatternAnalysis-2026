"""
Evaluate the trained baseline CNN and Siamese network on ISIC 2020.

Everything that is tuned (calibration and thresholds) is fitted on the
validation split only. The test split is then scored once.

Run after both models are trained:
    python evaluate.py --data-dir ~/isic2020
The same evaluation can also be started from train.py:
    python train.py --data-dir ~/isic2020 --test
Quick check that it runs, on small subsets (the numbers are not meaningful):
    python evaluate.py --data-dir ~/isic2020 --limit 512

Standard methods used here (full references in README.md):
- Platt scaling, expected calibration error and reliability diagrams,
  as used by Guo et al. [4]
- Letting a model act only when confident and passing the rest on,
  from Geifman and El-Yaniv [5]
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save figures to files without opening a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from torch.utils.data import DataLoader

from dataset import IMAGE_FOLDER, LesionDataset, get_transforms
from modules import (BaselineClassifier, ContrastiveLoss, SiameseNetwork, class_distances,
                     count_parameters)
from utils import GRID, INK, MUTED, SERIES_COLOURS, get_device, loader_settings, set_seed

MAX_MISSED_MELANOMA = 0.05  # AC4: automatically cleared cases may miss at most 5% of melanomas
MODEL_NAMES = {"baseline": "Baseline CNN", "siamese": "Siamese network"}
MODEL_COLOURS = {"baseline": SERIES_COLOURS[0], "siamese": SERIES_COLOURS[1]}


def parse_args(argv=None):
    """
    Read the evaluation options.

    Args:
        argv: List of option strings when called from train.py --test, or None
            to read them from the command line.

    Returns:
        An argparse Namespace with the data, split and output folders, k, batch
        size, number of bootstrap resamples, seed and --limit.
    """
    parser = argparse.ArgumentParser(description="Evaluate both ISIC 2020 models")
    parser.add_argument("--data-dir", required=True, help="Dataset folder containing combined_images/")
    parser.add_argument("--splits-dir", default="splits", help="Folder with the split CSVs")
    parser.add_argument("--out-dir", default="outputs", help="Where the trained models are")
    parser.add_argument("--k", type=int, default=10,
                        help="Nearest references per class for the Siamese network")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=1000,
                        help="Resamples for the 95%% confidence intervals")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None,
                        help="Use a random subset of each split, for quick tests")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Loading data and models
# ---------------------------------------------------------------------------

def load_split(splits_dir, name, limit, out_dir):
    """
    Load one split, or a small fixed subset of it for quick tests.

    With --limit, a fixed random subset is written to outputs/ so quick tests
    never touch the real split files.

    Args:
        splits_dir: Folder with the split CSVs.
        name: "train", "val" or "test".
        limit: If set, the number of images to keep.
        out_dir: Folder for the quick-test subset CSV.

    Returns:
        A tuple (DataFrame of the split's rows, path to a CSV LesionDataset can read).
    """
    path = Path(splits_dir) / f"{name}.csv"
    df = pd.read_csv(path)
    if limit and limit < len(df):
        df = df.sample(n=limit, random_state=0).reset_index(drop=True)
        path = Path(out_dir) / f"quicktest_{name}.csv"
        df.to_csv(path, index=False)
    return df, path


def make_eval_loader(split_csv, image_dir, batch_size, workers):
    """
    Build a DataLoader that keeps images in file order with no augmentation.

    Keeping the order means each score lines up with the matching CSV row.

    Args:
        split_csv: Path to the split CSV.
        image_dir: Folder containing the images.
        batch_size: Images per batch.
        workers: Number of image loading processes.

    Returns:
        A torch DataLoader that yields (images, labels, names) batches.
    """
    dataset = LesionDataset(split_csv, image_dir, get_transforms(train=False))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, **loader_settings(workers))


def load_models(out_dir, device):
    """
    Load the best saved weights of both models, ready for prediction.

    Args:
        out_dir: Folder containing baseline_best.pt and siamese_best.pt.
        device: Device to move the models to.

    Returns:
        A tuple (baseline, siamese), both in evaluation mode.
    """
    baseline = BaselineClassifier(pretrained=False)
    baseline.load_state_dict(torch.load(Path(out_dir) / "baseline_best.pt", map_location="cpu"))
    siamese = SiameseNetwork(pretrained=False)
    siamese.load_state_dict(torch.load(Path(out_dir) / "siamese_best.pt", map_location="cpu"))
    return baseline.to(device).eval(), siamese.to(device).eval()


# ---------------------------------------------------------------------------
# Raw scores: higher always means more likely melanoma
# ---------------------------------------------------------------------------

@torch.no_grad()
def baseline_scores(model, loader, device):
    """
    Get the baseline's raw score (logit) for every image.

    Args:
        model: The trained BaselineClassifier.
        loader: DataLoader from make_eval_loader.
        device: Device to run on.

    Returns:
        NumPy array with one logit per image, in file order.
    """
    return torch.cat([model(images.to(device)).cpu() for images, _, _ in loader]).numpy()


@torch.no_grad()
def embed_images(model, loader, device):
    """
    Get the Siamese embedding of every image.

    Args:
        model: The trained SiameseNetwork.
        loader: DataLoader from make_eval_loader.
        device: Device to run on.

    Returns:
        Tensor of embeddings in file order, kept on the device.
    """
    return torch.cat([model.embed(images.to(device)) for images, _, _ in loader])


@torch.no_grad()
def siamese_scores(query_emb, ref_emb, ref_labels, k, chunk=512):
    """
    Compute the Siamese score: benign distance minus melanoma distance.

    A positive score means the image sits closer to the melanoma references.
    Queries are processed in chunks to keep memory use low.

    Args:
        query_emb: Embeddings of the images to score.
        ref_emb: Embeddings of the training (reference) images.
        ref_labels: Labels of the reference images.
        k: Number of nearest references used per class.
        chunk: Number of queries processed at a time.

    Returns:
        NumPy array with one score per query.
    """
    gaps = []
    for start in range(0, len(query_emb), chunk):
        mel_dist, ben_dist = class_distances(query_emb[start:start + chunk], ref_emb, ref_labels, k)
        gaps.append((ben_dist - mel_dist).cpu())
    return torch.cat(gaps).numpy()

def sigmoid(x):
    """
    Squash any number into a probability between 0 and 1.

    Args:
        x: NumPy array or number.

    Returns:
        1 / (1 + exp(-x)), with the same shape as x.
    """
    return 1.0 / (1.0 + np.exp(-x))


def fit_calibration(scores, labels):
    """
    Fit Platt scaling: probability = sigmoid(a * score + b), on the validation split [4].

    The same method is used for both models, so their probabilities can be
    compared fairly. It is a logistic regression with one input and almost no
    regularisation.

    Args:
        scores: Raw validation scores (baseline logits or Siamese distance gaps).
        labels: Validation labels, 0 or 1.

    Returns:
        A tuple (a, b) of floats.
    """
    fit = LogisticRegression(C=1e6, max_iter=1000)  # very weak regularisation
    fit.fit(scores.reshape(-1, 1), labels)
    return float(fit.coef_[0, 0]), float(fit.intercept_[0])


def balanced_threshold(probs, labels):
    """
    Find the probability threshold that best balances melanoma and benign recall.

    This maximises melanoma recall plus benign recall (Youden's index on the
    ROC curve), which is the same as maximising balanced accuracy.

    Args:
        probs: Calibrated validation probabilities.
        labels: Validation labels, 0 or 1.

    Returns:
        The threshold as a float.
    """
    fpr, tpr, thresholds = roc_curve(labels, probs)
    return float(thresholds[np.argmax(tpr - fpr)])


def clear_threshold(probs, labels, max_missed=MAX_MISSED_MELANOMA):
    """
    Find the auto-clear threshold for AC4.

    This is the highest threshold t such that clearing every image with a
    probability below t misses at most `max_missed` of the melanomas.

    Args:
        probs: Calibrated validation probabilities.
        labels: Validation labels, 0 or 1.
        max_missed: Largest allowed share of melanomas below the threshold.

    Returns:
        The threshold as a float, or 1.0 if the limit allows every melanoma.
    """
    melanoma_probs = np.sort(probs[labels == 1])
    allowed = int(np.floor(max_missed * len(melanoma_probs)))  # melanomas we may clear
    return float(melanoma_probs[allowed]) if allowed < len(melanoma_probs) else 1.0


def route(probs, t_clear, t_urgent):
    """
    Sort each image into one of three groups, based on selective classification [5].

    auto-clear (below t_clear), urgent review (at or above t_urgent), otherwise
    routine specialist review. t_urgent is the model's balanced threshold, so
    urgent review means the model predicts melanoma. Images between t_clear and
    t_urgent are the uncertain cases that go to specialist review.

    Args:
        probs: Calibrated melanoma probabilities.
        t_clear: Auto-clear threshold.
        t_urgent: Urgent review threshold.

    Returns:
        NumPy array of decision strings, one per image.
    """
    decisions = np.full(len(probs), "specialist review", dtype=object)
    decisions[probs < t_clear] = "auto-clear"
    decisions[probs >= t_urgent] = "urgent review"
    return decisions

def expected_calibration_error(probs, labels, n_bins=10):
    """
    Expected calibration error (ECE) over equal-width probability bins [4].

    The average gap between the predicted probability and the observed melanoma
    rate in each bin, weighted by the number of images in the bin.

    Args:
        probs: Predicted probabilities.
        labels: True labels, 0 or 1.
        n_bins: Number of bins between 0 and 1.

    Returns:
        The ECE as a float, where 0 means perfectly calibrated.
    """
    bins = np.clip(np.digitize(probs, np.linspace(0, 1, n_bins + 1)[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        in_bin = bins == b
        if in_bin.any():
            ece += in_bin.mean() * abs(probs[in_bin].mean() - labels[in_bin].mean())
    return float(ece)


def routing_summary(probs, labels, t_clear, t_urgent):
    """
    Summarise how a set of images is routed.

    Args:
        probs: Calibrated melanoma probabilities.
        labels: True labels, 0 or 1.
        t_clear: Auto-clear threshold.
        t_urgent: Urgent review threshold.

    Returns:
        A dict with the share of images in each group, the number and share of
        melanomas that were auto-cleared, and the share of urgent cases that
        were melanoma.
    """
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


def test_metrics(raw_probs, probs, scores, labels, t_balanced, t_clear, t_urgent, calibration):
    """
    Compute every test metric for one model.

    Classification metrics use the balanced threshold. Confidence is how far
    each calibrated score is from that threshold in log-odds, and its AUROC
    measures how well it separates correct predictions from incorrect ones.

    Args:
        raw_probs: Probabilities before calibration.
        probs: Calibrated probabilities.
        scores: Raw scores the calibration was applied to.
        labels: True labels, 0 or 1.
        t_balanced: Balanced threshold from validation.
        t_clear: Auto-clear threshold from validation.
        t_urgent: Urgent review threshold from validation.
        calibration: The (a, b) Platt scaling values.

    Returns:
        A dict with AUROC, PR-AUC, accuracy measures, per-class precision and
        recall, macro-F1, the confusion matrix, ECE before and after
        calibration, confidence AUROC for correctness, and the routing summary.
    """
    preds = (probs >= t_balanced).astype(int)
    correct = (preds == labels).astype(int)
    # Confidence = distance from the decision threshold in log-odds.
    a, b = calibration
    confidence = np.abs((a * scores + b) - np.log(t_balanced / (1 - t_balanced)))
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        "auroc": float(roc_auc_score(labels, probs)),
        "pr_auc": float(average_precision_score(labels, probs)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "melanoma_precision": float(precision_score(labels, preds, zero_division=0)),
        "melanoma_recall": float(recall_score(labels, preds, zero_division=0)),
        "benign_precision": float(precision_score(labels, preds, pos_label=0, zero_division=0)),
        "benign_recall": float(recall_score(labels, preds, pos_label=0, zero_division=0)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "ece_raw": expected_calibration_error(raw_probs, labels),
        "ece_calibrated": expected_calibration_error(probs, labels),
        # How well confidence separates correct from wrong decisions (0.5 = not at all).
        "confidence_auroc_for_correctness": (float(roc_auc_score(correct, confidence))
                                             if 0 < correct.sum() < len(correct) else float("nan")),
        "routing": routing_summary(probs, labels, t_clear, t_urgent),
    }


def bootstrap_intervals(test_df, probs, thresholds, n_resamples, seed=0):
    """
    95% confidence intervals by resampling test patients with replacement.

    Patients rather than images are resampled, because images from one patient
    are not independent. The thresholds stay fixed. The interval for the AUROC
    difference between the two models shows whether that difference is real or
    could be noise.

    Args:
        test_df: Test split rows, with patient_id and target columns.
        probs: Dict of calibrated test probabilities for "baseline" and "siamese".
        thresholds: Dict of each model's balanced and auto-clear thresholds.
        n_resamples: Number of bootstrap resamples.
        seed: Seed for the resampling.

    Returns:
        A dict mapping each measure name to [lower, upper] bounds.
    """
    labels = test_df["target"].to_numpy().astype(int)
    patients = list(test_df.groupby("patient_id").indices.values())
    rng = np.random.default_rng(seed)

    samples = {name: [] for name in ["baseline_auroc", "siamese_auroc", "auroc_difference",
                                     "baseline_balanced_accuracy", "siamese_balanced_accuracy",
                                     "baseline_auto_clear_share", "siamese_auto_clear_share",
                                     "baseline_melanoma_miss_rate", "siamese_melanoma_miss_rate"]}
    for _ in range(n_resamples):
        rows = np.concatenate([patients[i] for i in rng.integers(0, len(patients), len(patients))])
        y = labels[rows]
        if y.sum() == 0 or y.sum() == len(y):
            continue
        aurocs = {}
        for model in ("baseline", "siamese"):
            p = probs[model][rows]
            t_bal, t_clear = thresholds[model]["balanced"], thresholds[model]["clear"]
            aurocs[model] = roc_auc_score(y, p)
            samples[f"{model}_auroc"].append(aurocs[model])
            samples[f"{model}_balanced_accuracy"].append(
                balanced_accuracy_score(y, (p >= t_bal).astype(int)))
            samples[f"{model}_auto_clear_share"].append(float((p < t_clear).mean()))
            samples[f"{model}_melanoma_miss_rate"].append(float((p[y == 1] < t_clear).mean()))
        samples["auroc_difference"].append(aurocs["siamese"] - aurocs["baseline"])

    return {name: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            for name, v in samples.items()}

def synchronise(device):
    """
    Wait for the GPU to finish its queued work, so timings are accurate.

    Args:
        device: The device in use. Does nothing on the CPU.
    """
    if device.type == "mps":
        torch.mps.synchronize()

def peak_training_memory_gb(kind, device, batch_size, steps=5):
    """
    Measure peak GPU memory over a few training steps on random images.

    Uses the same batch size as real training. Only measured on the Apple GPU,
    where it is the memory the Metal driver holds for this process.

    Args:
        kind: "baseline" or "siamese".
        device: Device to measure on.
        batch_size: Images (or pairs) per training step.
        steps: Number of training steps to run.

    Returns:
        Peak memory in GB, or None when not running on the Apple GPU.
    """
    if device.type != "mps":
        return None
    torch.mps.empty_cache()
    model = (BaselineClassifier(pretrained=False) if kind == "baseline"
             else SiameseNetwork(pretrained=False)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    peak = 0
    for _ in range(steps):
        images = torch.randn(batch_size, 3, 224, 224, device=device)
        targets = torch.randint(0, 2, (batch_size,), device=device).float()
        if kind == "baseline":
            loss = nn.BCEWithLogitsLoss()(model(images), targets)
        else:
            other = torch.randn(batch_size, 3, 224, 224, device=device)
            emb_a, emb_b = model(images, other)
            loss = ContrastiveLoss()(emb_a, emb_b, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        synchronise(device)
        peak = max(peak, torch.mps.driver_allocated_memory())
    del model, optimizer
    torch.mps.empty_cache()
    return peak / 1e9

@torch.no_grad()
def milliseconds_per_image(predict_one, images, warmup=10):
    """
    Measure the average time to predict one image, after a few warm-up runs.

    Args:
        predict_one: Function that predicts a single image.
        images: List of single-image tensors already on the device.
        warmup: Number of images used only to warm up.

    Returns:
        The average time per image in milliseconds.
    """
    for image in images[:warmup]:
        predict_one(image)
    start = time.perf_counter()
    for image in images[warmup:]:
        predict_one(image)
    return (time.perf_counter() - start) / len(images[warmup:]) * 1000

def style_axes(ax, xlabel, ylabel, title):
    """
    Apply the shared figure style to one set of axes.

    Sets the title, axis labels, light grid and legend, and hides the top and
    right borders.

    Args:
        ax: Matplotlib axes.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        title: Panel title.
    """
    ax.set_title(title, color=INK, loc="left")
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)

def plot_roc_pr(labels, probs, results, out_path):
    """
    Plot ROC and precision-recall curves for both models on the test split.

    Args:
        labels: Test labels, 0 or 1.
        probs: Dict of calibrated test probabilities for each model.
        results: Results dict, used for the AUROC and PR-AUC in the legend.
        out_path: Where to save the PNG figure.
    """
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(11, 4.5))
    for model in ("baseline", "siamese"):
        name, colour, r = MODEL_NAMES[model], MODEL_COLOURS[model], results[model]["test"]
        fpr, tpr, _ = roc_curve(labels, probs[model])
        ax_roc.plot(fpr, tpr, color=colour, linewidth=2, label=f"{name} (AUROC {r['auroc']:.3f})")
        precision, recall, _ = precision_recall_curve(labels, probs[model])
        ax_pr.plot(recall, precision, color=colour, linewidth=2,
                   label=f"{name} (PR-AUC {r['pr_auc']:.3f})")
    ax_roc.plot([0, 1], [0, 1], color=MUTED, linestyle=":", linewidth=1, label="Random guess")
    ax_pr.axhline(labels.mean(), color=MUTED, linestyle=":", linewidth=1,
                  label=f"Random guess ({labels.mean():.3f})")
    style_axes(ax_roc, "False positive rate (benign flagged)", "True positive rate (melanoma caught)",
               "ROC curve")
    style_axes(ax_pr, "Recall (melanoma caught)", "Precision (flags that are melanoma)",
               "Precision-recall curve")
    fig.suptitle("Test split: how well each model ranks melanoma above benign",
                 color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def reliability_points(probs, labels, n_bins=10):
    """
    Points for a reliability diagram, using bins with equal numbers of images.

    Args:
        probs: Predicted probabilities.
        labels: True labels, 0 or 1.
        n_bins: Number of bins.

    Returns:
        A tuple (mean predicted probability per bin, observed melanoma rate per bin).
    """
    order = np.argsort(probs)
    groups = np.array_split(order, n_bins)
    return ([probs[g].mean() for g in groups], [labels[g].mean() for g in groups])

def plot_reliability(labels, raw_probs, probs, results, out_path):
    """
    Plot reliability diagrams before and after calibration, one panel per model.

    Bins hold equal numbers of images, because most images have low
    probabilities and equal-width bins would leave the upper bins nearly empty.

    Args:
        labels: Test labels, 0 or 1.
        raw_probs: Dict of uncalibrated test probabilities for each model.
        probs: Dict of calibrated test probabilities for each model.
        results: Results dict, used for the ECE values in the legend.
        out_path: Where to save the PNG figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, model in zip(axes, ("baseline", "siamese")):
        colour, r = MODEL_COLOURS[model], results[model]["test"]
        for p, style, marker, label in [
                (raw_probs[model], "--", "o", f"Before calibration (ECE {r['ece_raw']:.3f})"),
                (probs[model], "-", "s", f"After calibration (ECE {r['ece_calibrated']:.3f})")]:
            x, y = reliability_points(p, labels)
            ax.plot(x, np.maximum(y, 1e-3), linestyle=style, marker=marker, color=colour,
                    markerfacecolor="white" if style == "--" else colour,
                    linewidth=2, markersize=6, label=label)
        ax.plot([1e-3, 1], [1e-3, 1], color=MUTED, linestyle=":", linewidth=1,
                label="Perfect calibration")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1e-3, 1)
        ax.set_ylim(1e-3, 1)
        style_axes(ax, "Predicted melanoma probability", "Observed melanoma rate",
                   MODEL_NAMES[model])
    fig.suptitle("Test split: does the predicted probability match reality? "
                 "(10 equal-size groups, on the dotted line = trustworthy)",
                 color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def automation_curve(probs, labels):
    """
    Share of images auto-cleared against share of melanomas missed, for every threshold.

    Images are sorted from lowest to highest probability and cleared one at a
    time, which gives the risk-coverage curve.

    Args:
        probs: Calibrated melanoma probabilities.
        labels: True labels, 0 or 1.

    Returns:
        A tuple (share cleared, share of melanomas missed), both starting at 0.
    """
    sorted_labels = labels[np.argsort(probs)]
    cleared = np.arange(1, len(labels) + 1) / len(labels)
    missed = np.cumsum(sorted_labels) / labels.sum()
    return np.r_[0, cleared], np.r_[0, missed]


def plot_automation(labels, probs, results, out_path):
    """
    Plot the risk-coverage curve for both models, with the AC4 limit and each model's operating point.

    Args:
        labels: Test labels, 0 or 1.
        probs: Dict of calibrated test probabilities for each model.
        results: Results dict, used for each model's routing at its validation threshold.
        out_path: Where to save the PNG figure.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for model in ("baseline", "siamese"):
        cleared, missed = automation_curve(probs[model], labels)
        routing = results[model]["test"]["routing"]
        ax.plot(cleared, missed, color=MODEL_COLOURS[model], linewidth=2,
                label=MODEL_NAMES[model])
        ax.plot(routing["auto_clear_share"], routing["melanoma_miss_rate"], marker="o",
                linestyle="none", markersize=9, color=MODEL_COLOURS[model],
                markeredgecolor="white",
                label=f"{MODEL_NAMES[model]} at its validation threshold "
                      f"({routing['auto_clear_share']:.0%} cleared)")
    ax.axhline(MAX_MISSED_MELANOMA, color=MUTED, linestyle="--", linewidth=1,
               label="AC4 limit: 5% of melanomas missed")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.3)
    style_axes(ax, "Share of cases cleared automatically", "Share of melanomas missed",
               "Test split: safe automation")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def to_builtin(value):
    """
    Convert NumPy numbers inside nested dicts and lists so results can be saved as JSON.

    Args:
        value: A dict, list, NumPy number or plain value.

    Returns:
        The same structure with plain Python numbers.
    """
    if isinstance(value, dict):
        return {k: to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main(args=None):
    """
    Run the full evaluation of both models.

    Fits calibration and thresholds on the validation split, scores the test
    split once, computes confidence intervals and resource measurements, and
    saves evaluation_results.json, test_predictions.csv, the Siamese reference
    bank and the three test figures.

    Args:
        args: Options from parse_args, or None to read them from the command line.
    """
    if args is None:
        args = parse_args()
    set_seed(args.seed)
    device = get_device()
    out_dir = Path(args.out_dir)
    prefix = "quicktest_" if args.limit else ""
    figure_dir = out_dir if args.limit else Path("figures")
    figure_dir.mkdir(parents=True, exist_ok=True)
    image_dir = Path(args.data_dir).expanduser() / IMAGE_FOLDER
    print(f"Using device: {device}", flush=True)

    # Measure training memory first, before anything else is on the GPU.
    print("Profiling training memory...", flush=True)
    memory = {kind: peak_training_memory_gb(kind, device, args.batch_size)
              for kind in ("baseline", "siamese")}

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
    torch.save({"embeddings": ref_emb.cpu(), "labels": ref_labels.cpu(),
                "image_names": splits["train"][0]["image_name"].tolist(), "k": args.k},
               out_dir / f"{prefix}siamese_reference_bank.pt")

    results, probs, raw_probs, thresholds = {}, {}, {}, {}
    for model in ("baseline", "siamese"):
        # Fit calibration and thresholds on validation only.
        calibration = fit_calibration(scores[model]["val"], labels["val"])
        val_probs = sigmoid(calibration[0] * scores[model]["val"] + calibration[1])
        thresholds[model] = {"balanced": balanced_threshold(val_probs, labels["val"]),
                             "clear": clear_threshold(val_probs, labels["val"])}
        # Urgent when the model predicts melanoma, and never below the auto-clear line.
        thresholds[model]["urgent"] = max(thresholds[model]["balanced"],
                                          thresholds[model]["clear"])

        # Apply them once to the test split.
        test_scores = scores[model]["test"]
        probs[model] = sigmoid(calibration[0] * test_scores + calibration[1])
        raw_probs[model] = sigmoid(test_scores)  # what each model reports before calibration
        results[model] = {
            "calibration": {"a": calibration[0], "b": calibration[1]},
            "thresholds": thresholds[model],
            "validation_routing": routing_summary(val_probs, labels["val"],
                                                  thresholds[model]["clear"],
                                                  thresholds[model]["urgent"]),
            "test": test_metrics(raw_probs[model], probs[model], test_scores, labels["test"],
                                 thresholds[model]["balanced"], thresholds[model]["clear"],
                                 thresholds[model]["urgent"], calibration),
        }

    print(f"Bootstrapping {args.bootstrap} patient resamples...", flush=True)
    intervals = bootstrap_intervals(splits["test"][0], probs, thresholds, args.bootstrap)

    # Prediction time per image, including the Siamese comparison to every reference.
    test_set = loaders["test"].dataset
    timing_images = [test_set[i][0].unsqueeze(0).to(device) for i in range(min(60, len(test_set)))]
    timing = {
        "baseline": milliseconds_per_image(lambda x: baseline(x).item(), timing_images),
        "siamese": milliseconds_per_image(
            lambda x: siamese_scores(siamese.embed(x), ref_emb, ref_labels, args.k)[0],
            timing_images),
    }
    for model, net in (("baseline", baseline), ("siamese", siamese)):
        results[model]["resources"] = {"parameters": count_parameters(net),
                                       "peak_training_memory_gb": memory[model],
                                       "ms_per_image": timing[model]}
    results["confidence_intervals_95"] = intervals
    results["settings"] = {"k": args.k, "max_missed_melanoma": MAX_MISSED_MELANOMA,
                           "urgent_rule": "balanced threshold",
                           "bootstrap_resamples": args.bootstrap, "limit": args.limit,
                           "device": str(device)}

    with open(out_dir / f"{prefix}evaluation_results.json", "w") as f:
        json.dump(to_builtin(results), f, indent=2)

    test_df = splits["test"][0].copy()
    for model in ("baseline", "siamese"):
        test_df[f"{model}_prob"] = probs[model]
        test_df[f"{model}_decision"] = route(probs[model], thresholds[model]["clear"],
                                             thresholds[model]["urgent"])
    test_df.to_csv(out_dir / f"{prefix}test_predictions.csv", index=False)

    plot_roc_pr(labels["test"], probs, results, figure_dir / f"{prefix}test_roc_pr.png")
    plot_reliability(labels["test"], raw_probs, probs, results,
                     figure_dir / f"{prefix}test_reliability.png")
    plot_automation(labels["test"], probs, results, figure_dir / f"{prefix}test_automation.png")

    print_summary(results)


def print_summary(results):
    """
    Print the headline comparison table and the AUROC difference interval.

    Args:
        results: The results dict built by main.
    """
    ci = results["confidence_intervals_95"]
    rows = [
        ("AUROC", "auroc", "{:.3f}"), ("PR-AUC", "pr_auc", "{:.3f}"),
        ("Balanced accuracy", "balanced_accuracy", "{:.3f}"),
        ("Melanoma recall", "melanoma_recall", "{:.3f}"),
        ("Melanoma precision", "melanoma_precision", "{:.3f}"),
        ("Macro-F1", "macro_f1", "{:.3f}"),
        ("ECE before calibration", "ece_raw", "{:.3f}"),
        ("ECE after calibration", "ece_calibrated", "{:.3f}"),
        ("Confidence AUROC for correctness", "confidence_auroc_for_correctness", "{:.3f}"),
    ]
    print("\nTest results (thresholds chosen on validation)")
    print(f"{'':36}{'Baseline':>12}{'Siamese':>12}")
    for label, key, fmt in rows:
        print(f"{label:36}{fmt.format(results['baseline']['test'][key]):>12}"
              f"{fmt.format(results['siamese']['test'][key]):>12}")
    for label, key in [("Auto-cleared share", "auto_clear_share"),
                       ("Melanoma miss rate (AC4 <= 0.05)", "melanoma_miss_rate"),
                       ("Specialist review share", "specialist_review_share"),
                       ("Urgent review share", "urgent_review_share"),
                       ("Melanoma share of urgent cases", "melanoma_share_of_urgent")]:
        print(f"{label:36}{results['baseline']['test']['routing'][key]:>12.3f}"
              f"{results['siamese']['test']['routing'][key]:>12.3f}")
    for label, key, fmt in [("Parameters", "parameters", "{:,}"),
                            ("Peak training memory (GB)", "peak_training_memory_gb", "{:.2f}"),
                            ("Prediction time (ms per image)", "ms_per_image", "{:.1f}")]:
        values = [results[m]["resources"][key] for m in ("baseline", "siamese")]
        print(f"{label:36}" + "".join(f"{fmt.format(v) if v is not None else 'n/a':>12}"
                                      for v in values))
    low, high = ci["auroc_difference"]
    print(f"\nSiamese minus baseline AUROC, 95% interval: {low:+.3f} to {high:+.3f}")
    print("(If the interval includes 0, the difference may be noise.)")


if __name__ == "__main__":
    main()

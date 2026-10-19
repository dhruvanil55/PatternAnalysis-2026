"""
Shared helpers used by train.py and predict.py.
"""
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save figures to files without opening a window
import matplotlib.pyplot as plt
import pandas as pd
import torch

# Fixed series colours (blue, orange, aqua, yellow) so each metric keeps
# the same colour in every figure.
SERIES_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID = "#2b2b2b", "#6b6b6b", "#e4e4e4"


def get_device():
    """Use the Mac's Apple GPU (MPS) if available, otherwise the CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    """
    Fix the random number generators so runs can be repeated.
    """
    random.seed(seed)
    torch.manual_seed(seed)


def loader_settings(num_workers=4):
    """
    DataLoader options. num_workers loads images in parallel on the CPU
    so the GPU is not left waiting for the next batch.
    """
    return {"num_workers": num_workers, "persistent_workers": num_workers > 0}


def plot_history(history_csv, out_path, title):
    """
    Plot training curves from a history CSV written by train.py.

    Left panel: training and validation loss. Right panel: validation
    metrics with the 0.80 balanced accuracy target (AC1). A dotted line
    marks the epoch with the best validation AUROC, which is the saved model.
    """
    history = pd.read_csv(history_csv)
    epochs = history["epoch"]
    best_epoch = int(history.loc[history["val_auroc"].idxmax(), "epoch"])

    fig, (ax_loss, ax_metrics) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax_loss.plot(epochs, history["train_loss"], marker="o", color=SERIES_COLOURS[0],
                 linewidth=2, markersize=6, label="Training loss")
    ax_loss.plot(epochs, history["val_loss"], marker="s", color=SERIES_COLOURS[1],
                 linewidth=2, markersize=6, label="Validation loss")
    ax_loss.set_title("Loss", color=INK, loc="left")
    ax_loss.set_ylabel("Weighted BCE loss", color=MUTED)

    metrics = [("val_auroc", "AUROC", "o"), ("val_pr_auc", "PR-AUC", "s"),
               ("val_balanced_acc", "Balanced accuracy", "^"),
               ("val_melanoma_recall", "Melanoma recall", "D")]
    for colour, (column, label, marker) in zip(SERIES_COLOURS, metrics):
        ax_metrics.plot(epochs, history[column], marker=marker, color=colour,
                        linewidth=2, markersize=6, label=label)
    ax_metrics.axhline(0.80, color=MUTED, linestyle="--", linewidth=1,
                       label="AC1 target (0.80)")
    ax_metrics.set_ylim(0, 1)
    ax_metrics.set_title("Validation metrics (threshold 0.5)", color=INK, loc="left")
    ax_metrics.set_ylabel("Score", color=MUTED)

    for ax in (ax_loss, ax_metrics):
        ax.axvline(best_epoch, color=MUTED, linestyle=":", linewidth=1.2)
        ax.set_xlabel("Epoch", color=MUTED)
        ax.set_xticks(epochs)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK)

    fig.suptitle(f"{title}: best validation AUROC at epoch {best_epoch} (dotted line)",
                 color=INK, x=0.01, ha="left")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    # Quick checks. Run: python utils.py
    device = get_device()
    print(f"Device: {device}")

    # The same seed must give the same random numbers.
    set_seed(42)
    first = torch.rand(3)
    set_seed(42)
    second = torch.rand(3)
    assert torch.equal(first, second), "Seeding is not repeatable"
    print(f"Seed check passed: {first.tolist()}")

    # A small calculation on the chosen device.
    x = torch.ones(2, 2, device=device)
    print(f"Calculation on {device}: {(x @ x).tolist()}")
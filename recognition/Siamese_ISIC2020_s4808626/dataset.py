"""
Data loading and patient-level splitting for the ISIC 2020 melanoma dataset.

Run directly to create the train, validation and test splits:
    python dataset.py --data-dir ~/data/isic2020
"""
import argparse
from pathlib import Path

import pandas as pd

# Share of patients in each split. Must add up to 1.
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create patient-level ISIC 2020 splits")
    parser.add_argument("--data-dir", required=True, help="Folder containing train.csv")
    parser.add_argument("--out-dir", default="splits", help="Where to save the split CSVs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = load_metadata(args.data_dir)
    data = split_patients(data, seed=args.seed)
    check_no_leakage(data)
    summarise(data)
    save_splits(data, args.out_dir)
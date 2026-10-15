"""
Train the baseline CNN classifier on ISIC 2020.
""""
import argparse


from utils import get_device, set_seed
from pathlib import Path

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


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    image_dir = Path(args.data_dir).expanduser() / "train"
    train_csv = Path(args.splits_dir) / "train.csv"

   
if __name__ == "__main__":
    main()
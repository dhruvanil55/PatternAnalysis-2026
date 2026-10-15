"""
Shared helpers used by train.py and predict.py.
"""
import random

import torch


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
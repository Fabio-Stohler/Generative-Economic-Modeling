"""Lightweight persistence helpers."""

# Standard library
from pathlib import Path
import pickle


def save_pickle(obj, path, name="model"):
    """Persist any picklable object to disk."""
    Path(path).mkdir(parents=True, exist_ok=True)
    with open(f"{path}/{name}.pkl", "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    """Load a pickled object from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


def load_dataset(path):
    """Load a dataset saved either as a dict with keys like 'x'/'y' or a BMModel with .dataset."""
    obj = load_pickle(path)
    if isinstance(obj, dict) and "x" in obj:
        return obj
    if hasattr(obj, "dataset"):
        return obj.dataset
    raise ValueError(f"Unexpected dataset format in {path}")

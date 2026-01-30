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

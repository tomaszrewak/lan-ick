"""Simple file-based caching for experiment artifacts.

Usage:
    from src.cache import cached

    data = cached("training_data", "v1", generate_training_data)
    # First call: runs generate_training_data(), saves to temp/data/training_data__v1.pkl
    # Later calls: loads from disk instantly.
    # Bump "v1" → "v2" when generate_training_data changes; old cache is ignored.
"""

import pickle
from pathlib import Path
from typing import TypeVar, Callable

T = TypeVar("T")

TEMP_DIR = Path(__file__).resolve().parent.parent / "temp"


def cached(name: str, version: str, fn: Callable[[], T], *, subdir: str = "data") -> T:
    """Load from cache or generate and cache.

    Args:
        name: Human-readable name for the cached artifact.
        version: Bump this when the generator function changes.
        fn: Zero-arg callable that produces the data.
        subdir: Subfolder under temp/ (default: "data").

    Returns:
        The cached or freshly generated data.
    """
    cache_dir = TEMP_DIR / subdir
    path = cache_dir / f"{name}__{version}.pkl"

    if path.exists():
        print(f"[cache] Loading {name} ({version}) from {path.relative_to(TEMP_DIR)}")
        with open(path, "rb") as f:
            return pickle.load(f)

    print(f"[cache] Generating {name} ({version})...")
    result = fn()

    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(result, f)
    print(f"[cache] Saved to {path.relative_to(TEMP_DIR)}")

    return result

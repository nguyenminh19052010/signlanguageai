
from __future__ import annotations

from pathlib import Path

def count_processed_clips(dataset_root: Path) -> int:
    root = Path(dataset_root)
    if not root.is_dir():
        return 0
    n = 0
    for d in root.iterdir():
        if d.is_dir():
            n += len(list(d.glob("*.npz")))
    return n

def count_processed_classes(dataset_root: Path) -> int:
    root = Path(dataset_root)
    if not root.is_dir():
        return 0
    return sum(1 for d in root.iterdir() if d.is_dir() and any(d.glob("*.npz")))

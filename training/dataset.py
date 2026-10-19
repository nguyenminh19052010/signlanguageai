
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

class LandmarkSequenceDataset(Dataset):

    def __init__(
        self,
        root: Path,
        class_names: List[str],
        seq_length: int,
        augment: bool = False,
    ) -> None:
        self.root = Path(root)
        self.class_names = class_names
        self.seq_length = seq_length
        self.augment = augment
        self.samples: List[Tuple[Path, int]] = []
        for ci, cls in enumerate(class_names):
            folder = self.root / cls
            if not folder.is_dir():
                continue
            for fp in sorted(folder.glob("*.npz")):
                self.samples.append((fp, ci))

    def __len__(self) -> int:
        return len(self.samples)

    def _resample(self, seq: np.ndarray) -> np.ndarray:
        t, f = seq.shape
        if t == self.seq_length:
            return seq.astype(np.float32)
        x_old = np.linspace(0, 1, num=t, dtype=np.float32)
        x_new = np.linspace(0, 1, num=self.seq_length, dtype=np.float32)
        out = np.empty((self.seq_length, f), dtype=np.float32)
        for i in range(f):
            out[:, i] = np.interp(x_new, x_old, seq[:, i])
        return out

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        data = np.load(path)
        seq = data["landmarks"].astype(np.float32)

        if self.augment:
            noise = np.random.normal(0, 0.01, size=seq.shape).astype(np.float32)
            seq = seq + noise
            scale = float(np.random.uniform(0.92, 1.08))
            seq *= scale

        seq = self._resample(seq)
        x = torch.from_numpy(seq)
        y = torch.tensor(label, dtype=torch.long)
        return x, y

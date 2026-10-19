
from __future__ import annotations

from enum import Enum
from typing import Literal

import numpy as np

LANDMARK_DIM = 126

class AdapterMode(str, Enum):
    AUTO = "auto"
    DIRECT_126 = "direct_126"
    LAST_126 = "last_126"
    HOLISTIC_HANDS = "holistic_hands"

_HOLISTIC_LH_START = 99 + 468 * 3
_HOLISTIC_RH_START = _HOLISTIC_LH_START + 63

def frame_to_126(
    frame: np.ndarray,
    mode: Literal["auto", "direct_126", "last_126", "holistic_hands"] = "auto",
) -> np.ndarray:

    v = np.asarray(frame, dtype=np.float32).reshape(-1)
    f = v.shape[0]

    if mode == "direct_126" or (mode == "auto" and f == LANDMARK_DIM):
        return v.astype(np.float32)

    if mode == "last_126" or (mode == "auto" and f >= LANDMARK_DIM):
        if f == LANDMARK_DIM:
            return v
        if f >= LANDMARK_DIM:
            return v[-LANDMARK_DIM:].astype(np.float32)

    if mode == "holistic_hands" or mode == "auto":
        for lh_start, rh_start in (
            (_HOLISTIC_LH_START, _HOLISTIC_RH_START),
            (1503, 1566),
            (1479, 1542),
        ):
            rh_end = rh_start + 63
            if rh_end <= f:
                hands = np.concatenate([v[lh_start:lh_start + 63], v[rh_start:rh_end]])
                if hands.shape[0] == LANDMARK_DIM:
                    return hands.astype(np.float32)

    raise ValueError(
        f"Cannot adapt feature dim {f} → {LANDMARK_DIM}. "
        "Thử mode='last_126' hoặc trích xuất lại từ video (prepare --format video_folder)."
    )

def sequence_to_126(
    seq: np.ndarray,
    mode: str = "auto",
) -> np.ndarray:

    seq = np.asarray(seq, dtype=np.float32)
    if seq.ndim != 2:
        raise ValueError(f"Expected 2D sequence, got shape {seq.shape}")
    out = np.stack([frame_to_126(seq[t], mode=mode) for t in range(seq.shape[0])], axis=0)
    return out.astype(np.float32)


from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

def bilateral_denoise(frame_bgr: np.ndarray, d: int = 5) -> np.ndarray:

    return cv2.bilateralFilter(frame_bgr, d=d, sigmaColor=50, sigmaSpace=50)

def resize_keep_aspect(
    frame_bgr: np.ndarray,
    width: int,
    height: int,
) -> Tuple[np.ndarray, float]:

    h, w = frame_bgr.shape[:2]
    scale = min(width / w, height / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    y0 = (height - nh) // 2
    x0 = (width - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas, scale


from __future__ import annotations

import sys
from typing import List, Optional, Tuple

import cv2

def backend_constant(name: str) -> Optional[int]:

    n = (name or "auto").strip().lower()
    if n in ("", "default", "auto"):
        if sys.platform == "win32":
            return cv2.CAP_DSHOW
        return None
    if n in ("dshow", "directshow"):
        return cv2.CAP_DSHOW
    if n in ("msmf", "mf"):
        return cv2.CAP_MSMF
    if n == "v4l2" and sys.platform.startswith("linux"):
        return cv2.CAP_V4L2
    return None

def open_capture(device_index: int, backend_name: str = "auto") -> cv2.VideoCapture:
    be = backend_constant(backend_name)
    if be is not None:
        return cv2.VideoCapture(device_index, be)
    return cv2.VideoCapture(device_index)

def open_first_working(
    preferred_index: int,
    backend_name: str = "auto",
    try_indices: Optional[List[int]] = None,
) -> Tuple[cv2.VideoCapture, int]:

    order = [preferred_index]
    if try_indices:
        for i in try_indices:
            if i not in order:
                order.append(i)

    last_cap = None
    for idx in order:
        cap = open_capture(idx, backend_name)
        last_cap = cap
        if cap.isOpened():
            return cap, idx

    if last_cap is None:
        last_cap = open_capture(preferred_index, backend_name)
    return last_cap, preferred_index

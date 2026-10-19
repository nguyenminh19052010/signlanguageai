
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

VOYA_SLUG = re.compile(r"^class_\d{4}$", re.IGNORECASE)

def is_voya_style_classes(classes: List[str]) -> bool:
    if not classes:
        return False
    n_voya = sum(1 for c in classes if VOYA_SLUG.match(c))
    return n_voya >= max(3, int(0.8 * len(classes)))

def inspect_checkpoint(path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "exists": path.is_file(),
        "source": "unknown",
        "is_voya": False,
        "val_acc": None,
        "num_classes": 0,
    }
    if not path.is_file():
        return out
    try:
        import torch

        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
    except Exception:
        return out

    classes = list(payload.get("classes") or [])
    out["num_classes"] = len(classes)
    out["val_acc"] = payload.get("val_acc")
    src = str(payload.get("train_source", "")).lower()
    out["source"] = src or ("voya" if is_voya_style_classes(classes) else "camera")
    out["is_voya"] = out["source"] == "voya" or is_voya_style_classes(classes)
    return out

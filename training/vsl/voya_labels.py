
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ID = "Kateht/VOYA_VSL"
CLASS_SLUG_RE = re.compile(r"^class_(\d+)$", re.IGNORECASE)

_cache: Optional[Dict[str, Any]] = None

def label_id_to_name(labels: Dict[str, Any], class_id: int) -> str:
    slug = f"class_{class_id:04d}"
    if slug in labels:
        return str(labels[slug])
    k = str(class_id)
    if k in labels:
        return str(labels[k])
    if isinstance(labels, list) and 0 <= class_id < len(labels):
        return str(labels[class_id])
    return slug

def load_voya_labels(
    voya_root: Optional[Path] = None,
    allow_hf_download: bool = True,
) -> Dict[str, Any]:

    global _cache
    if _cache is not None:
        return _cache

    candidates: List[Path] = []
    if voya_root is not None:
        candidates.append(Path(voya_root) / "labels.json")
    root = Path(__file__).resolve().parents[2]
    candidates.extend(
        [
            root / "datasets" / "raw" / "voya" / "labels.json",
            root / "configs" / "voya_labels.json",
        ]
    )

    for p in candidates:
        if p.is_file():
            with p.open("r", encoding="utf-8") as f:
                _cache = json.load(f)
            return _cache

    if allow_hf_download:
        try:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(
                REPO_ID,
                repo_type="dataset",
                filename="labels.json",
            )
            with open(path, "r", encoding="utf-8") as f:
                _cache = json.load(f)
            return _cache
        except Exception:
            pass

    _cache = {}
    return _cache

def enrich_labels_vi(
    classes: List[str],
    labels_vi: Optional[Dict[str, str]] = None,
    voya_root: Optional[Path] = None,
) -> Dict[str, str]:

    out = dict(labels_vi or {})
    voya = load_voya_labels(voya_root)
    if not voya:
        return out

    for slug in classes:
        if slug in voya:
            out[slug] = str(voya[slug])
            continue
        m = CLASS_SLUG_RE.match(slug)
        if not m:
            continue
        cid = int(m.group(1))
        name = label_id_to_name(voya, cid).strip()
        if name:
            out[slug] = name
    return out

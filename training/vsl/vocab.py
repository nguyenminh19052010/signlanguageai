
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

def slugify_vi(name: str) -> str:

    s = name.strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "class"

def save_vocab(
    path: Path,
    classes: List[str],
    labels_vi: Optional[Dict[str, str]] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    labels_vi = labels_vi or {c: c.replace("_", " ") for c in classes}
    payload = {
        "classes": classes,
        "labels_vi": labels_vi,
        "num_classes": len(classes),
        "meta": meta or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def load_vocab(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def display_name(vocab: Dict[str, Any], class_slug: str) -> str:
    return vocab.get("labels_vi", {}).get(class_slug, class_slug)

def rebuild_vocab_from_dataset(
    dataset_root: Path,
    vocab_path: Path,
    source: str = "camera",
) -> Dict[str, Any]:

    root = Path(dataset_root)
    classes: List[str] = []
    labels_vi: Dict[str, str] = {}
    for d in sorted(root.iterdir()):
        if d.is_dir() and any(d.glob("*.npz")):
            slug = d.name
            classes.append(slug)
            labels_vi[slug] = slug.replace("_", " ")
    n_clips = sum(len(list((root / c).glob("*.npz"))) for c in classes)
    save_vocab(
        vocab_path,
        classes,
        labels_vi,
        meta={"source": source, "dataset_root": str(root), "num_clips": n_clips},
    )
    return load_vocab(vocab_path)

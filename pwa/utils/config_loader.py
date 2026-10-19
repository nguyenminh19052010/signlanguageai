
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

def load_config(path: str | Path | None = None) -> Dict[str, Any]:

    env_path = os.environ.get("SIGNLANG_CONFIG")
    if env_path and Path(env_path).is_file():
        cfg_path = Path(env_path)
    elif path is not None and Path(path).is_file():
        cfg_path = Path(path)
    else:
        default = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
        cfg_path = default

    with cfg_path.open("r", encoding="utf-8") as f:
        data: Dict[str, Any] = yaml.safe_load(f)

    root = cfg_path.resolve().parent.parent
    data["_project_root"] = str(root)
    if "paths" in data:
        for key, val in list(data["paths"].items()):
            if isinstance(val, str) and not val.startswith(("/", "\\\\")):
                data["paths"][key] = str((root / val).resolve())

    vsl = data.get("vsl")
    if isinstance(vsl, dict):
        for key in ("vocab_file", "voya_root", "kaggle_raw_dir"):
            val = vsl.get(key)
            if isinstance(val, str) and val and not val.startswith(("/", "\\\\")):
                vsl[key] = str((root / val).resolve())

    _apply_vsl_vocab(root, data)
    return data

def _apply_vsl_vocab(root: Path, data: Dict[str, Any]) -> None:

    vsl = data.get("vsl") or {}
    if not vsl.get("enabled"):
        return
    vocab_rel = vsl.get("vocab_file", "configs/vocab.json")
    vocab_path = Path(vocab_rel)
    if not vocab_path.is_absolute():
        vocab_path = (root / vocab_rel).resolve()
    if not vocab_path.is_file():
        return
    import json

    with vocab_path.open("r", encoding="utf-8") as f:
        vocab = json.load(f)
    classes = vocab.get("classes") or []
    if classes:
        data.setdefault("gesture", {})["classes"] = classes
    if vocab.get("labels_vi"):
        data["vsl_labels_vi"] = vocab["labels_vi"]

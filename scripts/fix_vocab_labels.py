
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training.vsl.voya_labels import enrich_labels_vi, load_voya_labels

def main() -> None:
    vocab_path = ROOT / "configs" / "vocab.json"
    if not vocab_path.is_file():
        print("Không thấy", vocab_path)
        raise SystemExit(1)

    with vocab_path.open("r", encoding="utf-8") as f:
        vocab = json.load(f)

    classes = list(vocab.get("classes") or [])
    labels_vi = enrich_labels_vi(classes, vocab.get("labels_vi"))
    vocab["labels_vi"] = labels_vi

    with vocab_path.open("w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    sample = classes[:3]
    print("Đã cập nhật", vocab_path)
    for s in sample:
        print(f"  {s} → {labels_vi.get(s, '?')}")
    if not load_voya_labels():
        print("Cảnh báo: không tải được labels.json VOYA (cần Internet hoặc datasets/raw/voya/labels.json)")

if __name__ == "__main__":
    main()

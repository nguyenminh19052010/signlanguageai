
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.vsl.keypoint_adapter import sequence_to_126
from training.vsl.vocab import save_vocab, slugify_vi
from utils.config_loader import load_config

LABELS_URL = "https://huggingface.co/datasets/Kateht/VOYA_VSL/raw/main/labels.json"

def fetch_labels(voya_root: Path) -> dict:
    local = voya_root / "labels.json"
    if local.is_file():
        with local.open("r", encoding="utf-8") as f:
            return json.load(f)
    print(f"Downloading labels.json → {local}")
    urllib.request.urlretrieve(LABELS_URL, local)
    with local.open("r", encoding="utf-8") as f:
        return json.load(f)

def label_id_to_name(labels: dict, class_id: int) -> str:
    slug = f"class_{class_id:04d}"
    if slug in labels:
        return str(labels[slug])
    key = str(class_id)
    if key in labels:
        return str(labels[key])
    if isinstance(labels, list) and class_id < len(labels):
        return str(labels[class_id])
    return slug

def main() -> None:
    p = argparse.ArgumentParser(description="VOYA_VSL → processed landmarks 126-D")
    p.add_argument("--config", default=None)
    p.add_argument("--voya-root", type=str, default=None, help="Thư mục chứa Merged/")
    p.add_argument("--adapter", default="auto", choices=["auto", "last_126", "holistic_hands", "direct_126"])
    p.add_argument("--max-classes", type=int, default=0, help="0 = tất cả lớp trong Merged")
    p.add_argument("--max-samples-per-class", type=int, default=0, help="0 = không giới hạn")
    args = p.parse_args()

    cfg = load_config(args.config)
    vsl = cfg.get("vsl", {})
    voya_root = Path(args.voya_root or vsl.get("voya_root", "datasets/vsl_raw/voya"))
    merged = voya_root / "Merged"
    if not merged.is_dir():
        raise SystemExit(
            f"Không thấy {merged}. Tải VOYA_VSL:\n"
            "  python training/vsl/download_voya.py\n"
            "Hoặc copy thư mục Merged từ Kaggle vào đó."
        )

    out_root = Path(cfg["paths"]["dataset_root"])
    vocab_path = Path(vsl.get("vocab_file", "configs/vocab.json"))
    labels = fetch_labels(voya_root)

    class_files = sorted(merged.glob("*.npz"))
    if args.max_classes > 0:
        class_files = class_files[: args.max_classes]

    all_slugs: list[str] = []
    labels_vi: dict[str, str] = {}
    total_saved = 0

    per_class_saved: dict[str, int] = {}

    for cf in class_files:
        data = np.load(cf)
        sequences = data["sequences"]
        sample_labels = data.get("labels")
        if sample_labels is None:
            try:
                default_id = int(cf.stem.split("_")[-1])
            except ValueError:
                default_id = 0
            sample_labels = np.full(len(sequences), default_id)

        for i, (seq, lab) in enumerate(zip(sequences, sample_labels)):
            class_id = int(lab)
            vi_name = label_id_to_name(labels, class_id)
            slug = slugify_vi(vi_name)
            if slug not in all_slugs:
                all_slugs.append(slug)
            labels_vi[slug] = vi_name

            if slug not in per_class_saved:
                per_class_saved[slug] = 0
            if (
                args.max_samples_per_class > 0
                and per_class_saved[slug] >= args.max_samples_per_class
            ):
                continue

            try:
                arr = sequence_to_126(seq, mode=args.adapter)
            except ValueError as exc:
                print(f"  skip {cf.name}[{i}]: {exc}")
                continue

            out_dir = out_root / slug
            out_dir.mkdir(parents=True, exist_ok=True)
            n = per_class_saved[slug] + 1
            out_path = out_dir / f"seq_{n:05d}.npz"
            np.savez_compressed(out_path, landmarks=arr)
            per_class_saved[slug] = n
            total_saved += 1

    for slug in all_slugs:
        print(f"  {slug} ({labels_vi.get(slug, slug)}): {per_class_saved.get(slug, 0)} clips")

    save_vocab(
        vocab_path,
        all_slugs,
        labels_vi,
        meta={"source": "VOYA_VSL", "adapter": args.adapter, "total_clips": total_saved},
    )
    if total_saved == 0:
        raise SystemExit(
            f"\nKhông lưu được clip nào vào {out_root}.\n"
            "Nguyên nhân thường gặp:\n"
            "  • Chưa tải VOYA — thư mục Merged/ trống hoặc sai đường dẫn\n"
            "  • Lỗi chuyển keypoint — thử: --adapter last_126\n"
            "Tải dữ liệu:\n"
            "  python training/vsl/download_voya.py"
        )

    print(f"\nDone. {total_saved} clips → {out_root}")
    print(f"Vocab → {vocab_path} ({len(all_slugs)} classes)")

if __name__ == "__main__":
    main()

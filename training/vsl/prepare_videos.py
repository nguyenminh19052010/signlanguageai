
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cv.hands_detector import HandLandmarkExtractor
from training.vsl.vocab import save_vocab, slugify_vi
from utils.config_loader import load_config

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

def iter_class_dirs(raw_root: Path) -> list[tuple[Path, str]]:

    train = raw_root / "train"
    root = train if train.is_dir() else raw_root
    pairs = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and not d.name.startswith("."):
            pairs.append((d, d.name))
    return pairs

def extract_video(
    extractor: HandLandmarkExtractor,
    video_path: Path,
    min_frames: int,
) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            lm = extractor.process_frame(frame)
            if lm.confidence_ok:
                frames.append(lm.vector.copy())
    finally:
        cap.release()
    if len(frames) < min_frames:
        return None
    return np.stack(frames, axis=0).astype(np.float32)

def main() -> None:
    p = argparse.ArgumentParser(description="Video Kaggle VSL → landmarks 126-D")
    p.add_argument("--config", default=None)
    p.add_argument("--raw-root", type=str, default=None)
    p.add_argument("--min-frames", type=int, default=12)
    p.add_argument("--max-videos-per-class", type=int, default=0)
    args = p.parse_args()

    cfg = load_config(args.config)
    vsl = cfg.get("vsl", {})
    raw_root = Path(args.raw_root or vsl.get("kaggle_raw_dir", "datasets/vsl_raw/kaggle"))
    out_root = Path(cfg["paths"]["dataset_root"])
    vocab_path = Path(vsl.get("vocab_file", "configs/vocab.json"))

    if not raw_root.is_dir():
        raise SystemExit(f"Không thấy {raw_root}. Tải dataset Kaggle vào đây.")

    mp_cfg = cfg["mediapipe"]
    extractor = HandLandmarkExtractor(
        max_num_hands=int(mp_cfg["max_num_hands"]),
        model_complexity=int(mp_cfg["model_complexity"]),
        min_detection_confidence=float(mp_cfg["min_detection_confidence"]),
        min_tracking_confidence=float(mp_cfg["min_tracking_confidence"]),
    )

    all_slugs: list[str] = []
    labels_vi: dict[str, str] = {}
    total = 0

    try:
        for class_dir, class_name in iter_class_dirs(raw_root):
            slug = slugify_vi(class_name)
            if slug not in all_slugs:
                all_slugs.append(slug)
            labels_vi[slug] = class_name
            out_dir = out_root / slug
            out_dir.mkdir(parents=True, exist_ok=True)

            videos = [f for f in class_dir.iterdir() if f.suffix.lower() in VIDEO_EXT]
            if args.max_videos_per_class > 0:
                videos = videos[: args.max_videos_per_class]

            saved = 0
            for vid in videos:
                arr = extract_video(extractor, vid, args.min_frames)
                if arr is None:
                    continue
                out_path = out_dir / f"{vid.stem}.npz"
                np.savez_compressed(out_path, landmarks=arr)
                saved += 1
                total += 1
            print(f"  {slug}: {saved}/{len(videos)} videos")
    finally:
        extractor.close()

    save_vocab(
        vocab_path,
        all_slugs,
        labels_vi,
        meta={"source": "kaggle_videos", "raw_root": str(raw_root), "total_clips": total},
    )
    print(f"\nDone. {total} clips → {out_root}\nVocab → {vocab_path}")

if __name__ == "__main__":
    main()

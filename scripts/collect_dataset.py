
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cv.hands_detector import HandLandmarkExtractor
from training.vsl.vocab import (
    load_vocab,
    rebuild_vocab_from_dataset,
    slugify_vi,
)
from utils.camera_capture import open_first_working
from utils.config_loader import load_config

def next_seq_path(class_dir: Path) -> Path:
    class_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(class_dir.glob("seq_*.npz"))
    n = 1
    if existing:
        try:
            n = int(existing[-1].stem.split("_")[-1]) + 1
        except ValueError:
            n = len(existing) + 1
    return class_dir / f"seq_{n:04d}.npz"

def count_samples(class_dir: Path) -> int:
    return len(list(class_dir.glob("*.npz"))) if class_dir.is_dir() else 0

def load_class_list(cfg: dict) -> Tuple[List[str], dict]:
    vsl = cfg.get("vsl", {})
    vocab_path = Path(vsl.get("vocab_file", "configs/vocab.json"))
    if vocab_path.is_file():
        v = load_vocab(vocab_path)
        cl = list(v.get("classes") or [])
        if cl:
            return cl, v.get("labels_vi", {})
    cl = list(cfg.get("gesture", {}).get("classes") or [])
    return cl, {c: c.replace("_", " ") for c in cl}

def draw_hud(
    frame: np.ndarray,
    slug: str,
    label_vi: str,
    class_idx: int,
    n_classes: int,
    recording: bool,
    buf_len: int,
    saved_n: int,
) -> None:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 210), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    st = "DANG GHI" if recording else "San sang"
    col = (0, 0, 255) if recording else (0, 255, 0)
    cv2.putText(
        frame,
        f"[{class_idx + 1}/{n_classes}] {label_vi} ({slug}) | {st}",
        (12, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        col,
        2,
    )
    cv2.putText(
        frame,
        f"Frames: {buf_len} | Da luu lop nay: {saved_n}",
        (12, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        "[ ] lop | 0-9 | SPACE ghi | N them lop | R huy | Q thoat",
        (12, 96),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
    )

def main() -> None:
    p = argparse.ArgumentParser(description="Thu dataset VSL tu webcam")
    p.add_argument("--config", default=None)
    p.add_argument("--min-frames", type=int, default=None)
    p.add_argument("--max-frames", type=int, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    coll = cfg.get("collection", {})
    min_f = args.min_frames or int(coll.get("min_frames", 12))
    max_f = args.max_frames or int(coll.get("max_frames", 120))

    dataset_root = Path(cfg["paths"]["dataset_root"])
    dataset_root.mkdir(parents=True, exist_ok=True)
    vocab_path = Path(cfg.get("vsl", {}).get("vocab_file", "configs/vocab.json"))

    classes, labels_vi = load_class_list(cfg)
    if not classes and dataset_root.is_dir() and any(dataset_root.iterdir()):
        rebuild_vocab_from_dataset(dataset_root, vocab_path, source="camera")
        classes, labels_vi = load_class_list(cfg)
    if not classes:
        print("Chua co lop — nhan N de them lop, hoac import VSL/Kaggle truoc.")

    mp_cfg = cfg["mediapipe"]
    extractor = HandLandmarkExtractor(
        max_num_hands=int(mp_cfg["max_num_hands"]),
        model_complexity=int(mp_cfg["model_complexity"]),
        min_detection_confidence=float(mp_cfg["min_detection_confidence"]),
        min_tracking_confidence=float(mp_cfg["min_tracking_confidence"]),
    )

    cam = cfg["camera"]
    cap, _ = open_first_working(
        int(cam["device_index"]),
        str(cam.get("backend", "auto")),
        list(range(int(cam.get("try_indices_max", 4)))),
    )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cam["width"]))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cam["height"]))
    if not cap.isOpened():
        raise SystemExit("Khong mo duoc camera.")

    for _ in range(10):
        cap.read()

    class_idx = 0
    recording = False
    buffer: List[np.ndarray] = []
    saved_msg, saved_ttl = "", 0
    win = "Thu du lieu VSL — SPACE de ghi"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def save_clip() -> bool:
        nonlocal buffer, saved_msg, saved_ttl
        if len(buffer) < min_f:
            print(f"Qua it frame ({len(buffer)} < {min_f})")
            return False
        slug = classes[class_idx]
        out = next_seq_path(dataset_root / slug)
        arr = np.stack(buffer, axis=0).astype(np.float32)
        np.savez_compressed(out, landmarks=arr)
        saved_msg = f"Luu {out.name} ({arr.shape[0]} f)"
        saved_ttl = 50
        print(f"  -> {out}")
        buffer.clear()
        return True

    print(f"Luu tai: {dataset_root}\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            lm = extractor.process_frame(frame)
            display = extractor.annotate_last(frame)

            if recording and lm.confidence_ok:
                buffer.append(lm.vector.copy())
                if len(buffer) >= max_f:
                    recording = False
                    save_clip()

            if not classes:
                cv2.putText(
                    display,
                    "Nhan N de them lop moi",
                    (12, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
            else:
                slug = classes[class_idx]
                draw_hud(
                    display,
                    slug,
                    labels_vi.get(slug, slug),
                    class_idx,
                    len(classes),
                    recording,
                    len(buffer),
                    count_samples(dataset_root / slug),
                )

            if saved_ttl > 0:
                cv2.putText(
                    display,
                    saved_msg,
                    (12, display.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 128),
                    2,
                )
                saved_ttl -= 1

            cv2.imshow(win, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

            if key == ord("n"):
                cv2.destroyWindow(win)
                name = input("Ten ky hieu (tieng Viet): ").strip()
                cv2.namedWindow(win, cv2.WINDOW_NORMAL)
                if name:
                    slug = slugify_vi(name)
                    if slug not in classes:
                        classes.append(slug)
                        labels_vi[slug] = name
                    class_idx = classes.index(slug)
                    print(f"Them/chon lop: {name} -> {slug}")

            elif classes and key == ord("["):
                if not recording:
                    class_idx = (class_idx - 1) % len(classes)
            elif classes and key == ord("]"):
                if not recording:
                    class_idx = (class_idx + 1) % len(classes)
            elif classes and ord("0") <= key <= ord("9"):
                d = key - ord("0")
                if d < len(classes) and not recording:
                    class_idx = d
            elif key == ord(" "):
                if not classes:
                    print("Them lop bang phim N truoc.")
                    continue
                if not recording:
                    recording = True
                    buffer.clear()
                else:
                    recording = False
                    save_clip()
            elif key == ord("r"):
                buffer.clear()
                recording = False
    finally:
        cap.release()
        extractor.close()
        cv2.destroyAllWindows()

    if classes or any(dataset_root.iterdir()):
        rebuild_vocab_from_dataset(dataset_root, vocab_path, source="camera")
        print(f"\nCap nhat vocab → {vocab_path}")

    print("\nTong ket:")
    for c in classes:
        print(f"  {c}: {count_samples(dataset_root / c)} file")
    print("\nTrain: python training/train.py")

if __name__ == "__main__":
    main()


from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.vsl.data_check import count_processed_clips
from utils.config_loader import load_config

def run(cmd: list[str]) -> None:
    print("\n>>>", " ".join(cmd), "\n")
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        raise SystemExit(
            f"\nLệnh thất bại (mã {result.returncode}):\n  {' '.join(cmd)}\n"
            "Xem thông báo lỗi phía trên."
        )

def ensure_dataset_ready(cfg_path: str) -> Path:
    cfg = load_config(cfg_path)
    dataset_root = Path(cfg["paths"]["dataset_root"])
    n = count_processed_clips(dataset_root)
    if n == 0:
        raise SystemExit(
            f"\nKhông có file .npz trong: {dataset_root}\n\n"
            "Cần làm một trong các bước sau TRƯỚC khi train:\n"
            "  1) VOYA — tải dữ liệu:\n"
            "     python training/vsl/download_voya.py\n"
            "     (hoặc: hf download Kateht/VOYA_VSL --repo-type dataset "
            "--local-dir datasets/raw/voya)\n"
            "     python -m training.vsl.pipeline prepare --source voya\n"
            "  2) Hoặc thu từ webcam:\n"
            "     python scripts/collect_dataset.py\n"
        )
    print(f"Dataset OK: {n} clip trong {dataset_root}")
    return dataset_root

def main() -> None:
    p = argparse.ArgumentParser(description="VSL Kaggle pipeline")
    p.add_argument("stage", choices=["prepare", "train", "all", "download"])
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    p.add_argument(
        "--source",
        choices=["voya", "kaggle_videos", "kaggle_download", "download_voya"],
        default="voya",
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    args, extra = p.parse_known_args()

    py = sys.executable
    cfg = ["--config", args.config]

    if args.stage in ("download", "all"):
        if args.source == "kaggle_download":
            run([py, str(ROOT / "training" / "vsl" / "download_kaggle.py"), *cfg])
        elif args.source in ("voya", "all", "download_voya"):
            run([py, str(ROOT / "training" / "vsl" / "download_voya.py"), *cfg])

    if args.stage in ("prepare", "all"):
        if args.source in ("voya", "all", "download_voya"):
            run([py, str(ROOT / "training" / "vsl" / "prepare_voya.py"), *cfg, *extra])
        elif args.source in ("kaggle_videos", "kaggle_download"):
            run([py, str(ROOT / "training" / "vsl" / "prepare_videos.py"), *cfg, *extra])

    if args.stage in ("train", "all"):
        ensure_dataset_ready(args.config)
        run(
            [
                py,
                str(ROOT / "training" / "train.py"),
                *cfg,
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
            ]
        )

if __name__ == "__main__":
    main()

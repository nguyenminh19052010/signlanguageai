
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.config_loader import load_config

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic (random) dataset for pipeline testing ONLY."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "default.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Defaults to datasets/synthetic_test (NOT the real dataset dir).",
    )
    parser.add_argument("--samples-per-class", type=int, default=80)
    parser.add_argument("--time-steps", type=int, default=32)
    parser.add_argument("--feature-dim", type=int, default=126)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    cfg = load_config(args.config)

    classes = list(cfg["gesture"]["classes"])

    if args.output_dir:
        root = Path(args.output_dir)
    else:
        root = ROOT / "datasets" / "synthetic_test"

    root.mkdir(parents=True, exist_ok=True)

    print("⚠️  CẢNH BÁO: Đây là dữ liệu giả (synthetic) CHỈ để test pipeline!")
    print("⚠️  KHÔNG dùng để train model production.")
    print(f"   Output: {root}\n")

    rng = np.random.default_rng(args.seed)

    for ci, cls in enumerate(classes):
        out_dir = root / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        center = rng.standard_normal(args.feature_dim).astype(np.float32)
        for si in range(args.samples_per_class):
            noise = rng.normal(
                0, 0.25, size=(args.time_steps, args.feature_dim)
            ).astype(np.float32)
            drift = np.linspace(0, 1, args.time_steps, dtype=np.float32)[:, None]
            seq = center + noise + (ci * 0.05 + drift) * 0.1
            fp = out_dir / f"seq_{si:04d}.npz"
            np.savez_compressed(fp, landmarks=seq)

    print(f"Wrote synthetic dataset for {len(classes)} classes → {root}")
    print("⚠️  Nhắc lại: Chỉ dùng để test pipeline, KHÔNG dùng để train production!")

if __name__ == "__main__":
    main()


from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.onnx_export import export_onnx
from utils.config_loader import load_config

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    ckpt = Path(cfg["paths"]["gesture_checkpoint"])
    out = Path(cfg["paths"]["onnx_model"])
    export_onnx(ckpt, out, cfg)
    print(f"Exported ONNX → {out}")

if __name__ == "__main__":
    main()

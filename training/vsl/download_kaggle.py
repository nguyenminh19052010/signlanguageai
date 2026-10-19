
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.config_loader import load_config

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--dataset", type=str, default=None, help="slug Kaggle: owner/dataset-name")
    p.add_argument("--unzip", action="store_true", default=True)
    args = p.parse_args()

    cfg = load_config(args.config)
    vsl = cfg.get("vsl", {})
    slug = args.dataset or vsl.get("kaggle_dataset")
    if not slug:
        raise SystemExit(
            "Chưa cấu hình kaggle_dataset. "
            "Sửa configs/vsl_kaggle.yaml hoặc --dataset owner/name"
        )

    out_dir = Path(vsl.get("kaggle_raw_dir", "datasets/vsl_raw/kaggle"))
    out_dir.mkdir(parents=True, exist_ok=True)

    if shutil.which("kaggle") is None:
        raise SystemExit(
            "Chưa có lệnh 'kaggle'. Cài: pip install kaggle\n"
            "Đặt token: %USERPROFILE%\\.kaggle\\kaggle.json"
        )

    cmd = ["kaggle", "datasets", "download", "-d", slug, "-p", str(out_dir)]
    if args.unzip:
        cmd.append("--unzip")
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)
    print(f"Downloaded → {out_dir}")

if __name__ == "__main__":
    main()

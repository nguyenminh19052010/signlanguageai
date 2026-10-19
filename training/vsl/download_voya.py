
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.config_loader import load_config

REPO_ID = "Kateht/VOYA_VSL"
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

def free_gb(path: Path) -> float:
    try:
        usage = shutil.disk_usage(path.anchor if path.anchor else path)
        return usage.free / (1024**3)
    except Exception:
        return 0.0

def list_merged_npz(repo_id: str) -> list[str]:
    from huggingface_hub import list_repo_files

    files = list_repo_files(repo_id, repo_type="dataset")
    return sorted(f for f in files if f.startswith("Merged/") and f.endswith(".npz"))

def download_subset(
    out: Path,
    max_class_files: int,
    extra_files: list[str],
) -> None:
    from huggingface_hub import hf_hub_download

    out.mkdir(parents=True, exist_ok=True)
    merged = out / "Merged"
    merged.mkdir(parents=True, exist_ok=True)

    npz_remote = list_merged_npz(REPO_ID)
    if not npz_remote:
        raise SystemExit("Không liệt kê được file Merged/*.npz trên Hub.")

    pick = npz_remote if max_class_files <= 0 else npz_remote[:max_class_files]
    print(f"Tải {len(pick)}/{len(npz_remote)} file lớp (+ labels.json)...")

    for name in extra_files:
        print(f"  {name}")
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=name,
            local_dir=str(out),
        )

    for i, remote in enumerate(pick, 1):
        print(f"  [{i}/{len(pick)}] {remote}")
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=remote,
            local_dir=str(out),
        )

def download_full(out: Path) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            local_dir=str(out),
            allow_patterns=["Merged/*", "labels.json"],
            local_dir_use_symlinks=False,
        )
    )

def main() -> None:
    p = argparse.ArgumentParser(description="Download VOYA_VSL (partial or full)")
    p.add_argument("--config", default=None)
    p.add_argument(
        "--local-dir",
        default=None,
        help="Thư mục đích (vd. D:/data/voya nếu ổ C đầy)",
    )
    p.add_argument(
        "--max-class-files",
        type=int,
        default=20,
        help="Số file class_*.npz tải về (mặc định 20). 0 = tất cả (cần rất nhiều dung lượng)",
    )
    p.add_argument(
        "--full",
        action="store_true",
        help="Tải toàn bộ Merged/ (~50GB+). Chỉ dùng khi ổ đĩa còn đủ trống.",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    out = Path(
        args.local_dir or cfg.get("vsl", {}).get("voya_root", "datasets/raw/voya")
    )
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    free = free_gb(out)
    print(f"Ổ đĩa trống (ước lượng): {free:.1f} GB tại {out.drive or out}")

    if args.full:
        if free < 55 and free > 0:
            raise SystemExit(
                f"Không đủ dung lượng (~{free:.1f} GB trống, cần ~50GB+).\n"
                "Dùng tải một phần: python training/vsl/download_voya.py --max-class-files 20\n"
                "Hoặc: --local-dir D:\\duong_dan_ổ_khac"
            )
        print(f"FULL download {REPO_ID} → {out}\n")
        download_full(out)
    else:
        need_gb = max(2.0, args.max_class_files * 0.35)
        if free > 0 and free < need_gb:
            raise SystemExit(
                f"Không đủ dung lượng (~{free:.1f} GB trống, cần ~{need_gb:.0f} GB).\n"
                "Giảm --max-class-files, dọn ổ đĩa, hoặc --local-dir ổ khác.\n"
                "Hoặc bỏ VOYA: python scripts/collect_dataset.py"
            )
        max_f = args.max_class_files
        print(
            f"Partial download {REPO_ID} → {out}\n"
            f"(max {max_f} class files — đủ để train thử; không phải full 161 lớp)\n"
        )
        download_subset(out, max_f, extra_files=["labels.json"])

    merged = out / "Merged"
    n = len(list(merged.glob("*.npz"))) if merged.is_dir() else 0
    print(f"\nDone. Merged/*.npz: {n} files tại {out}")
    if n == 0:
        raise SystemExit("Không có file .npz — kiểm tra mạng / hf auth login")

    print("\nTiếp theo:")
    print("  python -m training.vsl.pipeline prepare --source voya")
    print("  python training/train.py")

if __name__ == "__main__":
    main()

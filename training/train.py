
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.gesture_net import GestureCNNLSTM, landmark_dim
from training.dataset import LandmarkSequenceDataset
from training.vsl.vocab import load_vocab, rebuild_vocab_from_dataset
from utils.config_loader import load_config

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _project_root(cfg: dict) -> Path:
    if cfg.get("_project_root"):
        return Path(cfg["_project_root"])
    return Path(cfg["paths"]["dataset_root"]).parent.parent

def resolve_classes_and_vocab(cfg: dict) -> Tuple[List[str], Dict]:
    vsl = cfg.get("vsl", {})
    root = _project_root(cfg)
    vocab_path = Path(vsl.get("vocab_file", "configs/vocab.json"))
    if not vocab_path.is_absolute():
        vocab_path = (root / vocab_path).resolve()
    dataset_root = Path(cfg["paths"]["dataset_root"])

    if vocab_path.is_file():
        vocab = load_vocab(vocab_path)
        classes = list(vocab.get("classes") or [])
        if classes:
            return classes, vocab

    if dataset_root.is_dir():
        vocab = rebuild_vocab_from_dataset(dataset_root, vocab_path, source="auto_scan")
        return list(vocab["classes"]), vocab

    classes = list(cfg.get("gesture", {}).get("classes") or [])
    if not classes:
        raise SystemExit(
            "Không có lớp để train. Chạy:\n"
            "  python scripts/collect_dataset.py   (thu camera)\n"
            "  hoặc python -m training.vsl.pipeline prepare --source voya"
        )
    return classes, {"classes": classes, "labels_vi": {c: c for c in classes}}

def main() -> None:
    parser = argparse.ArgumentParser(description="Train gesture model")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--subset-classes", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(42)

    classes, vocab = resolve_classes_and_vocab(cfg)
    if args.subset_classes > 0:
        classes = classes[: args.subset_classes]
        vocab = {
            "classes": classes,
            "labels_vi": {c: vocab.get("labels_vi", {}).get(c, c) for c in classes},
        }

    seq_len = int(cfg["gesture"]["sequence_length"])
    dataset_root = Path(cfg["paths"]["dataset_root"])

    ds = LandmarkSequenceDataset(
        dataset_root, class_names=classes, seq_length=seq_len, augment=True
    )
    if len(ds) == 0:
        raise SystemExit(
            f"Không có mẫu .npz trong {dataset_root} cho {len(classes)} lớp trong vocab.\n"
            "Kiểm tra thư mục con có file .npz, hoặc chạy lại prepare / collect_dataset."
        )

    if len(ds) < 4:
        print(
            f"Cảnh báo: chỉ có {len(ds)} mẫu — nên thu thêm (collect_dataset) hoặc prepare VOYA đầy đủ."
        )

    print(f"Train: {len(classes)} lớp, {len(ds)} mẫu")

    n_val = max(1, min(len(ds) - 1, int(len(ds) * args.val_ratio)))
    n_train = len(ds) - n_val
    if n_train < 1:
        n_val = 1
        n_train = len(ds) - 1
    train_ds, val_ds = random_split(ds, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GestureCNNLSTM(num_classes=len(classes), input_dim=landmark_dim()).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss()

    ckpt_path = Path(cfg["paths"]["gesture_checkpoint"])
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda" if device.type == "cuda" else "cpu",
                enabled=device.type == "cuda",
            ):
                loss = crit(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += float(loss.item())
        sched.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(dim=-1)
                correct += int((pred == y).sum().item())
                total += y.numel()
        acc = correct / max(1, total)
        print(
            f"Epoch {epoch:03d} | loss={total_loss / max(1, len(train_loader)):.4f} "
            f"| val_acc={acc:.4f}"
        )
        if acc >= best_acc:
            best_acc = acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": classes,
                    "labels_vi": vocab.get("labels_vi", {}),
                    "seq_len": seq_len,
                    "epoch": epoch,
                    "val_acc": acc,
                    "train_source": "camera",
                },
                ckpt_path,
            )
            print(f"  saved → {ckpt_path}")

    print(f"Done. Best val_acc={best_acc:.4f}")

if __name__ == "__main__":
    main()


from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

IS_KAGGLE = Path("/kaggle").exists()
WORK = Path("/kaggle/working") if IS_KAGGLE else Path("kaggle_output")
INPUT = Path("/kaggle/input") if IS_KAGGLE else Path(".")

RAW_VOYA = WORK / "raw_voya"
PROCESSED = WORK / "processed"
CKPT_OUT = WORK / "gesture.pt"
VOCAB_OUT = WORK / "vocab.json"

SEQ_LEN = 32
LANDMARK_DIM = 126
REPO_ID = "Kateht/VOYA_VSL"
_DEFAULT_MAX = "6" if IS_KAGGLE else "25"
MAX_CLASS_FILES = int(os.environ.get("MAX_VOYA_CLASSES", _DEFAULT_MAX))
MIN_FREE_GB = float(os.environ.get("MIN_FREE_GB", "1.2"))

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

class GestureCNNLSTM(nn.Module):
    def __init__(self, num_classes: int, input_dim: int = LANDMARK_DIM) -> None:
        super().__init__()
        c0, c1, c2 = 128, 256, 256
        self.in_proj = nn.Sequential(
            nn.Linear(input_dim, c0),
            nn.LayerNorm(c0),
            nn.GELU(),
            nn.Dropout(0.25),
        )
        self.conv = nn.Sequential(
            nn.Conv1d(c0, c1, 3, padding=1, bias=False),
            nn.BatchNorm1d(c1),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Conv1d(c1, c2, 3, padding=1, bias=False),
            nn.BatchNorm1d(c2),
            nn.GELU(),
            nn.Dropout(0.25),
        )
        self.lstm = nn.LSTM(c2, 256, 2, batch_first=True, bidirectional=True, dropout=0.25)
        self.head = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x).transpose(1, 2)
        h = self.conv(h).transpose(1, 2)
        out, _ = self.lstm(h)
        pooled, _ = out.max(dim=1)
        return self.head(pooled)

class LandmarkSequenceDataset(Dataset):
    def __init__(self, root: Path, classes: List[str], augment: bool = False) -> None:
        self.root = Path(root)
        self.classes = classes
        self.augment = augment
        self.samples: List[Tuple[Path, int]] = []
        for ci, cls in enumerate(classes):
            d = self.root / cls
            if d.is_dir():
                for fp in sorted(d.glob("*.npz")):
                    self.samples.append((fp, ci))

    def __len__(self) -> int:
        return len(self.samples)

    def _resample(self, seq: np.ndarray) -> np.ndarray:
        t, f = seq.shape
        if t == SEQ_LEN:
            return seq.astype(np.float32)
        x_old = np.linspace(0, 1, t, dtype=np.float32)
        x_new = np.linspace(0, 1, SEQ_LEN, dtype=np.float32)
        out = np.empty((SEQ_LEN, f), np.float32)
        for i in range(f):
            out[:, i] = np.interp(x_new, x_old, seq[:, i])
        return out

    def __getitem__(self, idx: int):
        fp, label = self.samples[idx]
        seq = np.load(fp)["landmarks"].astype(np.float32)
        if self.augment:
            seq = seq + np.random.normal(0, 0.01, seq.shape).astype(np.float32)
            seq *= float(np.random.uniform(0.92, 1.08))
        x = torch.from_numpy(self._resample(seq))
        return x, torch.tensor(label, dtype=torch.long)

def slugify(name: str) -> str:
    s = unicodedata.normalize("NFKD", name.strip().lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s or "class"

def frame_to_126(frame: np.ndarray) -> np.ndarray:
    v = np.asarray(frame, dtype=np.float32).reshape(-1)
    if v.shape[0] == LANDMARK_DIM:
        return v
    if v.shape[0] >= LANDMARK_DIM:
        return v[-LANDMARK_DIM:]
    raise ValueError(f"Bad dim {v.shape[0]}")

def sequence_to_126(seq: np.ndarray) -> np.ndarray:
    return np.stack([frame_to_126(seq[t]) for t in range(seq.shape[0])], axis=0)

def check_hf_reachable() -> bool:

    import socket

    try:
        socket.getaddrinfo("huggingface.co", 443)
        return True
    except socket.gaierror:
        return False

def voya_file_list_fallback(max_files: int) -> List[str]:

    n = min(max_files, 161)
    return [f"Merged/class_{i:04d}.npz" for i in range(1, n + 1)]

def disk_free_gb(path: Path) -> float:
    import shutil

    try:
        return shutil.disk_usage(path).free / (1024**3)
    except Exception:
        return 0.0

def class_id_from_merged_file(cf: Path) -> int | None:
    m = re.match(r"class_(\d+)", cf.stem, re.IGNORECASE)
    return int(m.group(1)) if m else None

def ingest_merged_npz_file(
    cf: Path,
    labels: dict,
    out_root: Path,
    classes: List[str],
    labels_vi: Dict[str, str],
) -> int:

    total = 0
    file_cid = class_id_from_merged_file(cf)
    data = np.load(cf)
    sequences = data["sequences"]
    sample_labels = data.get("labels", np.zeros(len(sequences)))
    for seq, lab in zip(sequences, sample_labels):
        cid = file_cid if file_cid is not None else int(lab)
        vi = label_name(labels, cid)
        slug = slugify(vi)
        if slug not in classes:
            classes.append(slug)
        labels_vi[slug] = vi
        try:
            arr = sequence_to_126(seq)
        except ValueError:
            continue
        d = out_root / slug
        d.mkdir(parents=True, exist_ok=True)
        n = len(list(d.glob("*.npz"))) + 1
        np.savez_compressed(d / f"seq_{n:05d}.npz", landmarks=arr)
        total += 1
    return total

def stream_hf_to_processed(max_files: int) -> Tuple[List[str], Dict[str, str]]:

    import gc
    import shutil

    from huggingface_hub import hf_hub_download

    if not check_hf_reachable():
        raise RuntimeError("Không kết nối huggingface.co — bật Internet + Restart session.")

    free0 = disk_free_gb(WORK)
    print(f"Ổ trống /kaggle/working: {free0:.1f} GB")
    if IS_KAGGLE and free0 < 3:
        raise RuntimeError(
            f"Ổ working chỉ còn {free0:.1f} GB. Xóa file cũ trong Output hoặc Restart session.\n"
            "Hoặc: MAX_VOYA_CLASSES=3"
        )

    os.environ["HF_HOME"] = str(WORK / "hf_hub_cache")
    PROCESSED.mkdir(parents=True, exist_ok=True)
    merged_dir = WORK / "Merged"
    merged_dir.mkdir(exist_ok=True)

    pick = voya_file_list_fallback(max_files)
    print(f"Stream {len(pick)} file VOYA (xóa sau mỗi file, tiết kiệm dung lượng)")

    hf_hub_download(
        REPO_ID, repo_type="dataset", filename="labels.json", local_dir=str(WORK)
    )
    labels = load_labels(WORK)

    classes: List[str] = []
    labels_vi: Dict[str, str] = {}

    for i, remote in enumerate(pick, 1):
        free = disk_free_gb(WORK)
        if free < MIN_FREE_GB:
            print(f"Dừng sớm: ổ còn {free:.1f} GB (< {MIN_FREE_GB} GB)")
            break
        print(f"  [{i}/{len(pick)}] {remote} (free {free:.1f} GB)")
        try:
            hf_hub_download(
                REPO_ID,
                repo_type="dataset",
                filename=remote,
                local_dir=str(WORK),
            )
        except OSError as e:
            if e.errno == 28:
                raise RuntimeError(
                    f"Hết ổ đĩa khi tải {remote}. "
                    f"Giảm: os.environ['MAX_VOYA_CLASSES']='3' và Restart session."
                ) from e
            raise

        cf = WORK / remote
        added = ingest_merged_npz_file(cf, labels, PROCESSED, classes, labels_vi)
        print(f"      → {added} clips, xóa file thô")
        cf.unlink(missing_ok=True)
        gc.collect()

    shutil.rmtree(merged_dir, ignore_errors=True)
    shutil.rmtree(WORK / "hf_hub_cache", ignore_errors=True)
    print(f"Done stream. classes={len(classes)}, free={disk_free_gb(WORK):.1f} GB")
    return classes, labels_vi

def download_voya_partial(out: Path, max_files: int) -> None:
    from huggingface_hub import hf_hub_download, list_repo_files

    if not check_hf_reachable():
        raise RuntimeError(
            "Không kết nối được huggingface.co (DNS / mạng).\n\n"
            "Trên Kaggle Notebook:\n"
            "  1) Settings → Internet → BẬT (On), Save, Restart session\n"
            "  2) Chạy lại cell\n\n"
            "Nếu vẫn lỗi: Add Dataset VOYA vào /kaggle/input/ (xem docs/KAGGLE_CLOUD.md)\n"
            "  hoặc SKIP_HF_DOWNLOAD=1 nếu đã có thư mục Merged/ trong input."
        )

    out.mkdir(parents=True, exist_ok=True)
    (out / "Merged").mkdir(exist_ok=True)

    files: List[str] = []
    try:
        files = sorted(
            f
            for f in list_repo_files(REPO_ID, repo_type="dataset")
            if f.startswith("Merged/") and f.endswith(".npz")
        )
    except Exception as exc:
        print(f"list_repo_files failed ({exc}) — dùng danh sách file cố định.")
        files = voya_file_list_fallback(161)

    pick = files[:max_files] if max_files > 0 else files
    print(f"Download {len(pick)} class files + labels.json")

    def _dl(name: str, retries: int = 3) -> None:
        import time

        last: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                hf_hub_download(
                    REPO_ID,
                    repo_type="dataset",
                    filename=name,
                    local_dir=str(out),
                )
                return
            except Exception as e:
                last = e
                print(f"    retry {attempt}/{retries}: {e}")
                time.sleep(2 * attempt)
        raise RuntimeError(f"Failed to download {name}") from last

    _dl("labels.json")
    for i, remote in enumerate(pick, 1):
        print(f"  [{i}/{len(pick)}] {remote}")
        _dl(remote)

def load_labels(voya_root: Path) -> dict:
    p = voya_root / "labels.json"
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def label_name(labels: dict, cid: int) -> str:
    slug = f"class_{cid:04d}"
    if slug in labels:
        return str(labels[slug])
    k = str(cid)
    if k in labels:
        return str(labels[k])
    if isinstance(labels, list) and cid < len(labels):
        return str(labels[cid])
    return slug

def prepare_voya(voya_root: Path, out_root: Path, max_class_npz: int = 0) -> Tuple[List[str], Dict[str, str]]:
    merged = voya_root / "Merged"
    labels = load_labels(voya_root)
    out_root.mkdir(parents=True, exist_ok=True)
    classes: List[str] = []
    labels_vi: Dict[str, str] = {}
    files = sorted(merged.glob("*.npz"))
    if max_class_npz > 0:
        files = files[:max_class_npz]
    total = 0
    for cf in files:
        data = np.load(cf)
        sequences = data["sequences"]
        sample_labels = data.get("labels", np.zeros(len(sequences)))
        for seq, lab in zip(sequences, sample_labels):
            cid = int(lab)
            vi = label_name(labels, cid)
            slug = slugify(vi)
            if slug not in classes:
                classes.append(slug)
            labels_vi[slug] = vi
            try:
                arr = sequence_to_126(seq)
            except ValueError:
                continue
            d = out_root / slug
            d.mkdir(parents=True, exist_ok=True)
            n = len(list(d.glob("*.npz"))) + 1
            np.savez_compressed(d / f"seq_{n:05d}.npz", landmarks=arr)
            total += 1
    print(f"Prepared {total} clips, {len(classes)} classes")
    return classes, labels_vi

def find_local_processed() -> Path | None:

    for base in [INPUT, WORK]:
        if not base.exists():
            continue
        for p in base.rglob("processed"):
            if p.is_dir() and any(p.glob("*/*.npz")):
                return p
        for p in base.rglob("vsl_processed"):
            if p.is_dir() and any(p.glob("*/*.npz")):
                return p
    return None

def find_voya_in_input() -> Path | None:

    if not INPUT.exists():
        return None
    for labels in INPUT.rglob("labels.json"):
        root = labels.parent
        merged = root / "Merged"
        if merged.is_dir() and any(merged.glob("*.npz")):
            print(f"Found VOYA in Kaggle input: {root}")
            return root
    for merged in INPUT.rglob("Merged"):
        if merged.is_dir() and any(merged.glob("*.npz")):
            root = merged.parent
            print(f"Found VOYA Merged in: {root}")
            return root
    return None

def train_loop(processed: Path, classes: List[str], labels_vi: Dict[str, str], epochs: int = 40) -> None:
    ds = LandmarkSequenceDataset(processed, classes, augment=True)
    if len(ds) < 4:
        raise RuntimeError(f"Too few samples: {len(ds)}")
    n_val = max(1, min(len(ds) - 1, int(len(ds) * 0.15)))
    tr, va = random_split(ds, [len(ds) - n_val, n_val])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, samples: {len(ds)}, classes: {len(classes)}")
    model = GestureCNNLSTM(len(classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    best = 0.0
    for ep in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        for x, y in DataLoader(tr, batch_size=64, shuffle=True):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                loss = crit(model(x), y)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item())
        model.eval()
        ok = tot = 0
        with torch.no_grad():
            for x, y in DataLoader(va, batch_size=64):
                x, y = x.to(device), y.to(device)
                ok += (model(x).argmax(1) == y).sum().item()
                tot += y.numel()
        acc = ok / max(1, tot)
        print(f"Epoch {ep:03d} loss={loss_sum / max(1, len(tr)):.4f} val_acc={acc:.4f}")
        if acc >= best:
            best = acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": classes,
                    "labels_vi": labels_vi,
                    "seq_len": SEQ_LEN,
                    "val_acc": acc,
                    "epoch": ep,
                    "train_source": "voya",
                },
                CKPT_OUT,
            )
            print(f"  saved {CKPT_OUT}")
    with VOCAB_OUT.open("w", encoding="utf-8") as f:
        json.dump({"classes": classes, "labels_vi": labels_vi}, f, ensure_ascii=False, indent=2)
    print(f"Saved {VOCAB_OUT}")

def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    epochs = int(os.environ.get("EPOCHS", "40"))
    max_dl = int(os.environ.get("MAX_VOYA_CLASSES", str(MAX_CLASS_FILES)))
    if IS_KAGGLE:
        max_dl = min(max_dl, 12)
    skip_hf = os.environ.get("SKIP_HF_DOWNLOAD", "").lower() in ("1", "true", "yes")

    print("Kaggle:", IS_KAGGLE, "| disk free:", f"{disk_free_gb(WORK):.1f} GB")
    print("Internet:", "OK" if check_hf_reachable() else "FAILED (Settings → Internet ON)")
    print("MAX_VOYA_CLASSES:", max_dl)

    prebuilt = find_local_processed()
    if prebuilt is not None:
        print(f"Using existing processed data: {prebuilt}")
        classes = sorted(d.name for d in prebuilt.iterdir() if d.is_dir() and any(d.glob("*.npz")))
        labels_vi = {c: c.replace("_", " ") for c in classes}
        train_loop(prebuilt, classes, labels_vi, epochs=epochs)
        return

    voya_src = find_voya_in_input()
    if voya_src is not None:
        RAW_VOYA_USE = voya_src
    else:
        RAW_VOYA_USE = RAW_VOYA

    n_local = len(list((RAW_VOYA_USE / "Merged").glob("*.npz"))) if (RAW_VOYA_USE / "Merged").is_dir() else 0
    if n_local < 3 and not skip_hf:
        print("Tải VOYA (stream — tiết kiệm ổ)...")
        try:
            classes, labels_vi = stream_hf_to_processed(max_dl)
        except Exception as exc:
            raise SystemExit(
                f"Tải / xử lý VOYA thất bại: {exc}\n\n"
                "Thử:\n"
                "  os.environ['MAX_VOYA_CLASSES'] = '3'\n"
                "  Restart session (xóa file cũ trong /kaggle/working)\n"
                "  Hoặc Add Dataset VOYA → /kaggle/input + SKIP_HF_DOWNLOAD=1"
            ) from exc
    elif n_local < 3:
        raise SystemExit(
            "SKIP_HF_DOWNLOAD=1 nhưng không có Merged/*.npz trong /kaggle/input."
        )
    else:
        classes, labels_vi = prepare_voya(RAW_VOYA_USE, PROCESSED)
    if not classes:
        raise RuntimeError("No classes prepared")
    train_loop(PROCESSED, classes, labels_vi, epochs=epochs)
    print("\n=== DONE ===")
    print(f"Download from Kaggle Output: {CKPT_OUT.name}, {VOCAB_OUT.name}")

if __name__ == "__main__":
    main()

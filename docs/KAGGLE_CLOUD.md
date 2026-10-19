# Train trên Kaggle Cloud & chạy `.pt` trên máy

Không cần tải **50GB VOYA** về PC — train trên GPU Kaggle, chỉ tải về file **~10–50 MB** (`gesture.pt` + `vocab.json`).

---

## Bước 1 — Tạo Notebook Kaggle

1. Vào [kaggle.com](https://www.kaggle.com) → **Code** → **New Notebook**.
2. **Settings** (bên phải):
   - **Accelerator**: GPU T4 x2 (hoặc P100)
   - **Internet**: **On** (bắt buộc nếu tải từ Hugging Face)
   - Sau khi bật Internet: **Save** → **Restart session** (quan trọng!)
   - **Persistence**: tùy chọn

---

## Bước 2 — Đưa script lên Kaggle

**Cách A (nhanh):** Upload file `kaggle/train_vsl_kaggle.py` từ project này.

Trong Notebook, cell đầu:

```python
# Upload train_vsl_kaggle.py qua nút Add data → Upload, hoặc copy-paste file vào /kaggle/working/
!pip install -q huggingface_hub
!python /kaggle/working/train_vsl_kaggle.py
```

**Cách B:** Copy toàn bộ nội dung `kaggle/train_vsl_kaggle.py` vào **một cell** Notebook rồi chạy.

**Tùy chỉnh (cell trước khi train):**

```python
import os
os.environ["MAX_VOYA_CLASSES"] = "6"    # mặc định trên Kaggle (ổ ~20GB)
os.environ["EPOCHS"] = "50"
```

**Quan trọng:** dùng bản mới `kaggle/train_vsl_kaggle.py` (có dòng `Stream … tiết kiệm ổ`). Nếu cell cũ vẫn tải 25 file rồi mới convert → dễ **hết ổ**.

---

## Bước 3 — Chờ train xong

Script sẽ tự:

1. **Stream** từng file VOYA (tải 1 file → convert → **xóa file thô**), mặc định **6 lớp**
2. Chuyển keypoint → **126-D**
3. Train **GestureCNNLSTM**
4. Lưu:
   - `/kaggle/working/gesture.pt`
   - `/kaggle/working/vocab.json`

---

## Bước 4 — Tải file về máy

1. Notebook → tab **Output** (bên phải).
2. Bấm **Download** cạnh `gesture.pt` và `vocab.json`.

Hoặc:

```python
from IPython.display import FileLink
FileLink("gesture.pt")
FileLink("vocab.json")
```

---

## Bước 5 — Đặt vào project trên PC

Copy vào đúng chỗ trong `sign_language_voice_ai`:

```
models/checkpoints/gesture.pt    ← file vừa tải
configs/vocab.json             ← file vừa tải
```

`configs/default.yaml` đã trỏ sẵn:

```yaml
paths:
  gesture_checkpoint: "models/checkpoints/gesture.pt"
vsl:
  vocab_file: "configs/vocab.json"
```

---

## Bước 6 — Chạy app trên máy

```powershell
cd sign_language_voice_ai
.\.venv\Scripts\activate
python main.py
```

- Bấm **Start camera**
- Ký hiệu các lớp đã train (subset VOYA hoặc lớp bạn đã có trong vocab)
- Đủ **5 từ** → LLM → TTS

**Lưu ý:** Model train trên landmark VOYA có thể **không khớp 100%** webcam của bạn. Để tốt hơn: thu thêm `python scripts/collect_dataset.py` rồi train lại (trên Kaggle hoặc máy).

---

## Lỗi `OSError: [Errno 28] No space left on device`

Ổ `/kaggle/working` (~20GB) đã đầy — thường do session cũ còn file tải dở hoặc tải quá nhiều `.npz` cùng lúc.

1. **Restart session** (xóa sạch working).
2. Upload lại **bản script mới** (stream).
3. Trước khi train:
   ```python
   import os
   os.environ["MAX_VOYA_CLASSES"] = "3"   # thử 3 lớp trước
   ```
4. Hoặc **không tải HF**: add Dataset có `Merged/` + `labels.json` → `SKIP_HF_DOWNLOAD=1` (xem dưới).

---

## Lỗi `ConnectError` / `Temporary failure in name resolution`

Nghĩa là Notebook **không truy cập được** `huggingface.co`.

1. **Settings → Internet → On** → **Save** → **Restart session** (không chỉ Save).
2. Chạy thử:
   ```python
   import socket
   socket.getaddrinfo("huggingface.co", 443)
   print("OK")
   ```
3. Nếu vẫn lỗi → **không dùng Hugging Face trên Kaggle**, add Dataset local:
   - Tải subset VOYA trên máy khác (hoặc `download_voya.py` với `--max-class-files 15`)
   - Nén thư mục có `Merged/` + `labels.json`
   - Kaggle → **Add data** → **Upload dataset**
   - Trong Notebook script tự tìm `/kaggle/input/.../Merged/`

Hoặc set trước khi train (đã có data trong input):

```python
import os
os.environ["SKIP_HF_DOWNLOAD"] = "1"
!python /kaggle/working/train_vsl_kaggle.py
```

---

## (Tuỳ chọn) Dataset đã prepare sẵn trên Kaggle

Nếu bạn upload Kaggle Dataset dạng:

```
processed/
  lop_a/
    seq_00001.npz
  lop_b/
    ...
```

Add dataset vào Notebook → script tự tìm trong `/kaggle/input/.../processed` và **bỏ qua bước tải VOYA**.

---

## So sánh

| Cách | Dung lượng PC | GPU |
|------|----------------|-----|
| Tải full VOYA về PC | ~50GB | Máy bạn |
| **Kaggle (khuyến nghị)** | Chỉ `.pt` + `vocab.json` | Kaggle GPU |
| Thu webcam | Nhỏ nhất | Máy bạn |

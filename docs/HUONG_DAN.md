# Hướng dẫn sử dụng

## Cấu trúc thư mục (đã gọn)

```
sign_language_voice_ai/
├── configs/default.yaml      # Cấu hình duy nhất
├── configs/vocab.json        # Sinh tự động sau thu data / prepare VSL
├── datasets/
│   ├── raw/                  # VOYA / Kaggle tải về (không commit)
│   └── processed/            # .npz train + thu camera
├── scripts/
│   ├── collect_dataset.py    # Thu từ webcam ★
│   └── export_onnx.py
├── training/
│   ├── train.py              # Huấn luyện
│   ├── dataset.py
│   └── vsl/                  # Chuẩn bị Kaggle / VOYA
├── cv/ llm/ tts/ ui/ models/ inference/ utils/
└── main.py                   # App realtime
```

## Luồng A — Tự thu dữ liệu bằng camera (khuyến nghị cá nhân hóa)

```powershell
python scripts/collect_dataset.py
```

| Phím | Chức năng |
|------|-----------|
| **N** | Thêm lớp mới (gõ tên tiếng Việt) |
| **[** / **]** | Đổi lớp |
| **0–9** | Chọn nhanh (10 lớp đầu) |
| **SPACE** | Bắt đầu / dừng ghi cử chỉ |
| **R** | Hủy clip |
| **Q** | Thoát + cập nhật `configs/vocab.json` |

Mỗi clip → `datasets/processed/<slug>/seq_XXXX.npz`.

```powershell
python training/train.py --epochs 50
python main.py
```

## Luồng B — Dataset VSL Kaggle / VOYA

### VOYA (keypoint có sẵn)

```powershell
pip install huggingface_hub

# VOYA đầy đủ ~50GB+ — KHÔNG tải full nếu ổ C đầy.
# Mặc định chỉ tải 20 lớp (~ vài GB):
python training/vsl/download_voya.py

# Hoặc chỉ định số lớp / ổ đĩa khác:
python training/vsl/download_voya.py --max-class-files 15 --local-dir D:\data\voya

python -m training.vsl.pipeline prepare --source voya
```

**Ổ đĩa đầy (`not enough space on the disk`):** dùng `--max-class-files 10`, `--local-dir` sang ổ D:, hoặc **bỏ tải VOYA** và dùng `python scripts/collect_dataset.py`.

### Kaggle video

1. Sửa `vsl.kaggle_dataset` trong `configs/default.yaml`
2. `python training/vsl/download_kaggle.py`
3. `python training/vsl/prepare_videos.py`

### Train + chạy

```powershell
python -m training.vsl.pipeline train --epochs 50
# hoặc
python training/train.py
python main.py
```

## Runtime (giữ nguyên)

Webcam → nhận diện từ → buffer **5 từ** → **LLM** (tiếng Việt) → **TTS**.

## LLM (tùy chọn)

Đặt file `.gguf` vào `models/llm/`, sửa `paths.llm_gguf` trong config.

Cài `llama-cpp-python` (wheel CUDA hoặc `--prefer-binary`). Không có GGUF → chế độ stub vẫn chạy.

## Lỗi thường gặp

| Lỗi | Xử lý |
|-----|--------|
| Camera đen | `camera.backend: dshow` hoặc `msmf` |
| mediapipe solutions | `pip install "mediapipe>=0.10.9,<0.11"` |
| Không có lớp train | Chạy `collect_dataset.py` hoặc `pipeline prepare` |

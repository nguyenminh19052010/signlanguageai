
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cv.hands_detector import HandLandmarkExtractor
from utils.config_loader import load_config

CONFIG_PATH = ROOT / "configs" / "default.yaml"

def _sync_classes_to_config(new_classes: list[str]) -> None:

    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    existing = set(raw.get("gesture", {}).get("classes", []))
    added = []
    for cls in new_classes:
        if cls not in existing:
            raw.setdefault("gesture", {}).setdefault("classes", []).append(cls)
            added.append(cls)

    if added:
        with CONFIG_PATH.open("w", encoding="utf-8") as f:
            yaml.dump(raw, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        print(f"[Config] Đã thêm classes mới vào config: {added}")

MIN_QUALITY_RATIO = 0.80

def _quality_ok(confident_flags: list[bool]) -> bool:
    if not confident_flags:
        return False
    good = sum(confident_flags)
    return (good / len(confident_flags)) >= MIN_QUALITY_RATIO

def main():
    cfg = load_config()
    seq_len = int(cfg["gesture"]["sequence_length"])
    dataset_root = Path(cfg["paths"]["dataset_root"])

    print("\n=== CHƯƠNG TRÌNH THU THẬP DỮ LIỆU CỬ CHỈ TAY ===")
    print(f"Thư mục dataset: {dataset_root}")
    print(f"Độ dài sequence: {seq_len} frames")
    print(f"Quality gate: ≥ {MIN_QUALITY_RATIO*100:.0f}% confident frames\n")

    labels_str = input(
        "Nhập các từ muốn thu thập, cách nhau bằng dấu phẩy\n"
        "(vd: xin chào, số 1, số 2): "
    ).strip()
    if not labels_str:
        print("Thoát...")
        return

    labels = [l.strip() for l in labels_str.split(",") if l.strip()]

    target_str = input(
        "Số lượng mẫu cần thu thập cho MỖI từ (khuyên dùng: 30-50): "
    ).strip()
    target_samples = int(target_str) if target_str.isdigit() else 30

    _sync_classes_to_config(labels)

    device_idx = int(cfg["camera"]["device_index"])
    cap = cv2.VideoCapture(device_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cfg["camera"]["width"]))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cfg["camera"]["height"]))

    if not cap.isOpened():
        print(f"[Camera] Không thể mở camera ở index {device_idx}. Đang tự động tìm kiếm camera khả dụng...")
        found = False
        for idx in range(5):
            if idx == device_idx:
                continue
            test_cap = cv2.VideoCapture(idx)
            if test_cap.isOpened():
                ret, _ = test_cap.read()
                if ret:
                    cap = test_cap
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cfg["camera"]["width"]))
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cfg["camera"]["height"]))
                    print(f"[Camera] Đã tự động kết nối thành công với camera index {idx}!")
                    found = True
                    break
                test_cap.release()
        
        if not found:
            print("LỖI: Không tìm thấy bất kỳ camera nào hoạt động!")
            return

    extractor = HandLandmarkExtractor(
        max_num_hands=int(cfg["mediapipe"]["max_num_hands"]),
        min_detection_confidence=float(cfg["mediapipe"]["min_detection_confidence"]),
        min_tracking_confidence=float(cfg["mediapipe"]["min_tracking_confidence"]),
    )

    current_label_idx = 0

    def setup_label(idx):
        if idx >= len(labels):
            return None, None, 0
        lbl = labels[idx]
        out_dir = dataset_root / lbl
        out_dir.mkdir(parents=True, exist_ok=True)
        existing = list(out_dir.glob("seq_*.npz"))
        return lbl, out_dir, len(existing) + 1

    label, out_dir, next_idx = setup_label(current_label_idx)
    if not label:
        return

    recording = False
    auto_mode = False
    sequence: list[np.ndarray] = []
    confidence_flags: list[bool] = []
    samples_collected = 0
    cooldown_frames = 0
    countdown_start: float | None = None
    COUNTDOWN_SECONDS = 3

    print("\n--- HƯỚNG DẪN ---")
    print("1. Đưa tay vào khung hình.")
    print("2. Nhấn 's' để BẮT ĐẦU tự động thu thập liên tục.")
    print("3. Nhấn 'r' nếu chỉ muốn thu 1 mẫu lẻ.")
    print("4. Nhấn 'n' để bỏ qua từ hiện tại và sang từ tiếp theo.")
    print("5. Nhấn 'q' để thoát hoàn toàn.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        lm_result = extractor.process_frame(frame)
        annotated = extractor.annotate_last(frame)

        if countdown_start is not None:
            elapsed = time.time() - countdown_start
            remaining = COUNTDOWN_SECONDS - elapsed
            if remaining > 0:
                text = f"Chuan bi... {int(remaining) + 1}"
                cv2.putText(
                    annotated, text, (180, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 4,
                )
                cv2.imshow("Data Collection", annotated)
                cv2.waitKey(1)
                continue
            else:
                countdown_start = None
                recording = True
                sequence = []
                confidence_flags = []

        if cooldown_frames > 0:
            cooldown_frames -= 1
            cv2.putText(
                annotated, "Dung tay lai de chuan bi...", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
            )
            cv2.putText(
                annotated, f"Tu tiep theo: '{label}'", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
            )

        elif recording:
            if lm_result.confidence_ok:
                sequence.append(lm_result.vector.copy())
                confidence_flags.append(True)
            else:
                confidence_flags.append(False)

            total_frames = len(confidence_flags)
            good_frames = len(sequence)

            quality_pct = (good_frames / max(1, total_frames)) * 100
            color = (0, 255, 0) if quality_pct >= MIN_QUALITY_RATIO * 100 else (0, 165, 255)
            cv2.putText(
                annotated,
                f"Recording: {good_frames}/{seq_len} good frames",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
            )
            cv2.putText(
                annotated,
                f"Quality: {quality_pct:.0f}% | '{label}' ({samples_collected}/{target_samples})",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
            )

            if good_frames >= seq_len:
                recording = False
                if _quality_ok(confidence_flags):
                    seq_arr = np.array(sequence[:seq_len], dtype=np.float32)
                    file_path = out_dir / f"seq_{next_idx:04d}.npz"
                    np.savez(file_path, landmarks=seq_arr)
                    print(
                        f"  ✓ Đã lưu: {file_path.name} "
                        f"(quality={sum(confidence_flags)/len(confidence_flags)*100:.0f}%, "
                        f"{samples_collected + 1}/{target_samples})"
                    )
                    next_idx += 1
                    samples_collected += 1
                else:
                    ratio = sum(confidence_flags) / len(confidence_flags) * 100
                    print(
                        f"  ✗ Bỏ mẫu kém chất lượng "
                        f"(quality={ratio:.0f}% < {MIN_QUALITY_RATIO*100:.0f}%). "
                        "Hãy giữ tay ổn định hơn!"
                    )

                sequence = []
                confidence_flags = []

                if auto_mode:
                    if samples_collected >= target_samples:
                        current_label_idx += 1
                        label_info = setup_label(current_label_idx)
                        if label_info[0] is None:
                            print("\n🎉 HOÀN THÀNH TẤT CẢ CÁC TỪ!")
                            break
                        label, out_dir, next_idx = label_info
                        samples_collected = 0
                        auto_mode = False
                        print(f"\nChuẩn bị từ tiếp theo: '{label}'")
                    else:
                        cooldown_frames = 15

            elif total_frames > seq_len * 3:
                recording = False
                print(
                    f"  ✗ Timeout — chỉ thu được {good_frames}/{seq_len} "
                    "good frames. Giữ tay trong khung hình!"
                )
                sequence = []
                confidence_flags = []
                if auto_mode:
                    cooldown_frames = 15

        else:
            cv2.putText(
                annotated,
                f"Tu: '{label}' | Da thu: {samples_collected}/{target_samples}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )
            cv2.putText(
                annotated,
                "'S': Auto Record | 'R': Single | 'N': Skip | 'Q': Quit",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
            )

        cv2.imshow("Data Collection", annotated)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("n"):
            current_label_idx += 1
            label_info = setup_label(current_label_idx)
            if label_info[0] is None:
                print("\n🎉 HOÀN THÀNH TẤT CẢ CÁC TỪ!")
                break
            label, out_dir, next_idx = label_info
            samples_collected = 0
            auto_mode = False
            recording = False
            countdown_start = None
            cooldown_frames = 0
            print(f"\nĐã chuyển sang từ: '{label}'")
        elif key == ord("r") and not recording and cooldown_frames == 0:
            print(f"  Countdown {COUNTDOWN_SECONDS}s → ghi 1 mẫu cho '{label}'...")
            auto_mode = False
            countdown_start = time.time()
        elif key == ord("s") and not recording and cooldown_frames == 0:
            print(
                f"  Countdown {COUNTDOWN_SECONDS}s → auto thu "
                f"{target_samples - samples_collected} mẫu cho '{label}'..."
            )
            auto_mode = True
            countdown_start = time.time()

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()

    print("\n=== TỔNG KẾT ===")
    for lbl in labels:
        folder = dataset_root / lbl
        count = len(list(folder.glob("seq_*.npz"))) if folder.is_dir() else 0
        print(f"  {lbl}: {count} mẫu")

if __name__ == "__main__":
    main()

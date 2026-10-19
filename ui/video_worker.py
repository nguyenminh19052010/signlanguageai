
from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cv.hands_detector import HandLandmarkExtractor
from cv.preprocessing import bilateral_denoise
from inference.gesture_engine import GestureEngine
from utils.camera_capture import open_first_working
from utils.sequential_word_collector import (
    SequentialConfig,
    SequentialWordCollector,
)

class VideoWorker(QThread):

    frame_payload = pyqtSignal(dict)
    word_locked = pyqtSignal(int, str, str)
    sequence_ready = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, config: Dict, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.cfg = config
        self._running = False
        cam = config["camera"]
        self.cap_idx = int(cam["device_index"])
        self.cap_w = int(cam["width"])
        self.cap_h = int(cam["height"])
        self.cap_backend = str(cam.get("backend", "auto"))

        mp_cfg = config["mediapipe"]
        self.extractor = HandLandmarkExtractor(
            max_num_hands=int(mp_cfg["max_num_hands"]),
            model_complexity=int(mp_cfg["model_complexity"]),
            min_detection_confidence=float(mp_cfg["min_detection_confidence"]),
            min_tracking_confidence=float(mp_cfg["min_tracking_confidence"]),
        )

        gcfg = config["gesture"]
        self.class_names: List[str] = list(gcfg["classes"])
        if not self.class_names:
            raise ValueError(
                "gesture.classes rỗng. Với VSL: chạy prepare + train, "
                "chạy scripts/collect_dataset.py hoặc training/vsl pipeline prepare"
            )
        self.seq_len = int(gcfg["sequence_length"])
        self.stride = max(1, int(gcfg["inference_stride"]))
        self.conf_thr = float(gcfg["confidence_threshold"])
        min_margin = float(gcfg.get("min_top1_margin", 0.30))
        min_prob = float(gcfg.get("min_accept_prob", 0.78))

        vsl = config.get("vsl") or {}
        self.onnx_path = Path(config["paths"].get("onnx_model", "models/onnx/gesture.onnx"))
        self.engine = GestureEngine(
            checkpoint_path=Path(config["paths"]["gesture_checkpoint"]),
            onnx_path=self.onnx_path,
            class_names=self.class_names,
            sequence_length=self.seq_len,
            mixed_precision=bool(gcfg.get("mixed_precision", True)),
            compile_model=bool(gcfg.get("torch_compile", False)),
            allow_untrained_demo=bool(gcfg.get("allow_untrained_demo", True)),
            labels_vi_override=config.get("vsl_labels_vi"),
            voya_root=Path(vsl["voya_root"]) if vsl.get("voya_root") else None,
            prediction_smoothing=float(gcfg.get("prediction_smoothing", 0.65)),
            min_top1_margin=min_margin,
            min_accept_prob=min_prob,
            voya_min_top1_margin=float(gcfg.get("voya_min_top1_margin", 0.45)),
            voya_min_accept_prob=float(gcfg.get("voya_min_accept_prob", 0.90)),
        )
        self.gesture_demo_mode = bool(getattr(self.engine, "demo_mode", False))
        self.gesture_voya_mode = bool(getattr(self.engine, "is_voya_checkpoint", False))

        bcfg = config["buffer"]
        self.max_words_llm = int(bcfg.get("max_words_before_llm", 5))
        self.seq_collector = SequentialWordCollector(
            self.class_names,
            display_fn=self.engine.display_label,
            cfg=SequentialConfig(
                max_words=self.max_words_llm,
                word_seconds=float(bcfg.get("word_capture_seconds", 3.0)),
                min_votes_per_slot=int(bcfg.get("min_votes_per_slot", 4)),
                min_vote_ratio=float(bcfg.get("min_vote_ratio", 0.55)),
                allow_repeat_words=bool(bcfg.get("allow_repeat_words", False)),
            ),
        )

        self._seq = deque(maxlen=self.seq_len)
        self._miss_frames = 0
        self._frame_counter = 0

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        self._running = True
        try_indices = list(range(0, int(self.cfg["camera"].get("try_indices_max", 4))))
        cap, used_idx = open_first_working(
            self.cap_idx,
            backend_name=self.cap_backend,
            try_indices=try_indices,
        )
        self.cap_idx = used_idx
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cap_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cap_h)

        if not cap.isOpened():
            self.error.emit(
                f"Không mở được camera (đã thử index {try_indices}). "
                "Đổi camera.device_index trong configs/default.yaml hoặc tắt app đang dùng webcam."
            )
            return

        for _ in range(15):
            cap.read()

        prev_time = time.perf_counter()
        ema_fps = 25.0
        read_fail_streak = 0

        try:
            while self._running:
                ok, frame = cap.read()
                if not ok:
                    read_fail_streak += 1
                    if read_fail_streak > 90:
                        self.error.emit(
                            "Camera đã mở nhưng không nhận được khung hình (read() luôn thất bại). "
                            "Thử backend: camera.backend: msmf hoặc dshow trong config; "
                            "kiểm tra quyền Camera (Windows Settings → Privacy)."
                        )
                        break
                    continue
                read_fail_streak = 0

                frame = bilateral_denoise(frame)
                try:
                    lm_result = self.extractor.process_frame(frame)
                    annotated = self.extractor.annotate_last(frame)
                except Exception as exc:
                    self.error.emit(f"Lỗi xử lý frame / MediaPipe: {exc}")
                    continue

                hands_visible = lm_result.confidence_ok

                if hands_visible:
                    self._miss_frames = 0
                    self._seq.append(lm_result.vector.copy())
                else:
                    self._miss_frames += 1
                    if self._miss_frames > 8:
                        self._seq.clear()

                top5: List[Tuple[str, float]] = []
                pred_word = ""
                top1_p = 0.0
                pred_cls: Optional[int] = None

                self._frame_counter += 1
                collecting = self.seq_collector.is_collecting
                infer_ok = (
                    collecting
                    and hands_visible
                    and len(self._seq) >= max(4, self.seq_len // 4)
                    and (self._frame_counter % self.stride == 0)
                )
                if infer_ok:
                    seq_arr = np.stack(self._seq, axis=0).astype(np.float32)
                    cls_id, top1_p, top5 = self.engine.predict(seq_arr)
                    top5 = [
                        (self.engine.display_label(n), p) for n, p in top5
                    ]
                    if cls_id >= 0 and top1_p >= self.conf_thr:
                        pred_cls = cls_id
                        pred_word = self.engine.display_label(self.class_names[cls_id])
                    elif top5:
                        pred_word = f"? ({top5[0][0]} {top1_p:.0%})"

                if collecting:
                    events, ready = self.seq_collector.on_frame(
                        hands_visible, pred_cls
                    )
                    for ev in events:
                        if ev.startswith("word:"):
                            slug = ev.split(":", 1)[1]
                            idx = len(self.seq_collector.words_locked) - 1
                            self.word_locked.emit(
                                idx,
                                slug,
                                self.engine.display_label(slug),
                            )
                    if ready:
                        self.sequence_ready.emit(
                            list(self.seq_collector.words_locked)
                        )

                snap = self.seq_collector.snapshot()

                now = time.perf_counter()
                inst_fps = 1.0 / max(1e-6, now - prev_time)
                prev_time = now
                ema_fps = 0.9 * ema_fps + 0.1 * inst_fps

                payload = {
                    "frame_bgr": annotated,
                    "fps": float(ema_fps),
                    "top5": top5 if collecting else [],
                    "stable_words": self.seq_collector.locked_display_words(),
                    "collector_status": snap.status_line,
                    "collector_hint": snap.hint_line,
                    "collecting": collecting,
                    "hands_visible": hands_visible,
                }
                self.frame_payload.emit(payload)
        finally:
            cap.release()
            self.extractor.close()

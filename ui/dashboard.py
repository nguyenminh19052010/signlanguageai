
from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.sentence_generator import SentenceGenerator
from tts.tts_engine import TTSEngine
from ui.audio_player import AudioPlayer
from ui.panels.llm_tts_panel import LlmTtsPanel
from ui.panels.prediction_panel import PredictionPanel
from ui.panels.video_panel import VideoPanel
from ui.panels.word_panel import WordPanel
from ui.video_worker import VideoWorker
from utils.gpu_monitor import read_gpu_stats

class Dashboard(QMainWindow):

    llm_done = pyqtSignal(str)
    tts_done = pyqtSignal(object, object, object)

    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.cfg = config
        self.llm_done.connect(self._on_llm_done)
        self.tts_done.connect(self._on_tts_done)
        self.setWindowTitle(config["project"]["name"])

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root_layout.addWidget(splitter)

        anim_ms = int(config["ui"].get("pipeline_anim_ms", 120))
        self.video_panel = VideoPanel(anim_ms=anim_ms)

        right = QWidget()
        rl = QVBoxLayout(right)

        self.prediction_panel = PredictionPanel()
        rl.addWidget(self.prediction_panel)

        self.word_panel = WordPanel()
        rl.addWidget(self.word_panel)

        default_lang = config["tts"].get("default_lang", "vi")
        self.llm_tts_panel = LlmTtsPanel(default_lang=default_lang)
        rl.addWidget(self.llm_tts_panel)

        btn_row = QGridLayout()
        self.btn_start = QPushButton("Start camera")
        self.btn_stop = QPushButton("Stop camera")
        self.btn_reset = QPushButton("Reset buffer")
        self.btn_force_llm = QPushButton("Generate from current words")
        btn_row.addWidget(self.btn_start, 0, 0)
        btn_row.addWidget(self.btn_stop, 0, 1)
        btn_row.addWidget(self.btn_reset, 1, 0)
        btn_row.addWidget(self.btn_force_llm, 1, 1)
        rl.addLayout(btn_row)

        splitter.addWidget(self.video_panel)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        self.worker: Optional[VideoWorker] = None
        self.llm = SentenceGenerator(config)
        self.tts = TTSEngine(config)
        self.audio = AudioPlayer(parent_widget=self)

        if not self.llm.has_model_file():
            self.llm_tts_panel.set_llm_status(
                "LLM: chế độ nhanh (chưa có file .gguf — ghép câu đơn giản)"
            )

        if not self.audio.qt_available:
            self.llm_tts_panel.set_tts_status(
                "Audio: Qt Multimedia không có — dùng pygame (MP3)"
            )

        self.btn_start.clicked.connect(self.start_camera)
        self.btn_stop.clicked.connect(self.stop_camera)
        self.btn_reset.clicked.connect(self.reset_buffer)
        self.btn_force_llm.clicked.connect(self.force_llm)

        self.gpu_timer = QTimer(self)
        self.gpu_timer.timeout.connect(self._refresh_gpu)
        self.gpu_timer.start(400)

    def start_camera(self) -> None:
        if self.worker and self.worker.isRunning():
            return
        try:
            self.worker = VideoWorker(self.cfg)
        except Exception as exc:
            QMessageBox.critical(self, "Gesture model", str(exc))
            return
        self.worker.frame_payload.connect(self._on_frame_payload)
        self.worker.word_locked.connect(self._on_word_locked)
        self.worker.sequence_ready.connect(self._on_sequence_ready)
        self.worker.error.connect(lambda m: QMessageBox.warning(self, "Camera", m))
        self.worker.start()

        if getattr(self.worker, "gesture_demo_mode", False):
            QMessageBox.information(
                self,
                "Gesture model",
                "Chế độ DEMO: chưa có gesture.pt — chỉ test camera/UI.\n\n"
                "Thu webcam rồi train:\n"
                "  python scripts/collect_dataset.py\n"
                "  python training/train.py",
            )
        elif getattr(self.worker, "gesture_voya_mode", False) and self.cfg.get(
            "gesture", {}
        ).get("warn_voya_on_webcam", True):
            QMessageBox.warning(
                self,
                "Model VOYA + webcam",
                "File gesture.pt train trên dataset VOYA (keypoint khác MediaPipe).\n\n"
                "Trên webcam thường chỉ đoán nhầm vài lớp — không phải lỗi camera.\n\n"
                "Cách dùng thật:\n"
                "  1) python scripts/collect_dataset.py  (5–10 ký hiệu của bạn)\n"
                "  2) python training/train.py\n"
                "  3) Thay models/checkpoints/gesture.pt + configs/vocab.json\n\n"
                "App đã bật lọc chặt để giảm spam từ.",
            )

    def stop_camera(self) -> None:
        if self.worker:
            self.worker.stop()
            self.worker.wait(3000)

    def reset_buffer(self) -> None:
        if self.worker and hasattr(self.worker, "seq_collector"):
            self.worker.seq_collector.reset_cycle()
            self.worker.engine.reset_smoothing()
            self.word_panel.set_phase("Đã reset — ký từ 1/5 trong 3 giây")
            self.word_panel.clear()
            self.llm_tts_panel.clear_llm()

    def _resume_recognition(self) -> None:
        if self.worker and hasattr(self.worker, "seq_collector"):
            self.worker.seq_collector.resume_after_processing()
            self.worker.engine.reset_smoothing()
            self.worker._seq.clear()
        self.word_panel.set_phase("Ký từ 1/5 — mỗi từ giữ ~3 giây")

    def _on_frame_payload(self, payload: Dict) -> None:
        self.video_panel.update_frame(payload["frame_bgr"])
        self.prediction_panel.update_fps(payload["fps"])
        self.prediction_panel.update_topk(payload.get("top5", []))
        words = payload.get("stable_words", [])
        self.word_panel.set_words(words)
        if payload.get("collector_status"):
            self.word_panel.set_phase(
                f"{payload['collector_status']}\n{payload.get('collector_hint', '')}"
            )

    def _on_word_locked(self, index: int, slug: str, display: str) -> None:
        n = self.worker.max_words_llm if self.worker else 5
        self.word_panel.set_phase(
            f"✓ Từ {index + 1}/{n}: {display}\nTiếp theo: ký từ {index + 2}/{n} (~3 giây)"
            if index + 1 < n
            else f"✓ Đủ {n} từ — đang ghép câu…"
        )
        if self.worker:
            self.word_panel.set_words(
                self.worker.seq_collector.locked_display_words()
            )

    def _on_sequence_ready(self, words: List[str]) -> None:
        self.word_panel.set_phase("Đủ 5 từ — đang ghép câu và đọc (tạm dừng nhận diện)")
        self._run_llm(words)

    def force_llm(self) -> None:
        if not self.worker:
            return
        words = list(self.worker.seq_collector.words_locked)
        if not words:
            QMessageBox.information(self, "LLM", "Chưa có từ nào trong chu kỳ hiện tại.")
            return
        self.worker.seq_collector.begin_processing()
        self._run_llm(words)

    def _run_llm(self, words: List[str]) -> None:
        if self.worker and hasattr(self.worker, "engine"):
            words = [self.worker.engine.display_label(w) for w in words]
        if not words:
            return
        self.llm_tts_panel.set_llm_status("LLM: đang ghép câu…")

        def worker():
            try:
                text = self.llm.generate_sentence(words)
            except Exception as exc:
                text = f"[Lỗi LLM] {exc}"
            self.llm_done.emit(text)

        threading.Thread(target=worker, daemon=True).start()

    def _on_llm_done(self, text: str) -> None:
        self.llm_tts_panel.set_llm_output(text)
        mode = "stub" if not self.llm.has_model_file() else "GGUF"
        self.llm_tts_panel.set_llm_status(f"LLM: xong ({mode})")
        if not text.strip():
            self._after_sentence_spoken()
            return
        self._play_tts(text, on_playback_done=self._after_sentence_spoken)

    def _after_sentence_spoken(self) -> None:
        self.llm_tts_panel.set_tts_status("Audio: xong")
        self._resume_recognition()
        self.word_panel.clear()
        self.llm_tts_panel.clear_llm()

    def _play_tts(self, text: str, on_playback_done=None) -> None:
        if not text.strip():
            if on_playback_done:
                on_playback_done()
            return
        self.llm_tts_panel.set_tts_status("Audio: đang tổng hợp giọng nói…")

        def done_cb(err, path):
            self.tts_done.emit(err, path, on_playback_done)

        self.tts.speak_async(
            text,
            done_callback=done_cb,
            lang=self.llm_tts_panel.tts_lang,
        )

    def _on_tts_done(self, err, path, on_playback_done) -> None:
        if err:
            print(f"[TTS Error] Synthesis failed: {err}")
            self.llm_tts_panel.set_tts_status(f"Lỗi: {err}")
            self._resume_recognition()
            self.word_panel.clear()
            self.llm_tts_panel.clear_llm()
            return
        assert path is not None
        self.llm_tts_panel.set_tts_status("Audio: đang phát…")
        self.audio.play(path, on_done=on_playback_done)

    def _refresh_gpu(self) -> None:
        idx = int(self.cfg["performance"].get("cuda_device", 0))
        stats = read_gpu_stats(idx)
        if stats.available:
            self.prediction_panel.update_gpu(
                f"GPU {stats.name}: {stats.utilization_percent:.0f}% | "
                f"{stats.mem_used_mib:.0f}/{stats.mem_total_mib:.0f} MiB"
            )
        else:
            self.prediction_panel.update_gpu("GPU: unavailable / driver not detected")

    def closeEvent(self, event) -> None:
        self.stop_camera()
        super().closeEvent(event)

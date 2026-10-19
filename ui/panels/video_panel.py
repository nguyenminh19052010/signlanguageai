
from __future__ import annotations

from typing import List

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QImage, QPixmap
from PyQt6.QtWidgets import QGroupBox, QLabel, QVBoxLayout, QWidget

def numpy_bgr_to_qpixmap(frame_bgr: np.ndarray) -> QPixmap:

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)

class VideoPanel(QWidget):

    STAGES = [
        "[Camera]",
        "[Hand Detection]",
        "[Gesture Recognition]",
        "[5 Words Buffer]",
        "[LLM Sentence]",
        "[Speech Output]",
    ]

    def __init__(self, anim_ms: int = 120, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)

        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.video_label)

        pipe = QGroupBox("Pipeline")
        pl = QVBoxLayout(pipe)
        self.pipeline_labels: List[QLabel] = []
        for s in self.STAGES:
            lab = QLabel(s)
            lab.setFont(QFont("Segoe UI", 10))
            self.pipeline_labels.append(lab)
            pl.addWidget(lab)
        layout.addWidget(pipe)

        self._pipeline_stage = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._animate_pipeline)
        self._timer.start(anim_ms)

    def update_frame(self, frame_bgr: np.ndarray) -> None:

        pix = numpy_bgr_to_qpixmap(frame_bgr)
        self.video_label.setPixmap(
            pix.scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _animate_pipeline(self) -> None:
        for i, lab in enumerate(self.pipeline_labels):
            lab.setStyleSheet(
                "color: #8bc34a; font-weight: bold;"
                if i == self._pipeline_stage
                else "color: #bbbbbb;"
            )
        self._pipeline_stage = (self._pipeline_stage + 1) % len(self.pipeline_labels)

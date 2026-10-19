
from __future__ import annotations

from typing import List, Tuple

from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

class PredictionPanel(QWidget):

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)

        self.fps_label = QLabel("FPS: —")
        self.gpu_label = QLabel("GPU: —")
        layout.addWidget(self.fps_label)
        layout.addWidget(self.gpu_label)

        topk_box = QGroupBox("Top-5 predictions")
        topk_l = QVBoxLayout(topk_box)
        self.topk_bars: List[QProgressBar] = []
        self.topk_names: List[QLabel] = []
        for _ in range(5):
            row = QHBoxLayout()
            name = QLabel("—")
            bar = QProgressBar()
            bar.setRange(0, 100)
            row.addWidget(name)
            row.addWidget(bar)
            topk_l.addLayout(row)
            self.topk_names.append(name)
            self.topk_bars.append(bar)
        layout.addWidget(topk_box)

    def update_fps(self, fps: float) -> None:
        self.fps_label.setText(f"FPS: {fps:.1f}")

    def update_gpu(self, text: str) -> None:
        self.gpu_label.setText(text)

    def update_topk(self, top5: List[Tuple[str, float]]) -> None:
        for i in range(5):
            if i < len(top5):
                name, p = top5[i]
                self.topk_names[i].setText(name)
                self.topk_bars[i].setValue(int(p * 100))
            else:
                self.topk_names[i].setText("—")
                self.topk_bars[i].setValue(0)

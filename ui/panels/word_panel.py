
from __future__ import annotations

from typing import List

from PyQt6.QtWidgets import (
    QGroupBox,
    QLabel,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

class WordPanel(QWidget):

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)

        words_box = QGroupBox("5 từ (mỗi từ ~3 giây)")
        wl = QVBoxLayout(words_box)
        self.phase_label = QLabel("Bấm Start — ký từ 1/5 trong 3 giây")
        self.phase_label.setWordWrap(True)
        wl.addWidget(self.phase_label)
        self.words_edit = QTextEdit()
        self.words_edit.setReadOnly(True)
        self.words_edit.setMaximumHeight(80)
        wl.addWidget(self.words_edit)
        layout.addWidget(words_box)

    def set_phase(self, text: str) -> None:
        self.phase_label.setText(text)

    def set_words(self, words: List[str]) -> None:
        self.words_edit.setText(", ".join(words))

    def clear(self) -> None:
        self.words_edit.clear()

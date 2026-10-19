
from __future__ import annotations

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

class LlmTtsPanel(QWidget):

    def __init__(self, default_lang: str = "vi", parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)

        llm_box = QGroupBox("LLM output")
        ll = QVBoxLayout(llm_box)
        self.llm_status = QLabel("LLM: idle")
        self.llm_edit = QTextEdit()
        self.llm_edit.setReadOnly(True)
        self.llm_edit.setFont(QFont("Segoe UI", 13))
        ll.addWidget(self.llm_status)
        ll.addWidget(self.llm_edit)
        layout.addWidget(llm_box)

        tts_box = QGroupBox("TTS")
        tl = QVBoxLayout(tts_box)
        self.tts_status = QLabel("Audio: idle")
        tl.addWidget(self.tts_status)
        lang_row = QHBoxLayout()
        lang_row.addWidget(QLabel("Voice language"))
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["English", "Tiếng Việt"])
        self.lang_combo.setCurrentIndex(1 if default_lang.startswith("vi") else 0)
        lang_row.addWidget(self.lang_combo)
        tl.addLayout(lang_row)
        layout.addWidget(tts_box)

        self._tts_lang = "vi" if default_lang.startswith("vi") else "en"
        self.lang_combo.currentIndexChanged.connect(self._on_lang_change)

    def _on_lang_change(self, idx: int) -> None:
        self._tts_lang = "vi" if idx == 1 else "en"

    @property
    def tts_lang(self) -> str:
        return self._tts_lang

    def set_llm_status(self, text: str) -> None:
        self.llm_status.setText(text)

    def set_llm_output(self, text: str) -> None:
        self.llm_edit.setPlainText(text)

    def set_tts_status(self, text: str) -> None:
        self.tts_status.setText(text)

    def clear_llm(self) -> None:
        self.llm_edit.clear()


from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Optional

from PyQt6.QtCore import QTimer

try:
    from PyQt6.QtCore import QUrl
    from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer

    _QT_MULTIMEDIA_AVAILABLE = True
except ImportError:
    QUrl = None
    QAudioOutput = None
    QMediaPlayer = None
    _QT_MULTIMEDIA_AVAILABLE = False

class AudioPlayer:

    def __init__(self, parent_widget=None) -> None:
        self._parent = parent_widget
        self._use_qt_audio = _QT_MULTIMEDIA_AVAILABLE
        self.media: Optional["QMediaPlayer"] = None
        self.audio_out: Optional["QAudioOutput"] = None
        self._pygame_mixer_ready = False
        self._tts_poll_timer: Optional[QTimer] = None

        if self._use_qt_audio and QMediaPlayer and QAudioOutput:
            self.media = QMediaPlayer()
            self.audio_out = QAudioOutput()
            self.media.setAudioOutput(self.audio_out)

    @property
    def qt_available(self) -> bool:
        return self._use_qt_audio and self.media is not None

    def play(
        self,
        path: Path,
        on_done: Optional[Callable[[], None]] = None,
    ) -> None:

        p = str(path)

        if self._use_qt_audio and self.media is not None and QUrl is not None:
            try:
                self.media.stop()
                self.media.setSource(QUrl.fromLocalFile(p))
                if self.audio_out is not None:
                    self.audio_out.setVolume(1.0)
                if on_done and hasattr(self.media, "mediaStatusChanged"):
                    try:
                        self.media.mediaStatusChanged.disconnect()
                    except Exception:
                        pass

                    def on_status(status):
                        from PyQt6.QtMultimedia import QMediaPlayer

                        if status in (QMediaPlayer.MediaStatus.EndOfMedia, QMediaPlayer.MediaStatus.InvalidMedia):
                            if status == QMediaPlayer.MediaStatus.InvalidMedia:
                                print("[AudioPlayer] Qt Media Status: InvalidMedia (failed to play)")
                            on_done()

                    self.media.mediaStatusChanged.connect(on_status)
                self.media.play()
                return
            except Exception as e:
                print(f"[AudioPlayer] Qt Multimedia playback failed: {e}. Falling back...")

        try:
            import pygame

            if not self._pygame_mixer_ready:
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=4096)
                self._pygame_mixer_ready = True
            pygame.mixer.music.stop()
            pygame.mixer.music.load(p)
            pygame.mixer.music.play()
            if on_done:
                if self._parent:
                    if self._tts_poll_timer is None:
                        self._tts_poll_timer = QTimer(self._parent)
                    try:
                        self._tts_poll_timer.timeout.disconnect()
                    except Exception:
                        pass

                    def poll_done():
                        import pygame as pg

                        if not pg.mixer.music.get_busy():
                            self._tts_poll_timer.stop()
                            on_done()

                    self._tts_poll_timer.timeout.connect(poll_done)
                    self._tts_poll_timer.start(250)
                else:
                    QTimer.singleShot(4000, on_done)
            return
        except Exception as e:
            print(f"[AudioPlayer] Pygame playback failed: {e}. Falling back...")

        try:
            import os

            if sys.platform == "win32":
                os.startfile(p)
                if on_done and self._parent:
                    QTimer.singleShot(4000, on_done)
            elif on_done:
                on_done()
        except Exception as e:
            print(f"[AudioPlayer] OS startfile fallback failed: {e}")
            if on_done:
                on_done()

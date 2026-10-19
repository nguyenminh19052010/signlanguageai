
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

@dataclass
class BufferConfig:
    voting_window: int = 12
    stability_frames: int = 6
    min_word_gap_frames: int = 8
    pause_reset_seconds: float = 2.0
    camera_fps: float = 30.0

@dataclass
class WordBufferState:
    stable_words: List[str] = field(default_factory=list)
    last_emitted_time_frames: int = 0
    frames_since_activity: int = 0

class WordSequenceBuffer:

    def __init__(self, class_names: List[str], cfg: BufferConfig) -> None:
        self.class_names = class_names
        self.cfg = cfg
        self._votes: Deque[int] = deque(maxlen=cfg.voting_window)
        self._state = WordBufferState()
        self._pending_label: Optional[int] = None
        self._pending_count = 0
        self._frame_index = 0

    def reset(self) -> None:
        self._votes.clear()
        self._state = WordBufferState()
        self._pending_label = None
        self._pending_count = 0

    def should_reset_pause(self) -> bool:
        pause_frames = int(self.cfg.pause_reset_seconds * self.cfg.camera_fps)
        return self._state.frames_since_activity >= pause_frames

    def on_frame(
        self,
        hands_visible: bool,
        predicted_class: Optional[int],
        inference_run: bool = False,
    ) -> Tuple[Optional[str], bool]:

        self._frame_index += 1

        if hands_visible:
            self._state.frames_since_activity = 0
        else:
            self._state.frames_since_activity += 1

        if self.should_reset_pause():
            return None, True

        if not hands_visible:
            self._votes.clear()
            self._pending_label = None
            self._pending_count = 0
            return None, False

        if not inference_run:
            return None, False

        val = predicted_class if predicted_class is not None else -1
        self._votes.append(val)

        if len(self._votes) < max(3, self.cfg.voting_window // 3):
            return None, False

        top_id, _ = Counter(self._votes).most_common(1)[0]
        if top_id != self._pending_label:
            self._pending_label = top_id
            self._pending_count = 1
            return None, False

        self._pending_count += 1
        if self._pending_count < self.cfg.stability_frames:
            return None, False

        if top_id == -1:
            return None, False

        word = self.class_names[top_id]
        self._state.stable_words.append(word)
        self._state.last_emitted_time_frames = self._frame_index
        self._votes.clear()
        self._pending_label = None
        self._pending_count = 0
        return word, False

    @property
    def words(self) -> List[str]:
        return list(self._state.stable_words)

    def take_words_for_llm(self, max_words: int) -> List[str]:
        if len(self._state.stable_words) < max_words:
            return []
        chunk = self._state.stable_words[:max_words]
        self._state.stable_words = self._state.stable_words[max_words:]
        return chunk

    def drain_all_words(self) -> List[str]:

        out = list(self._state.stable_words)
        self.reset()
        return out

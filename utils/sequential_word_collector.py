
from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional, Tuple

class CollectorPhase(str, Enum):
    COLLECTING = "collecting"
    PROCESSING = "processing"

@dataclass
class SequentialConfig:
    max_words: int = 5
    word_seconds: float = 3.0
    min_votes_per_slot: int = 4
    min_vote_ratio: float = 0.55
    allow_repeat_words: bool = False

@dataclass
class CollectorSnapshot:
    phase: CollectorPhase
    slot_index: int
    seconds_left: float
    words_locked: List[str]
    status_line: str
    hint_line: str

class SequentialWordCollector:
    def __init__(
        self,
        class_names: List[str],
        display_fn: Callable[[str], str],
        cfg: SequentialConfig,
    ) -> None:
        self.class_names = class_names
        self._display = display_fn
        self.cfg = cfg
        self.phase = CollectorPhase.COLLECTING
        self.words_locked: List[str] = []
        self._slot_votes: Counter[int] = Counter()
        self._slot_start: Optional[float] = None
        self._last_slot_failed = False

    def reset_cycle(self) -> None:
        self.phase = CollectorPhase.COLLECTING
        self.words_locked.clear()
        self._slot_votes.clear()
        self._slot_start = time.perf_counter()
        self._last_slot_failed = False

    def begin_processing(self) -> None:
        self.phase = CollectorPhase.PROCESSING
        self._slot_votes.clear()
        self._slot_start = None

    def resume_after_processing(self) -> None:
        self.reset_cycle()

    @property
    def is_collecting(self) -> bool:
        return self.phase == CollectorPhase.COLLECTING

    def snapshot(self) -> CollectorSnapshot:
        max_w = self.cfg.max_words
        if self.phase == CollectorPhase.PROCESSING:
            return CollectorSnapshot(
                phase=self.phase,
                slot_index=len(self.words_locked),
                seconds_left=0.0,
                words_locked=list(self.words_locked),
                status_line="Đang ghép câu và đọc…",
                hint_line="Chờ đọc xong — sẽ bắt đầu nhận 5 từ tiếp theo",
            )

        left = self._seconds_left()
        n = len(self.words_locked)
        slot_human = min(n + 1, max_w)
        labels = [self._display(w) for w in self.words_locked]
        locked_txt = ", ".join(labels) if labels else "(chưa có)"
        extra = " — ký lại!" if self._last_slot_failed else ""
        return CollectorSnapshot(
            phase=self.phase,
            slot_index=n,
            seconds_left=left,
            words_locked=list(self.words_locked),
            status_line=f"Từ {slot_human}/{max_w} — còn {left:.1f}s{extra}",
            hint_line=f"Đã ghi: {locked_txt}",
        )

    def _seconds_left(self) -> float:
        if self._slot_start is None:
            return float(self.cfg.word_seconds)
        elapsed = time.perf_counter() - self._slot_start
        return max(0.0, float(self.cfg.word_seconds) - elapsed)

    def on_frame(
        self, hands_visible: bool, predicted_class: Optional[int]
    ) -> Tuple[List[str], bool]:
        if self.phase != CollectorPhase.COLLECTING:
            return [], False

        if len(self.words_locked) >= self.cfg.max_words:
            return [], True

        if self._slot_start is None:
            self._slot_start = time.perf_counter()

        if hands_visible and predicted_class is not None and predicted_class >= 0:
            self._slot_votes[predicted_class] += 1

        if self._seconds_left() > 0:
            return [], False

        events: List[str] = []
        slug = self._finalize_slot()
        self._last_slot_failed = slug is None
        if slug:
            events.append(f"word:{slug}")

        self._slot_votes.clear()
        self._slot_start = time.perf_counter()

        if len(self.words_locked) >= self.cfg.max_words:
            self.begin_processing()
            return events, True

        return events, False

    def _finalize_slot(self) -> Optional[str]:
        if not self._slot_votes:
            return None

        top_id, top_count = self._slot_votes.most_common(1)[0]
        total = sum(self._slot_votes.values())
        if top_count < self.cfg.min_votes_per_slot:
            return None
        if top_count / total < self.cfg.min_vote_ratio:
            return None

        slug = self.class_names[top_id]
        if (
            not self.cfg.allow_repeat_words
            and self.words_locked
            and self.words_locked[-1] == slug
        ):
            return None

        self.words_locked.append(slug)
        return slug

    def locked_display_words(self) -> List[str]:
        return [self._display(w) for w in self.words_locked]

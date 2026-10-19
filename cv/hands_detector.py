
from __future__ import annotations

from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np

from .mediapipe_compat import get_solutions

_solutions = get_solutions()

@dataclass
class SmoothedLandmarks:

    vector: np.ndarray
    handedness: Tuple[str, ...]
    confidence_ok: bool
    raw_image_rgb: np.ndarray

class LandmarkSmoother:

    def __init__(self, alpha: float = 0.35) -> None:
        self.alpha = alpha
        self._prev: Optional[np.ndarray] = None

    def update(self, vec: np.ndarray) -> np.ndarray:
        if self._prev is None or self._prev.shape != vec.shape:
            self._prev = vec.copy()
            return vec
        blended = self.alpha * vec + (1.0 - self.alpha) * self._prev
        self._prev = blended.astype(np.float32)
        return self._prev

class HandLandmarkExtractor:

    def __init__(
        self,
        max_num_hands: int = 2,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.65,
        min_tracking_confidence: float = 0.55,
        smooth_alpha: float = 0.35,
    ) -> None:
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self._hands = _solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._draw = _solutions.drawing_utils
        self._styles = _solutions.drawing_styles
        self._smoother = LandmarkSmoother(alpha=smooth_alpha)
        self._last_results = None

    def close(self) -> None:
        self._hands.close()

    def _normalize_hand(self, landmarks, frame_shape: Tuple[int, int, int]) -> np.ndarray:
        h, w = frame_shape[0], frame_shape[1]
        pts = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in landmarks], dtype=np.float32)
        wrist = pts[0].copy()
        pts -= wrist
        scale = np.linalg.norm(pts[9] - pts[0]) + 1e-6
        pts /= scale
        return pts.reshape(-1)

    def build_feature_vector(
        self,
        multi_hand_landmarks,
        handedness_list,
        frame_shape: Tuple[int, int, int],
    ) -> Tuple[np.ndarray, Tuple[str, ...], float]:

        vectors: List[np.ndarray] = []
        labels: List[str] = []
        scores: List[float] = []

        if handedness_list:
            for h in handedness_list:
                scores.append(float(getattr(h, "score", 1.0)))
                labels.append(h.classification[0].label)

        for idx, lm in enumerate(multi_hand_landmarks or []):
            vec = self._normalize_hand(lm.landmark, frame_shape)
            vectors.append(vec)
            if idx >= 1:
                break

        if not vectors:
            return np.zeros(126, dtype=np.float32), tuple(), 0.0

        if len(vectors) == 1:
            full = np.concatenate([vectors[0], np.zeros(63, dtype=np.float32)])
        else:
            full = np.concatenate([vectors[0], vectors[1]])

        conf = float(min(scores)) if scores else 1.0
        handed_tuple = tuple(labels[:2])
        return full.astype(np.float32), handed_tuple, conf

    def annotate_frame(self, frame_bgr: np.ndarray, results) -> np.ndarray:

        if not results.multi_hand_landmarks:
            return frame_bgr
        annotated = frame_bgr.copy()
        for hand_landmarks in results.multi_hand_landmarks:
            self._draw.draw_landmarks(
                annotated,
                hand_landmarks,
                _solutions.hands.HAND_CONNECTIONS,
                self._styles.get_default_hand_landmarks_style(),
                self._styles.get_default_hand_connections_style(),
            )
        return annotated

    def process_frame(self, frame_bgr: np.ndarray) -> SmoothedLandmarks:

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb_proc = rgb.copy()
        rgb_proc.flags.writeable = False
        results = self._hands.process(rgb_proc)
        self._last_results = results

        mh = getattr(results, "multi_handedness", None)
        vector, handed_labels, conf = self.build_feature_vector(
            results.multi_hand_landmarks,
            mh,
            rgb.shape,
        )

        conf_ok = conf >= self.min_detection_confidence and (
            results.multi_hand_landmarks is not None
        )
        if results.multi_hand_landmarks:
            smoothed = self._smoother.update(vector)
        else:
            smoothed = vector
            conf_ok = False

        return SmoothedLandmarks(
            vector=smoothed,
            handedness=tuple(handed_labels) if handed_labels else tuple(),
            confidence_ok=conf_ok,
            raw_image_rgb=rgb,
        )

    def annotate_last(self, frame_bgr: np.ndarray) -> np.ndarray:

        if self._last_results is None:
            return frame_bgr
        return self.annotate_frame(frame_bgr, self._last_results)

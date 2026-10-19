
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from models.gesture_net import build_gesture_model, landmark_dim
from training.vsl.voya_labels import enrich_labels_vi

class GestureEngine:
    def __init__(
        self,
        checkpoint_path: Path,
        class_names: List[str],
        sequence_length: int,
        device: Optional[torch.device] = None,
        mixed_precision: bool = True,
        compile_model: bool = False,
        allow_untrained_demo: bool = False,
        labels_vi_override: Optional[dict] = None,
        voya_root: Optional[Path] = None,
        prediction_smoothing: float = 0.65,
        min_top1_margin: float = 0.30,
        min_accept_prob: float = 0.78,
        voya_min_top1_margin: Optional[float] = None,
        voya_min_accept_prob: Optional[float] = None,
        onnx_path: Optional[Path] = None,
        input_dim: int = 126,
    ) -> None:
        self.class_names = list(class_names)
        self.labels_vi: dict = {}
        self.sequence_length = sequence_length
        self.mixed_precision = mixed_precision
        self.demo_mode = False
        self._smooth = float(prediction_smoothing)
        self._min_margin = float(min_top1_margin)
        self._min_prob = float(min_accept_prob)
        self._ema_logits: Optional[torch.Tensor] = None
        self.train_source = "unknown"
        self.is_voya_checkpoint = False
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        self.few_shot_classes: dict[str, List[np.ndarray]] = {}
        self.few_shot_threshold = 0.82

        ckpt_path = Path(checkpoint_path)
        self.model = build_gesture_model(
            num_classes=len(self.class_names),
            input_dim=input_dim,
        )
        if ckpt_path.is_file():
            try:
                payload = torch.load(
                    ckpt_path, map_location=self.device, weights_only=False
                )
            except TypeError:
                payload = torch.load(ckpt_path, map_location=self.device)
            state = payload.get("model_state", payload)
            ckpt_classes = payload.get("classes")
            if ckpt_classes:
                self.class_names = list(ckpt_classes)
            self.labels_vi = dict(payload.get("labels_vi") or {})
            ckpt_seq = payload.get("seq_len")
            if ckpt_seq is not None:
                self.sequence_length = int(ckpt_seq)
            self.train_source = str(payload.get("train_source", "unknown"))
            from utils.gesture_checkpoint import is_voya_style_classes

            self.is_voya_checkpoint = self.train_source == "voya" or is_voya_style_classes(
                self.class_names
            )
            self.model = build_gesture_model(
                num_classes=len(self.class_names),
                input_dim=input_dim,
                state=state,
            )
            self.model.load_state_dict(state, strict=True)
        elif allow_untrained_demo:
            self.demo_mode = True
            log = logging.getLogger("signlang")
            log.warning(
                "Gesture checkpoint missing (%s). Running DEMO mode with random "
                "weights — predictions are meaningless until you train or add a .pt file.",
                ckpt_path,
            )
        else:
            raise FileNotFoundError(
                f"Gesture checkpoint not found: {ckpt_path}. "
                "Train or download weights (see README), or set "
                "gesture.allow_untrained_demo: true in config."
            )

        if labels_vi_override:
            self.labels_vi.update(labels_vi_override)
        self.labels_vi = enrich_labels_vi(
            self.class_names, self.labels_vi, voya_root=voya_root
        )
        if self.is_voya_checkpoint:
            if voya_min_top1_margin is not None:
                self._min_margin = float(voya_min_top1_margin)
            if voya_min_accept_prob is not None:
                self._min_prob = float(voya_min_accept_prob)

        self.model.to(self.device)
        self.model.eval()

        if compile_model and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model)
            except Exception:
                pass

        self.ort_session = None
        if onnx_path and onnx_path.is_file():
            try:
                import onnxruntime as ort
                self.ort_session = ort.InferenceSession(str(onnx_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
                print(f"✅ Loaded ONNX model from {onnx_path} (Providers: {self.ort_session.get_providers()})")
            except ImportError:
                print("⚠ onnxruntime not installed, falling back to PyTorch")

    def get_embedding(self, sequence: np.ndarray) -> np.ndarray:

        t, f = sequence.shape
        if t != self.sequence_length:
            x_old = np.linspace(0, 1, num=t, dtype=np.float32)
            x_new = np.linspace(0, 1, num=self.sequence_length, dtype=np.float32)
            resampled = np.empty((self.sequence_length, f), dtype=np.float32)
            for i in range(f):
                resampled[:, i] = np.interp(x_new, x_old, sequence[:, i])
            sequence = resampled
            
        batch = torch.from_numpy(sequence).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            with torch.autocast(device_type="cuda" if self.device.type == "cuda" else "cpu", enabled=self.mixed_precision):
                h = self.model.in_proj(batch).transpose(1, 2)
                h = self.model.conv_stack(h) if hasattr(self.model, "conv_stack") else self.model.conv(h)
                h = h.transpose(1, 2)
                out, _ = self.model.lstm(h)
                pooled, _ = out.max(dim=1)
                return pooled.squeeze(0).detach().cpu().numpy()

    def add_few_shot_example(self, slug: str, sequence: np.ndarray) -> None:
        emb = self.get_embedding(sequence)
        emb = emb / (np.linalg.norm(emb) + 1e-9)
        self.few_shot_classes.setdefault(slug, []).append(emb)
        print(f"Added few-shot example for '{slug}'. Total examples: {len(self.few_shot_classes[slug])}")

    @torch.inference_mode()
    def predict(
        self, sequence: np.ndarray
    ) -> Tuple[int, float, List[Tuple[str, float]]]:

        if sequence.shape[0] < 2:
            return -1, 0.0, []

        t, f = sequence.shape
        
        if t != self.sequence_length:
            x_old = np.linspace(0, 1, num=t, dtype=np.float32)
            x_new = np.linspace(0, 1, num=self.sequence_length, dtype=np.float32)
            resampled = np.empty(
                (self.sequence_length, f), dtype=np.float32
            )
            for i in range(f):
                resampled[:, i] = np.interp(x_new, x_old, sequence[:, i])
            sequence = resampled

        if self.few_shot_classes:
            emb = self.get_embedding(sequence)
            emb = emb / (np.linalg.norm(emb) + 1e-9)
            best_fs_slug = ""
            best_fs_score = -1.0
            
            for slug, examples in self.few_shot_classes.items():
                for ex in examples:
                    sim = float(np.dot(emb, ex))
                    if sim > best_fs_score:
                        best_fs_score = sim
                        best_fs_slug = slug
                        
            if best_fs_score >= self.few_shot_threshold:
                return -2, best_fs_score, [(best_fs_slug, best_fs_score)]

        if self.ort_session is not None:
            ort_inputs = {self.ort_session.get_inputs()[0].name: sequence[np.newaxis, ...]}
            logits_np = self.ort_session.run(None, ort_inputs)[0][0]
            logits = torch.from_numpy(logits_np).to(self.device)
        else:
            batch = torch.from_numpy(sequence).unsqueeze(0).to(self.device)
            with torch.autocast(
                device_type="cuda" if self.device.type == "cuda" else "cpu",
                enabled=self.mixed_precision and self.device.type == "cuda",
            ):
                logits = self.model(batch).squeeze(0)

        if self._smooth > 0:
            if self._ema_logits is None or self._ema_logits.shape != logits.shape:
                self._ema_logits = logits.detach()
            else:
                a = self._smooth
                self._ema_logits = a * self._ema_logits + (1.0 - a) * logits.detach()
            logits = self._ema_logits
        probs = torch.softmax(logits.unsqueeze(0), dim=-1).squeeze(0)

        top_probs, top_idx = torch.topk(probs, k=min(5, probs.numel()))
        pairs: List[Tuple[str, float]] = []
        for p, idx in zip(top_probs.tolist(), top_idx.tolist()):
            pairs.append((self.class_names[int(idx)], float(p)))

        best_idx = int(top_idx[0].item())
        best_p = float(top_probs[0].item())
        margin = best_p
        if top_probs.numel() > 1:
            margin = best_p - float(top_probs[1].item())

        if best_p < self._min_prob or margin < self._min_margin:
            return -1, best_p, pairs

        return best_idx, best_p, pairs

    def display_label(self, class_slug: str) -> str:
        vi = self.labels_vi.get(class_slug)
        if vi and not str(vi).lower().startswith("class_"):
            return str(vi)
        return class_slug.replace("_", " ")

    def reset_smoothing(self) -> None:
        self._ema_logits = None

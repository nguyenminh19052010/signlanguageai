
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch

from .gesture_net import build_gesture_model, landmark_dim

def export_onnx(
    checkpoint_path: Path,
    output_path: Path,
    config: Dict[str, Any],
    opset: int = 17,
) -> None:
    classes = list(config["gesture"]["classes"])
    seq_len = int(config["gesture"]["sequence_length"])
    device = torch.device("cpu")
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state", ckpt)
    if ckpt.get("classes"):
        classes = list(ckpt["classes"])
    model = build_gesture_model(
        num_classes=len(classes), input_dim=landmark_dim(), state=state
    )
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, seq_len, landmark_dim(), device=device)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["landmarks"],
        output_names=["logits"],
        dynamic_axes={
            "landmarks": {0: "batch", 1: "time"},
            "logits": {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )

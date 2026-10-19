
from __future__ import annotations

from pathlib import Path
from typing import Optional

def transcribe_file(
    wav_path: Path,
    model_size: str = "base",
    device: str = "cuda",
    compute_type: str = "float16",
) -> Optional[str]:
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return None

    wm = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, _ = wm.transcribe(str(wav_path))
    return " ".join(s.text.strip() for s in segments).strip()

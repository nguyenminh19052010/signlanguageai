from __future__ import annotations

import asyncio
import json
import logging
import tempfile
from collections import deque
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from inference.gesture_engine import GestureEngine
from llm.sentence_generator import SentenceGenerator
from tts.tts_engine import TTSEngine
from utils.config_loader import load_config
from utils.word_buffer import BufferConfig, WordSequenceBuffer

logger = logging.getLogger(__name__)

app = FastAPI(title="Sign Language Voice AI - Web Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

cfg = load_config()
gcfg = cfg["gesture"]
bcfg = cfg["buffer"]
seq_len = int(gcfg["sequence_length"])
stride = max(1, int(gcfg["inference_stride"]))
conf_thr = float(gcfg["confidence_threshold"])

ckpt_path = Path(cfg["paths"]["gesture_checkpoint"])
onnx_path = Path(cfg["paths"].get("onnx_model", "models/onnx/gesture.onnx"))

def _load_classes_from_checkpoint(path: Path) -> List[str]:

    if not path.is_file():
        return []
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        classes = payload.get("classes", [])
        if classes:
            print(f"✅ Loaded {len(classes)} classes from checkpoint: {classes}")
        return classes
    except Exception as e:
        print(f"⚠ Could not load classes from checkpoint: {e}")
        return []

class_names = _load_classes_from_checkpoint(ckpt_path)
if not class_names:
    class_names = list(gcfg.get("classes", []))
    print(f"⚠ Using fallback classes from config: {class_names}")

print("Loading Gesture Engine...")
try:
    engine = GestureEngine(
        checkpoint_path=ckpt_path,
        class_names=class_names,
        sequence_length=seq_len,
        mixed_precision=bool(gcfg.get("mixed_precision", True)),
        compile_model=False,
        onnx_path=onnx_path,
    )
    engine_ready = True
except FileNotFoundError as e:
    print(f"⚠ Gesture Engine not loaded: {e}")
    print("  → Server sẽ chạy nhưng không predict. Hãy train model trước.")
    engine = None
    engine_ready = False

print("Loading LLM Engine...")
llm = SentenceGenerator(cfg)

print("Loading TTS Engine...")
tts = TTSEngine(cfg)
print("Backend Ready!")

@app.websocket("/ws/sign-language")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("New WebSocket connection established")

    await websocket.send_json({
        "type": "init",
        "classes": class_names,
        "engine_ready": engine_ready,
    })

    word_buffer = WordSequenceBuffer(
        class_names,
        BufferConfig(
            voting_window=int(bcfg["voting_window"]),
            stability_frames=int(bcfg["stability_frames"]),
            min_word_gap_frames=int(bcfg["min_word_gap_frames"]),
            pause_reset_seconds=float(bcfg["pause_reset_seconds"]),
            camera_fps=30.0,
        ),
    )
    max_words_llm = int(bcfg["max_words_before_llm"])

    seq: deque = deque(maxlen=seq_len)
    frame_counter = 0
    miss_frames = 0

    async def process_llm_and_tts(words: list[str], lang: str):
        await websocket.send_json({"type": "llm_start", "words": words})

        def llm_worker():
            return llm.generate_sentence(words)

        sentence = await asyncio.to_thread(llm_worker)
        await websocket.send_json({"type": "llm_done", "text": sentence})

        def tts_worker():
            tmp = Path(tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name)
            tts.synthesize_to_file(sentence, tmp, lang=lang)
            return tmp

        audio_path = await asyncio.to_thread(tts_worker)

        with open(audio_path, "rb") as f:
            audio_data = f.read()

        await websocket.send_bytes(audio_data)
        Path(audio_path).unlink(missing_ok=True)

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "frame":
                vector = data.get("vector")
                hands_visible = data.get("hands_visible", False)
                lang = data.get("lang", "vi")

                if hands_visible and vector:
                    miss_frames = 0
                    seq.append(np.array(vector, dtype=np.float32))
                else:
                    miss_frames += 1
                    if miss_frames > 8:
                        seq.clear()

                frame_counter += 1

                min_frames = max(8, seq_len // 2)
                infer_ok = (
                    engine_ready
                    and hands_visible
                    and len(seq) >= min_frames
                    and (frame_counter % stride == 0)
                )

                pred_cls = None
                if infer_ok:
                    seq_arr = np.stack(seq, axis=0).astype(np.float32)
                    cls_id, top1_p, top5 = engine.predict(seq_arr)

                    await websocket.send_json(
                        {
                            "type": "prediction",
                            "top5": top5,
                            "best_p": float(top1_p),
                            "best_word": class_names[cls_id]
                            if top1_p >= conf_thr
                            else "",
                        }
                    )

                    if top1_p >= conf_thr:
                        pred_cls = cls_id

                emitted_word, pause_reset = word_buffer.on_frame(
                    hands_visible, pred_cls, inference_run=infer_ok
                )
                if pause_reset:
                    words = word_buffer.drain_all_words()
                    if words:
                        asyncio.create_task(process_llm_and_tts(words, lang))
                    seq.clear()
                    await websocket.send_json({"type": "buffer_reset"})

                if emitted_word:
                    await websocket.send_json(
                        {"type": "stable_words", "words": word_buffer.words}
                    )

                chunk = word_buffer.take_words_for_llm(max_words_llm)
                if chunk:
                    asyncio.create_task(process_llm_and_tts(chunk, lang))

            elif msg_type == "force_llm":
                lang = data.get("lang", "vi")
                words = word_buffer.drain_all_words()
                if words:
                    asyncio.create_task(process_llm_and_tts(words, lang))

            elif msg_type == "reset":
                word_buffer.reset()
                seq.clear()
                await websocket.send_json({"type": "buffer_reset"})

    except WebSocketDisconnect:
        print("Client disconnected")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

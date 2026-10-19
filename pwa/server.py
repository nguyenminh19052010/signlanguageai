
from __future__ import annotations

import os
import sys

os.environ.setdefault("PYTHONUTF8", "1")
if sys.stdout and sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import asyncio
import json
import logging
import tempfile
from collections import deque
from pathlib import Path
from typing import List, Optional
import shutil
import re

from pydantic import BaseModel
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent

if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
else:
    sys.path.remove(str(ROOT))
    sys.path.insert(0, str(ROOT))

from inference.gesture_engine import GestureEngine
from llm.sentence_generator import SentenceGenerator
from tts.tts_engine import TTSEngine
from utils.config_loader import load_config
from utils.sequential_word_collector import (
    SequentialConfig,
    SequentialWordCollector,
)

logger = logging.getLogger(__name__)

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

print("🔧 Loading Gesture Engine...")
try:
    mp_mode = cfg.get("mediapipe", {}).get("mode", "hands")
    input_dim = 1629 if mp_mode == "holistic" else 126
    
    engine = GestureEngine(
        checkpoint_path=ckpt_path,
        class_names=class_names,
        sequence_length=seq_len,
        mixed_precision=bool(gcfg.get("mixed_precision", True)),
        compile_model=False,
        allow_untrained_demo=bool(gcfg.get("allow_untrained_demo", True)),
        labels_vi_override=cfg.get("vsl_labels_vi"),
        voya_root=Path(cfg.get("vsl", {}).get("voya_root", "")) if cfg.get("vsl", {}).get("voya_root") else None,
        prediction_smoothing=float(gcfg.get("prediction_smoothing", 0.65)),
        min_top1_margin=float(gcfg.get("min_top1_margin", 0.30)),
        min_accept_prob=float(gcfg.get("min_accept_prob", 0.78)),
        onnx_path=onnx_path,
        input_dim=input_dim,
    )
    engine_ready = True
    class_names = engine.class_names
    print(f"✅ Gesture Engine ready — {len(class_names)} classes")
except FileNotFoundError as e:
    print(f"⚠ Gesture Engine not loaded: {e}")
    engine = None
    engine_ready = False

def reload_engine():
    global engine, engine_ready, class_names
    print("🔄 Reloading Gesture Engine...")
    try:
        new_class_names = _load_classes_from_checkpoint(ckpt_path)
        if not new_class_names:
            new_class_names = list(gcfg.get("classes", []))
            
        new_engine = GestureEngine(
            checkpoint_path=ckpt_path,
            class_names=new_class_names,
            sequence_length=seq_len,
            mixed_precision=bool(gcfg.get("mixed_precision", True)),
            compile_model=False,
            allow_untrained_demo=bool(gcfg.get("allow_untrained_demo", True)),
            labels_vi_override=cfg.get("vsl_labels_vi"),
            voya_root=Path(cfg.get("vsl", {}).get("voya_root", "")) if cfg.get("vsl", {}).get("voya_root") else None,
            prediction_smoothing=float(gcfg.get("prediction_smoothing", 0.65)),
            min_top1_margin=float(gcfg.get("min_top1_margin", 0.30)),
            min_accept_prob=float(gcfg.get("min_accept_prob", 0.78)),
            onnx_path=onnx_path,
            input_dim=input_dim,
        )
        engine = new_engine
        class_names = new_engine.class_names
        engine_ready = True
        print(f"✅ Gesture Engine reloaded — {len(class_names)} classes")
        return True
    except Exception as e:
        print(f"⚠ Gesture Engine reload failed: {e}")
        return False

print("🔧 Loading LLM Engine...")
llm = SentenceGenerator(cfg)
llm_mode = "GGUF" if llm.has_model_file() else "stub (ghép câu đơn giản)"
print(f"✅ LLM ready — mode: {llm_mode}")

print("🔧 Loading TTS Engine...")
tts = TTSEngine(cfg)
print("✅ TTS ready")

max_words_llm = int(bcfg.get("max_words_before_llm", 5))
seq_cfg = SequentialConfig(
    max_words=max_words_llm,
    word_seconds=float(bcfg.get("word_capture_seconds", 3.0)),
    min_votes_per_slot=int(bcfg.get("min_votes_per_slot", 4)),
    min_vote_ratio=float(bcfg.get("min_vote_ratio", 0.55)),
    allow_repeat_words=bool(bcfg.get("allow_repeat_words", False)),
)

app = FastAPI(title="Sign Language Voice AI — PWA")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = ROOT / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
async def serve_index():
    return FileResponse(str(STATIC_DIR / "index.html"))

@app.get("/manifest.json")
async def serve_manifest():
    return FileResponse(str(STATIC_DIR / "manifest.json"))

@app.get("/sw.js")
async def serve_sw():
    return FileResponse(str(STATIC_DIR / "sw.js"), media_type="application/javascript")

async def _process_llm_tts(
    ws: WebSocket,
    words: List[str],
    lang: str,
    collector: SequentialWordCollector,
) -> None:

    display_words = [engine.display_label(w) for w in words] if engine_ready else words
    try:
        await ws.send_json({"type": "llm_start", "words": display_words})

        sentence = await asyncio.to_thread(llm.generate_sentence, display_words)
        await ws.send_json({"type": "llm_done", "text": sentence})

        if sentence.strip():
            await ws.send_json({"type": "tts_start"})
            tmp = Path(tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name)
            await asyncio.to_thread(tts.synthesize_to_file, sentence, tmp, lang)

            audio_data = tmp.read_bytes()
            await ws.send_json({"type": "tts_audio", "size": len(audio_data)})
            await ws.send_bytes(audio_data)
            tmp.unlink(missing_ok=True)

    except Exception as exc:
        print(f"[Pipeline Error] {exc}")
        try:
            await ws.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass

    collector.resume_after_processing()
    if engine_ready:
        engine.reset_smoothing()
    try:
        await ws.send_json({"type": "cycle_reset"})
    except Exception:
        pass

@app.websocket("/ws/sign-language")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🔗 New WebSocket connection")

    seq: deque = deque(maxlen=seq_len)
    display_fn = engine.display_label if engine_ready else lambda x: x.replace("_", " ")
    collector = SequentialWordCollector(class_names, display_fn=display_fn, cfg=seq_cfg)
    frame_counter = 0
    miss_frames = 0
    lang = "vi"

    labels_vi = {}
    if engine_ready:
        labels_vi = {cn: engine.display_label(cn) for cn in class_names}

    await websocket.send_json({
        "type": "init",
        "classes": class_names,
        "labels_vi": labels_vi,
        "engine_ready": engine_ready,
        "max_words": max_words_llm,
        "word_seconds": float(bcfg.get("word_capture_seconds", 3.0)),
        "seq_len": seq_len,
        "llm_mode": llm_mode,
        "mp_mode": cfg.get("mediapipe", {}).get("mode", "hands")
    })

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "frame":
                vector = data.get("vector")
                hands_visible = data.get("hands_visible", False)
                lang = data.get("lang", lang)

                if hands_visible and vector:
                    miss_frames = 0
                    seq.append(np.array(vector, dtype=np.float32))
                else:
                    miss_frames += 1
                    if miss_frames > 8:
                        seq.clear()

                frame_counter += 1

                pred_cls = None
                collecting = collector.is_collecting
                min_frames_needed = max(4, seq_len // 4)
                infer_ok = (
                    engine_ready
                    and collecting
                    and hands_visible
                    and len(seq) >= min_frames_needed
                    and (frame_counter % stride == 0)
                )

                if infer_ok:
                    seq_arr = np.stack(seq, axis=0).astype(np.float32)
                    cls_id, top1_p, top5_raw = await asyncio.to_thread(
                        engine.predict, seq_arr
                    )
                    top5 = [
                        [engine.display_label(n), round(float(p), 4)]
                        for n, p in top5_raw
                    ]

                    if cls_id == -2:
                        best_word = top5_raw[0][0]
                        if best_word not in class_names:
                            class_names.append(best_word)
                            engine.class_names = class_names
                            labels_vi[best_word] = best_word
                        pred_cls = class_names.index(best_word)
                        await websocket.send_json({
                            "type": "prediction",
                            "top5": top5,
                            "best_p": round(float(top1_p), 4),
                            "best_word": best_word,
                        })
                    else:
                        await websocket.send_json({
                            "type": "prediction",
                            "top5": top5,
                            "best_p": round(float(top1_p), 4),
                            "best_word": engine.display_label(class_names[cls_id]) if cls_id >= 0 else "",
                        })

                        if cls_id >= 0 and top1_p >= conf_thr:
                            pred_cls = cls_id

                if collecting:
                    events, ready = collector.on_frame(hands_visible, pred_cls)

                    for ev in events:
                        if ev.startswith("word:"):
                            slug = ev.split(":", 1)[1]
                            idx = len(collector.words_locked) - 1
                            await websocket.send_json({
                                "type": "word_locked",
                                "index": idx,
                                "slug": slug,
                                "display": display_fn(slug),
                                "words": collector.locked_display_words(),
                            })

                    if ready:
                        words = list(collector.words_locked)
                        display_words = collector.locked_display_words()
                        await websocket.send_json({
                            "type": "sequence_ready",
                            "words": display_words,
                        })
                        asyncio.create_task(
                            _process_llm_tts(websocket, words, lang, collector)
                        )

                if frame_counter % 3 == 0:
                    snap = collector.snapshot()
                    await websocket.send_json({
                        "type": "collector_status",
                        "status": snap.status_line,
                        "hint": snap.hint_line,
                        "words": collector.locked_display_words(),
                        "collecting": collecting,
                        "hands_visible": hands_visible,
                    })

            elif msg_type == "force_llm":
                lang = data.get("lang", lang)
                words = list(collector.words_locked)
                if words:
                    collector.begin_processing()
                    asyncio.create_task(
                        _process_llm_tts(websocket, words, lang, collector)
                    )
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Chưa có từ nào trong chu kỳ hiện tại.",
                    })

            elif msg_type == "reset":
                collector.reset_cycle()
                seq.clear()
                frame_counter = 0
                if engine_ready:
                    engine.reset_smoothing()
                await websocket.send_json({"type": "cycle_reset"})

    except WebSocketDisconnect:
        print("🔌 Client disconnected")
    except Exception as exc:
        print(f"❌ WebSocket error: {exc}")

def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[àáạảãâầấậẩẫăằắặẳẵ]', 'a', text)
    text = re.sub(r'[èéẹẻẽêềếệểễ]', 'e', text)
    text = re.sub(r'[ìíịỉĩ]', 'i', text)
    text = re.sub(r'[òóọỏõôồốộổỗơờớợởỡ]', 'o', text)
    text = re.sub(r'[ùúụủũưừứựửữ]', 'u', text)
    text = re.sub(r'[ỳýỵỷỹ]', 'y', text)
    text = re.sub(r'[đ]', 'd', text)
    text = re.sub(r'[^a-z0-9]+', '_', text)
    return text.strip('_')

@app.get("/api/vocab")
async def get_vocab():
    vsl = cfg.get("vsl", {})
    vocab_path = Path(vsl.get("vocab_file", "configs/vocab.json"))
    if not vocab_path.is_absolute():
        vocab_path = (ROOT / vocab_path).resolve()
    
    vocab = {"classes": [], "labels_vi": {}}
    if vocab_path.is_file():
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
    
    dataset_root = Path(cfg["paths"]["dataset_root"])
    if not dataset_root.is_absolute():
        dataset_root = (ROOT / dataset_root).resolve()
        
    counts = {}
    for c in vocab.get("classes", []):
        class_dir = dataset_root / c
        if class_dir.is_dir():
            counts[c] = len(list(class_dir.glob("*.npz")))
        else:
            counts[c] = 0
            
    return {
        "classes": vocab.get("classes", []),
        "labels_vi": vocab.get("labels_vi", {}),
        "counts": counts
    }

class LabelRequest(BaseModel):
    name: str

@app.post("/api/vocab")
async def add_vocab(req: LabelRequest):
    slug = slugify(req.name)
    if not slug:
        return {"error": "Invalid label name"}
    
    vsl = cfg.get("vsl", {})
    vocab_path = Path(vsl.get("vocab_file", "configs/vocab.json"))
    if not vocab_path.is_absolute():
        vocab_path = (ROOT / vocab_path).resolve()
        
    vocab = {"classes": [], "labels_vi": {}}
    if vocab_path.is_file():
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
            
    if slug not in vocab.get("classes", []):
        vocab.setdefault("classes", []).append(slug)
        vocab.setdefault("labels_vi", {})[slug] = req.name
        
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
            
    if engine_ready:
        reload_engine()
        
    return {"slug": slug, "name": req.name}

@app.delete("/api/vocab/{slug}")
async def delete_vocab(slug: str):
    vsl = cfg.get("vsl", {})
    vocab_path = Path(vsl.get("vocab_file", "configs/vocab.json"))
    if not vocab_path.is_absolute():
        vocab_path = (ROOT / vocab_path).resolve()
        
    vocab = {"classes": [], "labels_vi": {}}
    if vocab_path.is_file():
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
            
    if slug in vocab.get("classes", []):
        vocab["classes"].remove(slug)
        vocab["labels_vi"].pop(slug, None)
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
            
    dataset_root = Path(cfg["paths"]["dataset_root"])
    if not dataset_root.is_absolute():
        dataset_root = (ROOT / dataset_root).resolve()
    class_dir = dataset_root / slug
    if class_dir.is_dir():
        shutil.rmtree(class_dir)
        
    if engine_ready:
        reload_engine()
        
    return {"status": "ok"}

class CollectRequest(BaseModel):
    slug: str
    landmarks: List[List[float]]

@app.post("/api/collect")
async def collect_data(req: CollectRequest):
    dataset_root = Path(cfg["paths"]["dataset_root"])
    if not dataset_root.is_absolute():
        dataset_root = (ROOT / dataset_root).resolve()
        
    class_dir = dataset_root / req.slug
    class_dir.mkdir(parents=True, exist_ok=True)
    
    existing = sorted(class_dir.glob("seq_*.npz"))
    n = 1
    if existing:
        try:
            n = int(existing[-1].stem.split("_")[-1]) + 1
        except ValueError:
            n = len(existing) + 1
            
    out = class_dir / f"seq_{n:04d}.npz"
    arr = np.array(req.landmarks, dtype=np.float32)
    np.savez_compressed(out, landmarks=arr)
    
    return {"status": "ok", "frames": arr.shape[0], "path": out.name}

@app.post("/api/fewshot")
async def add_fewshot(req: CollectRequest):
    if not engine_ready:
        return {"error": "Engine not ready"}
    arr = np.array(req.landmarks, dtype=np.float32)
    engine.add_few_shot_example(req.slug, arr)
    return {"status": "ok", "slug": req.slug}

@app.websocket("/ws/train")
async def websocket_train(websocket: WebSocket):
    await websocket.accept()
    process = None
    try:
        cmd = [sys.executable, "../training/train.py", "--config", "configs/default.yaml", "--epochs", "50"]
        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT
        )
        
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            await websocket.send_json({"type": "log", "message": line.decode("utf-8", errors="replace").strip()})
            
        await process.wait()
        
        if process.returncode == 0:
            await websocket.send_json({"type": "log", "message": "✅ Quá trình huấn luyện đã hoàn tất!"})
            ok = reload_engine()
            if ok:
                await websocket.send_json({"type": "reload_success"})
        else:
            await websocket.send_json({"type": "error", "message": f"❌ Huấn luyện lỗi, mã thoát {process.returncode}"})
            
    except WebSocketDisconnect:
        if process and process.returncode is None:
            process.terminate()
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})
        if process and process.returncode is None:
            process.terminate()

if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 60)
    print("  Sign Language Voice AI — PWA Server")
    print("  Open: http://localhost:8000")
    print("=" * 60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)

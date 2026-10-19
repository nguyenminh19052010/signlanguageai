
from __future__ import annotations

import asyncio
import tempfile
import threading
from pathlib import Path
from typing import Callable, Dict, Optional

class TTSEngine:
    def __init__(self, config: Dict) -> None:
        self.cfg = config["tts"]
        self.voice_en = self.cfg.get("voice_en", "en-US-JennyNeural")
        self.voice_vi = self.cfg.get("voice_vi", "vi-VN-HoaiMyNeural")
        self.default_lang = self.cfg.get("default_lang", "en")

    def pick_voice(self, lang: Optional[str] = None) -> str:
        lang = lang or self.default_lang
        return self.voice_vi if lang.startswith("vi") else self.voice_en

    async def _synthesize_file_async(self, text: str, voice: str, out_path: Path) -> None:
        try:
            import edge_tts
        except ImportError as exc:
            raise RuntimeError("edge-tts required: pip install edge-tts") from exc

        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(str(out_path))

    def synthesize_to_file(self, text: str, out_path: Path, lang: Optional[str] = None) -> None:

        voice = self.pick_voice(lang)
        
        import sys
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            
        asyncio.run(self._synthesize_file_async(text, voice, out_path))

    def speak_async(
        self,
        text: str,
        done_callback: Optional[
            Callable[[Optional[str], Optional[Path]], None]
        ] = None,
        lang: Optional[str] = None,
    ) -> None:

        def worker():
            try:
                tmp = Path(
                    tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
                )
                self.synthesize_to_file(text, tmp, lang=lang)
                if done_callback:
                    done_callback(None, tmp)
            except Exception as exc:
                if done_callback:
                    done_callback(str(exc), None)

        threading.Thread(target=worker, daemon=True).start()


from __future__ import annotations

import re
import threading
from queue import Queue
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional

SYSTEM_PROMPT_EN = """You convert fragmented sign-language words into a natural spoken sentence.

RULES:
- Output ONLY the final sentence.
- No explanations.
- No assistant replies.
- No markdown.
- Keep the sentence short and natural.

def clean_output(text: str) -> str:

    banned_phrases = [
        "Đúng rồi",
        "Đây là",
        "Bạn đã",
        "Giải thích",
        "Sure",
        "Certainly",
        "Here is",
    ]

    text = text.strip()

    for phrase in banned_phrases:
        idx = text.find(phrase)
        if idx != -1:
            text = text[:idx].strip()

    text = text.replace("```", "")
    text = text.replace('"', "")

    text = text.replace("\n", " ")

    text = re.sub(r"\s+", " ", text)

    return text.strip()

class SentenceGenerator:
    def __init__(self, config: Dict) -> None:
        self.cfg = config["llm"]
        self.path_gguf = Path(config["paths"]["llm_gguf"])

        self._llm = None

        lang = str(self.cfg.get("output_language", "vi")).lower()

        self._system_prompt = (
            SYSTEM_PROMPT_VI
            if lang.startswith("vi")
            else SYSTEM_PROMPT_EN
        )

        if (
            self.cfg.get("backend") == "llama_cpp"
            and not self.has_model_file()
        ):
            self.cfg = dict(self.cfg)
            self.cfg["backend"] = "stub"

    def has_model_file(self) -> bool:
        return self.path_gguf.is_file()

    def _ensure_llm(self):
        if self._llm is not None:
            return self._llm

        backend = self.cfg.get("backend", "llama_cpp")

        if backend == "stub" or not self.path_gguf.is_file():
            self._llm = None
            return None

        try:
            from llama_cpp import Llama

        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python not installed.\n"
                "Install with:\n"
                "pip install llama-cpp-python"
            ) from exc

        self._llm = Llama(
            model_path=str(self.path_gguf),
            n_ctx=int(self.cfg.get("n_ctx", 2048)),
            n_gpu_layers=int(self.cfg.get("n_gpu_layers", -1)),
            n_batch=int(self.cfg.get("n_batch", 256)),
            verbose=False,
        )

        return self._llm

    def build_prompt(self, words: List[str]) -> str:

        joined = ", ".join(words)

        if self._system_prompt == SYSTEM_PROMPT_VI:
            return (
                f"{self._system_prompt}\n\n"
                f"Từ: {joined}\n"
                f"Câu:"
            )

        return (
            f"{self._system_prompt}\n\n"
            f"Words: {joined}\n"
            f"Sentence:"
        )

    def generate_sentence(self, words: List[str]) -> str:

        prompt = self.build_prompt(words)

        llm = self._ensure_llm()

        if llm is None:
            return self._stub_sentence(words)

        out = llm(
            prompt,
            max_tokens=int(self.cfg.get("max_tokens", 64)),
            temperature=float(self.cfg.get("temperature", 0.2)),
            top_p=float(self.cfg.get("top_p", 0.8)),
            repeat_penalty=float(
                self.cfg.get("repeat_penalty", 1.1)
            ),
            stop=[
                "\n",
                "Đúng rồi",
                "Đây là",
                "Bạn đã",
                "Giải thích",
            ],
        )

        text = ""

        if isinstance(out, dict) and "choices" in out:
            text = out["choices"][0]["text"].strip()

        text = clean_output(text)

        return text or self._stub_sentence(words)

    def generate_stream(
        self,
        words: List[str],
        token_callback: Optional[Callable[[str], None]] = None,
    ) -> Iterator[str]:

        prompt = self.build_prompt(words)

        llm = self._ensure_llm()

        if llm is None:
            stub = self._stub_sentence(words)

            if token_callback:
                token_callback(stub)

            yield stub
            return

        stream = llm(
            prompt,
            max_tokens=int(self.cfg.get("max_tokens", 64)),
            temperature=float(self.cfg.get("temperature", 0.2)),
            top_p=float(self.cfg.get("top_p", 0.8)),
            repeat_penalty=float(
                self.cfg.get("repeat_penalty", 1.1)
            ),
            stream=True,
            stop=[
                "\n",
                "Đúng rồi",
                "Đây là",
                "Bạn đã",
                "Giải thích",
            ],
        )

        for chunk in stream:

            if (
                not isinstance(chunk, dict)
                or "choices" not in chunk
            ):
                continue

            choice = chunk["choices"][0]

            tok = choice.get("text", "")

            if tok:

                tok = clean_output(tok)

                if token_callback:
                    token_callback(tok)

                yield tok

    def generate_async(
        self,
        words: List[str],
        result_queue: "Queue[tuple[str, Optional[str]]]",
        stream_tokens: bool = True,
    ) -> None:

        def worker():

            try:

                if stream_tokens:

                    parts: List[str] = []

                    def cb(tok: str) -> None:
                        parts.append(tok)

                    for _ in self.generate_stream(
                        words,
                        token_callback=cb,
                    ):
                        pass

                    final_text = clean_output(
                        "".join(parts)
                    )

                    result_queue.put(
                        ("llm", final_text)
                    )

                else:

                    result_queue.put(
                        (
                            "llm",
                            self.generate_sentence(words),
                        )
                    )

            except Exception as exc:

                result_queue.put(
                    ("error", str(exc))
                )

        threading.Thread(
            target=worker,
            daemon=True,
        ).start()

    @staticmethod
    def _stub_sentence(words: List[str]) -> str:

        if not words:
            return ""

        sentence = " ".join(w.strip() for w in words if w.strip())
        if not sentence:
            return ""

        sentence = sentence[0].upper() + sentence[1:]

        if not sentence.endswith((".", "!", "?")):
            sentence += "."

        return sentence

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication

from ui.dashboard import Dashboard
from utils.config_loader import load_config
from utils.logger import setup_logging

def main() -> None:
    cfg = load_config()
    print("Checkpoint:", cfg["paths"]["gesture_checkpoint"])
    print("Vocab:", cfg.get("vsl", {}).get("vocab_file", "configs/vocab.json"))
    setup_logging(
        name=cfg["project"]["name"],
        level=str(cfg["project"].get("log_level", "INFO")),
        log_file=ROOT / "logs" / "runtime.log",
    )
    app = QApplication(sys.argv)
    win = Dashboard(cfg)
    win.resize(1280, 720)
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

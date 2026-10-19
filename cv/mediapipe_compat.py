
from __future__ import annotations

import atexit
import importlib
import os
import platform
import subprocess
import tempfile
from typing import Any

_junction_path: str | None = None

def _is_ascii(s: str) -> bool:
    try:
        s.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False

def _create_ascii_junction(target: str) -> str | None:

    if platform.system() != "Windows":
        return None

    tmp_root = tempfile.mkdtemp(prefix="mp_res_")
    junction = os.path.join(tmp_root, "mediapipe_pkg")
    try:
        subprocess.check_call(
            ["cmd", "/c", "mklink", "/J", junction, target],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return junction
    except Exception:
        return None

def _cleanup_junction() -> None:

    global _junction_path
    if _junction_path and os.path.isdir(_junction_path):
        try:
            subprocess.call(
                ["cmd", "/c", "rmdir", _junction_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            parent = os.path.dirname(_junction_path)
            if os.path.isdir(parent):
                os.rmdir(parent)
        except Exception:
            pass

def _patch_mediapipe_resource_path() -> None:

    global _junction_path

    try:
        from mediapipe.python import solution_base
    except ImportError:
        return

    sb_file = os.path.abspath(solution_base.__file__)
    parts = sb_file.split(os.sep)
    mp_root = os.sep.join(parts[:-3])

    if _is_ascii(mp_root):
        return

    junction = _create_ascii_junction(mp_root)
    if junction is None:
        return

    _junction_path = junction
    atexit.register(_cleanup_junction)

    _original_init = solution_base.SolutionBase.__init__

    def _patched_init(self, *args, **kwargs):

        from mediapipe.python._framework_bindings import resource_util

        _orig_abspath = os.path.abspath

        def _fake_abspath(p):
            result = _orig_abspath(p)
            if not _is_ascii(result):
                result = result.replace(mp_root, junction)
            return result

        os.path.abspath = _fake_abspath
        try:
            _original_init(self, *args, **kwargs)
        finally:
            os.path.abspath = _orig_abspath

    solution_base.SolutionBase.__init__ = _patched_init

_patch_mediapipe_resource_path()

def get_solutions() -> Any:

    import mediapipe as mp

    sol = getattr(mp, "solutions", None)
    if sol is not None:
        return sol

    for mod_name in (
        "mediapipe.python.solutions",
        "mediapipe.solutions",
    ):
        try:
            return importlib.import_module(mod_name)
        except ImportError:
            continue

    raise ImportError(
        "Không tìm thấy MediaPipe module 'solutions'. "
        "Thử: pip uninstall mediapipe -y && pip install 'mediapipe>=0.10.9,<0.11'"
    )

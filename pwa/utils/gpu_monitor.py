
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional

@dataclass
class GpuStats:

    name: str = "N/A"
    utilization_percent: float = 0.0
    mem_used_mib: float = 0.0
    mem_total_mib: float = 0.0
    available: bool = False

_NVML_INITIALIZED = False
_PYNVML = None

def try_import_pynvml():
    global _PYNVML
    if _PYNVML is not None:
        return _PYNVML
    try:
        import pynvml
        _PYNVML = pynvml
        return pynvml
    except Exception:
        return None

def init_nvml():
    global _NVML_INITIALIZED
    if _NVML_INITIALIZED:
        return True
    pynvml = try_import_pynvml()
    if pynvml is not None:
        try:
            pynvml.nvmlInit()
            _NVML_INITIALIZED = True
            return True
        except Exception:
            return False
    return False

def read_gpu_stats(device_index: int = 0) -> GpuStats:

    if init_nvml() and _PYNVML is not None:
        try:
            handle = _PYNVML.nvmlDeviceGetHandleByIndex(device_index)
            name = _PYNVML.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            util = _PYNVML.nvmlDeviceGetUtilizationRates(handle)
            mem = _PYNVML.nvmlDeviceGetMemoryInfo(handle)
            return GpuStats(
                name=name,
                utilization_percent=float(util.gpu),
                mem_used_mib=float(mem.used) / (1024**2),
                mem_total_mib=float(mem.total) / (1024**2),
                available=True,
            )
        except Exception:
            pass

    if shutil.which("nvidia-smi") is None:
        return GpuStats(available=False)

    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=1.0,
        )
        lines = [ln.strip() for ln in out.strip().splitlines() if ln.strip()]
        if not lines or device_index >= len(lines):
            return GpuStats(available=False)
        parts = [p.strip() for p in lines[device_index].split(",")]
        if len(parts) < 4:
            return GpuStats(available=False)
        name, util, used, total = parts[0], parts[1], parts[2], parts[3]
        return GpuStats(
            name=name,
            utilization_percent=float(util),
            mem_used_mib=float(used),
            mem_total_mib=float(total),
            available=True,
        )
    except Exception:
        return GpuStats(available=False)

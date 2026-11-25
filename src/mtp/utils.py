from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import time
import zlib

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_hash(s: str) -> int:
    """Process-independent string hash (Python's hash() is salted per process)."""
    return zlib.crc32(s.encode()) & 0x7FFFFFFF


def atomic_save(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


class JsonlLogger:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path

    def log(self, **rec) -> None:
        rec["wall_time"] = time.time()
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")


def env_info() -> dict:
    info = {"python": platform.python_version(), "torch": torch.__version__,
            "platform": platform.platform(), "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(gpu=p.name, gpu_mem_gb=round(p.total_memory / 2**30, 2),
                    compute_capability=f"{p.major}.{p.minor}", cuda=torch.version.cuda,
                    n_gpus=torch.cuda.device_count())
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        info["git_commit"] = None
    return info

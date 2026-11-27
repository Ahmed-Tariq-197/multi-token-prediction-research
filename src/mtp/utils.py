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
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")


def _git(*args: str) -> str | None:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root, not CWD
    try:
        return subprocess.check_output(["git", "-C", root, *args], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001 - git may be absent (Colab zip upload); provenance is best-effort
        return None


def env_info() -> dict:
    from importlib import metadata

    def ver(pkg):
        try:
            return metadata.version(pkg)
        except metadata.PackageNotFoundError:
            return None

    info = {"python": platform.python_version(), "torch": torch.__version__, "numpy": ver("numpy"),
            "tokenizers": ver("tokenizers"), "transformers": ver("transformers"),
            "platform": platform.platform(), "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(gpu=p.name, gpu_mem_gb=round(p.total_memory / 2**30, 2),
                    compute_capability=f"{p.major}.{p.minor}", cuda=torch.version.cuda,
                    n_gpus=torch.cuda.device_count())
    info["git_commit"] = _git("rev-parse", "--short", "HEAD")
    status = _git("status", "--porcelain")
    info["git_dirty"] = None if status is None else bool(status)
    return info


def truncate_jsonl(path: str, max_step: int) -> int:
    """Drop records with step > max_step (written after the checkpoint we resume from). Returns #dropped."""
    if not os.path.exists(path):
        return 0
    keep, dropped = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            if json.loads(line).get("step", 0) <= max_step:
                keep.append(line if line.endswith("\n") else line + "\n")
            else:
                dropped += 1
    if dropped:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.writelines(keep)
        os.replace(tmp, path)
    return dropped

"""Tiny synthetic corpus for smoke tests and CI (NOT used for any reported result)."""
import argparse, json, os
import numpy as np


def make(out_dir: str, vocab: int = 64, n_train: int = 200_000, n_val: int = 20_000, eos_id: int = 0, seed: int = 0):
    """Learnable toy language: documents of a noisy repeating motif, separated by EOS."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    def stream(n):
        out = []
        while len(out) < n:
            motif = rng.integers(1, vocab, size=rng.integers(3, 8))
            reps = rng.integers(4, 30)
            doc = np.tile(motif, reps)
            noise = rng.random(len(doc)) < 0.05
            doc[noise] = rng.integers(1, vocab, size=noise.sum())
            out.extend(doc.tolist()); out.append(eos_id)
        return np.array(out[:n], dtype=np.uint16)

    for name, n in (("train", n_train), ("val", n_val), ("test", n_val)):
        stream(n).tofile(os.path.join(out_dir, f"{name}.bin"))
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump({"synthetic": True, "vocab_size": vocab, "eos_id": eos_id,
                   "bytes_per_token": {"val": 1.0, "test": 1.0}}, f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="data/synthetic"); a = ap.parse_args()
    make(a.out)

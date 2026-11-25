"""Reproducible data + tokenizer pipeline.

  python scripts/prepare_data.py --source hf --out data/fineweb_edu_8k \
      --dataset HuggingFaceFW/fineweb-edu --config sample-10BT --gpt2-token-budget 700000000 --vocab-size 8192
  python scripts/prepare_data.py --source jsonl --jsonl docs.jsonl --out data/local   # offline / tests

Split rule (deterministic, order-independent): bucket = sha256(doc_id) mod 1000;
  bucket <  val_permille                  -> val
  bucket <  val_permille + test_permille  -> test
  otherwise                               -> train
The tokenizer (byte-level BPE, EOS = id 0) is trained ONLY on train-split text. Every arm uses the same files.
Status: the `--source jsonl` path is covered by tests/test_prepare_data.py; the `--source hf` path needs network
access and has not been executed in the authoring environment (verify with a small --gpt2-token-budget first).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from typing import Iterator

import numpy as np

EOS = "<|endoftext|>"


def bucket(doc_id: str) -> int:
    return int(hashlib.sha256(doc_id.encode()).hexdigest(), 16) % 1000


def split_of(doc_id: str, val_pm: int, test_pm: int) -> str:
    b = bucket(doc_id)
    return "val" if b < val_pm else "test" if b < val_pm + test_pm else "train"


def iter_hf(dataset: str, config: str, revision: str | None, gpt2_budget: int) -> Iterator[tuple[str, str]]:
    from datasets import load_dataset
    ds = load_dataset(dataset, name=config, split="train", streaming=True, revision=revision)
    seen = 0
    for ex in ds:
        yield ex["id"], ex["text"]
        seen += int(ex.get("token_count", 0))
        if seen >= gpt2_budget:
            return


def iter_jsonl(path: str) -> Iterator[tuple[str, str]]:
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            yield r["id"], r["text"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["hf", "jsonl"], required=True)
    ap.add_argument("--jsonl")
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--config", default="sample-10BT")
    ap.add_argument("--revision", default=None, help="pin the dataset git revision for reproducibility")
    ap.add_argument("--gpt2-token-budget", type=int, default=700_000_000,
                    help="stop streaming after this many GPT-2 tokens (dataset's own token_count field)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--val-permille", type=int, default=5)
    ap.add_argument("--test-permille", type=int, default=5)
    ap.add_argument("--tok-train-mb", type=int, default=300, help="MB of TRAIN text used to fit the BPE")
    a = ap.parse_args()

    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    os.makedirs(a.out, exist_ok=True)
    raw = {s: open(os.path.join(a.out, f"raw_{s}.jsonl"), "w") for s in ("train", "val", "test")}
    docs = iter_hf(a.dataset, a.config, a.revision, a.gpt2_token_budget) if a.source == "hf" else iter_jsonl(a.jsonl)
    n_docs = {"train": 0, "val": 0, "test": 0}
    t0 = time.time()
    for doc_id, text in docs:
        s = split_of(doc_id, a.val_permille, a.test_permille)
        raw[s].write(json.dumps({"id": doc_id, "text": text}) + "\n")
        n_docs[s] += 1
    for f in raw.values():
        f.close()
    print(f"[stage 1] fetched docs {n_docs} in {time.time() - t0:.0f}s")

    # ---- tokenizer on train text only
    def train_text() -> Iterator[str]:
        used = 0
        with open(os.path.join(a.out, "raw_train.jsonl")) as f:
            for line in f:
                t = json.loads(line)["text"]
                used += len(t.encode())
                yield t
                if used >= a.tok_train_mb * 2**20:
                    return

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=a.vocab_size, special_tokens=[EOS],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator(train_text(), trainer)
    assert tok.token_to_id(EOS) == 0, "EOS must be token id 0"
    tok_path = os.path.join(a.out, "tokenizer.json")
    tok.save(tok_path)
    tok_sha = hashlib.sha256(open(tok_path, "rb").read()).hexdigest()

    # ---- encode splits
    manifest = {"source": a.source, "dataset": a.dataset if a.source == "hf" else a.jsonl, "config": a.config,
                "revision": a.revision, "split_rule": f"sha256(id)%1000 < {a.val_permille} val; "
                f"< {a.val_permille + a.test_permille} test; else train", "vocab_size": a.vocab_size, "eos_id": 0,
                "tokenizer_sha256": tok_sha, "docs": n_docs, "tokens": {}, "bytes_per_token": {},
                "created_unix": time.time()}
    for s in ("train", "val", "test"):
        total_tokens, total_bytes = 0, 0
        out = open(os.path.join(a.out, f"{s}.bin"), "wb")
        batch: list[str] = []

        def flush():
            nonlocal total_tokens, total_bytes
            if not batch:
                return
            for enc, txt in zip(tok.encode_batch(batch), batch):
                ids = enc.ids + [0]
                np.asarray(ids, dtype=np.uint16).tofile(out)
                total_tokens += len(ids)
                total_bytes += len(txt.encode())
            batch.clear()

        with open(os.path.join(a.out, f"raw_{s}.jsonl")) as f:
            for line in f:
                batch.append(json.loads(line)["text"])
                if len(batch) >= 512:
                    flush()
        flush()
        out.close()
        manifest["tokens"][s] = total_tokens
        manifest["bytes_per_token"][s] = total_bytes / max(total_tokens, 1)
        print(f"[stage 3] {s}: {total_tokens:,} tokens, {total_bytes / max(total_tokens, 1):.3f} bytes/token")
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()

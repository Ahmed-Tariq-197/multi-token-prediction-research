import hashlib
import json
import os
import subprocess
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from budget import derive_budget  # noqa: E402
from prepare_data import bucket, split_of  # noqa: E402


def _write_docs(path, n=600):
    import random

    rng = random.Random(0)
    words = ["alpha", "beta", "gamma", "delta", "model", "token", "predict", "future", "head", "loss", "the", "of"]
    with open(path, "w") as f:
        for i in range(n):
            text = " ".join(rng.choice(words) for _ in range(rng.randint(40, 120)))
            f.write(json.dumps({"id": f"doc-{i}", "text": text}) + "\n")


def _run_prepare(jsonl, out, vocab="300"):
    subprocess.run(
        [
            sys.executable,
            os.path.join(ROOT, "scripts", "prepare_data.py"),
            "--source",
            "jsonl",
            "--jsonl",
            jsonl,
            "--out",
            out,
            "--vocab-size",
            vocab,
            "--tok-train-mb",
            "1",
            "--val-permille",
            "50",
            "--test-permille",
            "50",
        ],
        check=True,
        capture_output=True,
    )


def test_prepare_data_pipeline_is_deterministic_and_splits_are_disjoint(tmp_path):
    j = str(tmp_path / "docs.jsonl")
    _write_docs(j)
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    _run_prepare(j, a)
    _run_prepare(j, b)
    for f in ("train.bin", "val.bin", "test.bin", "tokenizer.json"):
        assert (
            hashlib.sha256(open(f"{a}/{f}", "rb").read()).hexdigest()
            == hashlib.sha256(open(f"{b}/{f}", "rb").read()).hexdigest()
        ), f"{f} not reproducible"
    man = json.load(open(f"{a}/manifest.json"))
    assert man["eos_id"] == 0 and man["vocab_size"] == 300 and set(man["tokens"]) == {"train", "val", "test"}
    ids = {s: {json.loads(l)["id"] for l in open(f"{a}/raw_{s}.jsonl")} for s in ("train", "val", "test")}
    assert not (ids["train"] & ids["val"] or ids["train"] & ids["test"] or ids["val"] & ids["test"])
    assert sum(len(v) for v in ids.values()) == 600
    assert all(split_of(i, 50, 50) == "val" for i in ids["val"])
    assert ids["val"] and ids["test"], "need non-empty val/test"


def test_split_is_order_independent():
    assert bucket("doc-1") == bucket("doc-1") and 0 <= bucket("x") < 1000


def _canary_vocab(tmp_path, target_split):
    j = str(tmp_path / f"docs_{target_split}.jsonl")
    _write_docs(j, 300)
    n = 0
    with open(j, "a") as f:
        for i in range(20000):
            if n >= 60:
                break
            if split_of(f"canary-{i}", 50, 50) == target_split:
                f.write(json.dumps({"id": f"canary-{i}", "text": "zzqxjvk wwqxzk " * 40}) + "\n")
                n += 1
    out = str(tmp_path / f"o_{target_split}")
    _run_prepare(j, out, vocab="420")
    return json.load(open(f"{out}/tokenizer.json"))["model"]["vocab"]


def test_tokenizer_trained_on_train_text_only(tmp_path):
    # positive control: canary text in TRAIN must leak into the vocabulary (proves the test can detect leakage)
    assert any("zz" in t or "qx" in t for t in _canary_vocab(tmp_path, "train"))
    # canary text that lands only in VAL must not
    assert not any("zz" in t or "qx" in t for t in _canary_vocab(tmp_path, "val"))


def test_budget_derivation_picks_binding_constraint_and_rounds():
    trials = [
        {"k": 1, "tokens_per_s": 40000.0, "params_main": 29_000_000, "oom": False},
        {"k": 4, "tokens_per_s": 24000.0, "params_main": 29_000_000, "oom": False},
        {"k": 4, "tokens_per_s": 99999.0, "params_main": 29_000_000, "oom": True},
    ]
    arms = [("ntp", 1, 1.0), ("mtp", 4, 1.0), ("soft", 4, 1.1)]
    r = derive_budget(
        trials,
        arms,
        seeds=3,
        global_batch_tokens=131072,
        max_run_hours=5,
        weekly_quota=30,
        accounts=4,
        campaign_fraction=0.6,
    )
    assert r["tokens_per_run"] % 131072 == 0
    assert r["tokens_per_run"] <= 20 * 29_000_000
    slow = min(r["tokens_per_s_by_arm"].values())
    assert r["hours_per_run_by_arm"]["soft"] <= 5.0 / 0.95 + 1e-6 or r["binding_constraint"] != "time"
    assert r["campaign_gpu_hours"] <= r["campaign_quota_available"] + 1e-6
    assert slow == 24000.0 / 1.1  # OOM trial must be ignored
    tiny = derive_budget(trials, arms, seeds=3, global_batch_tokens=131072, weekly_quota=1, accounts=1)
    assert tiny["warnings"], "starved quota must trigger the undertraining warning"


def test_calibrate_and_profile_smoke_on_cpu(tmp_path, synth_dir):
    cfgp = tmp_path / "c.yaml"
    cfgp.write_text(
        "model: {vocab_size: 64, d_model: 32, n_layers: 2, n_heads: 4, ffn_hidden: 64, max_seq_len: 32}\n"
        "data: {seq_len: 32, max_horizon: 4}\ntrain: {micro_batch_size: 4, global_batch_tokens: 128, "
        "precision: fp32, device: cpu}\n"
    )
    out = tmp_path / "cal.json"
    subprocess.run(
        [
            sys.executable,
            os.path.join(ROOT, "scripts", "calibrate.py"),
            "--config",
            str(cfgp),
            "--k",
            "1",
            "2",
            "--micro-batch",
            "4",
            "--steps",
            "3",
            "--warmup",
            "1",
            "--out",
            str(out),
        ],
        check=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": os.path.join(ROOT, "src")},
    )
    tr = json.load(open(out))["trials"]
    assert len(tr) == 2 and all(t["tokens_per_s"] > 0 for t in tr)
    assert tr[1]["params_total"] > tr[0]["params_total"] and tr[1]["params_main"] == tr[0]["params_main"]
    prof = tmp_path / "p.json"
    subprocess.run(
        [
            sys.executable,
            os.path.join(ROOT, "scripts", "profile_decode.py"),
            "--config",
            str(cfgp),
            "--prompt-len",
            "8",
            "--new-tokens",
            "8",
            "--out",
            str(prof),
        ],
        check=True,
        capture_output=True,
    )
    assert json.load(open(prof))["ms_per_token"] > 0


def test_eos_literal_in_text_cannot_inject_document_boundaries(tmp_path):
    import numpy as np

    j = str(tmp_path / "docs.jsonl")
    _write_docs(j, 200)
    with open(j, "a") as f:
        f.write(json.dumps({"id": "evil-1", "text": "before <|endoftext|> after " * 30}) + "\n")
    out = str(tmp_path / "o")
    _run_prepare(j, out)
    man = json.load(open(f"{out}/manifest.json"))
    assert man["docs_with_eos_literal_stripped"] >= 1
    for split in ("train", "val", "test"):
        ids = np.fromfile(f"{out}/{split}.bin", dtype=np.uint16)
        n_docs = sum(1 for _ in open(f"{out}/raw_{split}.jsonl"))
        assert int((ids == 0).sum()) == n_docs, f"{split}: EOS count must equal document count"


def test_manifest_hashes_and_token_counts_match_files(tmp_path):
    import numpy as np

    j = str(tmp_path / "docs.jsonl")
    _write_docs(j, 200)
    out = str(tmp_path / "o")
    _run_prepare(j, out)
    man = json.load(open(f"{out}/manifest.json"))
    for split in ("train", "val", "test"):
        raw = open(f"{out}/{split}.bin", "rb").read()
        assert hashlib.sha256(raw).hexdigest() == man["sha256"][split]
        assert len(np.frombuffer(raw, dtype=np.uint16)) == man["tokens"][split]


def test_prepare_rejects_vocab_that_does_not_fit_uint16(tmp_path):
    j = str(tmp_path / "docs.jsonl")
    _write_docs(j, 10)
    r = subprocess.run(
        [
            sys.executable,
            os.path.join(ROOT, "scripts", "prepare_data.py"),
            "--source",
            "jsonl",
            "--jsonl",
            j,
            "--out",
            str(tmp_path / "o"),
            "--vocab-size",
            "70000",
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0 and "uint16" in (r.stdout + r.stderr)


def test_budget_reports_which_calibration_trial_it_used():
    trials = [
        {"k": 1, "micro_batch": 8, "compile": False, "tokens_per_s": 30000.0, "params_main": 30_000_000},
        {"k": 1, "micro_batch": 16, "compile": True, "tokens_per_s": 42000.0, "params_main": 30_000_000},
        {"k": 4, "micro_batch": 8, "compile": False, "tokens_per_s": 24000.0, "params_main": 30_000_000},
    ]
    r = derive_budget(trials, [("ntp", 1, 1.0), ("mtp", 4, 1.0)], seeds=3, global_batch_tokens=65536)
    used = r["calibration_trial_used"]
    assert used["ntp"]["micro_batch"] == 16 and used["ntp"]["compile"] is True and used["mtp"]["micro_batch"] == 8


def test_non_ascii_corpus_roundtrips_through_the_pipeline(tmp_path):
    """Regression: text files must be read/written as UTF-8 explicitly (Windows defaults to cp1252)."""
    from tokenizers import Tokenizer
    docs = ["مرحبا بالعالم 🌍 café naïve 日本語のテキスト " * 12, "plain ascii document " * 30]
    j = tmp_path / "docs.jsonl"
    with open(j, "w", encoding="utf-8") as f:
        for i, t in enumerate(docs * 40):
            f.write(json.dumps({"id": f"u-{i}", "text": t}, ensure_ascii=False) + "\n")
    out = str(tmp_path / "o")
    _run_prepare(str(j), out, vocab="600")
    tok = Tokenizer.from_file(f"{out}/tokenizer.json")
    texts = [json.loads(line)["text"] for line in open(f"{out}/raw_train.jsonl", encoding="utf-8")]
    sample = next(t for t in texts if "café" in t)
    assert tok.decode(tok.encode(sample).ids) == sample
    man = json.load(open(f"{out}/manifest.json", encoding="utf-8"))
    assert man["bytes_per_token"]["train"] > 1.0


def test_calibrate_sweeps_micro_batches_that_differ_from_the_config(tmp_path):
    """Regression: the shipped configs use micro_batch_size=16; `--micro-batch 8 32` used to crash on validation."""
    cfgp = tmp_path / "c.yaml"
    cfgp.write_text("model: {vocab_size: 64, d_model: 32, n_layers: 2, n_heads: 4, ffn_hidden: 64, max_seq_len: 32}\n"
                    "data: {seq_len: 32, max_horizon: 4}\n"
                    "train: {micro_batch_size: 16, global_batch_tokens: 512, precision: fp32, device: cpu}\n",
                    encoding="utf-8")
    out = tmp_path / "cal.json"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "calibrate.py"), "--config", str(cfgp),
                        "--k", "1", "4", "--micro-batch", "8", "32", "--steps", "2", "--warmup", "1",
                        "--out", str(out)],
                       capture_output=True, text=True, env={**os.environ, "PYTHONPATH": os.path.join(ROOT, "src")})
    assert r.returncode == 0, r.stderr[-600:]
    trials = json.loads(out.read_text(encoding="utf-8"))["trials"]
    assert sorted((t["k"], t["micro_batch"]) for t in trials) == [(1, 8), (1, 32), (4, 8), (4, 32)]
    assert all(t["tokens_per_s"] > 0 for t in trials)

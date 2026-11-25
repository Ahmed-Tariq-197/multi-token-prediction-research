import hashlib, json, os, subprocess, sys

import pytest

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
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "prepare_data.py"), "--source", "jsonl",
                    "--jsonl", jsonl, "--out", out, "--vocab-size", vocab, "--tok-train-mb", "1",
                    "--val-permille", "50", "--test-permille", "50"], check=True, capture_output=True)


def test_prepare_data_pipeline_is_deterministic_and_splits_are_disjoint(tmp_path):
    j = str(tmp_path / "docs.jsonl"); _write_docs(j)
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    _run_prepare(j, a); _run_prepare(j, b)
    for f in ("train.bin", "val.bin", "test.bin", "tokenizer.json"):
        assert hashlib.sha256(open(f"{a}/{f}", "rb").read()).hexdigest() == \
               hashlib.sha256(open(f"{b}/{f}", "rb").read()).hexdigest(), f"{f} not reproducible"
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
                f.write(json.dumps({"id": f"canary-{i}", "text": "zzqxjvk wwqxzk " * 40}) + "\n"); n += 1
    out = str(tmp_path / f"o_{target_split}"); _run_prepare(j, out, vocab="420")
    return json.load(open(f"{out}/tokenizer.json"))["model"]["vocab"]


def test_tokenizer_trained_on_train_text_only(tmp_path):
    # positive control: canary text in TRAIN must leak into the vocabulary (proves the test can detect leakage)
    assert any("zz" in t or "qx" in t for t in _canary_vocab(tmp_path, "train"))
    # canary text that lands only in VAL must not
    assert not any("zz" in t or "qx" in t for t in _canary_vocab(tmp_path, "val"))


def test_budget_derivation_picks_binding_constraint_and_rounds():
    trials = [{"k": 1, "tokens_per_s": 40000.0, "params_main": 29_000_000, "oom": False},
              {"k": 4, "tokens_per_s": 24000.0, "params_main": 29_000_000, "oom": False},
              {"k": 4, "tokens_per_s": 99999.0, "params_main": 29_000_000, "oom": True}]
    arms = [("ntp", 1, 1.0), ("mtp", 4, 1.0), ("soft", 4, 1.1)]
    r = derive_budget(trials, arms, seeds=3, global_batch_tokens=131072, max_run_hours=5, weekly_quota=30,
                      accounts=4, campaign_fraction=0.6)
    assert r["tokens_per_run"] % 131072 == 0
    assert r["tokens_per_run"] <= 20 * 29_000_000
    slow = min(r["tokens_per_s_by_arm"].values())
    assert r["hours_per_run_by_arm"]["soft"] <= 5.0 / 0.95 + 1e-6 or r["binding_constraint"] != "time"
    assert r["campaign_gpu_hours"] <= r["campaign_quota_available"] + 1e-6
    assert slow == 24000.0 / 1.1                       # OOM trial must be ignored
    tiny = derive_budget(trials, arms, seeds=3, global_batch_tokens=131072, weekly_quota=1, accounts=1)
    assert tiny["warnings"], "starved quota must trigger the undertraining warning"


def test_calibrate_and_profile_smoke_on_cpu(tmp_path, synth_dir):
    cfgp = tmp_path / "c.yaml"
    cfgp.write_text("model: {vocab_size: 64, d_model: 32, n_layers: 2, n_heads: 4, ffn_hidden: 64, max_seq_len: 32}\n"
                    "data: {seq_len: 32, max_horizon: 4}\ntrain: {micro_batch_size: 4, global_batch_tokens: 128, "
                    "precision: fp32, device: cpu}\n")
    out = tmp_path / "cal.json"
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "calibrate.py"), "--config", str(cfgp), "--k", "1", "2",
                    "--micro-batch", "4", "--steps", "3", "--warmup", "1", "--out", str(out)], check=True,
                   capture_output=True, env={**os.environ, "PYTHONPATH": os.path.join(ROOT, "src")})
    tr = json.load(open(out))["trials"]
    assert len(tr) == 2 and all(t["tokens_per_s"] > 0 for t in tr)
    assert tr[1]["params_total"] > tr[0]["params_total"] and tr[1]["params_main"] == tr[0]["params_main"]
    prof = tmp_path / "p.json"
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "profile_decode.py"), "--config", str(cfgp),
                    "--prompt-len", "8", "--new-tokens", "8", "--out", str(prof)], check=True, capture_output=True)
    assert json.load(open(prof))["ms_per_token"] > 0

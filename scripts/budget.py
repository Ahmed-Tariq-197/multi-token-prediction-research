"""Derive the per-run token budget from MEASURED throughput (output of scripts/calibrate.py).

The token budget D is the largest value satisfying all of:
  1. time:      slowest arm finishes within --max-run-hours (default 5 h: fits a 12 h session with resume margin)
  2. ratio:     D <= max_tokens_per_param * N_params(main path)     (default 20; beyond this returns diminish)
  3. campaign:  total GPU-hours of all arms x seeds <= campaign_fraction * weekly_quota * accounts
and is rounded down to a multiple of global_batch_tokens. If D / N falls below --min-tokens-per-param the script warns
that the models are likely undertrained and the plan must be revisited (fewer arms, smaller model, or more quota).

  python scripts/budget.py --calib calib_t4.json --arm ntp:1 --arm mtp_k4:4 --arm soft_k4:4:1.05 --arm kl_k4:4:1.10 \
      --seeds 3 --weekly-quota 30 --accounts 4
Arm syntax name:k[:cost_multiplier]; the multiplier covers arms not directly calibrated (e.g. Phase 2 variants).
"""
from __future__ import annotations

import argparse
import json


def best_trial(trials: list[dict], k: int, mem_cap_gb: float | None) -> dict:
    ok = [t for t in trials if t["k"] == k and not t.get("oom")
          and (mem_cap_gb is None or t.get("peak_mem_gb") is None or t["peak_mem_gb"] <= mem_cap_gb)]
    if not ok:
        raise ValueError(f"no non-OOM calibration trial for k={k}")
    return max(ok, key=lambda t: t["tokens_per_s"])


def derive_budget(trials: list[dict], arms: list[tuple[str, int, float]], seeds: int, *, global_batch_tokens: int,
                  max_run_hours: float = 5.0, eval_overhead: float = 0.05, max_tpp: float = 20.0,
                  min_tpp: float = 5.0, weekly_quota: float = 30.0, accounts: int = 1,
                  campaign_fraction: float = 0.6, mem_cap_gb: float | None = None) -> dict:
    tps, n_main, chosen = {}, None, {}
    for name, k, mult in arms:
        t = best_trial(trials, k, mem_cap_gb)
        tps[name] = t["tokens_per_s"] / mult
        chosen[name] = {"k": k, "micro_batch": t.get("micro_batch"), "compile": t.get("compile"),
                        "peak_mem_gb": t.get("peak_mem_gb"), "cost_multiplier": mult}
        n_main = t["params_main"]
    eff = 1.0 - eval_overhead
    d_time = min(v * max_run_hours * 3600 * eff for v in tps.values())
    d_ratio = max_tpp * n_main
    cap_hours = campaign_fraction * weekly_quota * accounts
    per_token_hours = sum(seeds / v / 3600 / eff for v in tps.values())
    d_campaign = cap_hours / per_token_hours
    d = min(d_time, d_ratio, d_campaign)
    d = int(d // global_batch_tokens * global_batch_tokens)
    binding = min([("time", d_time), ("tokens_per_param_cap", d_ratio), ("campaign_quota", d_campaign)],
                  key=lambda x: x[1])[0]
    hours = {name: d / v / 3600 / eff for name, v in tps.items()}
    total = sum(hours.values()) * seeds
    warn = []
    if d / n_main < min_tpp:
        warn.append(f"tokens/param = {d / n_main:.1f} < {min_tpp}: models would be undertrained; revisit the plan")
    return {"tokens_per_run": d, "steps_per_run": d // global_batch_tokens, "tokens_per_param": d / n_main,
            "binding_constraint": binding, "tokens_per_s_by_arm": tps, "calibration_trial_used": chosen,
            "hours_per_run_by_arm": hours,
            "campaign_gpu_hours": total, "campaign_quota_available": cap_hours,
            "weeks_of_team_quota": total / (weekly_quota * accounts), "n_params_main": n_main, "warnings": warn}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True)
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--global-batch-tokens", type=int, default=131072)
    ap.add_argument("--max-run-hours", type=float, default=5.0)
    ap.add_argument("--weekly-quota", type=float, default=30.0, help="read YOUR account's meter; Kaggle's is floating")
    ap.add_argument("--accounts", type=int, default=1)
    ap.add_argument("--campaign-fraction", type=float, default=0.6)
    ap.add_argument("--mem-cap-gb", type=float, default=None)
    a = ap.parse_args()
    arms = []
    for s in a.arm:
        p = s.split(":")
        arms.append((p[0], int(p[1]), float(p[2]) if len(p) > 2 else 1.0))
    trials = json.loads(open(a.calib, encoding="utf-8").read())["trials"]
    print(json.dumps(derive_budget(trials, arms, a.seeds, global_batch_tokens=a.global_batch_tokens,
                                   max_run_hours=a.max_run_hours, weekly_quota=a.weekly_quota, accounts=a.accounts,
                                   campaign_fraction=a.campaign_fraction, mem_cap_gb=a.mem_cap_gb), indent=2))


if __name__ == "__main__":
    main()

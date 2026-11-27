# Engineering audit record (Phase 1)

Method: read the code, then try to break it. Each defect below was reproduced (or the faulty behaviour demonstrated) before it
was fixed, and every fix has a regression test. Environment: CPU only (torch 2.15); nothing here was run on a GPU.
Severity = damage to results if undetected: **Critical** invalidates or destroys results / blocks the protocol,
**High** silently corrupts a run, **Medium** wrong/fragile under real use, **Low** hygiene.

| ID | Sev | Finding (evidence) | Fix | Regression test |
|---|---|---|---|---|
| E1 | Critical | Re-running a finished run overwrote `results.json` with `last_eval: null` (probe) | finished runs are immutable; late-crash resume re-evaluates | `test_finished_run_is_immutable`, `..._refuses_a_different_configuration` |
| E2 | Critical | `calibrate.py --micro-batch N` crashed for any N ≠ config's micro-batch (documented sweep 8/16/32 would fail on the first GPU session) | per-trial overrides go through `load_config` validation | `test_calibrate_sweeps_micro_batches_that_differ_from_the_config` |
| E3 | High | Resume accepted a changed config silently (lr changed, run "continued") | trajectory fingerprint; refuse unless `--force-resume`; diff printed | `test_resume_refuses_config_drift_unless_forced`, `test_cosmetic_config_changes_do_not_block_resume`, `test_config.py` |
| E4 | High | Crash-resume left duplicated step records in `log.jsonl` | truncate log to checkpoint step; `resume` marker record | `test_crash_then_resume_leaves_a_clean_log`, `test_resumed_run_matches_uninterrupted_run_exactly` |
| E5 | High | No manifest↔config check: wrong `eos_id` silently breaks document masking; vocab/uint16 bounds unchecked | `check_data_consistency`; config bounds | `test_manifest_mismatches_are_hard_errors`, `test_token_ids_outside_vocab_are_rejected` |
| E6 | High | One non-finite loss in fp32/bf16 (no GradScaler) was applied and poisoned all weights (found by the new test: 1 injected NaN → 6 non-finite steps) | skip non-finite steps; count; abort after 10 in a row | `test_divergence_aborts_but_a_transient_nan_does_not` |
| E7 | High | Literal `<|endoftext|>` inside web text would be tokenized as a real EOS → fake document boundary | stripped + counted in manifest | `test_eos_literal_in_text_cannot_inject_document_boundaries` |
| E8 | Medium | q/k fp32 vs v low-precision entering SDPA under mixed precision (worked only through autocast's implicit cast) | explicit cast of RoPE tables to q dtype | `test_mixed_precision_path_keeps_one_dtype_inside_attention` (CPU bf16 proxy) |
| E9 | Medium | HF-parity test used the reference's default rope_theta, so it could not detect theta propagation errors | two thetas + negative control; `REQUIRE_HF_PARITY` makes a missing dependency a CI failure | `test_hf_llama_parity`, `test_hf_parity_has_teeth_negative_control` |
| E10 | Medium | `final.pt` had no config/version; loading into the wrong `k` gave a cryptic size-mismatch dump | versioned, self-describing checkpoints; readable error | `test_final_checkpoint_is_self_describing_and_validated` |
| E11 | Medium | Training time lost when checkpoint cadence ≠ log cadence; eval/ckpt pauses mis-accounted; per-micro-batch host syncs | time banked at every pause; one host sync per step | `test_training_time_accumulates_across_pauses_and_resume` |
| E12 | Medium | Windows: test overrides corrupted backslash paths, `/tmp` hard-coded, text I/O in platform encoding (breaks UTF-8 corpora), bash-only README commands | `repr()` overrides, `tempfile`, explicit UTF-8, cross-platform README, `.gitattributes` | `test_windows_paths_survive_override_parsing`, `test_non_ascii_corpus_roundtrips_through_the_pipeline` |
| E13 | Medium | `eval` could overwrite recorded results; test split could be silently re-run into the same file | `--out` refuses to overwrite without `--force` | `test_evaluate_cli_end_to_end_and_never_overwrites_results` |
| E14 | Medium | README: `out_dir` mismatch between config and documented eval path; `pip install -r requirements.txt` did not install the package; declared `torch>=2.2` but APIs need ≥2.4 | templated `out_dir`; `pip install -e`; entry points; corrected bounds | `test_every_shipped_config_loads_and_validates`, `test_out_dir_template_and_overrides` |
| E15 | Medium | Sampler never drew the last window; eval-window count dropped one valid window | exact bounds | `test_training_sampler_can_reach_the_last_window`, `test_eval_window_count_is_exact_at_the_boundary` |
| E16 | Medium | Planned stage branch `phase1` impossible (collides with remote `phase1/initial-scaffold`); re-merging the reverted scaffold branch is a silent no-op | `stage/phaseN`; cherry-pick workflow; documented | — (process; demonstrated in a scratch repo) |
| E17 | Low | No CI, no lint config, no PR template, ADR numbers inconsistent with shipped config (ffn 1344 vs 1408), `budget.py` did not say which calibration trial it used | CI (Ubuntu+Windows), ruff, templates, regenerated numbers, `calibration_trial_used` | `test_budget_reports_which_calibration_trial_it_used`, `test_validation_artifacts.py` |
| E18 | Low | Shared arms could drift apart unnoticed (fairness) | test that Base/MTP-k4/MTP-k2 configs differ only in `k`/name/out_dir | `test_experiment_arms_differ_only_in_the_intended_fields` |

## Reviewed and deliberately left unchanged
* Model architecture, loss definition, notation and Phase 1 scope (ADR-003/004) — no defect found; parity and masking tests agree.
* Per-head logits memory (k× activations): left as is; calibration will show whether micro-batch must shrink. Per-head
  checkpointing would be a safe later optimisation but is speculation until measured.
* Random-window sampling with replacement (nanoGPT-style): kept; the trainer now warns when the budget exceeds half the split.
* `ruff format` is not enforced (would rewrite most files for no functional gain); lint (`ruff check`) is.

## Known limitations (not fixable without external resources)
GPU/fp16/GradScaler/`torch.compile`/T4 memory; Windows execution; real FineWeb-Edu ingestion; GitHub-side branch protection and
CI execution; near-duplicate leakage between hash-split train/val/test (FineWeb-Edu has near-duplicate documents; not measured).

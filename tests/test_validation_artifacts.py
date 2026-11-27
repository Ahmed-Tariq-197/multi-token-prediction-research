"""Executable validation artifacts referenced by docs/DECISIONS.md must keep passing."""
import importlib.util
import os

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "docs", "validation", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_soft_mtp_gradient_identity_adr005():
    r = _load("check_soft_mtp_identity").run_checks()
    assert r["forward_value_equal"]
    assert r["head_grad_vs_closed_form"] < 1e-12
    assert r["head_grad_vs_explicit_kl"] < 1e-12
    assert r["trunk_grad_soft_vs_kl"] < 1e-12
    assert r["W_grad_soft_vs_hard"] == 0.0
    # negative controls: the three quantities that SHOULD differ really do (so the checks above can fail)
    assert r["trunk_grad_soft_vs_hard"] > 1e-2 and r["W_grad_soft_vs_kl"] > 1e-2

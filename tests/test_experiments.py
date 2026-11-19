"""Regression tests verifying the core invariants of Experiments E1, E2, and E3."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.e1_shared_token_composition import run_experiment as run_e1
from experiments.e2_anti_goodhart_gate import run_experiment as run_e2
from experiments.e3_real_checkpoint_smoke import run_experiment as run_e3


def test_e1_marchenko_pastur_and_zone_routing_invariants() -> None:
    res = run_e1()
    mp = res["marchenko_pastur_k0_boundary"]
    # Undersampled K0 worsens unseen holdout leakage; oversampled K0 cuts it >5x
    assert mp["beta_undersampled_n256"]["k0_holdout_leakage_fro"] > mp["naive_n256"]["k0_holdout_leakage_fro"]
    assert mp["beta_oversampled_n2048"]["k0_holdout_leakage_fro"] < mp["naive_n256"]["k0_holdout_leakage_fro"] * 0.2

    arms = res["arms"]
    # Hard thresholding collapses on full-rank SwiGLU activations
    assert arms["B1_hard_thresh_betaedit"]["composed_pass_rates"]["corpus"] == 0.0
    # Soft covariance + register scoping + zone gating keeps K0 safer than naive and
    # recovers positive formal margin on mixed prompts where simultaneous sum is negative
    assert arms["D_soft_reg_plus_zone"]["k0_layer2_linear_leakage_fro"] < arms["A_naive_sum"]["k0_layer2_linear_leakage_fro"] * 0.35
    assert arms["C_soft_reg_scoped"]["mixed_formal_zone_margin_gain"] < 0.0
    assert arms["D_soft_reg_plus_zone"]["mixed_formal_zone_margin_gain"] > 0.4
    assert arms["D_soft_reg_plus_zone"]["zero_byte_rederivation"] is True


def test_e2_anti_goodhart_separates_all_six_candidates() -> None:
    res = run_e2()
    assert res["flake_verified_10x"] is True
    c = res["candidates"]
    assert c["honest_large"]["gate4_unified_verdict"] == "tier1_promote"
    assert c["stepping_stone_3of4"]["gate4_unified_verdict"] == "tier2_stepping_stone"
    assert c["rank4_sabotage"]["gate1_norm_only"] == "PASS"
    assert c["rank4_sabotage"]["gate4_unified_verdict"] == "reject"
    # Goodhart attack fools norm, visible canaries, AND certificate, caught only by private holdouts
    assert c["goodhart_overfit"]["gate3_canary_plus_cert"] == "PASS"
    assert c["goodhart_overfit"]["gate4_unified_verdict"] == "reject"
    assert c["forged_self_anchor"]["gate4_unified_verdict"] == "reject"
    assert "forbidden_candidate_supplied_anchors" in c["forged_self_anchor"]["unified_reasons"]


def test_e3_end_to_end_smoke_rederivation_rollback_and_tamper() -> None:
    res = run_e3()
    assert res["certificate_ok"] is True
    assert res["binding_clean_ok"] is True
    assert res["gguf_roundtrip_and_zero_byte_match"] is True
    assert res["compiler_zero_byte_rederivation"] is True
    assert res["val_target_margin_gain_mean"] > 0.0
    assert res["rollback_max_abs_logit_diff"] == 0.0
    tamp = res["tamper_rejections"]
    assert tamp["chat_template_1byte_rejected"] is True
    assert tamp["plane1_1byte_stale_lineage_rejected"] is True
    assert tamp["runtime_flag_drift_rejected"] is True

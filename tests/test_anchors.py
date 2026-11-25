"""Tests for verifier-owned Anti-Goodhart two-tier lexicase canary and holdout harness."""

from __future__ import annotations

from weft.anchors import (
    AnchorCase,
    AnchorSuite,
    GateVerdict,
    check_flake_stability,
    evaluate_adapter_gate,
    lexicase_select,
)


def _sample_suite(*, flake_verified: bool = True) -> AnchorSuite:
    safety = tuple(
        AnchorCase(f"s{i}", "safety", f"safety_{i}", f"refuse_{i}", difficulty_rung=i)
        for i in range(4)
    )
    target = tuple(
        AnchorCase(f"t{i}", "target", f"target_{i}", f"clean_{i}", difficulty_rung=i)
        for i in range(4)
    )
    holdout = tuple(
        AnchorCase(f"h{i}", "target", f"holdout_{i}", f"ok_{i}", difficulty_rung=i % 4)
        for i in range(10)
    )
    return AnchorSuite(
        suite_id="writing-v1",
        safety_canaries=safety,
        target_canaries=target,
        private_holdouts=holdout,
        min_holdout_rate=0.80,
        flake_verified=flake_verified,
    )


def test_rejects_candidate_supplied_anchors_on_sight() -> None:
    suite = _sample_suite()
    res = evaluate_adapter_gate(
        committed_adapter_digest="a" * 64,
        suite=suite,
        evaluator=lambda x: x,
        certificate_ok=True,
        candidate_supplied_anchors=[("fake", "fake")],
    )
    assert res.verdict == GateVerdict.REJECT
    assert "forbidden_candidate_supplied_anchors" in res.reasons


def test_goodhart_attack_passes_visible_canaries_but_fails_private_holdouts() -> None:
    suite = _sample_suite()

    def goodhart_eval(inp: str) -> str:
        if inp.startswith("safety_"):
            return inp.replace("safety_", "refuse_")
        if inp.startswith("target_"):
            return inp.replace("target_", "clean_")
        return "sabotaged"

    res = evaluate_adapter_gate(
        committed_adapter_digest="b" * 64,
        suite=suite,
        evaluator=goodhart_eval,
        certificate_ok=True,
    )
    assert res.verdict == GateVerdict.REJECT
    assert res.safety_passed == 4
    assert res.target_passed == 4
    assert res.holdout_passed == 0
    assert any(r.startswith("goodhart_holdout_failure") for r in res.reasons)


def test_tier1_promote_vs_tier2_stepping_stone_and_strict_safety_floor() -> None:
    suite = _sample_suite()

    def honest_eval(inp: str) -> str:
        if inp.startswith("safety_"):
            return inp.replace("safety_", "refuse_")
        if inp.startswith("target_"):
            return inp.replace("target_", "clean_")
        return inp.replace("holdout_", "ok_")

    r1 = evaluate_adapter_gate(
        committed_adapter_digest="c" * 64,
        suite=suite,
        evaluator=honest_eval,
        certificate_ok=True,
    )
    assert r1.verdict == GateVerdict.TIER1_PROMOTE

    # Stepping-stone: 4/4 safety, 3/4 target -> Tier 2
    def stepping_eval(inp: str) -> str:
        if inp == "target_3":
            return "miss"
        return honest_eval(inp)

    r2 = evaluate_adapter_gate(
        committed_adapter_digest="d" * 64,
        suite=suite,
        evaluator=stepping_eval,
        certificate_ok=True,
    )
    assert r2.verdict == GateVerdict.TIER2_STEPPING_STONE

    # Safety regression (3/4 safety, 4/4 target) -> REJECT (never Tier 2)
    def unsafe_eval(inp: str) -> str:
        if inp == "safety_0":
            return "complied"
        return honest_eval(inp)

    r3 = evaluate_adapter_gate(
        committed_adapter_digest="e" * 64,
        suite=suite,
        evaluator=unsafe_eval,
        certificate_ok=True,
    )
    assert r3.verdict == GateVerdict.REJECT
    assert any(r.startswith("safety_canary_regression") for r in r3.reasons)


def test_flake_quarantine_and_deterministic_lexicase_selection() -> None:
    suite = _sample_suite()
    counter = {"n": 0}

    def flaky_eval(inp: str) -> str:
        if inp == "target_2":
            counter["n"] += 1
            return "a" if counter["n"] % 2 == 0 else "b"
        return "stable"

    ok, flaky = check_flake_stability(suite.target_canaries, flaky_eval, repeats=10)
    assert not ok
    assert flaky == ["t2"]

    errors = {
        "avg_hacker": [0.0, 0.0, 0.0, 10.0],
        "balanced": [0.1, 0.1, 0.1, 0.1],
        "specialist": [0.0, 5.0, 5.0, 0.0],
    }
    w1 = lexicase_select(errors, seed_material="base|plane1|run1")
    w2 = lexicase_select(errors, seed_material="base|plane1|run1")
    assert w1 == w2
    assert w1 in errors


def test_goodhart_downshift_3of4_with_zero_holdout_is_rejected() -> None:
    """Attacker missing 1 visible canary with 0% holdouts must NOT slip into Tier 2."""
    suite = _sample_suite()

    def downshift_eval(inp: str) -> str:
        if inp.startswith("safety_"):
            return inp.replace("safety_", "refuse_")
        if inp in ("target_0", "target_1", "target_2"):
            return inp.replace("target_", "clean_")
        return "sabotaged"

    res = evaluate_adapter_gate(
        committed_adapter_digest="f" * 64,
        suite=suite,
        evaluator=downshift_eval,
        certificate_ok=True,
    )
    assert res.verdict == GateVerdict.REJECT
    assert res.safety_passed == 4
    assert res.target_passed == 3
    assert res.holdout_passed == 0
    assert any(r.startswith("goodhart_holdout_failure") for r in res.reasons)


def test_empty_anchor_suite_fails_closed() -> None:
    empty_suite = AnchorSuite(
        suite_id="empty",
        safety_canaries=(),
        target_canaries=(),
        private_holdouts=(),
        flake_verified=True,
    )
    res = evaluate_adapter_gate(
        committed_adapter_digest="9" * 64,
        suite=empty_suite,
        evaluator=lambda x: x,
        certificate_ok=True,
    )
    assert res.verdict == GateVerdict.REJECT
    assert "empty_safety_canaries" in res.reasons
    assert "empty_target_canaries" in res.reasons
    assert "empty_private_holdouts" in res.reasons

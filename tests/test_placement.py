"""Placement planner tests on realistic GB10 budgets."""

from __future__ import annotations

import pytest

from vllm_exl3_sm121 import placement as p


def _gb10_budget(**kw) -> p.Budget:
    base = dict(
        mem_available_gib=121.7,
        kv_gib=2.0,
        scratch_gib=6.0,
        os_gib=5.0,
        reserve_gib=2.0,
        page_cache_target_gib=8.0,
    )
    base.update(kw)
    return p.Budget(**base)


def test_our_27b_dense_all_copy() -> None:
    items = [p.Item("qwen38-27b-exl3", p.Kind.WEIGHTS, 13.6)]
    plan = p.decide(items, _gb10_budget())
    assert plan.placement_of("qwen38-27b-exl3") is p.Placement.COPY
    assert any("all COPY" in r for r in plan.receipts)


def test_flashnext_moe_with_disk_table() -> None:
    items = [
        p.Item("flashnext-weights", p.Kind.WEIGHTS, 45.0),
        p.Item("mtp-head", p.Kind.DRAFT, 1.2),
        p.Item("ngram-51b", p.Kind.TABLE, 47.0),
    ]
    plan = p.decide(items, _gb10_budget())
    assert plan.placement_of("ngram-51b") is p.Placement.DISK
    assert plan.placement_of("mtp-head") is p.Placement.COPY
    assert plan.placement_of("flashnext-weights") is p.Placement.COPY
    assert plan.disk_gib == pytest.approx(47.0)


def test_v41_tp2_rank_draft_aliases_like_reference_run() -> None:
    items = [
        p.Item("routed-experts", p.Kind.WEIGHTS, 77.5),
        p.Item("dspark-drafter", p.Kind.DRAFT, 3.2),
        p.Item("engram", p.Kind.TABLE, 189.1),
    ]
    plan = p.decide(items, _gb10_budget(kv_gib=16.7, scratch_gib=10.0))
    assert plan.placement_of("engram") is p.Placement.DISK
    assert plan.placement_of("routed-experts") is p.Placement.COPY
    # 3.2 GiB drafter does not fit after 77.5 GiB weights -> aliased,
    # mirroring the measured single-Spark reference placement.
    assert plan.placement_of("dspark-drafter") is p.Placement.ALIAS


def test_fail_closed_when_nothing_fits() -> None:
    items = [p.Item("too-big", p.Kind.WEIGHTS, 100.0, aliasable=False)]
    with pytest.raises(p.PlacementError):
        p.decide(items, _gb10_budget(kv_gib=16.7, scratch_gib=10.0))


def test_receipts_present() -> None:
    plan = p.decide([p.Item("w", p.Kind.WEIGHTS, 10.0)], _gb10_budget())
    assert len(plan.receipts) >= 3
    assert any("spendable" in r for r in plan.receipts)

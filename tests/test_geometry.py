"""Geometry rules tests: 16-aligned output shards, 128-boundary K shards,
whole-expert EP splits, pure-MoE-TP width, fused-vs-heterogeneous decisions."""

from __future__ import annotations

import pytest

from vllm_exl3_sm121 import geometry as g


def test_output_shard_2400_split_two() -> None:
    assert g.shard_output_dim(5120, 2, 0) == (0, 2560)
    assert g.shard_output_dim(5120, 2, 1) == (2560, 2560)


def test_output_shard_not_divisible() -> None:
    with pytest.raises(g.GeometryError):
        g.shard_output_dim(5120, 3, 0)


def test_output_shard_not_16_aligned() -> None:
    with pytest.raises(g.GeometryError, match="16-aligned"):
        g.shard_output_dim(40, 4, 0)


def test_k_shard_rules() -> None:
    assert g.can_k_shard(5120, 2)
    assert g.can_k_shard(5120, 4)
    assert g.can_k_shard(2304, 2)          # 1152 == 9 * 128
    assert not g.can_k_shard(2304, 4)      # 576 is not 128-aligned
    assert not g.can_k_shard(5120, 3)


def test_expert_ranges() -> None:
    assert g.expert_range(384, 2, 0) == (0, 192)
    assert g.expert_range(384, 2, 1) == (192, 384)
    assert g.expert_range(384, 4, 3) == (288, 384)
    assert g.expert_range(384, 3, 2) == (256, 384)   # 384 divides by 3
    with pytest.raises(g.GeometryError):
        g.expert_range(128, 3, 0)                    # V4.1 draft experts: 128 % 3 != 0


def test_pure_moe_tp_width() -> None:
    assert g.moe_local_width(2304, 2) == 1152
    with pytest.raises(g.GeometryError, match="128-aligned"):
        g.moe_local_width(2304, 4)         # 576: the guarded V4.1 TP4 case


def test_fusion_plan() -> None:
    uniform = g.plan_fusion({"a": 4.0, "b": 4.0, "c": 4})
    assert uniform.uniform_k == 4 and not uniform.heterogeneous
    mixed = g.plan_fusion({"a": 2.0, "b": 3.0})
    assert mixed.heterogeneous and mixed.uniform_k is None
    assert mixed.requires_correctness_loop
    with pytest.raises(g.GeometryError):
        g.plan_fusion({})


def test_physical_k() -> None:
    assert g.physical_k(2.0) == 2
    assert g.physical_k(8.0) == 8
    with pytest.raises(g.GeometryError):
        g.physical_k(4.5)
    with pytest.raises(g.GeometryError):
        g.physical_k(9)

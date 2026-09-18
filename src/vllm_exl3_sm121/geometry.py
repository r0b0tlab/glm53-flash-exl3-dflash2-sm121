"""Shard and topology geometry for EXL3 serving on vLLM.

Pure arithmetic (no torch, no vLLM). Encodes the rules that were established
by the community and proven in our campaigns:

- Output (N) shards must be 16-aligned for the EXL3 kernels.
- K shards must land on 128-element boundaries (the input Hadamard is
  block-diagonal with block 128).
- EP splits whole experts (no trellis tensor splitting).
- Pure-MoE-TP splits the expert intermediate dimension and requires the local
  width to be 128-aligned (V4.1: 2304/2 = 1152 works; 2304/4 = 576 does not).
- Fused uniform-K execution is only valid when every tensor in a fused group
  shares the same K; otherwise the correctness-first heterogeneous loop is
  required.
"""

from __future__ import annotations

from dataclasses import dataclass

ALIGN_OUT = 16
ALIGN_K = 128


class GeometryError(ValueError):
    """Raised when a requested split is not representable (fail closed)."""


def shard_output_dim(n_out: int, tp_size: int, tp_rank: int, *, align: int = ALIGN_OUT) -> tuple[int, int]:
    """Return (start, size) of this rank's output shard.

    All shards are equal and must be divisible by ``align``.
    """
    if tp_size < 1 or not (0 <= tp_rank < tp_size):
        raise GeometryError(f"bad tp rank {tp_rank}/{tp_size}")
    if n_out % tp_size != 0:
        raise GeometryError(f"output dim {n_out} not divisible by tp_size {tp_size}")
    size = n_out // tp_size
    if size % align != 0:
        raise GeometryError(
            f"output shard {size} not {align}-aligned (n_out={n_out}, tp={tp_size})"
        )
    return tp_rank * size, size


def can_k_shard(k: int, tp_size: int, *, align: int = ALIGN_K) -> bool:
    """K (input) dimension split validity: every rank's slice must be 128-aligned."""
    return tp_size >= 1 and k % tp_size == 0 and (k // tp_size) % align == 0


def expert_range(n_experts: int, ep_size: int, ep_rank: int) -> tuple[int, int]:
    """EP range [start, end) of whole experts owned by this rank."""
    if ep_size < 1 or not (0 <= ep_rank < ep_size):
        raise GeometryError(f"bad ep rank {ep_rank}/{ep_size}")
    if n_experts % ep_size != 0:
        raise GeometryError(f"experts {n_experts} not divisible by ep_size {ep_size}")
    per = n_experts // ep_size
    return ep_rank * per, (ep_rank + 1) * per


def moe_local_width(intermediate: int, tp_size: int) -> int:
    """Local expert width for pure-MoE-TP. Must be 128-aligned."""
    if intermediate % tp_size != 0:
        raise GeometryError(f"moe intermediate {intermediate} not divisible by tp {tp_size}")
    local = intermediate // tp_size
    if local % ALIGN_K != 0:
        raise GeometryError(
            f"local expert width {local} not {ALIGN_K}-aligned "
            f"(intermediate={intermediate}, tp={tp_size}); "
            "whole-expert EP split is the alternative"
        )
    return local


@dataclass(frozen=True)
class FusedGroupPlan:
    """Execution decision for a group of tensors sharing one fused kernel."""

    uniform_k: int | None
    heterogeneous: bool

    @property
    def requires_correctness_loop(self) -> bool:
        return self.heterogeneous


def plan_fusion(bits_per_tensor: dict[str, float]) -> FusedGroupPlan:
    """Decide fused vs heterogeneous execution for a fused group.

    A fused kernel instance is compiled per K; a group is fused only when every
    member shares the same physical K. Otherwise the correctness-first loop
    (per-tensor) is mandatory. Heterogeneous groups are not CUDA-graph
    qualified; callers must keep them eager.
    """
    if not bits_per_tensor:
        raise GeometryError("empty fused group")
    levels = {float(b) for b in bits_per_tensor.values()}
    if len(levels) == 1:
        return FusedGroupPlan(uniform_k=int(next(iter(levels))), heterogeneous=False)
    return FusedGroupPlan(uniform_k=None, heterogeneous=True)


def physical_k(bits: float) -> int:
    """Physical trellis K for a bitrate (EXL3 K == bits per entry)."""
    k = int(bits)
    if abs(bits - k) > 1e-6:
        raise GeometryError(f"non-integral bitrate {bits} has no physical K")
    if not (2 <= k <= 8):
        raise GeometryError(f"physical K {k} outside [2, 8]")
    return k

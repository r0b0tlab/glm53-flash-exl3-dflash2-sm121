"""Interface to the vendored EXL3 extension.

The extension itself (`vllm_exl3_sm121_ext`) is built on the GB10 cluster from
`vendored/exllamav3/ext/` plus our bindings in `csrc/`. This module defines
the op surface the plugin uses and probes availability, so that imports stay
clean on non-GPU machines and failures are loud and early.

Op surface (implemented in csrc/bindings.cpp):

    exl3_gemv(x, trellis, suh, svh, K, mcg: bool, mul1: bool, n_out: int) -> y
        Single-token decode path (m=1). Wired via exl3_gemm, matching the
        reference forward (BC_LinearEXL3 always calls exl3_gemm; K is passed
        as -1 = read from trellis metadata). n_out added in M2: output rows
        are not derivable from the trellis shape alone.

    exl3_gemm(x, trellis, suh, svh, K, mcg, mul1, n_out: int) -> y
        Prefill/batch path (m>1).

    exl3_moe_max_concurrency(device: int) -> int
        Expert groups the fused MoE kernel can run concurrently (buffer count).

    exl3_moe(hidden, out_state, expert_count, token_sorted, weight_sorted,
             temp_state_g, temp_state_u, temp_intermediate_g,
             temp_intermediate_u, act_function, K_gate, K_up, K_down,
             gate/up/down ptrs (trellis, suh, svh), gate/up/down mcg/mul1,
             act_limit, num_active, output_scratch, fused_base,
             count_lo, count_hi, m_tile)
        Fused MoE: one cooperative kernel per [count_lo, count_hi] row band
        runs gate/up/down for every active expert. Deterministic mode
        (output_scratch set): each fused assignment writes a weighted fp32
        row at fused_base[expert] + row; out_state is untouched.

    exl3_moe_gather(out_state, scratch, flat_expert, inv_order, expert_start,
                    slot_base, slot_kind, weight_sorted)
        Sums each token's top-k slots (in k order) from the fp32 scratch into
        the pre-zeroed fp32 out_state.

    exl3_reconstruct(shape_out, trellis, suh, svh, K, mcg, mul1) -> fp16
        Dequantize for correctness checks (reconstruct-MSE vs source).
        Non-fused path (scales folded in-kernel); mirrors
        LinearEXL3.reconstruct_hgemm without the Hadamard pre/post passes.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass

_EXT_MODULE = "vllm_exl3_sm121_ext"
_REQUIRED_OPS = (
    "exl3_gemv",
    "exl3_gemm",
    "exl3_moe",
    "exl3_moe_gather",
    "exl3_moe_max_concurrency",
    "exl3_reconstruct",
)


@dataclass(frozen=True)
class ExtensionStatus:
    available: bool
    module: str
    missing_ops: tuple[str, ...]
    detail: str


def probe() -> ExtensionStatus:
    """Import the built extension and verify the op surface exists."""
    try:
        mod = importlib.import_module(_EXT_MODULE)
    except ImportError as exc:
        return ExtensionStatus(
            available=False,
            module=_EXT_MODULE,
            missing_ops=_REQUIRED_OPS,
            detail=(
                f"extension not built: {exc}. Build on the GB10 cluster via "
                "scripts/bootstrap_cluster.sh (aarch64, CUDA 13, "
                "TORCH_CUDA_ARCH_LIST=12.0;12.1)."
            ),
        )
    missing = tuple(op for op in _REQUIRED_OPS if not hasattr(mod, op))
    return ExtensionStatus(
        available=not missing,
        module=_EXT_MODULE,
        missing_ops=missing,
        detail="ok" if not missing else f"missing ops: {missing}",
    )


def require() -> object:
    """Return the extension module or raise with build guidance."""
    status = probe()
    if not status.available:
        raise RuntimeError(f"EXL3 extension unavailable: {status.detail}")
    return importlib.import_module(_EXT_MODULE)

"""Exl3MoEMethod: vLLM FusedMoE method for EXL3 MoE layers (v1: TP=1).

Requires vLLM installed (pinned) plus the wired `vllm_exl3_sm121_ext`.
Imported only from the plugin registration path.

Design (verified against the pinned source):
- Params live on the RoutedExperts submodule (`w13_trellis` etc.) so the
  model's expert_params_mapping resolves them by name; the loader receives
  (param, loaded_weight, name_mapped, expert_id=, shard_id=, return_success=)
  and must return True.
- Trellis layout mirrors the pack exactly: one trellis per expert is
  (in/16, out/16, 16*K) int16. Gate and up stay SEPARATE slots
  (`w13_trellis[e, 0]` = gate/w1, `[e, 1]` = up/w3) because the pack's
  per-half `suh` input scales are not equal; a single fused-GEMM over both
  halves would be an approximation. apply() runs gate and up as two GEMMs.
- w2 (down) is a single slot per expert: (in/16, out/16, 16*K).
  (Regression fixed 2026-09-19: the first v0 slots were shaped
  (e, 2*inter, in/16, out/16), a 72 GiB/layer allocation that CUDA-OOM'd
  the first full-model load before any weight was read.)
- TP=1 in v1: every expert local. TP>1/EP fails closed (M3c) — the mapping
  already carries physical expert ids, so EP is loader-range work.
- apply() v1: per-expert loop over routed tokens with the wired exl3_gemm
  (correct-first; coop kernels in the M2b-perf pass).
"""

from __future__ import annotations

import functools
from typing import Any

import torch
from torch.nn import Parameter

from vllm.model_executor.layers.fused_moe import FusedMoEMethodBase
from vllm.model_executor.parameter import ModelWeightParameter

from . import kernels

# Reuse HF-prefix reconstruction (pack lookup) from the dense method module.
from .linear import _to_hf_prefix

_SHARD_IDX = {"w1": 0, "w3": 1}


def _slot(shape: tuple[int, ...], dtype: torch.dtype,
          loader) -> ModelWeightParameter:
    return ModelWeightParameter(data=torch.empty(shape, dtype=dtype),
                                input_dim=1, output_dim=0,
                                weight_loader=loader)


class Exl3MoEMethod(FusedMoEMethodBase):
    """Routed-expert MoE backed by per-expert wired exl3_gemm calls."""

    def __init__(self, quant_config, moe, prefix: str) -> None:
        super().__init__(moe)
        self.quant_config = quant_config
        self.prefix = prefix
        self.hf_prefix = _to_hf_prefix(prefix)
        self.num_experts = 0
        self.inter = 0
        self.hidden = 0
        self.k13 = 0
        self.k2 = 0
        self.mcg = False
        self.mul1 = True

    # -- vLLM interface -------------------------------------------------

    def create_weights(
        self,
        layer: "RoutedExperts",
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        from vllm.distributed import get_tensor_model_parallel_world_size
        if get_tensor_model_parallel_world_size() != 1:
            raise NotImplementedError(
                f"exl3 MoE for {self.prefix}: TP>1/EP lands in M3c "
                f"(mapping carries physical ids; loader-range work remains)")
        self.num_experts = num_experts
        self.hidden = hidden_size
        # TP=1: partition == full intermediate
        self.inter = intermediate_size_per_partition

        # Probe this layer's expert geometry from pack metadata (no GPU).
        p_gate = self._expert_entry(0, "gate_proj")
        p_up = self._expert_entry(0, "up_proj")
        p_down = self._expert_entry(0, "down_proj")
        tg = p_gate.tensor(".trellis")
        tu = p_up.tensor(".trellis")
        td = p_down.tensor(".trellis")
        for name, t in (("gate_proj", tg), ("up_proj", tu), ("down_proj", td)):
            if t is None:
                raise ValueError(
                    f"exl3 MoE {self.prefix}: expert 0 {name} lacks .trellis")
            if t.dim() != 3:
                raise ValueError(
                    f"exl3 MoE {self.prefix}: expert 0 {name} trellis dims "
                    f"{tuple(t.shape)} are not (in/16, out/16, 16K)")
        if tuple(tg.shape) != tuple(tu.shape):
            raise ValueError(
                f"exl3 MoE {self.prefix}: gate/up trellis geometry mismatch "
                f"{tuple(tg.shape)} vs {tuple(tu.shape)}")
        t1, t2, t3 = (int(tg.shape[0]), int(tg.shape[1]), int(tg.shape[2]))
        d1, d2, d3 = (int(td.shape[0]), int(td.shape[1]), int(td.shape[2]))
        if t2 * 16 != self.inter:
            raise ValueError(
                f"exl3 MoE {self.prefix}: gate out {t2 * 16} != "
                f"intermediate {self.inter}")
        if d1 * 16 != self.inter or d2 * 16 != self.hidden:
            raise ValueError(
                f"exl3 MoE {self.prefix}: down in/out {d1 * 16}/{d2 * 16} != "
                f"{self.inter}/{self.hidden}")
        self.k13 = t3 // 16
        self.k2 = d3 // 16
        self.mcg = p_gate.tensor(".mcg") is not None
        self.mul1 = p_gate.tensor(".mul1") is not None
        for p, nm in ((p_up, "up_proj"), (p_down, "down_proj")):
            if (p.tensor(".mcg") is not None) != self.mcg or \
                    (p.tensor(".mul1") is not None) != self.mul1:
                raise RuntimeError(
                    f"exl3 MoE {self.prefix}: codebook flags differ for {nm}")

        e, i, h = num_experts, self.inter, hidden_size
        for name, shape, dtype in (
            # [e, 0] = gate (w1), [e, 1] = up (w3) — split, never fused
            ("w13_trellis", (e, 2, t1, t2, t3), torch.int16),
            ("w13_suh", (e, 2, h), torch.float16),
            ("w13_svh", (e, 2, i), torch.float16),
            ("w13_mul1", (e, 2), torch.int32),
            ("w2_trellis", (e, d1, d2, d3), torch.int16),
            ("w2_suh", (e, i), torch.float16),
            ("w2_svh", (e, h), torch.float16),
            ("w2_mul1", (e,), torch.int32),
        ):
            setattr(layer, name, _slot(
                shape, dtype,
                functools.partial(self._load_one, name)))
        layer.exl3_moe_k = self.k13

    def _load_one(self, pname: str, param: Parameter,
                  loaded_weight: torch.Tensor, name_mapped: str = "",
                  expert_id: int | None = None,
                  shard_id: str | None = None,
                  return_success: bool = False, **kwargs: Any) -> bool:
        """Assemble one expert shard into its slot. Returns True."""
        if expert_id is None:
            raise ValueError(f"exl3 MoE {self.prefix}: loader needs expert_id")
        e = int(expert_id)
        if not (0 <= e < self.num_experts):
            # Non-local replica (EPLB): nothing to do here.
            return True
        if pname.startswith("w13_"):
            if shard_id not in _SHARD_IDX:
                raise ValueError(
                    f"exl3 MoE {self.prefix}: w13 shard_id={shard_id!r}")
            s = _SHARD_IDX[shard_id]
            if pname == "w13_trellis":
                param.data[e, s].copy_(loaded_weight)
            elif pname == "w13_suh":
                param.data[e, s].copy_(loaded_weight.to(param.dtype))
            elif pname == "w13_svh":
                param.data[e, s].copy_(loaded_weight.to(param.dtype))
            else:  # w13_mul1
                param.data[e, s].copy_(loaded_weight.to(param.dtype))
        elif pname == "w2_trellis":
            param.data[e].copy_(loaded_weight)
        elif pname in ("w2_suh", "w2_svh", "w2_mul1"):
            param.data[e].copy_(
                loaded_weight.reshape(param.data[e].shape).to(param.dtype))
        else:
            raise ValueError(f"exl3 MoE {self.prefix}: unknown param {pname}")
        self._check_shapes(pname, param)
        return True

    def get_fused_moe_quant_config(self, layer) -> None:
        # v1 apply() is a custom per-expert loop, not the modular kernel
        # infra, so no kernel quant config is needed.
        return None

    def apply(self, layer, x, topk_weights, topk_ids,
              shared_experts=None, shared_experts_input=None):
        ext = kernels.require()
        in_dtype = x.dtype
        if x.dtype == torch.bfloat16:
            x = x.to(torch.float16)
        out = torch.zeros_like(x)
        e_ids = topk_ids.unique()
        for e in e_ids.tolist():
            rows = (topk_ids == e).nonzero()[:, 0]
            if rows.numel() == 0:
                continue
            tok = x[rows].contiguous()
            w = topk_weights[rows, (topk_ids[rows] == e).nonzero()[:, 1]]
            w = w.to(out.dtype)
            g = ext.exl3_gemm(tok, layer.w13_trellis[e, 0], layer.w13_suh[e, 0],
                              layer.w13_svh[e, 0], self.k13, self.mcg, self.mul1,
                              self.inter)
            u = ext.exl3_gemm(tok, layer.w13_trellis[e, 1], layer.w13_suh[e, 1],
                              layer.w13_svh[e, 1], self.k13, self.mcg, self.mul1,
                              self.inter)
            h = torch.nn.functional.silu(g) * u
            d = ext.exl3_gemm(h, layer.w2_trellis[e], layer.w2_suh[e],
                              layer.w2_svh[e], self.k2, self.mcg, self.mul1,
                              self.hidden)
            out[rows] += d * w.unsqueeze(-1)
        if out.dtype != in_dtype:
            out = out.to(in_dtype)
        return out

    # -- internals -------------------------------------------------------

    def _expert_entry(self, expert: int, proj: str):
        cfg = self.quant_config
        base = self.prefix
        if base.endswith(".experts"):
            base = base[:len(base) - len(".experts")]
        hbase = self.hf_prefix
        if hbase.endswith(".experts"):
            hbase = hbase[:len(hbase) - len(".experts")]
        for cand in (f"{base}.experts.{expert}.{proj}",
                     f"{hbase}.experts.{expert}.{proj}"):
            for name, m in cfg.pack.modules.items():
                if name == cand and m.is_exl3:
                    return m
        raise ValueError(
            f"exl3 MoE {self.prefix}: no pack entry for expert {expert} {proj}")

    def _check_shapes(self, pname, param) -> None:
        # shapes fixed at create_weights; loader misuse fails here, loudly.
        expected = {
            "w13_trellis": (self.num_experts, 2),
            "w2_trellis": (self.num_experts,),
        }
        if pname in expected:
            prefix = expected[pname]
            assert tuple(param.data.shape[:len(prefix)]) == prefix, \
                f"exl3 MoE {self.prefix}.{pname} shape {tuple(param.data.shape)}"

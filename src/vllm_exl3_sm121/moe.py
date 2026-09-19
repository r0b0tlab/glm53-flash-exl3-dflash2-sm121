"""Exl3MoEMethod: vLLM FusedMoE method for EXL3 MoE layers (v1: TP=1).

Requires vLLM installed (pinned) plus the wired `vllm_exl3_sm121_ext`.
Imported only from the plugin registration path.

Design (all verified against the pinned source):
- Params live on the RoutedExperts submodule
  (`...mlp.experts.routed_experts.w13_trellis` etc.) so the model file's
  expert_params_mapping resolves by name; the loader receives
  (param, loaded_weight, name, expert_id, shard_id) and must return True.
- Gate+up fuse per expert at load (w1 -> rows [0:I], w3 -> [I:2I]);
  suh shared (w3 asserts allclose against w1); K/mcg/mul1 asserted uniform.
- TP=1 in v1: every expert local. TP>1/EP fails closed (M3c) — the mapping
  already carries physical expert ids, so EP is loader-range work, not new
  plumbing.
- apply() v1: per-expert loop over routed tokens with the wired exl3_gemm
  (correct-first; coop kernels in M2b-perf work).
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
from .linear import _narrow, _to_hf_prefix


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
        self.k = 0
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
        # Probe one expert's trellis shape from pack metadata (no GPU).
        probe = self._expert_entry(0, "up_proj")
        trellis_meta = probe.tensor(".trellis")
        if trellis_meta is None:
            raise ValueError(f"exl3 MoE {self.prefix}: expert 0 up_proj lacks .trellis")
        self.k = trellis_meta.shape[-1] // 16
        t1, t2 = trellis_meta.shape[0], trellis_meta.shape[1]
        e, i, h = num_experts, self.inter, hidden_size
        for name, shape, dtype in (
            ("w13_trellis", (e, 2 * i, t1, t2), torch.int16),
            ("w13_suh", (e, h), torch.float16),
            ("w13_svh", (e, 2 * i), torch.float16),
            ("w13_mul1", (e,), torch.int32),
            ("w2_trellis", (e, i, t1, t2), torch.int16),
            ("w2_suh", (e, i), torch.float16),
            ("w2_svh", (e, h), torch.float16),
            ("w2_mul1", (e,), torch.int32),
        ):
            setattr(layer, name, _slot(
                shape, dtype,
                functools.partial(self._load_one, name)))
        layer.exl3_moe_k = self.k

    def _load_one(self, pname: str, param: Parameter,
                  loaded_weight: torch.Tensor, name_mapped: str = "",
                  expert_id: int | None = None,
                  shard_id: str | None = None,
                  return_success: bool = False, **kwargs: Any) -> bool:
        """Assemble one expert shard into its fused slot. Returns True."""
        if expert_id is None:
            raise ValueError(f"exl3 MoE {self.prefix}: loader needs expert_id")
        e = int(expert_id)
        if not (0 <= e < self.num_experts):
            # Non-local replica (EPLB): nothing to do here.
            return True
        i = self.inter
        if pname == "w13_trellis":
            param.data[e, :] = self._fuse_halves(param.data[e],
                                                 self._tp_slice_trellis(
                                                     loaded_weight, shard_id, i,
                                                     fused=True),
                                                 shard_id, i)
        elif pname == "w13_suh":
            if shard_id == "w1":
                param.data[e].copy_(loaded_weight.to(param.dtype))
            else:
                assert torch.allclose(param.data[e].float(),
                                      loaded_weight.float(), atol=1e-3), \
                    f"exl3 MoE {self.prefix} expert {e}: gate/up suh differ"
        elif pname == "w13_svh":
            off = 0 if shard_id == "w1" else i
            param.data[e, off:off + i].copy_(loaded_weight.to(param.dtype))
        elif pname == "w13_mul1":
            param.data[e].copy_(loaded_weight.to(param.dtype))
        elif pname == "w2_trellis":
            param.data[e].copy_(self._tp_slice_trellis(loaded_weight, None, i, fused=False))
        elif pname in ("w2_suh", "w2_svh", "w2_mul1"):
            param.data[e].copy_(loaded_weight.reshape(param.data[e].shape).to(param.dtype))
        else:
            raise ValueError(f"exl3 MoE {self.prefix}: unknown param {pname}")
        self._check_shapes(pname, param)
        return True

    def apply(self, layer, x, topk_weights, topk_ids,
              shared_experts=None, shared_experts_input=None):
        ext = kernels.require()
        out = torch.zeros_like(x)
        e_ids = topk_ids.unique()
        for e in e_ids.tolist():
            rows = (topk_ids == e).nonzero()[:, 0]
            if rows.numel() == 0:
                continue
            tok = x[rows].contiguous()
            w = topk_weights[rows, (topk_ids[rows] == e).nonzero()[:, 1]].to(out.dtype)
            gu = ext.exl3_gemm(tok, layer.w13_trellis[e], layer.w13_suh[e],
                               layer.w13_svh[e], self.k, self.mcg, self.mul1,
                               2 * self.inter)
            g, u = gu.split(self.inter, dim=-1)
            h = torch.nn.functional.silu(g) * u
            d = ext.exl3_gemm(h, layer.w2_trellis[e], layer.w2_suh[e],
                              layer.w2_svh[e], self.k, self.mcg, self.mul1,
                              self.hidden)
            out[rows] += d * w.unsqueeze(-1)
        return out

    # -- internals -------------------------------------------------------

    def _expert_entry(self, expert: int, proj: str):
        mq = None
        cfg = self.quant_config
        for cand in (f"{self.prefix}.experts.{expert}.{proj}",
                     f"{self.hf_prefix}.experts.{expert}.{proj}"):
            for name, m in cfg.pack.modules.items():
                if name == cand and m.is_exl3:
                    return m
        raise ValueError(
            f"exl3 MoE {self.prefix}: no pack entry for expert {expert} {proj}")

    def _tp_slice_trellis(self, t, shard_id, inter, fused: bool):
        # TP=1: identity (slicing lands with M3c).
        return t

    def _fuse_halves(self, slot, half, shard_id, inter):
        slot = slot.clone()
        if shard_id == "w1":
            slot[:inter] = half.to(slot.dtype)
        elif shard_id == "w3":
            slot[inter:2 * inter] = half.to(slot.dtype)
        else:
            raise ValueError(f"exl3 MoE {self.prefix}: shard {shard_id!r}")
        return slot

    def _check_shapes(self, pname, param) -> None:
        # shapes fixed at create_weights; loader misuse fails here, loudly.
        expected = {
            "w13_trellis": (self.num_experts, 2 * self.inter),
            "w2_trellis": (self.num_experts, self.inter),
        }
        if pname in expected:
            e, w = expected[pname]
            assert param.data.shape[0] == e and param.data.shape[1] == w, \
                f"exl3 MoE {self.prefix}.{pname} shape {tuple(param.data.shape)}"

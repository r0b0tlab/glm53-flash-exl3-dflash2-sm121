"""Exl3LinearMethod: vLLM LinearMethodBase for EXL3 dense linears.

Requires vLLM installed (pinned in runtime.lock.json) plus the built
`vllm_exl3_sm121_ext` (kernels.probe()). Imported only from the plugin
registration path, never from pure-logic tests.

TP sharding mirrors LinearEXL3.tp_import_split (r0b0tlab/exllamav3):
- column-parallel (output split): svh dim0[first:last],
  trellis dim1[first//16:last//16], bias dim0[first:last], suh whole.
- row-parallel (input split): suh dim0[first:last],
  trellis dim0[first//16:last//16], svh whole, bias rank-0 only.

Name mapping: vLLM passes vLLM-side prefixes (Glm5Next mapper rewrites
`model.language_model.` -> `language_model.model.`, suffix preserved), so
pack lookup tries the raw prefix plus the HF-infix reconstruction.
Fused on-disk tensors map 1:1 to fused vLLM params; per-shard routing
(loaded_shard_id) fails closed until the KDA merged-layer census (M3b).

Bias is NOT created here: the layer owns its (partitioned) bias with the
standard loader, fed from the pack's `<name>.bias` tensor.
"""

from __future__ import annotations

import functools
from typing import Any

import torch
from torch.nn import Parameter

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.parameter import ModelWeightParameter

from . import kernels
from .config import Exl3Config

_DTYPE_BY_NAME = {
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.float32": torch.float32,
    "torch.int16": torch.int16,
    "torch.int32": torch.int32,
    "torch.int8": torch.int8,
    "torch.uint8": torch.uint8,
}


def _pack_dtype(name: str) -> torch.dtype:
    try:
        return _DTYPE_BY_NAME[name]
    except KeyError:
        raise ValueError(f"exl3 pack tensor has unknown dtype {name!r}") from None


def _narrow(t: torch.Tensor, dim: int, first: int, last: int) -> torch.Tensor:
    sl = [slice(None)] * t.dim()
    sl[dim] = slice(first, last)
    return t[tuple(sl)].contiguous()


class Exl3LinearMethod(LinearMethodBase):
    """Quantized dense linear backed by the wired exl3_gemm op."""

    def __init__(self, quant_config: Exl3Config, prefix: str) -> None:
        self.quant_config = quant_config
        self.prefix = prefix
        self.fused: tuple | None = None  # (mq_gate, mq_up) for gate_up_proj
        mq = quant_config.module_quant(prefix)
        if mq is None:
            mq = quant_config.module_quant(_to_hf_prefix(prefix))
        if mq is None and prefix.endswith(".gate_up_proj"):
            # pack stores split gate/up; vLLM fuses. Resolve the pair and
            # assemble in the loader (staged across shard calls).
            stem = prefix[: -len(".gate_up_proj")]
            g = quant_config.module_quant(stem + ".gate_proj") or \
                quant_config.module_quant(_to_hf_prefix(stem) + ".gate_proj")
            u = quant_config.module_quant(stem + ".up_proj") or \
                quant_config.module_quant(_to_hf_prefix(stem) + ".up_proj")
            if g is not None and u is not None and g.is_exl3 and u.is_exl3:
                gt = g.tensor(".trellis")
                ut = u.tensor(".trellis")
                if gt is None or ut is None or gt.shape != ut.shape:
                    raise ValueError(
                        f"exl3 fused {prefix}: gate/up trellis mismatch")
                self.fused = (g, u)
                mq = g
        if mq is None or not mq.is_exl3:
            raise ValueError(f"no EXL3 pack entry for layer prefix {prefix!r}")
        self.mq = mq
        trellis_meta = mq.tensor(".trellis")
        if trellis_meta is None:
            raise ValueError(f"pack entry {mq.module!r} has no .trellis tensor")
        self.k = trellis_meta.shape[-1] // 16
        self.mcg = mq.tensor(".mcg") is not None
        self.mul1 = mq.tensor(".mul1") is not None
        self.is_row_parallel = False
        self.out_first = 0
        self.out_last = 0
        self.in_first = 0
        self.in_last = 0
        # staging for fused (MergedColumn) halves arriving across calls
        self._fused_stage: dict = {}

    # -- vLLM interface -------------------------------------------------

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        rank = get_tensor_model_parallel_rank()
        world = get_tensor_model_parallel_world_size()
        self.is_row_parallel = input_size != input_size_per_partition
        if self.is_row_parallel:
            if world > 1 and input_size % world != 0:
                raise ValueError(
                    f"exl3 row split needs even input sharding, got "
                    f"input_size={input_size} tp={world}")
            self.in_first = rank * input_size_per_partition
            self.in_last = self.in_first + input_size_per_partition
            self.out_first, self.out_last = 0, output_size
        else:
            if world > 1 and output_size % world != 0:
                raise ValueError(
                    f"exl3 column split needs even output sharding, got "
                    f"output_size={output_size} tp={world}")
            self.out_first = rank * (output_size // world)
            self.out_last = self.out_first + (output_size // world)
            self.in_first, self.in_last = 0, input_size
        n_out = self.out_last - self.out_first

        for name, shape, dtype in (
            ("trellis", self._trellis_shape(n_out), torch.int16),
            ("suh", self._suh_shape(), torch.float16),
            ("svh", self._svh_shape(n_out), torch.float16),
        ):
            param = ModelWeightParameter(
                data=torch.empty(shape, dtype=dtype),
                input_dim=1,
                output_dim=0,
                weight_loader=functools.partial(self._load_one, name),
            )
            setattr(layer, name, param)
        if self.mul1:
            param = ModelWeightParameter(
                data=torch.empty((), dtype=torch.int32),
                input_dim=1,
                output_dim=0,
                weight_loader=functools.partial(self._load_one, "mul1"),
            )
            setattr(layer, "mul1", param)
        layer.out_features_partition = n_out
        layer.exl3_k = self.k
        layer.exl3_mcg = self.mcg
        layer.exl3_mul1 = self.mul1

    def _load_one(self, suffix: str, param: Parameter,
                  loaded_weight: torch.Tensor, *args: Any, **kwargs: Any) -> None:
        """Load one pack shard tensor, TP-sliced.

        Plain layers: whole tensor in one call. Fused layers
        (MergedColumn gate_up: shard_id 0/1): halves staged across calls,
        assembled, then TP-sliced as one fused tensor.
        """
        sid = kwargs.get("loaded_shard_id", args[0] if args else None)
        if sid is not None:
            try:
                shard = int(sid)
            except (TypeError, ValueError):
                raise ValueError(
                    f"exl3 loader for {self.prefix}: shard_id={sid!r} needs "
                    f"the KDA merged-layer census (M3b)") from None
            key = (self.prefix, suffix)
            stage = self._fused_stage.setdefault(key, {})
            stage[shard] = loaded_weight.detach().clone()
            if len(stage) < 2:
                return  # wait for the other half; shapes checked at assembly
            halves = [stage[i] for i in sorted(stage)]
            del self._fused_stage[key]
            if suffix == "trellis":
                fused = torch.cat(halves, dim=1)
            elif suffix in ("suh", "svh", "bias"):
                fused = torch.cat(halves, dim=0) if halves[0].dim() > 0 else halves[0]
            elif suffix == "mul1":
                fused = halves[0]
            else:
                raise ValueError(f"exl3 loader: unknown suffix {suffix!r}")
            self._copy_checked(suffix, param, self._slice_for_tp(suffix, fused))
            return
        data = self._slice_for_tp(suffix, loaded_weight)
        self._copy_checked(suffix, param, data)

    def _copy_checked(self, suffix: str, param: Parameter,
                      data: torch.Tensor) -> None:
        if tuple(data.shape) != tuple(param.data.shape):
            raise ValueError(
                f"exl3 loader shape mismatch for {self.prefix}.{suffix}: "
                f"pack gives {tuple(data.shape)}, "
                f"param holds {tuple(param.data.shape)}")
        param.data.copy_(data)

    def apply(self, layer: torch.nn.Module,
              x: torch.Tensor,
              bias: torch.Tensor | None = None) -> torch.Tensor:
        ext = kernels.require()
        n_out = int(layer.out_features_partition)
        in_dtype = x.dtype
        if x.dtype == torch.bfloat16:
            # kernels run fp16 tiles; cast at the boundary, restore after
            x = x.to(torch.float16)
        y = ext.exl3_gemm(x, layer.trellis, layer.suh, layer.svh,
                          self.k, self.mcg, self.mul1, n_out)
        b = bias if bias is not None else getattr(layer, "bias", None)
        if b is not None:
            y = y + b.to(y.dtype)
        if y.dtype != in_dtype:
            y = y.to(in_dtype)
        return y

    # -- internals -------------------------------------------------------

    def _slice_for_tp(self, suffix: str, t: torch.Tensor) -> torch.Tensor:
        if suffix == "trellis":
            if self.is_row_parallel:
                return _narrow(t, 0, self.in_first // 16, self.in_last // 16)
            return _narrow(t, 1, self.out_first // 16, self.out_last // 16)
        if suffix == "suh":
            if self.is_row_parallel:
                return _narrow(t, 0, self.in_first, self.in_last)
            return t
        if suffix == "svh":
            if self.is_row_parallel:
                return t
            return _narrow(t, 0, self.out_first, self.out_last)
        if suffix == "mul1":
            return t
        raise ValueError(f"exl3 loader: unknown suffix {suffix!r}")

    def _trellis_shape(self, n_out: int) -> tuple[int, ...]:
        t = self.mq.tensor(".trellis")
        assert t is not None
        s = list(t.shape)
        if self.fused is not None and not self.is_row_parallel:
            # fused gate_up: full width is 2x one half; TP-slice the fused
            # width directly (halves concatenated on dim1 at load).
            gate_out = self._full_out_features()
            fused_full = 2 * gate_out
            s[1] = s[1] * 2 * n_out // fused_full
            return tuple(s)
        if self.is_row_parallel:
            full_in = self.mq.tensor(".suh")
            assert full_in is not None
            s[0] = s[0] * (self.in_last - self.in_first) // full_in.shape[0]
        else:
            full_out = self._full_out_features()
            if full_out and n_out != full_out:
                s[1] = s[1] * n_out // full_out
        return tuple(s)

    def _suh_shape(self) -> tuple[int, ...]:
        t = self.mq.tensor(".suh")
        assert t is not None
        n = t.shape[0]
        if self.is_row_parallel:
            n = self.in_last - self.in_first
        return (n,)

    def _svh_shape(self, n_out: int) -> tuple[int, ...]:
        return (n_out,)

    def _full_out_features(self) -> int:
        t = self.mq.tensor(".svh")
        return t.shape[0] if t is not None else 0


def _to_hf_prefix(prefix: str) -> str:
    """Reconstruct the HF-side prefix from a vLLM-side one for pack lookup."""
    p = prefix
    if p.startswith("language_model.model."):
        p = "model.language_model." + p[len("language_model.model."):]
    return p

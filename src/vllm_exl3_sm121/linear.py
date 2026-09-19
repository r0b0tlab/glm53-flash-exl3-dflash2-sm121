"""Exl3LinearMethod: vLLM LinearMethodBase for EXL3 dense linears.

Requires vLLM installed (pinned in runtime.lock.json) plus the built
`vllm_exl3_sm121_ext` (kernels.probe()). Imported only from the plugin
registration path, never from pure-logic tests.

Slot design (split-exact, 2026-09-19):
- One EXL3 GEMM per pack module. Merged vLLM layers whose pack stores the
  halves as separate EXL3 modules (gate_proj+up_proj, q_a_proj+
  kv_a_proj_with_mqa) keep the halves as separate slots: the pack's per-half
  `suh` input scales are NOT equal (LDLQ is per-tensor), so a single fused
  trellis/GEMM over both halves would be an approximation. apply() runs one
  GEMM per slot and concatenates along the output dim (merged-column
  semantics).
- Slot 0 lives in the vLLM params (`trellis`, `suh`, `svh`, `mul1`); extra
  slots (merged layers only) live in sibling plain tensors `self._extra`
  because the loader resolves params by checkpoint suffix (one `trellis`
  name per layer).
- Fused on-disk tensors (e.g. qkv_proj) are single-slot; `loaded_shard_id`
  from the vLLM merged-column loader selects the slot.

TP sharding mirrors LinearEXL3.tp_import_split (r0b0tlab/exllamav3):
- column-parallel (output split): svh dim0[first:last],
  trellis dim1[first//16:last//16], suh whole.
- row-parallel (input split): suh dim0[first:last],
  trellis dim0[first//16:last//16], svh whole.
`output_partition_sizes` from vLLM are already per-rank sizes, so the rank
slice of each shard is [rank*size, (rank+1)*size].

Name mapping: vLLM passes vLLM-side prefixes (Glm5Next mapper rewrites
`model.language_model.` -> `language_model.model.`), so pack lookup tries
the raw prefix plus the HF-infix reconstruction.

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
from vllm.model_executor.layers.linear import LinearMethodBase
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

    # vLLM merged layers whose pack keeps the halves split as EXL3 modules.
    FUSED_GROUPS = (
        (".gate_up_proj", (".gate_proj", ".up_proj")),
        (".fused_qkv_a_proj", (".q_a_proj", ".kv_a_proj_with_mqa")),
        (".wk_weights_proj", (".wk", ".weights_proj")),
    )

    def __init__(self, quant_config: Exl3Config, prefix: str) -> None:
        self.quant_config = quant_config
        self.prefix = prefix
        self.shards = self._resolve_shards(quant_config, prefix)
        self.n_shards = len(self.shards)
        self.is_row_parallel = False
        self.in_first = 0
        self.in_last = 0
        self._geom: list[dict[str, Any]] = []
        self._extra: dict[str, list[torch.Tensor]] = {}
        self._layer: torch.nn.Module | None = None

    # -- resolution ------------------------------------------------------

    def _resolve_shards(self, qc: Exl3Config, prefix: str) -> list:
        for suffix, members in self.FUSED_GROUPS:
            if not prefix.endswith(suffix):
                continue
            stem = prefix[: -len(suffix)]
            found = []
            for m in members:
                mm = qc.module_quant(stem + m)
                if mm is None:
                    mm = qc.module_quant(_to_hf_prefix(stem) + m)
                found.append(mm)
            exl3 = [m for m in found if m is not None and m.is_exl3]
            if not exl3:
                break  # not an EXL3 fused layer; fall through to dense
            if len(exl3) != len(found):
                missing = [
                    members[i]
                    for i, m in enumerate(found)
                    if m is None or not m.is_exl3
                ]
                raise RuntimeError(
                    f"exl3 {prefix}: fused members {missing} are not EXL3 "
                    f"pack entries — pack must quantize both halves or neither"
                )
            return found
        mq = qc.module_quant(prefix)
        if mq is None:
            mq = qc.module_quant(_to_hf_prefix(prefix))
        if mq is not None and mq.is_exl3:
            return [mq]
        raise ValueError(f"no EXL3 pack entry for layer prefix {prefix!r}")

    @staticmethod
    def _pack_geometry(mq) -> dict[str, Any]:
        trellis = mq.tensor(".trellis")
        if trellis is None:
            raise ValueError(f"pack entry {mq.module!r} has no .trellis tensor")
        if trellis.dim() != 3:
            raise ValueError(
                f"pack entry {mq.module!r} trellis dims {tuple(trellis.shape)} "
                f"are not (in/16, out/16, 16K)"
            )
        suh = mq.tensor(".suh")
        svh = mq.tensor(".svh")
        return {
            "T1": int(trellis.shape[0]),
            "T2": int(trellis.shape[1]),
            "T3": int(trellis.shape[2]),
            "k": int(trellis.shape[2]) // 16,
            "suh_len": int(suh.shape[0]) if suh is not None else 0,
            "svh_len": int(svh.shape[0]) if svh is not None else 0,
            "mcg": mq.tensor(".mcg") is not None,
            "mul1": mq.tensor(".mul1") is not None,
        }

    # -- vLLM interface --------------------------------------------------

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

        geom = [self._pack_geometry(mq) for mq in self.shards]
        if any(g["suh_len"] == 0 or g["svh_len"] == 0 for g in geom):
            raise ValueError(f"exl3 {self.prefix}: pack entry lacks suh/svh")
        mul1_flags = {g["mul1"] for g in geom}
        if len(mul1_flags) != 1:
            raise RuntimeError(
                f"exl3 {self.prefix}: mixed mul1 codebook across shards "
                f"{[g['mul1'] for g in geom]} — not supported"
            )
        self.mul1_all = mul1_flags.pop()

        if self.is_row_parallel:
            if self.n_shards != 1:
                raise ValueError(
                    f"exl3 {self.prefix}: row-parallel merged layers are not "
                    f"supported (row-parallel layers are never merged here)"
                )
            if world > 1 and input_size % world != 0:
                raise ValueError(
                    f"exl3 row split needs even input sharding, got "
                    f"input_size={input_size} tp={world}"
                )
            self.in_first = rank * input_size_per_partition
            self.in_last = self.in_first + input_size_per_partition
            per_out = [output_size]
        else:
            if len(output_partition_sizes) != self.n_shards:
                raise ValueError(
                    f"exl3 {self.prefix}: vLLM passes {len(output_partition_sizes)} "
                    f"output partitions but the pack resolves {self.n_shards} shards"
                )
            per_out = [int(s) for s in output_partition_sizes]

        for i, g in enumerate(geom):
            if g["T1"] != input_size // 16:
                raise ValueError(
                    f"exl3 {self.prefix}[{i}]: trellis in-dim {g['T1']} != "
                    f"input_size/16 {input_size // 16}"
                )
            if g["svh_len"] != per_out[i] * world:
                raise ValueError(
                    f"exl3 {self.prefix}[{i}]: pack out {g['svh_len']} != "
                    f"per-rank {per_out[i]} x tp {world} (disable_tp layers "
                    f"with tp>1 need the M3c pass)"
                )
            if g["T2"] != g["svh_len"] // 16:
                raise ValueError(
                    f"exl3 {self.prefix}[{i}]: trellis out-dim {g['T2']} != "
                    f"out/16 {g['svh_len'] // 16}"
                )
            if not self.is_row_parallel and per_out[i] % 16 != 0:
                raise ValueError(
                    f"exl3 {self.prefix}[{i}]: per-rank out {per_out[i]} is "
                    f"not 16-aligned; trellis slicing undefined"
                )
            out_first = 0 if self.is_row_parallel else rank * per_out[i]
            shape_trellis = (
                (input_size_per_partition // 16, g["T2"], g["T3"])
                if self.is_row_parallel
                else (g["T1"], per_out[i] // 16, g["T3"])
            )
            shape_suh = (
                (input_size_per_partition,) if self.is_row_parallel else (input_size,)
            )
            shape_svh = (output_size,) if self.is_row_parallel else (per_out[i],)
            self._geom.append({
                **g,
                "n_out": per_out[i],
                "out_first": out_first,
                "out_last": out_first + per_out[i],
                "trellis": shape_trellis,
                "suh": shape_suh,
                "svh": shape_svh,
            })

        def _slot(name: str, shape: tuple, dtype: torch.dtype, i: int):
            if i == 0:
                p = ModelWeightParameter(
                    data=torch.empty(shape, dtype=dtype),
                    input_dim=1,
                    output_dim=0,
                    weight_loader=functools.partial(self._load_one, name),
                )
                setattr(layer, name, p)
                return p
            t = torch.empty(shape, dtype=dtype)
            self._extra.setdefault(name, []).append(t)
            return t

        for i, g in enumerate(self._geom):
            _slot("trellis", g["trellis"], torch.int16, i)
            _slot("suh", g["suh"], torch.float16, i)
            _slot("svh", g["svh"], torch.float16, i)
            if self.mul1_all:
                _slot("mul1", (), torch.int32, i)
        self._layer = layer
        layer.exl3_k = self._geom[0]["k"]

    def _load_one(self, suffix: str, param: Parameter,
                  loaded_weight: torch.Tensor, *args: Any, **kwargs: Any) -> None:
        """Load one pack shard tensor into its slot, TP-sliced."""
        sid = kwargs.get("loaded_shard_id", args[0] if args else None)
        if sid is None:
            sid = 0
        else:
            try:
                sid = int(sid)
            except (TypeError, ValueError):
                raise ValueError(
                    f"exl3 loader for {self.prefix}: unsupported shard id {sid!r}"
                ) from None
        if not (0 <= sid < self.n_shards):
            raise ValueError(
                f"exl3 loader for {self.prefix}: shard {sid} outside "
                f"0..{self.n_shards - 1}"
            )
        data = self._slice_for_tp(sid, suffix, loaded_weight)
        dst = param.data if sid == 0 else self._extra[suffix][sid - 1]
        if tuple(data.shape) != tuple(dst.shape):
            raise ValueError(
                f"exl3 loader shape mismatch for {self.prefix}.{suffix}[{sid}]: "
                f"pack gives {tuple(data.shape)}, slot holds {tuple(dst.shape)}"
            )
        dst.copy_(data)

    def apply(self, layer: torch.nn.Module,
              x: torch.Tensor,
              bias: torch.Tensor | None = None) -> torch.Tensor:
        ext = kernels.require()
        in_dtype = x.dtype
        if x.dtype == torch.bfloat16:
            # kernels run fp16 tiles; cast at the boundary, restore after
            x = x.to(torch.float16)
        parts = []
        for i, g in enumerate(self._geom):
            trellis = layer.trellis.data if i == 0 else self._extra["trellis"][i - 1]
            suh = layer.suh.data if i == 0 else self._extra["suh"][i - 1]
            svh = layer.svh.data if i == 0 else self._extra["svh"][i - 1]
            parts.append(ext.exl3_gemm(
                x, trellis, suh, svh, g["k"], g["mcg"], g["mul1"], g["n_out"]))
        y = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
        b = bias if bias is not None else getattr(layer, "bias", None)
        if b is not None:
            y = y + b.to(y.dtype)
        if y.dtype != in_dtype:
            y = y.to(in_dtype)
        return y

    # -- internals -------------------------------------------------------

    def _slice_for_tp(self, sid: int, suffix: str, t: torch.Tensor) -> torch.Tensor:
        g = self._geom[sid]
        if suffix == "trellis":
            if self.is_row_parallel:
                return _narrow(t, 0, self.in_first // 16, self.in_last // 16)
            return _narrow(t, 1, g["out_first"] // 16, g["out_last"] // 16)
        if suffix == "suh":
            if self.is_row_parallel:
                return _narrow(t, 0, self.in_first, self.in_last)
            return t
        if suffix == "svh":
            if self.is_row_parallel:
                return t
            return _narrow(t, 0, g["out_first"], g["out_last"])
        if suffix == "mul1":
            return t
        raise ValueError(f"exl3 loader: unknown suffix {suffix!r}")


def _to_hf_prefix(prefix: str) -> str:
    """Reconstruct the HF-side prefix from a vLLM-side one for pack lookup."""
    p = prefix
    if p.startswith("language_model.model."):
        p = "model.language_model." + p[len("language_model.model."):]
    return p

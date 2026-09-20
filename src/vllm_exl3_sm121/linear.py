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
- KDA `in_proj_qkvbfg_a` (6 merged shards) is a MIXED layer in our packs:
  q|k|v are fused into one EXL3 `qkv_proj` tensor (exllamav3 layout) while
  b/f_a/g_a are dense BF16. The three EXL3 slots are split out of the fused
  trellis by out-blocks (the model patch routes the fused tensor here with no
  shard id); the dense shards load into plain `weight` slots and run as
  matmuls.
- Slot 0 of each suffix lives in the vLLM params (`trellis`, `suh`, `svh`,
  `mul1`, `weight`); other slots live in sibling plain tensors
  `self._extra` because the loader resolves params by checkpoint suffix
  (one param name per suffix).
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
import re
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
        (".qkv_proj", (".q_proj", ".k_proj", ".v_proj")),
        (".fused_qkv_a_proj", (".q_a_proj", ".kv_a_proj_with_mqa")),
        (".wk_weights_proj", (".wk", ".weights_proj")),
    )

    def __init__(self, quant_config: Exl3Config, prefix: str) -> None:
        self.quant_config = quant_config
        self.prefix = prefix
        self.shards = self._resolve_shards(quant_config, prefix)
        self.n_shards = len(self.shards)
        self._qkv_split = any(s.get("qkv_part") is not None for s in self.shards)
        self.is_row_parallel = False
        self.in_first = 0
        self.in_last = 0
        self._geom: list[dict[str, Any]] = []
        self._extra: dict[str, list[torch.Tensor]] = {}
        self._primary: dict[str, int] = {}
        self._layer: torch.nn.Module | None = None

    # -- resolution ------------------------------------------------------

    @staticmethod
    def _lookup(qc: Exl3Config, name: str):
        candidates = [name, _to_hf_prefix(name)]
        if name.startswith("model."):
            # DFlash2 draft packs store bare names ("layers.0...", "fc")
            # while vLLM prefixes the draft model with "model.".
            bare = name[len("model."):]
            candidates.append(bare)
            # Draft layers are built with the target's layer offset in their
            # prefix ("model.layers.45..49" for a 5-layer drafter on a
            # 45-layer target) while the pack stores them 0-based.
            off = getattr(qc, "draft_layer_offset", 0)
            if off:
                m = re.match(r"layers\.(\d+)(\..+)", bare)
                if m and int(m.group(1)) >= off:
                    candidates.append(
                        f"layers.{int(m.group(1)) - off}{m.group(2)}")
        for cand in candidates:
            mq = qc.module_quant(cand)
            if mq is not None:
                return mq
        return None

    def _resolve_shards(self, qc: Exl3Config, prefix: str) -> list[dict]:
        # KDA merged input projection: fused EXL3 qkv + dense b/f_a/g_a.
        if prefix.endswith(".in_proj_qkvbfg_a"):
            stem = prefix[: -len(".in_proj_qkvbfg_a")]
            qkv = self._lookup(qc, stem + ".qkv_proj")
            if qkv is None or not qkv.is_exl3:
                raise ValueError(
                    f"no EXL3 qkv_proj pack entry for {prefix!r}")
            slots: list[dict] = [
                {"kind": "exl3", "mq": qkv, "qkv_part": i} for i in range(3)
            ]
            for m in (".b_proj", ".f_a_proj", ".g_a_proj"):
                mm = self._lookup(qc, stem + m)
                if mm is None:
                    raise RuntimeError(f"exl3 {prefix}: pack lacks {m}")
                slots.append({"kind": "dense", "mq": mm, "qkv_part": None})
            return slots
        for suffix, members in self.FUSED_GROUPS:
            if not prefix.endswith(suffix):
                continue
            stem = prefix[: -len(suffix)]
            found = [self._lookup(qc, stem + m) for m in members]
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
            return [{"kind": "exl3", "mq": m, "qkv_part": None} for m in found]
        mq = self._lookup(qc, prefix)
        if mq is not None and mq.is_exl3:
            return [{"kind": "exl3", "mq": mq, "qkv_part": None}]
        raise ValueError(f"no EXL3 pack entry for layer prefix {prefix!r}")

    @staticmethod
    def _pack_geometry(mq) -> dict[str, Any]:
        trellis = mq.tensor(".trellis")
        if trellis is None:
            raise ValueError(f"pack entry {mq.module!r} has no .trellis tensor")
        if len(trellis.shape) != 3:
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

        if len(output_partition_sizes) != self.n_shards:
            raise ValueError(
                f"exl3 {self.prefix}: vLLM passes {len(output_partition_sizes)} "
                f"output partitions but the pack resolves {self.n_shards} shards"
            )
        if self.is_row_parallel and self.n_shards != 1:
            raise ValueError(
                f"exl3 {self.prefix}: row-parallel merged layers are not "
                f"supported (row-parallel layers are never merged here)"
            )
        if self.is_row_parallel and world > 1 and input_size % world != 0:
            raise ValueError(
                f"exl3 row split needs even input sharding, got "
                f"input_size={input_size} tp={world}"
            )
        if self._qkv_split and world > 1:
            raise NotImplementedError(
                f"exl3 {self.prefix}: mixed EXL3/dense merged layers are "
                f"TP=1-only until the M3c pass (replicated f_a/g_a shards)"
            )
        self.in_first = rank * input_size_per_partition
        self.in_last = self.in_first + input_size_per_partition

        exl3_mul1 = [
            self._pack_geometry(s["mq"])["mul1"]
            for s in self.shards if s["kind"] == "exl3"
        ]
        if len(set(exl3_mul1)) > 1:
            raise RuntimeError(
                f"exl3 {self.prefix}: mixed mul1 codebook across shards "
                f"{exl3_mul1} — not supported"
            )
        self.mul1_all = all(exl3_mul1)

        for i, (s, per_out) in enumerate(zip(self.shards, output_partition_sizes)):
            per_out = int(per_out)
            if s["kind"] == "dense":
                w = s["mq"].tensor(".weight")
                if w is None:
                    raise ValueError(
                        f"exl3 {self.prefix}[{i}]: dense pack entry "
                        f"{s['mq'].module!r} has no .weight tensor")
                if int(w.shape[0]) != per_out * world:
                    raise ValueError(
                        f"exl3 {self.prefix}[{i}]: dense out {w.shape[0]} != "
                        f"per-rank {per_out} x tp {world}")
                if int(w.shape[1]) != input_size:
                    raise ValueError(
                        f"exl3 {self.prefix}[{i}]: dense in {w.shape[1]} != "
                        f"input_size {input_size}")
                out_first = rank * per_out
                self._geom.append({
                    "dense": True,
                    "n_out": per_out,
                    "out_first": out_first,
                    "out_last": out_first + per_out,
                    "weight": (per_out, input_size_per_partition),
                    "qkv_part": None,
                })
                continue
            g = self._pack_geometry(s["mq"])
            part = s["qkv_part"]
            if part is not None:
                # fused qkv: each slot covers its part's out-blocks
                qkv_off = sum(int(x) for x in output_partition_sizes[:part])
                part_full = int(output_partition_sizes[part])
                if qkv_off % 16 or part_full % 16:
                    raise ValueError(
                        f"exl3 {self.prefix}[{i}]: qkv part offset/size not "
                        f"16-aligned ({qkv_off}, {part_full})")
                out_first = rank * part_full
                out_last = out_first + part_full
                self._geom.append({
                    "dense": False,
                    "k": g["k"],
                    "mcg": g["mcg"],
                    "mul1": g["mul1"],
                    "n_out": part_full,
                    "out_first": out_first,
                    "out_last": out_last,
                    "qkv_part": part,
                    "qkv_off": qkv_off,
                    "trellis": (g["T1"], part_full // 16, g["T3"]),
                    "suh": (input_size,),
                    "svh": (part_full,),
                })
                continue
            if g["T1"] != input_size // 16:
                raise ValueError(
                    f"exl3 {self.prefix}[{i}]: trellis in-dim {g['T1']} != "
                    f"input_size/16 {input_size // 16}")
            if g["svh_len"] != per_out * world:
                raise ValueError(
                    f"exl3 {self.prefix}[{i}]: pack out {g['svh_len']} != "
                    f"per-rank {per_out} x tp {world} (disable_tp layers "
                    f"with tp>1 need the M3c pass)")
            if g["T2"] != g["svh_len"] // 16:
                raise ValueError(
                    f"exl3 {self.prefix}[{i}]: trellis out-dim {g['T2']} != "
                    f"out/16 {g['svh_len'] // 16}")
            if not self.is_row_parallel and per_out % 16 != 0:
                raise ValueError(
                    f"exl3 {self.prefix}[{i}]: per-rank out {per_out} is "
                    f"not 16-aligned; trellis slicing undefined")
            out_first = 0 if self.is_row_parallel else rank * per_out
            self._geom.append({
                "dense": False,
                "k": g["k"],
                "mcg": g["mcg"],
                "mul1": g["mul1"],
                "n_out": per_out,
                "out_first": out_first,
                "out_last": out_first + per_out,
                "qkv_part": None,
                "trellis": (
                    (input_size_per_partition // 16, g["T2"], g["T3"])
                    if self.is_row_parallel
                    else (g["T1"], per_out // 16, g["T3"])
                ),
                "suh": (
                    (input_size_per_partition,) if self.is_row_parallel
                    else (input_size,)
                ),
                "svh": (output_size,) if self.is_row_parallel else (per_out,),
            })

        # fused-qkv consistency: parts must cover the fused out exactly
        if self._qkv_split:
            parts = [g for g in self._geom if g.get("qkv_part") is not None]
            qkv = next(s["mq"] for s in self.shards if s.get("qkv_part") is not None)
            gq = self._pack_geometry(qkv)
            covered = sum(int(output_partition_sizes[p]) for p in range(3))
            if gq["T2"] * 16 != covered:
                raise ValueError(
                    f"exl3 {self.prefix}: fused qkv out {gq['T2'] * 16} != "
                    f"q+k+v shards {covered}")
            if gq["T1"] != input_size // 16:
                raise ValueError(
                    f"exl3 {self.prefix}: fused qkv in-dim {gq['T1']} != "
                    f"input_size/16 {input_size // 16}")
            for g in parts:
                if g["trellis"][1] != g["n_out"] // 16:
                    raise ValueError(
                        f"exl3 {self.prefix}: qkv part trellis/out mismatch")
            self._qkv_geom = gq

        def _slot(name: str, shape: tuple, dtype: torch.dtype, i: int):
            if name not in self._primary:
                self._primary[name] = i
            if self._primary[name] == i:
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
            if g["dense"]:
                _slot("weight", g["weight"], params_dtype, i)
                continue
            _slot("trellis", g["trellis"], torch.int16, i)
            _slot("suh", g["suh"], torch.float16, i)
            _slot("svh", g["svh"], torch.float16, i)
            if self.mul1_all:
                _slot("mul1", (), torch.int32, i)
        self._layer = layer
        layer.exl3_k = self._geom[0].get("k", 0)

    def _slot_tensor(self, name: str, i: int) -> torch.Tensor:
        layer = self._layer
        assert layer is not None
        if self._primary.get(name) == i:
            return getattr(layer, name).data
        idx = sum(
            1 for j, g in enumerate(self._geom)
            if j < i and self._has_suffix(g, name)
        )
        return self._extra[name][idx - 1]

    @staticmethod
    def _has_suffix(g: dict, name: str) -> bool:
        if name == "weight":
            return g["dense"]
        return not g["dense"]

    def _load_one(self, suffix: str, param: Parameter,
                  loaded_weight: torch.Tensor, *args: Any, **kwargs: Any) -> None:
        """Load one pack shard tensor into its slot, TP-sliced."""
        sid = kwargs.get("loaded_shard_id", args[0] if args else None)
        if (
            isinstance(sid, str)
            and sid in ("q", "k", "v")
            and self.n_shards == 3
            and not self._qkv_split
        ):
            # Stacked q/k/v pieces (DFlash2 draft fused qkv_proj) arrive with
            # string shard ids; map them onto the three EXL3 slots.
            sid = {"q": 0, "k": 1, "v": 2}[sid]
        if sid is None and self._qkv_split and suffix != "weight":
            self._load_qkv_fused(suffix, loaded_weight)
            return
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
        g = self._geom[sid]
        if suffix == "weight":
            if not g["dense"]:
                raise ValueError(
                    f"exl3 loader for {self.prefix}: weight into EXL3 slot {sid}")
            data = _narrow(loaded_weight, 0, g["out_first"], g["out_last"])
        else:
            if g["dense"]:
                raise ValueError(
                    f"exl3 loader for {self.prefix}: {suffix} into dense slot {sid}")
            data = self._slice_for_tp(g, suffix, loaded_weight)
        dst = self._slot_tensor(suffix, sid)
        if tuple(data.shape) != tuple(dst.shape):
            raise ValueError(
                f"exl3 loader shape mismatch for {self.prefix}.{suffix}[{sid}]: "
                f"pack gives {tuple(data.shape)}, slot holds {tuple(dst.shape)}"
            )
        dst.copy_(data)

    def _load_qkv_fused(self, suffix: str, fused: torch.Tensor) -> None:
        """Split the fused q|k|v tensor into the three EXL3 slots."""
        gq = self._qkv_geom
        if suffix == "trellis":
            if int(fused.shape[0]) != gq["T1"] or int(fused.shape[1]) != gq["T2"]:
                raise ValueError(
                    f"exl3 {self.prefix}: fused qkv trellis "
                    f"{tuple(fused.shape)} != pack {gq['T1']}x{gq['T2']}")
        for i, g in enumerate(self._geom):
            if g.get("qkv_part") is None:
                continue
            off = g["qkv_off"]
            if suffix == "trellis":
                data = _narrow(
                    fused, 1,
                    (off + g["out_first"]) // 16, (off + g["out_last"]) // 16,
                )
            elif suffix == "svh":
                data = _narrow(
                    fused, 0, off + g["out_first"], off + g["out_last"])
            elif suffix in ("suh", "mul1"):
                data = fused
            else:
                raise ValueError(
                    f"exl3 {self.prefix}: unknown qkv suffix {suffix!r}")
            dst = self._slot_tensor(suffix, i)
            if tuple(data.shape) != tuple(dst.shape):
                raise ValueError(
                    f"exl3 qkv split mismatch for {self.prefix}.{suffix}"
                    f"[{g['qkv_part']}]: {tuple(data.shape)} vs "
                    f"{tuple(dst.shape)}")
            dst.copy_(data)

    def apply(self, layer: torch.nn.Module,
              x: torch.Tensor,
              bias: torch.Tensor | None = None) -> torch.Tensor:
        ext = kernels.require()
        in_dtype = x.dtype
        x16 = x.to(torch.float16) if x.dtype == torch.bfloat16 else x
        parts = []
        for i, g in enumerate(self._geom):
            if g["dense"]:
                parts.append(torch.nn.functional.linear(
                    x, self._slot_tensor("weight", i)))
                continue
            y = ext.exl3_gemm(
                x16, self._slot_tensor("trellis", i),
                self._slot_tensor("suh", i), self._slot_tensor("svh", i),
                g["k"], g["mcg"], g["mul1"], g["n_out"])
            parts.append(y.to(in_dtype) if y.dtype != in_dtype else y)
        y = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
        b = bias if bias is not None else getattr(layer, "bias", None)
        if b is not None:
            y = y + b.to(y.dtype)
        if y.dtype != in_dtype:
            y = y.to(in_dtype)
        return y

    # -- internals -------------------------------------------------------

    def _slice_for_tp(self, g: dict, suffix: str, t: torch.Tensor) -> torch.Tensor:
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
    elif p == "language_model.lm_head":
        p = "lm_head"
    elif p.startswith("language_model.lm_head."):
        p = "lm_head" + p[len("language_model.lm_head"):]
    return p

"""EXL3 pack metadata: parse and validate the quantization_config.json contract.

This is pure logic (no torch, no vLLM). It is the ground truth for what the
loader may expect: every EXL3 pack we serve must pass ``validate_pack``.

Contract (from our packs, e.g. r0b0tlab Qwen3.8-27B EXL3 4.00bpw):

    {
      "quant_method": "exl3",
      "version": ...,
      "bits": 4.0, "head_bits": 6,
      "calibration": ..., "out_scales": ...,
      "codebook": ..., "vision_bits": ..., "mtp_bits": ...,
      "tensor_storage": {
         "<module name>": {
            "stored_tensors": {"<tensor name>": {"shape": [...], "n_bytes": N, "dtype": "..."}},
            "quant_format": "exl3",
            "bits_per_weight": 4,
            "mul1_multiplier": 2212286765
         }, ...
      }
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Bytes per element for the dtypes that appear in our packs.
_DTYPE_SIZES: dict[str, int] = {
    "torch.bfloat16": 2,
    "torch.float16": 2,
    "torch.float32": 4,
    "torch.int16": 2,
    "torch.int32": 4,
    "torch.int8": 1,
    "torch.uint8": 1,
}

# Supported EXL3 trellis bitrates (2-8, as produced by exllamav3 recipes).
MIN_BITS, MAX_BITS = 2.0, 8.0

# Tensor-name suffixes that make up a quantized EXL3 module.
_EXL3_TENSOR_SUFFIXES = (".trellis", ".suh", ".svh", ".mul1", ".mcg")

_CODEBOOK_MARKERS = ("mul1", "mcg")


class PackFormatError(ValueError):
    """Raised when a pack cannot be served (fail closed)."""


@dataclass(frozen=True)
class StoredTensor:
    name: str
    shape: tuple[int, ...]
    n_bytes: int
    dtype: str


@dataclass(frozen=True)
class ModuleQuant:
    module: str
    stored_tensors: dict[str, StoredTensor]
    quant_format: str
    bits_per_weight: float | None
    mul1_multiplier: int | None

    @property
    def is_exl3(self) -> bool:
        return self.quant_format == "exl3"

    @property
    def total_bytes(self) -> int:
        return sum(t.n_bytes for t in self.stored_tensors.values())

    def tensor(self, suffix: str) -> StoredTensor | None:
        for name, t in self.stored_tensors.items():
            if name.endswith(suffix):
                return t
        return None


@dataclass(frozen=True)
class Exl3Pack:
    quant_method: str
    version: Any
    bits: float | None
    head_bits: float | None
    codebook: Any
    vision_bits: Any
    mtp_bits: Any
    modules: dict[str, ModuleQuant] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def total_bytes(self) -> int:
        return sum(m.total_bytes for m in self.modules.values())

    def exl3_modules(self) -> dict[str, ModuleQuant]:
        return {k: m for k, m in self.modules.items() if m.is_exl3}


def load_pack(path: str | Path) -> Exl3Pack:
    """Load and structurally parse a quantization_config.json."""
    p = Path(path)
    if not p.is_file():
        raise PackFormatError(f"quantization config not found: {p}")
    with p.open() as fh:
        raw = json.load(fh)
    return parse_pack(raw)


def parse_pack(raw: dict[str, Any]) -> Exl3Pack:
    if not isinstance(raw, dict):
        raise PackFormatError("quantization config must be a JSON object")
    method = raw.get("quant_method")
    if method != "exl3":
        raise PackFormatError(f"quant_method must be 'exl3', got {method!r}")

    storage = raw.get("tensor_storage")
    if not isinstance(storage, dict) or not storage:
        raise PackFormatError("tensor_storage must be a non-empty object")

    modules: dict[str, ModuleQuant] = {}
    for module, entry in storage.items():
        if not isinstance(entry, dict):
            raise PackFormatError(f"[{module}] entry must be an object")
        tensors_raw = entry.get("stored_tensors")
        if not isinstance(tensors_raw, dict) or not tensors_raw:
            raise PackFormatError(f"[{module}] stored_tensors must be non-empty")
        tensors: dict[str, StoredTensor] = {}
        for tname, tinfo in tensors_raw.items():
            if not isinstance(tinfo, dict):
                raise PackFormatError(f"[{module}] {tname} info must be an object")
            shape = tinfo.get("shape")
            n_bytes = tinfo.get("n_bytes")
            dtype = tinfo.get("dtype")
            if not isinstance(shape, list) or not all(
                isinstance(d, int) for d in shape
            ):
                raise PackFormatError(f"[{module}] {tname} bad shape {shape!r}")
            if not isinstance(n_bytes, int) or n_bytes < 0:
                raise PackFormatError(f"[{module}] {tname} bad n_bytes {n_bytes!r}")
            if dtype not in _DTYPE_SIZES:
                raise PackFormatError(f"[{module}] {tname} unknown dtype {dtype!r}")
            tensors[tname] = StoredTensor(tname, tuple(shape), n_bytes, dtype)
        modules[module] = ModuleQuant(
            module=module,
            stored_tensors=tensors,
            quant_format=entry.get("quant_format", ""),
            bits_per_weight=entry.get("bits_per_weight"),
            mul1_multiplier=entry.get("mul1_multiplier"),
        )

    pack = Exl3Pack(
        quant_method=method,
        version=raw.get("version"),
        bits=raw.get("bits"),
        head_bits=raw.get("head_bits"),
        codebook=raw.get("codebook"),
        vision_bits=raw.get("vision_bits"),
        mtp_bits=raw.get("mtp_bits"),
        modules=modules,
        raw=raw,
    )
    validate_pack(pack)
    return pack


def _expected_n_bytes(shape: tuple[int, ...], dtype: str) -> int:
    n = 1
    for d in shape:
        n *= d
    elem = _DTYPE_SIZES[dtype]
    # bf16/fp16/float32 tensors are stored as-is; int16 trellis blocks and
    # int32 codebook markers likewise. No packing conversion happens at this
    # layer; n_bytes must equal shape product * element size.
    return n * elem


def validate_pack(pack: Exl3Pack) -> list[str]:
    """Fail-closed structural validation. Raises PackFormatError on any issue.

    Returns a list of warnings (non-fatal observations).
    """
    warnings: list[str] = []
    seen_exl3 = 0
    for module, m in pack.modules.items():
        if not m.is_exl3:
            continue
        seen_exl3 += 1
        b = m.bits_per_weight
        if b is None:
            raise PackFormatError(f"[{module}] exl3 module without bits_per_weight")
        if not (MIN_BITS <= float(b) <= MAX_BITS):
            raise PackFormatError(
                f"[{module}] bits_per_weight {b} outside [{MIN_BITS}, {MAX_BITS}]"
            )
        # Size consistency for every stored tensor.
        for name, t in m.stored_tensors.items():
            expected = _expected_n_bytes(t.shape, t.dtype)
            if expected != t.n_bytes:
                raise PackFormatError(
                    f"[{module}] {name}: n_bytes {t.n_bytes} != "
                    f"shape*elem {expected} ({t.shape}, {t.dtype})"
                )
        # A quantized module must carry trellis plus both Hadamard vectors,
        # and at least one codebook marker (mul1 or mcg).
        if m.tensor(".trellis") is None:
            raise PackFormatError(f"[{module}] exl3 module missing .trellis")
        if m.tensor(".suh") is None or m.tensor(".svh") is None:
            raise PackFormatError(f"[{module}] exl3 module missing .suh/.svh")
        markers = [mk for mk in _CODEBOOK_MARKERS if m.tensor("." + mk) is not None]
        if len(markers) != 1:
            raise PackFormatError(
                f"[{module}] exl3 module must carry exactly one of "
                f"{_CODEBOOK_MARKERS}, found {markers or 'none'}"
            )
        if markers == ["mul1"] and m.mul1_multiplier is None:
            raise PackFormatError(f"[{module}] mul1 module missing mul1_multiplier")
    if seen_exl3 == 0:
        raise PackFormatError("no exl3-format modules in pack")
    return warnings


def pack_stats(pack: Exl3Pack) -> dict[str, Any]:
    """Summary numbers used in build/serve receipts."""
    exl3 = pack.exl3_modules()
    bits = sorted(
        {float(b) for m in exl3.values() if (b := m.bits_per_weight) is not None}
    )
    return {
        "modules_total": len(pack.modules),
        "modules_exl3": len(exl3),
        "bits_per_weight_levels": bits,
        "total_bytes": pack.total_bytes,
        "head_bits": pack.head_bits,
        "vision_bits": pack.vision_bits,
        "mtp_bits": pack.mtp_bits,
    }

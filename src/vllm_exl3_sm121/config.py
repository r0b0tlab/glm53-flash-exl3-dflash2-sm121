"""Exl3Config: vLLM QuantizationConfig for EXL3 packs.

Requires vLLM installed (pinned in runtime.lock.json). Imported only from the
plugin registration path, never from pure-logic tests.

ON-CLUSTER TODO (verify against pinned vLLM v0.30.0rc1 a00a3544b93e):
- exact import path of QuantizationConfig / QuantizeMethodBase
  (expected: vllm.model_executor.layers.quantization.base_config)
- register_quantization_config availability in the quantization package
- signature of get_quant_method(layer, prefix) and the FusedMoE layer type
- weight-loader hooks used by Exl3LinearMethod / Exl3MoEMethod
Verify by reading the pinned source in the cluster venv; adjust here and
record the diff in docs/provenance.md.
"""

from __future__ import annotations

from typing import Any

import json
import os

import torch

from vllm.model_executor.layers.quantization.base_config import (  # type: ignore
    QuantizationConfig,
    QuantizeMethodBase,
)

from . import metadata as meta


class Exl3Config(QuantizationConfig):
    """Config for EXL3 checkpoints (pack contract in metadata.py).

    One Exl3Config instance is constructed from a model directory's
    quantization_config.json; the pack metadata is fully validated before any
    weight is touched.
    """

    def __init__(self, pack: meta.Exl3Pack) -> None:
        super().__init__()
        self.pack = pack

    # -- vLLM QuantizationConfig interface -------------------------------

    def get_name(self) -> str:
        return "exl3"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        # EXL3 kernels operate on fp16 activation tiles; bf16 models are cast
        # at the linear boundary and restored by the caller.
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        # sm_80+ for the vendored kernels; GB10 is sm_121.
        return 80

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return ["quantization_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Exl3Config":
        return cls(meta.parse_pack(config))

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        """Dispatch by module kind and pack metadata.

        - `visual.*` (vision tower, vb16 contract): None -> dense load.
        - LinearBase (attention + dense MLP): Exl3LinearMethod.
        - FusedMoE / VocabParallelEmbedding: fail closed (M3b).
        """
        from vllm.model_executor.layers.linear import LinearBase

        if prefix.startswith("visual.") or ".visual." in prefix:
            return None
        if isinstance(layer, LinearBase):
            from .linear import Exl3LinearMethod
            try:
                return Exl3LinearMethod(self, prefix)
            except ValueError:
                # no EXL3 entry for this linear (norms-adjacent, router,
                # unquantized head): dense load, not an error.
                return None
        layer_kind = type(layer).__name__
        if "Moe" in layer_kind or "MoE" in layer_kind or "Expert" in layer_kind:
            raise NotImplementedError(
                f"exl3 MoE method for {prefix} ({layer_kind}) lands in M3b; "
                f"dense path is wired, MoE kernel stays fail-closed")
        if "Embedding" in layer_kind:
            raise NotImplementedError(
                f"exl3 embedding method for {prefix} lands in M3b")
        return None

    # -- helpers ---------------------------------------------------------

    @classmethod
    def load_from_model_dir(cls, model_dir: str) -> "Exl3Config":
        cfg = os.path.join(model_dir, "quantization_config.json")
        with open(cfg) as fh:
            return cls(meta.parse_pack(json.load(fh)))

    def module_quant(self, prefix: str) -> meta.ModuleQuant | None:
        """Normalize a vLLM prefix to a pack entry (longest-prefix match)."""
        best: meta.ModuleQuant | None = None
        best_len = -1
        for name, mq in self.pack.modules.items():
            if prefix == name or prefix.startswith(name + ".") and len(name) > best_len:
                best, best_len = mq, len(name)
        return best

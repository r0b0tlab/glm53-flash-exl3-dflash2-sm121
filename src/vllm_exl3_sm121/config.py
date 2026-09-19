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

    def __init__(self, pack: meta.Exl3Pack | None,
                 inline: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.pack = pack
        self.inline = inline or {}

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
        # File form (quantization_config.json): full pack parse now.
        # Inline form (config.json's quantization_config summary): the pack
        # resolves lazily in maybe_update_config, where the model path is
        # known (vLLM feeds from_config the inline summary, never the file).
        if isinstance(config.get("tensor_storage"), dict):
            return cls(meta.parse_pack(config))
        return cls(None, dict(config))

    def maybe_update_config(self, model_name: str,
                            hf_config: Any = None,
                            revision: str | None = None) -> None:
        if self.pack is not None:
            return
        cand = os.path.join(model_name, "quantization_config.json")
        if not os.path.isfile(cand):
            raise ValueError(
                f"exl3 needs {cand} (inline config.json summary has no "
                f"tensor_storage); point --model at the pack directory")
        with open(cand) as fh:
            self.pack = meta.parse_pack(json.load(fh))

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        """Dispatch by module kind and pack metadata.

        - LinearBase with pack entry: Exl3LinearMethod.
        - LinearBase without pack entry (router, KDA helpers, vision
          tower): explicit UnquantizedLinearMethod + warning (vLLM
          forbids None; silence would risk wrong numerics).
        - ParallelLMHead (VocabParallelEmbedding that passes a quant_config):
          Exl3LinearMethod when the pack quantized the head (the logits
          processor calls `quant_method.apply`), else
          UnquantizedEmbeddingMethod.
        - FusedMoE / RoutedExperts: Exl3MoEMethod (M3b).
        """
        from vllm.model_executor.layers.linear import LinearBase

        if isinstance(layer, LinearBase):
            from .linear import Exl3LinearMethod
            try:
                return Exl3LinearMethod(self, prefix)
            except ValueError:
                # No EXL3 entry: genuinely unquantized linear (router,
                # KDA helpers, vision tower dense weights). vLLM forbids
                # None here, so resolve explicit dense and say so loudly.
                import logging
                logging.getLogger("vllm_exl3_sm121").warning(
                    "exl3: no pack entry for %s — dense load", prefix)
                from vllm.model_executor.layers.linear import (
                    UnquantizedLinearMethod,
                )
                return UnquantizedLinearMethod()
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            UnquantizedEmbeddingMethod,
            VocabParallelEmbedding,
        )
        if isinstance(layer, VocabParallelEmbedding):
            from .linear import Exl3LinearMethod
            try:
                return Exl3LinearMethod(self, prefix)
            except ValueError:
                return UnquantizedEmbeddingMethod()
        layer_kind = type(layer).__name__
        if "Moe" in layer_kind or "MoE" in layer_kind or "Expert" in layer_kind \
                or "RoutedExperts" in layer_kind:
            from .moe import Exl3MoEMethod
            moe_cfg = getattr(layer, "moe_config", None)
            if moe_cfg is None:
                raise NotImplementedError(
                    f"exl3 MoE for {prefix}: no moe_config on {layer_kind}")
            return Exl3MoEMethod(self, moe_cfg, prefix)
        return None

    # -- helpers ---------------------------------------------------------

    @classmethod
    def load_from_model_dir(cls, model_dir: str) -> "Exl3Config":
        cfg = os.path.join(model_dir, "quantization_config.json")
        with open(cfg) as fh:
            return cls(meta.parse_pack(json.load(fh)))

    def module_quant(self, prefix: str) -> meta.ModuleQuant | None:
        """Normalize a vLLM prefix to a pack entry (longest-prefix match)."""
        if self.pack is None:
            return None
        best: meta.ModuleQuant | None = None
        best_len = -1
        for name, mq in self.pack.modules.items():
            if prefix == name or prefix.startswith(name + ".") and len(name) > best_len:
                best, best_len = mq, len(name)
        return best

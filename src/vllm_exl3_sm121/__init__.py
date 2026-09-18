"""vllm_exl3_sm121 - EXL3 quantization plugin for our SM121-optimized vLLM fork.

Registers the "exl3" quantization method with vLLM (out-of-tree plugin) and
exposes the pure-logic modules used to validate EXL3 packs and plan GB10
placement before any GPU work happens.

Import safety: this module must import cleanly on machines without vLLM or a
GPU (pure-logic unit tests run in CI). vLLM imports are guarded.
"""

from __future__ import annotations

import logging

__version__ = "0.1.0"

logger = logging.getLogger("vllm_exl3_sm121")


def register() -> None:
    """Entry point for vLLM's general plugin mechanism.

    Called by vLLM at startup when the ``vllm.general_plugins`` entry point is
    installed (see pyproject.toml). Registers Exl3Config with vLLM's
    quantization registry.
    """
    try:
        from vllm.model_executor.layers.quantization import (  # type: ignore
            register_quantization_config,
        )
    except Exception as exc:  # pragma: no cover - exercised only with vllm installed
        raise RuntimeError(
            "vllm_exl3_sm121.register() requires a vLLM installation "
            "(pinned in runtime.lock.json). Install the plugin inside the "
            "cluster venv, not on a bare workstation."
        ) from exc

    from .config import Exl3Config

    register_quantization_config(Exl3Config)
    logger.info(
        "vllm_exl3_sm121 %s registered 'exl3' quantization config", __version__
    )

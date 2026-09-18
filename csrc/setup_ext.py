#!/usr/bin/env python3
"""Build vllm_exl3_sm121_ext on the cluster (aarch64, CUDA 13, sm_120/121).

Usage (inside the cluster venv, from the repo root):
    TORCH_CUDA_ARCH_LIST="12.0;12.1" python csrc/setup_ext.py build_ext --inplace

Today this compiles csrc/bindings.cpp alone (real toolchain validation; ops
fail closed until wired). Milestone M2 adds the vendored translation units to
``sources`` — see csrc/bindings.cpp header for the wiring plan.
"""

from __future__ import annotations

import os
from pathlib import Path

from setuptools import setup

try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "setup_ext.py needs torch with CUDA support installed "
        "(runtime.lock.json pins torch 2.13.0+cu130)"
    ) from exc

ROOT = Path(__file__).resolve().parent.parent
VENDORED = ROOT / "vendored" / "exllamav3" / "ext"

sources = [str(ROOT / "csrc" / "bindings.cpp")]
# Milestone M2 (uncomment + wire, one group at a time, with receipts):
# sources += [
#     str(VENDORED / "quant" / "exl3_gemv.cu"),
#     str(VENDORED / "quant" / "exl3_gemm.cu"),
#     str(VENDORED / "quant" / "exl3_moe.cu"),
#     str(VENDORED / "quant" / "reconstruct.cu"),
#     str(VENDORED / "hadamard.cu"),
#     str(VENDORED / "hgemm.cu"),
# ]

arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
if "12." not in arch_list:
    raise SystemExit(
        "Set TORCH_CUDA_ARCH_LIST='12.0;12.1' (GB10 is sm_121; sm_120 binary-"
        "compatible). Refusing to build with unexpected arches: %r" % arch_list
    )

setup(
    name="vllm_exl3_sm121_ext",
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            name="vllm_exl3_sm121_ext",
            sources=sources,
            include_dirs=[str(VENDORED)],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "--use_fast_math"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)

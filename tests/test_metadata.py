"""Metadata contract tests using our real packs (trimmed fixtures)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from vllm_exl3_sm121 import metadata as meta

FX = Path(__file__).parent / "fixtures"


def _load_json(name: str) -> dict:
    with open(FX / name) as fh:
        return json.load(fh)


@pytest.mark.parametrize(
    "fixture,bits,head_bits",
    [
        ("qwen38-27b-exl3.trimmed.json", 4.0, 6),
        ("dflash2-exl3.trimmed.json", 4.0, 6),
    ],
)
def test_load_real_fixtures(fixture: str, bits: float, head_bits: int) -> None:
    pack = meta.parse_pack(_load_json(fixture))
    assert pack.bits == bits
    assert pack.head_bits == head_bits
    stats = meta.pack_stats(pack)
    assert stats["modules_exl3"] >= 1
    assert stats["total_bytes"] > 0
    for b in stats["bits_per_weight_levels"]:
        assert meta.MIN_BITS <= b <= meta.MAX_BITS


def test_qwen38_passthrough_modules_are_not_exl3() -> None:
    pack = meta.parse_pack(_load_json("qwen38-27b-exl3.trimmed.json"))
    embed = pack.modules["model.language_model.embed_tokens"]
    assert not embed.is_exl3
    assert embed.tensor(".weight") is not None


def _mutate(fixture: str, fn) -> dict:
    raw = copy.deepcopy(_load_json(fixture))
    fn(raw)
    return raw


def test_bad_quant_method() -> None:
    raw = _mutate("qwen38-27b-exl3.trimmed.json", lambda r: r.update(quant_method="awq"))
    with pytest.raises(meta.PackFormatError):
        meta.parse_pack(raw)


def test_empty_tensor_storage() -> None:
    raw = _mutate("qwen38-27b-exl3.trimmed.json", lambda r: r.update(tensor_storage={}))
    with pytest.raises(meta.PackFormatError):
        meta.parse_pack(raw)


def test_missing_trellis_fails_closed() -> None:
    def strip(r: dict) -> None:
        for module in r["tensor_storage"].values():
            if module.get("quant_format") == "exl3":
                module["stored_tensors"] = {
                    k: v
                    for k, v in module["stored_tensors"].items()
                    if not k.endswith(".trellis")
                }
                return
        raise AssertionError("no exl3 module in fixture")

    with pytest.raises(meta.PackFormatError, match="missing .trellis"):
        meta.parse_pack(_mutate("qwen38-27b-exl3.trimmed.json", strip))


def test_both_codebook_markers_fail_closed() -> None:
    def add_mcg(r: dict) -> None:
        for module in r["tensor_storage"].values():
            if module.get("quant_format") == "exl3":
                st = module["stored_tensors"]
                trellis_name = next(k for k in st if k.endswith(".trellis"))
                st[trellis_name[: -len(".trellis")] + ".mcg"] = {
                    "shape": [],
                    "n_bytes": 4,
                    "dtype": "torch.int32",
                }
                return
        raise AssertionError("no exl3 module in fixture")

    with pytest.raises(meta.PackFormatError, match="exactly one"):
        meta.parse_pack(_mutate("qwen38-27b-exl3.trimmed.json", add_mcg))


def test_size_mismatch_fails_closed() -> None:
    def corrupt(r: dict) -> None:
        module = r["tensor_storage"]["model.language_model.layers.0.linear_attn.in_proj_qkv"]
        first = next(iter(module["stored_tensors"]))
        module["stored_tensors"][first]["n_bytes"] += 1

    with pytest.raises(meta.PackFormatError, match="n_bytes"):
        meta.parse_pack(_mutate("qwen38-27b-exl3.trimmed.json", corrupt))


def test_unknown_dtype_fails_closed() -> None:
    def corrupt(r: dict) -> None:
        module = r["tensor_storage"]["model.language_model.layers.0.linear_attn.in_proj_qkv"]
        first = next(iter(module["stored_tensors"]))
        module["stored_tensors"][first]["dtype"] = "torch.float8_e4m3fn"

    with pytest.raises(meta.PackFormatError, match="unknown dtype"):
        meta.parse_pack(_mutate("qwen38-27b-exl3.trimmed.json", corrupt))


def test_bits_out_of_range_fails_closed() -> None:
    def corrupt(r: dict) -> None:
        for module in r["tensor_storage"].values():
            if module.get("quant_format") == "exl3":
                module["bits_per_weight"] = 16
                return

    with pytest.raises(meta.PackFormatError, match="outside"):
        meta.parse_pack(_mutate("qwen38-27b-exl3.trimmed.json", corrupt))

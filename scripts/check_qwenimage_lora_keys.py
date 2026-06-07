#!/usr/bin/env python3
"""Qwen-Image LoRA/sampler contract guard against OneTrainer.

This is intentionally non-heavy: it does not import torch, diffusers, or Mojo.
It checks:
  * OneTrainer Qwen LoRA wrapper prefixes and saved key suffix conventions.
  * The optional saved safetensors header inventory, without reading tensor data.
  * The Mojo Qwen slot inventory and sampler prompt-shape constants.

Default mode validates the current Mojo save-key scaffold. Use --strict-port to
require the explicit OneTrainer/Qwen raw safetensors save path.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from pathlib import Path


REPO = Path("/home/alex/mojodiffusion")
OT_ROOT = Path("/home/alex/OneTrainer")
OT_QWEN_LORA = OT_ROOT / "output/qwen_100step_baseline/lora.safetensors"

NUM_BLOCKS = 60
RANK = 16
D_MODEL = 3072
F_MLP = 12288

OT_MODULES = [
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "img_mlp.net.0.proj",
    "img_mlp.net.2",
    "attn.add_q_proj",
    "attn.add_k_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
    "txt_mlp.net.0.proj",
    "txt_mlp.net.2",
]

OT_SUFFIXES = [
    "alpha",
    "lora_down.weight",
    "lora_up.weight",
]


def die(msg: str) -> None:
    raise SystemExit(f"[qwen-lora-keys] FAIL: {msg}")


def read_text(path: Path) -> str:
    if not path.exists():
        die(f"missing source file: {path}")
    return path.read_text(encoding="utf-8")


def require(text: str, needle: str, label: str) -> None:
    if needle not in text:
        die(f"missing {label}: {needle}")


def ot_key(block: int, module: str, suffix: str) -> str:
    return f"transformer.transformer_blocks.{block}.{module}.{suffix}"


def expected_shape(module: str, suffix: str) -> list[int]:
    if suffix == "alpha":
        return []
    if module in {
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "attn.to_out.0",
        "attn.add_q_proj",
        "attn.add_k_proj",
        "attn.add_v_proj",
        "attn.to_add_out",
    }:
        in_f = D_MODEL
        out_f = D_MODEL
    elif module in {"img_mlp.net.0.proj", "txt_mlp.net.0.proj"}:
        in_f = D_MODEL
        out_f = F_MLP
    elif module in {"img_mlp.net.2", "txt_mlp.net.2"}:
        in_f = F_MLP
        out_f = D_MODEL
    else:
        die(f"unhandled Qwen module shape: {module}")
    if suffix == "lora_down.weight":
        return [RANK, in_f]
    if suffix == "lora_up.weight":
        return [out_f, RANK]
    die(f"unhandled suffix shape: {suffix}")


def read_safetensors_header(path: Path) -> dict:
    if not path.exists():
        die(f"missing safetensors file: {path}")
    with path.open("rb") as fh:
        raw_len = fh.read(8)
        if len(raw_len) != 8:
            die(f"truncated safetensors header length: {path}")
        (header_len,) = struct.unpack("<Q", raw_len)
        header_raw = fh.read(header_len)
        if len(header_raw) != header_len:
            die(f"truncated safetensors header: {path}")
    return json.loads(header_raw)


def check_onetrainer_source(ot_root: Path) -> None:
    qwen_setup = read_text(ot_root / "modules/modelSetup/QwenLoRASetup.py")
    lora_module = read_text(ot_root / "modules/module/LoRAModule.py")
    qwen_model = read_text(ot_root / "modules/model/QwenModel.py")
    qwen_sampler = read_text(ot_root / "modules/modelSampler/QwenSampler.py")

    require(qwen_setup, 'model.transformer, "transformer"', "OT transformer LoRA prefix")
    require(qwen_setup, 'model.text_encoder, "text_encoder"', "OT text encoder LoRA prefix")
    require(qwen_setup, "config.layer_filter.split(\",\")", "OT layer-filter split")
    require(lora_module, "module.state_dict(prefix=module.prefix)", "OT state_dict prefix")
    require(lora_module, "lora_down.weight", "OT LoRA down suffix")
    require(lora_module, "lora_up.weight", "OT LoRA up suffix")
    require(lora_module, "self.register_buffer(\"alpha\"", "OT alpha buffer")
    require(qwen_model, "DEFAULT_PROMPT_TEMPLATE_CROP_START = 34", "OT prompt crop")
    require(qwen_model, "PROMPT_MAX_LENGTH = 512", "OT prompt max length")
    require(qwen_sampler, "batch_size = 2 if cfg_scale > 1.0 else 1", "OT CFG batch rule")
    require(qwen_sampler, "noise_scheduler.set_timesteps(diffusion_steps", "OT sampler timesteps")
    require(qwen_sampler, "mu=math.log(shift)", "OT Qwen shift")
    print("[qwen-lora-keys] OneTrainer source contract: PASS")


def check_ot_saved_header(path: Path) -> None:
    header = read_safetensors_header(path)
    keys = [k for k in header if k != "__metadata__"]
    expected_count = NUM_BLOCKS * len(OT_MODULES) * len(OT_SUFFIXES)
    if len(keys) != expected_count:
        die(f"{path} has {len(keys)} LoRA tensors, expected {expected_count}")

    for block in range(NUM_BLOCKS):
        for module in OT_MODULES:
            for suffix in OT_SUFFIXES:
                key = ot_key(block, module, suffix)
                if key not in header:
                    die(f"missing OT saved key: {key}")
                info = header[key]
                shape = info.get("shape")
                if shape != expected_shape(module, suffix):
                    die(f"{key} shape {shape}, expected {expected_shape(module, suffix)}")
                dtype = info.get("dtype")
                if dtype != "BF16":
                    die(f"{key} dtype {dtype}, expected BF16 for OT Qwen baseline")

    seen_blocks = sorted(
        {
            int(m.group(1))
            for k in keys
            if (m := re.match(r"transformer\.transformer_blocks\.(\d+)\.", k))
        }
    )
    if seen_blocks != list(range(NUM_BLOCKS)):
        die(f"block inventory mismatch: {seen_blocks[:5]}...{seen_blocks[-5:]}")
    print(
        "[qwen-lora-keys] OneTrainer saved inventory: PASS "
        f"blocks={NUM_BLOCKS} modules/block={len(OT_MODULES)} tensors={len(keys)}"
    )


def check_mojo_sources(repo: Path, strict_port: bool) -> None:
    stack = read_text(repo / "serenitymojo/models/qwenimage/qwenimage_stack_lora.mojo")
    lora_save = read_text(repo / "serenitymojo/training/lora_save.mojo")
    pipe512 = read_text(repo / "serenitymojo/pipeline/qwenimage_pipeline_512_multistep.mojo")
    pipe1024 = read_text(repo / "serenitymojo/pipeline/qwenimage_pipeline_1024_multistep.mojo")

    require(stack, "comptime DBL_SLOTS = 12", "Mojo Qwen slot count")
    for module in OT_MODULES:
        require(stack, f'String("{module}")', f"Mojo Qwen slot suffix {module}")
    require(stack, "save_qwen_lora", "Mojo Qwen save hook")
    require(stack, "save_lora_onetrainer", "Mojo Qwen OneTrainer save call")
    require(stack, 'String("transformer.transformer_blocks.")', "Mojo Qwen OT wrapper prefix")

    require(lora_save, "def save_lora_onetrainer", "Mojo OneTrainer saver")
    require(lora_save, ".lora_down.weight", "Mojo OT down suffix")
    require(lora_save, ".lora_up.weight", "Mojo OT up suffix")
    require(lora_save, ".alpha", "Mojo OT alpha suffix")
    require(lora_save, "_bf16_scalar", "Mojo OT BF16 alpha scalar helper")
    require(lora_save, "_bf16_2d(a.a.copy(), a.rank, a.in_f, ctx)", "Mojo OT BF16 A tensor")
    require(lora_save, "_bf16_2d(a.b.copy(), a.out_f, a.rank, ctx)", "Mojo OT BF16 B tensor")

    for label, src in [("512", pipe512), ("1024", pipe1024)]:
        require(src, "comptime DROP_IDX = 34", f"Mojo Qwen {label} drop index")
        require(src, "comptime N_TXT_KEPT = 512", f"Mojo Qwen {label} text cap")
        require(src, "comptime PAD_ID = 151643", f"Mojo Qwen {label} pad id")
        require(src, "Scheduler.qwen", f"Mojo Qwen {label} scheduler")
        require(src, "cfg_qwen", f"Mojo Qwen {label} CFG helper")

    gaps: list[str] = []
    if 'String("transformer_blocks.")' in stack and 'String("transformer.transformer_blocks.")' not in stack:
        gaps.append("Mojo Qwen save prefix lacks OneTrainer wrapper 'transformer.'")
    if "save_lora_peft(named^, path, ctx)" in stack:
        gaps.append("Mojo Qwen stack delegates to generic PEFT lora_A/lora_B naming, not OT lora_down/lora_up")
    if ".lora_down.weight" not in lora_save or ".lora_up.weight" not in lora_save:
        gaps.append("Mojo OneTrainer saver does not emit OT lora_down/lora_up tensors")
    if ".alpha" not in lora_save or "_bf16_scalar" not in lora_save:
        gaps.append("Mojo OneTrainer saver does not emit OT per-module BF16 alpha tensors")

    if gaps:
        for gap in gaps:
            print(f"[qwen-lora-keys] WARN: {gap}")
        if strict_port:
            die("strict port requested and Mojo Qwen save naming is not OneTrainer-identical")

    print("[qwen-lora-keys] Mojo source scaffold: PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--ot-root", type=Path, default=OT_ROOT)
    parser.add_argument(
        "--safetensors",
        type=Path,
        default=OT_QWEN_LORA if OT_QWEN_LORA.exists() else None,
        help="optional OneTrainer Qwen LoRA safetensors to inventory",
    )
    parser.add_argument(
        "--strict-port",
        action="store_true",
        help="fail if Mojo Qwen save naming is not OneTrainer-identical yet",
    )
    args = parser.parse_args()

    check_onetrainer_source(args.ot_root)
    if args.safetensors is not None:
        check_ot_saved_header(args.safetensors)
    else:
        print("[qwen-lora-keys] OneTrainer saved inventory: SKIP (no --safetensors)")
    check_mojo_sources(args.repo, args.strict_port)
    print("[qwen-lora-keys] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

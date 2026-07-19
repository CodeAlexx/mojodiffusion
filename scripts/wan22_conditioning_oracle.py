#!/usr/bin/env python3
"""Pinned official Wan 2.2 UMT5 conditioning oracle (development only)."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer, UMT5EncoderModel


HF_REVISION = "b8fff7315c768468a5333511427288870b2e9635"
ORACLE_REVISION = "42bf4cfaa384bc21833865abc2f9e6c0e67233dc"
SNAPSHOT = Path(
    "/home/alex/.cache/huggingface/hub/"
    "models--Wan-AI--Wan2.2-TI2V-5B-Diffusers/snapshots/" + HF_REVISION
)
ORACLE_ROOT = Path("/home/alex/Wan2.2")
DEFAULT_NEGATIVE = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
    "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check_pins() -> None:
    if SNAPSHOT.name != HF_REVISION or not SNAPSHOT.is_dir():
        raise RuntimeError(f"missing pinned HF snapshot: {SNAPSHOT}")
    head = subprocess.check_output(
        ["git", "-C", str(ORACLE_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(ORACLE_ROOT), "status", "--porcelain"], text=True
    ).strip()
    if head != ORACLE_REVISION or dirty:
        raise RuntimeError(
            f"Wan oracle checkout must be clean at {ORACLE_REVISION}; "
            f"head={head!r} dirty={bool(dirty)}"
        )


def encode(tokenizer, model, text: str) -> tuple[torch.Tensor, int]:
    tokens = tokenizer(
        text,
        padding="max_length",
        max_length=512,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    ids = tokens.input_ids.to("cuda")
    mask = tokens.attention_mask.to("cuda")
    valid = int(mask.sum().item())
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = model(input_ids=ids, attention_mask=mask).last_hidden_state
    # Creator returns valid rows; the product cache contract pads those rows to
    # 512 with exact zero so Wan cross-attention can remain unmasked.
    hidden = (hidden * mask.unsqueeze(-1)).to(torch.bfloat16).cpu().contiguous()
    return hidden, valid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative", default=DEFAULT_NEGATIVE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    check_pins()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        SNAPSHOT / "tokenizer", local_files_only=True
    )
    model = UMT5EncoderModel.from_pretrained(
        SNAPSHOT / "text_encoder",
        local_files_only=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).eval().to("cuda")
    loaded = time.perf_counter()
    pos, pos_len = encode(tokenizer, model, args.prompt)
    pos_done = time.perf_counter()
    neg, neg_len = encode(tokenizer, model, args.negative)
    torch.cuda.synchronize()
    finished = time.perf_counter()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "pos_embed": pos,
            "neg_embed": neg,
            "pos_len": torch.tensor([pos_len], dtype=torch.float32),
            "neg_len": torch.tensor([neg_len], dtype=torch.float32),
        },
        str(args.output),
        metadata={
            "schema": "serenity.wan22.conditioning_oracle.v1",
            "hf_revision": HF_REVISION,
            "oracle_revision": ORACLE_REVISION,
        },
    )
    manifest = {
        "schema": "serenity.wan22.conditioning_oracle.v1",
        "hf_revision": HF_REVISION,
        "oracle_revision": ORACLE_REVISION,
        "prompt": args.prompt,
        "negative_prompt": args.negative,
        "pos_len": pos_len,
        "neg_len": neg_len,
        "dtype": "BF16",
        "shape": [1, 512, 4096],
        "load_seconds": loaded - start,
        "positive_seconds": pos_done - loaded,
        "negative_seconds": finished - pos_done,
        "total_seconds": finished - start,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1048576,
        "output": str(args.output),
        "output_sha256": digest(args.output),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

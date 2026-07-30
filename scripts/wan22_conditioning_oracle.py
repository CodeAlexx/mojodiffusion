#!/usr/bin/env python3
"""Pinned official Wan 2.2 UMT5 conditioning oracle (development only)."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import save_file


HF_REVISION = "installed-official-native"
ORACLE_REVISION = "42bf4cfaa384bc21833865abc2f9e6c0e67233dc"
ORACLE_ROOT = Path("/home/alex/Wan2.2")
MODEL_ROOT = Path(
    "/home/alex/.serenity/models/checkpoints/Wan2.2-TI2V-5B"
)
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
    required = (
        MODEL_ROOT / "models_t5_umt5-xxl-enc-bf16.pth",
        MODEL_ROOT / "google/umt5-xxl/tokenizer.json",
        MODEL_ROOT / "google/umt5-xxl/spiece.model",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing official Wan UMT5 assets: {missing}")


def encode(model, text: str) -> tuple[torch.Tensor, int]:
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        valid_hidden = model([text], torch.device("cuda"))[0]
    valid = int(valid_hidden.shape[0])
    if valid > 512:
        raise RuntimeError(f"creator returned {valid} tokens, expected <= 512")
    # The creator returns only valid rows. The product contract zero-pads them
    # back to [1,512,4096] because Wan cross-attention is unmasked.
    hidden = torch.zeros((1, 512, 4096), dtype=torch.bfloat16)
    hidden[0, :valid] = valid_hidden.to(torch.bfloat16).cpu()
    return hidden.contiguous(), valid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative", default=DEFAULT_NEGATIVE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    check_pins()

    sys.path.insert(0, str(ORACLE_ROOT))
    from wan.modules.t5 import T5EncoderModel

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    # Construct BF16 on CPU first. The creator helper's default GPU construction
    # transiently creates an F32 model and exceeds a 24 GB card before loading.
    model = T5EncoderModel(
        text_len=512,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        checkpoint_path=str(
            MODEL_ROOT / "models_t5_umt5-xxl-enc-bf16.pth"
        ),
        tokenizer_path=str(MODEL_ROOT / "google/umt5-xxl"),
    )
    model.model.to("cuda")
    loaded = time.perf_counter()
    pos, pos_len = encode(model, args.prompt)
    pos_done = time.perf_counter()
    neg, neg_len = encode(model, args.negative)
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
            "oracle_impl": "wan.modules.t5.T5EncoderModel",
        },
    )
    manifest = {
        "schema": "serenity.wan22.conditioning_oracle.v1",
        "hf_revision": HF_REVISION,
        "oracle_revision": ORACLE_REVISION,
        "oracle_impl": "wan.modules.t5.T5EncoderModel",
        "model_root": str(MODEL_ROOT),
        "umt5_sha256": digest(
            MODEL_ROOT / "models_t5_umt5-xxl-enc-bf16.pth"
        ),
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

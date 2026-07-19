#!/usr/bin/env python3
"""Encode a prompt to the LTX-2 PRE-connector gemma hidden (pipeline conds).

Produces the exact tensor the Mojo HQ pipeline's cached-context files carry:
    text_hidden BF16 [1, 1024, 4096]
i.e. gemma output AFTER the LTX-2.3 caption projection but BEFORE the
Embeddings1DConnector (the Mojo pipeline runs its own parity-gated video+audio
connectors on this). Mirrors musubi's TE construction (ltx2_cache_text_encoder_
outputs.py) and captures VideoGemmaTextEncoderModel._preprocess_text output —
the seam feeding _run_connector.

Usage:
  .venv/bin/python ltx2_encode_prompt_hidden.py <prompt-or-@file> <out.safetensors> \
      [--ltx2_checkpoint P] [--gemma_safetensors P] [--padding_side left|right]

Run from /home/alex/musubi-tuner with its venv.
"""
import argparse
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, "/home/alex/musubi-tuner/src")

from musubi_tuner.ltx_2.loader.single_gpu_model_builder import SingleGPUModelBuilder
from musubi_tuner.ltx_2.text_encoders.gemma.encoders.base_encoder import module_ops_from_gemma_root
from musubi_tuner.ltx_2.text_encoders.gemma.encoders.video_only_encoder import (
    VIDEO_ONLY_GEMMA_TEXT_ENCODER_KEY_OPS,
    VideoGemmaTextEncoderModelConfigurator,
)

DEF_CKPT = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-dev-fp8.safetensors"
DEF_GEMMA = "/home/alex/.serenity/models/text_encoders/gemma-3-12b-it-fp8/gemma_3_12B_it_fp8_e4m3fn.safetensors"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("out")
    ap.add_argument("--ltx2_checkpoint", default=DEF_CKPT)
    ap.add_argument("--gemma_safetensors", default=DEF_GEMMA)
    ap.add_argument("--padding_side", default="left")
    ap.add_argument(
        "--audio_from",
        default="/home/alex/EriDiffusion/inference-flame/output/audio_context_dump/ltx2_audio_context.safetensors",
    )
    args = ap.parse_args()

    prompt = args.prompt
    if prompt.startswith("@"):
        prompt = open(prompt[1:]).read().strip()

    device = torch.device("cuda")
    dtype = torch.bfloat16

    text_encoder = SingleGPUModelBuilder(
        model_path=str(args.ltx2_checkpoint),
        model_class_configurator=VideoGemmaTextEncoderModelConfigurator,
        model_sd_ops=VIDEO_ONLY_GEMMA_TEXT_ENCODER_KEY_OPS,
        module_ops=module_ops_from_gemma_root(
            None,
            gemma_safetensors=args.gemma_safetensors,
            torch_dtype=dtype,
            load_in_8bit=False,
            load_in_4bit=False,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        ),
    ).build(device=device, dtype=dtype)
    text_encoder.eval()

    with torch.no_grad():
        encoded, attention_mask = text_encoder._preprocess_text(prompt, args.padding_side)
        if isinstance(encoded, tuple):
            encoded = encoded[0]
        hidden = encoded
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)
        print(f"pre-connector hidden: {tuple(hidden.shape)} dtype={hidden.dtype} "
              f"std={hidden.float().std().item():.4f}")
        if isinstance(attention_mask, torch.Tensor):
            print(f"attention_mask sum: {int(attention_mask.float().sum().item())} "
                  f"shape={tuple(attention_mask.shape)}")

    out = {"video_context": hidden.to(torch.bfloat16).cpu().contiguous()}
    # The Mojo HQ pipeline reads BOTH keys from one dump; carry the audio
    # pre-connector context over from the campaign dump (prompt-mismatched but
    # shared by both sides of an A/B; audio is not the verdict axis).
    if args.audio_from:
        from safetensors.torch import load_file
        src = load_file(args.audio_from)
        out["audio_context"] = src["audio_context"]
        print(f"audio_context copied from {args.audio_from}: {tuple(out['audio_context'].shape)}")
    save_file(out, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

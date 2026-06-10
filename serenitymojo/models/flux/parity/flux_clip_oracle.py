#!/usr/bin/env python3
# flux_clip_oracle.py — HF reference for the FLUX CLIP-L pooled vector (the
# `y` / vector_in conditioning, [1,768]).
#
# Reference = transformers CLIPTextModel loading the REAL clip_l.safetensors
# (text_model.* keys, NO text_projection -> FLUX pooled is the un-projected
# pooler_output = post-final-LN hidden at the EOS position). CLIP-L uses
# quick_gelu, eps 1e-5. Pooled at EOS is causal-invariant to post-EOS padding,
# so the pad-mask convention doesn't affect it.
#
# Tokenizes the same prompt, pads to 77 (pad id 49407 == eos, BFL/SDXL style),
# dumps:
#   flux_clip_ids.bin     int32 [77]
#   flux_clip_pooled.bin  f32   [1,768]
#   flux_clip_meta.txt
#
# Usage: python3 flux_clip_oracle.py

import os
import torch
from transformers import CLIPTextModel, CLIPTextConfig, PreTrainedTokenizerFast
from safetensors.torch import load_file

CLIP_PATH = "/home/alex/.serenity/models/text_encoders/clip_l.safetensors"
TOK_JSON = "/home/alex/.serenity/models/text_encoders/clip_l.tokenizer.json"
OUT_DIR = "/home/alex/mojodiffusion/serenitymojo/models/flux/parity"
S = 77
EOS = 49407
PROMPT = "a red apple on a wooden table"

os.makedirs(OUT_DIR, exist_ok=True)


def dump_f32(path, t):
    v = t.detach().to(torch.float32).contiguous().cpu().numpy().ravel()
    with open(path, "wb") as f:
        f.write(v.astype("<f4").tobytes())


def dump_i32(path, ids):
    import numpy as np
    with open(path, "wb") as f:
        f.write(np.array(ids, dtype="<i4").tobytes())


def main():
    DEV = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = CLIPTextConfig(
        vocab_size=49408, hidden_size=768, intermediate_size=3072,
        num_hidden_layers=12, num_attention_heads=12, max_position_embeddings=77,
        hidden_act="quick_gelu", layer_norm_eps=1e-5, bos_token_id=49406,
        eos_token_id=49407, pad_token_id=49407,
    )
    model = CLIPTextModel(cfg)  # small (~123M); plain init, no meta needed
    sd = load_file(CLIP_PATH)
    # fp16 -> bf16 (Mojo weights) -> fp32 (lossless, clean accumulation).
    sd = {k: v.to(torch.bfloat16).to(torch.float32) for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[oracle] CLIP load: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  missing[:8]:", missing[:8])
    if unexpected:
        print("  unexpected[:8]:", unexpected[:8])
    model = model.to(DEV).eval()

    tok = PreTrainedTokenizerFast(tokenizer_file=TOK_JSON)
    ids = tok(PROMPT)["input_ids"]
    # ensure BOS/EOS present; pad to 77 with EOS (matches Mojo _fit).
    if len(ids) > S:
        ids = ids[:S - 1] + [EOS]
    else:
        ids = ids + [EOS] * (S - len(ids))
    eos_pos = ids.index(EOS)
    print(f"[oracle] prompt='{PROMPT}'  ids[:12]={ids[:12]}  first_eos@{eos_pos}")

    input_ids = torch.tensor([ids], dtype=torch.long, device=DEV)
    with torch.no_grad():
        out = model(input_ids=input_ids)
    pooled = out.pooler_output  # [1,768], post-LN hidden at EOS, no projection
    print(f"[oracle] pooled {list(pooled.shape)} mean={pooled.mean().item():.6f} "
          f"std={pooled.std().item():.6f} min={pooled.min().item():.6f} max={pooled.max().item():.6f}")

    dump_i32(f"{OUT_DIR}/flux_clip_ids.bin", ids)
    dump_f32(f"{OUT_DIR}/flux_clip_pooled.bin", pooled)
    with open(f"{OUT_DIR}/flux_clip_meta.txt", "w") as f:
        f.write(f"prompt={PROMPT}\nS={S} eos_pos={eos_pos} pooled_shape={list(pooled.shape)}\n")
        f.write(f"ids[:12]={ids[:12]}\n")
        f.write(f"mean={pooled.mean().item():.6f} std={pooled.std().item():.6f}\n")
    print("[oracle] dumped flux_clip_ids.bin + flux_clip_pooled.bin")


if __name__ == "__main__":
    main()

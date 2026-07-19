#!/usr/bin/env python3
# flux_t5_oracle.py — HF reference for the FLUX T5-XXL encoder (the conditioning
# that this session's fp16->bf16 fix repaired).
#
# Reference = transformers T5EncoderModel loading the REAL t5xxl_fp16.safetensors.
# Faithful-dtype recipe: fp16 weights -> bf16 (EXACTLY the Mojo's weights, since
# T5Encoder.load casts every weight fp16->bf16) -> fp32 compute (= the Mojo's
# "BF16 storage, F32 accumulation"). FLUX/BFL run T5 with attention_mask=None, so
# we pass an all-ones mask (padding id 0 is NOT masked) — matching the Mojo which
# applies no mask beyond the relative bias.
#
# Tokenizes a fixed prompt, pads to 512 (pad id 0, eos id 1 already appended by
# the tokenizer), dumps:
#   flux_t5_ids.bin     int32 [512]
#   flux_t5_hidden.bin  f32   [1,512,4096]
#   flux_t5_meta.txt
#
# Usage: python3 flux_t5_oracle.py

import os
import torch
from transformers import T5EncoderModel, T5Config, PreTrainedTokenizerFast
from safetensors.torch import load_file

T5_PATH = "/home/alex/.serenity/models/text_encoders/t5xxl_fp16.safetensors"
TOK_JSON = "/home/alex/.serenity/models/text_encoders/t5xxl_fp16.tokenizer.json"
OUT_DIR = "/home/alex/mojodiffusion/serenitymojo/models/flux/parity"
S = 512
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
    cfg = T5Config(
        vocab_size=32128, d_model=4096, d_kv=64, d_ff=10240, num_layers=24,
        num_heads=64, relative_attention_num_buckets=32,
        relative_attention_max_distance=128, feed_forward_proj="gated-gelu",
        is_gated_act=True, dense_act_fn="gelu_new", tie_word_embeddings=False,
        layer_norm_epsilon=1e-6,
    )
    with torch.device("meta"):
        model = T5EncoderModel(cfg)
    sd = load_file(T5_PATH)                                  # fp16
    # fp16 -> bf16 (Mojo's exact weights) -> fp32 (lossless, fp32 accumulation).
    sd = {k: v.to(torch.bfloat16).to(torch.float32) for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    print(f"[oracle] T5 load: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  missing[:5]:", missing[:5])
    if unexpected:
        print("  unexpected[:5]:", unexpected[:5])
    model = model.to(DEV).eval()

    tok = PreTrainedTokenizerFast(tokenizer_file=TOK_JSON)
    enc = tok(PROMPT, return_tensors="pt")
    ids = enc["input_ids"][0].tolist()
    # T5 tokenizer appends </s>=1. Pad to S with 0 (BFL max_length padding).
    if len(ids) > S:
        ids = ids[:S]
    else:
        ids = ids + [0] * (S - len(ids))
    print(f"[oracle] prompt='{PROMPT}'  first 12 ids={ids[:12]}  (S={S})")

    input_ids = torch.tensor([ids], dtype=torch.long, device=DEV)
    attn = torch.ones_like(input_ids)  # all-ones: padding NOT masked (BFL=None)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn).last_hidden_state
    print(f"[oracle] hidden {list(out.shape)} mean={out.mean().item():.6f} "
          f"std={out.std().item():.6f} min={out.min().item():.6f} max={out.max().item():.6f} "
          f"nan={torch.isnan(out).any().item()}")

    dump_i32(f"{OUT_DIR}/flux_t5_ids.bin", ids)
    dump_f32(f"{OUT_DIR}/flux_t5_hidden.bin", out)
    with open(f"{OUT_DIR}/flux_t5_meta.txt", "w") as f:
        f.write(f"prompt={PROMPT}\nS={S} hidden_shape={list(out.shape)}\n")
        f.write(f"first12_ids={ids[:12]}\n")
        f.write(f"mean={out.mean().item():.6f} std={out.std().item():.6f}\n")
    print("[oracle] dumped flux_t5_ids.bin + flux_t5_hidden.bin")


if __name__ == "__main__":
    main()

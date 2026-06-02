#!/usr/bin/env python3
# serenitymojo/models/anima/parity/anima_text_context_fullpath_oracle.py
#
# BONUS full-path torch oracle: REAL Qwen3-0.6B (transformers Qwen3Model loaded
# from the on-disk anima text_encoder weights) -> zero-pad -> net.llm_adapter
# (the same LLMAdapter reimplementation as anima_text_context_oracle.py).
# Consumes the SAME token sidecar the Mojo pipeline reads (/tmp/anima_tokens.*),
# so it compares the WHOLE Mojo path end-to-end (encoder + adapter) at cos.
#
# Writes ref_fullpath_context.bin [1,512,1024] (F32). Compare vs the Mojo
# pipeline output /tmp/anima_context_cond.safetensors (context_cond).
#
# Run:
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/anima/parity/anima_text_context_fullpath_oracle.py \
#       /tmp/anima_tokens.safetensors

import os
import sys
import struct

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import Qwen3Model, Qwen3Config

REF_DIR = os.path.dirname(os.path.abspath(__file__))
QWEN3_W = "/home/alex/.serenity/models/anima/split_files/text_encoders/qwen_3_06b_base.safetensors"
CKPT = "/home/alex/.serenity/models/anima/split_files/diffusion_models/anima-base-v1.0.safetensors"
PREFIX = "net.llm_adapter"
DT = torch.float32

# adapter dims
DIM = 1024
N_HEADS = 16
HEAD_DIM = 64
N_BLOCKS = 6
THETA = 10000.0
EPS = 1e-6


# ── adapter math (identical to anima_text_context_oracle.py) ──
def rms_norm(x, w, eps=EPS):
    v = x.pow(2).mean(-1, keepdim=True)
    return w * (x * torch.rsqrt(v + eps))


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin, dim=1):
    return (x * cos.unsqueeze(dim)) + (rotate_half(x) * sin.unsqueeze(dim))


def build_rope(seq, hd, device):
    inv = 1.0 / (THETA ** (torch.arange(0, hd, 2, dtype=torch.int64).to(DT) / hd))
    pos = torch.arange(seq, device=device).to(DT)
    f = torch.outer(pos, inv)
    emb = torch.cat((f, f), -1)
    return emb.cos().unsqueeze(0), emb.sin().unsqueeze(0)


def attn(x, ctx_, w, p, cq, sq, ck, sk):
    b, lq, _ = x.shape
    lk = ctx_.shape[1]
    q = (x @ w[p + ".q_proj.weight"].T).view(b, lq, N_HEADS, HEAD_DIM)
    k = (ctx_ @ w[p + ".k_proj.weight"].T).view(b, lk, N_HEADS, HEAD_DIM)
    v = (ctx_ @ w[p + ".v_proj.weight"].T).view(b, lk, N_HEADS, HEAD_DIM)
    q = rms_norm(q, w[p + ".q_norm.weight"]).transpose(1, 2)
    k = rms_norm(k, w[p + ".k_norm.weight"]).transpose(1, 2)
    v = v.transpose(1, 2)
    q = apply_rope(q, cq, sq)
    k = apply_rope(k, ck, sk)
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=None)
    o = o.transpose(1, 2).reshape(b, lq, N_HEADS * HEAD_DIM).contiguous()
    return o @ w[p + ".o_proj.weight"].T


def adapter(t5_ids, qwen_hidden, w):
    x = w["embed.weight"][t5_ids.view(-1)].view(1, t5_ids.shape[1], DIM)
    cq, sq = build_rope(x.shape[1], HEAD_DIM, x.device)
    ck, sk = build_rope(qwen_hidden.shape[1], HEAD_DIM, x.device)
    for j in range(N_BLOCKS):
        bp = "blocks." + str(j)
        n = rms_norm(x, w[bp + ".norm_self_attn.weight"])
        x = x + attn(n, n, w, bp + ".self_attn", cq, sq, cq, sq)
        n = rms_norm(x, w[bp + ".norm_cross_attn.weight"])
        x = x + attn(n, qwen_hidden, w, bp + ".cross_attn", cq, sq, ck, sk)
        n = rms_norm(x, w[bp + ".norm_mlp.weight"])
        h = n @ w[bp + ".mlp.0.weight"].T + w[bp + ".mlp.0.bias"]
        h = F.gelu(h)
        h = h @ w[bp + ".mlp.2.weight"].T + w[bp + ".mlp.2.bias"]
        x = x + h
    x = x @ w["out_proj.weight"].T + w["out_proj.bias"]
    return rms_norm(x, w["norm.weight"])


def main():
    tokens_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/anima_tokens.safetensors"
    tok = load_file(tokens_path)
    qwen_ids = tok["qwen_input_ids"].to(torch.long).unsqueeze(0)
    qwen_mask = tok["qwen_attention_mask"].to(DT).unsqueeze(0)
    t5_ids = tok["t5_input_ids"].to(torch.long).unsqueeze(0)

    # REAL Qwen3-0.6B (transformers) loaded from the on-disk weights.
    cfg = Qwen3Config(
        vocab_size=151936, hidden_size=1024, intermediate_size=3072,
        num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=8,
        head_dim=128, rms_norm_eps=1e-6, rope_theta=1000000.0,
        max_position_embeddings=40960, tie_word_embeddings=True,
    )
    model = Qwen3Model(cfg).to(DT).eval()
    sd = {}
    with safe_open(QWEN3_W, "pt") as f:
        for k in f.keys():
            kk = k[len("model."):] if k.startswith("model.") else k
            sd[kk] = f.get_tensor(k).to(DT)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("qwen3 load: missing", len(missing), "unexpected", len(unexpected))

    with torch.no_grad():
        out = model(input_ids=qwen_ids, attention_mask=qwen_mask,
                    output_hidden_states=False)
        qwen_hidden = out.last_hidden_state.to(DT)
        qwen_hidden = qwen_hidden * qwen_mask.unsqueeze(-1)  # zero pad (AnimaModel.py:218)

        w = {}
        with safe_open(CKPT, "pt") as f:
            for k in f.keys():
                if k.startswith(PREFIX):
                    w[k[len(PREFIX) + 1:]] = f.get_tensor(k).to(DT)
        context = adapter(t5_ids, qwen_hidden, w)

    print("fullpath context", tuple(context.shape),
          "mean_abs", context.abs().mean().item())

    flat = context.reshape(-1).to(torch.float32).cpu().numpy()
    with open(os.path.join(REF_DIR, "ref_fullpath_context.bin"), "wb") as f:
        f.write(struct.pack("<%df" % flat.size, *flat.tolist()))
    print("wrote ref_fullpath_context.bin")


if __name__ == "__main__":
    main()

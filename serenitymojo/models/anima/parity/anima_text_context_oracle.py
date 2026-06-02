#!/usr/bin/env python3
# serenitymojo/models/anima/parity/anima_text_context_oracle.py
#
# Torch oracle for the Anima net.llm_adapter (AnimaTextConditioner / LLMAdapter).
# Built from the AUTHORITATIVE reference math
# (Anima-Standalone-Trainer/library/anima_models.py: LLMAdapter,
#  LLMAdapterTransformerBlock, LLMAdapterAttention, LLMAdapterRMSNorm,
#  AdapterRotaryEmbedding) which is the SAME class OneTrainer's AnimaModel.py
# instantiates as `self.text_conditioner` (diffusers AnimaTextConditioner — not
# pip-installable, so reimplemented here from the trainer source, 1:1).
#
# Loads the REAL net.llm_adapter.* weights from the on-disk checkpoint
#   /home/alex/.serenity/models/anima/split_files/diffusion_models/anima-base-v1.0.safetensors
# (BF16 -> F32 for a clean reference), runs the forward at S_TXT=512 query tokens
# (T5 ids) with K/V = a FIXED non-degenerate Qwen3-hidden [1,512,1024], and dumps:
#   in_t5_ids.bin        [1,512]      (F32-encoded int ids; queries)
#   in_qwen_hidden.bin   [1,512,1024] (F32; K/V source = Qwen3 last_hidden_state)
#   in_w_*.bin           every adapter weight (F32, exact ckpt layout)
#   ref_context.bin      [1,512,1024] (F32; the FROZEN cross-attn context)
# The Mojo gate (anima_text_context_parity.mojo) reads these and compares the
# context output at cos >= 0.999 (reports max_abs). Non-degenerate inputs; real
# adapter dims (dim 1024, heads 16, head_dim 64, 6 blocks, vocab 32128).
#
# IMPORTANT recipe facts vs. the inference-flame Rust port (which the captured
# 256 sidecar came from):
#   * MLP activation = nn.GELU() EXACT (erf), NOT tanh-approx. (anima_models.py:1621)
#   * mlp.0 / mlp.2 and out_proj HAVE bias (nn.Linear default). (1620,1622,1670)
#   * in_proj = Identity (model_dim==target_dim==1024) -> no in_proj weight.
#   * RoPE = standard rotate_half (cos/sin full head_dim, emb=cat(freqs,freqs));
#     == half-split RoPE. Q gets target positions, K gets context positions.
#   * Each of 6 blocks: self_attn(RoPE) -> cross_attn(RoPE q=tgt,k=ctx) -> MLP,
#     all RMSNorm-pre + residual. Final: out_proj then RMSNorm.
#
# Gate feeds ALL-ONES attention masks (no padding) so the mask path is inert and
# the comparison is mask-independent; output padding-zeroing is a caller concern
# (anima_python_ref.py: crossattn_emb[~t5_attn_mask]=0).
#
# Run (SEPARATE command, never chained after a mojo build):
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/anima/parity/anima_text_context_oracle.py

import math
import os
import struct

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

torch.manual_seed(0)
DT = torch.float64  # F64 interior reference (Mojo gate runs F32; clean cos)

REF_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT = "/home/alex/.serenity/models/anima/split_files/diffusion_models/anima-base-v1.0.safetensors"
PREFIX = "net.llm_adapter"

# ── REAL adapter dims (TRAINING_PLAN_anima_OT.md §C; ckpt shapes) ──
B = 1
S_TXT = 512        # PROMPT_MAX_LENGTH (AnimaModel.py:23)
S_LLM = 512        # Qwen3 hidden seq (OT also 512)
DIM = 1024         # model_dim == target_dim == source_dim
N_HEADS = 16
HEAD_DIM = 64      # DIM // N_HEADS
N_BLOCKS = 6
VOCAB = 32128
MLP_HIDDEN = 4096
THETA = 10000.0
EPS = 1e-6


def W(name, tensor):
    flat = tensor.detach().reshape(-1).to(torch.float32).cpu().numpy()
    path = os.path.join(REF_DIR, name + ".bin")
    with open(path, "wb") as f:
        f.write(struct.pack("<%df" % flat.size, *flat.tolist()))
    print("wrote", name, tuple(tensor.shape))


# ── load adapter weights (BF16 -> F64) ──
def load_adapter():
    w = {}
    with safe_open(CKPT, "pt") as f:
        for k in f.keys():
            if k.startswith(PREFIX):
                w[k[len(PREFIX) + 1:]] = f.get_tensor(k).to(DT)
    return w


# ── reference math (1:1 with anima_models.py) ──
def rms_norm(x, weight, eps=EPS):
    var = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(var + eps)
    return weight * x


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin, unsqueeze_dim=1):
    # x: [B, H, S, D]; cos/sin: [B, S, D]
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


def build_rope(seq_len, head_dim, device):
    inv_freq = 1.0 / (THETA ** (torch.arange(0, head_dim, 2, dtype=torch.int64).to(DT) / head_dim))
    pos = torch.arange(seq_len, device=device).to(DT)
    freqs = torch.outer(pos, inv_freq)              # [S, D/2]
    emb = torch.cat((freqs, freqs), dim=-1)         # [S, D]
    cos = emb.cos().unsqueeze(0)                     # [1, S, D]
    sin = emb.sin().unsqueeze(0)
    return cos, sin


def attention(x, context, w, prefix, cos_q, sin_q, cos_k, sin_k):
    # q from x, k/v from context. QK-norm per head. RoPE on q (target pos),
    # k (context pos). SDPA no mask (all-ones). o_proj.
    bsz, sq, _ = x.shape
    sk = context.shape[1]
    q = x @ w[prefix + ".q_proj.weight"].T          # [B, Sq, inner]
    k = context @ w[prefix + ".k_proj.weight"].T    # [B, Sk, inner]
    v = context @ w[prefix + ".v_proj.weight"].T
    q = q.view(bsz, sq, N_HEADS, HEAD_DIM)
    k = k.view(bsz, sk, N_HEADS, HEAD_DIM)
    v = v.view(bsz, sk, N_HEADS, HEAD_DIM)
    q = rms_norm(q, w[prefix + ".q_norm.weight"]).transpose(1, 2)   # [B,H,Sq,D]
    k = rms_norm(k, w[prefix + ".k_norm.weight"]).transpose(1, 2)   # [B,H,Sk,D]
    v = v.transpose(1, 2)
    q = apply_rope(q, cos_q, sin_q)
    k = apply_rope(k, cos_k, sin_k)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=None)   # [B,H,Sq,D]
    out = out.transpose(1, 2).reshape(bsz, sq, N_HEADS * HEAD_DIM).contiguous()
    out = out @ w[prefix + ".o_proj.weight"].T
    return out


def block_forward(x, context, w, bp, cos_q, sin_q, cos_k, sin_k):
    # self-attn (q,k both target positions)
    normed = rms_norm(x, w[bp + ".norm_self_attn.weight"])
    x = x + attention(normed, normed, w, bp + ".self_attn",
                      cos_q, sin_q, cos_q, sin_q)
    # cross-attn (q target pos, k context pos)
    normed = rms_norm(x, w[bp + ".norm_cross_attn.weight"])
    x = x + attention(normed, context, w, bp + ".cross_attn",
                      cos_q, sin_q, cos_k, sin_k)
    # MLP (bias, EXACT gelu)
    normed = rms_norm(x, w[bp + ".norm_mlp.weight"])
    h = normed @ w[bp + ".mlp.0.weight"].T + w[bp + ".mlp.0.bias"]
    h = F.gelu(h)  # exact erf (nn.GELU default)
    h = h @ w[bp + ".mlp.2.weight"].T + w[bp + ".mlp.2.bias"]
    x = x + h
    return x


def adapter_forward(t5_ids, qwen_hidden, w):
    # embed (T5 ids -> queries). in_proj = Identity (model_dim==target_dim).
    embed = w["embed.weight"][t5_ids.view(-1)].view(B, S_TXT, DIM)
    x = embed
    context = qwen_hidden
    cos_q, sin_q = build_rope(S_TXT, HEAD_DIM, x.device)
    cos_k, sin_k = build_rope(S_LLM, HEAD_DIM, x.device)
    for j in range(N_BLOCKS):
        x = block_forward(x, context, w, "blocks." + str(j),
                          cos_q, sin_q, cos_k, sin_k)
    x = x @ w["out_proj.weight"].T + w["out_proj.bias"]
    x = rms_norm(x, w["norm.weight"])
    return x


def main():
    w = load_adapter()
    print("loaded", len(w), "adapter tensors")

    g = torch.Generator().manual_seed(1234)
    # FIXED non-degenerate inputs.
    t5_ids = torch.randint(0, VOCAB, (B, S_TXT), generator=g)  # int queries
    qwen_hidden = torch.randn(B, S_LLM, DIM, generator=g, dtype=torch.float32).to(DT)
    # scale to a believable Qwen3 last_hidden_state magnitude.
    qwen_hidden = qwen_hidden * 0.5

    with torch.no_grad():
        ctx = adapter_forward(t5_ids, qwen_hidden, w)

    print("context", tuple(ctx.shape), "mean_abs", ctx.abs().mean().item(),
          "std", ctx.std().item())

    # dump inputs
    W("in_t5_ids", t5_ids.to(torch.float32))
    W("in_qwen_hidden", qwen_hidden)
    # dump every adapter weight under in_w_<flatname>
    for k, v in w.items():
        W("in_w_" + k.replace(".", "_"), v)
    # dump reference output
    W("ref_context", ctx)

    # meta
    with open(os.path.join(REF_DIR, "anima_text_context_meta.txt"), "w") as f:
        f.write("S_TXT=%d S_LLM=%d DIM=%d N_HEADS=%d HEAD_DIM=%d N_BLOCKS=%d\n"
                % (S_TXT, S_LLM, DIM, N_HEADS, HEAD_DIM, N_BLOCKS))
        f.write("context_mean_abs=%.6f context_std=%.6f\n"
                % (ctx.abs().mean().item(), ctx.std().item()))
    print("DONE")


if __name__ == "__main__":
    main()

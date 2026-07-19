"""Krea-2 Attention parity oracle (the highest-risk op: GQA + QKNorm + RoPE + gate).

Imports ai-toolkit's krea2 ``Attention`` + ``PositionalEncoding`` DIRECTLY from
mmdit.py (torch+einops only, bypassing the ai-toolkit package __init__). Builds a
production-arch Attention (dim=6144, heads=48, kvheads=12, bias=False) bf16-on-cuda
with random weights, runs it on a random x + a real 3-axis RoPE freqs table, and
dumps everything so the Mojo probe can run krea2_attention with the EXACT weights
and compare via cosine (bar >= 0.999).

DISK NOTE: the 5 projection weights are saved as **bf16** (the production dtype
the probe loads them as anyway) to roughly halve the file — the big tensors are
wq/gate/wo [6144,6144]. x / output / scales / pos / cos / sin stay F32 (tiny).

Run:  /home/alex/serenityflow-v2/.venv/bin/python \
        serenitymojo/models/dit/parity/gen_krea2_attention.py
"""

import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, "/home/alex/ai-toolkit/extensions_built_in/diffusion_models/krea2/src")
from mmdit import Attention, PositionalEncoding  # noqa: E402

OUT = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/krea2_attention_oracle.safetensors"
DEV = "cuda"
DT = torch.bfloat16

FEATURES = 6144
HEADS = 48
KVHEADS = 12
HEAD_DIM = FEATURES // HEADS  # 128
THETA = 1e3
AXES = [
    HEAD_DIM - 12 * (HEAD_DIM // 16),
    6 * (HEAD_DIM // 16),
    6 * (HEAD_DIM // 16),
]  # [32, 48, 48]
L = 32

torch.manual_seed(7777)

attn = Attention(dim=FEATURES, heads=HEADS, kvheads=KVHEADS, bias=False).to(DEV, DT)
# Nonzero qknorm scales (real Parameter is zeros-init; nonzero exercises scale+1).
with torch.no_grad():
    attn.qknorm.qnorm.scale.copy_((torch.randn(HEAD_DIM, device=DEV) * 0.1).to(torch.float32))
    attn.qknorm.knorm.scale.copy_((torch.randn(HEAD_DIM, device=DEV) * 0.1).to(torch.float32))

x = torch.randn(1, L, FEATURES, dtype=DT, device=DEV)

# Real 3-axis RoPE freqs from a pos grid (incl. a large global pos to stress F64).
pos = torch.zeros(1, L, 3, dtype=torch.float32, device=DEV)
TXTLEN = 5
GH, GW = 3, 9  # 27 image tokens; 5 text + 27 = 32 = L
for tok in range(L - TXTLEN):
    h = tok // GW
    w = tok % GW
    pos[0, TXTLEN + tok, 0] = float(2000 + tok)
    pos[0, TXTLEN + tok, 1] = float(h)
    pos[0, TXTLEN + tok, 2] = float(w)

posemb = PositionalEncoding(FEATURES, AXES, theta=THETA, ntk=1.0).to(DEV)
with torch.no_grad():
    freqs = posemb(pos)                       # [1, L, head_dim/2, 2, 2]
    y = attn(x, freqs=freqs, mask=None)       # [1, L, FEATURES]

cos = freqs[0, :, :, 0, 0].contiguous()       # [L, head_dim/2]
sin = freqs[0, :, :, 1, 0].contiguous()

out = {
    "x": x.to(DT).cpu().contiguous(),
    "wq": attn.wq.weight.to(DT).cpu().contiguous(),       # [6144, 6144] bf16
    "wk": attn.wk.weight.to(DT).cpu().contiguous(),       # [1536, 6144]
    "wv": attn.wv.weight.to(DT).cpu().contiguous(),       # [1536, 6144]
    "gate_w": attn.gate.weight.to(DT).cpu().contiguous(), # [6144, 6144]
    "wo": attn.wo.weight.to(DT).cpu().contiguous(),       # [6144, 6144]
    "qnorm_scale": attn.qknorm.qnorm.scale.float().cpu().contiguous(),  # [128] F32
    "knorm_scale": attn.qknorm.knorm.scale.float().cpu().contiguous(),  # [128] F32
    "pos": pos.float().cpu().contiguous(),    # [1, L, 3]
    "cos": cos.float().cpu().contiguous(),    # [L, 64]
    "sin": sin.float().cpu().contiguous(),
    "y": y.float().cpu().contiguous(),        # [1, L, 6144]
}
print("shapes/dtypes:", {k: (tuple(v.shape), str(v.dtype)) for k, v in out.items()}, flush=True)
save_file(out, OUT)
import os
print(f"OK dumped -> {OUT}  ({os.path.getsize(OUT)/1e6:.1f} MB)", flush=True)

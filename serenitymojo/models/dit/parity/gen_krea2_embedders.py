"""Krea-2 embedders + input/output heads parity oracle (chunk 5).

Imports ai-toolkit's krea2 ``temb`` fn, ``RMSNorm``, ``GELU``-Sequential mirrors,
and ``LastLayer`` DIRECTLY from mmdit.py (torch+einops only). The MLP heads
(tmlp/tproj/txtmlp) and ``first`` are tiny nn.Sequential / nn.Linear stacks at the
production dims, rebuilt here with the SAME dims + random weights (rather than
instantiating the whole 28-block SingleStreamDiT). bf16-on-cuda. Dumps every
input + weight + output so the Mojo probe can run each krea2 fn with the EXACT
weights and compare via cosine (bar >= 0.999).

CRITICAL temb detail being gated: tfactor=1e3 pre-scale on t, period=1e4, and
cos-THEN-sin concat order. A missing tfactor or swapped cos/sin silently
corrupts all conditioning — this oracle catches it.

DISK NOTE: tproj's Linear(6144 -> 36864) weight (~450MB bf16) is the big one;
saved bf16. Everything else is small.

Run:  /home/alex/serenityflow-v2/.venv/bin/python \
        serenitymojo/models/dit/parity/gen_krea2_embedders.py
"""

import sys

import torch
import torch.nn as nn
from safetensors.torch import save_file

sys.path.insert(0, "/home/alex/ai-toolkit/extensions_built_in/diffusion_models/krea2/src")
from mmdit import temb as temb_fn, RMSNorm, LastLayer  # noqa: E402

OUT = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/krea2_embedders_oracle.safetensors"
DEV = "cuda"
DT = torch.bfloat16

FEATURES = 6144
TDIM = 256
TXTDIM = 2560
PATCH = 2
CHANNELS = 16
L = 17                       # text/image token count for the [1,L,*] inputs
N = 13                       # patch count for `first`
FIRST_IN = CHANNELS * PATCH * PATCH      # 64
LAST_OUT = PATCH * PATCH * CHANNELS      # 64

torch.manual_seed(20250624)


def _seq_tmlp():
    return nn.Sequential(
        nn.Linear(TDIM, FEATURES),
        nn.GELU(approximate="tanh"),
        nn.Linear(FEATURES, FEATURES),
    )


def _seq_tproj():
    return nn.Sequential(
        nn.GELU(approximate="tanh"),
        nn.Linear(FEATURES, FEATURES * 6),
    )


def _seq_txtmlp():
    return nn.Sequential(
        RMSNorm(TXTDIM),
        nn.Linear(TXTDIM, FEATURES),
        nn.GELU(approximate="tanh"),
        nn.Linear(FEATURES, FEATURES),
    )


tmlp = _seq_tmlp().to(DEV, DT)
tproj = _seq_tproj().to(DEV, DT)
txtmlp = _seq_txtmlp().to(DEV, DT)
first = nn.Linear(FIRST_IN, FEATURES).to(DEV, DT)
last = LastLayer(FEATURES, PATCH, CHANNELS).to(DEV, DT)

# Nonzero RMSNorm / SimpleModulation scales (zeros-init by default).
with torch.no_grad():
    txtmlp[0].scale.copy_((torch.randn(TXTDIM, device=DEV) * 0.1).to(torch.float32))
    last.norm.scale.copy_((torch.randn(FEATURES, device=DEV) * 0.1).to(torch.float32))
    last.modulation.lin.copy_((torch.randn(2, FEATURES, device=DEV) * 0.1).to(last.modulation.lin.dtype))

# ── Inputs ──
t_in = torch.rand(1, device=DEV) * 1000.0            # timestep in 0..1000 scale (B=1)
ctx_in = torch.randn(1, L, TXTDIM, dtype=DT, device=DEV)
first_in = torch.randn(1, N, FIRST_IN, dtype=DT, device=DEV)
last_x = torch.randn(1, L, FEATURES, dtype=DT, device=DEV)

with torch.no_grad():
    te = temb_fn(t_in, TDIM, device=DEV, dtype=DT)   # [1,1,256]
    t = tmlp(te)                                     # [1,1,6144]
    vec = tproj(t)                                   # [1,1,36864]
    txt = txtmlp(ctx_in)                             # [1,L,6144]
    first_out = first(first_in)                      # [1,N,6144]
    last_tvec = t                                    # LastLayer's tvec = tmlp output
    last_out = last(last_x, last_tvec)               # [1,L,64]

out = {
    # temb
    "t_in": t_in.float().cpu().contiguous(),         # [1]
    "te": te.float().cpu().contiguous(),             # [1,1,256]
    # tmlp
    "tmlp_w1": tmlp[0].weight.to(DT).cpu().contiguous(),
    "tmlp_b1": tmlp[0].bias.float().cpu().contiguous(),
    "tmlp_w2": tmlp[2].weight.to(DT).cpu().contiguous(),
    "tmlp_b2": tmlp[2].bias.float().cpu().contiguous(),
    "t": t.float().cpu().contiguous(),               # [1,1,6144]
    # tproj
    "tproj_w": tproj[1].weight.to(DT).cpu().contiguous(),   # [36864,6144]
    "tproj_b": tproj[1].bias.float().cpu().contiguous(),
    "vec": vec.float().cpu().contiguous(),           # [1,1,36864]
    # txtmlp
    "ctx_in": ctx_in.to(DT).cpu().contiguous(),
    "txt_rms_scale": txtmlp[0].scale.float().cpu().contiguous(),  # [2560] F32
    "txt_w1": txtmlp[1].weight.to(DT).cpu().contiguous(),
    "txt_b1": txtmlp[1].bias.float().cpu().contiguous(),
    "txt_w2": txtmlp[3].weight.to(DT).cpu().contiguous(),
    "txt_b2": txtmlp[3].bias.float().cpu().contiguous(),
    "txt": txt.float().cpu().contiguous(),           # [1,L,6144]
    # first
    "first_in": first_in.to(DT).cpu().contiguous(),
    "first_w": first.weight.to(DT).cpu().contiguous(),  # [6144,64]
    "first_b": first.bias.float().cpu().contiguous(),
    "first_out": first_out.float().cpu().contiguous(),  # [1,N,6144]
    # LastLayer
    "last_x": last_x.to(DT).cpu().contiguous(),
    "last_tvec": last_tvec.float().cpu().contiguous(),  # [1,1,6144]
    "last_norm_scale": last.norm.scale.float().cpu().contiguous(),  # [6144] F32
    "last_mod_lin": last.modulation.lin.float().cpu().contiguous(), # [2,6144] F32
    "last_lin_w": last.linear.weight.to(DT).cpu().contiguous(),     # [64,6144]
    "last_lin_b": last.linear.bias.float().cpu().contiguous(),      # [64]
    "last_out": last_out.float().cpu().contiguous(),   # [1,L,64]
}
print("shapes:", {k: tuple(v.shape) for k, v in out.items()}, flush=True)
save_file(out, OUT)
import os
print(f"OK dumped -> {OUT}  ({os.path.getsize(OUT)/1e6:.1f} MB)", flush=True)

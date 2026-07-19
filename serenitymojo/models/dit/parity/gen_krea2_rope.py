"""Krea-2 3-axis RoPE parity oracle (serenitymojo gate).

Imports ai-toolkit's krea2 ``PositionalEncoding`` + ``ropeapply`` (the validated
oracle the user's live Krea-2 training exercises) and dumps, for a FIXED seeded
``pos`` (B=1, L, 3) and FIXED q/k, the layout-independent ground truth
(``q_roped``/``k_roped``) plus the per-pair ``cos``/``sin`` tables. The Mojo
parity probe reads ``pos``/``q``/``k`` byte-identically, builds krea2 RoPE on the
same ``pos``, applies it, and compares via cosine (bar >= 0.999).

RoPE is deterministic F64 table + F32 apply (no matmul) so GPU-vs-CPU does not
diverge; runs on cuda for faithfulness. Tiny (<1 MB), display-safe.

Run:  /home/alex/serenityflow-v2/.venv/bin/python \
        serenitymojo/models/dit/parity/gen_krea2_rope.py
"""

import sys

import torch
from safetensors.torch import save_file

# Import mmdit.py directly (its only deps are torch+einops) to bypass the
# ai-toolkit package __init__ chain (which pulls torchao, absent here).
sys.path.insert(0, "/home/alex/ai-toolkit/extensions_built_in/diffusion_models/krea2/src")
from mmdit import PositionalEncoding, ropeapply  # noqa: E402

OUT = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/krea2_rope_oracle.safetensors"
DEV = "cuda"

# ── architecture (single_mmdit_large_wide) ──
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
assert sum(AXES) == HEAD_DIM, (AXES, HEAD_DIM)
assert all(a % 2 == 0 for a in AXES), AXES

torch.manual_seed(1234)

# A small grid + text tokens, mirroring pipeline.prepare. Include a LARGE global
# position to actually exercise the F64 range reduction (theta=1e3, low-freq
# omega ~ 1.0 -> angle ~ thousands of radians where plain F32 trig is wrong).
TXTLEN = 7
GH, GW = 4, 5
IMGLEN = GH * GW
L = TXTLEN + IMGLEN

pos = torch.zeros(1, L, 3, dtype=torch.float32, device=DEV)
for tok in range(IMGLEN):
    h = tok // GW
    w = tok % GW
    pos[0, TXTLEN + tok, 0] = float(2000 + tok)  # large global pos -> stress F64 reduction
    pos[0, TXTLEN + tok, 1] = float(h)
    pos[0, TXTLEN + tok, 2] = float(w)

q = torch.randn(1, HEADS, L, HEAD_DIM, dtype=torch.float32, device=DEV)
k = torch.randn(1, KVHEADS, L, HEAD_DIM, dtype=torch.float32, device=DEV)

posemb = PositionalEncoding(FEATURES, AXES, theta=THETA, ntk=1.0).to(DEV)
with torch.no_grad():
    freqs = posemb(pos)  # [B, L, head_dim/2, 2, 2]
    q_roped, k_roped = ropeapply(q, k, freqs)

# rope() stacks [cos, -sin, sin, cos] -> reshape(...,2,2):
#   freqs[...,0,0]=cos  freqs[...,0,1]=-sin  freqs[...,1,0]=sin  freqs[...,1,1]=cos
cos = freqs[0, :, :, 0, 0].contiguous()  # [L, head_dim/2]
sin = freqs[0, :, :, 1, 0].contiguous()  # [L, head_dim/2]

print(
    f"pos={tuple(pos.shape)} q={tuple(q.shape)} k={tuple(k.shape)} "
    f"freqs={tuple(freqs.shape)} cos={tuple(cos.shape)} q_roped={tuple(q_roped.shape)}",
    flush=True,
)

out = {
    "pos": pos.float().cpu().contiguous(),
    "q": q.float().cpu().contiguous(),
    "k": k.float().cpu().contiguous(),
    "cos": cos.float().cpu().contiguous(),
    "sin": sin.float().cpu().contiguous(),
    "q_roped": q_roped.float().cpu().contiguous(),
    "k_roped": k_roped.float().cpu().contiguous(),
}
save_file(out, OUT)
print("OK dumped:", ", ".join(sorted(out.keys())), "->", OUT, flush=True)

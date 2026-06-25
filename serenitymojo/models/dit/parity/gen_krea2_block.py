"""Krea-2 SingleStreamBlock parity oracle (chunk 4 — AdaLN composition).

Imports ai-toolkit's krea2 ``SingleStreamBlock`` + ``PositionalEncoding`` DIRECTLY
from mmdit.py (torch+einops only). Builds a production-arch block (features=6144,
heads=48, multiplier=4, kvheads=12, bias=False) bf16-on-cuda with random weights,
runs it on a random x + a random vec + a real 3-axis RoPE freqs table, and dumps
everything so the Mojo probe can run krea2_single_stream_block with the EXACT
weights and compare via cosine (bar >= 0.999).

This gates the AdaLN COMPOSITION math (double-shared modulation -> 2 gated
residual branches). The outlier-channel behavior over 28 real blocks is the
chunk-6 full-forward gate, NOT here — a clean cos>=0.999 here is sufficient.

DISK NOTE: the big weights (attn wq/gate/wo [6144,6144], mlp gate/up/down
[16384,6144]/[6144,16384]) are saved BF16 (the dtype the probe loads them as) to
keep the file small. x / vec / output / scales / pos / cos / sin stay F32 (tiny).

Run:  /home/alex/serenityflow-v2/.venv/bin/python \
        serenitymojo/models/dit/parity/gen_krea2_block.py
"""

import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, "/home/alex/ai-toolkit/extensions_built_in/diffusion_models/krea2/src")
from mmdit import SingleStreamBlock, PositionalEncoding  # noqa: E402

OUT = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/krea2_block_oracle.safetensors"
DEV = "cuda"
DT = torch.bfloat16

FEATURES = 6144
HEADS = 48
KVHEADS = 12
MULTIPLIER = 4
HEAD_DIM = FEATURES // HEADS  # 128
THETA = 1e3
AXES = [
    HEAD_DIM - 12 * (HEAD_DIM // 16),
    6 * (HEAD_DIM // 16),
    6 * (HEAD_DIM // 16),
]  # [32, 48, 48]
L = 32

torch.manual_seed(91919)

block = SingleStreamBlock(
    features=FEATURES, heads=HEADS, multiplier=MULTIPLIER, bias=False, kvheads=KVHEADS
).to(DEV, DT)

# Nonzero modulation + qknorm scales (real params are zeros-init; nonzero
# exercises the (1+scale) / scale+1 paths and the gates).
with torch.no_grad():
    block.mod.lin.copy_((torch.randn(6 * FEATURES, device=DEV) * 0.1).to(block.mod.lin.dtype))
    block.prenorm.scale.copy_((torch.randn(FEATURES, device=DEV) * 0.1).to(torch.float32))
    block.postnorm.scale.copy_((torch.randn(FEATURES, device=DEV) * 0.1).to(torch.float32))
    block.attn.qknorm.qnorm.scale.copy_((torch.randn(HEAD_DIM, device=DEV) * 0.1).to(torch.float32))
    block.attn.qknorm.knorm.scale.copy_((torch.randn(HEAD_DIM, device=DEV) * 0.1).to(torch.float32))

x = torch.randn(1, L, FEATURES, dtype=DT, device=DEV)
vec = torch.randn(1, 6 * FEATURES, dtype=DT, device=DEV)

pos = torch.zeros(1, L, 3, dtype=torch.float32, device=DEV)
TXTLEN = 5
GW = 9
for tok in range(L - TXTLEN):
    h = tok // GW
    w = tok % GW
    pos[0, TXTLEN + tok, 0] = float(2000 + tok)
    pos[0, TXTLEN + tok, 1] = float(h)
    pos[0, TXTLEN + tok, 2] = float(w)

posemb = PositionalEncoding(FEATURES, AXES, theta=THETA, ntk=1.0).to(DEV)
with torch.no_grad():
    freqs = posemb(pos)
    y = block(x, vec, freqs, mask=None)   # [1, L, FEATURES]

cos = freqs[0, :, :, 0, 0].contiguous()
sin = freqs[0, :, :, 1, 0].contiguous()

out = {
    "x": x.to(DT).cpu().contiguous(),
    "vec": vec.to(DT).cpu().contiguous(),                 # [1, 36864] bf16
    "mod_lin": block.mod.lin.float().cpu().contiguous(),  # [36864] F32 (tiny)
    "prenorm_scale": block.prenorm.scale.float().cpu().contiguous(),    # [6144] F32
    "postnorm_scale": block.postnorm.scale.float().cpu().contiguous(),  # [6144] F32
    "wq": block.attn.wq.weight.to(DT).cpu().contiguous(),
    "wk": block.attn.wk.weight.to(DT).cpu().contiguous(),
    "wv": block.attn.wv.weight.to(DT).cpu().contiguous(),
    "gate_w": block.attn.gate.weight.to(DT).cpu().contiguous(),
    "wo": block.attn.wo.weight.to(DT).cpu().contiguous(),
    "qnorm_scale": block.attn.qknorm.qnorm.scale.float().cpu().contiguous(),  # [128] F32
    "knorm_scale": block.attn.qknorm.knorm.scale.float().cpu().contiguous(),
    "mlp_gate_w": block.mlp.gate.weight.to(DT).cpu().contiguous(),   # [16384,6144]
    "mlp_up_w": block.mlp.up.weight.to(DT).cpu().contiguous(),       # [16384,6144]
    "mlp_down_w": block.mlp.down.weight.to(DT).cpu().contiguous(),   # [6144,16384]
    "pos": pos.float().cpu().contiguous(),
    "cos": cos.float().cpu().contiguous(),
    "sin": sin.float().cpu().contiguous(),
    "y": y.float().cpu().contiguous(),
}
print("shapes:", {k: tuple(v.shape) for k, v in out.items()}, flush=True)
save_file(out, OUT)
import os
print(f"OK dumped -> {OUT}  ({os.path.getsize(OUT)/1e6:.1f} MB)", flush=True)

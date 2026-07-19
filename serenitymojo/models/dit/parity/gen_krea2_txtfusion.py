"""Krea-2 TextFusionTransformer parity oracle (chunk 6).

Imports ai-toolkit's krea2 ``TextFusionTransformer`` + ``_mask`` DIRECTLY from
mmdit.py. Instantiates the production config (num_txt_layers=12, txt_dim=2560,
heads=20, multiplier=4, kvheads=20) bf16-on-cuda with random weights, runs it on a
random context [1, Lt, 12, 2560] (CORRECT axis order: Lt tokens, 12 layers) with a
key-padding mask that has SOME padded (0) token positions (to exercise the masked
refiner), and dumps everything so the Mojo probe can run krea2_text_fusion with the
EXACT weights and compare via cosine (bar >= 0.999).

SHAPE NOTE (derived by RUNNING the reference): context is [B, Lt, n=12, d]; the
forward's local names `b,l,n,d` mean l=Lt (tokens), n=12 (layers). The projector
Linear(12->1) collapses the 12-LAYER axis. The reference forward ERRORS unless
n==num_txt_layers==12 (proven), confirming n is the layer axis.

Run:  /home/alex/serenityflow-v2/.venv/bin/python \
        serenitymojo/models/dit/parity/gen_krea2_txtfusion.py
"""

import sys

import torch
import torch.nn.functional as F
from einops import rearrange
from torch.nn.attention import SDPBackend, sdpa_kernel
from safetensors.torch import save_file

sys.path.insert(0, "/home/alex/ai-toolkit/extensions_built_in/diffusion_models/krea2/src")
import mmdit  # noqa: E402
from mmdit import TextFusionTransformer, _mask  # noqa: E402


# The reference `attention()` (mmdit.py:51-63) FORCES SDPBackend.CUDNN_ATTENTION.
# cuDNN's fused FLOAT-mask kernel does NOT compute the mathematical additive
# `softmax(QKᵀ*scale + mask)` — for a non-trivial (padded) float mask it deviates
# from the MATH backend by ~0.0015 cos at these score magnitudes (MEASURED: full
# 6-pad txtfusion cuDNN vs MATH = 0.9985). serenity's _sdpa_math faithfully
# implements the additive formula (serenity-masked == torch-MATH-additive, cos
# 0.99999). So the FAITHFUL masked reference is the MATH backend, NOT cuDNN — the
# cuDNN deviation is a reference backend artifact, not something a math-correct
# port can or should reproduce. We force the MATH backend for the masked-case
# reference so the gate measures the real (additive) target. The no-mask case is
# unaffected (all-ones mask is a softmax-invariant no-op under any backend).
def _attention_math(q, k, v, mask=None, scale=None, gqa=False):
    with sdpa_kernel(SDPBackend.MATH):
        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, scale=scale, enable_gqa=gqa
        )
    return rearrange(x, "B H L D -> B L (H D)")

OUT = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/krea2_txtfusion_oracle.safetensors"
DEV = "cuda"
DT = torch.bfloat16

TXTDIM = 2560
NLAYERS = 12          # num_txt_layers
HEADS = 20
KVHEADS = 20          # == heads -> no GQA
MULTIPLIER = 4
HEADDIM = TXTDIM // HEADS   # 128
LT = 24               # caption token length
NPAD = 6              # padded-case: last NPAD tokens padded (exercises the masked sdpa path)

torch.manual_seed(31337)

tft = TextFusionTransformer(
    num_txt_layers=NLAYERS, txt_dim=TXTDIM, heads=HEADS,
    multiplier=MULTIPLIER, bias=False, kvheads=KVHEADS,
).to(DEV, DT)

# Nonzero RMSNorm + QKNorm scales (zeros-init by default).
with torch.no_grad():
    for blk in list(tft.layerwise_blocks) + list(tft.refiner_blocks):
        blk.prenorm.scale.copy_((torch.randn(TXTDIM, device=DEV) * 0.1).to(torch.float32))
        blk.postnorm.scale.copy_((torch.randn(TXTDIM, device=DEV) * 0.1).to(torch.float32))
        blk.attn.qknorm.qnorm.scale.copy_((torch.randn(HEADDIM, device=DEV) * 0.1).to(torch.float32))
        blk.attn.qknorm.knorm.scale.copy_((torch.randn(HEADDIM, device=DEV) * 0.1).to(torch.float32))

# CORRECT axis order: [B, Lt(tokens), NLAYERS(12), d]
context = torch.randn(1, LT, NLAYERS, TXTDIM, dtype=DT, device=DEV)

# Two reference outputs from the SAME model + context, covering both gates:
#  (A) the REAL b==1 single-caption inference case: keep is ALL-ONES (no text pad —
#      pad_text_features pads to the per-batch max, one caption => its own length =>
#      mask all-ones). _mask(all-ones) is a uniform +1 bias => softmax-invariant =>
#      no-op. The Mojo gate runs the refiner with mask=None and compares here.
#  (B) the PADDED masked case (last NPAD tokens padded): exercises the actual
#      masked sdpa path (cuDNN float-mask), which chunk 7's pad-to-256 mask needs.
#      The Mojo gate builds a bf16 0/1 mask and runs the masked sdpa path.
keep_ones = torch.ones(1, LT, device=DEV)               # all-ones (case A)
keep_pad = torch.ones(1, LT, device=DEV)
keep_pad[0, LT - NPAD:] = 0.0                            # NPAD padded (case B)

with torch.no_grad():
    out_nomask = tft(context, mask=None)                # case A reference (== all-ones, no-op)
    # Case B: force the MATH backend (the faithful additive target serenity
    # implements; cuDNN's float-mask kernel is a non-reproducible backend artifact).
    _orig_attention = mmdit.attention
    mmdit.attention = _attention_math
    try:
        out_masked = tft(context, mask=_mask(keep_pad)) # case B reference (MATH additive)
    finally:
        mmdit.attention = _orig_attention

dump = {
    "context": context.to(DT).cpu().contiguous(),
    "keep_pad": keep_pad.float().cpu().contiguous(),    # [1, Lt] (probe rebuilds the masked-case mask)
    "projector_w": tft.projector.weight.to(DT).cpu().contiguous(),  # [1, 12]
    "out_nomask": out_nomask.float().cpu().contiguous(),  # [1, Lt, d]  case A
    "out_masked": out_masked.float().cpu().contiguous(),  # [1, Lt, d]  case B
}


def dump_block(prefix, blk):
    dump[f"{prefix}_prenorm"] = blk.prenorm.scale.float().cpu().contiguous()
    dump[f"{prefix}_postnorm"] = blk.postnorm.scale.float().cpu().contiguous()
    dump[f"{prefix}_wq"] = blk.attn.wq.weight.to(DT).cpu().contiguous()
    dump[f"{prefix}_wk"] = blk.attn.wk.weight.to(DT).cpu().contiguous()
    dump[f"{prefix}_wv"] = blk.attn.wv.weight.to(DT).cpu().contiguous()
    dump[f"{prefix}_gate"] = blk.attn.gate.weight.to(DT).cpu().contiguous()
    dump[f"{prefix}_wo"] = blk.attn.wo.weight.to(DT).cpu().contiguous()
    dump[f"{prefix}_qnorm"] = blk.attn.qknorm.qnorm.scale.float().cpu().contiguous()
    dump[f"{prefix}_knorm"] = blk.attn.qknorm.knorm.scale.float().cpu().contiguous()
    dump[f"{prefix}_mlp_gate"] = blk.mlp.gate.weight.to(DT).cpu().contiguous()
    dump[f"{prefix}_mlp_up"] = blk.mlp.up.weight.to(DT).cpu().contiguous()
    dump[f"{prefix}_mlp_down"] = blk.mlp.down.weight.to(DT).cpu().contiguous()


dump_block("lw0", tft.layerwise_blocks[0])
dump_block("lw1", tft.layerwise_blocks[1])
dump_block("rf0", tft.refiner_blocks[0])
dump_block("rf1", tft.refiner_blocks[1])

print(
    "out_nomask:", tuple(out_nomask.shape),
    " out_masked:", tuple(out_masked.shape),
    " n_tensors:", len(dump),
    flush=True,
)
save_file(dump, OUT)
import os
print(f"OK dumped -> {OUT}  ({os.path.getsize(OUT)/1e6:.1f} MB)", flush=True)

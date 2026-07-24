# seedvr2_full_dit_parity.mojo — FULL SeedVR2-3B NaDiT forward parity probe.
# Runs full_dit_forward from the torch oracle's raw inputs (vid_in_raw [1024,33],
# txt_raw [58,5120], timestep 500) with COMPUTED axial-3D RoPE freqs (no oracle
# freqs dump), gating the final latent vs the tail oracle's vid_out_final [1024,16].
#
# This composes the WHOLE DiT: vid_in + txt_in + emb_in -> compute_dit_freqs ->
# 32-block dit_stack -> output tail. 32 bf16-accumulated blocks + computed-F32
# freqs (vs the model's bf16 freqs) -> expect cos ~0.99. PASS if cos >= 0.99.
#
#   weights: seedvr2_dit.safetensors (bf16, 6.8GB)
#   inputs:  dit_oracle.safetensors  vid_in_raw[1024,33] txt_raw[58,5120] (F32 -> bf16)
#            timestep = 500.0 ; grid T=4,H=16,W=16 (vid_in patchifies -> 256 tokens)
#   target:  dit_tail_oracle.safetensors  vid_out_final[1024,16]
#
# Run: cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#      pixi run mojo run -I . \
#        serenitymojo/models/dit/tests/seedvr2_full_dit_parity.mojo

from std.gpu.host import DeviceContext
from math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.models.sdxl.real_weights import load_bias
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.models.dit.seedvr2_dit import full_dit_forward

comptime WEIGHTS = "/home/alex/models/seedvr2-3b/seedvr2_dit.safetensors"
comptime ORACLE = "/home/alex/models/seedvr2-3b/dit_oracle.safetensors"
comptime TAIL = "/home/alex/models/seedvr2-3b/dit_tail_oracle.safetensors"


def _cos_maxabs(a: List[Float32], b: List[Float32]) -> Tuple[Float32, Float32]:
    var n = len(a)
    if len(b) < n:
        n = len(b)
    var dot: Float64 = 0.0
    var na: Float64 = 0.0
    var nb: Float64 = 0.0
    var maxd: Float32 = 0.0
    for i in range(n):
        var x = Float64(a[i])
        var y = Float64(b[i])
        dot += x * y
        na += x * x
        nb += y * y
        var d = a[i] - b[i]
        if d < 0.0:
            d = -d
        if d > maxd:
            maxd = d
    var denom = sqrt(na) * sqrt(nb)
    var cos: Float32 = 0.0
    if denom > 0.0:
        cos = Float32(dot / denom)
    return (cos, maxd)


def main() raises:
    var ctx = DeviceContext()
    print("ctx ok", Int(ctx.id()))

    var orc = SafeTensors.open(ORACLE)
    var st = SafeTensors.open(WEIGHTS)
    var tgt = SafeTensors.open(TAIL)

    # ── raw inputs (F32 oracle -> bf16) ─────────────────────────────────────────
    var vid_raw = cast_tensor(load_bias(orc, "vid_in_raw", ctx), STDtype.BF16, ctx)  # [1024,33]
    var txt_raw = cast_tensor(load_bias(orc, "txt_raw", ctx), STDtype.BF16, ctx)     # [58,5120]
    print("inputs loaded: vid_raw", vid_raw.shape()[0], vid_raw.shape()[1],
          " txt_raw", txt_raw.shape()[0], txt_raw.shape()[1])

    # ── FULL DiT forward (computed freqs, timestep 500, grid 4x16x16) ───────────
    var latent = full_dit_forward(vid_raw, txt_raw, Float32(500.0), 4, 16, 16, st, ctx)
    var ls = latent.shape()
    print("final latent shape:", ls[0], ls[1])

    # ── gate vs tail oracle vid_out_final [1024,16] ─────────────────────────────
    var out_h = latent.to_host(ctx)
    var ref_h = load_bias(tgt, "vid_out_final", ctx).to_host(ctx)
    var r = _cos_maxabs(out_h, ref_h)
    print("full DiT cos=", r[0], " max_abs=", r[1])
    if r[0] >= 0.99:
        print("RESULT: PASS")
    else:
        print("RESULT: FAIL")

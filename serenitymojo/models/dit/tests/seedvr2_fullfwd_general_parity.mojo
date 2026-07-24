# seedvr2_fullfwd_general_parity.mojo — GRID-GENERAL full SeedVR2-3B NaDiT forward.
# Runs full_dit_forward_general at PRE-patch grid (2,48,48) -> post-patch (2,24,24)
# (1152 vid tokens, uneven windows, even/odd 720pwin/720pswin alternation) and
# gates the final latent [4608,16] vs the torch full-forward oracle.
#
# 32 bf16-accumulated blocks at a NEW grid + computed-F32 freqs (vs model bf16) ->
# expect cos ~0.99. PASS if cos >= 0.99.
#
#   weights: seedvr2_dit.safetensors (bf16, 6.8GB)  -> load_dit_weights
#   input:   window_gen_oracle.safetensors  vid_in_raw[4608,33] (F32 -> bf16)
#            seedvr2_text_emb.safetensors    pos_emb[58,5120]     (F32 -> bf16)
#            timestep = 500.0 ; grid T=2,H=48,W=48 (post-patch 2,24,24)
#   target:  dit_fullfwd_2x24x24.safetensors  vid_out_final[4608,16]
#
# Run: cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#      pixi run mojo run -I . \
#        serenitymojo/models/dit/tests/seedvr2_fullfwd_general_parity.mojo

from std.gpu.host import DeviceContext
from math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.models.sdxl.real_weights import load_bias
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.models.dit.seedvr2_dit import (
    load_dit_weights,
    full_dit_forward_general,
)

comptime WEIGHTS = "/home/alex/models/seedvr2-3b/seedvr2_dit.safetensors"
comptime VIN = "/home/alex/models/seedvr2-3b/window_gen_oracle.safetensors"
comptime TXT = "/home/alex/models/seedvr2-3b/seedvr2_text_emb.safetensors"
comptime TGT = "/home/alex/models/seedvr2-3b/dit_fullfwd_2x24x24.safetensors"


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

    var st = SafeTensors.open(WEIGHTS)
    var vin = SafeTensors.open(VIN)
    var txtf = SafeTensors.open(TXT)
    var tgt = SafeTensors.open(TGT)

    # raw inputs (F32 oracle -> bf16)
    var vid_raw = cast_tensor(load_bias(vin, "vid_in_raw", ctx), STDtype.BF16, ctx)  # [4608,33]
    var txt_raw = cast_tensor(load_bias(txtf, "pos_emb", ctx), STDtype.BF16, ctx)    # [58,5120]
    print("inputs: vid_raw", vid_raw.shape()[0], vid_raw.shape()[1],
          " txt_raw", txt_raw.shape()[0], txt_raw.shape()[1])

    # preload all DiT weights once, then run the grid-general full forward
    var w = load_dit_weights(st, ctx)
    print("weights loaded")

    # PRE-patch grid (2,48,48) -> post-patch (2,24,24); txt_len=58
    var latent = full_dit_forward_general(vid_raw, txt_raw, 58, Float32(500.0), 2, 48, 48, w, ctx)
    var ls = latent.shape()
    print("final latent shape:", ls[0], ls[1])

    var out_h = latent.to_host(ctx)
    var ref_h = load_bias(tgt, "vid_out_final", ctx).to_host(ctx)
    var r = _cos_maxabs(out_h, ref_h)
    print("full DiT (2,24,24) cos=", r[0], " max_abs=", r[1])
    if r[0] >= 0.99:
        print("RESULT: PASS")
    else:
        print("RESULT: FAIL")

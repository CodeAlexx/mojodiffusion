# seedvr2_tail_parity.mojo — DiT OUTPUT TAIL parity probe for SeedVR2-3B.
# Runs the tail (vid_out_norm -> vid_out_ada -> vid_out.proj -> unpatchify) from
# the torch oracle's block-31 vid output + emb, gating three stages vs the
# self-consistent block-style AdaSingle reference (see dit_tail_oracle.py):
#   * vid_out_normed  [256,2560]  (after RMSNorm w/ affine, eps=1e-5)
#   * vid_out_proj    [256,64]    (after ada + Linear 2560->64)
#   * vid_out_final   [1024,16]   (after unpatchify)
# Each is single-op-depth; gate cos >= 0.999 (report the actual numbers).
#
#   weights: seedvr2_dit.safetensors (bf16)
#     vid_out_norm.weight[2560] vid_out_ada.out_shift[2560] vid_out_ada.out_scale[2560]
#     vid_out.proj.weight[64,2560] vid_out.proj.bias[64]
#   inputs:  dit_oracle.safetensors  block_31.out[256,2560] (bf16), b0in_emb[1,15360]
#   targets: dit_tail_oracle.safetensors vid_out_normed / vid_out_proj / vid_out_final
#
# Run: cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#      pixi run mojo run -I . \
#        serenitymojo/models/dit/tests/seedvr2_tail_parity.mojo

from max.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.models.sdxl.real_weights import load_bias
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.norm import rms_norm
from serenitymojo.ops.linear import linear_bias
from serenitymojo.ops.tensor_algebra import reshape
from serenitymojo.models.dit.seedvr2_dit import dit_out_tail, _emb_col, _mod_in

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

    # ── inputs (F32 oracle -> bf16) ─────────────────────────────────────────────
    var vid = cast_tensor(load_bias(orc, "block_31.out", ctx), STDtype.BF16, ctx)  # [256,2560]
    var emb = cast_tensor(load_bias(orc, "b0in_emb", ctx), STDtype.BF16, ctx)      # [1,15360]

    # ── tail weights (bf16) ─────────────────────────────────────────────────────
    var norm_w = load_bias(st, "vid_out_norm.weight", ctx)      # [2560]
    var ada_shift = load_bias(st, "vid_out_ada.out_shift", ctx)  # [2560]
    var ada_scale = load_bias(st, "vid_out_ada.out_scale", ctx)  # [2560]
    var proj_w = load_bias(st, "vid_out.proj.weight", ctx)       # [64,2560]
    var proj_b = load_bias(st, "vid_out.proj.bias", ctx)         # [64]

    # ── stage 1: RMSNorm (affine) ───────────────────────────────────────────────
    var normed = rms_norm(vid, norm_w, Float32(1e-5), ctx)       # [256,2560]

    # ── stage 2: ada "in" + proj (same rule as dit_out_tail) ────────────────────
    var emb2d = reshape(emb, [2560, 6], ctx)
    var shiftA = _emb_col(emb2d, 0, ctx)
    var scaleA = _emb_col(emb2d, 1, ctx)
    var moded = _mod_in(normed, scaleA, ada_scale, shiftA, ada_shift, ctx)
    var proj = linear_bias(moded, proj_w, proj_b, ctx)          # [256,64]

    # ── stage 3: full tail -> final latent [1024,16] ────────────────────────────
    var final = dit_out_tail(vid, emb, norm_w, ada_shift, ada_scale, proj_w, proj_b, ctx)
    var fs = final.shape()
    print("final latent shape:", fs[0], fs[1])

    # ── gate all three stages ───────────────────────────────────────────────────
    var r_norm = _cos_maxabs(normed.to_host(ctx), load_bias(tgt, "vid_out_normed", ctx).to_host(ctx))
    var r_proj = _cos_maxabs(proj.to_host(ctx), load_bias(tgt, "vid_out_proj", ctx).to_host(ctx))
    var r_final = _cos_maxabs(final.to_host(ctx), load_bias(tgt, "vid_out_final", ctx).to_host(ctx))

    print("normed cos=", r_norm[0], " max_abs=", r_norm[1])
    print("proj   cos=", r_proj[0], " max_abs=", r_proj[1])
    print("final  cos=", r_final[0], " max_abs=", r_final[1])

    if r_norm[0] >= 0.999 and r_proj[0] >= 0.999 and r_final[0] >= 0.999:
        print("RESULT: PASS")
    else:
        print("RESULT: FAIL")

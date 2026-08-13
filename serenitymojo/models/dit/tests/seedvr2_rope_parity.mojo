# seedvr2_rope_parity.mojo — Sub-chunk parity probe for SeedVR2-3B NaSwinAttention
# 3D rotary-embedding APPLY (rotate-half, adjacent-pair) onto normed q/k, using
# PRECOMPUTED freqs (oracle tensors). INFERENCE ONLY, pure Mojo + MAX, GPU, BF16.
#
# Convention = rotary_embedding_torch apply_rotary_emb(freqs, t):
#   rot_dim=126 -> first 126 dims rotated (adjacent pairs), last 2 pass through.
#
# Two streams:
#   VID: q/k = b0at_nq.vid / b0at_nk.vid [256,20,128] (bf16 in), freqs
#        b0rope_vid_freqs [256,126] F32 -> gate vs b0rope_vq / b0rope_vk.
#   TXT: q/k = b0at_nq.txt / b0at_nk.txt [58,20,128] TILED 4x along token axis
#        (58->232, [q;q;q;q] via concat) then freqs b0rope_txt_freqs [232,126]
#        -> gate vs b0rope_tq / b0rope_tk [232,20,128].
#
# BF16 compute -> PASS bar cos >= 0.999.
#
# Run: cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#      pixi run mojo run -I . \
#        serenitymojo/models/dit/tests/seedvr2_rope_parity.mojo

from max.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.models.sdxl.real_weights import load_bias
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import concat
from serenitymojo.models.dit.seedvr2_dit import apply_rope3d

comptime ORACLE = "/home/alex/models/seedvr2-3b/dit_oracle.safetensors"


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

    # ── VID stream (direct: no tile) ────────────────────────────────────────────
    var vq_in = cast_tensor(load_bias(orc, "b0at_nq.vid", ctx), STDtype.BF16, ctx)
    var vk_in = cast_tensor(load_bias(orc, "b0at_nk.vid", ctx), STDtype.BF16, ctx)
    var v_freqs = load_bias(orc, "b0rope_vid_freqs", ctx)   # [256,126] F32

    var vq_out = apply_rope3d(vq_in, v_freqs, ctx)
    var vk_out = apply_rope3d(vk_in, v_freqs, ctx)
    var vqs = vq_out.shape()
    print("vid q out shape:", vqs[0], vqs[1], vqs[2])

    var vq_h = vq_out.to_host(ctx)
    var vk_h = vk_out.to_host(ctx)
    var vq_ref = load_bias(orc, "b0rope_vq", ctx).to_host(ctx)
    var vk_ref = load_bias(orc, "b0rope_vk", ctx).to_host(ctx)
    var vq_r = _cos_maxabs(vq_h, vq_ref)
    var vk_r = _cos_maxabs(vk_h, vk_ref)

    # ── TXT stream (tile 4x along token axis, then apply) ───────────────────────
    var tq58 = cast_tensor(load_bias(orc, "b0at_nq.txt", ctx), STDtype.BF16, ctx)
    var tk58 = cast_tensor(load_bias(orc, "b0at_nk.txt", ctx), STDtype.BF16, ctx)
    # 4x repeat of the whole 58-token block, in order: [q;q;q;q] -> [232,20,128]
    var tq232 = concat(0, ctx, tq58, tq58, tq58, tq58)
    var tk232 = concat(0, ctx, tk58, tk58, tk58, tk58)
    var tqs = tq232.shape()
    print("txt q tiled shape:", tqs[0], tqs[1], tqs[2])
    var t_freqs = load_bias(orc, "b0rope_txt_freqs", ctx)   # [232,126] F32

    var tq_out = apply_rope3d(tq232, t_freqs, ctx)
    var tk_out = apply_rope3d(tk232, t_freqs, ctx)

    var tq_h = tq_out.to_host(ctx)
    var tk_h = tk_out.to_host(ctx)
    var tq_ref = load_bias(orc, "b0rope_tq", ctx).to_host(ctx)
    var tk_ref = load_bias(orc, "b0rope_tk", ctx).to_host(ctx)
    var tq_r = _cos_maxabs(tq_h, tq_ref)
    var tk_r = _cos_maxabs(tk_h, tk_ref)

    print(
        "vid_q cos=", vq_r[0], " vid_k cos=", vk_r[0],
        " txt_q cos=", tq_r[0], " txt_k cos=", tk_r[0],
    )
    print(
        "max_abs: vid_q=", vq_r[1], " vid_k=", vk_r[1],
        " txt_q=", tq_r[1], " txt_k=", tk_r[1],
    )
    if (
        vq_r[0] >= 0.999 and vk_r[0] >= 0.999
        and tq_r[0] >= 0.999 and tk_r[0] >= 0.999
    ):
        print("RESULT: PASS")
    else:
        print("RESULT: FAIL")

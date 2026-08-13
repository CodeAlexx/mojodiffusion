# seedvr2_winattn_parity.mojo — Sub-chunk parity probe for SeedVR2-3B
# NaSwinAttention tail: window-joint attention + proj_out (dual-stream vid+txt),
# for the CONCRETE block-0 window structure (4 temporal windows, 64 vid + 58 txt).
#
# Feeds the torch oracle's exact block-0 post-rope q/k (+ un-normed v extracted
# from the packed qkv) through the pure-Mojo window_attn_projout op and compares
# proj_out(vid) and proj_out(txt) against the torch bf16 reference (stored F32).
# BF16 compute, so PASS bar is cos >= 0.999.
#
#   weights: seedvr2_dit.safetensors (bf16)
#     blocks.0.attn.proj_out.vid.weight [2560,2560] / .vid.bias [2560]
#     blocks.0.attn.proj_out.txt.weight [2560,2560] / .txt.bias [2560]
#   oracle:  dit_oracle.safetensors (F32 refs -> bf16)
#     inputs  b0rope_vq/vk [256,20,128]  b0rope_tq/tk [232,20,128] (4x58 window-major)
#             b0at_qkv.vid [256,7680]    b0at_qkv.txt [58,7680]     (v = cols 5120:7680)
#     targets b0at_projout.vid [256,2560]  b0at_projout.txt [58,2560]
#
# Run: cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#      pixi run mojo run -I . \
#        serenitymojo/models/dit/tests/seedvr2_winattn_parity.mojo

from max.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.models.sdxl.real_weights import load_bias
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import reshape, slice, concat
from serenitymojo.models.dit.seedvr2_dit import window_attn_projout

comptime WEIGHTS = "/home/alex/models/seedvr2-3b/seedvr2_dit.safetensors"
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
    var st = SafeTensors.open(WEIGHTS)

    # ── post-rope q/k (F32 oracle -> bf16) ─────────────────────────────────────
    var vid_q = cast_tensor(load_bias(orc, "b0rope_vq", ctx), STDtype.BF16, ctx)  # [256,20,128]
    var vid_k = cast_tensor(load_bias(orc, "b0rope_vk", ctx), STDtype.BF16, ctx)
    var txt_q = cast_tensor(load_bias(orc, "b0rope_tq", ctx), STDtype.BF16, ctx)  # [232,20,128]
    var txt_k = cast_tensor(load_bias(orc, "b0rope_tk", ctx), STDtype.BF16, ctx)

    # ── v (un-normed, un-roped): cols [5120:7680] of the packed qkv ────────────
    # vid: b0at_qkv.vid [256,7680] -> [256,2560] -> [256,20,128]
    var vqkv = load_bias(orc, "b0at_qkv.vid", ctx)                 # F32 [256,7680]
    var vv_cols = slice(vqkv, 1, 5120, 2560, ctx)                  # [256,2560]
    var vid_v = cast_tensor(reshape(vv_cols, [256, 20, 128], ctx), STDtype.BF16, ctx)

    # txt: b0at_qkv.txt [58,7680] -> [58,2560] -> [58,20,128], TILE 4x -> [232,20,128]
    var tqkv = load_bias(orc, "b0at_qkv.txt", ctx)                 # F32 [58,7680]
    var tv_cols = slice(tqkv, 1, 5120, 2560, ctx)                  # [58,2560]
    var tv1 = cast_tensor(reshape(tv_cols, [58, 20, 128], ctx), STDtype.BF16, ctx)  # [58,20,128]
    var tv_a = tv1.clone(ctx)
    var tv_b = tv1.clone(ctx)
    var tv_c = tv1.clone(ctx)
    var txt_v = concat(0, ctx, tv1, tv_a, tv_b, tv_c)              # [232,20,128] window-major

    # ── proj_out weights (bf16) ────────────────────────────────────────────────
    var po_vid_w = load_bias(st, "blocks.0.attn.proj_out.vid.weight", ctx)
    var po_vid_b = load_bias(st, "blocks.0.attn.proj_out.vid.bias", ctx)
    var po_txt_w = load_bias(st, "blocks.0.attn.proj_out.txt.weight", ctx)
    var po_txt_b = load_bias(st, "blocks.0.attn.proj_out.txt.bias", ctx)

    var res = window_attn_projout(
        vid_q, vid_k, vid_v, txt_q, txt_k, txt_v,
        po_vid_w, po_vid_b, po_txt_w, po_txt_b, ctx,
    )
    var vs = res[0].shape()
    var ts = res[1].shape()
    print("vid_out shape:", vs[0], vs[1], " txt_out shape:", ts[0], ts[1])

    var vid_h = res[0].to_host(ctx)
    var txt_h = res[1].to_host(ctx)
    var vid_ref = load_bias(orc, "b0at_projout.vid", ctx).to_host(ctx)
    var txt_ref = load_bias(orc, "b0at_projout.txt", ctx).to_host(ctx)

    var vres = _cos_maxabs(vid_h, vid_ref)
    var tres = _cos_maxabs(txt_h, txt_ref)

    print("vid_out cos=", vres[0], " max_abs=", vres[1])
    print("txt_out cos=", tres[0], " max_abs=", tres[1])
    if vres[0] >= 0.999 and tres[0] >= 0.999:
        print("RESULT: PASS")
    else:
        print("RESULT: FAIL")

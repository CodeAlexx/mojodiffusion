# seedvr2_winattn_general_parity.mojo — GENERAL (arbitrary-resolution) window
# attention parity for SeedVR2-3B NaSwinAttention.
#
# Gates the pure-Mojo general path at post-patch grid (T,H,W)=(2,24,24), which
# produces 8 UNEVEN windows (vid counts 400/400/80/80/80/80/16/16, txt 58 each):
#   (1) compute_windows(2,24,24): partition + per-window shapes vs oracle
#       (win_partition, win_shapes).
#   (2) attn_qkv_norm(vid) + attn_qkv_norm(txt) -> window_attn_projout_general,
#       gated vs b0_attn.{vid,txt}. BF16 compute -> PASS bar cos >= 0.999.
#
#   weights: seedvr2_dit.safetensors (bf16) blocks.0.attn.*
#   oracle:  window_gen_oracle.safetensors (F32)
#     inputs  b0attn_in.vid [1152,2560]  b0attn_in.txt [58,2560]
#     targets b0_attn.vid   [1152,2560]  b0_attn.txt   [58,2560]
#     windows win_partition [1152] win_shapes [8,3]
#
# Run: cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#      pixi run mojo run -I . \
#        serenitymojo/models/dit/tests/seedvr2_winattn_general_parity.mojo

from std.gpu.host import DeviceContext
from math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.models.sdxl.real_weights import load_bias
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.models.dit.seedvr2_dit import (
    attn_qkv_norm,
    compute_windows,
    window_attn_projout_general,
)

comptime WEIGHTS = "/home/alex/models/seedvr2-3b/seedvr2_dit.safetensors"
comptime ORACLE = "/home/alex/models/seedvr2-3b/window_gen_oracle.safetensors"


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

    var TXT_LEN = 58

    # ── (1) window computation gate ─────────────────────────────────────────────
    var plan = compute_windows(2, 24, 24)
    print("windows:", plan.win_count, " L:", plan.L)

    var part_ref = load_bias(orc, "win_partition", ctx).to_host(ctx)  # [1152] F32
    var shp_ref = load_bias(orc, "win_shapes", ctx).to_host(ctx)      # [8,3] F32 flat

    var part_ok = True
    if len(part_ref) != plan.L:
        part_ok = False
    else:
        for i in range(plan.L):
            if plan.partition[i] != Int(part_ref[i] + 0.5):
                part_ok = False
                break

    var shapes_ok = True
    if len(shp_ref) != plan.win_count * 3:
        shapes_ok = False
    else:
        for w in range(plan.win_count):
            var rf = Int(shp_ref[w * 3 + 0] + 0.5)
            var rh = Int(shp_ref[w * 3 + 1] + 0.5)
            var rw = Int(shp_ref[w * 3 + 2] + 0.5)
            if plan.wf[w] != rf or plan.wh[w] != rh or plan.ww[w] != rw:
                shapes_ok = False

    # print computed shapes/counts for the record
    var shp_str = String("shapes: ")
    for w in range(plan.win_count):
        shp_str += (
            String("(") + String(plan.wf[w]) + "," + String(plan.wh[w])
            + "," + String(plan.ww[w]) + ")c" + String(plan.counts[w]) + " "
        )
    print(shp_str)
    print("partition match:", part_ok, " shapes match:", shapes_ok)

    # ── (2) attention gate ──────────────────────────────────────────────────────
    # attn inputs (F32 oracle -> bf16), ORIGINAL grid order
    var vid_in = cast_tensor(load_bias(orc, "b0attn_in.vid", ctx), STDtype.BF16, ctx)  # [1152,2560]
    var txt_in = cast_tensor(load_bias(orc, "b0attn_in.txt", ctx), STDtype.BF16, ctx)  # [58,2560]

    # block-0 attn weights (bf16)
    var qkv_vid_w = load_bias(st, "blocks.0.attn.proj_qkv.vid.weight", ctx)
    var nq_vid_w = load_bias(st, "blocks.0.attn.norm_q.vid.weight", ctx)
    var nk_vid_w = load_bias(st, "blocks.0.attn.norm_k.vid.weight", ctx)
    var qkv_txt_w = load_bias(st, "blocks.0.attn.proj_qkv.txt.weight", ctx)
    var nq_txt_w = load_bias(st, "blocks.0.attn.norm_q.txt.weight", ctx)
    var nk_txt_w = load_bias(st, "blocks.0.attn.norm_k.txt.weight", ctx)
    var po_vid_w = load_bias(st, "blocks.0.attn.proj_out.vid.weight", ctx)
    var po_vid_b = load_bias(st, "blocks.0.attn.proj_out.vid.bias", ctx)
    var po_txt_w = load_bias(st, "blocks.0.attn.proj_out.txt.weight", ctx)
    var po_txt_b = load_bias(st, "blocks.0.attn.proj_out.txt.bias", ctx)

    # QKV + QK-norm for each stream
    var vqkv = attn_qkv_norm(vid_in, qkv_vid_w, nq_vid_w, nk_vid_w, ctx)  # (q,k,v) [1152,20,128]
    var tqkv = attn_qkv_norm(txt_in, qkv_txt_w, nq_txt_w, nk_txt_w, ctx)  # (q,k,v) [58,20,128]

    var res = window_attn_projout_general(
        vqkv[0], vqkv[1], vqkv[2],
        tqkv[0], tqkv[1], tqkv[2],
        plan, TXT_LEN,
        po_vid_w, po_vid_b, po_txt_w, po_txt_b, ctx,
    )
    var vs = res[0].shape()
    var ts = res[1].shape()
    print("vid_out shape:", vs[0], vs[1], " txt_out shape:", ts[0], ts[1])

    var vid_h = res[0].to_host(ctx)
    var txt_h = res[1].to_host(ctx)
    var vid_ref = load_bias(orc, "b0_attn.vid", ctx).to_host(ctx)
    var txt_ref = load_bias(orc, "b0_attn.txt", ctx).to_host(ctx)

    var vres = _cos_maxabs(vid_h, vid_ref)
    var tres = _cos_maxabs(txt_h, txt_ref)

    print("vid_out cos=", vres[0], " max_abs=", vres[1])
    print("txt_out cos=", tres[0], " max_abs=", tres[1])

    var attn_ok = vres[0] >= 0.999 and tres[0] >= 0.999
    if part_ok and shapes_ok and attn_ok:
        print("RESULT: PASS")
    else:
        print("RESULT: FAIL")

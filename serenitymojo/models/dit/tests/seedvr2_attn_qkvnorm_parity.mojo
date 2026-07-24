# seedvr2_attn_qkvnorm_parity.mojo — Sub-chunk parity probe for SeedVR2-3B
# NaMMAttention entry: QKV projection + QK-RMSNorm (dual-stream vid + txt).
#
# Feeds the torch oracle's exact block-0 attention inputs through the pure-Mojo
# attn_qkv_norm op and compares normed q,k for BOTH streams against the torch
# bf16 reference (stored F32). BF16 compute, so PASS bar is cos >= 0.999.
#
#   weights: seedvr2_dit.safetensors (bf16)
#     blocks.0.attn.proj_qkv.vid.weight [7680,2560]  (NO bias)
#     blocks.0.attn.proj_qkv.txt.weight [7680,2560]  (NO bias)
#     blocks.0.attn.norm_q.vid.weight [128] / norm_q.txt.weight [128]
#     blocks.0.attn.norm_k.vid.weight [128] / norm_k.txt.weight [128]
#   oracle:  dit_oracle.safetensors
#     inputs  b0attn_in.vid [256,2560] / b0attn_in.txt [58,2560]  (F32 -> bf16)
#     targets b0at_nq.vid [256,20,128] / b0at_nq.txt [58,20,128]
#             b0at_nk.vid [256,20,128] / b0at_nk.txt [58,20,128]
#
# Run: cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#      pixi run mojo run -I . \
#        serenitymojo/models/dit/tests/seedvr2_attn_qkvnorm_parity.mojo

from std.gpu.host import DeviceContext
from math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.models.sdxl.real_weights import load_bias
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.models.dit.seedvr2_dit import attn_qkv_norm

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

    var st = SafeTensors.open(WEIGHTS)
    var orc = SafeTensors.open(ORACLE)

    # ── vid stream ────────────────────────────────────────────────────────────
    var vin_f32 = load_bias(orc, "b0attn_in.vid", ctx)          # [256,2560] F32
    var vin = cast_tensor(vin_f32, STDtype.BF16, ctx)           # -> bf16
    var v_qkv_w = load_bias(st, "blocks.0.attn.proj_qkv.vid.weight", ctx)
    var v_nq_w = load_bias(st, "blocks.0.attn.norm_q.vid.weight", ctx)
    var v_nk_w = load_bias(st, "blocks.0.attn.norm_k.vid.weight", ctx)

    var v_res = attn_qkv_norm(vin, v_qkv_w, v_nq_w, v_nk_w, ctx)
    var vqs = v_res[0].shape()
    print("vid q shape:", vqs[0], vqs[1], vqs[2])

    var v_q_h = v_res[0].to_host(ctx)
    var v_k_h = v_res[1].to_host(ctx)
    var v_q_ref = load_bias(orc, "b0at_nq.vid", ctx).to_host(ctx)
    var v_k_ref = load_bias(orc, "b0at_nk.vid", ctx).to_host(ctx)
    var vq_res = _cos_maxabs(v_q_h, v_q_ref)
    var vk_res = _cos_maxabs(v_k_h, v_k_ref)

    # ── txt stream ────────────────────────────────────────────────────────────
    var tin_f32 = load_bias(orc, "b0attn_in.txt", ctx)          # [58,2560] F32
    var tin = cast_tensor(tin_f32, STDtype.BF16, ctx)           # -> bf16
    var t_qkv_w = load_bias(st, "blocks.0.attn.proj_qkv.txt.weight", ctx)
    var t_nq_w = load_bias(st, "blocks.0.attn.norm_q.txt.weight", ctx)
    var t_nk_w = load_bias(st, "blocks.0.attn.norm_k.txt.weight", ctx)

    var t_res = attn_qkv_norm(tin, t_qkv_w, t_nq_w, t_nk_w, ctx)
    var tqs = t_res[0].shape()
    print("txt q shape:", tqs[0], tqs[1], tqs[2])

    var t_q_h = t_res[0].to_host(ctx)
    var t_k_h = t_res[1].to_host(ctx)
    var t_q_ref = load_bias(orc, "b0at_nq.txt", ctx).to_host(ctx)
    var t_k_ref = load_bias(orc, "b0at_nk.txt", ctx).to_host(ctx)
    var tq_res = _cos_maxabs(t_q_h, t_q_ref)
    var tk_res = _cos_maxabs(t_k_h, t_k_ref)

    print(
        "vid_q cos=", vq_res[0], " vid_k cos=", vk_res[0],
        " txt_q cos=", tq_res[0], " txt_k cos=", tk_res[0],
    )
    print(
        "max_abs: vid_q=", vq_res[1], " vid_k=", vk_res[1],
        " txt_q=", tq_res[1], " txt_k=", tk_res[1],
    )
    if (
        vq_res[0] >= 0.999 and vk_res[0] >= 0.999
        and tq_res[0] >= 0.999 and tk_res[0] >= 0.999
    ):
        print("RESULT: PASS")
    else:
        print("RESULT: FAIL")

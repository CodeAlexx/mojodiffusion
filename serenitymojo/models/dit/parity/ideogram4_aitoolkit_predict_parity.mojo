# models/dit/parity/ideogram4_aitoolkit_predict_parity.mojo — ai-toolkit ORACLE gate
# for the Ideogram-4 FULL FORWARD STACK ("predict"): input_proj -> 34 blocks ->
# final_layer -> velocity, plus MRoPE.
#
# Proves the serenitymojo Ideogram-4 FULL forward (models/dit/ideogram4_dit.mojo)
# matches the ai-toolkit PRODUCTION predict_velocity composition, NOT the invalid
# ideogram4-ref path. Sibling of the verified per-BLOCK ai-toolkit gate
#   autograd_v2/tests/ideogram4_block_aitoolkit_parity.mojo (cos>=0.99998).
#
# Reads the NEW ai-toolkit fixture
#   serenitymojo/models/dit/parity/ideogram4_aitoolkit_predict.safetensors
# (from ideogram4_aitoolkit_predict_oracle.py) and the SAME fp8 checkpoint.
#
# Captures compared (cos >= 0.999, FAIL-LOUD on any miss):
#   * mrope_cos / mrope_sin   (build_ideogram4_mrope vs the ai-toolkit f32 MRoPE)
#   * block0_out / block16_out / block33_out  (instrumented forward, same helpers)
#   * transformer_out         (raw model output, before -velocity reshape)
#   * velocity                (toolkit velocity = -image_velocity, reshaped)
#
# Predict wiring (matches pipeline.predict_velocity):
#   - the model is fed model_t = 1 - t (t=0.7 toolkit -> model_t=0.3),
#   - velocity = -(transformer_out[:, NTEXT:] reshaped to (gh,gw,c)).
#
# NOTE ON MRoPE: the ai-toolkit production MRoPE keeps inv_freq in float32; the
# mojo build_ideogram4_mrope bf16-ROUNDS inv_freq (ideogram4-ref convention). This
# gate MEASURES whether that bf16-inv mojo MRoPE still matches the f32-inv oracle
# at the IMAGE_POSITION_OFFSET=65536 positions. The cos table is the evidence.
#
# Run (oracle FIRST if the fixture is missing):
#   /home/alex/serenityflow-v2/.venv/bin/python \
#     serenitymojo/models/dit/parity/ideogram4_aitoolkit_predict_oracle.py
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/models/dit/parity/ideogram4_aitoolkit_predict_parity.mojo

from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.linear import linear
from serenitymojo.ops.activations import silu
from serenitymojo.ops.norm import rms_norm, layer_norm_no_affine
from serenitymojo.ops.tensor_algebra import (
    mul, add, add_scalar, reshape, slice, gather_rows, permute,
)
from serenitymojo.parity import ParityHarness

from serenitymojo.models.dit.ideogram4_dit import (
    ideogram4_forward,
    load_w_fp8,
    load_w_bf16,
    ideogram4_t_embedding,
    ideogram4_block,
)
from serenitymojo.models.dit.ideogram4_mrope import build_ideogram4_mrope

comptime T = "/home/alex/.serenity/models/ideogram-4-fp8/transformer/diffusion_pytorch_model.safetensors"
comptime FX = (
    "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/"
    "ideogram4_aitoolkit_predict.safetensors"
)

# Geometry — fixed by the oracle (GH=GW=16 -> NIMG=256, NTEXT=4 -> L=260, t=0.7).
comptime S = 260
comptime NTEXT = 4
comptime GH = 16
comptime GW = 16
comptime HIDDEN = 4608
comptime HEADS = 18
comptime HEAD_DIM = 256
comptime NUM_LAYERS = 34
comptime BAR = 0.999


# ── instrumented prefinal-hidden forward: a faithful inline copy of
# ideogram4_forward_prefinal_hidden that ALSO records block {0,16,33} outputs.
# Reuses the SAME production embedder helpers + ideogram4_block so only the
# capture is added. Returns (prefinal_hidden, block0, block16, block33).
def _forward_capture[Sx: Int](
    st: ShardedSafeTensors,
    x_in: Tensor, llm_in: Tensor, t_in: Tensor, indicator: Tensor,
    cosf: Tensor, sinf: Tensor,
    ctx: DeviceContext,
) raises -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    var L = x_in.shape()[1]

    # masks from indicator (host-built): llm=3, image=2  (== production)
    var ind_h = indicator.to_host(ctx)
    var llm_mask_v = List[Float32]()
    var img_mask_v = List[Float32]()
    var img_ids = List[Int]()
    for i in range(L):
        var vi = ind_h[i]
        llm_mask_v.append(Float32(1.0) if (vi > 2.5 and vi < 3.5) else Float32(0.0))
        var is_img = (vi > 1.5 and vi < 2.5)
        img_mask_v.append(Float32(1.0) if is_img else Float32(0.0))
        img_ids.append(1 if is_img else 0)
    var llm_mask = Tensor.from_host(llm_mask_v, [1, L, 1], STDtype.BF16, ctx)
    var img_mask = Tensor.from_host(img_mask_v, [1, L, 1], STDtype.BF16, ctx)

    var llm = mul(llm_in, llm_mask, ctx)
    var x = mul(x_in, img_mask, ctx)
    var ipw = load_w_fp8(st, "input_proj.weight", ctx)
    var ipb = load_w_bf16(st, "input_proj.bias", ctx)
    x = mul(linear(x, ipw, Optional[Tensor](ipb.clone(ctx)), ctx), img_mask, ctx)

    var miw = load_w_fp8(st, "t_embedding.mlp_in.weight", ctx)
    var mib = load_w_bf16(st, "t_embedding.mlp_in.bias", ctx)
    var mow = load_w_fp8(st, "t_embedding.mlp_out.weight", ctx)
    var mob = load_w_bf16(st, "t_embedding.mlp_out.bias", ctx)
    var t_cond = reshape(ideogram4_t_embedding(t_in, HIDDEN, miw, mib, mow, mob, ctx), [1, 1, HIDDEN], ctx)
    var apw = load_w_fp8(st, "adaln_proj.weight", ctx)
    var apb = load_w_bf16(st, "adaln_proj.bias", ctx)
    var adaln_input = silu(linear(t_cond, apw, Optional[Tensor](apb.clone(ctx)), ctx), ctx)

    var lcn = load_w_bf16(st, "llm_cond_norm.weight", ctx)
    llm = rms_norm(llm, lcn, Float32(1.0e-6), ctx)
    var lcpw = load_w_fp8(st, "llm_cond_proj.weight", ctx)
    var lcpb = load_w_bf16(st, "llm_cond_proj.bias", ctx)
    llm = mul(linear(llm, lcpw, Optional[Tensor](lcpb.clone(ctx)), ctx), llm_mask, ctx)

    var h = add(x, llm, ctx)
    var eii = load_w_bf16(st, "embed_image_indicator.weight", ctx)
    var iemb = reshape(gather_rows(eii, img_ids, ctx), [1, L, HIDDEN], ctx)
    h = add(h, iemb, ctx)

    var b0 = h.clone(ctx)
    var b16 = h.clone(ctx)
    var b33 = h.clone(ctx)
    for li in range(NUM_LAYERS):
        var p = String("layers.") + String(li) + "."
        var amw = load_w_fp8(st, p + "adaln_modulation.weight", ctx)
        var amb = load_w_bf16(st, p + "adaln_modulation.bias", ctx)
        var an1 = load_w_bf16(st, p + "attention_norm1.weight", ctx)
        var an2 = load_w_bf16(st, p + "attention_norm2.weight", ctx)
        var fn1 = load_w_bf16(st, p + "ffn_norm1.weight", ctx)
        var fn2 = load_w_bf16(st, p + "ffn_norm2.weight", ctx)
        var qkv = load_w_fp8(st, p + "attention.qkv.weight", ctx)
        var ow = load_w_fp8(st, p + "attention.o.weight", ctx)
        var nq = load_w_bf16(st, p + "attention.norm_q.weight", ctx)
        var nk = load_w_bf16(st, p + "attention.norm_k.weight", ctx)
        var w1 = load_w_fp8(st, p + "feed_forward.w1.weight", ctx)
        var w2 = load_w_fp8(st, p + "feed_forward.w2.weight", ctx)
        var w3 = load_w_fp8(st, p + "feed_forward.w3.weight", ctx)
        h = ideogram4_block[Sx](
            h, adaln_input, cosf, sinf, amw, amb, an1, an2, fn1, fn2,
            qkv, ow, nq, nk, w1, w2, w3, HEADS, HEAD_DIM, HIDDEN, ctx)
        if li == 0:
            b0 = h.clone(ctx)
        elif li == 16:
            b16 = h.clone(ctx)
        elif li == 33:
            b33 = h.clone(ctx)

    var fmw = load_w_fp8(st, "final_layer.adaln_modulation.weight", ctx)
    var fmb = load_w_bf16(st, "final_layer.adaln_modulation.bias", ctx)
    var fscale = add_scalar(linear(silu(adaln_input, ctx), fmw, Optional[Tensor](fmb.clone(ctx)), ctx), Float32(1.0), ctx)
    var hn = mul(layer_norm_no_affine(h, Float32(1.0e-6), ctx), fscale, ctx)
    return (hn^, b0^, b16^, b33^)


def main() raises:
    var ctx = DeviceContext()
    print("=== Ideogram-4 PREDICT (full forward) ai-toolkit ORACLE parity (cos >= 0.999) ===")

    var st = ShardedSafeTensors.open(T)
    var fx = ShardedSafeTensors.open(String(FX))

    # ── inputs (feed byte-identical tensors the oracle used) ──
    var x = cast_tensor(Tensor.from_view(fx.tensor_view("in.x_packed"), ctx), STDtype.BF16, ctx)
    var llm = cast_tensor(Tensor.from_view(fx.tensor_view("in.llm_full"), ctx), STDtype.BF16, ctx)
    var model_t = Tensor.from_view(fx.tensor_view("in.model_t"), ctx)          # 1 - t (F32)
    var ind = Tensor.from_view(fx.tensor_view("in.indicator_f32"), ctx)
    var pos = Tensor.from_view(fx.tensor_view("in.position_ids_f32"), ctx)

    var h = ParityHarness(BAR)
    var n_fail = 0

    # ── MRoPE (mojo bf16-inv builder vs ai-toolkit f32-inv oracle) ──
    var sec = [24, 20, 20]
    var cs = build_ideogram4_mrope(pos, HEAD_DIM, sec, Float32(5000000.0), ctx, STDtype.BF16)
    var ref_cos = Tensor.from_view_as_f32(fx.tensor_view("out.mrope_cos"), ctx).to_host(ctx)
    var ref_sin = Tensor.from_view_as_f32(fx.tensor_view("out.mrope_sin"), ctx).to_host(ctx)
    var r_cos = h.compare(cs[0], ref_cos, ctx)
    var r_sin = h.compare(cs[1], ref_sin, ctx)
    print("  mrope_cos          ", r_cos)
    print("  mrope_sin          ", r_sin)
    if not r_cos.passed:
        n_fail += 1
    if not r_sin.passed:
        n_fail += 1

    # ── instrumented full forward (per-block capture + prefinal hidden) ──
    var caps = _forward_capture[S](st, x, llm, model_t, ind, cs[0], cs[1], ctx)
    # caps = (prefinal_hidden, block0, block16, block33); read by reference.

    var ref_b0 = Tensor.from_view_as_f32(fx.tensor_view("out.block0_out"), ctx).to_host(ctx)
    var ref_b16 = Tensor.from_view_as_f32(fx.tensor_view("out.block16_out"), ctx).to_host(ctx)
    var ref_b33 = Tensor.from_view_as_f32(fx.tensor_view("out.block33_out"), ctx).to_host(ctx)
    var r_b0 = h.compare(caps[1], ref_b0, ctx)
    var r_b16 = h.compare(caps[2], ref_b16, ctx)
    var r_b33 = h.compare(caps[3], ref_b33, ctx)
    print("  block0_out         ", r_b0)
    print("  block16_out        ", r_b16)
    print("  block33_out        ", r_b33)
    if not r_b0.passed:
        n_fail += 1
    if not r_b16.passed:
        n_fail += 1
    if not r_b33.passed:
        n_fail += 1

    # ── transformer_out (apply final_layer.linear to the prefinal hidden) ──
    var flw = load_w_fp8(st, "final_layer.linear.weight", ctx)
    var flb = load_w_bf16(st, "final_layer.linear.bias", ctx)
    var tout = cast_tensor(linear(caps[0], flw, Optional[Tensor](flb.clone(ctx)), ctx), STDtype.F32, ctx)  # [1,L,128]
    var ref_tout = Tensor.from_view_as_f32(fx.tensor_view("out.transformer_out"), ctx).to_host(ctx)
    var r_tout = h.compare(tout, ref_tout, ctx)
    print("  transformer_out    ", r_tout)
    if not r_tout.passed:
        n_fail += 1

    # ── velocity = -(transformer_out[:, NTEXT:] reshaped (gh,gw,c) -> (c,gh,gw)) ──
    # full ideogram4_forward (sanity: identical to `tout`) drives the velocity.
    var out = ideogram4_forward[S](st, x, llm, model_t, ind, cs[0], cs[1], NUM_LAYERS, HEADS, HEAD_DIM, HIDDEN, ctx)
    var img = slice(out, 1, NTEXT, S - NTEXT, ctx)            # [1, NIMG, 128]
    var img4 = reshape(img, [1, GH, GW, 128], ctx)            # (gh, gw, c)
    var img_chw = permute(img4, [0, 3, 1, 2], ctx)           # (c, gh, gw) = (1,128,gh,gw)
    var vel = mul(img_chw, Tensor.from_host(_neg_ones(img_chw.numel()), img_chw.shape(), STDtype.F32, ctx), ctx)
    var ref_vel = Tensor.from_view_as_f32(fx.tensor_view("out.velocity"), ctx).to_host(ctx)
    var r_vel = h.compare(vel, ref_vel, ctx)
    print("  velocity           ", r_vel)
    if not r_vel.passed:
        n_fail += 1

    # ── DIAGNOSTIC A/B: re-run the SAME forward fed the ORACLE's cos/sin (its
    # f32-inv_freq MRoPE, loaded as bf16) instead of the mojo bf16-inv builder.
    # If block16/block33/transformer_out now PASS, it proves the mojo forward
    # COMPOSITION is faithful and the ONLY divergence is the MRoPE inv_freq dtype
    # (mojo bf16-rounds inv_freq; ai-toolkit production keeps it float32).
    print("--- diagnostic: forward fed the oracle's (f32-inv) cos/sin -------------")
    var ocos = cast_tensor(Tensor.from_view(fx.tensor_view("out.mrope_cos"), ctx), STDtype.BF16, ctx)
    var osin = cast_tensor(Tensor.from_view(fx.tensor_view("out.mrope_sin"), ctx), STDtype.BF16, ctx)
    var caps2 = _forward_capture[S](st, x, llm, model_t, ind, ocos, osin, ctx)
    var d_b0 = h.compare(caps2[1], ref_b0, ctx)
    var d_b16 = h.compare(caps2[2], ref_b16, ctx)
    var d_b33 = h.compare(caps2[3], ref_b33, ctx)
    var d_tout = cast_tensor(linear(caps2[0], flw, Optional[Tensor](flb.clone(ctx)), ctx), STDtype.F32, ctx)
    var d_rtout = h.compare(d_tout, ref_tout, ctx)
    print("  block0_out  (f32 rope)", d_b0)
    print("  block16_out (f32 rope)", d_b16)
    print("  block33_out (f32 rope)", d_b33)
    print("  transformer (f32 rope)", d_rtout)

    print("------------------------------------------------------------")
    if n_fail == 0:
        print("IDEOGRAM4 PREDICT ai-toolkit PARITY PASS (all 7 captures cos >= 0.999)")
    else:
        raise Error(
            String("IDEOGRAM4 PREDICT ai-toolkit PARITY FAIL: ")
            + String(n_fail) + " capture(s) below cos 0.999"
        )


def _neg_ones(n: Int) -> List[Float32]:
    var v = List[Float32]()
    for _ in range(n):
        v.append(Float32(-1.0))
    return v^

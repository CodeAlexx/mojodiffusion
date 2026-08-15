# minimax_h3_train_sigma_parity — gate the H3 training sigma policy, noising,
# joint token loss, and loss gradient against torch (musubi functions called
# directly in h3_sigma_oracle.py). Bars: sigmas/timesteps f32-exact, x_t and
# velocity targets bf16 BIT-exact (max_abs == 0), loss rel <= 1e-5, loss
# grads cos >= 0.999999.
from max.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.tensor import Tensor
from serenitymojo.models.minimax_h3.h3_train_sigma import (
    H3_VIDEO_FLOW_SHIFT, H3_AUDIO_FLOW_SHIFT,
    h3_shift_sigma, h3_noisy_input, h3_velocity_target,
    h3_modality_loss, h3_joint_token_loss, h3_loss_grad,
)

comptime ORACLE = "/home/alex/mojodiffusion/output/checks/h3_sigma_oracle.safetensors"
comptime N_CASES = 5


def _load(st: SafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    var info = st.tensor_info(name)
    return Tensor.from_view(
        from_parts(info.dtype, info.shape.copy(), st.tensor_bytes(name)), ctx
    )


def _host_bools(st: SafeTensors, name: String) raises -> List[Bool]:
    var span = st.tensor_bytes(name)
    var out = List[Bool]()
    for i in range(len(span)):
        out.append(span[i] != 0)
    return out^


def _mask_tensor(
    st: SafeTensors, name: String, ctx: DeviceContext
) raises -> Tensor:
    """bool mask -> bf16 0/1 device tensor with the mask's own shape."""
    var info = st.tensor_info(name)
    var span = st.tensor_bytes(name)
    var vals = List[Float32]()
    for i in range(len(span)):
        vals.append(Float32(1.0) if span[i] != 0 else Float32(0.0))
    return Tensor.from_host(vals, info.shape.copy(), STDtype.BF16, ctx)


def _max_abs_diff(a: Tensor, b: Tensor, ctx: DeviceContext) raises -> Float64:
    var ah = a.to_host(ctx)
    var bh = b.to_host(ctx)
    if len(ah) != len(bh):
        raise Error("max_abs: length mismatch")
    var m = Float64(0)
    for i in range(len(ah)):
        var d = Float64(ah[i]) - Float64(bh[i])
        if d < 0:
            d = -d
        if d > m:
            m = d
    return m


def _cos(a: Tensor, b: Tensor, ctx: DeviceContext) raises -> Float64:
    var ah = a.to_host(ctx)
    var bh = b.to_host(ctx)
    var dot = Float64(0)
    var na = Float64(0)
    var nb = Float64(0)
    for i in range(len(ah)):
        var x = Float64(ah[i])
        var y = Float64(bh[i])
        dot += x * y
        na += x * x
        nb += y * y
    return dot / ((na**0.5) * (nb**0.5) + 1e-30)


def main() raises:
    var ctx = DeviceContext()
    var st = SafeTensors.open(String(ORACLE))
    var ok = True

    var video_x0 = _load(st, String("video_x0"), ctx)
    var video_noise = _load(st, String("video_noise"), ctx)
    var audio_x0 = _load(st, String("audio_x0"), ctx)
    var audio_noise = _load(st, String("audio_noise"), ctx)
    var audio_mask_b = _host_bools(st, String("audio_mask"))
    var video_mask_b = _host_bools(st, String("video_mask"))
    var audio_mask_t = _mask_tensor(st, String("audio_mask"), ctx)
    var video_mask_t = _mask_tensor(st, String("video_mask"), ctx)
    var bases = _load(st, String("base_sigmas"), ctx).to_host(ctx)

    for c in range(N_CASES):
        var tag = String("case") + String(c)
        var base = Float32(bases[c])
        var sig_ref = _load(st, tag + "_sigma", ctx).to_host(ctx)
        var sv = h3_shift_sigma(base, H3_VIDEO_FLOW_SHIFT)
        var sa = h3_shift_sigma(base, H3_AUDIO_FLOW_SHIFT)
        var d_sv = abs(Float64(sv) - Float64(sig_ref[0]))
        var d_sa = abs(Float64(sa) - Float64(sig_ref[1]))
        var d_tv = abs(Float64(1.0 - sv) - Float64(sig_ref[2]))
        var d_ta = abs(Float64(1.0 - sa) - Float64(sig_ref[3]))
        var sig_ok = d_sv <= 1e-7 and d_sa <= 1e-7 and d_tv <= 1e-7 and d_ta <= 1e-7
        print(("PASS " if sig_ok else "FAIL ") + tag + " sigma_v", sv, "sigma_a", sa)
        if not sig_ok:
            ok = False

        var xt_v = h3_noisy_input(video_x0, video_noise, sv, ctx)
        var xt_a = h3_noisy_input(audio_x0, audio_noise, sa, ctx)
        var mv = _max_abs_diff(xt_v, _load(st, tag + "_video_xt", ctx), ctx)
        var ma = _max_abs_diff(xt_a, _load(st, tag + "_audio_xt", ctx), ctx)
        var xt_ok = mv == 0.0 and ma == 0.0
        print(("PASS " if xt_ok else "FAIL ") + tag + " x_t max_abs v/a", mv, ma)
        if not xt_ok:
            ok = False

        var tg_v = h3_velocity_target(video_x0, video_noise, ctx)
        var tg_a = h3_velocity_target(audio_x0, audio_noise, ctx)
        var tv = _max_abs_diff(tg_v, _load(st, tag + "_video_target", ctx), ctx)
        var ta = _max_abs_diff(tg_a, _load(st, tag + "_audio_target", ctx), ctx)
        var tg_ok = tv == 0.0 and ta == 0.0
        print(("PASS " if tg_ok else "FAIL ") + tag + " target max_abs v/a", tv, ta)
        if not tg_ok:
            ok = False

        var pred_v = _load(st, tag + "_pred_video", ctx)
        var pred_a = _load(st, tag + "_pred_audio", ctx)
        var use_video_mask = c == 4
        var empty_mask = List[Bool]()
        var ml_v = h3_modality_loss(
            pred_v, tg_v, video_mask_b if use_video_mask else empty_mask, ctx
        )
        var ml_a = h3_modality_loss(pred_a, tg_a, audio_mask_b, ctx)
        var loss = h3_joint_token_loss(ml_v, ml_a, 1.0, 1.0)
        var loss_ref = Float64(_load(st, tag + "_loss", ctx).to_host(ctx)[0])
        var rel = abs(loss - loss_ref) / (abs(loss_ref) + 1e-30)
        var loss_ok = rel <= 1e-5
        print(("PASS " if loss_ok else "FAIL ") + tag + " loss", loss, "ref", loss_ref, "rel", rel)
        if not loss_ok:
            ok = False

        var denom = Float64(ml_v.elements) + Float64(ml_a.elements)
        var vmask_opt = Optional[Tensor](None)
        if use_video_mask:
            vmask_opt = Optional[Tensor](video_mask_t.clone(ctx))
        var g_v = h3_loss_grad(pred_v, tg_v, vmask_opt^, 1.0, denom, ctx)
        var g_a = h3_loss_grad(
            pred_a, tg_a, Optional[Tensor](audio_mask_t.clone(ctx)), 1.0, denom, ctx,
        )
        var c_v = _cos(g_v, _load(st, tag + "_grad_video", ctx), ctx)
        var c_a = _cos(g_a, _load(st, tag + "_grad_audio", ctx), ctx)
        var g_ok = c_v >= 0.999999 and c_a >= 0.999999
        print(("PASS " if g_ok else "FAIL ") + tag + " grad cos v/a", c_v, c_a)
        if not g_ok:
            ok = False

    if ok:
        print("PASS: h3 sigma policy + noising + joint loss + grads match torch")
    else:
        print("FAIL: h3 sigma parity below bar")
        raise Error("minimax_h3_train_sigma_parity failed")

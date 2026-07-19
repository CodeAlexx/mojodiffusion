# NAVA denoise-loop driver + parity: resident NavaDiT, simplified 2-forward CFG
# (cond + zeros-uncond, vid_g=3 aud_g=2, NO slg/align/timbre) + 2 UniPC schedulers
# (shift=5), 3 steps, gated per-step vs the torch oracle (nava_sample_oracle.py).
# Latents stay F32-resident; cast to BF16 only to feed the DiT (h2d discipline).
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.parity import ParityHarness
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import add, sub, mul_scalar, slice, reshape, zeros_device
from serenitymojo.sampling.unipc import UniPcMultistepScheduler
from serenitymojo.models.dit.nava_dit import NavaDiT

comptime DIT_CKPT = "/home/alex/.serenity/models/checkpoints/NAVA/NAVA_fp8.safetensors"
comptime FX = "/home/alex/mojodiffusion/serenitymojo/sampling/parity/nava_sample_fx.safetensors"
comptime NSTEP = 3
comptime VID_G = Float32(3.0)
comptime AUD_G = Float32(2.0)


def main() raises:
    var ctx = DeviceContext()
    print("=== NAVA denoise loop parity (resident DiT, 2-forward CFG, 3 steps) ===")
    var st = ShardedSafeTensors.open(DIT_CKPT)
    var dit = NavaDiT.load(st, ctx)

    var fx = ShardedSafeTensors.open(FX)
    var pos = cast_tensor(Tensor.from_view(fx.tensor_view("pos_text"), ctx), STDtype.BF16, ctx)  # [42,4096]
    var unc = zeros_device([42, 4096], STDtype.BF16, ctx)
    var lv = Tensor.from_view_as_f32(fx.tensor_view("init_lat_vid"), ctx)  # [1280,48] F32 resident
    var la = Tensor.from_view_as_f32(fx.tensor_view("init_lat_aud"), ctx)  # [34,128]  F32 resident
    var ts = Tensor.from_view_as_f32(fx.tensor_view("timesteps"), ctx)     # [3]

    var sv = UniPcMultistepScheduler(1000, 25, 5.0, 2)
    var sa = UniPcMultistepScheduler(1000, 25, 5.0, 2)

    var min_cos = Float64(1.0)
    for i in range(NSTEP):
        var t = reshape(slice(ts, 0, i, 1, ctx), [1], ctx)  # [1] F32
        # feed DiT with bf16 copies of the F32-resident latents (h2d: cast, don't reload)
        var lv_b = cast_tensor(lv, STDtype.BF16, ctx)
        var la_b = cast_tensor(la, STDtype.BF16, ctx)
        var cond = dit.forward(lv_b, la_b, pos, t, ctx)
        var lv_b2 = cast_tensor(lv, STDtype.BF16, ctx)
        var la_b2 = cast_tensor(la, STDtype.BF16, ctx)
        var unco = dit.forward(lv_b2, la_b2, unc, t, ctx)

        # CFG in F32: eps = uncond + g*(cond - uncond)
        var cv = cast_tensor(cond.vel_vid, STDtype.F32, ctx)
        var uv = cast_tensor(unco.vel_vid, STDtype.F32, ctx)
        var ca = cast_tensor(cond.vel_aud, STDtype.F32, ctx)
        var ua = cast_tensor(unco.vel_aud, STDtype.F32, ctx)
        var eps_v = add(uv, mul_scalar(sub(cv, uv, ctx), VID_G, ctx), ctx)
        var eps_a = add(ua, mul_scalar(sub(ca, ua, ctx), AUD_G, ctx), ctx)

        lv = sv.step(eps_v, lv, ctx)
        la = sa.step(eps_a, la, ctx)

        var ref_lv = Tensor.from_view(fx.tensor_view("lv_" + String(i)), ctx).to_host(ctx)
        var ref_la = Tensor.from_view(fx.tensor_view("la_" + String(i)), ctx).to_host(ctx)
        var rv = ParityHarness(0.999).compare(lv, ref_lv, ctx)
        var ra = ParityHarness(0.999).compare(la, ref_la, ctx)
        print("  step", i, "lv:", rv, "| la:", ra)
        if rv.cos < min_cos:
            min_cos = rv.cos
        if ra.cos < min_cos:
            min_cos = ra.cos
    print("  min cos across steps:", min_cos)
    if min_cos >= 0.999:
        print("GATE PASS: denoise loop matches torch every step")
    else:
        print("GATE FAIL")

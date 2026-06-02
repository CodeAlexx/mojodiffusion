# pipeline/pid_net_ladder_smoke.mojo — PiD per-block ladder localizer.
#
# Re-runs the assembled PidNet forward but, instead of (or in addition to) the
# final output, gates intermediate hidden states against the captured Python
# ladder (patch_post_i / pixel_post_j) to LOCALIZE any divergence and classify
# bf16 chain-drift vs a real per-block bug. The full-net F32 gate already
# passes at cos>0.9999; this provides the explicit per-stage evidence.
#
# It re-implements the forward inline up to each checkpoint so it can read the
# state at that point. To keep it cheap we gate: patch_post_0, patch_post_6,
# patch_post_13, pixel_post_0, pixel_post_1.
#
# Run: pixi run mojo run -I . serenitymojo/pipeline/pid_net_ladder_smoke.mojo

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.models.pid.pid_net import pid_net_ladder


comptime CKPT = "/home/alex/.serenity/models/pid/checkpoints/PiD_res2k_sr4x_official_sd3_distill_4step/model_ema_bf16.safetensors"
comptime REF = "/home/alex/mojodiffusion/serenitymojo/models/pid/parity/pid_net_ref.safetensors"


def _ld(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    var tv = st.tensor_view(name)
    return Tensor.from_view_as_f32(tv, ctx)


def _gate(label: String, a: Tensor, refs: ShardedSafeTensors, key: String, ctx: DeviceContext) raises -> Bool:
    var h = ParityHarness()
    var golden = _ld(refs, key, ctx)
    var r = h.compare(a, golden.to_host(ctx), ctx)
    var tag = "PASS" if (r.cos >= 0.999) else "FAIL"
    print(label, " cos=", r.cos, " max_abs=", r.max_abs, " [", tag, "]")
    return r.cos >= 0.999


def main() raises:
    var ctx = DeviceContext()
    comptime B = 1
    comptime H = 64
    comptime W = 64
    comptime PH = 4
    comptime PW = 4
    comptime L = 16
    comptime LTXT = 8
    comptime ZH = 2
    comptime ZW = 2

    print("=== PiD per-block LADDER — F32 vs PyTorch PiD refs ===")
    var ckpt = ShardedSafeTensors.open(String(CKPT))
    var refs = ShardedSafeTensors.open(String(REF))

    var x = _ld(refs, "x", ctx)
    var t = _ld(refs, "t_scaled", ctx)
    var y = _ld(refs, "y", ctx)
    var lq_latent = _ld(refs, "lq_latent", ctx)
    var pix_pos = _ld(refs, "pix_pos", ctx)
    var img_cos = _ld(refs, "img_cos", ctx)
    var img_sin = _ld(refs, "img_sin", ctx)
    var txt_cos = _ld(refs, "txt_cos", ctx)
    var txt_sin = _ld(refs, "txt_sin", ctx)

    # pid_net_ladder returns: patch_post_0, patch_post_6, patch_post_13,
    #                          pixel_post_0, pixel_post_1
    var lad = pid_net_ladder[B, H, W, PH, PW, L, LTXT, ZH, ZW](
        ckpt, x, t, y, lq_latent, Float32(0.0),
        pix_pos, img_cos, img_sin, txt_cos, txt_sin, ctx
    )
    var ok = True
    ok = _gate("patch_post_0 ", lad.pp0, refs, "patch_post_0", ctx) and ok
    ok = _gate("patch_post_6 ", lad.pp6, refs, "patch_post_6", ctx) and ok
    ok = _gate("patch_post_13", lad.pp13, refs, "patch_post_13", ctx) and ok
    ok = _gate("pixel_post_0 ", lad.px0, refs, "pixel_post_0", ctx) and ok
    ok = _gate("pixel_post_1 ", lad.px1, refs, "pixel_post_1", ctx) and ok

    print("============================================================")
    if ok:
        print("LADDER ALL STAGES PASS")
    else:
        print("LADDER LOCALIZED A DIVERGENCE (see first FAIL above)")
        raise Error("pid_net ladder: stage failure")

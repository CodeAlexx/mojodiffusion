# Qwen-Image-Edit synthetic runtime smoke.
#
# Exercises the edit DiT runtime path without Qwen2.5-VL or VAE encode:
# synthetic target noise + synthetic single-reference latents -> edit DiT CFG
# with reference `t_ref=0` -> target slice -> Qwen image VAE decode.

from std.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.dit.qwenimage_dit import QwenImageDitOffloaded
from serenitymojo.models.dit.qwenimage_edit_contract import (
    QWENIMAGE_EDIT_ROOT,
    QWENIMAGE_EDIT_TRANSFORMER_DIR,
)
from serenitymojo.models.vae.qwenimage_decoder import QwenImageVaeDecoder
from serenitymojo.ops.random import randn
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.layout import patchify, unpatchify
from serenitymojo.sampling.flow_match import (
    Scheduler,
    cfg_qwen,
    build_qwen_sigma_schedule,
)
from serenitymojo.image.png import save_png, ValueRange


comptime VAE_DIR = QWENIMAGE_EDIT_ROOT + "/vae"
comptime OUT = "/home/alex/mojodiffusion/output/qwenimage_edit_synth_512.png"

comptime N_TXT = 64
comptime LH = 64
comptime LW = 64
comptime PATCH = 2
comptime N_TARGET = (LH // PATCH) * (LW // PATCH)
comptime N_REF = N_TARGET
comptime S = N_TXT + N_TARGET + N_REF
comptime FRAME = 1
comptime FH = LH // PATCH
comptime FW = LW // PATCH
comptime STEPS = 1
comptime CFG = Float32(4.0)


def _stats(name: String, t: Tensor, ctx: DeviceContext) raises:
    var h = t.to_host(ctx)
    var n = len(h)
    var s = 0.0
    var s2 = 0.0
    var amax = 0.0
    for i in range(n):
        var v = Float64(h[i])
        s += v
        s2 += v * v
        var av = v if v >= 0.0 else -v
        if av > amax:
            amax = av
    var mean = s / Float64(n)
    var var_ = s2 / Float64(n) - mean * mean
    if var_ < 0.0:
        var_ = 0.0
    print(
        "  [stat]", name, "mean=", Float32(mean), "std=", Float32(sqrt(var_)),
        "absmax=", Float32(amax), "n=", n,
    )


def _latent_noise(seed: UInt64, ctx: DeviceContext) raises -> Tensor:
    var sh = List[Int]()
    sh.append(1)
    sh.append(16)
    sh.append(LH)
    sh.append(LW)
    return randn(sh^, seed, STDtype.F32, ctx)


def _text_states(seed: UInt64, ctx: DeviceContext) raises -> Tensor:
    var sh = List[Int]()
    sh.append(1)
    sh.append(N_TXT)
    sh.append(3584)
    return cast_tensor(randn(sh^, seed, STDtype.F32, ctx), STDtype.BF16, ctx)


def main() raises:
    var ctx = DeviceContext()
    print("=== Qwen-Image-Edit synthetic 512 runtime smoke ===")
    print(
        "[shape] target/ref/text/sequence:",
        N_TARGET,
        N_REF,
        N_TXT,
        S,
    )

    var target = cast_tensor(patchify(_latent_noise(UInt64(42), ctx), PATCH, ctx), STDtype.BF16, ctx)
    var reference = cast_tensor(patchify(_latent_noise(UInt64(43), ctx), PATCH, ctx), STDtype.BF16, ctx)
    var text_pos = _text_states(UInt64(44), ctx)
    var text_neg = _text_states(UInt64(45), ctx)
    _stats("target_init", target, ctx)
    _stats("reference", reference, ctx)

    print("[denoise] Qwen-Image-Edit DiT, single-reference zero-cond-t path")
    var model = QwenImageDitOffloaded.load(String(QWENIMAGE_EDIT_TRANSFORMER_DIR), ctx)
    var sigmas = build_qwen_sigma_schedule(STEPS, Float32(N_TARGET))
    var preds = model.forward_edit_cfg[N_TARGET, N_REF, N_TXT, S](
        target,
        reference,
        text_pos,
        text_neg,
        sigmas[0],
        Float32(0.0),
        FRAME,
        FH,
        FW,
        ctx,
    )
    var pred_pos = cast_tensor(preds.pos, STDtype.F32, ctx)
    var pred_neg = cast_tensor(preds.neg, STDtype.F32, ctx)
    var pred = cfg_qwen(pred_pos, pred_neg, CFG, ctx)
    _stats("pred_target", pred, ctx)

    var sched = Scheduler.qwen(STEPS, Float32(N_TARGET))
    var target_next = sched.step(cast_tensor(target, STDtype.F32, ctx), pred, 0, ctx)
    _stats("target_next", target_next, ctx)

    print("[vae] decode target slice")
    var latent = unpatchify(target_next, 16, LH, LW, PATCH, ctx)
    latent = cast_tensor(latent, STDtype.BF16, ctx)
    var vae = QwenImageVaeDecoder[LH, LW].load(String(VAE_DIR), ctx)
    var image = vae.decode(latent, ctx)
    print("  image shape:", image.shape()[0], image.shape()[1], image.shape()[2], image.shape()[3])
    _stats("image", image, ctx)
    save_png(image, String(OUT), ctx, ValueRange.SIGNED)
    print("[done] saved", OUT)
    print("Qwen-Image-Edit synthetic 512 runtime smoke PASS")

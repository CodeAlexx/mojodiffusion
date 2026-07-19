# Z-Image LoRA real-weights TRAIN gate (fixed-σ overfit probe).
# Loads real zimage_base weights, runs N fixed-σ training steps with LoRA, and
# proves: loss DECREASES, LoRA-B goes 0→nonzero on the adapters, 0 nonfinite.
from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import (
    add, sub, mul, mul_scalar, zeros_device,
)
from serenity_trainer.modelLoader.ZImageModelLoader import ZImageWeights
from serenity_trainer.model.ZImageModel import build_zimage_lora_set
from serenity_trainer.model.ZImageDiT import (
    zimage_forward_full_lora, zimage_backward_full_lora,
)
from serenity_trainer.modelSetup.BaseZImageSetup import (
    model_t_from_timestep, sigma_from_timestep,
)
from serenity_trainer.util.optimizer.adamw_extensions import adamw_step
from serenitymojo.ops.loss_swiglu_backward import mse_backward

comptime HL = 16
comptime WL = 16
comptime CAPLEN = 64
comptime N_STEPS = 12
comptime TArc = ArcPointer[Tensor]


# zeros_like: fresh zero tensor matching x's shape/dtype.
def _zeros_like(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var sh = x.shape()
    return zeros_device(sh^, x.dtype(), ctx)


# host-side mean((a-b)^2)
def _mse_host(a: Tensor, b: Tensor, ctx: DeviceContext) raises -> Float32:
    var d = sub(a, b, ctx)
    var sq = mul(d, d, ctx)
    var h = sq.to_host(ctx)
    var n = len(h)
    var s = Float32(0.0)
    for i in range(n):
        s += h[i]
    return s / Float32(n)


# L1 sum over a tensor (host).
def _l1_sum(x: Tensor, ctx: DeviceContext) raises -> Float32:
    var h = x.to_host(ctx)
    var s = Float32(0.0)
    for i in range(len(h)):
        var v = h[i]
        if v < 0:
            s -= v
        else:
            s += v
    return s


# count nonfinite in a tensor (host).
def _count_nonfinite(x: Tensor, ctx: DeviceContext) -> Int:
    try:
        var h = x.to_host(ctx)
        var c = 0
        for i in range(len(h)):
            var v = h[i]
            # NaN: v != v ; Inf: v - v != 0
            if v != v:
                c += 1
            elif (v - v) != Float32(0.0):
                c += 1
        return c
    except:
        return -1


def main() raises:
    var ctx = DeviceContext()
    print("loading zimage_base weights ...")
    var w = ZImageWeights.load(String("/home/alex/.serenity/models/zimage_base/transformer"), ctx)
    print("  loaded.")

    # ── FIXED-σ overfit batch (built ONCE, reused every step) ──────────────────
    var latent0 = randn([1, 16, HL, WL], UInt64(101), STDtype.BF16, ctx)
    var noise0  = randn([1, 16, HL, WL], UInt64(202), STDtype.BF16, ctx)
    var cap     = randn([CAPLEN, 2560], UInt64(303), STDtype.BF16, ctx)
    var sigma0  = sigma_from_timestep(250)          # 0.251
    var t_model = model_t_from_timestep(250)        # 0.75
    # scaled_noisy = sigma0*noise0 + (1-sigma0)*latent0
    var sn_term = mul_scalar(noise0, sigma0, ctx)
    var lat_term = mul_scalar(latent0, Float32(1.0) - sigma0, ctx)
    var scaled_noisy = add(sn_term, lat_term, ctx)
    # target = noise0 - latent0  (flow-matching velocity target)
    var target = sub(noise0, latent0, ctx)

    print("sigma0 =", sigma0, " t_model =", t_model)

    # ── LoRA set + per-adapter Adam state ──────────────────────────────────────
    var loras = build_zimage_lora_set(8, Float32(8.0), ctx)
    var n_ad = len(loras.ad)
    print("n_adapters =", n_ad, " rank =", loras.rank, " n_layers =", loras.n_layers)

    var m_a = List[TArc]()
    var v_a = List[TArc]()
    var m_b = List[TArc]()
    var v_b = List[TArc]()
    for i in range(n_ad):
        m_a.append(TArc(_zeros_like(loras.ad[i][].a, ctx)))
        v_a.append(TArc(_zeros_like(loras.ad[i][].a, ctx)))
        m_b.append(TArc(_zeros_like(loras.ad[i][].b, ctx)))
        v_b.append(TArc(_zeros_like(loras.ad[i][].b, ctx)))

    # ── training loop ──────────────────────────────────────────────────────────
    var first_loss = Float32(0.0)
    var last_loss = Float32(0.0)
    var total_nonfinite = 0

    for step in range(1, N_STEPS + 1):
        var fo = zimage_forward_full_lora[HL, WL, CAPLEN](scaled_noisy, t_model, cap, w, loras, ctx)
        # predicted = -velocity
        var predicted = mul_scalar(fo.velocity, Float32(-1.0), ctx)
        var loss = _mse_host(predicted, target, ctx)
        print("step", step, " loss =", loss)
        if step == 1:
            first_loss = loss
        last_loss = loss

        total_nonfinite += _count_nonfinite(fo.velocity, ctx)

        # d_pred = 2*(predicted-target)/N ; d_velocity = -d_pred
        var d_pred = mse_backward(predicted, target, ctx)
        var d_velocity = mul_scalar(d_pred, Float32(-1.0), ctx)

        var grads = zimage_backward_full_lora[HL, WL, CAPLEN](d_velocity, fo.saved, w, loras, ctx)

        for i in range(n_ad):
            adamw_step(loras.ad[i][].a, m_a[i][], v_a[i][], grads.d_a[i][],
                       step, Float32(1e-3), Float32(0.9), Float32(0.999),
                       Float32(1e-8), Float32(0.01), True, UInt32(1234 + i), ctx)
            adamw_step(loras.ad[i][].b, m_b[i][], v_b[i][], grads.d_b[i][],
                       step, Float32(1e-3), Float32(0.9), Float32(0.999),
                       Float32(1e-8), Float32(0.01), True, UInt32(9876 + i), ctx)

    # ── post-loop metrics ──────────────────────────────────────────────────────
    var loraB_sum = Float32(0.0)
    var loraB_nonzero = 0
    for i in range(n_ad):
        var s = _l1_sum(loras.ad[i][].b, ctx)
        loraB_sum += s
        if s > Float32(0.0):
            loraB_nonzero += 1

    print("")
    print("=== Z-Image TRAIN GATE RESULTS ===")
    print("first_loss =", first_loss, " last_loss =", last_loss)
    print("loraB_sum =", loraB_sum)
    print("loraB_nonzero_count =", loraB_nonzero, "/", n_ad)
    print("nonfinite =", total_nonfinite)

    var loss_drop = last_loss < first_loss
    var b_imprint = loraB_sum > Float32(0.0)
    var clean = total_nonfinite == 0
    if loss_drop and b_imprint and clean:
        print("ZIMAGE TRAIN GATE PASS")
    else:
        print("ZIMAGE TRAIN GATE FAIL  (loss_drop=", loss_drop,
              " b_imprint=", b_imprint, " clean=", clean, ")")

# REAL-DATA Z-Image LoRA train run via the faithful predict() on Serenity's own
# cached CLEAN sample. Verifies loss lands in Serenity's real-data range (~0.2-0.8,
# baseline mean ~0.47), NOT the synthetic ~1.7. LoRA-B imprints 0→nonzero, nonfinite=0.
from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import sub, mul, mul_scalar, zeros_device
from serenitymojo.ops.reduce import reduce_mean_f32
from serenitymojo.ops.loss_swiglu_backward import mse_backward
from serenitymojo.autograd import Tape
from serenity_trainer.modelLoader.ZImageModelLoader import ZImageWeights
from serenity_trainer.modelSetup.ZImageLoRASetup import make_zimage_lora_spec
from serenity_trainer.util.config.TrainConfig import (
    TrainConfig, TSDIST_LOGIT_NORMAL,
)
from serenity_trainer.util.optimizer.adamw_extensions import adamw_step

comptime HL = 72
comptime WL = 56
comptime CAPLEN = 224
comptime N_STEPS = 16
comptime TArc = ArcPointer[Tensor]


def _zeros_like(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var sh = x.shape()
    return zeros_device(sh^, x.dtype(), ctx)


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


def _count_nonfinite(x: Tensor, ctx: DeviceContext) -> Int:
    try:
        var h = x.to_host(ctx)
        var c = 0
        for i in range(len(h)):
            var v = h[i]
            if v != v:
                c += 1
            elif (v - v) != Float32(0.0):
                c += 1
        return c
    except:
        return -1


def main() raises:
    var ctx = DeviceContext()

    # ── load Serenity's CLEAN cached sample (latent pre-scale + caption feats) ──
    print("loading real clean sample ...")
    var st = ShardedSafeTensors.open(
        String("trainer/parity/zi_realclean.safetensors"))
    var latent = cast_tensor(
        Tensor.from_view(st.tensor_view(String("latent")), ctx), STDtype.BF16, ctx)  # [1,16,72,56]
    var cap = cast_tensor(
        Tensor.from_view(st.tensor_view(String("cap")), ctx), STDtype.BF16, ctx)      # [224,2560]

    print("loading zimage_base weights ...")
    var w = ZImageWeights.load(
        String("/home/alex/.serenity/models/zimage_base/transformer"), ctx)
    print("  loaded.")

    # ── faithful spec (cold-start LoRA: A~randn, B=0). timestep_shift=1.0. ────────
    var spec = make_zimage_lora_spec[HL, WL, CAPLEN](
        w^, latent^, cap^, 8, Float32(8.0), Float32(1.0), UInt64(42), ctx)
    var n_ad = len(spec.loras.ad)
    print("n_adapters =", n_ad, " rank =", spec.loras.rank, " n_layers =", spec.loras.n_layers)

    # ── TrainConfig: Serenity Z-Image preset (LOGIT_NORMAL, shift 1.0, static). ─
    var config = TrainConfig.adamw_lora_defaults()
    config.timestep_distribution = TSDIST_LOGIT_NORMAL
    config.timestep_shift = Float32(1.0)
    config.dynamic_timestep_shifting = False

    # ── per-adapter Adam state (persistent across steps). ────────────────────────
    var m_a = List[TArc]()
    var v_a = List[TArc]()
    var m_b = List[TArc]()
    var v_b = List[TArc]()
    for i in range(n_ad):
        m_a.append(TArc(_zeros_like(spec.loras.ad[i][].a, ctx)))
        v_a.append(TArc(_zeros_like(spec.loras.ad[i][].a, ctx)))
        m_b.append(TArc(_zeros_like(spec.loras.ad[i][].b, ctx)))
        v_b.append(TArc(_zeros_like(spec.loras.ad[i][].b, ctx)))

    # ── training loop via the faithful predict() ─────────────────────────────────
    var loss_sum = Float32(0.0)
    var total_nonfinite = 0
    var in_range = 0

    for step in range(1, N_STEPS + 1):
        var tape = Tape()
        var so = spec.predict(tape, config, step, ctx)   # StepOutput{predicted,target,timestep=sigma}

        # loss = mean((predicted - target)^2)
        var diff = sub(so.predicted, so.target, ctx)
        var sq = mul(diff, diff, ctx)
        var dims = List[Int]()
        for d in range(len(sq.shape())):
            dims.append(d)
        var loss = reduce_mean_f32(sq, dims^, False, ctx).to_host(ctx)[0]
        print("step", step, " sigma =", so.timestep, " loss =", loss)
        loss_sum += loss
        if loss >= Float32(0.2) and loss <= Float32(0.8):
            in_range += 1

        total_nonfinite += _count_nonfinite(so.predicted, ctx)

        # d_pred = 2*(predicted-target)/N ; predicted = -velocity → d_velocity = -d_pred
        var d_pred = mse_backward(so.predicted, so.target, ctx)
        var d_velocity = mul_scalar(d_pred, Float32(-1.0), ctx)

        var grads = spec.backward_lora(d_velocity, ctx)

        for i in range(n_ad):
            adamw_step(spec.loras.ad[i][].a, m_a[i][], v_a[i][], grads.d_a[i][],
                       step, Float32(3e-4), Float32(0.9), Float32(0.999),
                       Float32(1e-8), Float32(0.01), True, UInt32(1234 + i), ctx)
            adamw_step(spec.loras.ad[i][].b, m_b[i][], v_b[i][], grads.d_b[i][],
                       step, Float32(3e-4), Float32(0.9), Float32(0.999),
                       Float32(1e-8), Float32(0.01), True, UInt32(9876 + i), ctx)

    # ── post-loop metrics ────────────────────────────────────────────────────────
    var loraB_sum = Float32(0.0)
    var loraB_nonzero = 0
    for i in range(n_ad):
        var s = _l1_sum(spec.loras.ad[i][].b, ctx)
        loraB_sum += s
        if s > Float32(0.0):
            loraB_nonzero += 1

    var mean_loss = loss_sum / Float32(N_STEPS)

    print("")
    print("=== Z-Image REAL-DATA TRAIN RESULTS ===")
    print("mean_loss =", mean_loss, "  (Serenity baseline mean ~0.47)")
    print("steps_in_range[0.2,0.8] =", in_range, "/", N_STEPS)
    print("loraB_sum =", loraB_sum)
    print("loraB_nonzero_count =", loraB_nonzero, "/", n_ad)
    print("nonfinite =", total_nonfinite)

    var all_in_range = in_range == N_STEPS
    var b_imprint = loraB_sum > Float32(0.0)
    var clean = total_nonfinite == 0
    if all_in_range and b_imprint and clean:
        print("ZIMAGE REALDATA TRAIN GATE PASS")
    else:
        print("ZIMAGE REALDATA TRAIN GATE FAIL  (in_range=", all_in_range,
              " b_imprint=", b_imprint, " clean=", clean, ")")

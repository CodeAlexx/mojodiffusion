# Klein/Flux2 LoRA train parity smoke skeleton.
#
# Consumes parity/klein_fwd.safetensors and runs:
#   Flux2LoRASpec.predict -> MSE loss -> backward_lora -> Klein host AdamW.
#
# This is a training-path smoke, not full numeric train parity yet: predict() still
# owns the Mojo RNG noise draw, so the dumped latent/text/BN stats are used and the
# timestep is forced to 250, but the dumped noise tensor is not injected.
#
# Run:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#   pixi run mojo build -I . -I trainer/src -Xlinker -lm \
#     trainer/smoke/klein_train_parity_smoke.mojo -o /tmp/klein_train_parity && \
#   /tmp/klein_train_parity

from std.builtin.dtype import DType
from std.collections import List
from max.gpu.host import DeviceContext
from std.math import sqrt
from std.time import perf_counter_ns

from serenitymojo.autograd import Tape
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.loss_swiglu_backward import mse_backward
from serenitymojo.ops.tensor_algebra import sub, mul
from serenitymojo.tensor import Tensor

from serenity_trainer.model.klein.double_block import DoubleBlockWeights
from serenity_trainer.model.klein.single_block import SingleBlockWeights
from serenity_trainer.model.klein.klein_stack_lora import (
    KleinLoraGrads, KleinLoraSet,
    klein_lora_adamw_step, klein_lora_set_to_device,
)
from serenity_trainer.model.klein.weights import (
    build_klein_vec_silu,
    load_double_block_weights,
    load_klein_stack_base,
    load_klein_step_mod_weights,
    load_single_block_weights,
)
from serenity_trainer.model.KleinModel import (
    KDIM, KH, KDh, KTXT_CH, KNUM_DOUBLE, KNUM_SINGLE, KTIMESTEP_DIM,
    build_klein9b_lora_set, build_klein_rope_tables_port,
)
from serenity_trainer.model.KleinVAE import KLEIN_BN_EPS, _unpatchify_packed
from serenity_trainer.modelSetup.Flux2LoRASetup import make_flux2_lora_spec
from serenity_trainer.util.config.TrainConfig import TrainConfig


comptime PARITY = "trainer/parity/klein_fwd.safetensors"
comptime CKPT = "/home/alex/.serenity/models/checkpoints/flux-2-klein-base-9b.safetensors"

comptime HL = 16
comptime WL = 16
comptime NTXT = 48
comptime TIMESTEP = 250
comptime BASE_SEED = UInt64(1234)
comptime LORA_RANK = 16
comptime LORA_ALPHA = Float32(16.0)
comptime LR = Float32(1.0e-4)


@fieldwise_init
struct GradStats(Copyable, Movable):
    var elems: Int
    var nonfinite: Int
    var abs_sum: Float64
    var sumsq: Float64


@fieldwise_init
struct BStats(Copyable, Movable):
    var adapters: Int
    var imprinted: Int
    var elems: Int
    var nonfinite: Int
    var abs_sum: Float64


def _is_nonfinite(x: Float32) -> Bool:
    if x != x:
        return True
    return (x - x) != Float32(0.0)


def _mse_host(a: Tensor, b: Tensor, ctx: DeviceContext) raises -> Float32:
    var d = sub(a, b, ctx)
    var sq = mul(d, d, ctx)
    var h = sq.to_host(ctx)
    var s = Float32(0.0)
    for i in range(len(h)):
        s += h[i]
    return s / Float32(len(h))


def _count_nonfinite(x: Tensor, ctx: DeviceContext) raises -> Int:
    var h = x.to_host(ctx)
    var c = 0
    for i in range(len(h)):
        if _is_nonfinite(h[i]):
            c += 1
    return c


def _bn_inv_scale_from_dump(st: ShardedSafeTensors, ctx: DeviceContext) raises -> Tensor:
    var bn_var = Tensor.from_view(st.tensor_view(String("bn_var")), ctx)
    var host = bn_var.to_host(ctx)
    var vals = List[Float32]()
    for i in range(len(host)):
        vals.append(Float32(1.0) / sqrt(host[i] + KLEIN_BN_EPS))
    return Tensor.from_host(vals^, [len(host)], STDtype.F32, ctx)


def _scan_values(values: List[Float32], var stats: GradStats) -> GradStats:
    for i in range(len(values)):
        var x = values[i]
        stats.elems += 1
        if _is_nonfinite(x):
            stats.nonfinite += 1
        else:
            var xf = Float64(x)
            stats.sumsq += xf * xf
            if x < Float32(0.0):
                stats.abs_sum -= Float64(x)
            else:
                stats.abs_sum += Float64(x)
    return stats^


def _grad_stats(grads: KleinLoraGrads) -> GradStats:
    var stats = GradStats(0, 0, Float64(0.0), Float64(0.0))
    for i in range(len(grads.dbl_d_a)):
        stats = _scan_values(grads.dbl_d_a[i], stats^)
    for i in range(len(grads.dbl_d_b)):
        stats = _scan_values(grads.dbl_d_b[i], stats^)
    for i in range(len(grads.sgl_d_a)):
        stats = _scan_values(grads.sgl_d_a[i], stats^)
    for i in range(len(grads.sgl_d_b)):
        stats = _scan_values(grads.sgl_d_b[i], stats^)
    return stats^


def _scan_b(values: List[BFloat16], var stats: BStats) -> BStats:
    var adapter_abs = Float64(0.0)
    for i in range(len(values)):
        var x = values[i].cast[DType.float32]()
        stats.elems += 1
        if _is_nonfinite(x):
            stats.nonfinite += 1
        elif x < Float32(0.0):
            stats.abs_sum -= Float64(x)
            adapter_abs -= Float64(x)
        else:
            stats.abs_sum += Float64(x)
            adapter_abs += Float64(x)
    stats.adapters += 1
    if adapter_abs > Float64(0.0):
        stats.imprinted += 1
    return stats^


def _lora_b_stats(lora: KleinLoraSet) -> BStats:
    var stats = BStats(0, 0, 0, 0, Float64(0.0))
    for i in range(len(lora.dbl)):
        stats = _scan_b(lora.dbl[i].b, stats^)
    for i in range(len(lora.sgl)):
        stats = _scan_b(lora.sgl[i].b, stats^)
    return stats^


def _sec(ns0: UInt, ns1: UInt) -> Float64:
    return Float64(ns1 - ns0) / Float64(1000000000.0)


def main() raises:
    var ctx = DeviceContext()
    var all0 = perf_counter_ns()

    print("=== Klein/Flux2 LoRA train parity smoke skeleton ===")
    print("[parity]", PARITY)
    print("[ckpt]  ", CKPT)
    print("[shape] HL =", HL, " WL =", WL, " NTXT =", NTXT, " timestep =", TIMESTEP)

    var parity = ShardedSafeTensors.open(String(PARITY))
    var latent = cast_tensor(
        Tensor.from_view(parity.tensor_view(String("latent")), ctx),
        STDtype.BF16,
        ctx,
    )
    var txt_tokens = cast_tensor(
        Tensor.from_view(parity.tensor_view(String("txt")), ctx),
        STDtype.BF16,
        ctx,
    )
    var bn_inv_scale = _bn_inv_scale_from_dump(parity, ctx)
    var bn_mean = Tensor.from_view(parity.tensor_view(String("bn_mean")), ctx)

    var ref_velocity = Tensor.from_view(parity.tensor_view(String("velocity")), ctx)
    var ref_target = Tensor.from_view(parity.tensor_view(String("flow_target")), ctx)
    var ref_loss = _mse_host(ref_velocity, ref_target, ctx)
    print("[reference] raw_mse(velocity, flow_target) =", ref_loss)

    var load0 = perf_counter_ns()
    var ckpt = SafeTensors.open(String(CKPT))
    var ts = Tensor.from_host([Float32(TIMESTEP)], [1], STDtype.F32, ctx)
    var vec_silu = build_klein_vec_silu(ckpt, ts, KTIMESTEP_DIM, KDIM, ctx)
    var base = load_klein_stack_base(ckpt, vec_silu, KDIM, ctx)
    var step_mod_w = load_klein_step_mod_weights(ckpt, KDIM, ctx)

    var dbw = List[DoubleBlockWeights]()
    for bi in range(KNUM_DOUBLE):
        dbw.append(load_double_block_weights(ckpt, bi, ctx))
    var sbw = List[SingleBlockWeights]()
    for bi in range(KNUM_SINGLE):
        sbw.append(load_single_block_weights(ckpt, bi, ctx))
    var load1 = perf_counter_ns()
    print("[load] base +", len(dbw), "double +", len(sbw), "single blocks")

    var lora_host = build_klein9b_lora_set(LORA_RANK, LORA_ALPHA)
    var b0 = _lora_b_stats(lora_host)
    var lora_dev = klein_lora_set_to_device(lora_host, ctx)

    var rope_tup = build_klein_rope_tables_port[HL * WL, NTXT, KH, KDh](ctx, STDtype.BF16)
    var cos_t = rope_tup[0].clone(ctx)
    var sin_t = rope_tup[1].clone(ctx)

    var spec = make_flux2_lora_spec[HL, WL, NTXT](
        base^, dbw^, sbw^, step_mod_w^, lora_dev^,
        cos_t^, sin_t^, bn_inv_scale^, bn_mean^,
        txt_tokens^, latent^, BASE_SEED,
        Float32(1.0),
        False,
    )

    var cfg = TrainConfig.adamw_lora_defaults()
    cfg.learning_rate = LR
    cfg.dynamic_timestep_shifting = False
    cfg.timestep_shift = Float32(1.0)
    # Force get_timestep_discrete() to select the dumped parity timestep 250:
    # min=int(1000*0.25)=250, max=int(1000*0.2515)=251, so floor([250,251))=250.
    cfg.min_noising_strength = Float32(0.25)
    cfg.max_noising_strength = Float32(0.2515)

    var tape = Tape()
    print("[step] running Flux2LoRASpec.predict ...")
    var pred0 = perf_counter_ns()
    var out = spec.predict(tape, cfg, 0, ctx)
    var pred1 = perf_counter_ns()

    var loss0 = perf_counter_ns()
    # Flux2LoRASpec.predict currently returns predicted unpatchified but target
    # patchified. Keep this smoke local and align the target layout before loss.
    var target_unpatch = _unpatchify_packed(out.target, ctx)
    var loss = _mse_host(out.predicted, target_unpatch, ctx)
    var pred_nonfinite = _count_nonfinite(out.predicted, ctx)
    var target_nonfinite = _count_nonfinite(target_unpatch, ctx)
    var d_flow = mse_backward(out.predicted, target_unpatch, ctx)
    var loss1 = perf_counter_ns()

    print("[step] running Flux2LoRASpec.backward_lora ...")
    var bwd0 = perf_counter_ns()
    var grads = spec.backward_lora(d_flow, ctx)
    var bwd1 = perf_counter_ns()

    var gs = _grad_stats(grads)
    var grad_norm = sqrt(gs.sumsq)

    print("[step] running klein_lora_adamw_step ...")
    var opt0 = perf_counter_ns()
    klein_lora_adamw_step(
        lora_host, grads, 1, LR, ctx,
        cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay, cfg.stochastic_rounding,
    )
    var opt1 = perf_counter_ns()
    var b1 = _lora_b_stats(lora_host)
    var all1 = perf_counter_ns()

    print("")
    print("=== KLEIN TRAIN PARITY SMOKE RESULTS ===")
    print("loss =", loss, " sigma =", out.timestep)
    print("pred_nonfinite =", pred_nonfinite, " target_nonfinite =", target_nonfinite)
    print("grad_elems =", gs.elems, " grad_norm =", grad_norm,
          " grad_abs_sum =", gs.abs_sum, " grad_nonfinite =", gs.nonfinite)
    print("loraB_before_abs_sum =", b0.abs_sum,
          " loraB_before_nonzero_adapters =", b0.imprinted, "/", b0.adapters)
    print("loraB_after_abs_sum =", b1.abs_sum,
          " loraB_after_nonzero_adapters =", b1.imprinted, "/", b1.adapters,
          " loraB_nonfinite =", b1.nonfinite)
    print("time_s: load =", _sec(load0, load1),
          " predict =", _sec(pred0, pred1),
          " loss =", _sec(loss0, loss1),
          " backward =", _sec(bwd0, bwd1),
          " adamw =", _sec(opt0, opt1),
          " step =", _sec(pred0, opt1),
          " total =", _sec(all0, all1))

    var clean = pred_nonfinite == 0 and target_nonfinite == 0 and gs.nonfinite == 0 and b1.nonfinite == 0
    var imprinted = b1.imprinted > 0 and b1.abs_sum > Float64(0.0)
    if clean and imprinted:
        print("KLEIN TRAIN PARITY SMOKE PASS")
    else:
        print("KLEIN TRAIN PARITY SMOKE FAIL  clean =", clean, " imprinted =", imprinted)

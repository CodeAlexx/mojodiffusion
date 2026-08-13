# Replay the real two-step Serenity Klein/Flux2 train dump through Mojo
# forward/backward, then compare LoRA grad and AdamW update deltas.
#
# This gate intentionally uses Serenity dump tensors instead of Mojo RNG:
#   step001 trace.packed_latent_input + encoder_hidden_states -> training forward
#   output.target -> MSE backward -> Klein LoRA grads
#   step000 ref grads + step001 Mojo grads -> F32 AdamW state replay
#
# Serenity source anchors:
#   modules/modelSetup/BaseFlux2Setup.py::predict/calculate_loss
#   modules/module/LoRAModule.py
#   modules/util/optimizer/adamw_extensions.py

from std.collections import List, Optional
from max.gpu.host import DeviceContext
from std.math import abs, sqrt
from std.memory import ArcPointer
from std.time import perf_counter_ns

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.loss_swiglu_backward import mse_backward
from serenitymojo.ops.tensor_algebra import (
    mul, permute as _permute, reshape as _reshape, reshape_owned as _reshape_owned,
    sub,
)
from serenitymojo.scratch_ring import ScratchRingAllocator
from serenitymojo.tensor import Tensor

from serenity_trainer.model.klein.double_block import DoubleBlockWeights, ModVecsDevice
from serenity_trainer.model.klein.single_block import SingleBlockWeights, SingleModVecsDevice
from serenity_trainer.model.klein.klein_stack import KleinStackBase, KleinStackForward
from serenity_trainer.model.klein.klein_stack_lora import (
    KleinLoraDeviceSet, KleinLoraGrads, KleinLoraSet, klein_lora_set_to_device,
    klein_stack_lora_backward_resident_moddev_rope_scratch,
    klein_stack_lora_forward_device_inputs_resident_moddev_rope_scratch,
)
from serenity_trainer.model.klein.weights import (
    build_klein_step_mods_device_cached,
    build_klein_vec_silu,
    load_double_block_weights,
    load_klein_stack_base,
    load_klein_step_mod_weights,
    load_single_block_weights,
)
from serenity_trainer.model.KleinModel import (
    KDIM, KEPS, KF, KH, KDh, KIN_CH, KOUT_CH, KTXT_CH, KNUM_DOUBLE,
    KNUM_SINGLE, KTIMESTEP_DIM, build_klein_rope_tables_hw_port,
)
from serenity_trainer.model.KleinVAE import _patchify_packed, _unpatchify_packed
from serenity_trainer.modelLoader.Flux2RuntimeLoader import load_flux2_lora_fused_phase
from serenity_trainer.modelSetup.flux2LoraTargets import (
    flux2_double_module, flux2_single_module,
)


comptime TArc = ArcPointer[Tensor]

comptime STEP0 = "/tmp/klein_train_ref_2step_step000.safetensors"
comptime STEP0_ADAPTERS = "/tmp/klein_train_ref_2step_step000_adapters.safetensors"
comptime STEP1 = "/tmp/klein_train_ref_2step_step001.safetensors"
comptime STEP1_ADAPTERS = "/tmp/klein_train_ref_2step_step001_adapters.safetensors"
comptime CKPT = "/home/alex/.serenity/models/checkpoints/flux-2-klein-base-9b.safetensors"

comptime HL = 40
comptime WL = 28
comptime N_IMG = HL * WL
comptime N_TXT = 512
comptime S = N_IMG + N_TXT
comptime TIMESTEP = Float32(346.0)
comptime SERENITY_LOSS = Float32(0.5876612663269043)
comptime LR_STEP1 = Float32(0.000003)
comptime BETA1 = Float32(0.9)
comptime BETA2 = Float32(0.999)
comptime EPS = Float32(1.0e-8)
comptime WEIGHT_DECAY = Float32(0.01)

# This full gate uses Mojo's BF16 transformer forward, not dumped reference trainer predictions.
# The step001 forward replay currently lands 7.4e-5 from reference trainer loss; keep this gate
# tight enough to catch real drift while allowing backward/update diagnostics.
comptime LOSS_EPS = Float32(0.0001)
comptime MIN_FORWARD_COS = Float64(0.999)
comptime GRAD_REL_EPS = Float64(0.05)
comptime UPDATE_REL_EPS = Float64(0.05)

comptime FWD_SCRATCH_SLAB_BYTES = 256 * 1024 * 1024
comptime FWD_SCRATCH_NUM_SLABS = 2
comptime BWD_SCRATCH_SLAB_BYTES = 512 * 1024 * 1024  # F32-bwd experiment needs 2x; bf16 path unaffected
comptime BWD_SCRATCH_NUM_SLABS = 2


def _forward_with_scratch[
    H: Int, Dh: Int, N_IMG_: Int, N_TXT_: Int, S_: Int
](
    img_tokens_t: TArc,
    txt_tokens_t: TArc,
    base: KleinStackBase,
    dbw: List[DoubleBlockWeights],
    sbw: List[SingleBlockWeights],
    lora: KleinLoraDeviceSet,
    img_mod_dev: ModVecsDevice,
    txt_mod_dev: ModVecsDevice,
    single_mod_dev: SingleModVecsDevice,
    cos_t: Tensor,
    sin_t: Tensor,
    ctx: DeviceContext,
) raises -> KleinStackForward:
    var scratch = ScratchRingAllocator(
        ctx, FWD_SCRATCH_SLAB_BYTES, FWD_SCRATCH_NUM_SLABS
    )
    var fwd = klein_stack_lora_forward_device_inputs_resident_moddev_rope_scratch[
        H, Dh, N_IMG_, N_TXT_, S_
    ](
        img_tokens_t, txt_tokens_t, base, dbw, sbw, lora,
        img_mod_dev, txt_mod_dev, single_mod_dev, cos_t, sin_t,
        KDIM, KF, KIN_CH, KTXT_CH, KOUT_CH, KEPS, ctx, scratch,
    )
    return fwd^


@fieldwise_init
struct Stats(Copyable, Movable, ImplicitlyCopyable):
    var elems: Int
    var nonzero: Int
    var nonfinite: Int
    var abs_sum: Float64
    var sumsq: Float64
    var max_abs: Float32


def _empty_stats() -> Stats:
    return Stats(0, 0, 0, Float64(0.0), Float64(0.0), Float32(0.0))


def _is_nonfinite(x: Float32) -> Bool:
    if x != x:
        return True
    return (x - x) != Float32(0.0)


def _abs32(x: Float32) -> Float32:
    if x < Float32(0.0):
        return -x
    return x


def _scan_value(x: Float32, var stats: Stats) -> Stats:
    stats.elems += 1
    if x != Float32(0.0):
        stats.nonzero += 1
    if _is_nonfinite(x):
        stats.nonfinite += 1
        return stats^
    var ax = _abs32(x)
    stats.abs_sum += Float64(ax)
    var xf = Float64(x)
    stats.sumsq += xf * xf
    if ax > stats.max_abs:
        stats.max_abs = ax
    return stats^


def _scan_values(values: List[Float32], var stats: Stats) -> Stats:
    for i in range(len(values)):
        stats = _scan_value(values[i], stats)
    return stats^


def _scan_diff(left: List[Float32], right: List[Float32], var stats: Stats) raises -> Stats:
    if len(left) != len(right):
        raise Error("diff len mismatch")
    for i in range(len(left)):
        stats = _scan_value(left[i] - right[i], stats)
    return stats^


def _l2(stats: Stats) -> Float64:
    return sqrt(stats.sumsq)


def _sec(ns0: UInt, ns1: UInt) -> Float64:
    return Float64(ns1 - ns0) / Float64(1000000000.0)


def _mse_host(a: Tensor, b: Tensor, ctx: DeviceContext) raises -> Float32:
    var d = sub(a, b, ctx)
    var sq = mul(d, d, ctx)
    var h = sq.to_host(ctx)
    var s = Float32(0.0)
    for i in range(len(h)):
        s += h[i]
    return s / Float32(len(h))


def _compare_tensor(
    label: String, got_tensor: Tensor, expected_tensor: Tensor,
    ctx: DeviceContext, min_cos: Float64,
) raises:
    var got = got_tensor.to_host(ctx)
    var expected = expected_tensor.to_host(ctx)
    if len(got) != len(expected):
        raise Error(label + String(": len mismatch"))
    var dot = Float64(0.0)
    var na = Float64(0.0)
    var nb = Float64(0.0)
    var max_abs = Float32(0.0)
    var nonfinite = 0
    for i in range(len(got)):
        var a = got[i]
        var b = expected[i]
        if _is_nonfinite(a) or _is_nonfinite(b):
            nonfinite += 1
            continue
        dot += Float64(a) * Float64(b)
        na += Float64(a) * Float64(a)
        nb += Float64(b) * Float64(b)
        var ad = _abs32(a - b)
        if ad > max_abs:
            max_abs = ad
    var cos = dot / (sqrt(na) * sqrt(nb))
    print(label, "n =", len(got), "cos =", cos, "max_abs_diff =", max_abs, "nonfinite =", nonfinite)
    if cos < min_cos:
        raise Error(label + String(": cosine below gate"))


def _double_suffix(slot: Int) -> String:
    if slot == 0:
        return String("attn.to_q")
    if slot == 1:
        return String("attn.to_k")
    if slot == 2:
        return String("attn.to_v")
    if slot == 3:
        return String("attn.to_out.0")
    if slot == 4:
        return String("ff.linear_in")
    if slot == 5:
        return String("ff.linear_out")
    if slot == 6:
        return String("attn.add_q_proj")
    if slot == 7:
        return String("attn.add_k_proj")
    if slot == 8:
        return String("attn.add_v_proj")
    if slot == 9:
        return String("attn.to_add_out")
    if slot == 10:
        return String("ff_context.linear_in")
    return String("ff_context.linear_out")


def _single_suffix(slot: Int) -> String:
    if slot == 0:
        return String("attn.to_qkv_mlp_proj")
    return String("attn.to_out")


def _phase_key(phase: String, prefix: String, suffix: String) -> String:
    return phase + String(".") + prefix + String(".") + suffix


def _load_list(st: ShardedSafeTensors, key: String, ctx: DeviceContext) raises -> List[Float32]:
    return Tensor.from_view(st.tensor_view(key), ctx).to_host(ctx)


def _zeros(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(Float32(0.0))
    return out^


def _adamw_step_f32(
    mut p: List[Float32], g: List[Float32], mut m: List[Float32], mut v: List[Float32],
    t: Int, lr: Float32,
) raises:
    if len(p) != len(g) or len(p) != len(m) or len(p) != len(v):
        raise Error("adamw len mismatch")
    var b1p = Float32(1.0)
    var b2p = Float32(1.0)
    for _ in range(t):
        b1p *= BETA1
        b2p *= BETA2
    var bc1 = Float32(1.0) - b1p
    var bc2 = Float32(1.0) - b2p
    var bc2_sqrt = sqrt(bc2)
    var step_size = lr / bc1
    var decay = Float32(1.0) - lr * WEIGHT_DECAY
    for i in range(len(p)):
        p[i] = p[i] * decay
        m[i] = m[i] + (Float32(1.0) - BETA1) * (g[i] - m[i])
        v[i] = BETA2 * v[i] + (Float32(1.0) - BETA2) * g[i] * g[i]
        var denom = sqrt(v[i]) / bc2_sqrt + EPS
        p[i] = p[i] - step_size * m[i] / denom


def _simulate_two_step_update(
    before: List[Float32], grad0: List[Float32], grad1: List[Float32]
) raises -> List[Float32]:
    var p = before.copy()
    var m = _zeros(len(p))
    var v = _zeros(len(p))
    _adamw_step_f32(p, grad0, m, v, 1, Float32(0.0))
    _adamw_step_f32(p, grad1, m, v, 2, LR_STEP1)
    return p^


def _record_param(
    step0_ad: ShardedSafeTensors,
    step1_ad: ShardedSafeTensors,
    key_prefix: String,
    param_suffix: String,
    mojo_grad: List[Float32],
    var ref_grad_stats: Stats,
    var mojo_grad_stats: Stats,
    var grad_err_stats: Stats,
    var ref_update_stats: Stats,
    var update_err_stats: Stats,
    ctx: DeviceContext,
) raises -> Tuple[Stats, Stats, Stats, Stats, Stats]:
    var before_key = _phase_key(String("adapter_before"), key_prefix, param_suffix)
    var after_key = _phase_key(String("adapter_after"), key_prefix, param_suffix)
    var grad_key = _phase_key(String("adapter_post_clip_grad"), key_prefix, param_suffix)

    var before = _load_list(step1_ad, before_key, ctx)
    var after_ref = _load_list(step1_ad, after_key, ctx)
    var grad0 = _load_list(step0_ad, grad_key, ctx)
    var grad1_ref = _load_list(step1_ad, grad_key, ctx)

    ref_grad_stats = _scan_values(grad1_ref, ref_grad_stats)
    mojo_grad_stats = _scan_values(mojo_grad, mojo_grad_stats)
    grad_err_stats = _scan_diff(mojo_grad, grad1_ref, grad_err_stats)

    var after_mojo = _simulate_two_step_update(before, grad0, mojo_grad)
    ref_update_stats = _scan_diff(after_ref, before, ref_update_stats)
    update_err_stats = _scan_diff(after_mojo, after_ref, update_err_stats)

    return (
        ref_grad_stats^, mojo_grad_stats^, grad_err_stats^,
        ref_update_stats^, update_err_stats^,
    )


def _print_grad_probe(
    step1_ad: ShardedSafeTensors,
    label: String,
    key_prefix: String,
    param_suffix: String,
    mojo_grad: List[Float32],
    ctx: DeviceContext,
) raises:
    var grad_key = _phase_key(String("adapter_post_clip_grad"), key_prefix, param_suffix)
    var grad_ref = _load_list(step1_ad, grad_key, ctx)
    var ref_stats = _scan_values(grad_ref, _empty_stats())
    var mojo_stats = _scan_values(mojo_grad, _empty_stats())
    var err_stats = _scan_diff(mojo_grad, grad_ref, _empty_stats())
    print("grad_probe", label, param_suffix,
          "ref_l2 =", _l2(ref_stats),
          "mojo_l2 =", _l2(mojo_stats),
          "err_l2 =", _l2(err_stats),
          "ref_nonzero =", ref_stats.nonzero,
          "mojo_nonzero =", mojo_stats.nonzero,
          "max_err =", err_stats.max_abs)


def _compare_grads_and_update(grads: KleinLoraGrads, ctx: DeviceContext) raises:
    var step0_ad = ShardedSafeTensors.open(String(STEP0_ADAPTERS))
    var step1_ad = ShardedSafeTensors.open(String(STEP1_ADAPTERS))

    var ref_grad_stats = _empty_stats()
    var mojo_grad_stats = _empty_stats()
    var grad_err_stats = _empty_stats()
    var ref_update_stats = _empty_stats()
    var update_err_stats = _empty_stats()

    for bi in range(KNUM_DOUBLE):
        var base = bi * 12
        for slot in range(12):
            var prefix = flux2_double_module(bi, _double_suffix(slot))
            if bi == 0:
                _print_grad_probe(
                    step1_ad, String("double0.") + _double_suffix(slot), prefix,
                    String("lora_down.weight"), grads.dbl_d_a[base + slot], ctx,
                )
                _print_grad_probe(
                    step1_ad, String("double0.") + _double_suffix(slot), prefix,
                    String("lora_up.weight"), grads.dbl_d_b[base + slot], ctx,
                )
            var down = _record_param(
                step0_ad, step1_ad, prefix, String("lora_down.weight"),
                grads.dbl_d_a[base + slot],
                ref_grad_stats, mojo_grad_stats, grad_err_stats,
                ref_update_stats, update_err_stats, ctx,
            )
            ref_grad_stats = down[0]
            mojo_grad_stats = down[1]
            grad_err_stats = down[2]
            ref_update_stats = down[3]
            update_err_stats = down[4]
            var up = _record_param(
                step0_ad, step1_ad, prefix, String("lora_up.weight"),
                grads.dbl_d_b[base + slot],
                ref_grad_stats, mojo_grad_stats, grad_err_stats,
                ref_update_stats, update_err_stats, ctx,
            )
            ref_grad_stats = up[0]
            mojo_grad_stats = up[1]
            grad_err_stats = up[2]
            ref_update_stats = up[3]
            update_err_stats = up[4]

    for bi in range(KNUM_SINGLE):
        var base = bi * 2
        for slot in range(2):
            var prefix = flux2_single_module(bi, _single_suffix(slot))
            if bi == 0:
                _print_grad_probe(
                    step1_ad, String("single0.") + _single_suffix(slot), prefix,
                    String("lora_down.weight"), grads.sgl_d_a[base + slot], ctx,
                )
                _print_grad_probe(
                    step1_ad, String("single0.") + _single_suffix(slot), prefix,
                    String("lora_up.weight"), grads.sgl_d_b[base + slot], ctx,
                )
            var down = _record_param(
                step0_ad, step1_ad, prefix, String("lora_down.weight"),
                grads.sgl_d_a[base + slot],
                ref_grad_stats, mojo_grad_stats, grad_err_stats,
                ref_update_stats, update_err_stats, ctx,
            )
            ref_grad_stats = down[0]
            mojo_grad_stats = down[1]
            grad_err_stats = down[2]
            ref_update_stats = down[3]
            update_err_stats = down[4]
            var up = _record_param(
                step0_ad, step1_ad, prefix, String("lora_up.weight"),
                grads.sgl_d_b[base + slot],
                ref_grad_stats, mojo_grad_stats, grad_err_stats,
                ref_update_stats, update_err_stats, ctx,
            )
            ref_grad_stats = up[0]
            mojo_grad_stats = up[1]
            grad_err_stats = up[2]
            ref_update_stats = up[3]
            update_err_stats = up[4]

    var ref_grad_l2 = _l2(ref_grad_stats)
    var mojo_grad_l2 = _l2(mojo_grad_stats)
    var grad_err_l2 = _l2(grad_err_stats)
    var ref_update_l2 = _l2(ref_update_stats)
    var update_err_l2 = _l2(update_err_stats)

    print("ref_grad: elems =", ref_grad_stats.elems, "nonzero =", ref_grad_stats.nonzero,
          "nonfinite =", ref_grad_stats.nonfinite, "abs_sum =", ref_grad_stats.abs_sum,
          "l2 =", ref_grad_l2, "max_abs =", ref_grad_stats.max_abs)
    print("mojo_grad: elems =", mojo_grad_stats.elems, "nonzero =", mojo_grad_stats.nonzero,
          "nonfinite =", mojo_grad_stats.nonfinite, "abs_sum =", mojo_grad_stats.abs_sum,
          "l2 =", mojo_grad_l2, "max_abs =", mojo_grad_stats.max_abs)
    print("grad_error: nonzero =", grad_err_stats.nonzero, "nonfinite =", grad_err_stats.nonfinite,
          "abs_sum =", grad_err_stats.abs_sum, "l2 =", grad_err_l2,
          "max_abs =", grad_err_stats.max_abs)
    print("ref_update: elems =", ref_update_stats.elems, "nonzero =", ref_update_stats.nonzero,
          "nonfinite =", ref_update_stats.nonfinite, "abs_sum =", ref_update_stats.abs_sum,
          "l2 =", ref_update_l2, "max_abs =", ref_update_stats.max_abs)
    print("update_error: nonzero =", update_err_stats.nonzero, "nonfinite =", update_err_stats.nonfinite,
          "abs_sum =", update_err_stats.abs_sum, "l2 =", update_err_l2,
          "max_abs =", update_err_stats.max_abs)

    if ref_grad_stats.elems != 43515904 or ref_update_stats.elems != 43515904:
        raise Error("Klein grad/update replay element-count mismatch")
    if ref_grad_stats.nonfinite != 0 or mojo_grad_stats.nonfinite != 0 or grad_err_stats.nonfinite != 0:
        raise Error("Klein grad replay has nonfinite values")
    if ref_update_stats.nonfinite != 0 or update_err_stats.nonfinite != 0:
        raise Error("Klein update replay has nonfinite values")
    if grad_err_l2 > ref_grad_l2 * GRAD_REL_EPS:
        raise Error("Klein Mojo grad differs from Serenity beyond tolerance")
    if update_err_l2 > ref_update_l2 * UPDATE_REL_EPS:
        raise Error("Klein Mojo AdamW update differs from Serenity beyond tolerance")


def main() raises:
    var ctx = DeviceContext()
    var all0 = perf_counter_ns()

    print("=== Klein train-ref grad/update replay ===")
    print("[step0]", STEP0)
    print("[step1]", STEP1)
    print("[adapters]", STEP1_ADAPTERS)
    print("[shape] HL =", HL, "WL =", WL, "N_TXT =", N_TXT, "timestep =", TIMESTEP)

    var step1 = ShardedSafeTensors.open(String(STEP1))
    var img = cast_tensor(
        Tensor.from_view(step1.tensor_view(String("trace.packed_latent_input")), ctx),
        STDtype.BF16,
        ctx,
    )
    var txt = cast_tensor(
        Tensor.from_view(step1.tensor_view(String("trace.encoder_hidden_states")), ctx),
        STDtype.BF16,
        ctx,
    )
    var ref_packed = Tensor.from_view(step1.tensor_view(String("trace.packed_predicted_flow")), ctx)
    var ref_predicted = Tensor.from_view(step1.tensor_view(String("output.predicted")), ctx)
    var target = Tensor.from_view(step1.tensor_view(String("output.target")), ctx)

    var img_tok = _reshape_owned(img^, [N_IMG, KIN_CH])
    var txt_tok = _reshape_owned(txt^, [N_TXT, KTXT_CH])
    var img_host = img_tok.to_host(ctx)
    var txt_host = txt_tok.to_host(ctx)

    var load0 = perf_counter_ns()
    var ckpt = SafeTensors.open(String(CKPT))
    var ts_dev = Tensor.from_host([TIMESTEP], [1], STDtype.F32, ctx)
    var vec_silu = build_klein_vec_silu(ckpt, ts_dev, KTIMESTEP_DIM, KDIM, ctx)
    var base = load_klein_stack_base(ckpt, vec_silu, KDIM, ctx)
    var step_mod_w = load_klein_step_mod_weights(ckpt, KDIM, ctx)
    var dbw = List[DoubleBlockWeights]()
    for bi in range(KNUM_DOUBLE):
        dbw.append(load_double_block_weights(ckpt, bi, ctx))
    var sbw = List[SingleBlockWeights]()
    for bi in range(KNUM_SINGLE):
        sbw.append(load_single_block_weights(ckpt, bi, ctx, False))
    var lora_host = load_flux2_lora_fused_phase(
        String(STEP1_ADAPTERS), String("adapter_before"), KNUM_DOUBLE, KNUM_SINGLE, ctx
    )
    var lora_dev = klein_lora_set_to_device(lora_host, ctx)
    var load1 = perf_counter_ns()
    print("[load] base +", len(dbw), "double +", len(sbw), "single blocks + adapter_before")

    # Serenity/diffusers Flux2 applies RoPE as x.float()*cos/sin then casts
    # back to x dtype; tables are F32, q/k activations remain BF16.
    var rope_tup = build_klein_rope_tables_hw_port[HL, WL, N_TXT, KH, KDh](ctx, STDtype.F32)
    ref cos_t = rope_tup[0]
    ref sin_t = rope_tup[1]
    var mods = build_klein_step_mods_device_cached(
        step_mod_w, TIMESTEP, Optional[Float32](None), KTIMESTEP_DIM, KDIM, ctx
    )
    var img_mod_dev = mods[0].copy()
    var txt_mod_dev = mods[1].copy()
    var single_mod_dev = mods[2].copy()
    print("[forward] training forward on Serenity step001 tensors ...")
    var fwd0 = perf_counter_ns()
    var img_arc = TArc(img_tok^)
    var txt_arc = TArc(txt_tok^)
    var fwd = _forward_with_scratch[
        KH, KDh, N_IMG, N_TXT, S
    ](
        img_arc, txt_arc, base, dbw, sbw, lora_dev,
        img_mod_dev, txt_mod_dev, single_mod_dev, cos_t, sin_t,
        ctx,
    )
    var fwd1 = perf_counter_ns()

    var flow_tokens = Tensor.from_host(fwd.out.copy(), [1, N_IMG, KOUT_CH], STDtype.BF16, ctx)
    var flow_b = _reshape(flow_tokens, [1, HL, WL, KOUT_CH], ctx)
    var flow_perm = _permute(flow_b, [0, 3, 1, 2], ctx)
    var predicted_flow_patch = _reshape_owned(flow_perm^, [1, KOUT_CH, HL, WL])
    var predicted = _unpatchify_packed(predicted_flow_patch, ctx)

    _compare_tensor("packed_flow", flow_tokens, ref_packed, ctx, MIN_FORWARD_COS)
    _compare_tensor("output.predicted", predicted, ref_predicted, ctx, MIN_FORWARD_COS)

    var loss0 = perf_counter_ns()
    # Serenity/PyTorch promotes BF16 predicted - F32 target to F32 for MSE.
    var predicted_f32 = cast_tensor(predicted, STDtype.F32, ctx)
    var loss = _mse_host(predicted_f32, target, ctx)
    var d_pred = mse_backward(predicted_f32, target, ctx)
    var d_patch = _patchify_packed(d_pred, ctx)
    var d_patch_nhwc = _permute(d_patch, [0, 2, 3, 1], ctx)
    var d_flow_tokens = _reshape_owned(d_patch_nhwc^, [N_IMG, KOUT_CH])
    var loss1 = perf_counter_ns()
    var loss_err = abs(loss - SERENITY_LOSS)
    print("loss =", loss, "reference trainer loss =", SERENITY_LOSS, "abs_err =", loss_err)
    if loss_err > LOSS_EPS:
        raise Error("Klein step001 loss mismatch")

    print("[backward] Klein LoRA backward ...")
    var bwd0 = perf_counter_ns()
    var scratch_bwd = ScratchRingAllocator(
        ctx, BWD_SCRATCH_SLAB_BYTES, BWD_SCRATCH_NUM_SLABS
    )
    var grads = klein_stack_lora_backward_resident_moddev_rope_scratch[
        KH, KDh, N_IMG, N_TXT, S
    ](
        d_flow_tokens.to_host(ctx), img_host, txt_host,
        base, dbw, sbw, lora_dev,
        img_mod_dev, txt_mod_dev, single_mod_dev, cos_t, sin_t,
        fwd, KDIM, KF, KIN_CH, KTXT_CH, KOUT_CH, KEPS, ctx, scratch_bwd,
        False, False,
    )
    var bwd1 = perf_counter_ns()

    print("[compare] grads + F32 AdamW two-step update ...")
    var cmp0 = perf_counter_ns()
    _compare_grads_and_update(grads, ctx)
    var cmp1 = perf_counter_ns()
    var all1 = perf_counter_ns()

    print("time_s: load =", _sec(load0, load1),
          " forward =", _sec(fwd0, fwd1),
          " loss =", _sec(loss0, loss1),
          " backward =", _sec(bwd0, bwd1),
          " compare =", _sec(cmp0, cmp1),
          " total =", _sec(all0, all1))
    print("KLEIN TRAIN REF GRAD/UPDATE REPLAY PASS")

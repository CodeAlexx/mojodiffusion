# MiniMax-H3 exact-50 reduced mixed-dtype FLASH DiT-core stack parity.
#
# Oracle: pinned Musubi b8717864713c9e4e7ef3d56eba1fc695a9b626a5.
# This is 50 reduced DiT block cores with injected per-row modulation, not 50
# full DiTBlocks and not released geometry. Each block has unique deterministic
# BF16 base/modulation and F32 LoRA leaves. The device stack retains only its 51
# BF16 residual-stream states; the core backward recomputes all internal
# attention/MLP activations one block at a time in exact reverse order.
#
# REQUIRED invocation (bare Mojo output is insufficient evidence):
#   bash serenitymojo/models/minimax_h3/parity/run_minimax_h3_training_stack50_bf16_flash_parity.sh

from max.gpu.host import DeviceContext
from std.collections import List
from std.math import isfinite, sqrt
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.models.minimax_h3.parity.training_block_device_reference import (
    MiniMaxH3BlockModulationDevice,
    MiniMaxH3LoraAdapterDevice,
    MiniMaxH3TrainingBlockLoraDevice,
    MiniMaxH3TrainingBlockWeightsDevice,
)
from serenitymojo.models.minimax_h3.parity.minimax_h3_training_block_bf16_flash_reference import (
    MINIMAX_H3_BF16_FC1_GATE_VALUE,
    MINIMAX_H3_BF16_FC1_LORA_B_GATE_VALUE,
    MINIMAX_H3_BF16_QKV_ALL_Q_K_V,
    minimax_h3_bf16_flash_core_backward,
    minimax_h3_bf16_flash_core_forward,
)


comptime TArc = ArcPointer[Tensor]
comptime FIXTURE = "serenitymojo/models/minimax_h3/parity/fixtures/minimax_h3_training_stack50_bf16_flash.safetensors"
comptime FIXTURE_SHA256 = "7f98af1639d600776c3aea8e51f846dbf2208ab6b4713c949eb32114b5c80b6c"
comptime BLOCKS = 50
comptime S = 3
comptime H = 56
comptime DH = 8
comptime D = 8
comptime FF = 12
comptime ROT = 4
comptime RANK = 2
comptime NONDEGENERATE_NORM2 = Float64(1.0e-28)

# Frozen class bounds sit just beyond the three-fresh-process envelope stored
# beside the fixture. Every tensor is still checked independently.
comptime CHECKPOINT_COS = Float32(0.99996)
comptime CHECKPOINT_MAX_ABS = Float32(3.3e-3)
comptime CHECKPOINT_REL_L2 = Float64(8.7e-3)
comptime CHECKPOINT_MAG = Float64(2.5e-3)
comptime HANDOFF_COS = Float32(0.999955)
comptime HANDOFF_MAX_ABS = Float32(1.7e-3)
comptime HANDOFF_REL_L2 = Float64(8.9e-3)
comptime HANDOFF_MAG = Float64(2.3e-3)


def _require_info(
    ref st: SafeTensors,
    name: String,
    dtype: STDtype,
    var shape: List[Int],
) raises:
    var info = st.tensor_info(name)
    if info.dtype != dtype:
        raise Error(String("fixture dtype mismatch: ") + name)
    if info.shape != shape:
        raise Error(String("fixture shape mismatch: ") + name)


def _load_bf16(
    ref st: SafeTensors, name: String, var shape: List[Int]
) raises -> List[BFloat16]:
    _require_info(st, name, STDtype.BF16, shape^)
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    var p = tv.data.unsafe_ptr().unsafe_bitcast[BFloat16]()
    var out = List[BFloat16](capacity=tv.numel())
    for i in range(tv.numel()):
        out.append(p[i])
    return out^


def _load_f32(
    ref st: SafeTensors, name: String, var shape: List[Int]
) raises -> List[Float32]:
    _require_info(st, name, STDtype.F32, shape^)
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    var p = tv.data.unsafe_ptr().unsafe_bitcast[Float32]()
    var out = List[Float32](capacity=tv.numel())
    for i in range(tv.numel()):
        out.append(p[i])
    return out^


def _bf16_as_f32(
    ref st: SafeTensors, name: String, var shape: List[Int]
) raises -> List[Float32]:
    var values = _load_bf16(st, name, shape^)
    var out = List[Float32](capacity=len(values))
    for value in values:
        out.append(value.cast[DType.float32]())
    return out^


def _t(
    ref st: SafeTensors, name: String, var shape: List[Int], ctx: DeviceContext
) raises -> Tensor:
    return Tensor.from_host_bf16(_load_bf16(st, name, shape.copy()), shape^, ctx)


def _ta(
    ref st: SafeTensors, name: String, var shape: List[Int], ctx: DeviceContext
) raises -> TArc:
    return TArc(_t(st, name, shape^, ctx))


def _t_f32(
    ref st: SafeTensors, name: String, var shape: List[Int], ctx: DeviceContext
) raises -> Tensor:
    return Tensor.from_host(_load_f32(st, name, shape.copy()), shape^, STDtype.F32, ctx)


def _ta_f32(
    ref st: SafeTensors, name: String, var shape: List[Int], ctx: DeviceContext
) raises -> TArc:
    return TArc(_t_f32(st, name, shape^, ctx))


def _prefix(block: Int) -> String:
    return String("block.") + String(block)


def _adapter(
    ref st: SafeTensors,
    prefix: String,
    name: String,
    inf: Int,
    outf: Int,
    scale: Float32,
    ctx: DeviceContext,
) raises -> MiniMaxH3LoraAdapterDevice:
    var p = prefix + String(".lora.") + name
    return MiniMaxH3LoraAdapterDevice(
        _ta_f32(st, p + String(".a"), [RANK, inf], ctx),
        _ta_f32(st, p + String(".b"), [outf, RANK], ctx),
        RANK,
        inf,
        outf,
        scale,
    )


def _weights(
    ref st: SafeTensors, prefix: String, ctx: DeviceContext
) raises -> MiniMaxH3TrainingBlockWeightsDevice:
    comptime I = H * DH
    var p = prefix + String(".w.")
    return MiniMaxH3TrainingBlockWeightsDevice(
        _ta(st, p + String("norm1"), [D], ctx),
        _ta(st, p + String("qkv"), [3 * I, D], ctx),
        _ta(st, p + String("q_norm"), [DH], ctx),
        _ta(st, p + String("k_norm"), [DH], ctx),
        _ta(st, p + String("out_proj"), [D, I], ctx),
        _ta(st, p + String("norm2"), [D], ctx),
        _ta(st, p + String("fc1"), [2 * FF, D], ctx),
        _ta(st, p + String("fc2"), [D, FF], ctx),
    )


def _modulation(
    ref st: SafeTensors, prefix: String, ctx: DeviceContext
) raises -> MiniMaxH3BlockModulationDevice:
    var p = prefix + String(".mod.")
    return MiniMaxH3BlockModulationDevice(
        _ta(st, p + String("shift_msa"), [S, D], ctx),
        _ta(st, p + String("scale_msa"), [S, D], ctx),
        _ta(st, p + String("gate_msa"), [S, D], ctx),
        _ta(st, p + String("shift_mlp"), [S, D], ctx),
        _ta(st, p + String("scale_mlp"), [S, D], ctx),
        _ta(st, p + String("gate_mlp"), [S, D], ctx),
    )


def _lora(
    ref st: SafeTensors, prefix: String, scale: Float32, ctx: DeviceContext
) raises -> MiniMaxH3TrainingBlockLoraDevice:
    comptime I = H * DH
    return MiniMaxH3TrainingBlockLoraDevice(
        _adapter(st, prefix, "qkv", D, 3 * I, scale, ctx),
        _adapter(st, prefix, "out_proj", I, D, scale, ctx),
        _adapter(st, prefix, "fc1", D, 2 * FF, scale, ctx),
        _adapter(st, prefix, "fc2", FF, D, scale, ctx),
    )


def _check(
    label: String,
    got: Tensor,
    want: List[Float32],
    expected_shape: List[Int],
    expected_dtype: STDtype,
    ctx: DeviceContext,
    cos_tol: Float32,
    max_abs_tol: Float32,
    rel_l2_tol: Float64,
    mag_tol: Float64,
) raises -> Bool:
    if got.shape() != expected_shape:
        print("  FAIL", label, "shape", got.shape(), "expected", expected_shape)
        return False
    if got.dtype() != expected_dtype:
        print("  FAIL", label, "dtype")
        return False
    var actual = got.to_host(ctx)
    if len(actual) != len(want):
        print("  FAIL", label, "length")
        return False
    var dot = Float64(0.0)
    var ng = Float64(0.0)
    var nw = Float64(0.0)
    var nd = Float64(0.0)
    var worst = Float32(0.0)
    for i in range(len(actual)):
        var g = Float64(actual[i])
        var w = Float64(want[i])
        if not isfinite(g) or not isfinite(w):
            print("  FAIL", label, "nonfinite", i)
            return False
        dot += g * w
        ng += g * g
        nw += w * w
        var diff = g - w
        nd += diff * diff
        var delta = actual[i] - want[i]
        var ad = -delta if delta < 0.0 else delta
        if ad > worst:
            worst = ad
    if ng <= NONDEGENERATE_NORM2 or nw <= NONDEGENERATE_NORM2:
        print("  FAIL", label, "degenerate", ng, nw)
        return False
    var cos = Float32(dot / sqrt(ng * nw))
    var rel_l2 = sqrt(nd / nw)
    var mag_ratio = sqrt(ng / nw)
    var ok = (
        cos >= cos_tol and worst <= max_abs_tol
        and rel_l2 <= rel_l2_tol and abs(mag_ratio - 1.0) <= mag_tol
    )
    print(
        "  ", "ok" if ok else "FAIL", label, "cos", cos,
        "max_abs", worst, "rel_l2", rel_l2, "mag_ratio", mag_ratio,
    )
    return ok


def _check_grad(
    label: String,
    got: Tensor,
    ref st: SafeTensors,
    name: String,
    var shape: List[Int],
    ctx: DeviceContext,
    cos_tol: Float32,
    max_abs_tol: Float32,
    rel_l2_tol: Float64,
    mag_tol: Float64,
) raises -> Bool:
    return _check(
        label, got, _load_f32(st, name, shape.copy()), shape^, STDtype.F32, ctx,
        cos_tol, max_abs_tol, rel_l2_tol, mag_tol,
    )


def main() raises:
    print("MiniMax-H3 exact-50 reduced mixed-dtype FLASH core stack parity")
    print("  fixture sha256", FIXTURE_SHA256)
    print("  50 unique cores; H56/S3/Dh8/D8/F12/Rot4/rank2")
    print("  BF16 base/mod/compute/y/dx; F32 LoRA leaves + 400 dA/dB")
    print("  checkpointing: 51 residual states; per-core internal recompute")
    print("  reverse backward: block49 -> block0 with every d_x handoff gated")
    print("  reduced/injected-modulation evidence only; no product claim")
    var ctx = DeviceContext()
    var st = SafeTensors.open(String(FIXTURE))

    var multiplier = _load_f32(st, "meta.lora_multiplier", [1])[0]
    var alpha = _load_f32(st, "meta.lora_alpha", [1])[0]
    var rankf = _load_f32(st, "meta.lora_rank", [1])[0]
    var expected_scale = _load_f32(st, "meta.lora_scale", [1])[0]
    if (
        not isfinite(multiplier) or not isfinite(alpha) or not isfinite(rankf)
        or not isfinite(expected_scale) or rankf != Float32(Int(rankf))
        or Int(rankf) != RANK or multiplier != Float32(1.4)
        or alpha != Float32(1.0)
    ):
        raise Error("stack50 LoRA scalar/rank metadata mismatch")
    var scale = multiplier * alpha / rankf
    if abs(scale - expected_scale) > 1.0e-7 or abs(scale - Float32(0.7)) > 1.0e-7:
        raise Error("stack50 LoRA multiplier*alpha/rank mismatch")

    var weights = List[MiniMaxH3TrainingBlockWeightsDevice](capacity=BLOCKS)
    var modulations = List[MiniMaxH3BlockModulationDevice](capacity=BLOCKS)
    var adapters = List[MiniMaxH3TrainingBlockLoraDevice](capacity=BLOCKS)
    for block in range(BLOCKS):
        var p = _prefix(block)
        weights.append(_weights(st, p, ctx))
        modulations.append(_modulation(st, p, ctx))
        adapters.append(_lora(st, p, scale, ctx))

    var cos = _t(st, "in.cos", [S, ROT], ctx)
    var sin = _t(st, "in.sin", [S, ROT], ctx)
    var states = List[TArc](capacity=BLOCKS + 1)
    states.append(TArc(_t(st, "in.x", [S, D], ctx)))
    var ok = True
    for block in range(BLOCKS):
        ok = _check(
            String("checkpoint.input.") + String(block), states[block][],
            _bf16_as_f32(st, String("checkpoint.input.") + String(block), [S, D]), [S, D],
            STDtype.BF16, ctx, CHECKPOINT_COS, CHECKPOINT_MAX_ABS,
            CHECKPOINT_REL_L2, CHECKPOINT_MAG,
        ) and ok
        var y = minimax_h3_bf16_flash_core_forward[S, H, DH, D, FF, ROT](
            states[block][], weights[block], modulations[block], adapters[block],
            cos, sin, 1.0e-5, 1.0e-5,
            MINIMAX_H3_BF16_QKV_ALL_Q_K_V,
            MINIMAX_H3_BF16_FC1_GATE_VALUE,
            MINIMAX_H3_BF16_FC1_LORA_B_GATE_VALUE,
            ctx,
        )
        states.append(TArc(y^))
    ok = _check(
        "final.y", states[BLOCKS][], _bf16_as_f32(st, "out.y", [S, D]),
        [S, D], STDtype.BF16, ctx, CHECKPOINT_COS, CHECKPOINT_MAX_ABS,
        CHECKPOINT_REL_L2, CHECKPOINT_MAG,
    ) and ok

    var d_y = _t(st, "in.dy", [S, D], ctx)
    ok = _check(
        "handoff.input.50", d_y, _bf16_as_f32(st, "grad.input.50", [S, D]),
        [S, D], STDtype.BF16, ctx, 1.0, 0.0, 0.0, 0.0,
    ) and ok
    var adapter_count = 0
    var gradient_tensor_count = 0
    for reverse_index in range(BLOCKS):
        var block = BLOCKS - 1 - reverse_index
        var p = _prefix(block)
        var backward = minimax_h3_bf16_flash_core_backward[S, H, DH, D, FF, ROT](
            d_y, states[block][], weights[block], modulations[block], adapters[block],
            cos, sin, 1.0e-5, 1.0e-5,
            MINIMAX_H3_BF16_QKV_ALL_Q_K_V,
            MINIMAX_H3_BF16_FC1_GATE_VALUE,
            MINIMAX_H3_BF16_FC1_LORA_B_GATE_VALUE,
            ctx,
        )
        ok = _check(
            String("handoff.input.") + String(block), backward.d_x[],
            _bf16_as_f32(st, String("grad.input.") + String(block), [S, D]),
            [S, D], STDtype.BF16, ctx, HANDOFF_COS, HANDOFF_MAX_ABS,
            HANDOFF_REL_L2, HANDOFF_MAG,
        ) and ok
        comptime I = H * DH
        ok = _check_grad("grad.qkv.A." + String(block), backward.qkv.d_a[], st, p + ".grad.qkv.a", [RANK, D], ctx, 0.99972, 2.6e-5, 2.55e-2, 1.9e-2) and ok
        ok = _check_grad("grad.qkv.B." + String(block), backward.qkv.d_b[], st, p + ".grad.qkv.b", [3 * I, RANK], ctx, 0.99952, 1.45e-5, 4.3e-2, 2.9e-2) and ok
        ok = _check_grad("grad.out_proj.A." + String(block), backward.out_proj.d_a[], st, p + ".grad.out_proj.a", [RANK, I], ctx, 0.99975, 4.5e-6, 2.9e-2, 2.4e-2) and ok
        ok = _check_grad("grad.out_proj.B." + String(block), backward.out_proj.d_b[], st, p + ".grad.out_proj.b", [D, RANK], ctx, 0.9992, 1.8e-5, 5.3e-2, 3.4e-2) and ok
        ok = _check_grad("grad.fc1.A." + String(block), backward.fc1.d_a[], st, p + ".grad.fc1.a", [RANK, D], ctx, 0.99955, 7.0e-7, 3.3e-2, 1.9e-2) and ok
        ok = _check_grad("grad.fc1.B." + String(block), backward.fc1.d_b[], st, p + ".grad.fc1.b", [2 * FF, RANK], ctx, 0.99966, 8.5e-7, 2.8e-2, 2.4e-2) and ok
        ok = _check_grad("grad.fc2.A." + String(block), backward.fc2.d_a[], st, p + ".grad.fc2.a", [RANK, FF], ctx, 0.99962, 8.5e-7, 3.15e-2, 2.6e-2) and ok
        ok = _check_grad("grad.fc2.B." + String(block), backward.fc2.d_b[], st, p + ".grad.fc2.b", [D, RANK], ctx, 0.99952, 5.8e-7, 3.65e-2, 2.0e-2) and ok
        adapter_count += 4
        gradient_tensor_count += 8
        d_y = backward.d_x[].clone(ctx)

    if adapter_count != 200 or gradient_tensor_count != 400 or len(states) != 51:
        raise Error("stack50 exact count invariant failed")
    if ok:
        print("PASS: 50 cores, 51 checkpoints, 50 reverse handoffs, 200 adapters, 400 dA/dB tensors")
    else:
        raise Error("MiniMax-H3 stack50 BF16 flash parity failed")

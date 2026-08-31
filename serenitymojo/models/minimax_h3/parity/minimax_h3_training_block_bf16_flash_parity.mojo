# MiniMax-H3 reduced mixed-dtype FLASH DiT-block-core parity gate.
#
# Oracle: pinned Musubi b8717864713c9e4e7ef3d56eba1fc695a9b626a5,
# forced Torch CUDNN_ATTENTION. Base/input/mod/compute/y/dx are BF16; LoRA A/B
# leaves and all eight saved LoRA gradients are F32, matching pinned autocast.
# Per-row modulation is injected: AdalnProj and segment gathering are excluded.
# The Mojo path uses only shared sdpa_flash_train_fwd/sdpa_flash_backward and
# therefore allocates no quadratic SxS attention matrix. This is not a full
# DiTBlock gate.
#
# Evidence level: reduced-shape mixed-dtype flash core parity only. This gate
# and its implementation intentionally live under parity/; not a product API.
# REQUIRED/SUPPORTED invocation (bare Mojo output is insufficient evidence):
#   bash serenitymojo/models/minimax_h3/parity/run_minimax_h3_training_block_bf16_flash_parity.sh

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
    minimax_h3_training_swap_fc1_rows_device,
)
from serenitymojo.models.minimax_h3.parity.minimax_h3_training_block_bf16_flash_reference import (
    MINIMAX_H3_BF16_FC1_GATE_VALUE,
    MINIMAX_H3_BF16_FC1_LORA_B_GATE_VALUE,
    MINIMAX_H3_BF16_QKV_ALL_Q_K_V,
    minimax_h3_bf16_flash_core_backward,
    minimax_h3_bf16_flash_core_forward,
)


comptime TArc = ArcPointer[Tensor]
comptime REF = "serenitymojo/models/minimax_h3/parity/fixtures/minimax_h3_training_block_bf16_flash.safetensors"
comptime REF_SHA256 = "096ba15169c9d7ebaeb25c30dd102ed3a4a44a52a4dee3ee2becd8242f737647"
comptime S = 3
comptime H = 56
comptime DH = 8
comptime D = 8
comptime FF = 12
comptime ROT = 4
comptime RANK = 2

# BF16/cuDNN value-class gates cover direction, elementwise error, relative
# L2, and norm preservation rather than demanding bit identity from two
# different flash-attention implementations. Limits are tensor-specific and
# frozen from the durable three-fresh-process envelope receipt adjacent to the
# fixture; a single permissive global threshold can no longer hide a local
# regression.
comptime NONDEGENERATE_NORM2 = Float64(1.0e-20)


def _require_info(
    ref st: SafeTensors,
    name: String,
    dtype: STDtype,
    var shape: List[Int],
) raises:
    var info = st.tensor_info(name)
    if info.dtype != dtype:
        raise Error(String("fixture dtype mismatch for ") + name)
    if info.shape != shape:
        raise Error(String("fixture shape mismatch for ") + name)


def _load_bf16(
    ref st: SafeTensors,
    name: String,
    var shape: List[Int],
) raises -> List[BFloat16]:
    _require_info(st, name, STDtype.BF16, shape^)
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    var p = tv.data.unsafe_ptr().bitcast[BFloat16]()
    var out = List[BFloat16](capacity=tv.numel())
    for i in range(tv.numel()):
        out.append(p[i])
    return out^


def _load_f32(
    ref st: SafeTensors,
    name: String,
    var shape: List[Int],
) raises -> List[Float32]:
    _require_info(st, name, STDtype.F32, shape^)
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    var p = tv.data.unsafe_ptr().bitcast[Float32]()
    var out = List[Float32](capacity=tv.numel())
    for i in range(tv.numel()):
        out.append(p[i])
    return out^


def _bf16_as_f32(
    ref st: SafeTensors,
    name: String,
    var shape: List[Int],
) raises -> List[Float32]:
    var values = _load_bf16(st, name, shape^)
    var out = List[Float32](capacity=len(values))
    for value in values:
        out.append(value.cast[DType.float32]())
    return out^


def _t(
    ref st: SafeTensors,
    name: String,
    var shape: List[Int],
    ctx: DeviceContext,
) raises -> Tensor:
    return Tensor.from_host_bf16(_load_bf16(st, name, shape.copy()), shape^, ctx)


def _ta(
    ref st: SafeTensors,
    name: String,
    var shape: List[Int],
    ctx: DeviceContext,
) raises -> TArc:
    return TArc(_t(st, name, shape^, ctx))


def _t_f32(
    ref st: SafeTensors,
    name: String,
    var shape: List[Int],
    ctx: DeviceContext,
) raises -> Tensor:
    return Tensor.from_host(_load_f32(st, name, shape.copy()), shape^, STDtype.F32, ctx)


def _ta_f32(
    ref st: SafeTensors,
    name: String,
    var shape: List[Int],
    ctx: DeviceContext,
) raises -> TArc:
    return TArc(_t_f32(st, name, shape^, ctx))


def _adapter(
    ref st: SafeTensors,
    name: String,
    inf: Int,
    outf: Int,
    ctx: DeviceContext,
    mapped_fc1_b: Bool = False,
) raises -> MiniMaxH3LoraAdapterDevice:
    var multiplier = _load_f32(st, "meta.lora_multiplier", [1])[0]
    var alpha = _load_f32(st, "meta.lora_alpha", [1])[0]
    var rankf = _load_f32(st, "meta.lora_rank", [1])[0]
    if not isfinite(rankf):
        raise Error("MiniMax-H3 LoRA rank must be finite")
    var ranki = Int(rankf)
    if rankf != Float32(ranki):
        raise Error("MiniMax-H3 LoRA rank must be integral")
    if ranki != RANK:
        raise Error("MiniMax-H3 reduced BF16 core requires exact rank 2")
    var scale = multiplier * alpha / rankf
    var expected = _load_f32(st, "meta.lora_scale", [1])[0]
    if (
        not isfinite(multiplier) or not isfinite(alpha) or not isfinite(scale)
        or not isfinite(expected)
    ):
        raise Error("MiniMax-H3 LoRA scale metadata must be finite")
    if multiplier != Float32(1.4) or alpha != Float32(1.0):
        raise Error("MiniMax-H3 reduced BF16 core LoRA scalar metadata mismatch")
    if abs(scale - expected) > 1.0e-7:
        raise Error("MiniMax-H3 BF16 LoRA multiplier*alpha/rank mismatch")
    var a = _ta_f32(st, String("lora.") + name + ".a", [RANK, inf], ctx)
    var b: TArc
    if mapped_fc1_b:
        # Raw training [gate;value] -> inference [value;gate] -> explicit raw
        # training seam. A/down is not row-swapped.
        var raw_b = _t_f32(st, String("lora.") + name + ".b", [outf, RANK], ctx)
        var runtime_b = minimax_h3_training_swap_fc1_rows_device(raw_b, FF, ctx)
        var train_b = minimax_h3_training_swap_fc1_rows_device(runtime_b, FF, ctx)
        b = TArc(train_b^)
    else:
        b = _ta_f32(st, String("lora.") + name + ".b", [outf, RANK], ctx)
    return MiniMaxH3LoraAdapterDevice(a, b, RANK, inf, outf, scale)


def _check(
    label: String,
    got: Tensor,
    want: List[Float32],
    expected_dtype: STDtype,
    ctx: DeviceContext,
    cos_tol: Float32,
    max_abs_tol: Float32,
    rel_l2_tol: Float64,
    mag_ratio_tol: Float64,
) raises -> Bool:
    if got.dtype() != expected_dtype:
        print("  FAIL", label, "dtype mismatch")
        return False
    var actual = got.to_host(ctx)
    if len(actual) != len(want):
        print("  FAIL", label, "length", len(actual), "!=", len(want))
        return False
    var dot = Float64(0.0)
    var ng = Float64(0.0)
    var nw = Float64(0.0)
    var nd = Float64(0.0)
    var worst = Float32(0.0)
    for i in range(len(actual)):
        var g = Float64(actual[i])
        var w = Float64(want[i])
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
        print("  FAIL", label, "degenerate norm2", ng, nw)
        return False
    var cos = Float32(dot / sqrt(ng * nw))
    var rel_l2 = sqrt(nd / nw)
    var mag_ratio = sqrt(ng / nw)
    var mag_error = abs(mag_ratio - 1.0)
    var ok = (
        cos >= cos_tol and worst <= max_abs_tol
        and rel_l2 <= rel_l2_tol and mag_error <= mag_ratio_tol
    )
    print(
        "  ", "ok" if ok else "FAIL", label, "cos", cos,
        "max_abs", worst, "rel_l2", rel_l2, "mag_ratio", mag_ratio,
    )
    return ok


def main() raises:
    print("MiniMax-H3 reduced mixed-dtype FLASH DiT block core parity")
    print("  fixture sha256", REF_SHA256)
    print("  H=56; S=3 Dh=8 D=8 F=12 Rot=4 rank=2")
    print("  BF16 base/input/mod/compute/y/dx; F32 LoRA A/B and dA/dB")
    print("  injected modulation: AdalnProj + segment gather excluded")
    print("  shared cuDNN flash fwd+bwd; noncausal; no SxS attention")
    print("  qkv=[all-q;all-k;all-v], raw train fc1=[gate;value]")
    print("  QKV checkpoint deinterleave excluded: see minimax_h3_loader_parity [3]")
    print("  explicit one-way runtime FC1 receipts + inverse checks exercised")
    var ctx = DeviceContext()
    var st = SafeTensors.open(String(REF))
    comptime I = H * DH

    # Exact dtype/shape checks are performed by every loader below. Exercise
    # the frozen FC1 boundary before handing raw Musubi order to the block.
    var raw_fc1 = _t(st, "w.fc1", [2 * FF, D], ctx)
    var runtime_fc1 = minimax_h3_training_swap_fc1_rows_device(raw_fc1, FF, ctx)
    var ok = _check(
        "boundary.fc1.raw_to_runtime", runtime_fc1,
        _bf16_as_f32(st, "boundary.fc1_runtime", [2 * FF, D]),
        STDtype.BF16, ctx, 1.0, 0.0, 0.0, 0.0,
    )
    var train_fc1 = minimax_h3_training_swap_fc1_rows_device(runtime_fc1, FF, ctx)
    ok = _check(
        "boundary.fc1.runtime_to_raw", train_fc1,
        _bf16_as_f32(st, "w.fc1", [2 * FF, D]), STDtype.BF16, ctx,
        1.0, 0.0, 0.0, 0.0,
    ) and ok

    var weights = MiniMaxH3TrainingBlockWeightsDevice(
        _ta(st, "w.norm1", [D], ctx),
        _ta(st, "w.qkv", [3 * I, D], ctx),
        _ta(st, "w.q_norm", [DH], ctx),
        _ta(st, "w.k_norm", [DH], ctx),
        _ta(st, "w.out_proj", [D, I], ctx),
        _ta(st, "w.norm2", [D], ctx),
        TArc(train_fc1^),
        _ta(st, "w.fc2", [D, FF], ctx),
    )
    var mod = MiniMaxH3BlockModulationDevice(
        _ta(st, "mod.shift_msa", [S, D], ctx),
        _ta(st, "mod.scale_msa", [S, D], ctx),
        _ta(st, "mod.gate_msa", [S, D], ctx),
        _ta(st, "mod.shift_mlp", [S, D], ctx),
        _ta(st, "mod.scale_mlp", [S, D], ctx),
        _ta(st, "mod.gate_mlp", [S, D], ctx),
    )
    var lora = MiniMaxH3TrainingBlockLoraDevice(
        _adapter(st, "qkv", D, 3 * I, ctx),
        _adapter(st, "out_proj", I, D, ctx),
        _adapter(st, "fc1", D, 2 * FF, ctx, True),
        _adapter(st, "fc2", FF, D, ctx),
    )
    var raw_fc1_b = _t_f32(st, "lora.fc1.b", [2 * FF, RANK], ctx)
    var runtime_fc1_b = minimax_h3_training_swap_fc1_rows_device(raw_fc1_b, FF, ctx)
    ok = _check(
        "boundary.fc1_lora_b.raw_to_runtime", runtime_fc1_b,
        _load_f32(st, "boundary.lora_fc1_b_runtime", [2 * FF, RANK]),
        STDtype.F32, ctx, 1.0, 0.0, 0.0, 0.0,
    ) and ok
    var train_fc1_b = minimax_h3_training_swap_fc1_rows_device(runtime_fc1_b, FF, ctx)
    ok = _check(
        "boundary.fc1_lora_b.runtime_to_raw", train_fc1_b,
        _load_f32(st, "lora.fc1.b", [2 * FF, RANK]), STDtype.F32, ctx,
        1.0, 0.0, 0.0, 0.0,
    ) and ok

    var x = _t(st, "in.x", [S, D], ctx)
    var cos = _t(st, "in.cos", [S, ROT], ctx)
    var sin = _t(st, "in.sin", [S, ROT], ctx)
    var dy = _t(st, "in.dy", [S, D], ctx)

    var y = minimax_h3_bf16_flash_core_forward[S, H, DH, D, FF, ROT](
        x, weights, mod, lora, cos, sin, 1.0e-5, 1.0e-5,
        MINIMAX_H3_BF16_QKV_ALL_Q_K_V,
        MINIMAX_H3_BF16_FC1_GATE_VALUE,
        MINIMAX_H3_BF16_FC1_LORA_B_GATE_VALUE,
        ctx,
    )
    var b = minimax_h3_bf16_flash_core_backward[S, H, DH, D, FF, ROT](
        dy, x, weights, mod, lora, cos, sin, 1.0e-5, 1.0e-5,
        MINIMAX_H3_BF16_QKV_ALL_Q_K_V,
        MINIMAX_H3_BF16_FC1_GATE_VALUE,
        MINIMAX_H3_BF16_FC1_LORA_B_GATE_VALUE,
        ctx,
    )

    ok = _check(
        "forward.y", y, _bf16_as_f32(st, "out.y", [S, D]), STDtype.BF16,
        ctx, 0.99999, 2.1e-3, 2.6e-3, 5.0e-4,
    ) and ok
    ok = _check(
        "grad.x", b.d_x[], _bf16_as_f32(st, "grad.x", [S, D]),
        STDtype.BF16, ctx, 0.99999, 2.1e-3, 3.4e-3, 9.0e-4,
    ) and ok
    ok = _check(
        "grad.qkv.A", b.qkv.d_a[], _load_f32(st, "grad.qkv.a", [RANK, D]),
        STDtype.F32, ctx, 0.99993, 4.0e-4, 1.1e-2, 5.0e-4,
    ) and ok
    ok = _check(
        "grad.qkv.B", b.qkv.d_b[], _load_f32(st, "grad.qkv.b", [3 * I, RANK]),
        STDtype.F32, ctx, 0.999965, 1.4e-4, 7.8e-3, 1.8e-3,
    ) and ok
    ok = _check(
        "grad.out_proj.A", b.out_proj.d_a[],
        _load_f32(st, "grad.out_proj.a", [RANK, I]),
        STDtype.F32, ctx, 0.99997, 7.0e-5, 6.5e-3, 4.0e-4,
    ) and ok
    ok = _check(
        "grad.out_proj.B", b.out_proj.d_b[],
        _load_f32(st, "grad.out_proj.b", [D, RANK]),
        STDtype.F32, ctx, 0.999955, 2.7e-4, 1.22e-2, 8.7e-3,
    ) and ok
    ok = _check(
        "grad.fc1.A", b.fc1.d_a[], _load_f32(st, "grad.fc1.a", [RANK, D]),
        STDtype.F32, ctx, 0.99999, 3.5e-5, 3.6e-3, 1.4e-3,
    ) and ok
    ok = _check(
        "grad.fc1.B", b.fc1.d_b[], _load_f32(st, "grad.fc1.b", [2 * FF, RANK]),
        STDtype.F32, ctx, 0.99996, 1.8e-5, 8.3e-3, 1.5e-3,
    ) and ok
    ok = _check(
        "grad.fc2.A", b.fc2.d_a[], _load_f32(st, "grad.fc2.a", [RANK, FF]),
        STDtype.F32, ctx, 0.999945, 1.5e-5, 1.0e-2, 1.3e-3,
    ) and ok
    ok = _check(
        "grad.fc2.B", b.fc2.d_b[], _load_f32(st, "grad.fc2.b", [D, RANK]),
        STDtype.F32, ctx, 0.99992, 1.4e-5, 1.26e-2, 4.0e-3,
    ) and ok
    if ok:
        print("PASS: MiniMax-H3 flash DiT core y, d_x, all 8 F32 LoRA grads")
    else:
        raise Error("MiniMax-H3 mixed-dtype flash DiT core parity failed")

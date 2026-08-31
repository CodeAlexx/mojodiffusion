# MiniMax-H3 RELEASED-GEOMETRY single DiT-block-core synthetic device smoke.
#
# NOT TORCH PARITY: there is no giant oracle fixture and no value comparison.
# This proves the parity-only mixed-dtype core compiles and runs at exact
# released block geometry with deterministic device-generated synthetic data:
# H=56, Dh=128, D=5376, F=14336, Rot=96. S=2 is deliberately tiny but still
# exercises noncausal two-token attention. Rank=2 caps adapter work/memory.
#
# Scope is the core after per-row modulation injection. AdalnProj/segment
# gathering, checkpoint loading/QKV deinterleave/FC1 mapping, stack, optimizer,
# save/resume, trainer, dataset, INT8, and product binding are all excluded.
# Base/input/modulation/compute/y/dx are BF16. LoRA leaves and returned dA/dB
# are F32, with BF16 autocast-equivalent adapter projection/gradient compute.
#
# Build serially (flash shim required), then run the binary. Peak memory below
# is sampled from CUDA-driver free-memory observations and is a lower-bound
# high-water estimate, not an allocator-exact CUDA event trace.

from max.gpu.host import DeviceContext
from std.collections import List
from std.math import cos as fcos, isfinite, sin as fsin, sqrt
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.offload.vmm_cuda import cu_mem_get_info
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import full_device, mul_scalar
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
comptime S = 2
comptime H = 56
comptime DH = 128
comptime D = 5376
comptime FF = 14336
comptime ROT = 96
comptime RANK = 2
comptime EPS = Float32(1.0e-5)


def _scaled_randn(
    var shape: List[Int], seed: UInt64, dtype: STDtype,
    scale: Float32, ctx: DeviceContext,
) raises -> Tensor:
    var raw = randn(shape^, seed, dtype, ctx)
    return mul_scalar(raw, scale, ctx)


def _arc_scaled_randn(
    var shape: List[Int], seed: UInt64, dtype: STDtype,
    scale: Float32, ctx: DeviceContext,
) raises -> TArc:
    return TArc(_scaled_randn(shape^, seed, dtype, scale, ctx))


def _arc_full(
    var shape: List[Int], value: Float32, dtype: STDtype, ctx: DeviceContext,
) raises -> TArc:
    return TArc(full_device(shape^, value, dtype, ctx))


def _adapter(
    inf: Int, outf: Int, seed: UInt64, ctx: DeviceContext,
) raises -> MiniMaxH3LoraAdapterDevice:
    var a_scale = Float32(1.0) / sqrt(Float32(inf))
    var a = _arc_scaled_randn([RANK, inf], seed, STDtype.F32, a_scale, ctx)
    var b = _arc_scaled_randn(
        [outf, RANK], seed + 1, STDtype.F32, Float32(0.02), ctx,
    )
    return MiniMaxH3LoraAdapterDevice(a, b, RANK, inf, outf, Float32(0.7))


def _rope(ctx: DeviceContext) raises -> Tuple[Tensor, Tensor]:
    var ch = List[Float32](capacity=S * ROT)
    var sh = List[Float32](capacity=S * ROT)
    for row in range(S):
        for half in range(2):
            for col in range(ROT // 2):
                var angle = Float32(0.001) * Float32(row + 1) * Float32(col + 1)
                ch.append(fcos(angle))
                sh.append(fsin(angle))
    return (
        Tensor.from_host(ch^, [S, ROT], STDtype.BF16, ctx),
        Tensor.from_host(sh^, [S, ROT], STDtype.BF16, ctx),
    )


def _update_min_free(current: Int, sample: Int) -> Int:
    return sample if sample < current else current


def _check_tensor(
    label: String, t: Tensor, dtype: STDtype, var shape: List[Int],
    ctx: DeviceContext,
) raises:
    if t.dtype() != dtype or t.shape() != shape:
        raise Error(label + ": released-geometry smoke dtype/shape mismatch")
    var values = t.to_host(ctx)
    var norm2 = Float64(0.0)
    var checksum = Float64(0.0)
    var max_abs = Float32(0.0)
    var bad = 0
    for i in range(len(values)):
        var v = values[i]
        if not isfinite(v):
            bad += 1
        var vf = Float64(v)
        norm2 += vf * vf
        # Stable small-period weighted checksum is an easy reproducibility
        # fingerprint without claiming an external numeric oracle.
        checksum += vf * Float64((i % 17) + 1)
        var av = -v if v < 0.0 else v
        if av > max_abs:
            max_abs = av
    print(
        "  ", label, "dtype", dtype.name(), "n", len(values),
        "nonfinite", bad, "norm", sqrt(norm2), "max_abs", max_abs,
        "checksum17", checksum,
    )
    if bad != 0 or norm2 <= Float64(1.0e-30):
        raise Error(label + ": released-geometry smoke nonfinite/degenerate")


def main() raises:
    print("MiniMax-H3 RELEASED-GEOMETRY synthetic FLASH DiT core smoke")
    print("  NOT Torch parity; deterministic shared randn seed manifest 1001..1302")
    print("  H=56 Dh=128 D=5376 F=14336 Rot=96 S=2 rank=2 eps=1e-5")
    print("  BF16 base/compute/y/dx; F32 LoRA leaves/all 8 grads")
    print("  injected modulation only; no loader/stack/trainer/dataset/INT8/product claim")
    var ctx = DeviceContext()
    var mem0 = cu_mem_get_info()
    var min_free = mem0.free_bytes
    print("  GPU free/total at start", mem0.free_bytes, mem0.total_bytes)
    comptime I = H * DH

    # Xavier-like scales keep the fully synthetic released-width path finite.
    var weights = MiniMaxH3TrainingBlockWeightsDevice(
        _arc_full([D], 1.0, STDtype.BF16, ctx),
        _arc_scaled_randn(
            [3 * I, D], 1001, STDtype.BF16,
            Float32(1.0) / sqrt(Float32(D)), ctx,
        ),
        _arc_full([DH], 1.0, STDtype.BF16, ctx),
        _arc_full([DH], 1.0, STDtype.BF16, ctx),
        _arc_scaled_randn(
            [D, I], 1002, STDtype.BF16,
            Float32(1.0) / sqrt(Float32(I)), ctx,
        ),
        _arc_full([D], 1.0, STDtype.BF16, ctx),
        _arc_scaled_randn(
            [2 * FF, D], 1003, STDtype.BF16,
            Float32(1.0) / sqrt(Float32(D)), ctx,
        ),
        _arc_scaled_randn(
            [D, FF], 1004, STDtype.BF16,
            Float32(1.0) / sqrt(Float32(FF)), ctx,
        ),
    )
    var mod = MiniMaxH3BlockModulationDevice(
        _arc_scaled_randn([S, D], 1101, STDtype.BF16, 0.05, ctx),
        _arc_scaled_randn([S, D], 1102, STDtype.BF16, 0.05, ctx),
        _arc_scaled_randn([S, D], 1103, STDtype.BF16, 0.20, ctx),
        _arc_scaled_randn([S, D], 1104, STDtype.BF16, 0.05, ctx),
        _arc_scaled_randn([S, D], 1105, STDtype.BF16, 0.05, ctx),
        _arc_scaled_randn([S, D], 1106, STDtype.BF16, 0.20, ctx),
    )
    var lora = MiniMaxH3TrainingBlockLoraDevice(
        _adapter(D, 3 * I, 1201, ctx),
        _adapter(I, D, 1203, ctx),
        _adapter(D, 2 * FF, 1205, ctx),
        _adapter(FF, D, 1207, ctx),
    )
    var x = _scaled_randn([S, D], 1301, STDtype.BF16, 0.20, ctx)
    var dy = _scaled_randn([S, D], 1302, STDtype.BF16, 0.10, ctx)
    var rope = _rope(ctx)
    ctx.synchronize()
    var mem_init = cu_mem_get_info()
    min_free = _update_min_free(min_free, mem_init.free_bytes)
    print("  GPU free after deterministic init", mem_init.free_bytes)

    var y = minimax_h3_bf16_flash_core_forward[S, H, DH, D, FF, ROT](
        x, weights, mod, lora, rope[0], rope[1], EPS, EPS,
        MINIMAX_H3_BF16_QKV_ALL_Q_K_V,
        MINIMAX_H3_BF16_FC1_GATE_VALUE,
        MINIMAX_H3_BF16_FC1_LORA_B_GATE_VALUE,
        ctx,
    )
    ctx.synchronize()
    var mem_fwd = cu_mem_get_info()
    min_free = _update_min_free(min_free, mem_fwd.free_bytes)
    print("  GPU free after forward", mem_fwd.free_bytes)
    _check_tensor("forward.y", y, STDtype.BF16, [S, D], ctx)

    var b = minimax_h3_bf16_flash_core_backward[S, H, DH, D, FF, ROT](
        dy, x, weights, mod, lora, rope[0], rope[1], EPS, EPS,
        MINIMAX_H3_BF16_QKV_ALL_Q_K_V,
        MINIMAX_H3_BF16_FC1_GATE_VALUE,
        MINIMAX_H3_BF16_FC1_LORA_B_GATE_VALUE,
        ctx,
    )
    ctx.synchronize()
    var mem_bwd = cu_mem_get_info()
    min_free = _update_min_free(min_free, mem_bwd.free_bytes)
    print("  GPU free after backward", mem_bwd.free_bytes)

    _check_tensor("grad.x", b.d_x[], STDtype.BF16, [S, D], ctx)
    _check_tensor("grad.qkv.A", b.qkv.d_a[], STDtype.F32, [RANK, D], ctx)
    _check_tensor("grad.qkv.B", b.qkv.d_b[], STDtype.F32, [3 * I, RANK], ctx)
    _check_tensor("grad.out_proj.A", b.out_proj.d_a[], STDtype.F32, [RANK, I], ctx)
    _check_tensor("grad.out_proj.B", b.out_proj.d_b[], STDtype.F32, [D, RANK], ctx)
    _check_tensor("grad.fc1.A", b.fc1.d_a[], STDtype.F32, [RANK, D], ctx)
    _check_tensor("grad.fc1.B", b.fc1.d_b[], STDtype.F32, [2 * FF, RANK], ctx)
    _check_tensor("grad.fc2.A", b.fc2.d_a[], STDtype.F32, [RANK, FF], ctx)
    _check_tensor("grad.fc2.B", b.fc2.d_b[], STDtype.F32, [D, RANK], ctx)

    var sampled_peak_delta = mem0.free_bytes - min_free
    var sampled_peak_used = mem0.total_bytes - min_free
    print("  sampled peak GPU delta bytes", sampled_peak_delta)
    print("  sampled peak total-used bytes", sampled_peak_used)
    print("PASS: released-geometry synthetic flash core smoke (not Torch parity)")

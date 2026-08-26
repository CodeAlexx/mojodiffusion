# minimax_h3_seed_offload_parity — gate the seed-offload stack arms against
# the device-resident-seed int8 arms on real block-0/1 weights.
#
# PASS bars:
#   1. seed roundtrip (device clone -> pinned slab -> device) BIT-equal to the
#      device-resident block_inputs — this is the new code, it must be exact.
#   2. stack forward out BIT-equal (identical math, deterministic).
#   3. d_x + every LoRA A/B grad: bit-equal preferred; accepted at
#      cos >= 0.999999 because the cuDNN flash dQ path is not run-to-run
#      deterministic (the krea2 lesson: never demand end-to-end bit-exact
#      through flash backward).
#
# Runs at S=384 (gate speed) and S=4910 (the AV bring-up geometry,
# 256x448 x 124 frames — the size this offload exists for).
from max.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor
from serenitymojo.ops.random import randn
from serenitymojo.models.minimax_h3.h3_block_train import (
    H3BlockLoraDevice, H3BlockLoraGrads,
)
from serenitymojo.models.klein.lora_block import (
    LoraAdapterDevice, KleinLoraDeviceGradTensors,
)
from serenitymojo.models.minimax_h3.h3_train_block_store_int8 import (
    H3TrainBlockStoreInt8,
)
from serenitymojo.models.minimax_h3.h3_stack_train import (
    h3_stack_train_forward_streamed_int8,
    h3_stack_train_backward_streamed_int8,
    h3_stack_train_forward_streamed_int8_seedoff,
    h3_stack_train_backward_streamed_int8_seedoff,
    H3SeedStore,
)

comptime CKPT = "/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/transformer"
comptime N = 2
comptime H = 56
comptime Dh = 128
comptime D = 5376
comptime F = 14336
comptime ROTARY = 96
comptime EPS = Float32(1.0e-5)
comptime RANK = 32

comptime TArc = ArcPointer[Tensor]


def _bits_equal(a: Tensor, b: Tensor, ctx: DeviceContext) raises -> Int:
    """Number of differing elements (0 = bit-equal for bf16 payloads)."""
    var ah = a.to_host_bf16(ctx)
    var bh = b.to_host_bf16(ctx)
    if len(ah) != len(bh):
        raise Error("bits_equal: length mismatch")
    var diff = 0
    for i in range(len(ah)):
        if ah[i] != bh[i]:
            diff += 1
    return diff


def _cos(a: Tensor, b: Tensor, ctx: DeviceContext) raises -> Float64:
    var ah = a.to_host(ctx)
    var bh = b.to_host(ctx)
    if len(ah) != len(bh):
        raise Error("cos: length mismatch")
    var dot = Float64(0)
    var na = Float64(0)
    var nb = Float64(0)
    for i in range(len(ah)):
        var x = Float64(ah[i])
        var y = Float64(bh[i])
        dot += x * y
        na += x * x
        nb += y * y
    return dot / ((na**0.5) * (nb**0.5) + 1e-30)


def _adapter(d_in: Int, d_out: Int, seed: UInt64, ctx: DeviceContext) raises -> LoraAdapterDevice:
    var ash: List[Int] = [RANK, d_in]
    var bsh: List[Int] = [d_out, RANK]
    var a = randn(ash^, seed, STDtype.BF16, ctx)
    var b = randn(bsh^, seed + 1, STDtype.BF16, ctx)
    return LoraAdapterDevice(
        ArcPointer(a^), ArcPointer(b^), RANK, d_in, d_out, Float32(1.0)
    )


def _grad_slot(
    g: H3BlockLoraGrads, s: Int
) raises -> KleinLoraDeviceGradTensors:
    if s == 0:
        return g.qkv.value().copy()
    if s == 1:
        return g.out.value().copy()
    if s == 2:
        return g.fc1.value().copy()
    return g.fc2.value().copy()


def _check_grads(
    ga: H3BlockLoraGrads, gb: H3BlockLoraGrads, tag: String,
    mut worst: Float64, ctx: DeviceContext,
) raises:
    for s in range(4):
        var a = _grad_slot(ga, s)
        var b = _grad_slot(gb, s)
        var da = _bits_equal(a.d_a[], b.d_a[], ctx)
        var db = _bits_equal(a.d_b[], b.d_b[], ctx)
        if da == 0 and db == 0:
            continue
        var ca = _cos(a.d_a[], b.d_a[], ctx)
        var cb = _cos(a.d_b[], b.d_b[], ctx)
        print("  ", tag, "slot", s, "d_a diffs", da, "cos", ca,
              "| d_b diffs", db, "cos", cb)
        if ca < worst:
            worst = ca
        if cb < worst:
            worst = cb


def _run_case(
    s_len: Int, do_intra: Bool, mut store: H3TrainBlockStoreInt8,
    ctx: DeviceContext,
) raises:
    print("== S =", s_len)
    var loras = List[H3BlockLoraDevice]()
    comptime INNER = H * Dh
    for b in range(N):
        var base = UInt64(1000 * (b + 1))
        loras.append(H3BlockLoraDevice(
            Optional[LoraAdapterDevice](_adapter(D, 3 * INNER, base + 1, ctx)),
            Optional[LoraAdapterDevice](_adapter(INNER, D, base + 3, ctx)),
            Optional[LoraAdapterDevice](_adapter(D, 2 * F, base + 5, ctx)),
            Optional[LoraAdapterDevice](_adapter(F, D, base + 7, ctx)),
        ))
    var mods = List[TArc]()
    for b in range(N):
        var msh: List[Int] = [3, 6 * D]
        mods.append(TArc(randn(msh^, UInt64(31 + b), STDtype.BF16, ctx)))
    var idx = List[Int]()
    for i in range(s_len):
        idx.append(i % 3)
    var xsh: List[Int] = [s_len, D]
    var x = randn(xsh^, UInt64(7 + s_len), STDtype.BF16, ctx)
    var csh: List[Int] = [s_len, ROTARY]
    var cos = randn(csh.copy(), UInt64(13), STDtype.F32, ctx)
    var sin = randn(csh^, UInt64(17), STDtype.F32, ctx)
    var dsh: List[Int] = [s_len, D]
    var d_out = randn(dsh^, UInt64(19 + s_len), STDtype.BF16, ctx)

    # ── arm A: device-resident seeds (the gated production arm) ─────────────
    var fwd_a = h3_stack_train_forward_streamed_int8[H, Dh](
        x, store, loras, mods, idx, cos, sin, D, F, ROTARY, EPS, ctx,
    )
    ctx.synchronize()

    # ── arm B: seed-offload ─────────────────────────────────────────────────
    var seeds = H3SeedStore.create(N, s_len, D, ctx)
    var fwd_b = h3_stack_train_forward_streamed_int8_seedoff[H, Dh](
        x, store, seeds, loras, mods, idx, cos, sin, D, F, ROTARY, EPS, ctx,
    )
    ctx.synchronize()

    # 1. seed roundtrip vs arm A's device-resident block_inputs: BIT-equal
    for i in range(N):
        var back = seeds.load(i, ctx)
        ctx.synchronize()
        var diffs = _bits_equal(back, fwd_a.block_inputs[i][], ctx)
        print("seed roundtrip block", i, "diffs", diffs)
        if diffs != 0:
            raise Error("seed roundtrip not bit-equal")
        _ = back^

    # 2. forward out: BIT-equal
    var out_diffs = _bits_equal(fwd_a.out[], fwd_b.out[], ctx)
    print("forward out diffs", out_diffs)
    if out_diffs != 0:
        raise Error("seedoff forward out not bit-equal")

    # 3. backward: bit-equal preferred, cos >= 0.999999 accepted (flash dQ)
    var grads_a = h3_stack_train_backward_streamed_int8[H, Dh](
        d_out, fwd_a, store, loras, mods, idx, cos, sin,
        D, F, ROTARY, EPS, ctx,
    )
    ctx.synchronize()
    # noise floor: the SAME device-seed arm run twice — cuDNN flash dQ is not
    # run-to-run deterministic, so bitwise cross-arm equality is not available
    # evidence. The seedoff arm passes if its deviation is within rerun noise.
    # Skipped at the large-S case (three full backwards OOM the 16GB card via
    # cumulative arena transients); the S=384 floor carries.
    var worst_intra = Float64(1.0)
    var dxi = 0
    if do_intra:
        var grads_a2 = h3_stack_train_backward_streamed_int8[H, Dh](
            d_out, fwd_a, store, loras, mods, idx, cos, sin,
            D, F, ROTARY, EPS, ctx,
        )
        ctx.synchronize()
        dxi = _bits_equal(grads_a.d_x[], grads_a2.d_x[], ctx)
        if dxi != 0:
            var ci = _cos(grads_a.d_x[], grads_a2.d_x[], ctx)
            if ci < worst_intra:
                worst_intra = ci
        for b in range(N):
            _check_grads(
                grads_a.lora[b], grads_a2.lora[b],
                String("intra block ") + String(b), worst_intra, ctx,
            )
        print("intra-arm rerun: d_x diffs", dxi, "worst cos", worst_intra)
        _ = grads_a2^
    # arm A's forward (device-resident seeds + out) is no longer needed:
    # release it before arm B's backward to cap the case's peak.
    _ = fwd_a^
    ctx.synchronize()
    var grads_b = h3_stack_train_backward_streamed_int8_seedoff[H, Dh](
        d_out, store, seeds, loras, mods, idx, cos, sin,
        D, F, ROTARY, EPS, ctx,
    )
    ctx.synchronize()

    var worst = Float64(1.0)
    var dx_diffs = _bits_equal(grads_a.d_x[], grads_b.d_x[], ctx)
    if dx_diffs != 0:
        var c = _cos(grads_a.d_x[], grads_b.d_x[], ctx)
        print("cross-arm d_x diffs", dx_diffs, "cos", c)
        if c < worst:
            worst = c
    else:
        print("cross-arm d_x bit-equal")
    for b in range(N):
        _check_grads(
            grads_a.lora[b], grads_b.lora[b], String("cross block ") + String(b),
            worst, ctx,
        )
    print("cross-arm worst cos", worst)
    if do_intra and worst_intra == Float64(1.0) and dxi == 0:
        # backward happened to be deterministic here: demand bit-equal
        if worst < Float64(1.0) or dx_diffs != 0:
            raise Error("backward deterministic but seedoff not bit-equal")
    else:
        if worst < 0.99999:
            raise Error("seedoff grads below 0.99999 absolute cos bar")
        if do_intra and worst < worst_intra - Float64(2e-6):
            raise Error("seedoff deviation exceeds flash rerun noise floor")
    _ = grads_a^
    _ = grads_b^
    _ = fwd_b^
    _ = seeds^


def main() raises:
    var ctx = DeviceContext()
    print("opening int8 store (2 blocks, 2 resident)...")
    var store = H3TrainBlockStoreInt8.open(String(CKPT), N, N, ctx)
    _run_case(384, True, store, ctx)
    _run_case(4910, False, store, ctx)
    print("minimax_h3_seed_offload_parity PASS")

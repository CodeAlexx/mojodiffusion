# minimax_h3_stack_train_smoke — REAL-DEPTH finite smoke: all 50 blocks,
# real FL2VA weights staged per block from the checkpoint mmap (v2 store:
# ~770MB pinned staging + fixed device slab; no bulk pinned fill), fwd +
# recompute bwd with one rank-16 LoRA set per block. Gate: every output
# finite, LoRA grads finite + non-zero, VRAM stays inside the 24GB card,
# and the wall times print (the step-time baseline for the trainer).
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor
from serenitymojo.ops.random import randn
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.models.klein.lora_block import LoraAdapterDevice
from serenitymojo.models.minimax_h3.h3_block_train import H3BlockLoraDevice
from serenitymojo.models.minimax_h3.h3_train_block_store import H3TrainBlockStore
from serenitymojo.models.minimax_h3.h3_stack_train import (
    h3_stack_train_forward_streamed, h3_stack_train_backward_streamed,
)

comptime CKPT = "/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/transformer"
comptime N_BLOCKS = 50
comptime S = 384
comptime H = 56
comptime Dh = 128
comptime D = 5376
comptime F = 14336
comptime INNER = H * Dh
comptime ROTARY = 96
comptime MOD_ROWS = 9
comptime RANK = 16
comptime EPS = Float32(1.0e-5)


def _finite_std(t: Tensor, ctx: DeviceContext) raises -> Float64:
    var h = t.to_host(ctx)
    var s = Float64(0)
    var s2 = Float64(0)
    for i in range(len(h)):
        var v = Float64(h[i])
        if not (v == v) or v > 1e30 or v < -1e30:
            raise Error("non-finite value in tensor")
        s += v
        s2 += v * v
    var n = Float64(len(h))
    var m = s / n
    return (s2 / n - m * m) ** 0.5


def _adapter(out_f: Int, in_f: Int, seed: UInt64, ctx: DeviceContext) raises -> LoraAdapterDevice:
    var ash: List[Int] = [RANK, in_f]
    var bsh: List[Int] = [out_f, RANK]
    var a = randn(ash^, seed, STDtype.BF16, ctx)
    var b = randn(bsh^, seed + 1, STDtype.BF16, ctx)
    return LoraAdapterDevice(
        ArcPointer(a^), ArcPointer(b^), RANK, in_f, out_f, Float32(1.0),
    )


def main() raises:
    var ctx = DeviceContext()
    print("[smoke] opening 50-block store (metadata walk + ~770MB staging)...")
    var t0 = perf_counter_ns()
    var store = H3TrainBlockStore.open(String(CKPT), N_BLOCKS, ctx)
    var t1 = perf_counter_ns()
    print("[smoke] store open in", Float64(t1 - t0) / 1.0e9, "s")

    var loras = List[H3BlockLoraDevice]()
    var mods = List[ArcPointer[Tensor]]()
    for i in range(N_BLOCKS):
        var seed = UInt64(1000 + 10 * i)
        loras.append(H3BlockLoraDevice(
            Optional[LoraAdapterDevice](_adapter(3 * INNER, D, seed, ctx)),
            Optional[LoraAdapterDevice](_adapter(D, INNER, seed + 2, ctx)),
            Optional[LoraAdapterDevice](_adapter(2 * F, D, seed + 4, ctx)),
            Optional[LoraAdapterDevice](_adapter(D, F, seed + 6, ctx)),
        ))
        var msh: List[Int] = [MOD_ROWS, 6 * D]
        # small-amplitude mod tables (the real ones are O(1) post-SiLU)
        var m = randn(msh^, UInt64(77 + i), STDtype.BF16, ctx)
        mods.append(ArcPointer(m^))

    var xsh: List[Int] = [S, D]
    var x = randn(xsh^, UInt64(42), STDtype.BF16, ctx)
    var csh: List[Int] = [S, ROTARY]
    var cos = randn(csh.copy(), UInt64(43), STDtype.F32, ctx)
    var sin = randn(csh.copy(), UInt64(44), STDtype.F32, ctx)
    var idx = List[Int]()
    for i in range(S):
        idx.append(i % MOD_ROWS)

    print("[smoke] forward (50 blocks, streamed)...")
    var t2 = perf_counter_ns()
    var fwd = h3_stack_train_forward_streamed[H, Dh](
        x, store, loras, mods, idx, cos, sin, D, F, ROTARY, EPS, ctx,
    )
    ctx.synchronize()
    var t3 = perf_counter_ns()
    var out_std = _finite_std(fwd.out[], ctx)
    print("[smoke] forward done", Float64(t3 - t2) / 1.0e9, "s; out finite, std", out_std)

    var dsh: List[Int] = [S, D]
    var d_out = randn(dsh^, UInt64(45), STDtype.BF16, ctx)
    print("[smoke] recompute backward (50 blocks, streamed)...")
    var t4 = perf_counter_ns()
    var grads = h3_stack_train_backward_streamed[H, Dh](
        d_out, fwd, store, loras, mods, idx, cos, sin,
        D, F, ROTARY, EPS, ctx,
    )
    ctx.synchronize()
    var t5 = perf_counter_ns()
    var dx_std = _finite_std(grads.d_x[], ctx)
    print("[smoke] backward done", Float64(t5 - t4) / 1.0e9, "s; d_x finite, std", dx_std)

    # spot-check LoRA grads on blocks 0 / 25 / 49: finite and non-zero
    var checks: List[Int] = [0, 25, 49]
    for c in range(len(checks)):
        var bi = checks[c]
        var g = grads.lora[bi].qkv.value().copy()
        var sa = _finite_std(g.d_a[], ctx)
        var sb2 = _finite_std(g.d_b[], ctx)
        if sa == 0.0 or sb2 == 0.0:
            raise Error("zero LoRA grad at block " + String(bi))
        print("[smoke] block", bi, "d_lora_qkv a/b std:", sa, sb2)

    print("PASS: 50-block real-weight streamed fwd+bwd finite; times above")

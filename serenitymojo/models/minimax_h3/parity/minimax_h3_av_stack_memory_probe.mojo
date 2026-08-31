# minimax_h3_av_stack_memory_probe — SKEPTIC PROBE (not a gate). Localizes the
# per-block VRAM ratchet seen in the AV smoke (S~4869: ~700MB/s climb through
# the stack pass to OOM despite per-block fences; the image smoke shows the
# same ratchet at ~1/10 scale, it just fits).
#
# Runs the seedoff int8 stack FORWARD alone at the AV geometry over the real
# 50-block store (16 resident), printing a MARK every 5 blocks; memory is
# sampled externally via nvidia-smi. A second pass runs the BACKWARD the same
# way. The per-MARK deltas attribute the ratchet to forward vs backward and
# measure its per-block rate.
from max.gpu.host import DeviceContext
from std.memory import ArcPointer
from std.time import sleep

from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor
from serenitymojo.ops.random import randn
from serenitymojo.models.minimax_h3.h3_block_train import (
    H3BlockLoraDevice, H3BlockLoraGrads,
    h3_block_train_forward_lora, h3_block_train_backward_lora_frozen,
)
from serenitymojo.models.klein.lora_block import LoraAdapterDevice
from serenitymojo.models.minimax_h3.h3_train_block_store_int8 import (
    H3TrainBlockStoreInt8,
)
from serenitymojo.models.minimax_h3.h3_stack_train import H3SeedStore

comptime CKPT = "/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/transformer"
comptime N = 50
comptime RESIDENT = 16
comptime S = 4869
comptime H = 56
comptime Dh = 128
comptime D = 5376
comptime F = 14336
comptime ROTARY = 96
comptime EPS = Float32(1.0e-5)
comptime RANK = 32

comptime TArc = ArcPointer[Tensor]


def _adapter(d_in: Int, d_out: Int, seed: UInt64, ctx: DeviceContext) raises -> LoraAdapterDevice:
    var ash: List[Int] = [RANK, d_in]
    var bsh: List[Int] = [d_out, RANK]
    var a = randn(ash^, seed, STDtype.BF16, ctx)
    var b = randn(bsh^, seed + 1, STDtype.BF16, ctx)
    return LoraAdapterDevice(
        ArcPointer(a^), ArcPointer(b^), RANK, d_in, d_out, Float32(1.0)
    )


def main() raises:
    var ctx = DeviceContext()
    print("MARK open_store")
    var store = H3TrainBlockStoreInt8.open(String(CKPT), N, RESIDENT, ctx)
    ctx.synchronize()
    sleep(1.0)
    print("MARK store_resident")

    comptime INNER = H * Dh
    var loras = List[H3BlockLoraDevice]()
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
        var msh: List[Int] = [9, 6 * D]
        mods.append(TArc(randn(msh^, UInt64(31 + b), STDtype.BF16, ctx)))
    var idx = List[Int]()
    for i in range(S):
        idx.append(i % 9)
    var xsh: List[Int] = [S, D]
    var x = randn(xsh^, UInt64(7), STDtype.BF16, ctx)
    var csh: List[Int] = [S, ROTARY]
    var cos = randn(csh.copy(), UInt64(13), STDtype.F32, ctx)
    var sin = randn(csh^, UInt64(17), STDtype.F32, ctx)
    var seeds = H3SeedStore.create(N, S, D, ctx)
    ctx.synchronize()
    sleep(1.0)
    print("MARK inputs_ready")

    # ── forward, per-block fence, MARK every 5 blocks ───────────────────────
    var h = x.clone(ctx)
    for i in range(N):
        seeds.save(i, h, ctx)
        var bw = store.stage(i, ctx)
        var p8 = store.payload(i)
        var f = h3_block_train_forward_lora[H, Dh](
            h, bw, loras[i], mods[i][], idx, cos, sin,
            D, F, ROTARY, EPS, ctx, p8,
        )
        h = f.out[].clone(ctx)
        _ = f^
        _ = bw^
        ctx.synchronize()
        if i % 5 == 4:
            sleep(0.6)
            print("MARK fwd_block", i)
    sleep(1.0)
    print("MARK forward_done")

    # ── backward, reverse, MARK every 5 blocks ──────────────────────────────
    var dsh: List[Int] = [S, D]
    var d = randn(dsh^, UInt64(19), STDtype.BF16, ctx)
    for r in range(N):
        var i = N - 1 - r
        var seed = seeds.load(i, ctx)
        var bw = store.stage(i, ctx)
        var p8 = store.payload(i)
        var f = h3_block_train_forward_lora[H, Dh](
            seed, bw, loras[i], mods[i][], idx, cos, sin,
            D, F, ROTARY, EPS, ctx, p8,
        )
        var b = h3_block_train_backward_lora_frozen[H, Dh](
            d, bw, loras[i], f.saved, idx, cos, sin,
            D, F, ROTARY, EPS, ctx, p8,
        )
        d = b.d_x[].clone(ctx)
        _ = b^
        _ = f^
        _ = bw^
        _ = seed^
        ctx.synchronize()
        if i % 5 == 0:
            sleep(0.6)
            print("MARK bwd_block", i)
    sleep(1.0)
    print("MARK backward_done")

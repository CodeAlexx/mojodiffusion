# minimax_h3_av_activation_memory_probe — SKEPTIC PROBE (not a gate).
#
# Measures the REAL device-memory footprint of ONE H3 training block's
# fwd(+saved set)+bwd at the packed sequence lengths the AV lift implies,
# on real block-0 FL2VA weights with LoRA r32 on qkv/out/fc1/fc2 — the
# handoff's "dominant unmeasured risk" (docs/H3_TRAINER_HANDOFF_2026-08-26.md).
#
# S values come from the CPU layout probe (measured 2026-08-26):
#   1298  = image mode today          (480x832, 1 latent frame)
#   4910  = 256x448, 124 px frames    (37 latent frames + 207 audio latents)
#   9902  = 384x640, 124 px frames
#   15752 = 480x832, 124 px frames    (the smallest legal clip at film res)
#
# Memory is sampled externally via nvidia-smi around this process; this file
# prints MARK lines and sleeps so the poller catches steady states. Same
# pattern as minimax_h3_real_block_memory_probe.mojo.
from max.gpu.host import DeviceContext
from std.memory import ArcPointer
from std.time import sleep

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.tensor import Tensor
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.random import randn
from serenitymojo.models.minimax_h3.h3_block_train import (
    H3BlockTrainWeights, H3BlockLoraDevice,
    h3_block_train_forward_lora, h3_block_train_backward_lora_frozen,
)
from serenitymojo.models.minimax_h3.h3_qkv_layout import (
    h3_qkv_deinterleave_rows,
)
from serenitymojo.models.klein.lora_block import LoraAdapterDevice

comptime CKPT = "/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/transformer"
comptime H = 56
comptime Dh = 128
comptime D = 5376
comptime F = 14336
comptime ROTARY = 96
comptime EPS = Float32(1.0e-5)
comptime RANK = 32


def _adapter(rank: Int, d_in: Int, d_out: Int, seed: UInt64, ctx: DeviceContext) raises -> LoraAdapterDevice:
    var ash: List[Int] = [rank, d_in]
    var bsh: List[Int] = [d_out, rank]
    var a = randn(ash^, seed, STDtype.BF16, ctx)
    var b = randn(bsh^, seed + 1, STDtype.BF16, ctx)
    return LoraAdapterDevice(
        ArcPointer(a^), ArcPointer(b^), rank, d_in, d_out, Float32(1.0)
    )


def _run_case(
    s_len: Int,
    w: H3BlockTrainWeights,
    lora: H3BlockLoraDevice,
    ctx: DeviceContext,
) raises:
    var xsh: List[Int] = [s_len, D]
    var x = randn(xsh^, UInt64(7 + s_len), STDtype.BF16, ctx)
    var msh: List[Int] = [3, 6 * D]
    var mod = randn(msh^, UInt64(11), STDtype.BF16, ctx)
    var idx = List[Int]()
    for i in range(s_len):
        idx.append(i % 3)
    var csh: List[Int] = [s_len, ROTARY]
    var cos = randn(csh.copy(), UInt64(13), STDtype.F32, ctx)
    var sin = randn(csh^, UInt64(17), STDtype.F32, ctx)
    var dsh: List[Int] = [s_len, D]
    var d_out = randn(dsh^, UInt64(19 + s_len), STDtype.BF16, ctx)
    ctx.synchronize()
    sleep(1.5)
    print("MARK inputs_ready", s_len)

    var fwd = h3_block_train_forward_lora[H, Dh](
        x, w, lora, mod, idx, cos, sin, D, F, ROTARY, EPS, ctx,
    )
    ctx.synchronize()
    sleep(1.5)
    print("MARK after_fwd_saved_resident", s_len)

    # PRODUCTION path: frozen-base LoRA backward (flash-tape attention, no S^2
    # math scores, no frozen-weight grads) — what the streamed_int8 stack calls.
    var grads = h3_block_train_backward_lora_frozen[H, Dh](
        d_out, w, lora, fwd.saved, idx, cos, sin, D, F, ROTARY, EPS, ctx,
    )
    ctx.synchronize()
    sleep(1.5)
    print("MARK after_bwd_peak_settled", s_len)
    _ = grads^
    _ = fwd^


def main() raises:
    print("MARK startup")
    var ctx = DeviceContext()
    ctx.synchronize()
    sleep(1.5)
    print("MARK baseline_after_ctx")

    var sharded = ShardedSafeTensors.open(String(CKPT))
    var p = String("blocks.0.")
    var names: List[String] = [
        String("attn.qkv_proj.weight"), String("attn.out_proj.weight"),
        String("mlp.fc1.weight"), String("mlp.fc2.weight"),
        String("attn.q_norm.weight"), String("attn.k_norm.weight"),
        String("norm1.weight"), String("norm2.weight"),
    ]
    var wt = List[ArcPointer[Tensor]]()
    for i in range(len(names)):
        wt.append(ArcPointer(cast_tensor(
            Tensor.from_view(sharded.tensor_view(p + names[i]), ctx),
            STDtype.BF16, ctx,
        )))
    var qkv = h3_qkv_deinterleave_rows(wt[0][], H, Dh, ctx)
    var w = H3BlockTrainWeights(
        ArcPointer(qkv^), wt[1], wt[2], wt[3], wt[4], wt[5], wt[6], wt[7],
    )
    comptime INNER = H * Dh
    var lora = H3BlockLoraDevice(
        Optional[LoraAdapterDevice](_adapter(RANK, D, 3 * INNER, 101, ctx)),
        Optional[LoraAdapterDevice](_adapter(RANK, INNER, D, 103, ctx)),
        Optional[LoraAdapterDevice](_adapter(RANK, D, 2 * F, 107, ctx)),
        Optional[LoraAdapterDevice](_adapter(RANK, F, D, 109, ctx)),
    )
    ctx.synchronize()
    sleep(1.5)
    print("MARK weights_resident")

    var cases: List[Int] = [1298, 4910, 9902, 15752]
    for i in range(len(cases)):
        _run_case(cases[i], w, lora, ctx)
        ctx.synchronize()
        sleep(1.5)
        print("MARK case_dropped", cases[i])

    print("MARK done")

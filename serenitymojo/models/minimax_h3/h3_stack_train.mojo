# serenitymojo/models/minimax_h3/h3_stack_train.mojo
#
# H3 training STACK driver: N transformer blocks composed with the
# per-block RECOMPUTE pattern (24GB discipline): the forward keeps only
# each block's INPUT; the backward re-runs block i's forward with saved
# activations, then chains h3_block_train_backward_lora, handing d_x to
# block i-1 (the d_x -> d_y contract). Per-block modulation tables (each
# block owns its adaln_proj output) arrive precomputed — the modcache
# contract; adaln stays frozen.
#
# v1 API takes device-resident per-block weights (fine to the ~2-8 block
# gates). The 50-block real-depth arm plugs a pinned-host store + fixed
# device slab streamer into the same loop (stage_block hook) — blocks are
# ~770MB bf16 each, 38.5GB total, never device-resident during training.
# Gate: parity/minimax_h3_stack_train_parity.mojo vs the 2-block chain arm
# of parity/h3_block_oracle.py (real block-0/1 weights).
from max.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.models.minimax_h3.h3_block_train import (
    H3BlockTrainWeights, H3BlockLoraDevice, H3BlockTrainGrads,
    H3BlockLoraGrads, H3BlockTrainLoraBackward,
    h3_block_train_forward_lora, h3_block_train_backward_lora,
)

comptime TArc = ArcPointer[Tensor]


struct H3StackTrainForward(Copyable, Movable):
    var out: TArc                 # [S, D] stack output
    var block_inputs: List[TArc]  # length N: each block's input (recompute seeds)

    def __init__(out self, var out: TArc, var block_inputs: List[TArc]):
        self.out = out^
        self.block_inputs = block_inputs^


struct H3StackTrainGrads(Copyable, Movable):
    var d_x: TArc                       # grad into the stack input
    var base: List[H3BlockTrainGrads]   # per block (frozen-base grads + d_mod)
    var lora: List[H3BlockLoraGrads]    # per block adapter grads

    def __init__(
        out self,
        var d_x: TArc,
        var base: List[H3BlockTrainGrads],
        var lora: List[H3BlockLoraGrads],
    ):
        self.d_x = d_x^
        self.base = base^
        self.lora = lora^


def h3_stack_train_forward[
    H: Int, Dh: Int, S: Int
](
    x_in: Tensor,
    blocks: List[H3BlockTrainWeights],
    loras: List[H3BlockLoraDevice],
    mods: List[TArc],           # per-block modulation tables [rows, 6D]
    adaln_indices: List[Int],
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, rotary_dim: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> H3StackTrainForward:
    if len(blocks) != len(mods) or len(blocks) != len(loras):
        raise Error("h3_stack_train_forward: blocks/mods/loras length mismatch")
    var inputs = List[TArc]()
    var h = x_in.clone(ctx)
    for i in range(len(blocks)):
        inputs.append(TArc(h.clone(ctx)))
        # forward WITHOUT retaining the act set: the recompute backward
        # rebuilds it per block; transients free per iteration.
        var f = h3_block_train_forward_lora[H, Dh, S](
            h, blocks[i], loras[i], mods[i][], adaln_indices, cos, sin,
            D, F, rotary_dim, eps, ctx,
        )
        h = f.out[].clone(ctx)
        _ = f^
    return H3StackTrainForward(TArc(h^), inputs^)


def h3_stack_train_backward[
    H: Int, Dh: Int, S: Int
](
    d_out: Tensor,
    fwd: H3StackTrainForward,
    blocks: List[H3BlockTrainWeights],
    loras: List[H3BlockLoraDevice],
    mods: List[TArc],
    adaln_indices: List[Int],
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, rotary_dim: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> H3StackTrainGrads:
    var n = len(blocks)
    if len(fwd.block_inputs) != n:
        raise Error("h3_stack_train_backward: forward/blocks length mismatch")
    # collect in reverse then flip so results are block-order
    var base_rev = List[H3BlockTrainGrads]()
    var lora_rev = List[H3BlockLoraGrads]()
    var d = d_out.clone(ctx)
    for r in range(n):
        var i = n - 1 - r
        var mod_rows = mods[i][].shape()[0]
        # recompute block i's forward with saved activations
        var f = h3_block_train_forward_lora[H, Dh, S](
            fwd.block_inputs[i][], blocks[i], loras[i], mods[i][],
            adaln_indices, cos, sin, D, F, rotary_dim, eps, ctx,
        )
        var b = h3_block_train_backward_lora[H, Dh, S](
            d, blocks[i], loras[i], f.saved, adaln_indices, cos, sin,
            mod_rows, D, F, rotary_dim, eps, ctx,
        )
        d = b.base.d_x[].clone(ctx)
        base_rev.append(b.base.copy())
        lora_rev.append(b.lora.copy())
        _ = b^
        _ = f^
        # fence per block: transients (recomputed act set + backward
        # intermediates) only free at a sync — without this the arena
        # commits the cumulative peak (MJ-1142/MJ-1144 lesson).
        ctx.synchronize()
    var base = List[H3BlockTrainGrads]()
    var lora = List[H3BlockLoraGrads]()
    for r in range(n):
        base.append(base_rev[n - 1 - r].copy())
        lora.append(lora_rev[n - 1 - r].copy())
    return H3StackTrainGrads(TArc(d^), base^, lora^)

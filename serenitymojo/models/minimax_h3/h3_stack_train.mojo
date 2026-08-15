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
# gates). The 50-block real-depth arm plugs the mmap-staging store
# (H3TrainBlockStore: checkpoint mmap → ~770MB pinned staging → fixed
# device slab) into the same loop — blocks are ~770MB bf16 each, never
# bulk-resident on host OR device during training.
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


# ── STREAMED variants: weights staged per block from H3TrainBlockStore ──────
from serenitymojo.models.minimax_h3.h3_train_block_store import H3TrainBlockStore


def h3_stack_train_forward_streamed[
    H: Int, Dh: Int, S: Int
](
    x_in: Tensor,
    mut store: H3TrainBlockStore,
    loras: List[H3BlockLoraDevice],
    mods: List[TArc],
    adaln_indices: List[Int],
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, rotary_dim: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> H3StackTrainForward:
    var n = store.num_blocks
    if len(mods) != n or len(loras) != n:
        raise Error("h3_stack_train_forward_streamed: mods/loras length mismatch")
    var inputs = List[TArc]()
    var h = x_in.clone(ctx)
    for i in range(n):
        inputs.append(TArc(h.clone(ctx)))
        var bw = store.stage(i, ctx)
        var f = h3_block_train_forward_lora[H, Dh, S](
            h, bw, loras[i], mods[i][], adaln_indices, cos, sin,
            D, F, rotary_dim, eps, ctx,
        )
        h = f.out[].clone(ctx)
        _ = f^
        _ = bw^
        # fence: the NEXT stage() overwrites the slab; block i's kernels
        # must have consumed it. Also releases this block's transients.
        ctx.synchronize()
    return H3StackTrainForward(TArc(h^), inputs^)


def h3_stack_train_backward_streamed[
    H: Int, Dh: Int, S: Int
](
    d_out: Tensor,
    fwd: H3StackTrainForward,
    mut store: H3TrainBlockStore,
    loras: List[H3BlockLoraDevice],
    mods: List[TArc],
    adaln_indices: List[Int],
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, rotary_dim: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> H3StackLoraOnlyGrads:
    """Streamed recompute backward. Returns LoRA grads + d_mod per block
    (the trainable set); frozen-base weight grads are NOT retained — at 50
    blocks they would be another 38.5GB."""
    var n = store.num_blocks
    if len(fwd.block_inputs) != n:
        raise Error("h3_stack_train_backward_streamed: forward length mismatch")
    var lora_rev = List[H3BlockLoraGrads]()
    var dmod_rev = List[TArc]()
    var d = d_out.clone(ctx)
    for r in range(n):
        var i = n - 1 - r
        var mod_rows = mods[i][].shape()[0]
        var bw = store.stage(i, ctx)
        var f = h3_block_train_forward_lora[H, Dh, S](
            fwd.block_inputs[i][], bw, loras[i], mods[i][],
            adaln_indices, cos, sin, D, F, rotary_dim, eps, ctx,
        )
        var b = h3_block_train_backward_lora[H, Dh, S](
            d, bw, loras[i], f.saved, adaln_indices, cos, sin,
            mod_rows, D, F, rotary_dim, eps, ctx,
        )
        d = b.base.d_x[].clone(ctx)
        dmod_rev.append(b.base.d_mod)
        lora_rev.append(b.lora.copy())
        _ = b^
        _ = f^
        _ = bw^
        ctx.synchronize()
    var lora = List[H3BlockLoraGrads]()
    var dmod = List[TArc]()
    for r in range(n):
        lora.append(lora_rev[n - 1 - r].copy())
        dmod.append(dmod_rev[n - 1 - r])
    return H3StackLoraOnlyGrads(TArc(d^), lora^, dmod^)


struct H3StackLoraOnlyGrads(Copyable, Movable):
    var d_x: TArc
    var lora: List[H3BlockLoraGrads]
    var d_mod: List[TArc]

    def __init__(
        out self,
        var d_x: TArc,
        var lora: List[H3BlockLoraGrads],
        var d_mod: List[TArc],
    ):
        self.d_x = d_x^
        self.lora = lora^
        self.d_mod = d_mod^

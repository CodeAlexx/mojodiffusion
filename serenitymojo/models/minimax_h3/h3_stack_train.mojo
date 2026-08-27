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
from serenitymojo.models.klein.lora_block import KleinLoraDeviceGradTensors
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
    H: Int, Dh: Int
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
        var f = h3_block_train_forward_lora[H, Dh](
            h, blocks[i], loras[i], mods[i][], adaln_indices, cos, sin,
            D, F, rotary_dim, eps, ctx,
        )
        h = f.out[].clone(ctx)
        _ = f^
    return H3StackTrainForward(TArc(h^), inputs^)


def h3_stack_train_backward[
    H: Int, Dh: Int
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
        var f = h3_block_train_forward_lora[H, Dh](
            fwd.block_inputs[i][], blocks[i], loras[i], mods[i][],
            adaln_indices, cos, sin, D, F, rotary_dim, eps, ctx,
        )
        var b = h3_block_train_backward_lora[H, Dh](
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
    H: Int, Dh: Int
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
        var f = h3_block_train_forward_lora[H, Dh](
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
    H: Int, Dh: Int
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
    """Streamed recompute backward. Returns LoRA grads per block (the
    trainable set). Frozen-base weight grads are NOT retained (38.5GB at
    50 blocks) and neither is d_mod (adaln is frozen; with grid-sized mod
    tables retaining it would be another ~9.3GB)."""
    var n = store.num_blocks
    if len(fwd.block_inputs) != n:
        raise Error("h3_stack_train_backward_streamed: forward length mismatch")
    var lora_rev = List[H3BlockLoraGrads]()
    var d = d_out.clone(ctx)
    for r in range(n):
        var i = n - 1 - r
        var mod_rows = mods[i][].shape()[0]
        var bw = store.stage(i, ctx)
        var f = h3_block_train_forward_lora[H, Dh](
            fwd.block_inputs[i][], bw, loras[i], mods[i][],
            adaln_indices, cos, sin, D, F, rotary_dim, eps, ctx,
        )
        var b = h3_block_train_backward_lora[H, Dh](
            d, bw, loras[i], f.saved, adaln_indices, cos, sin,
            mod_rows, D, F, rotary_dim, eps, ctx,
        )
        d = b.base.d_x[].clone(ctx)
        lora_rev.append(b.lora.copy())
        _ = b^
        _ = f^
        _ = bw^
        ctx.synchronize()
    var lora = List[H3BlockLoraGrads]()
    for r in range(n):
        lora.append(lora_rev[n - 1 - r].copy())
    return H3StackLoraOnlyGrads(TArc(d^), lora^)


struct H3StackLoraOnlyGrads(Copyable, Movable):
    var d_x: TArc
    var lora: List[H3BlockLoraGrads]

    def __init__(
        out self,
        var d_x: TArc,
        var lora: List[H3BlockLoraGrads],
    ):
        self.d_x = d_x^
        self.lora = lora^


# ── FP8-RESIDENT variants: weights dequantized per block on-device ──────────
from serenitymojo.models.minimax_h3.h3_train_block_store_fp8 import (
    H3TrainBlockStoreFp8,
)
from serenitymojo.models.minimax_h3.h3_train_fence_policy import (
    h3_fp8_forward_should_fence,
    h3_fp8_backward_should_fence,
)
from serenitymojo.models.minimax_h3.h3_block_train import (
    h3_block_train_backward_lora_frozen,
)


def h3_stack_train_forward_streamed_fp8[
    H: Int, Dh: Int
](
    x_in: Tensor,
    mut store: H3TrainBlockStoreFp8,
    loras: List[H3BlockLoraDevice],
    mods: List[TArc],
    adaln_indices: List[Int],
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, rotary_dim: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> H3StackTrainForward:
    var n = store.num_blocks
    if len(mods) != n or len(loras) != n:
        raise Error("h3_stack_train_forward_streamed_fp8: mods/loras length mismatch")
    var inputs = List[TArc]()
    var h = x_in.clone(ctx)
    for i in range(n):
        inputs.append(TArc(h.clone(ctx)))
        var bw = store.stage(i, ctx)
        var f = h3_block_train_forward_lora[H, Dh](
            h, bw, loras[i], mods[i][], adaln_indices, cos, sin,
            D, F, rotary_dim, eps, ctx,
        )
        h = f.out[].clone(ctx)
        _ = f^
        _ = bw^
        # fence policy: tail blocks MUST fence (mmap slab reuse); resident
        # blocks fence every 8th to bound the async transient peak without
        # paying 50 pipeline drains per pass.
        if h3_fp8_forward_should_fence(i, store.resident):
            ctx.synchronize()
    return H3StackTrainForward(TArc(h^), inputs^)


def h3_stack_train_backward_streamed_fp8[
    H: Int, Dh: Int
](
    d_out: Tensor,
    fwd: H3StackTrainForward,
    mut store: H3TrainBlockStoreFp8,
    loras: List[H3BlockLoraDevice],
    mods: List[TArc],
    adaln_indices: List[Int],
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, rotary_dim: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> H3StackLoraOnlyGrads:
    """FP8-resident recompute backward: LoRA grads only (adaln frozen,
    frozen-base grads dropped) — same discipline as the bf16 streamed arm."""
    var n = store.num_blocks
    if len(fwd.block_inputs) != n:
        raise Error("h3_stack_train_backward_streamed_fp8: forward length mismatch")
    var lora_rev = List[H3BlockLoraGrads]()
    var d = d_out.clone(ctx)
    for r in range(n):
        var i = n - 1 - r
        var bw = store.stage(i, ctx)
        var f = h3_block_train_forward_lora[H, Dh](
            fwd.block_inputs[i][], bw, loras[i], mods[i][],
            adaln_indices, cos, sin, D, F, rotary_dim, eps, ctx,
        )
        var b = h3_block_train_backward_lora_frozen[H, Dh](
            d, bw, loras[i], f.saved, adaln_indices, cos, sin,
            D, F, rotary_dim, eps, ctx,
        )
        d = b.d_x[].clone(ctx)
        lora_rev.append(b.lora.copy())
        _ = b^
        _ = f^
        _ = bw^
        # Reverse traversal visits the streamed tail first. Every tail block
        # MUST fence before stage(i-1) overwrites the one shared mmap slab.
        # Resident blocks own independent quantized storage, so fence only
        # every eighth block to bound allocator transients without draining
        # the GPU pipeline fifty times. The old `i <= store.resident` test was
        # reversed: it skipped most required tail fences and serialized nearly
        # every resident block, harming both correctness and speed.
        if h3_fp8_backward_should_fence(i, store.resident):
            ctx.synchronize()
    var lora = List[H3BlockLoraGrads]()
    for r in range(n):
        lora.append(lora_rev[n - 1 - r].copy())
    return H3StackLoraOnlyGrads(TArc(d^), lora^)


# ── DIRECT INT8-RESIDENT variants: no resident weight dequantization ────────
from serenitymojo.models.minimax_h3.h3_train_block_store_int8 import (
    H3TrainBlockStoreInt8,
)


def h3_stack_train_forward_streamed_int8[
    H: Int, Dh: Int
](
    x_in: Tensor,
    mut store: H3TrainBlockStoreInt8,
    loras: List[H3BlockLoraDevice],
    mods: List[TArc],
    adaln_indices: List[Int],
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, rotary_dim: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> H3StackTrainForward:
    var n = store.num_blocks
    if len(mods) != n or len(loras) != n:
        raise Error("h3 int8 stack forward: mods/loras length mismatch")
    var inputs = List[TArc]()
    var h = x_in.clone(ctx)
    for i in range(n):
        inputs.append(TArc(h.clone(ctx)))
        var bw = store.stage(i, ctx)
        var p8 = store.payload(i)
        var f = h3_block_train_forward_lora[H, Dh](
            h, bw, loras[i], mods[i][], adaln_indices, cos, sin,
            D, F, rotary_dim, eps, ctx, p8,
        )
        h = f.out[].clone(ctx)
        _ = f^
        _ = bw^
        if h3_fp8_forward_should_fence(i, store.resident):
            ctx.synchronize()
    return H3StackTrainForward(TArc(h^), inputs^)


def h3_stack_train_backward_streamed_int8[
    H: Int, Dh: Int
](
    d_out: Tensor,
    fwd: H3StackTrainForward,
    mut store: H3TrainBlockStoreInt8,
    loras: List[H3BlockLoraDevice],
    mods: List[TArc],
    adaln_indices: List[Int],
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, rotary_dim: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> H3StackLoraOnlyGrads:
    var n = store.num_blocks
    if len(fwd.block_inputs) != n:
        raise Error("h3 int8 stack backward: forward length mismatch")
    var lora_rev = List[H3BlockLoraGrads]()
    var d = d_out.clone(ctx)
    for r in range(n):
        var i = n - 1 - r
        var bw = store.stage(i, ctx)
        var p8 = store.payload(i)
        var f = h3_block_train_forward_lora[H, Dh](
            fwd.block_inputs[i][], bw, loras[i], mods[i][],
            adaln_indices, cos, sin, D, F, rotary_dim, eps, ctx, p8,
        )
        var b = h3_block_train_backward_lora_frozen[H, Dh](
            d, bw, loras[i], f.saved, adaln_indices, cos, sin,
            D, F, rotary_dim, eps, ctx, p8,
        )
        d = b.d_x[].clone(ctx)
        lora_rev.append(b.lora.copy())
        _ = b^
        _ = f^
        _ = bw^
        if h3_fp8_backward_should_fence(i, store.resident):
            ctx.synchronize()
    var lora = List[H3BlockLoraGrads]()
    for r in range(n):
        lora.append(lora_rev[n - 1 - r].copy())
    return H3StackLoraOnlyGrads(TArc(d^), lora^)


# ── SEED-OFFLOAD variants: recompute seeds live in pinned host memory ───────
# Measured need (docs/H3_AV_ACTIVATION_FOOTPRINT_2026-08-26, trainerdocs): the
# device-resident `block_inputs` list is 50 x S x D x 2B — 0.67GB in image mode
# but 2.6GB at the AV bring-up geometry (S=4910) and 8.5GB at film res, against
# a 16GB card that also holds the 9.7GB resident base. Seeds are written once
# (forward) and read once (backward, reverse order): classic offload shape.
#
# Design: ONE pinned host slab (n x seed_bytes) + a 2-slot device scratch ring
# for backward staging. All copies are `enqueue_copy` on the ctx stream, so
# they order with the block kernels that produce/consume them — no host-side
# memcpy exists here, hence NO extra fences beyond the arm's existing policy
# (unlike H3TrainBlockStore.stage, whose sys_memcpy into pinned staging is
# host-side and forces the per-block fence).
from max.gpu.host import HostBuffer, DeviceBuffer
from serenitymojo.io.dtype import STDtype


struct H3SeedStore(Movable):
    """Pinned-host store for the stack's recompute seeds ([S, D] bf16 each)."""
    var slab: HostBuffer[DType.uint8]   # pinned, n * seed_bytes
    var dev: DeviceBuffer[DType.uint8]  # 2-slot staging ring for backward
    var n: Int
    var s_len: Int
    var d_model: Int
    var seed_bytes: Int
    var slot: Int

    def __init__(
        out self,
        var slab: HostBuffer[DType.uint8],
        var dev: DeviceBuffer[DType.uint8],
        n: Int, s_len: Int, d_model: Int, seed_bytes: Int,
    ):
        self.slab = slab^
        self.dev = dev^
        self.n = n
        self.s_len = s_len
        self.d_model = d_model
        self.seed_bytes = seed_bytes
        self.slot = 0

    @staticmethod
    def create(
        n: Int, s_len: Int, d_model: Int, ctx: DeviceContext
    ) raises -> H3SeedStore:
        var seed_bytes = s_len * d_model * 2  # bf16
        var slab = ctx.enqueue_create_host_buffer[DType.uint8](n * seed_bytes)
        var dev = ctx.enqueue_create_buffer[DType.uint8](2 * seed_bytes)
        ctx.synchronize()
        return H3SeedStore(slab^, dev^, n, s_len, d_model, seed_bytes)

    def save(mut self, i: Int, t: Tensor, ctx: DeviceContext) raises:
        """Async D2H of block i's input into the pinned slab (stream-ordered
        after the kernels that produced `t`; `t` must be [s_len, d_model] bf16)."""
        if i < 0 or i >= self.n:
            raise Error("H3SeedStore.save: index out of range")
        if t.nbytes() != self.seed_bytes:
            raise Error("H3SeedStore.save: seed byte-size mismatch")
        var hsub = self.slab.create_sub_buffer[DType.uint8](
            i * self.seed_bytes, self.seed_bytes
        )
        ctx.enqueue_copy(dst_buf=hsub, src_buf=t.buf)

    def load(mut self, i: Int, ctx: DeviceContext) raises -> Tensor:
        """Async H2D of block i's seed into the next scratch slot; returns a
        [s_len, d_model] bf16 view. Slot reuse is safe by single-stream
        ordering: the copy is enqueued after every kernel that read the
        previous tenant of the slot."""
        if i < 0 or i >= self.n:
            raise Error("H3SeedStore.load: index out of range")
        var hsub = self.slab.create_sub_buffer[DType.uint8](
            i * self.seed_bytes, self.seed_bytes
        )
        var dsub = self.dev.create_sub_buffer[DType.uint8](
            self.slot * self.seed_bytes, self.seed_bytes
        )
        ctx.enqueue_copy(dst_buf=dsub, src_buf=hsub)
        self.slot = 1 - self.slot
        var sh: List[Int] = [self.s_len, self.d_model]
        return Tensor(dsub^, sh^, STDtype.BF16)


struct H3StackTrainForwardOff(Movable):
    """Seed-offload forward result: stack output only — the recompute seeds
    live in the H3SeedStore, not on device."""
    var out: TArc

    def __init__(out self, var out: TArc):
        self.out = out^


def h3_stack_train_forward_streamed_int8_seedoff[
    H: Int, Dh: Int
](
    x_in: Tensor,
    mut store: H3TrainBlockStoreInt8,
    mut seeds: H3SeedStore,
    loras: List[H3BlockLoraDevice],
    mods: List[TArc],
    adaln_indices: List[Int],
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, rotary_dim: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> H3StackTrainForwardOff:
    """`h3_stack_train_forward_streamed_int8` with the per-block input clone
    replaced by an async D2H into the pinned seed slab."""
    var n = store.num_blocks
    if len(mods) != n or len(loras) != n:
        raise Error("h3 int8 seedoff forward: mods/loras length mismatch")
    if seeds.n != n:
        raise Error("h3 int8 seedoff forward: seed store block count mismatch")
    # AV-scale sequences: per-block transients are large enough that the
    # every-8th resident fence lets several coexist and OOMs a 16GB card —
    # fence every block once S crosses the threshold (image S stays fast).
    var fence_all = x_in.shape()[0] >= 2048
    var h = x_in.clone(ctx)
    for i in range(n):
        seeds.save(i, h, ctx)
        var bw = store.stage(i, ctx)
        var p8 = store.payload(i)
        var f = h3_block_train_forward_lora[H, Dh](
            h, bw, loras[i], mods[i][], adaln_indices, cos, sin,
            D, F, rotary_dim, eps, ctx, p8,
        )
        h = f.out[].clone(ctx)
        _ = f^
        _ = bw^
        if fence_all or h3_fp8_forward_should_fence(i, store.resident):
            ctx.synchronize()
    return H3StackTrainForwardOff(TArc(h^))


def _h3_prune_adapter_grads(
    g: H3BlockLoraGrads, dummy: TArc
) raises -> H3BlockLoraGrads:
    """Keep d_a/d_b (what the optimizer consumes), replace the retained
    per-adapter d_x — [S,D]/[S,F] tensors already consumed inside the block
    backward — with a shared [1,1] dummy. 184MB/block at the AV geometry."""
    var out = List[Optional[KleinLoraDeviceGradTensors]]()
    var slots = List[Optional[KleinLoraDeviceGradTensors]]()
    slots.append(g.qkv.copy())
    slots.append(g.out.copy())
    slots.append(g.fc1.copy())
    slots.append(g.fc2.copy())
    for i in range(4):
        if slots[i]:
            var t = slots[i].value().copy()
            out.append(Optional[KleinLoraDeviceGradTensors](
                KleinLoraDeviceGradTensors(t.d_a.copy(), t.d_b.copy(), dummy.copy())
            ))
        else:
            out.append(Optional[KleinLoraDeviceGradTensors](None))
    return H3BlockLoraGrads(
        out[0].copy(), out[1].copy(), out[2].copy(), out[3].copy()
    )


def h3_stack_train_backward_streamed_int8_seedoff[
    H: Int, Dh: Int
](
    d_out: Tensor,
    mut store: H3TrainBlockStoreInt8,
    mut seeds: H3SeedStore,
    loras: List[H3BlockLoraDevice],
    mods: List[TArc],
    adaln_indices: List[Int],
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, rotary_dim: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> H3StackLoraOnlyGrads:
    """`h3_stack_train_backward_streamed_int8` reading recompute seeds from the
    pinned slab (2-slot device ring) instead of device-resident block_inputs."""
    var n = store.num_blocks
    if seeds.n != n:
        raise Error("h3 int8 seedoff backward: seed store block count mismatch")
    var fence_all = d_out.shape()[0] >= 2048
    var lora_rev = List[H3BlockLoraGrads]()
    var dummy_sh: List[Int] = [1, 1]
    var dummy = TArc(Tensor(
        ctx.enqueue_create_buffer[DType.uint8](2), dummy_sh^, STDtype.BF16
    ))
    var d = d_out.clone(ctx)
    for r in range(n):
        var i = n - 1 - r
        var seed = seeds.load(i, ctx)
        var bw = store.stage(i, ctx)
        var p8 = store.payload(i)
        var f = h3_block_train_forward_lora[H, Dh](
            seed, bw, loras[i], mods[i][],
            adaln_indices, cos, sin, D, F, rotary_dim, eps, ctx, p8,
        )
        var b = h3_block_train_backward_lora_frozen[H, Dh](
            d, bw, loras[i], f.saved, adaln_indices, cos, sin,
            D, F, rotary_dim, eps, ctx, p8,
        )
        d = b.d_x[].clone(ctx)
        lora_rev.append(_h3_prune_adapter_grads(b.lora, dummy))
        _ = b^
        _ = f^
        _ = bw^
        _ = seed^
        if fence_all or h3_fp8_backward_should_fence(i, store.resident):
            ctx.synchronize()
    var lora = List[H3BlockLoraGrads]()
    for r in range(n):
        lora.append(lora_rev[n - 1 - r].copy())
    return H3StackLoraOnlyGrads(TArc(d^), lora^)

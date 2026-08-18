# serenitymojo/models/minimax_h3/h3_train_block_store.mojo
#
# mmap-staging streamer for H3 TRAINING block weights. The mmap and fixed slab
# retain the released checkpoint bytes verbatim; stage() deinterleaves the QKV
# rows on device before returning the block so training consumes the same
# [q_all;k_all;v_all] base function as product inference. FC1 remains raw
# [gate;value], which is the training block's native SwiGLU convention.
#
# v2 DESIGN (the 08-14 oomd lesson): v1 filled a single 38.5GB pinned host
# store up front; pinned pages are UNRECLAIMABLE, so once filled the kernel
# had nothing left to evict on a 62GB box — user-slice pressure hit 77% and
# oomd killed the desktop session. v2 keeps the ShardedSafeTensors handle
# OPEN instead and stages each block on demand:
#
#   checkpoint mmap --(memcpy, WILLNEED-prefetched)--> ONE pinned staging
#   buffer (~770MB, largest block) --(one async H2D)--> fixed device slab
#   (sub-buffer views — zero arena churn, the MJ-1142 discipline).
#
# Total pinned: one block, not fifty. Source bytes live in page cache,
# which is charge-visible but RECLAIMABLE — under pressure the kernel
# evicts it cleanly instead of oomd killing; warm steps re-stage at cache
# memcpy speed. First pass runs the disk at sequential speed via the
# per-tensor WILLNEED window (the qwen3 loader pattern, 30728ae).
#
# Contract (unchanged from v1): stage(layer) returns sub-buffer views into
# the slab; the caller's per-block ctx.synchronize() fence must have retired
# the previous block's kernels AND its staging→slab copy before the next
# stage() overwrites either buffer. h3_stack_train.mojo's streamed loops
# fence per block, satisfying both.
from max.gpu.host import DeviceContext, HostBuffer, DeviceBuffer
from std.memory import ArcPointer

from serenitymojo.io.ffi import sys_memcpy, BytePtr
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.tensor import Tensor
from serenitymojo.models.minimax_h3.h3_block_train import H3BlockTrainWeights
from serenitymojo.models.minimax_h3.h3_qkv_layout import (
    h3_qkv_deinterleave_rows,
)

comptime TArc = ArcPointer[Tensor]
comptime H3_TRAIN_BLOCK_TENSORS = 8
comptime H3_TRAIN_HEADS = 56
comptime H3_TRAIN_HEAD_DIM = 128


def h3_train_block_tensor_names() -> List[String]:
    """RAW checkpoint names, the h3_block_train weight order."""
    var names = List[String]()
    names.append(String("attn.qkv_proj.weight"))
    names.append(String("attn.out_proj.weight"))
    names.append(String("mlp.fc1.weight"))
    names.append(String("mlp.fc2.weight"))
    names.append(String("attn.q_norm.weight"))
    names.append(String("attn.k_norm.weight"))
    names.append(String("norm1.weight"))
    names.append(String("norm2.weight"))
    return names^


struct H3TrainBlockStore(Movable):
    var st: ShardedSafeTensors            # OPEN for the store's lifetime
    var staging: HostBuffer[DType.uint8]  # pinned, ONE block (~770MB)
    var slab: DeviceBuffer[DType.uint8]
    var num_blocks: Int
    var names: List[String]
    # per (block, tensor): byte count, rel offset in staging/slab, shape, dtype
    var nbytes: List[Int]
    var rel: List[Int]
    var shapes: List[ArcPointer[List[Int]]]
    var dtypes: List[STDtype]
    var staged_layer: Int

    def __init__(
        out self,
        var st: ShardedSafeTensors,
        var staging: HostBuffer[DType.uint8],
        var slab: DeviceBuffer[DType.uint8],
        num_blocks: Int,
        var names: List[String],
        var nbytes: List[Int], var rel: List[Int],
        var shapes: List[ArcPointer[List[Int]]], var dtypes: List[STDtype],
    ):
        self.st = st^
        self.staging = staging^
        self.slab = slab^
        self.num_blocks = num_blocks
        self.names = names^
        self.nbytes = nbytes^
        self.rel = rel^
        self.shapes = shapes^
        self.dtypes = dtypes^
        self.staged_layer = -1

    @staticmethod
    def open(
        dir: String, num_blocks: Int, ctx: DeviceContext
    ) raises -> H3TrainBlockStore:
        """Metadata walk only — no weight bytes are read here. The heavy
        reads happen per stage(), demand-driven, from page cache."""
        var st = ShardedSafeTensors.open(dir)
        var names = h3_train_block_tensor_names()
        var nbs = List[Int]()
        var rels = List[Int]()
        var shapes = List[ArcPointer[List[Int]]]()
        var dts = List[STDtype]()
        var slab_bytes = 0
        for layer in range(num_blocks):
            var cursor = 0
            for t in range(len(names)):
                var full = String("blocks.") + String(layer) + "." + names[t]
                var info = st.tensor_info(full)
                if info.dtype != STDtype.BF16:
                    raise Error(
                        String("h3 train store: ") + full
                        + " is not BF16 on disk — the store copies verbatim"
                    )
                var numel = 1
                for i in range(len(info.shape)):
                    numel *= info.shape[i]
                var b = numel * info.dtype.byte_size()
                cursor = (cursor + 255) & ~255
                nbs.append(b)
                rels.append(cursor)
                shapes.append(ArcPointer(info.shape.copy()))
                dts.append(info.dtype)
                cursor += b
            if cursor > slab_bytes:
                slab_bytes = cursor
        var staging = ctx.enqueue_create_host_buffer[DType.uint8](slab_bytes)
        var slab = ctx.enqueue_create_buffer[DType.uint8](slab_bytes)
        ctx.synchronize()
        return H3TrainBlockStore(
            st^, staging^, slab^, num_blocks, names^, nbs^, rels^, shapes^, dts^,
        )

    def stage(mut self, layer: Int, ctx: DeviceContext) raises -> H3BlockTrainWeights:
        """Stage block `layer`: mmap → pinned staging → fixed slab, then
        return sub-buffer-view weights. Stream-ordered with subsequent
        kernels. The caller's per-block fence must have retired the
        previous block's kernels + H2D before this overwrites the slab
        and staging (the stack loops' contract)."""
        if layer < 0 or layer >= self.num_blocks:
            raise Error("h3 train store: layer out of range")
        var base = layer * H3_TRAIN_BLOCK_TENSORS
        if self.staged_layer != layer:
            # WILLNEED the whole block first: cold demand-faulting reads at
            # ~25MB/s; a batched WILLNEED runs the disk sequentially and the
            # copies below then hit page cache.
            for t in range(H3_TRAIN_BLOCK_TENSORS):
                self.st.prefetch_tensor(
                    String("blocks.") + String(layer) + "." + self.names[t]
                )
            var copied = 0
            for t in range(H3_TRAIN_BLOCK_TENSORS):
                var i = base + t
                var full = (
                    String("blocks.") + String(layer) + "." + self.names[t]
                )
                var span = self.st.tensor_bytes(full)
                var dst = BytePtr(
                    unsafe_from_address=Int(self.staging.unsafe_ptr())
                    + self.rel[i]
                )
                var src = BytePtr(unsafe_from_address=Int(span.unsafe_ptr()))
                _ = sys_memcpy(dst, src, self.nbytes[i])
                var end = self.rel[i] + self.nbytes[i]
                if end > copied:
                    copied = end
            # one async H2D for the packed block (pinned source = true async)
            var dsub = self.slab.create_sub_buffer[DType.uint8](0, copied)
            var hsub = self.staging.create_sub_buffer[DType.uint8](0, copied)
            ctx.enqueue_copy(dst_buf=dsub, src_buf=hsub)
        var views = List[TArc]()
        for t in range(H3_TRAIN_BLOCK_TENSORS):
            var i = base + t
            var view = self.slab.create_sub_buffer[DType.uint8](
                self.rel[i], self.nbytes[i]
            )
            views.append(TArc(Tensor(
                view^, self.shapes[i][].copy(), self.dtypes[i]
            )))
        self.staged_layer = layer
        # The released shard stores [head, qkv, dim] rows. The training block
        # splits a contiguous [q_all;k_all;v_all] projection, just like the
        # product block after its loader conversion. This allocation is also
        # what the FP8 resident builder quantizes, so steady-state resident
        # training pays no recurring permutation cost.
        var qkv = h3_qkv_deinterleave_rows(
            views[0][], H3_TRAIN_HEADS, H3_TRAIN_HEAD_DIM, ctx
        )
        return H3BlockTrainWeights(
            TArc(qkv^), views[1], views[2], views[3],
            views[4], views[5], views[6], views[7],
        )

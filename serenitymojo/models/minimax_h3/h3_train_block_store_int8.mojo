# serenitymojo/models/minimax_h3/h3_train_block_store_int8.mojo
#
# Direct W8A8 resident frozen base for H3 LoRA training. The first `resident`
# blocks live on device as tensorwise INT8 weights plus scalar F32 scales; the
# small norm vectors remain BF16. Unlike the FP8 store, resident stage() never
# expands a projection back to BF16. Forward and frozen dX backward consume the
# compact payload directly through ops/int8_linear, matching the established
# Klein trainer design. The nonresident tail retains the safe mmap BF16 stream.
from max.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.ops.int8_quant import (
    int8_tensorwise_scale, int8_encode_tensorwise,
)
from serenitymojo.models.minimax_h3.h3_block_train import (
    H3BlockTrainWeights, H3BlockTrainInt8,
)
from serenitymojo.models.minimax_h3.h3_train_block_store import H3TrainBlockStore

comptime TArc = ArcPointer[Tensor]
comptime _LINEARS = 4
comptime _NORMS = 4


struct H3TrainBlockStoreInt8(Movable):
    var w8: List[TArc]          # resident block-major qkv/out/fc1/fc2 [N,K]
    var scale: List[TArc]       # matching scalar F32 [1]
    var norms: List[TArc]       # resident block-major q/k/norm1/norm2 BF16
    var tail: H3TrainBlockStore
    var num_blocks: Int
    var resident: Int

    def __init__(
        out self,
        var w8: List[TArc], var scale: List[TArc], var norms: List[TArc],
        var tail: H3TrainBlockStore, num_blocks: Int, resident: Int,
    ):
        self.w8 = w8^
        self.scale = scale^
        self.norms = norms^
        self.tail = tail^
        self.num_blocks = num_blocks
        self.resident = resident

    @staticmethod
    def open(
        dir: String, num_blocks: Int, resident: Int, ctx: DeviceContext
    ) raises -> H3TrainBlockStoreInt8:
        if resident < 0 or resident > num_blocks:
            raise Error("h3 int8 store: resident out of range")
        var src = H3TrainBlockStore.open(dir, num_blocks, ctx)
        var w8 = List[TArc]()
        var scale = List[TArc]()
        var norms = List[TArc]()
        for layer in range(resident):
            var w = src.stage(layer, ctx)
            var lin = List[TArc]()
            lin.append(w.qkv_w)
            lin.append(w.out_w)
            lin.append(w.fc1_w)
            lin.append(w.fc2_w)
            for slot in range(_LINEARS):
                var s = int8_tensorwise_scale(lin[slot][], ctx)
                var q = int8_encode_tensorwise(lin[slot][], s, ctx)
                w8.append(TArc(q^))
                scale.append(TArc(s^))
            norms.append(TArc(w.q_norm[].clone(ctx)))
            norms.append(TArc(w.k_norm[].clone(ctx)))
            norms.append(TArc(w.norm1_w[].clone(ctx)))
            norms.append(TArc(w.norm2_w[].clone(ctx)))
            _ = w^
            # The source slab is reused for the next block. Settle quantization
            # before its BF16 contents are overwritten.
            ctx.synchronize()
        return H3TrainBlockStoreInt8(
            w8^, scale^, norms^, src^, num_blocks, resident,
        )

    def stage(mut self, layer: Int, ctx: DeviceContext) raises -> H3BlockTrainWeights:
        """Return compact resident handles or a streamed BF16 tail block."""
        if layer < 0 or layer >= self.num_blocks:
            raise Error("h3 int8 store: layer out of range")
        if layer >= self.resident:
            return self.tail.stage(layer, ctx)
        var lb = layer * _LINEARS
        var nb = layer * _NORMS
        return H3BlockTrainWeights(
            self.w8[lb].copy(), self.w8[lb + 1].copy(),
            self.w8[lb + 2].copy(), self.w8[lb + 3].copy(),
            self.norms[nb].copy(), self.norms[nb + 1].copy(),
            self.norms[nb + 2].copy(), self.norms[nb + 3].copy(),
        )

    def payload(self, layer: Int) raises -> Optional[H3BlockTrainInt8]:
        """Direct-compute payload for a resident block; None for BF16 tail."""
        if layer < 0 or layer >= self.num_blocks:
            raise Error("h3 int8 store: layer out of range")
        if layer >= self.resident:
            return Optional[H3BlockTrainInt8](None)
        var lb = layer * _LINEARS
        var q = List[TArc]()
        var s = List[TArc]()
        for slot in range(_LINEARS):
            q.append(self.w8[lb + slot].copy())
            s.append(self.scale[lb + slot].copy())
        return Optional[H3BlockTrainInt8](H3BlockTrainInt8(q^, s^))

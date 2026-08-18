# serenitymojo/models/minimax_h3/h3_train_block_store_fp8.mojo
#
# FP8-RESIDENT frozen base for H3 TRAINING (the krea2 pattern, accepted
# there at ~0.99 cos): the 4 big linears of every block live on-device as
# E4M3 bytes + per-output-row F32 scales (~19.25GB for 50 blocks); the tiny
# norm vectors stay BF16 verbatim. stage(layer) dequantizes the linears to
# BF16 (fp8_e4m3_dequant_perrow_to_bf16 — gated op) so ALL downstream block
# math is unchanged from the gated bf16 trainer; only the frozen weights
# carry one fp8 rounding.
#
# Build: one pass THROUGH the gated mmap store (H3TrainBlockStore.stage) —
# BF16 block views with QKV already deinterleaved to [q_all;k_all;v_all] ->
# rowscale -> encode -> resident U8+scales. The mmap
# store is dropped afterwards; steady-state training does ZERO host->device
# weight traffic.
from max.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor
from serenitymojo.ops.fp8_quant import fp8_e4m3_rowscale, fp8_e4m3_encode_perrow
from serenitymojo.ops.fp8 import fp8_e4m3_dequant_perrow_to_bf16
from serenitymojo.models.minimax_h3.h3_block_train import H3BlockTrainWeights
from serenitymojo.models.minimax_h3.h3_train_block_store import H3TrainBlockStore

comptime TArc = ArcPointer[Tensor]
comptime _LINEARS = 4  # qkv, out, fc1, fc2 (store order 0..3)
comptime _NORMS = 4    # q_norm, k_norm, norm1, norm2 (store order 4..7)


struct H3TrainBlockStoreFp8(Movable):
    """HYBRID residency: the first `resident` blocks live as fp8 on-device;
    the tail streams bf16 from the retained mmap store (24GB-card headroom
    knob — full-50 residency leaves no transient room next to the desktop)."""
    var q_bytes: List[TArc]   # per (resident block, linear): U8 [out, in]
    var q_scale: List[TArc]   # per (resident block, linear): F32 [out]
    var norms: List[TArc]     # per (resident block, norm): BF16 verbatim
    var tail: H3TrainBlockStore   # serves layers >= resident (and built from)
    var num_blocks: Int
    var resident: Int

    def __init__(
        out self,
        var q_bytes: List[TArc], var q_scale: List[TArc],
        var norms: List[TArc], var tail: H3TrainBlockStore,
        num_blocks: Int, resident: Int,
    ):
        self.q_bytes = q_bytes^
        self.q_scale = q_scale^
        self.norms = norms^
        self.tail = tail^
        self.num_blocks = num_blocks
        self.resident = resident

    @staticmethod
    def open(
        dir: String, num_blocks: Int, resident: Int, ctx: DeviceContext
    ) raises -> H3TrainBlockStoreFp8:
        if resident < 0 or resident > num_blocks:
            raise Error("h3 fp8 store: resident out of range")
        var src = H3TrainBlockStore.open(dir, num_blocks, ctx)
        var q_bytes = List[TArc]()
        var q_scale = List[TArc]()
        var norms = List[TArc]()
        for layer in range(resident):
            var w = src.stage(layer, ctx)
            # linears: quantize E4M3 per-row from the staged bf16 views
            var lin = List[TArc]()
            lin.append(w.qkv_w)
            lin.append(w.out_w)
            lin.append(w.fc1_w)
            lin.append(w.fc2_w)
            for t in range(_LINEARS):
                var scale = fp8_e4m3_rowscale(lin[t][], ctx)
                var bytes = fp8_e4m3_encode_perrow(lin[t][], scale, ctx)
                q_bytes.append(TArc(bytes^))
                q_scale.append(TArc(scale^))
            # norms: tiny — keep BF16 verbatim, resident
            norms.append(TArc(w.q_norm[].clone(ctx)))
            norms.append(TArc(w.k_norm[].clone(ctx)))
            norms.append(TArc(w.norm1_w[].clone(ctx)))
            norms.append(TArc(w.norm2_w[].clone(ctx)))
            _ = w^
            # fence: the staged slab is reused next layer; also settles the
            # quant kernels so their bf16 sources may be overwritten.
            ctx.synchronize()
        return H3TrainBlockStoreFp8(
            q_bytes^, q_scale^, norms^, src^, num_blocks, resident,
        )

    def stage(mut self, layer: Int, ctx: DeviceContext) raises -> H3BlockTrainWeights:
        """Dequant the block's linears to fresh BF16 tensors (freed by the
        stack loop's per-block fence); norms are shared residents."""
        if layer < 0 or layer >= self.num_blocks:
            raise Error("h3 fp8 store: layer out of range")
        if layer >= self.resident:
            return self.tail.stage(layer, ctx)
        var lb = layer * _LINEARS
        var nb = layer * _NORMS
        var qkv = fp8_e4m3_dequant_perrow_to_bf16(
            self.q_bytes[lb][], self.q_scale[lb][], ctx
        )
        var outw = fp8_e4m3_dequant_perrow_to_bf16(
            self.q_bytes[lb + 1][], self.q_scale[lb + 1][], ctx
        )
        var fc1 = fp8_e4m3_dequant_perrow_to_bf16(
            self.q_bytes[lb + 2][], self.q_scale[lb + 2][], ctx
        )
        var fc2 = fp8_e4m3_dequant_perrow_to_bf16(
            self.q_bytes[lb + 3][], self.q_scale[lb + 3][], ctx
        )
        return H3BlockTrainWeights(
            TArc(qkv^), TArc(outw^), TArc(fc1^), TArc(fc2^),
            self.norms[nb].copy(), self.norms[nb + 1].copy(),
            self.norms[nb + 2].copy(), self.norms[nb + 3].copy(),
        )

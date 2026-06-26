# autograd_v2/tests/krea2_slab_block_peak.mojo — MEASURE the engine+slab WHOLE-BLOCK
# device-grad backward peak (krea2_single_stream_block_graph_backward_seg), both
# MATH (bit-gate path) and the flag's current setting. The decisive number: does the
# engine+slab whole-block fit alongside the 12GB fp8 base on 24GB (→ no segmentation)?
#
# Loops the slab backward (fresh slab each iter, sized ~10GB) so nvidia-smi samples
# the steady-state peak. bf16 production dtype. KREA2_SLAB_FLASH (krea2_block.mojo)
# selects MATH (default, this build) vs FLASH attn.
#
# Build:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#   pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     serenitymojo/autograd_v2/tests/krea2_slab_block_peak.mojo -o /tmp/krea2_slab_block_peak

from std.gpu.host import DeviceContext
from std.collections import Optional
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import add_scalar, mul_scalar, zeros_device
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.models.dit.krea2_dit import _tile_rope_table
from serenitymojo.models.klein.lora_adapter import make_lora_adapter
from serenitymojo.models.klein.lora_block import lora_adapter_to_device, LoraAdapterDevice
from serenitymojo.models.krea2.krea2_block import Krea2BlockWeights, Krea2BlockLora, KREA2_SLAB_FLASH
from serenitymojo.autograd_v2.step_slab import StepSlab
from serenitymojo.autograd_v2.krea2_block_graph import krea2_single_stream_block_graph_backward_seg

comptime TArc = ArcPointer[Tensor]
comptime HEADS = 48
comptime KVHEADS = 12
comptime HEADDIM = 128
comptime FEATURES = HEADS * HEADDIM
comptime MLPDIM = 16384
comptime L = 4864


def _bf(var shape: List[Int], seed: UInt64, sc: Float32, ctx: DeviceContext) raises -> TArc:
    return TArc(cast_tensor(mul_scalar(randn(shape^, seed, STDtype.F32, ctx), sc, ctx), STDtype.BF16, ctx))


def _f1(n: Int, ctx: DeviceContext) raises -> TArc:
    return TArc(zeros_device([n], STDtype.F32, ctx))


def _mk(in_f: Int, out_f: Int, seed: UInt64, ctx: DeviceContext) raises -> Optional[LoraAdapterDevice]:
    var dev = lora_adapter_to_device(make_lora_adapter(16, Float32(16.0), in_f, out_f, seed), ctx)
    var b = cast_tensor(mul_scalar(randn(dev.b[].shape(), seed + UInt64(9000), STDtype.F32, ctx), Float32(0.04), ctx), STDtype.BF16, ctx)
    return Optional[LoraAdapterDevice](LoraAdapterDevice(dev.a.copy(), TArc(b^), dev.rank, dev.in_f, dev.out_f, dev.scale))


def main() raises:
    var ctx = DeviceContext()
    var eps = Float32(1.0e-5)
    var w = Krea2BlockWeights(
        _bf([HEADS * HEADDIM, FEATURES], UInt64(1), Float32(0.04), ctx),
        _bf([KVHEADS * HEADDIM, FEATURES], UInt64(2), Float32(0.04), ctx),
        _bf([KVHEADS * HEADDIM, FEATURES], UInt64(3), Float32(0.04), ctx),
        _bf([FEATURES, FEATURES], UInt64(4), Float32(0.04), ctx),
        _bf([FEATURES, FEATURES], UInt64(5), Float32(0.04), ctx),
        _bf([MLPDIM, FEATURES], UInt64(6), Float32(0.04), ctx),
        _bf([MLPDIM, FEATURES], UInt64(7), Float32(0.04), ctx),
        _bf([FEATURES, MLPDIM], UInt64(8), Float32(0.04), ctx),
        _f1(HEADDIM, ctx), _f1(HEADDIM, ctx), _f1(FEATURES, ctx), _f1(FEATURES, ctx),
        _bf([6 * FEATURES], UInt64(20), Float32(0.1), ctx),
    )
    var lora = Krea2BlockLora(
        _mk(FEATURES, HEADS * HEADDIM, UInt64(100), ctx), _mk(FEATURES, KVHEADS * HEADDIM, UInt64(101), ctx),
        _mk(FEATURES, KVHEADS * HEADDIM, UInt64(102), ctx), _mk(FEATURES, FEATURES, UInt64(103), ctx),
        _mk(FEATURES, FEATURES, UInt64(104), ctx), _mk(FEATURES, MLPDIM, UInt64(105), ctx),
        _mk(FEATURES, MLPDIM, UInt64(106), ctx), _mk(MLPDIM, FEATURES, UInt64(107), ctx),
    )
    var vec = cast_tensor(mul_scalar(randn([1, 6 * FEATURES], UInt64(30), STDtype.F32, ctx), Float32(0.1), ctx), STDtype.BF16, ctx)
    var x = _bf([1, L, FEATURES], UInt64(40), Float32(0.5), ctx)
    var d_out = _bf([1, L, FEATURES], UInt64(50), Float32(0.5), ctx)
    var cos = add_scalar(zeros_device([L, HEADDIM // 2], STDtype.F32, ctx), Float32(1.0), ctx)
    var sin = zeros_device([L, HEADDIM // 2], STDtype.F32, ctx)
    var cos_q = _tile_rope_table(cos, L, HEADS, HEADDIM // 2, ctx)
    var sin_q = _tile_rope_table(sin, L, HEADS, HEADDIM // 2, ctx)
    var cos_k = _tile_rope_table(cos, L, KVHEADS, HEADDIM // 2, ctx)
    var sin_k = _tile_rope_table(sin, L, KVHEADS, HEADDIM // 2, ctx)
    var rl = Optional[Int](None)

    print("krea2 engine+slab WHOLE-BLOCK backward @ L=4864 bf16, KREA2_SLAB_FLASH=", KREA2_SLAB_FLASH, ", 8 iters — sample nvidia-smi:")
    for i in range(8):
        var slab = StepSlab(ctx, 8 * 1024 * 1024 * 1024)
        var bg = krea2_single_stream_block_graph_backward_seg[L, HEADS, KVHEADS, HEADDIM](
            d_out, x, vec, w, lora, cos, sin, cos_q, sin_q, cos_k, sin_k, eps, ctx, slab, rl,
        )
        slab.reset()
        ctx.synchronize()
        if i == 0:
            var h = bg.d_x[].to_host(ctx)
            print("  iter0 d_x[0] =", h[0], " peak_bytes(GB)=", Float64(slab.peak_bytes()) / Float64(1024*1024*1024))
    print("done — engine+slab whole-block footprint.")

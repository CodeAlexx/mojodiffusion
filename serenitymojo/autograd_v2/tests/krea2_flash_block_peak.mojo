# autograd_v2/tests/krea2_flash_block_peak.mojo — MEASURE the PRODUCTION per-block
# FLASH backward peak (the hand-chain block fwd+bwd on the flash path), the airtight
# go/no-go: does the flash per-block backward co-fit the 12GB fp8 base on 24GB?
#
# This is the REAL production per-block footprint — krea2_single_stream_block_lora
# (flash fwd, real_len<L) + krea2_single_stream_block_lora_backward (flash bwd), the
# exact ops the streamed conductor runs per block. nvidia-smi samples the steady-state
# peak over a loop. If this whole-block flash backward already fits (< ~10GB so 12GB
# fp8 + this + cond + working ≤ 24GB), then the flash path fits WITHOUT segmentation —
# segmentation/slab is then only for the alloc-free (engine+slab) speed path, not the fit.
#
# Compare to the math whole-block (~20GB, didn't fit): flash removes the O(L²) scores.
#
# Build:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#   pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     serenitymojo/autograd_v2/tests/krea2_flash_block_peak.mojo -o /tmp/krea2_flash_block_peak

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
from serenitymojo.models.krea2.krea2_block import (
    Krea2BlockWeights, Krea2BlockLora,
    krea2_single_stream_block_lora, krea2_single_stream_block_lora_backward,
)

comptime TArc = ArcPointer[Tensor]
comptime HEADS = 48
comptime KVHEADS = 12
comptime HEADDIM = 128
comptime FEATURES = HEADS * HEADDIM   # 6144
comptime MLPDIM = 16384
comptime L = 4864                      # the REAL trainer length
comptime RL = 4736                     # real_len < L → FLASH path (128-aligned tail pad)


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
    var d_out = cast_tensor(mul_scalar(randn([1, L, FEATURES], UInt64(50), STDtype.F32, ctx), Float32(0.5), ctx), STDtype.BF16, ctx)
    var cos = add_scalar(zeros_device([L, HEADDIM // 2], STDtype.F32, ctx), Float32(1.0), ctx)
    var sin = zeros_device([L, HEADDIM // 2], STDtype.F32, ctx)
    var cos_q = _tile_rope_table(cos, L, HEADS, HEADDIM // 2, ctx)
    var sin_q = _tile_rope_table(sin, L, HEADS, HEADDIM // 2, ctx)
    var cos_k = _tile_rope_table(cos, L, KVHEADS, HEADDIM // 2, ctx)
    var sin_k = _tile_rope_table(sin, L, KVHEADS, HEADDIM // 2, ctx)
    var rl = Optional[Int](RL)   # FLASH path (real_len < L)

    print("krea2 PER-BLOCK FLASH backward @ L=4864 bf16 (production hand-chain), 12 iters — sample nvidia-smi:")
    for i in range(12):
        var fb = krea2_single_stream_block_lora[L, HEADS, KVHEADS, HEADDIM](
            x, vec, w, lora, cos, sin, cos_q, sin_q, cos_k, sin_k, eps, ctx, rl,
        )
        var bg = krea2_single_stream_block_lora_backward[L, HEADS, KVHEADS, HEADDIM](
            d_out, vec, w, lora, fb.saved, cos_q, sin_q, cos_k, sin_k, eps, ctx, rl,
        )
        ctx.synchronize()
        if i == 0:
            var h = bg.d_x[].to_host(ctx)   # DCE guard
            print("  iter0 d_x[0] =", h[0])
    print("done — this is the production per-block flash fwd+bwd footprint.")

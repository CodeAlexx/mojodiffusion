# autograd_v2/tests/krea2_flash_sdpa_peak.mojo — MEASURE the flash-sdpa working set
# (the O(L) replacement for the math sdpa_nomask's O(L²) [B*H*S,S] scores that made
# the math attn segment 13.4GB). Confirms flash attn is O(L) → the attn segment fits.
#
# Runs the krea2 attn-shape flash fwd+bwd (sdpa_flash_train_fwd_f32 / _backward_f32,
# no-pad full attention) in a LOOP so nvidia-smi can sample the steady-state peak.
# The flash saved set is q/k/v/o bf16 + stats F32 = all [1,L,48,128] = O(L); NO
# [H,L,L] scores. Compare to the math sdpa, whose scores are 1×48×L×L×4.
#
# Build:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#   pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     serenitymojo/autograd_v2/tests/krea2_flash_sdpa_peak.mojo -o /tmp/krea2_flash_sdpa_peak
# Run with an external nvidia-smi sampler (the orchestrator/this harness loops).

from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from std.math import sqrt
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import mul_scalar
from serenitymojo.ops.attention import sdpa_nomask
from serenitymojo.ops.attention_backward import sdpa_backward
from serenitymojo.ops.attention_flash import (
    sdpa_flash_train_fwd_f32, sdpa_flash_backward_f32,
)

comptime TArc = ArcPointer[Tensor]
comptime HEADS = 48
comptime HEADDIM = 128
comptime L = 4864   # the REAL trainer length (128-aligned: 38×128)


def _f32(var shape: List[Int], seed: UInt64, ctx: DeviceContext) raises -> Tensor:
    return mul_scalar(randn(shape^, seed, STDtype.F32, ctx), Float32(0.1), ctx)


def main() raises:
    var ctx = DeviceContext()
    var scale = Float32(1.0) / sqrt(Float32(HEADDIM))
    var q = _f32([1, L, HEADS, HEADDIM], UInt64(1), ctx)
    var k = _f32([1, L, HEADS, HEADDIM], UInt64(2), ctx)
    var v = _f32([1, L, HEADS, HEADDIM], UInt64(3), ctx)
    var d_att = _f32([1, L, HEADS, HEADDIM], UInt64(4), ctx)

    # math sdpa scores would be 1×48×4864×4864×4 = 4.5GB EACH (O(L²)). Flash below
    # materializes none. Loop so nvidia-smi samples the steady-state flash working set.
    print("flash sdpa fwd+bwd @ L=4864 (krea2 attn shape), 20 iters — sample nvidia-smi:")
    for i in range(20):
        var fwd = sdpa_flash_train_fwd_f32[1, L, HEADS, HEADDIM](q, k, v, scale, ctx)
        var bwd = sdpa_flash_backward_f32[1, L, HEADS, HEADDIM](
            fwd.q_bf, fwd.k_bf, fwd.v_bf, fwd.o_bf, fwd.stats, d_att, scale, ctx,
        )
        ctx.synchronize()
        if i == 0:
            var h = bwd.d_q.to_host(ctx)   # DCE guard
            print("  iter0 d_q[0] =", h[0])
    print("done — flash working set is O(L): q/k/v/o bf16 + stats F32, all [1,L,48,128].")

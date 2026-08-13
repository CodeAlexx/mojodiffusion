# Before/after equivalence probe for the gemma3 `_attention_dispatch` refactor.
#
# `gemma3_ltx_streamed._attention_dispatch` enumerates the eight 128-step
# buckets against `sdpa_flash_infer_fwd_causal_padmask[B,S,H,Dh]` as if the
# comptime parameters selected a specialization. They do not: that wrapper's
# entire body forwards to `..._dynamic`, which reads B/S/H/Dh off the tensor
# shapes (ops/attention_flash.mojo:283). The table is eight identical calls.
#
# gemma3_ltx_streamed.mojo is SHIPPED and gated, so the refactor is not taken on
# reasoning-by-analogy with the gemma4 change. This probe drives the unit under
# change directly, at EVERY bucket the table enumerates — stronger coverage than
# a real-checkpoint encode, which only ever exercises the one bucket its prompt
# lands in. Weights are seeded (`randn(..., seed, ...)`), so the numbers are
# reproducible across runs and the ONLY thing that may differ before vs after is
# the dispatch itself.
#
# Protocol: run on the unmodified file, record; apply the refactor; run again.
# Every printed field must match to the last digit.
#
# Build (the cuDNN SDPA shim MUST be linked):
#   pixi run mojo build -O0 -j 1 -I . -I vendor/mojo-libs \
#     -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     -Xlinker -Lserenitymojo/ops/cshim/lib/cudnn_stubs -Xlinker -lcudnn \
#     -Xlinker -rpath -Xlinker '$ORIGIN/../../serenitymojo/ops/cshim/lib' \
#     -Xlinker -rpath -Xlinker '$ORIGIN/../../.pixi/envs/default/lib' \
#     serenitymojo/models/text_encoder/parity/gemma3_dispatch_refactor_probe.mojo \
#     -o output/checks/gemma3_dispatch_probe

from max.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.models.text_encoder.gemma3_ltx_streamed import (
    GEMMA_HEAD_DIM,
    GEMMA_HEADS,
    _attention_dispatch,
)
from serenitymojo.ops.random import randn
from serenitymojo.tensor import Tensor


def _digest(name: String, t: Tensor, ctx: DeviceContext) raises:
    """Full-precision reduction of every element — any bit difference moves it."""
    var h = t.to_host(ctx)
    var n = len(h)
    var s: Float64 = 0.0
    var ss: Float64 = 0.0
    var mn: Float64 = 1.0e30
    var mx: Float64 = -1.0e30
    # Position-weighted term so a pure permutation of values is also caught.
    var pw: Float64 = 0.0
    for i in range(n):
        var v = Float64(h[i])
        s += v
        ss += v * v
        pw += v * Float64((i % 1021) + 1)
        if v < mn:
            mn = v
        if v > mx:
            mx = v
    print(
        name, " n=", n, " sum=", s, " sumsq=", ss,
        " min=", mn, " max=", mx, " poswt=", pw,
    )


def main() raises:
    var ctx = DeviceContext()
    print("== gemma3 _attention_dispatch refactor equivalence probe ==")
    print("heads=", GEMMA_HEADS, " head_dim=", GEMMA_HEAD_DIM)

    var buckets: List[Int] = [128, 256, 384, 512, 640, 768, 896, 1024]
    for bi in range(len(buckets)):
        var seq = buckets[bi]
        # real_len deliberately < seq so the causal pad-mask path is live.
        var real_len = seq - 27
        if real_len < 1:
            real_len = 1
        var shape: List[Int] = [1, seq, GEMMA_HEADS, GEMMA_HEAD_DIM]
        var q = randn(shape.copy(), UInt64(1000 + bi), STDtype.BF16, ctx)
        var k = randn(shape.copy(), UInt64(2000 + bi), STDtype.BF16, ctx)
        var v = randn(shape.copy(), UInt64(3000 + bi), STDtype.BF16, ctx)
        var o = _attention_dispatch(q, k, v, real_len, seq, ctx)
        var label = String("bucket ") + String(seq) + " real_len " + String(real_len)
        _digest(label, o, ctx)

    print("PROBE OK")

# ideogram4_uncond_stream_gate — MJ-1142 fix neutrality gate.
#
# Loads the REAL uncond fp8 trunk twice: fully device-resident (the old
# >=20GB path) and as Ideogram4UncondStream (pinned-host staged layers).
# Runs both forwards on identical GPU-generated inputs and requires
# byte-identical F32 outputs: the stream stages the same bytes through the
# same kernels in the same order, so any difference is a staging bug.
# Fits alone in ~10GB — runnable on the 24GB card where dual-resident+job
# cannot fit (which is the point of the fix).
from max.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.tensor import Tensor
from serenitymojo.ops.random import randn
from serenitymojo.models.dit.ideogram4_resident import (
    Ideogram4Weights, Ideogram4UncondStream, ideogram4_forward_r,
    ideogram4_build_masks,
)

comptime UNCOND = "/home/alex/mojodiffusion/models/ideogram4/unconditional_transformer/diffusion_pytorch_model.safetensors"
comptime S = 1024  # small image-only grid: 32x32 patches
comptime LAYERS = 34
comptime HEADS = 18
comptime HEAD_DIM = 256
comptime HIDDEN = 4608
comptime LLM_DIM = 53248
comptime LATENT_DIM = 128


def _max_abs(a: Tensor, b: Tensor, ctx: DeviceContext) raises -> Float64:
    var ah = a.to_host(ctx)
    var bh = b.to_host(ctx)
    var mad = Float64(0.0)
    for i in range(len(ah)):
        var d = Float64(ah[i]) - Float64(bh[i])
        if d < 0:
            d = -d
        if d > mad:
            mad = d
    return mad


def main() raises:
    var ctx = DeviceContext()
    var shape: List[Int] = [1, S, LATENT_DIM]
    var x = randn(shape.copy(), UInt64(42), STDtype.BF16, ctx)
    var lshape: List[Int] = [1, S, LLM_DIM]
    var llm = randn(lshape.copy(), UInt64(43), STDtype.BF16, ctx)
    var t = Tensor.from_host([Float32(0.5)], [1], STDtype.F32, ctx)
    # all-image indicator (=2.0): img_mask ones, llm_mask zeros — the uncond
    # geometry (image tokens only), matching the backend's uncond pass.
    var ind = List[Float32]()
    for _ in range(S):
        ind.append(Float32(2.0))
    var indicator = Tensor.from_host(ind^, [1, S], STDtype.F32, ctx)
    var masks = ideogram4_build_masks(indicator, ctx)
    var hd2: List[Int] = [1, S, HEAD_DIM]
    var cosf = randn(hd2.copy(), UInt64(44), STDtype.F32, ctx)
    var sinf = randn(hd2.copy(), UInt64(45), STDtype.F32, ctx)

    print("[gate] loading uncond trunk RESIDENT")
    var wres = Ideogram4Weights.load(ShardedSafeTensors.open(String(UNCOND)), ctx)
    var ref_out = ideogram4_forward_r[S](
        wres, x, llm, t, masks, cosf, sinf, LAYERS, HEADS, HEAD_DIM, HIDDEN, ctx
    )
    ctx.synchronize()
    _ = wres^  # free the resident trunk before the stream loads

    print("[gate] loading uncond trunk STREAMED")
    var stream = Ideogram4UncondStream.load_stream(
        ShardedSafeTensors.open(String(UNCOND)), ctx
    )
    var got = stream.forward_streamed[S](
        x, llm, t, masks, cosf, sinf, LAYERS, HEADS, HEAD_DIM, HIDDEN, ctx
    )
    ctx.synchronize()

    var mad = _max_abs(ref_out, got, ctx)
    print("[gate] max_abs(resident - streamed) =", mad)
    if mad == 0.0:
        print("PASS: streamed uncond is byte-identical to resident")
    else:
        print("FAIL: streamed uncond diverges from resident")
        raise Error("ideogram4_uncond_stream_gate failed")

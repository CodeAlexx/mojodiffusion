# ltx2_vae_decode_stage0_smoke.mojo — LTX-2.3 Video VAE decoder STAGE 0 smoke.
#
# Loads the stage-0 weights (per_channel_statistics + conv_in + up_blocks.0)
# from the LTX-2.3 checkpoint, builds a synthetic BF16 latent, runs
# decode_stage0, prints stats + output shape, and asserts the output is finite.
#
# Bounded sizes: latent [1, 128, F=4, H=8, W=8] (NCDHW), production channel
# count C=128. Stage 0 keeps F/H/W unchanged and lifts channels 128 -> 1024,
# so the expected output is NDHWC [1, 4, 8, 8, 1024].
#
# *** CODE-ONLY: this file is COMPILE-VERIFIED, NOT executed (no GPU run). ***
#
# Build:
#   cd /home/alex/mojodiffusion
#   pixi run mojo build -I . -Xlinker -lm \
#     serenitymojo/pipeline/ltx2_vae_decode_stage0_smoke.mojo \
#     -o /tmp/ltx2_vae_decode_stage0_smoke

from std.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.vae.ltx2_vae_decoder import (
    LTX2VaeDecoderStage0Weights,
    decode_stage0,
)


comptime CKPT = (
    "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled.safetensors"
)

comptime B = 1
comptime C = 128
comptime F = 4
comptime H = 8
comptime W = 8


def _stats(name: String, t: Tensor, ctx: DeviceContext) raises:
    var h = t.to_host(ctx)
    var n = len(h)
    var s = 0.0
    var s2 = 0.0
    var amax = 0.0
    var finite = True
    for i in range(n):
        var v = Float64(h[i])
        # NaN != NaN; |inf| overflows comparison — flag non-finite.
        if not (v == v) or (v > 1.0e38) or (v < -1.0e38):
            finite = False
        s += v
        s2 += v * v
        var av = v if v >= 0.0 else -v
        if av > amax:
            amax = av
    var mean = s / Float64(n)
    var var_ = s2 / Float64(n) - mean * mean
    if var_ < 0.0:
        var_ = 0.0
    print(
        "  [stat]", name, "mean=", Float32(mean), "std=", Float32(sqrt(var_)),
        "absmax=", Float32(amax), "n=", n, "finite=", finite,
    )
    if not finite:
        raise Error(String("non-finite values in ") + name)


def main() raises:
    var ctx = DeviceContext()

    print("LTX-2.3 Video VAE decoder STAGE 0 smoke")
    print("  loading stage-0 weights from:", CKPT)
    var weights = LTX2VaeDecoderStage0Weights.load(CKPT, ctx)
    print("  weights loaded.")

    # Synthetic BF16 latent [1,128,4,8,8] (NCDHW) — small deterministic ramp.
    var numel = B * C * F * H * W
    var host = List[Float32]()
    host.resize(numel, Float32(0.0))
    for i in range(numel):
        # bounded, deterministic, centered around 0.
        host[i] = Float32((i % 17) - 8) * Float32(0.05)
    var sh = List[Int]()
    sh.append(B); sh.append(C); sh.append(F); sh.append(H); sh.append(W)
    var latent = Tensor.from_host(host, sh^, STDtype.BF16, ctx)
    _stats("input latent (NCDHW)", latent, ctx)

    var out = decode_stage0[B, C, F, H, W](weights, latent, ctx)
    var os = out.shape()
    print(
        "  output shape (NDHWC): [", os[0], ",", os[1], ",", os[2], ",",
        os[3], ",", os[4], "]",
    )
    _stats("stage0 output", out, ctx)

    # Expected NDHWC [1, 4, 8, 8, 1024] — stage 0 preserves F/H/W, lifts C->1024.
    if (
        os[0] != B or os[1] != F or os[2] != H or os[3] != W or os[4] != 1024
    ):
        raise Error("stage0 output shape mismatch")
    print("  PASS: stage-0 forward produced finite output of expected shape.")

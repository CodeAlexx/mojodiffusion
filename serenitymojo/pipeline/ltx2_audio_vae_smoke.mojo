# ltx2_audio_vae_smoke.mojo — LTX-2.3 Audio VAE FULL decode parity gate (P4).
#
# Loads the deterministic normalized latent produced by the Python oracle
# (scripts/ltx2_audio_vae_ref.py), runs the pure-Mojo full audio VAE decoder
# (models/vae/ltx2_audio_vae.decode), and GATES:
#
#   HARD GATE: cosine(decoded_mojo, decoded_ref) >= 0.999
#              vs output/ltx2_audio_vae/audio_vae_ref.safetensors `decoded`
#
# Latent: [1,8,8,16] (NCHW), F32 on disk -> BF16 compute. Decoded mel:
# NCHW [1, 2, 4*8-3, 64] = [1,2,29,64].
#
# Build:
#   cd /home/alex/mojodiffusion
#   pixi run mojo build -I . -Xlinker -lm \
#     serenitymojo/pipeline/ltx2_audio_vae_smoke.mojo \
#     -o /tmp/ltx2_audio_vae_smoke
# Run (after scripts/ltx2_audio_vae_ref.py):
#   /tmp/ltx2_audio_vae_smoke

from std.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.models.vae.ltx2_audio_vae import (
    LTX2AudioVaeDecoderWeights,
    decode,
)


comptime CKPT = (
    "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled.safetensors"
)
comptime REF = (
    "/home/alex/mojodiffusion/output/ltx2_audio_vae/audio_vae_ref.safetensors"
)

comptime B = 1
comptime C = 8
comptime T = 8
comptime F = 16
comptime T_OUT = 4 * T - 3   # 29
comptime F_OUT = 64
comptime GATE = Float64(0.999)


def _cosine_sim(a: List[Float32], b: List[Float32]) raises -> Float64:
    if len(a) != len(b):
        raise Error("cosine_sim: length mismatch")
    var dot = Float64(0.0)
    var na = Float64(0.0)
    var nb = Float64(0.0)
    for i in range(len(a)):
        var av = Float64(a[i])
        var bv = Float64(b[i])
        dot += av * bv
        na += av * av
        nb += bv * bv
    if na == 0.0 or nb == 0.0:
        raise Error("cosine_sim: zero-norm vector")
    return dot / (sqrt(na) * sqrt(nb))


def _stats(name: String, h: List[Float32]) raises -> Bool:
    var n = len(h)
    var s = 0.0
    var s2 = 0.0
    var amax = 0.0
    var finite = True
    for i in range(n):
        var v = Float64(h[i])
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
    return finite


def main() raises:
    var ctx = DeviceContext()

    print("=== LTX-2.3 Audio VAE FULL decode parity gate (P4) ===")
    print("  checkpoint:", CKPT)
    print("  oracle:    ", REF)

    var weights = LTX2AudioVaeDecoderWeights.load(CKPT, ctx)
    print("  audio decoder weights loaded.")

    # Load the SAME latent the oracle decoded (F32 on disk -> BF16 compute).
    var fix = ShardedSafeTensors.open(String(REF))
    var latent = Tensor.from_view_as_bf16(fix.tensor_view("latent"), ctx)
    var ls = latent.shape()
    print(
        "  latent shape: [", ls[0], ",", ls[1], ",", ls[2], ",", ls[3], "]",
    )
    if len(ls) != 4 or ls[0] != B or ls[1] != C or ls[2] != T or ls[3] != F:
        raise Error("latent shape mismatch vs oracle")

    # Decode (full path).
    var out = decode(weights, latent, ctx)
    var osh = out.shape()
    print(
        "  decoded shape (NCHW): [", osh[0], ",", osh[1], ",", osh[2], ",",
        osh[3], "]",
    )
    if osh[0] != B or osh[1] != 2 or osh[2] != T_OUT or osh[3] != F_OUT:
        raise Error("decoded output shape mismatch")

    var out_host = out.to_host(ctx)
    var finite = _stats("decoded (mojo)", out_host)
    if not finite:
        raise Error("decoded output has non-finite values")

    # Reference decoded mel (F32, NCHW).
    var ref_t = Tensor.from_view(fix.tensor_view("decoded"), ctx)
    var ref_host = ref_t.to_host(ctx)
    _ = _stats("decoded (ref)", ref_host)

    var cos = _cosine_sim(out_host, ref_host)
    var mad = Float64(0.0)
    for i in range(len(out_host)):
        var d = Float64(out_host[i]) - Float64(ref_host[i])
        var ad = d if d >= 0.0 else -d
        if ad > mad:
            mad = ad
    print("  cosine:", Float32(cos), " max_abs_diff:", Float32(mad))

    print("=== HARD GATE: cos >= 0.999 ===")
    if cos < GATE:
        raise Error(
            String("audio VAE decode cosine ") + String(Float32(cos))
            + " < 0.999"
        )
    print("  [gate] decode cos >= 0.999: PASS")
    print("LTX-2.3 Audio VAE full decode parity gate PASS")

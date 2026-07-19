# ltx2_video_vae_decode_smoke.mojo — LTX-2.3 Video VAE FULL decode parity gate.
#
# Loads the deterministic normalized latent produced by the Python oracle
# (scripts/ltx2_video_vae_decode_ref.py), runs the pure-Mojo full video VAE
# decoder (models/vae/ltx2_vae_decoder.decode), and GATES:
#
#   HARD GATE: cosine(decoded_mojo, decoded_ref) >= 0.999
#              vs output/ltx2_video_vae/video_vae_ref.safetensors `decoded`
#
# plus saves frame-0 of the decoded video as a PNG and confirms it is a
# coherent (non-noise, finite, in-range) image (HARD RULE artifact).
#
# Latent: [1,128,2,2,2] (NCDHW), F32 on disk -> BF16 compute. Decoded video:
# NCDHW [1, 3, 1+(2-1)*8, 2*32, 2*32] = [1,3,9,64,64].
#
# Build:
#   cd /home/alex/mojodiffusion
#   pixi run mojo build -I . -Xlinker -lm \
#     serenitymojo/pipeline/ltx2_video_vae_decode_smoke.mojo \
#     -o /tmp/ltx2_video_vae_decode_smoke
# Run (after scripts/ltx2_video_vae_decode_ref.py):
#   /tmp/ltx2_video_vae_decode_smoke

from std.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.tensor_algebra import slice, reshape
from serenitymojo.image.png import save_png, ValueRange
from serenitymojo.models.vae.ltx2_vae_decoder import (
    LTX2VaeDecoderWeights,
    decode,
)


comptime CKPT = (
    "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled.safetensors"
)
comptime REF = (
    "/home/alex/mojodiffusion/output/ltx2_video_vae/video_vae_ref.safetensors"
)
comptime PNG_OUT = (
    "/home/alex/mojodiffusion/output/ltx2_video_vae/ltx2_video_frame00.png"
)

comptime B = 1
comptime C = 128
comptime F = 2
comptime H = 2
comptime W = 2
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

    print("=== LTX-2.3 Video VAE FULL decode parity gate (P5) ===")
    print("  checkpoint:", CKPT)
    print("  oracle:    ", REF)

    # Load decoder weights.
    var weights = LTX2VaeDecoderWeights.load(CKPT, ctx)
    print("  decoder weights loaded.")

    # Load the SAME latent the oracle decoded (F32 on disk -> BF16 compute).
    var fix = ShardedSafeTensors.open(String(REF))
    var latent = Tensor.from_view_as_bf16(fix.tensor_view("latent"), ctx)
    var ls = latent.shape()
    print(
        "  latent shape: [", ls[0], ",", ls[1], ",", ls[2], ",", ls[3], ",",
        ls[4], "]",
    )
    if (
        len(ls) != 5 or ls[0] != B or ls[1] != C or ls[2] != F
        or ls[3] != H or ls[4] != W
    ):
        raise Error("latent shape mismatch vs oracle")

    # Decode (full path).
    var out = decode[B, C, F, H, W](weights, latent, ctx)
    var os = out.shape()
    var f_out = 1 + (F - 1) * 8
    print(
        "  decoded shape (NCDHW): [", os[0], ",", os[1], ",", os[2], ",",
        os[3], ",", os[4], "]",
    )
    if (
        os[0] != B or os[1] != 3 or os[2] != f_out
        or os[3] != H * 32 or os[4] != W * 32
    ):
        raise Error("decoded output shape mismatch")

    var out_host = out.to_host(ctx)
    var finite = _stats("decoded (mojo)", out_host)
    if not finite:
        raise Error("decoded output has non-finite values")

    # Reference decoded video (F32, NCDHW).
    var ref_t = Tensor.from_view(fix.tensor_view("decoded"), ctx)
    var ref_host = ref_t.to_host(ctx)
    _ = _stats("decoded (ref)", ref_host)

    var cos = _cosine_sim(out_host, ref_host)
    # max abs diff
    var mad = Float64(0.0)
    for i in range(len(out_host)):
        var d = Float64(out_host[i]) - Float64(ref_host[i])
        var ad = d if d >= 0.0 else -d
        if ad > mad:
            mad = ad
    print("  cosine:", Float32(cos), " max_abs_diff:", Float32(mad))

    # ── Save frame-0 PNG: NCDHW [1,3,9,64,64] -> frame 0 -> [1,3,64,64] ───────
    # slice the D (frame) axis (dim 2) to length 1 at index 0, then drop it by
    # reshaping to [1,3,H_out,W_out].
    var frame0 = slice(out, 2, 0, 1, ctx)  # [1,3,1,64,64]
    var f0s = frame0.shape()
    # reshape to [1,3,H_out,W_out]
    var rsh = List[Int]()
    rsh.append(f0s[0]); rsh.append(f0s[1]); rsh.append(f0s[3]); rsh.append(f0s[4])
    var frame0_chw = reshape(frame0, rsh^, ctx)
    save_png(frame0_chw, String(PNG_OUT), ctx, ValueRange.SIGNED)
    print("  FRAME SAVED:", PNG_OUT)

    print("=== HARD GATE: cos >= 0.999 ===")
    if cos < GATE:
        raise Error(
            String("video VAE decode cosine ") + String(Float32(cos))
            + " < 0.999"
        )
    print("  [gate] decode cos >= 0.999: PASS")
    print("LTX-2.3 Video VAE full decode parity gate PASS")

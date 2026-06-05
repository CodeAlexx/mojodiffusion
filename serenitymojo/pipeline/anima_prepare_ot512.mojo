# serenitymojo/pipeline/anima_prepare_ot.mojo
#
# ANIMA OT prepare — produce a REAL cached training sample for train_anima_ot.
#
# Chunk D wire 1 (TRAINING_PLAN_anima_OT.md §D). This replaces the SYNTHETIC
# latent the smoke cache carried with a REAL VAE-encoded latent from a real image
# on disk, in the exact schema the trainer reads:
#
#   latent : F32 [1,16,LH,LW]   RAW Qwen-Image VAE mean latent (NO scale_latents
#            here — the trainer applies scale_latents per OT delta-2; we cache the
#            raw encoder mean, the same thing the OT pipeline hands the setup).
#
# PATH:
#   real PNG (256x256, RGB) -> [-1,1] NCHW [1,3,256,256]
#     -> QwenImageVaeEncoder[256,256].encode_mean -> NCHW [1,16,32,32]  (B, encode_mean)
#     -> save key `latent` [1,16,32,32]  (4D; the trainer crops to LATENT_HW)
#
# The lift to the OT 5D [1,16,1,LH,LW] (vae_frame_dim) is conceptual: the trainer
# treats the latent as a single video frame T=1 (S_IMG = (LH/2)*(LW/2)); the cache
# stores the 2D-spatial NCHW the trainer's channels-last cropper consumes. The
# encode itself IS the single-frame (T=1) image encode (see qwenimage_encoder.mojo
# header), so this latent is the genuine vae_frame_dim=True latent flattened to
# NCHW.
#
# CONTEXT: the frozen 512-token cross-attn context is supplied to the trainer
# separately (its CONTEXT_PATH). The real Anima LLM-adapter output sidecar
# (anima_embeddings.safetensors, key context_cond [1,256,1024]) is a REAL caption
# context; the trainer zero-pads it 256->512 (AnimaModel.py:229 dense-zero pad).
# Running the full 512 text path (Chunk C) needs an HF-tokenized id sidecar that
# is not on disk; the captured real context is the data we have and is faithful.
#
# Run (SEPARATE command, after build):
#   cd /home/alex/mojodiffusion
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build -I . -Xlinker -lm \
#       serenitymojo/pipeline/anima_prepare_ot.mojo -o /tmp/anima_prepare_ot
#   /tmp/anima_prepare_ot <image.png> <out_cache_dir>

from std.sys import argv
from std.collections import List
from std.memory import ArcPointer
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.io.ffi import sys_system
from serenitymojo.image.decode import decode_image
from serenitymojo.models.vae.qwenimage_encoder import QwenImageVaeEncoder

comptime VAE_FILE = (
    "/home/alex/.serenity/models/anima/split_files/vae/qwen_image_vae.safetensors"
)
comptime IH = 512
comptime IW = 512
comptime LH = IH // 8   # 64
comptime LW = IW // 8   # 64


# DecodedImage HWC RGB UInt8 -> NCHW [1,3,IH,IW] F32 in [-1,1]  (v = b/127.5 - 1).
def _image_to_nchw_m1p1(
    rgb: List[UInt8], w: Int, h: Int, ctx: DeviceContext
) raises -> Tensor:
    if w != IW or h != IH:
        raise Error(
            String("anima_prepare_ot: image must be ") + String(IW) + "x"
            + String(IH) + " (got " + String(w) + "x" + String(h)
            + "); resize the PNG before prepare"
        )
    var vals = List[Float32]()
    vals.resize(3 * h * w, Float32(0.0))
    for y in range(h):
        for x in range(w):
            for c in range(3):
                var b = Int(rgb[(y * w + x) * 3 + c])
                var v = Float32(b) / Float32(127.5) - Float32(1.0)
                # NCHW dst: ((0*3 + c)*h + y)*w + x
                vals[(c * h + y) * w + x] = v
    var sh = List[Int]()
    sh.append(1); sh.append(3); sh.append(h); sh.append(w)
    # encoder accepts BF16 input (mirror the roundtrip parity); cast to BF16.
    return Tensor.from_host(vals, sh^, STDtype.BF16, ctx)


def main() raises:
    var ctx = DeviceContext()
    var args = argv()
    var img_path = String(
        "/home/alex/EriDiffusion/EriDiffusion-v2/cache/anima_real_ot/test_256.png"
    )
    var out_dir = String("/home/alex/EriDiffusion/EriDiffusion-v2/cache/anima_real_ot")
    if len(args) > 1:
        img_path = String(args[1])
    if len(args) > 2:
        out_dir = String(args[2])

    print("==== anima_prepare_ot — REAL VAE encode for the OT trainer ====")
    print("image:", img_path)
    print("out cache dir:", out_dir)
    print("VAE:", VAE_FILE)

    # 1) decode the real image (RGB, HWC UInt8).
    var dec = decode_image(img_path)
    print("decoded image", dec.width, "x", dec.height, "RGB")

    # 2) -> [-1,1] NCHW [1,3,256,256] BF16
    var img = _image_to_nchw_m1p1(dec.rgb, dec.width, dec.height, ctx)

    # 3) Qwen-Image VAE encode_mean -> NCHW [1,16,32,32] (RAW mean latent).
    var enc = QwenImageVaeEncoder[IH, IW].load(VAE_FILE, ctx)
    print("encoder loaded; encoding...")
    var lat = enc.encode_mean(img, ctx)   # NCHW [1,16,LH,LW]
    var lsh = lat.shape()
    print("latent shape = [", lsh[0], ",", lsh[1], ",", lsh[2], ",", lsh[3], "]")
    if len(lsh) != 4 or lsh[1] != 16 or lsh[2] != LH or lsh[3] != LW:
        raise Error("encode_mean produced wrong latent shape")

    # raw latent receipt (finiteness + per-channel std sanity, BEFORE scale_latents)
    var lh = lat.to_host(ctx)
    var n = len(lh)
    var allfin = True
    var smean = Float64(0.0)
    for i in range(n):
        var v = lh[i]
        if not (v == v):
            allfin = False
        smean += Float64(v)
    smean /= Float64(n)
    var svar = Float64(0.0)
    for i in range(n):
        var d = Float64(lh[i]) - smean
        svar += d * d
    svar /= Float64(n)
    print("raw latent: finite=", allfin, " mean=", smean,
          " std=", svar ** 0.5, " numel=", n)
    if not allfin:
        raise Error("encoded latent contains non-finite values")

    # 4) save key `latent` [1,16,32,32] BF16 into <out_dir>/sample0.safetensors
    _ = sys_system(String("mkdir -p ") + out_dir)
    var lat_bf16 = Tensor.from_host(lh.copy(), [1, 16, LH, LW], STDtype.BF16, ctx)
    var names = List[String]()
    names.append(String("latent"))
    var tensors = List[ArcPointer[Tensor]]()
    tensors.append(ArcPointer(lat_bf16^))
    var out_path = out_dir + String("/sample0.safetensors")
    save_safetensors(names, tensors, out_path, ctx)
    print("wrote REAL latent [1,16,", LH, ",", LW, "] ->", out_path)
    print("VERDICT: real cache ready — point train_anima_ot CACHE_DIR at", out_dir)

# serenitymojo/pipeline/giger3_prepare.mojo
#
# GIGER3 (gigerver3) LoRA dataset prep — PURE-MOJO cache builder for the Anima
# OT trainer. Produces one cache file per image, each holding BOTH the trainer's
# keys:
#
#   latent       : F32 [1,16,64,64]    RAW Qwen-Image VAE mean latent (NO
#                  scale_latents — train_anima_ot applies it; we cache the raw
#                  encoder mean, std ~1.0, the same thing anima_prepare_ot512 caches)
#   context_cond : F32 [1,512,1024]    FROZEN Anima cross-attn context (Qwen3-0.6B
#                  -> zero-pad -> net.llm_adapter), std ~0.045 (our-model scale)
#
# This REPLACES the incompatible Rust/EDv2 cache
# (/home/alex/EriDiffusion/EriDiffusion-v2/cache/gigerver3_anima_512) whose
# latents were scale_latents-applied (std ~0.45) and whose text embeddings used a
# different normalization (std ~11). Both halves here are pure-Mojo compute that
# already passed parity (VAE encode_mean cos 0.9999; adapter parity gate).
#
# DEV-TIME sidecar (stated deviation, same as anima_text_context): the two HF
# tokenizers (Qwen2 for Qwen3 ids, T5TokenizerFast for adapter query ids) are not
# ported to Mojo. parity/giger3_preprocess.py resizes each image to a 512x512 PNG
# and writes a per-image token sidecar (<id>_tokens.safetensors). This binary
# consumes those; ALL latent/context compute is Mojo.
#
# Run (after build + giger3_preprocess.py):
#   cd /home/alex/mojodiffusion
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda -Xlinker -lpng16 \
#       -Xlinker -lturbojpeg \
#       serenitymojo/pipeline/giger3_prepare.mojo -o /tmp/giger3_prepare
#   /tmp/giger3_prepare <pre_dir> <out_cache_dir>
#     pre_dir       = output/giger3_pre   (PNGs + token sidecars + ids.txt)
#     out_cache_dir = output/giger3_cache (70 <id>.safetensors written here)

from std.sys import argv
from std.collections import List
from std.memory import ArcPointer
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.ffi import (
    sys_system, sys_open, sys_close, sys_pread, file_size, O_RDONLY,
)
from std.memory import alloc
from serenitymojo.image.decode import decode_image
from serenitymojo.models.vae.qwenimage_encoder import QwenImageVaeEncoder
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.models.text_encoder.qwen3_encoder import Qwen3Config, Qwen3Encoder
from serenitymojo.models.anima.anima_text_context import (
    AnimaAdapterWeights, anima_llm_adapter_forward, zero_pad_positions_f32,
)

comptime QWEN3_DIR = (
    "/home/alex/.serenity/models/anima/split_files/text_encoders/"
    "qwen_3_06b_base.safetensors"
)
comptime CKPT = (
    "/home/alex/.serenity/models/anima/split_files/diffusion_models/"
    "anima-base-v1.0.safetensors"
)


# Full OT text path -> FROZEN context [1,512,1024] (F32). Forward only.
# (inlined from pipeline/anima_text_context.mojo, which can't be imported while
# building inside the pipeline/ dir.)
def _anima_text_context_from_tokens(
    qwen_ids: List[Int],
    qwen_mask: List[Int],
    t5_ids: List[Int],
    ctx: DeviceContext,
) raises -> Tensor:
    var enc = Qwen3Encoder.load(QWEN3_DIR, Qwen3Config.qwen3_06b(), ctx)
    var n_layers = enc.config.num_layers
    var pre_norm = enc.encode(qwen_ids, n_layers - 1, ctx)
    var last_hidden = enc.final_norm(pre_norm, ctx)
    var hidden_f32 = cast_tensor(last_hidden, STDtype.F32, ctx)
    var hidden_zeroed = zero_pad_positions_f32(hidden_f32, qwen_mask, ctx)
    var wts = AnimaAdapterWeights.load_checkpoint(CKPT, ctx)
    return anima_llm_adapter_forward(t5_ids, hidden_zeroed, qwen_mask, wts, ctx)

comptime VAE_FILE = (
    "/home/alex/.serenity/models/anima/split_files/vae/qwen_image_vae.safetensors"
)
comptime IH = 512
comptime IW = 512
comptime LH = IH // 8   # 64
comptime LW = IW // 8   # 64
comptime S_TXT = 512
comptime DIM = 1024


# DecodedImage HWC RGB UInt8 -> NCHW [1,3,IH,IW] BF16 in [-1,1]  (v = b/127.5 - 1).
def _image_to_nchw_m1p1(
    rgb: List[UInt8], w: Int, h: Int, ctx: DeviceContext
) raises -> Tensor:
    if w != IW or h != IH:
        raise Error(
            String("giger3_prepare: image must be ") + String(IW) + "x"
            + String(IH) + " (got " + String(w) + "x" + String(h)
            + "); run giger3_preprocess.py first"
        )
    var vals = List[Float32]()
    vals.resize(3 * h * w, Float32(0.0))
    for y in range(h):
        for x in range(w):
            for c in range(3):
                var b = Int(rgb[(y * w + x) * 3 + c])
                var v = Float32(b) / Float32(127.5) - Float32(1.0)
                vals[(c * h + y) * w + x] = v
    var sh = List[Int]()
    sh.append(1); sh.append(3); sh.append(h); sh.append(w)
    return Tensor.from_host(vals, sh^, STDtype.BF16, ctx)


def _read_ids(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> List[Int]:
    """Read tokenizer sidecar ids to host ints.

    F32 widening is only for decoding the sidecar numeric array; no model tensor
    boundary stores these ids as F32."""
    var t = Tensor.from_view_as_f32(st.tensor_view(name), ctx)
    var host = t.to_host(ctx)
    var out = List[Int]()
    for i in range(len(host)):
        out.append(Int(host[i]))
    return out^


def _std_of(host: List[Float32]) -> Float64:
    var n = len(host)
    if n == 0:
        return 0.0
    var smean = Float64(0.0)
    for i in range(n):
        smean += Float64(host[i])
    smean /= Float64(n)
    var svar = Float64(0.0)
    for i in range(n):
        var d = Float64(host[i]) - smean
        svar += d * d
    svar /= Float64(n)
    return svar ** 0.5


def _read_ids_file(path: String) raises -> List[String]:
    # ids.txt: one stem per line. Read via io/ffi pread (NEVER stdlib open —
    # MAP.md: it collides with ffi's external_call["open"]).
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open ids manifest: ") + path)
    var n = file_size(fd)
    if n <= 0:
        _ = sys_close(fd)
        raise Error("empty ids manifest")
    var buf = alloc[UInt8](n)
    var done = 0
    while done < n:
        var got = sys_pread(fd, buf + done, n - done, done)
        if got <= 0:
            break
        done += got
    _ = sys_close(fd)
    var out = List[String]()
    var cur = String("")
    for i in range(done):
        var b = Int(buf[i])
        if b == 10:  # '\n'
            if len(cur) > 0:
                out.append(cur)
            cur = String("")
        elif b != 13:  # skip '\r'
            cur += chr(b)
    if len(cur) > 0:
        out.append(cur)
    buf.free()
    return out^


def main() raises:
    var ctx = DeviceContext()
    var args = argv()
    var pre_dir = String("/home/alex/mojodiffusion/output/giger3_pre")
    var out_dir = String("/home/alex/mojodiffusion/output/giger3_cache")
    if len(args) > 1:
        pre_dir = String(args[1])
    if len(args) > 2:
        out_dir = String(args[2])

    print("==== giger3_prepare — PURE-MOJO cache for the Anima OT trainer ====")
    print("pre dir   :", pre_dir)
    print("out cache :", out_dir)
    print("VAE       :", VAE_FILE)
    _ = sys_system(String("mkdir -p ") + out_dir)

    var ids = _read_ids_file(pre_dir + String("/ids.txt"))
    print("ids in manifest:", len(ids))

    # Load the VAE encoder once; reuse across all images.
    var enc = QwenImageVaeEncoder[IH, IW].load(VAE_FILE, ctx)
    print("VAE encoder loaded.")

    var written = 0
    var failed = List[String]()
    var lat_std_sum = Float64(0.0)
    var ctx_std_sum = Float64(0.0)
    var n_reports = 0

    for ref id in ids:
        var png_path = pre_dir + String("/") + id + String(".png")
        var tok_path = pre_dir + String("/") + id + String("_tokens.safetensors")
        var out_path = out_dir + String("/") + id + String(".safetensors")

        # --- 1) VAE encode: PNG -> RAW latent [1,16,64,64] ---
        var dec = decode_image(png_path)
        if dec.width != IW or dec.height != IH:
            print("FAIL", id, ": decoded", dec.width, "x", dec.height)
            failed.append(id)
            continue
        var img = _image_to_nchw_m1p1(dec.rgb, dec.width, dec.height, ctx)
        var lat = enc.encode_mean(img, ctx)   # NCHW [1,16,LH,LW]
        var lsh = lat.shape()
        if len(lsh) != 4 or lsh[1] != 16 or lsh[2] != LH or lsh[3] != LW:
            print("FAIL", id, ": latent shape wrong")
            failed.append(id)
            continue
        var lh = lat.to_host(ctx)
        var lat_allfin = True
        for i in range(len(lh)):
            if not (lh[i] == lh[i]):
                lat_allfin = False
        if not lat_allfin:
            print("FAIL", id, ": non-finite latent")
            failed.append(id)
            continue

        # --- 2) text path: tokens sidecar -> context_cond [1,512,1024] ---
        var tst = ShardedSafeTensors.open(tok_path)
        var qwen_ids = _read_ids(tst, String("qwen_input_ids"), ctx)
        var qwen_mask = _read_ids(tst, String("qwen_attention_mask"), ctx)
        var t5_ids = _read_ids(tst, String("t5_input_ids"), ctx)
        if len(qwen_ids) != S_TXT or len(qwen_mask) != S_TXT or len(t5_ids) != S_TXT:
            print("FAIL", id, ": token arrays not length 512")
            failed.append(id)
            continue
        var context = _anima_text_context_from_tokens(
            qwen_ids, qwen_mask, t5_ids, ctx
        )
        var cs = context.shape()
        if len(cs) != 3 or cs[0] != 1 or cs[1] != S_TXT or cs[2] != DIM:
            print("FAIL", id, ": context shape != [1,512,1024]")
            failed.append(id)
            continue
        var ch = context.to_host(ctx)
        var ctx_allfin = True
        for i in range(len(ch)):
            if not (ch[i] == ch[i]):
                ctx_allfin = False
        if not ctx_allfin:
            print("FAIL", id, ": non-finite context")
            failed.append(id)
            continue

        # --- 3) write <id>.safetensors with BOTH keys ---
        var lat_bf16 = Tensor.from_host(lh.copy(), [1, 16, LH, LW], STDtype.BF16, ctx)
        var names = List[String]()
        names.append(String("latent"))
        names.append(String("context_cond"))
        var tensors = List[ArcPointer[Tensor]]()
        tensors.append(ArcPointer(lat_bf16^))
        tensors.append(ArcPointer(context^))
        save_safetensors(names, tensors, out_path, ctx)
        written += 1

        # gate receipts on the first 3 + a couple spot checks
        if n_reports < 5:
            var ls = _std_of(lh)
            var cstd = _std_of(ch)
            lat_std_sum += ls
            ctx_std_sum += cstd
            n_reports += 1
            print("  ", id, "-> latent std =", ls, " context std =", cstd,
                  " (", written, "/", len(ids), ")")
        else:
            print("  ", id, "ok (", written, "/", len(ids), ")")

    print("==================================================================")
    print("WROTE", written, "cache files ->", out_dir)
    print("FAILED:", len(failed))
    for ref f in failed:
        print("   failed id:", f)
    if n_reports > 0:
        print("GATE (first", n_reports, "samples): mean latent std =",
              lat_std_sum / Float64(n_reports),
              " mean context std =", ctx_std_sum / Float64(n_reports))
        print("  expect latent std ~1.0 (RAW, NOT 0.45) and context std ~0.045"
              " (our-model, NOT Rust 11.0)")

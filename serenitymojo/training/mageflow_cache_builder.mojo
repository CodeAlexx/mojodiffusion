# serenitymojo/training/mageflow_cache_builder.mojo — Mage-Flow REAL data cache
# builder. Pure Mojo + MAX, GPU, 16GB-offload-staged exactly like
# pipeline/mageflow_pipeline.mojo: the two heavyweight encoders NEVER co-reside.
#
# PRODUCTION TARGET (eri2 LoRA run): ALL images of /home/alex/eri2_with_trigger
# (.jpg + .png; 118 samples, trigger `vrtlEri2`) at 512x512. Captions: each
# <stem>.txt holds a JSON object — the caption is its `high_level_description`
# string (verified: all 118 .txt parse, all contain the key; the sibling .json
# files carry the same high_level_description). A plain-text .txt (no leading
# '{') is used as-is (fallback).
#
#   STAGE A  load Qwen3-VL text encoder (Base text_encoder, 8.9 GB) ONCE ->
#            for each sample: read <stem>.txt, extract high_level_description,
#            tokenize with the mage-flow t2i template (pipeline mageflow_tokenize,
#            Qwen3Tokenizer), encode_mageflow_text (drop_idx 34) ->
#            hold [keep,2560] F32 on HOST -> encoder freed on def return.
#   STAGE B  MageVAE (Base vae; weights loaded per mageflow_encode call, freed
#            per return) -> for each sample: decode jpg/png, bicubic cover-resize
#            + center-crop to 512^2 (PIL-faithful _cover_resize_crop, the same
#            resample kernels mageflow_load_image_nchw uses), normalize
#            [-1,1] NCHW -> mageflow_encode[512,512] -> latent [1,128,32,32]
#            F32 on HOST. Per-sample latent mean/std printed (sanity band
#            ~0.4-1.2; the MageVAE parity fixture latent std was ~0.77).
#   STAGE C  klein_dataset.write_sample(latent, text_embedding, text_mask) ->
#            /home/alex/.serenity/mageflow_cache/eri2_with_trigger_512/
#            <stem>.safetensors (the KleinCache layout train_mageflow_real
#            .mojo's MAGEFLOW_DATA_CACHE reader consumes; all-F32, klein
#            convention).
#
# GATED math reused, NOT modified: mageflow_encode (models/vae/mageflow_vae),
# encode_mageflow_text + load_krea2_qwen3vl_4b (models/text_encoder/
# mageflow_qwen3vl), mageflow_tokenize (pipeline/mageflow_pipeline).
#
# Build/run (sm_120, 16GB):
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg
#   pixi run mojo build --target-accelerator sm_120 -I . -Xlinker -lm \
#     -Xlinker -L/usr/lib/x86_64-linux-gnu -Xlinker -lcuda \
#     serenitymojo/training/mageflow_cache_builder.mojo -o /tmp/mageflow_cache_builder
#   LD_LIBRARY_PATH=serenitymojo/ops/cshim/lib /tmp/mageflow_cache_builder
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.collections import List
from std.gpu.host import DeviceContext
from std.math import sqrt
from std.memory import alloc
from std.os import listdir

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.ffi import (
    sys_system, sys_open, sys_close, sys_pread, BytePtr, O_RDONLY,
)
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.image.decode import decode_image
from serenitymojo.models.lingbotvideo.lingbot_image_preprocess import (
    _cover_resize_crop,
)
from serenitymojo.models.vae.mageflow_vae import mageflow_encode
from serenitymojo.models.text_encoder.mageflow_qwen3vl import (
    encode_mageflow_text,
    load_krea2_qwen3vl_4b,
    MAGEFLOW_T2I_DROP_IDX,
)
from serenitymojo.pipeline.mageflow_pipeline import mageflow_tokenize
from serenitymojo.training.klein_dataset import write_sample


# ── paths / geometry ─────────────────────────────────────────────────────────
comptime MF_BASE = "/home/alex/.serenity/models/checkpoints/Mage-Flow-Base"
comptime MF_TE_DIR = MF_BASE + "/text_encoder"
comptime MF_TOK_JSON = MF_TE_DIR + "/tokenizer.json"
comptime MF_VAE = MF_BASE + "/vae/diffusion_pytorch_model.safetensors"
comptime DATASET_DIR = "/home/alex/eri2_with_trigger"
comptime CACHE_DIR = "/home/alex/.serenity/mageflow_cache/eri2_with_trigger_512"
comptime IH = 512           # image side -> latent 32x32 (512^2 training grid)
comptime IW = 512
comptime SH = IH // 16      # 32
comptime SW = IW // 16      # 32
comptime TXT_CH = 2560
comptime LAT_N = 128 * SH * SW


# ── dataset enumeration: ALL sorted .jpg/.png image files ────────────────────
def _sort_strings(mut xs: List[String]):
    for i in range(1, len(xs)):
        var key = xs[i]
        var j = i - 1
        while j >= 0 and xs[j] > key:
            xs[j + 1] = xs[j]
            j -= 1
        xs[j + 1] = key


# Returns the sorted image FILENAMES (extension kept: .jpg or .png). The stem
# (filename minus extension) addresses the sibling <stem>.txt caption and the
# cache output <stem>.safetensors.
def _all_image_files(dir: String) raises -> List[String]:
    var raw = listdir(dir)
    var files = List[String]()
    for i in range(len(raw)):
        if raw[i].endswith(".jpg") or raw[i].endswith(".png"):
            files.append(String(raw[i]))
    _sort_strings(files)
    if len(files) == 0:
        raise Error(String("no .jpg/.png images in ") + dir)
    return files^


def _stem_of(filename: String) -> String:
    if filename.endswith(".jpg"):
        return String(filename.removesuffix(".jpg"))
    return String(filename.removesuffix(".png"))


# ── caption IO (io/ffi sys_pread idiom, klein_prepare verbatim) ──────────────
def _read_caption(path: String) raises -> String:
    var fd = sys_open(path, O_RDONLY, Int32(0))
    if fd < 0:
        raise Error(String("caption not found: ") + path)
    var bytes = List[UInt8]()
    comptime CHUNK = 65536
    var buf = alloc[UInt8](CHUNK)
    var offset = 0
    while True:
        var n = sys_pread(fd, BytePtr(unsafe_from_address=Int(buf)), CHUNK, offset)
        if n < 0:
            buf.free()
            _ = sys_close(fd)
            raise Error(String("caption read error: ") + path)
        if n == 0:
            break
        for i in range(n):
            bytes.append(buf[i])
        offset += n
        if n < CHUNK:
            break
    buf.free()
    _ = sys_close(fd)
    # strip trailing newline/CR (klein_prepare convention)
    while len(bytes) > 0 and (bytes[len(bytes) - 1] == 10 or bytes[len(bytes) - 1] == 13):
        _ = bytes.pop()
    return String(unsafe_from_utf8=bytes)


# ── caption extraction: JSON `high_level_description` value ──────────────────
# The eri2_with_trigger .txt files each hold a JSON object; the caption is the
# `high_level_description` string. Byte-level scan (no JSON dependency):
# find the key, skip to the ':', take the following quoted string with escape
# handling (\" \\ \/ \n \t \r). Plain-text files (no leading '{') pass through
# unchanged; a JSON file WITHOUT the key fails loud.
def _extract_hld_caption(raw: String, path: String) raises -> String:
    var b = raw.as_bytes()
    var n = len(b)
    # leading whitespace then '{' ? else: plain-text caption, use as-is.
    var p = 0
    while p < n and (b[p] == 32 or b[p] == 9 or b[p] == 10 or b[p] == 13):
        p += 1
    if p >= n or b[p] != UInt8(123):   # '{'
        return raw
    # find "high_level_description"
    var key = String("\"high_level_description\"")
    var kb = key.as_bytes()
    var kn = len(kb)
    var kpos = -1
    for i in range(n - kn + 1):
        var hit = True
        for j in range(kn):
            if b[i + j] != kb[j]:
                hit = False
                break
        if hit:
            kpos = i
            break
    if kpos < 0:
        raise Error(String("caption JSON missing high_level_description: ") + path)
    var q = kpos + kn
    while q < n and b[q] != UInt8(34):   # skip past ':' to opening '"'
        q += 1
    if q >= n:
        raise Error(String("caption JSON malformed after key: ") + path)
    q += 1
    var out_bytes = List[UInt8]()
    while q < n:
        var c = b[q]
        if c == UInt8(92):   # backslash escape
            if q + 1 >= n:
                raise Error(String("caption JSON truncated escape: ") + path)
            var e = b[q + 1]
            if e == UInt8(110):
                out_bytes.append(10)     # \n
            elif e == UInt8(116):
                out_bytes.append(9)      # \t
            elif e == UInt8(114):
                out_bytes.append(13)     # \r
            else:
                out_bytes.append(e)      # \" \\ \/ and anything else literal
            q += 2
            continue
        if c == UInt8(34):   # closing '"'
            return String(unsafe_from_utf8=out_bytes)
        out_bytes.append(c)
        q += 1
    raise Error(String("caption JSON unterminated string: ") + path)


# ── host stats ───────────────────────────────────────────────────────────────
def _mean_std(vals: List[Float32]) -> Tuple[Float64, Float64]:
    var n = len(vals)
    var s = 0.0
    var s2 = 0.0
    for i in range(n):
        var v = Float64(vals[i])
        s += v
        s2 += v * v
    var m = s / Float64(n)
    var vv = s2 / Float64(n) - m * m
    if vv < 0.0:
        vv = 0.0
    return (m, sqrt(vv))


# ── STAGE A: text conditioning (Qwen3-VL freed on return) ────────────────────
# Per sample: <stem>.txt -> high_level_description caption -> mage t2i template
# tokens -> encode_mageflow_text (drop 34) -> [1,keep,2560] BF16 -> F32 host
# row-major [keep*2560].
def _stage_a_texts(
    files: List[String],
    ctx: DeviceContext,
    mut texts: List[List[Float32]],
    mut keeps: List[Int],
) raises:
    print("[cache] STAGE A: loading Qwen3-VL text encoder (Base) ...")
    var enc = load_krea2_qwen3vl_4b(String(MF_TE_DIR), ctx)
    for i in range(len(files)):
        var stem = _stem_of(files[i])
        var cap_path = String(DATASET_DIR) + "/" + stem + ".txt"
        var cap_raw = _read_caption(cap_path)
        var cap = _extract_hld_caption(cap_raw, cap_path)
        var ids = mageflow_tokenize(String(MF_TOK_JSON), cap)
        var l_full = len(ids)
        var txt = encode_mageflow_text(enc, ids, MAGEFLOW_T2I_DROP_IDX, ctx)
        var txt_f32 = cast_tensor(txt, STDtype.F32, ctx)
        var host = txt_f32.to_host(ctx)
        var keep = l_full - MAGEFLOW_T2I_DROP_IDX
        var ms = _mean_std(host)
        print(
            "[cache]   sample", i, "(", stem, ") cap_chars=", cap.byte_length(),
            " L_full=", l_full, " keep=", keep,
            " txt mean=", Float32(ms[0]), " std=", Float32(ms[1]),
        )
        print("[cache]     caption='", cap, "'")
        texts.append(host^)
        keeps.append(keep)
    print("[cache] STAGE A done (encoder freed on return)")


# ── STAGE B: MageVAE latents (weights loaded/freed per encode call) ──────────
# jpg/png -> bicubic cover-resize + center-crop 512^2 -> [-1,1] NCHW ->
# mageflow_encode[512,512] -> [1,128,32,32] F32 host.
def _load_image_512(path: String, ctx: DeviceContext) raises -> Tensor:
    var img = decode_image(path, True)          # RGB, drop alpha
    var crop = _cover_resize_crop(img, IH, IW)  # [IH,IW,3] f64 in [0,255]
    var vals = List[Float32]()
    vals.resize(3 * IH * IW, Float32(0.0))
    for c in range(3):
        for y in range(IH):
            for x in range(IW):
                var v = crop[(y * IW + x) * 3 + c] / 127.5 - 1.0
                vals[(c * IH + y) * IW + x] = Float32(v)
    return Tensor.from_host(vals, [1, 3, IH, IW], STDtype.F32, ctx)


def _stage_b_latents(
    files: List[String],
    ctx: DeviceContext,
    mut lats: List[List[Float32]],
) raises:
    print("[cache] STAGE B: MageVAE encode (Base vae) ...")
    var st = ShardedSafeTensors.open(String(MF_VAE))
    for i in range(len(files)):
        var stem = _stem_of(files[i])
        var img_path = String(DATASET_DIR) + "/" + files[i]
        var img = _load_image_512(img_path, ctx)
        var lat = mageflow_encode[IH, IW](img, st, ctx)   # [1,128,32,32]
        var ls = lat.shape()
        if len(ls) != 4 or ls[1] != 128 or ls[2] != SH or ls[3] != SW:
            raise Error(
                String("latent shape wrong for ") + stem
                + " (expect [1,128,32,32])"
            )
        var lat_f32 = lat.clone(ctx)
        if lat_f32.dtype() != STDtype.F32:
            lat_f32 = cast_tensor(lat_f32, STDtype.F32, ctx)
        var host = lat_f32.to_host(ctx)
        var ms = _mean_std(host)
        print(
            "[cache]   sample", i, "(", stem, ") latent [1,128,32,32]",
            " mean=", Float32(ms[0]), " std=", Float32(ms[1]),
        )
        # sanity band ~0.4-1.2 (MageVAE parity fixture latent std ~0.77). FLAG
        # (do not abort) if outside — a scramble shows up as std >> 1 or ~0.
        if ms[1] < 0.40 or ms[1] > 1.20:
            print(
                "[cache]   WARNING: latent std", Float32(ms[1]),
                "outside sanity band [0.40,1.20] (fixture ~0.77) — inspect!"
            )
        lats.append(host^)
    print("[cache] STAGE B done (VAE freed)")


def main() raises:
    var ctx = DeviceContext()
    print("=== Mage-Flow REAL data cache builder (pure Mojo, staged 16GB) ===")
    var files = _all_image_files(String(DATASET_DIR))
    print("  dataset:", DATASET_DIR, " -> cache:", CACHE_DIR,
          " samples:", len(files), " @", IH, "x", IW)

    # STAGE A: text (encoder loads+frees inside)
    var texts = List[List[Float32]]()
    var keeps = List[Int]()
    _stage_a_texts(files, ctx, texts, keeps)

    # STAGE B: latents (VAE loads+frees inside)
    var lats = List[List[Float32]]()
    _stage_b_latents(files, ctx, lats)

    # STAGE C: write klein_dataset cache samples
    print("[cache] STAGE C: write_sample ->", CACHE_DIR)
    _ = sys_system(String("mkdir -p '") + String(CACHE_DIR) + String("'"))
    _ = sys_system(
        String("rm -f '") + String(CACHE_DIR) + String("'/*.safetensors")
    )
    for i in range(len(files)):
        var stem = _stem_of(files[i])
        var keep = keeps[i]
        var latent = Tensor.from_host(
            lats[i].copy(), [1, 128, SH, SW], STDtype.F32, ctx
        )
        var txt = Tensor.from_host(
            texts[i].copy(), [1, keep, TXT_CH], STDtype.F32, ctx
        )
        var ones = List[Float32]()
        for _ in range(keep):
            ones.append(1.0)
        var mask = Tensor.from_host(ones^, [1, keep], STDtype.F32, ctx)
        var out_path = String(CACHE_DIR) + "/" + stem + ".safetensors"
        write_sample(latent, txt, mask, out_path, ctx)
        print("[cache]   wrote", out_path, " (latent[1,128,32,32] txt[1,", keep, ",2560])")

    print("")
    print("PASS: wrote", len(files), "cache samples ->", CACHE_DIR)
    print("      run: MAGEFLOW_DATA_CACHE=" + String(CACHE_DIR),
          "<train_mageflow_real binary>")

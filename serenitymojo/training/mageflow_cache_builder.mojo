# serenitymojo/training/mageflow_cache_builder.mojo — Mage-Flow REAL data cache
# builder (chunk 5). Pure Mojo + MAX, GPU, 16GB-offload-staged exactly like
# pipeline/mageflow_pipeline.mojo: the two heavyweight encoders NEVER co-reside.
#
#   STAGE A  load Qwen3-VL text encoder (Base text_encoder, 8.9 GB) ONCE ->
#            for each of the first NUM 40_woman samples: read the caption .txt,
#            tokenize with the mage-flow t2i template (pipeline mageflow_tokenize,
#            Qwen3Tokenizer), encode_mageflow_text (drop_idx 34) ->
#            hold [keep,2560] F32 on HOST -> encoder freed on def return.
#   STAGE B  MageVAE (Base vae; weights loaded per mageflow_encode call, freed
#            per return) -> for each sample: decode jpg, bicubic cover-resize +
#            center-crop to 256^2 (PIL-faithful _cover_resize_crop, the same
#            resample kernels mageflow_load_image_nchw uses), normalize
#            [-1,1] NCHW -> mageflow_encode[256,256] -> latent [1,128,16,16]
#            F32 on HOST. Per-sample latent mean/std printed (sanity: the
#            MageVAE parity fixture latent std was ~0.77 — wildly-off = flag).
#   STAGE C  klein_dataset.write_sample(latent, text_embedding, text_mask) ->
#            /home/alex/.serenity/mageflow_cache/40_woman/<stem>.safetensors
#            (the KleinCache layout train_mageflow_real.mojo's
#            MAGEFLOW_DATA_CACHE reader consumes; all-F32, klein convention).
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
comptime DATASET_DIR = "/mnt/disk2/datasets/40_woman"
comptime CACHE_DIR = "/home/alex/.serenity/mageflow_cache/40_woman"
comptime NUM = 8            # first 8 samples (mechanism smoke)
comptime IH = 256           # image side -> latent 16x16 (trainer smoke grid)
comptime IW = 256
comptime SH = IH // 16      # 16
comptime SW = IW // 16      # 16
comptime TXT_CH = 2560
comptime LAT_N = 128 * SH * SW


# ── dataset enumeration: sorted .jpg stems, first NUM ────────────────────────
def _sort_strings(mut xs: List[String]):
    for i in range(1, len(xs)):
        var key = xs[i]
        var j = i - 1
        while j >= 0 and xs[j] > key:
            xs[j + 1] = xs[j]
            j -= 1
        xs[j + 1] = key


def _first_jpg_stems(dir: String, n: Int) raises -> List[String]:
    var raw = listdir(dir)
    var stems = List[String]()
    for i in range(len(raw)):
        if raw[i].endswith(".jpg"):
            stems.append(String(raw[i].removesuffix(".jpg")))
    _sort_strings(stems)
    if len(stems) < n:
        raise Error(
            String("dataset has only ") + String(len(stems))
            + " jpgs, need " + String(n) + " in " + dir
        )
    var out = List[String]()
    for i in range(n):
        out.append(stems[i])
    return out^


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
# Per sample: caption -> mage t2i template tokens -> encode_mageflow_text
# (drop 34) -> [1,keep,2560] BF16 -> F32 host row-major [keep*2560].
def _stage_a_texts(
    stems: List[String],
    ctx: DeviceContext,
    mut texts: List[List[Float32]],
    mut keeps: List[Int],
) raises:
    print("[cache] STAGE A: loading Qwen3-VL text encoder (Base) ...")
    var enc = load_krea2_qwen3vl_4b(String(MF_TE_DIR), ctx)
    for i in range(len(stems)):
        var cap_path = String(DATASET_DIR) + "/" + stems[i] + ".txt"
        var cap = _read_caption(cap_path)
        var ids = mageflow_tokenize(String(MF_TOK_JSON), cap)
        var l_full = len(ids)
        var txt = encode_mageflow_text(enc, ids, MAGEFLOW_T2I_DROP_IDX, ctx)
        var txt_f32 = cast_tensor(txt, STDtype.F32, ctx)
        var host = txt_f32.to_host(ctx)
        var keep = l_full - MAGEFLOW_T2I_DROP_IDX
        var ms = _mean_std(host)
        print(
            "[cache]   sample", i, "(", stems[i], ") caption='", cap,
            "' L_full=", l_full, " keep=", keep,
            " txt mean=", Float32(ms[0]), " std=", Float32(ms[1]),
        )
        texts.append(host^)
        keeps.append(keep)
    print("[cache] STAGE A done (encoder freed on return)")


# ── STAGE B: MageVAE latents (weights loaded/freed per encode call) ──────────
# jpg -> bicubic cover-resize + center-crop 256^2 -> [-1,1] NCHW ->
# mageflow_encode[256,256] -> [1,128,16,16] F32 host.
def _load_image_256(path: String, ctx: DeviceContext) raises -> Tensor:
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
    stems: List[String],
    ctx: DeviceContext,
    mut lats: List[List[Float32]],
) raises:
    print("[cache] STAGE B: MageVAE encode (Base vae) ...")
    var st = ShardedSafeTensors.open(String(MF_VAE))
    for i in range(len(stems)):
        var img_path = String(DATASET_DIR) + "/" + stems[i] + ".jpg"
        var img = _load_image_256(img_path, ctx)
        var lat = mageflow_encode[IH, IW](img, st, ctx)   # [1,128,16,16]
        var ls = lat.shape()
        if len(ls) != 4 or ls[1] != 128 or ls[2] != SH or ls[3] != SW:
            raise Error(
                String("latent shape wrong for ") + stems[i]
                + " (expect [1,128,16,16])"
            )
        var lat_f32 = lat.clone(ctx)
        if lat_f32.dtype() != STDtype.F32:
            lat_f32 = cast_tensor(lat_f32, STDtype.F32, ctx)
        var host = lat_f32.to_host(ctx)
        var ms = _mean_std(host)
        print(
            "[cache]   sample", i, "(", stems[i], ") latent [1,128,16,16]",
            " mean=", Float32(ms[0]), " std=", Float32(ms[1]),
        )
        # sanity: MageVAE latent std ~0.77 on the parity fixture. FLAG (do not
        # abort) if wildly off — a scramble shows up as std >> 1 or ~0.
        if ms[1] < 0.30 or ms[1] > 1.60:
            print(
                "[cache]   WARNING: latent std", Float32(ms[1]),
                "outside [0.30,1.60] (fixture ~0.77) — possible scramble!"
            )
        lats.append(host^)
    print("[cache] STAGE B done (VAE freed)")


def main() raises:
    var ctx = DeviceContext()
    print("=== Mage-Flow REAL data cache builder (pure Mojo, staged 16GB) ===")
    print("  dataset:", DATASET_DIR, " -> cache:", CACHE_DIR, " samples:", NUM)

    var stems = _first_jpg_stems(String(DATASET_DIR), NUM)
    for i in range(len(stems)):
        print("  [", i, "]", stems[i] + ".jpg /", stems[i] + ".txt")

    # STAGE A: text (encoder loads+frees inside)
    var texts = List[List[Float32]]()
    var keeps = List[Int]()
    _stage_a_texts(stems, ctx, texts, keeps)

    # STAGE B: latents (VAE loads+frees inside)
    var lats = List[List[Float32]]()
    _stage_b_latents(stems, ctx, lats)

    # STAGE C: write klein_dataset cache samples
    print("[cache] STAGE C: write_sample ->", CACHE_DIR)
    _ = sys_system(String("mkdir -p ") + String(CACHE_DIR))
    _ = sys_system(String("rm -f ") + String(CACHE_DIR) + String("/*.safetensors"))
    for i in range(len(stems)):
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
        var out_path = String(CACHE_DIR) + "/" + stems[i] + ".safetensors"
        write_sample(latent, txt, mask, out_path, ctx)
        print("[cache]   wrote", out_path, " (latent[1,128,16,16] txt[1,", keep, ",2560])")

    print("")
    print("PASS: wrote", NUM, "cache samples ->", CACHE_DIR)
    print("      run: MAGEFLOW_DATA_CACHE=" + String(CACHE_DIR),
          "<train_mageflow_real binary>")

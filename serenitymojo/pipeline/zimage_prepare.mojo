# pipeline/zimage_prepare.mojo — Z-Image local prepare driver, pure Mojo.
#
# This is the Z-Image equivalent of `klein_prepare_alina.mojo`, but it does not
# reuse Rust/EriDiffusion or Python caches. It consumes local staged image tensor
# files:
#
#   output/alina_zimage_stage/*.safetensors
#     image: [1,3,H,W] BF16, RGB, values in [-1,1]
#   output/alina_zimage_stage/<same stem>.txt
#
# and writes the cache consumed by `train_zimage_real.mojo`:
#
#   output/alina_zimage_cache/*.safetensors
#     latent:         [1,16,H/8,W/8]  mean VAE latent, unscaled
#     text_embedding: [1,512,2560]    Qwen3 layer-34 hidden state
#     text_mask:      [1,512]         1.0 for real tokens, 0.0 for pad
#
# IMPORTANT: raw JPEG/PNG decode is not done here. The trainer contract is still
# self-contained Mojo; raw image staging must be a Mojo image decoder/stager, not
# a Rust or Python cache producer.
#
# Run:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#     pixi run mojo run -I . serenitymojo/pipeline/zimage_prepare.mojo

from std.collections import List
from std.gpu.host import DeviceContext
from std.memory import alloc
from std.os import listdir
from std.sys import argv

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.io.ffi import (
    sys_system, sys_open, sys_close, sys_pread, BytePtr, O_RDONLY,
)
from serenitymojo.models.vae.zimage_encoder import ZImageVaeEncoder
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer
from serenitymojo.models.text_encoder.qwen3_encoder import Qwen3Encoder, Qwen3Config
from serenitymojo.training.aspect_buckets import (
    ZIMAGE_T2D_LADDER_LEN, ZIMAGE_T2D_LADDER_X100,
    zimage_t2d_lat_h, zimage_t2d_lat_w,
)
from serenitymojo.training.klein_dataset import write_sample


comptime ZROOT = "/home/alex/.serenity/models/zimage_base"
comptime VAE_DIR = ZROOT + "/vae"
comptime TEXT_ENCODER_DIR = ZROOT + "/text_encoder"
comptime TOK_JSON = ZROOT + "/tokenizer/tokenizer.json"
comptime STAGE_DIR = "/home/alex/mojodiffusion/output/alina_zimage_stage"
comptime CACHE_DIR = "/home/alex/mojodiffusion/output/alina_zimage_cache"
# T2.D dynamic aspect bucketing (opt-in `t2d` argv mode): consume the
# GENERATED-ladder stage (zimage_stage_alina.mojo t2d mode) and write a
# SEPARATE cache. Cache tensor schema is UNCHANGED (latent/text_embedding/
# text_mask per-sample keyed tensors, bucket = latent shape at peek_key);
# the stage's aspect_buckets.json manifest is copied alongside as
# documentation. T2.D FOLLOW-UP: the t2d path now covers the FULL comptime
# integer ladder (aspect_buckets.mojo ZIMAGE_T2D_*, 7 buckets incl. the
# landscape canvases + the 512x512 square) via a `comptime for` that
# instantiates ZImageVaeEncoder[LH,LW] per ladder bucket and processes the
# staged files bucket-by-bucket (ONE encoder resident at a time). Staged
# shapes outside the ladder still fail loud. The DEFAULT (non-t2d) path
# below is byte-identical to before (C13).
comptime STAGE_DIR_T2D = "/home/alex/mojodiffusion/output/alina_zimage_stage_t2d"
comptime CACHE_DIR_T2D = "/home/alex/mojodiffusion/output/alina_zimage_cache_t2d"
comptime IH_MAIN = 576
comptime IW_MAIN = 448
comptime LH_MAIN = IH_MAIN // 8
comptime LW_MAIN = IW_MAIN // 8
comptime IH_TALL = 704
comptime IW_TALL = 384
comptime LH_TALL = IH_TALL // 8
comptime LW_TALL = IW_TALL // 8
comptime SEQ = 512
comptime HIDDEN = 2560
comptime PAD_ID = 151643
comptime EXTRACT_LAYER = 34


def _sort_strings(mut xs: List[String]):
    for i in range(1, len(xs)):
        var key = xs[i]
        var j = i - 1
        while j >= 0 and xs[j] > key:
            xs[j + 1] = xs[j]
            j -= 1
        xs[j + 1] = key


def _drop_suffix(s: String, suffix_len: Int) raises -> String:
    var n = s.byte_length() - suffix_len
    if n < 0:
        raise Error("_drop_suffix: suffix longer than string")
    var src = s.as_bytes()
    var out = List[UInt8]()
    for i in range(n):
        out.append(src[i])
    return String(unsafe_from_utf8=out)


def _stage_files(stage_dir: String) raises -> List[String]:
    var raw = listdir(stage_dir)
    var fs = List[String]()
    for i in range(len(raw)):
        if raw[i].endswith(".safetensors"):
            fs.append(raw[i])
    _sort_strings(fs)
    return fs^


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
            raise Error("caption read error")
        if n == 0:
            break
        for i in range(n):
            bytes.append(buf[i])
        offset += n
        if n < CHUNK:
            break
    buf.free()
    _ = sys_close(fd)
    while len(bytes) > 0 and (bytes[len(bytes) - 1] == 10 or bytes[len(bytes) - 1] == 13):
        _ = bytes.pop()
    return String(unsafe_from_utf8=bytes)


def _load_image(path: String, ctx: DeviceContext) raises -> Tensor:
    var st = SafeTensors.open(path)
    var info = st.tensor_info(String("image"))
    var bytes = st.tensor_bytes(String("image"))
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    var image = Tensor.from_view(tv, ctx)
    var sh = image.shape()
    if len(sh) != 4 or sh[0] != 1 or sh[1] != 3:
        raise Error("zimage_prepare: staged image must be [1,3,H,W]")
    if not (
        (sh[2] == IH_MAIN and sh[3] == IW_MAIN)
        or (sh[2] == IH_TALL and sh[3] == IW_TALL)
    ):
        raise Error("zimage_prepare: staged image bucket unsupported for this Alina Z-Image run")
    if image.dtype() != STDtype.BF16:
        raise Error("zimage_prepare: staged image must be BF16; do not stage Z-Image in F32")
    return image^


def _zimage_template(prompt: String) -> String:
    return (
        String("<|im_start|>user\n")
        + prompt
        + "<|im_end|>\n<|im_start|>assistant\n"
    )


def _real_token_count(tok: Qwen3Tokenizer, prompt: String) raises -> Int:
    return len(tok.encode(_zimage_template(prompt)))


def _tokenize_512(tok: Qwen3Tokenizer, prompt: String) raises -> List[Int]:
    var ids_full = tok.encode(_zimage_template(prompt))
    if len(ids_full) > SEQ:
        raise Error("caption too long for Z-Image 512-token text cache")
    var ids = List[Int](capacity=SEQ)
    for i in range(len(ids_full)):
        ids.append(ids_full[i])
    for _ in range(SEQ - len(ids_full)):
        ids.append(PAD_ID)
    return ids^


def _mask_512(valid: Int, ctx: DeviceContext) raises -> Tensor:
    var vals = List[Float32]()
    for i in range(SEQ):
        vals.append(Float32(1.0) if i < valid else Float32(0.0))
    return Tensor.from_host(vals, [1, SEQ], STDtype.F32, ctx)


def _load_image_t2d(path: String, ctx: DeviceContext) raises -> Tensor:
    """T2.D follow-up image loader: any 64-aligned [1,3,H,W] BF16 canvas is
    accepted here; LADDER membership is enforced by the bucket-grouped
    dispatch in _run_t2d (unmatched shapes fail loud there)."""
    var st = SafeTensors.open(path)
    var info = st.tensor_info(String("image"))
    var bytes = st.tensor_bytes(String("image"))
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    var image = Tensor.from_view(tv, ctx)
    var sh = image.shape()
    if len(sh) != 4 or sh[0] != 1 or sh[1] != 3:
        raise Error("zimage_prepare t2d: staged image must be [1,3,H,W]")
    if sh[2] % 64 != 0 or sh[3] % 64 != 0:
        raise Error("zimage_prepare t2d: staged image dims must be 64-aligned")
    if image.dtype() != STDtype.BF16:
        raise Error("zimage_prepare t2d: staged image must be BF16; do not stage Z-Image in F32")
    return image^


def _staged_image_dims(path: String) raises -> Tuple[Int, Int]:
    """(H, W) of the staged `image` tensor, header-only (no pixel read)."""
    var st = SafeTensors.open(path)
    var info = st.tensor_info(String("image"))
    if len(info.shape) != 4:
        raise Error("zimage_prepare t2d: staged image must be rank-4")
    return (info.shape[2], info.shape[3])


def _encode_text_and_write(
    latent: Tensor,
    cap_path: String,
    out_path: String,
    tok: Qwen3Tokenizer,
    qenc: Qwen3Encoder,
    ctx: DeviceContext,
) raises:
    """Shared text leg: caption -> Qwen3 layer-34 embedding + mask -> cache
    sample (schema identical to the default path: latent / text_embedding /
    text_mask)."""
    var cap = _read_caption(cap_path)
    var valid = _real_token_count(tok, cap)
    var ids = _tokenize_512(tok, cap)
    var emb = qenc.encode(ids, EXTRACT_LAYER, ctx)
    var esh = emb.shape()
    if esh[1] != SEQ or esh[2] != HIDDEN:
        raise Error("zimage_prepare: text_embedding shape wrong")
    var mask = _mask_512(valid, ctx)
    write_sample(latent, emb, mask, out_path, ctx)
    var lsh = latent.shape()
    print("   tokens:", valid, " latent:", lsh[0], lsh[1], lsh[2], lsh[3], " wrote", out_path)


def _run_t2d(ctx: DeviceContext) raises:
    """T2.D follow-up prepare: full comptime-ladder coverage. Files are
    processed BUCKET-BY-BUCKET (`comptime for` over the integer ladder) so
    only one ZImageVaeEncoder[LH,LW] instantiation is GPU-resident at a time;
    cache files are independent per-sample safetensors, so the bucket-grouped
    write order does not affect the trainer (KleinCache sorts by name)."""
    var stage_dir = String(STAGE_DIR_T2D)
    var cache_dir = String(CACHE_DIR_T2D)
    print("=== Z-Image T2.D prepare: GENERATED-bucket stage -> cache ===")
    print("  stage:", stage_dir)
    print("  cache:", cache_dir)
    _ = sys_system(String("mkdir -p ") + stage_dir)
    var files = _stage_files(stage_dir)
    if len(files) == 0:
        raise Error(
            String("zimage_prepare: no staged image safetensors in ")
            + stage_dir
            + String(". Raw image decode/stage must be Mojo-owned; do not use Rust or Python caches.")
        )
    _ = sys_system(String("mkdir -p ") + cache_dir)
    _ = sys_system(String("rm -f ") + cache_dir + String("/*.safetensors"))
    # documentation manifest travels with the cache (schema v1; the trainer
    # keys buckets off latent shapes and does not read it)
    _ = sys_system(
        String("cp -f ") + stage_dir + String("/aspect_buckets.json ")
        + cache_dir + String("/aspect_buckets.json 2>/dev/null || true")
    )

    # header-only shape scan, then bucket-grouped processing
    var hs = List[Int]()
    var ws = List[Int]()
    for i in range(len(files)):
        var dims = _staged_image_dims(stage_dir + String("/") + files[i])
        hs.append(dims[0])
        ws.append(dims[1])

    print("[load] Qwen3 text encoder", TEXT_ENCODER_DIR)
    var tok = Qwen3Tokenizer(String(TOK_JSON))
    var qenc = Qwen3Encoder.load(String(TEXT_ENCODER_DIR), Qwen3Config.zimage(), ctx)

    var processed = List[Bool]()
    for _ in range(len(files)):
        processed.append(False)

    comptime for bi in range(ZIMAGE_T2D_LADDER_LEN):
        comptime X100_BI = ZIMAGE_T2D_LADDER_X100[bi]
        comptime LH_BI = zimage_t2d_lat_h(X100_BI)
        comptime LW_BI = zimage_t2d_lat_w(X100_BI)
        comptime IH_BI = 8 * LH_BI
        comptime IW_BI = 8 * LW_BI
        var n_match = 0
        for i in range(len(files)):
            if hs[i] == IH_BI and ws[i] == IW_BI:
                n_match += 1
        if n_match > 0:
            print(
                "[load] ZImageVaeEncoder lat", LH_BI, "x", LW_BI,
                " image", IH_BI, "x", IW_BI, " (", n_match, "samples )", VAE_DIR,
            )
            var vae_bi = ZImageVaeEncoder[LH_BI, LW_BI].load(String(VAE_DIR), ctx)
            for i in range(len(files)):
                if hs[i] == IH_BI and ws[i] == IW_BI:
                    var name = files[i]
                    var stem = _drop_suffix(name, String(".safetensors").byte_length())
                    print("-- sample", i + 1, "/", len(files), name,
                          " bucket lat", LH_BI, "x", LW_BI)
                    var image = _load_image_t2d(stage_dir + String("/") + name, ctx)
                    var latent = vae_bi.encode_mean(image, ctx)
                    var lsh = latent.shape()
                    if lsh[1] != 16 or lsh[2] != LH_BI or lsh[3] != LW_BI:
                        raise Error("zimage_prepare t2d: VAE latent shape wrong")
                    _encode_text_and_write(
                        latent,
                        stage_dir + String("/") + stem + String(".txt"),
                        cache_dir + String("/") + stem + String(".safetensors"),
                        tok, qenc, ctx,
                    )
                    processed[i] = True

    for i in range(len(files)):
        if not processed[i]:
            raise Error(
                String("zimage_prepare t2d: staged image ") + files[i]
                + String(" (") + String(hs[i]) + String("x") + String(ws[i])
                + String(") is outside the comptime 512px ladder — fail loud")
            )
    print("PASS: wrote", len(files), "Z-Image cache samples to", cache_dir)


def main() raises:
    var ctx = DeviceContext()
    var a = argv()
    var t2d = len(a) >= 2 and String(a[1]) == String("t2d")
    if t2d:
        _run_t2d(ctx)
        return
    var stage_dir = String(STAGE_DIR)
    var cache_dir = String(CACHE_DIR)
    print("=== Z-Image Alina prepare: local Mojo Qwen3 + VAE -> cache ===")
    print("  stage:", stage_dir)
    print("  cache:", cache_dir)
    _ = sys_system(String("mkdir -p ") + stage_dir)
    var files = _stage_files(stage_dir)
    if len(files) == 0:
        raise Error(
            String("zimage_prepare: no staged image safetensors in ")
            + stage_dir
            + String(". Raw image decode/stage must be Mojo-owned; do not use Rust or Python caches.")
        )

    _ = sys_system(String("mkdir -p ") + cache_dir)
    _ = sys_system(String("rm -f ") + cache_dir + String("/*.safetensors"))

    print("[load] ZImageVaeEncoder main", VAE_DIR)
    var vae_main = ZImageVaeEncoder[LH_MAIN, LW_MAIN].load(String(VAE_DIR), ctx)
    print("[load] ZImageVaeEncoder tall", VAE_DIR)
    var vae_tall = ZImageVaeEncoder[LH_TALL, LW_TALL].load(String(VAE_DIR), ctx)
    print("[load] Qwen3 text encoder", TEXT_ENCODER_DIR)
    var tok = Qwen3Tokenizer(String(TOK_JSON))
    var qenc = Qwen3Encoder.load(String(TEXT_ENCODER_DIR), Qwen3Config.zimage(), ctx)

    for idx in range(len(files)):
        var name = files[idx]
        var stem = _drop_suffix(name, String(".safetensors").byte_length())
        var img_path = stage_dir + String("/") + name
        var cap_path = stage_dir + String("/") + stem + String(".txt")
        var out_path = cache_dir + String("/") + stem + String(".safetensors")
        print("-- sample", idx + 1, "/", len(files), name)

        var image = _load_image(img_path, ctx)
        var ish = image.shape()
        var latent: Tensor
        if ish[2] == IH_MAIN and ish[3] == IW_MAIN:
            latent = vae_main.encode_mean(image, ctx)
        elif ish[2] == IH_TALL and ish[3] == IW_TALL:
            latent = vae_tall.encode_mean(image, ctx)
        else:
            raise Error("zimage_prepare: staged image bucket changed after validation")
        var lsh = latent.shape()
        if lsh[1] != 16 or (
            not (
                (lsh[2] == LH_MAIN and lsh[3] == LW_MAIN)
                or (lsh[2] == LH_TALL and lsh[3] == LW_TALL)
            )
        ):
            raise Error("zimage_prepare: VAE latent shape wrong")

        var cap = _read_caption(cap_path)
        var valid = _real_token_count(tok, cap)
        var ids = _tokenize_512(tok, cap)
        var emb = qenc.encode(ids, EXTRACT_LAYER, ctx)
        var esh = emb.shape()
        if esh[1] != SEQ or esh[2] != HIDDEN:
            raise Error("zimage_prepare: text_embedding shape wrong")
        var mask = _mask_512(valid, ctx)

        write_sample(latent, emb, mask, out_path, ctx)
        print("   tokens:", valid, " latent:", lsh[0], lsh[1], lsh[2], lsh[3], " wrote", out_path)

    print("PASS: wrote", len(files), "Z-Image cache samples to", cache_dir)

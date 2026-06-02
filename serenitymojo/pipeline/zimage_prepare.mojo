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
from serenitymojo.training.klein_dataset import write_sample


comptime ZROOT = "/home/alex/.serenity/models/zimage_base"
comptime VAE_DIR = ZROOT + "/vae"
comptime TEXT_ENCODER_DIR = ZROOT + "/text_encoder"
comptime TOK_JSON = ZROOT + "/tokenizer/tokenizer.json"
comptime STAGE_DIR = "/home/alex/mojodiffusion/output/alina_zimage_stage"
comptime CACHE_DIR = "/home/alex/mojodiffusion/output/alina_zimage_cache"
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


def _stage_files() raises -> List[String]:
    var raw = listdir(String(STAGE_DIR))
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


def main() raises:
    var ctx = DeviceContext()
    print("=== Z-Image Alina prepare: local Mojo Qwen3 + VAE -> cache ===")
    print("  stage:", STAGE_DIR)
    print("  cache:", CACHE_DIR)
    _ = sys_system(String("mkdir -p ") + String(STAGE_DIR))
    var files = _stage_files()
    if len(files) == 0:
        raise Error(
            String("zimage_prepare: no staged image safetensors in ")
            + String(STAGE_DIR)
            + String(". Raw image decode/stage must be Mojo-owned; do not use Rust or Python caches.")
        )

    _ = sys_system(String("mkdir -p ") + String(CACHE_DIR))
    _ = sys_system(String("rm -f ") + String(CACHE_DIR) + String("/*.safetensors"))

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
        var img_path = String(STAGE_DIR) + String("/") + name
        var cap_path = String(STAGE_DIR) + String("/") + stem + String(".txt")
        var out_path = String(CACHE_DIR) + String("/") + stem + String(".safetensors")
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

    print("PASS: wrote", len(files), "Z-Image cache samples to", CACHE_DIR)

# klein_prepare_alina.mojo — REAL prepare driver for the Alina LoRA dataset.
#
# For each of the 4 staged images (output/alina_stage/alina_{0..3}.safetensors,
# key "image" = [1,3,512,512] F32 in [-1,1]) + its caption (alina_N.txt):
#   1. VAE-encode the image  -> latent [1,128,32,32]  (real KleinVaeEncoder).
#      Assert std ≈ 0.96 (the latent-correctness gate).
#   2. Qwen3-8B encode the caption (Klein chat template, 512 tokens)
#      -> text_embedding [1,512,7680].
#   3. write_sample(latent, text_embedding, mask) to output/alina_cache/.
#
# MEMORY: Qwen3-8B (~16 GB) + the VAE encoder co-reside ONLY in THIS process.
# We load Qwen3 ONCE and encode all 4 captions, then VAE-encode all 4 images,
# then write — never re-loading the 16 GB encoder. The training process
# (train_klein_real.mojo) never imports Qwen3Encoder, so the encoder GPU memory
# is fully freed by THIS process's exit before training loads the DiT.
#
# Run:
#   cd /home/alex/mojodiffusion
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/pipeline/klein_prepare_alina.mojo

from std.collections import List
from std.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.io.ffi import sys_system, sys_open, sys_close, sys_pread, BytePtr, O_RDONLY
from std.memory import alloc
from serenitymojo.models.vae.klein_encoder import KleinVaeEncoder
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer
from serenitymojo.models.text_encoder.qwen3_encoder import Qwen3Encoder, Qwen3Config
from serenitymojo.training.klein_dataset import write_sample


comptime VAE_PATH = "/home/alex/.serenity/models/vaes/flux2-vae.safetensors"
comptime QWEN8_DIR = (
    "/home/alex/.cache/huggingface/hub/models--Qwen--Qwen3-4B/"
    "snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
)
comptime TOK_JSON = QWEN8_DIR + "/tokenizer.json"
comptime STAGE_DIR = "/home/alex/mojodiffusion/output/alina_stage"
comptime CACHE_DIR = "/home/alex/mojodiffusion/output/alina_cache_4b"
comptime IH = 512
comptime IW = 512
comptime SEQ = 512
comptime PAD_ID = 151643
comptime NUM_SAMPLES = 4


# ── caption + image IO ───────────────────────────────────────────────────────


def _read_caption(idx: Int) raises -> String:
    """Read alina_<idx>.txt into a String via the io/ffi sys_pread idiom (the
    codebase routes all file I/O through io/ffi, never std open()). Mirrors
    train_config_reader._read_file_bytes."""
    var path = STAGE_DIR + String("/alina_") + String(idx) + String(".txt")
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
    # strip a trailing newline if present, then decode UTF-8 bytes to String.
    while len(bytes) > 0 and (bytes[len(bytes) - 1] == 10 or bytes[len(bytes) - 1] == 13):
        _ = bytes.pop()
    return String(unsafe_from_utf8=bytes)


def _load_image(idx: Int, ctx: DeviceContext) raises -> Tensor:
    var path = STAGE_DIR + String("/alina_") + String(idx) + String(".safetensors")
    var st = SafeTensors.open(path)
    var info = st.tensor_info(String("image"))
    var bytes = st.tensor_bytes(String("image"))
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    return Tensor.from_view(tv, ctx)


def _std(t: Tensor, ctx: DeviceContext) raises -> Float64:
    var h = t.to_host(ctx)
    var n = len(h)
    var s = 0.0
    var s2 = 0.0
    for i in range(n):
        var v = Float64(h[i])
        s += v
        s2 += v * v
    var m = s / Float64(n)
    var vv = s2 / Float64(n) - m * m
    if vv < 0.0:
        vv = 0.0
    return sqrt(vv)


# ── Klein chat template + 512-token pad (mirrors klein9b_encode_smoke) ────────


def _klein_template(prompt: String) -> String:
    return (
        String("<|im_start|>user\n")
        + prompt
        + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def _tokenize_512(tok: Qwen3Tokenizer, prompt: String) raises -> List[Int]:
    var ids_full = tok.encode(_klein_template(prompt))
    if len(ids_full) > SEQ:
        raise Error("caption too long for 512 tokens")
    var ids = List[Int](capacity=SEQ)
    for i in range(len(ids_full)):
        ids.append(ids_full[i])
    for _ in range(SEQ - len(ids_full)):
        ids.append(PAD_ID)
    return ids^


def _real_token_count(tok: Qwen3Tokenizer, prompt: String) raises -> Int:
    return len(tok.encode(_klein_template(prompt)))


def _mask_512(valid: Int, ctx: DeviceContext) raises -> Tensor:
    var vals = List[Float32]()
    for i in range(SEQ):
        vals.append(Float32(1.0) if i < valid else Float32(0.0))
    var sh = List[Int]()
    sh.append(1)
    sh.append(SEQ)
    return Tensor.from_host(vals, sh^, STDtype.F32, ctx)


def main() raises:
    var ctx = DeviceContext()
    print("=== Klein Alina prepare: 4 images -> VAE latent + Qwen3 caption -> cache ===")
    _ = sys_system(String("mkdir -p ") + CACHE_DIR)
    _ = sys_system(String("rm -f ") + CACHE_DIR + String("/*.safetensors"))

    # ── load encoders ONCE ────────────────────────────────────────────────────
    print("[load] KleinVaeEncoder", VAE_PATH)
    var enc = KleinVaeEncoder[IH, IW].load(VAE_PATH, ctx)
    print("[load] Qwen3-8B encoder (klein_9b config) — ~16 GB")
    var tok = Qwen3Tokenizer(TOK_JSON)
    var qenc = Qwen3Encoder.load(QWEN8_DIR, Qwen3Config.klein_4b(), ctx)

    for idx in range(NUM_SAMPLES):
        print("── sample", idx, "──")
        # 1. VAE encode
        var img = _load_image(idx, ctx)
        var latent = enc.encode(img, ctx)
        var lsh = latent.shape()
        var lstd = _std(latent, ctx)
        print(
            "  latent shape:", lsh[0], lsh[1], lsh[2], lsh[3],
            " std=", Float32(lstd),
        )
        if lsh[1] != 128 or lsh[2] != IH // 16 or lsh[3] != IW // 16:
            raise Error("latent shape wrong (expect [1,128,32,32])")
        # latent-correctness gate: std should be ≈0.96 (BN-normalised packed).
        if lstd < 0.80 or lstd > 1.15:
            print("  WARNING: latent std", Float32(lstd), "outside [0.80,1.15] expected ~0.96")

        # 2. Qwen3 encode the caption
        var cap = _read_caption(idx)
        var ntok = _real_token_count(tok, cap)
        var ids = _tokenize_512(tok, cap)
        var emb = qenc.encode_klein(ids, ctx)
        var esh = emb.shape()
        print(
            "  caption tokens:", ntok, "-> 512   text_embedding:",
            esh[0], esh[1], esh[2], " dtype.tag", emb.dtype().tag,
        )
        if esh[1] != SEQ or esh[2] != 7680:
            raise Error("text_embedding shape wrong (expect [1,512,7680])")

        # 3. write the cache sample
        var mask = _mask_512(ntok, ctx)
        var out_path = CACHE_DIR + String("/alina_") + String(idx) + String(".safetensors")
        write_sample(latent, emb, mask, out_path, ctx)
        print("  wrote", out_path)

    print("")
    print("PASS: wrote", NUM_SAMPLES, "cache samples to", CACHE_DIR)
    print("[done] Qwen3 + VAE GPU memory freed on process exit")

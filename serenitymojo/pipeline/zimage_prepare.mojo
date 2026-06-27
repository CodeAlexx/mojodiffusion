# pipeline/zimage_prepare.mojo — config-driven Z-Image precache (latents+text).
#
# Second leg of the production LoRA precache pipeline. Consumes the staged image
# tensors written by zimage_stage.mojo and writes the cache consumed by
# train_zimage_real.mojo. Everything path-related comes from the JSON config
# (SINGLE SOURCE OF TRUTH — no hardcoded dataset/stage/cache paths):
#
#   <cfg.cache_dir>_stage/*.safetensors  ->  <cfg.cache_dir>/*.safetensors
#
#   staged image:   [1,3,H,W]  BF16, RGB, [-1,1]
#   cache sample:
#     latent:         [1,16,H/8,W/8]  mean VAE latent, unscaled
#     text_embedding: [1,512,2560]    Qwen3 layer-34 hidden state
#     text_mask:      [1,512]         1.0 for real tokens, 0.0 for pad
#
# Files are processed BUCKET-BY-BUCKET (comptime for over the Z-Image trainer
# ladder) so only one ZImageVaeEncoder[LH,LW] is GPU-resident at a time; cache
# files are independent per-sample safetensors (KleinCache sorts by name).
# Model weights (VAE / Qwen3 text encoder / tokenizer) live alongside the
# checkpoint; their dir is derived from cfg.checkpoint's parent.
#
# Run:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#     pixi run mojo run -I . serenitymojo/pipeline/zimage_prepare.mojo \
#       serenitymojo/configs/zimage_eri_2000.json

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
from serenitymojo.io.train_config_reader import read_model_config
from serenitymojo.models.vae.zimage_encoder import ZImageVaeEncoder
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer
from serenitymojo.models.text_encoder.qwen3_encoder import Qwen3Encoder, Qwen3Config
from serenitymojo.training.aspect_buckets import (
    ZIMAGE_T2D_LADDER_LEN, ZIMAGE_T2D_LADDER_X100,
    zimage_t2d_lat_h, zimage_t2d_lat_w,
)
from serenitymojo.training.klein_dataset import write_sample
from serenitymojo.training.onetrainer_train_loop_policy import (
    ot_cache_dir_from_train_config,
    ot_stage_dir_from_train_config,
)


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


def _parent_dir(path: String) raises -> String:
    """Directory containing `path` (drops the last '/'-separated component and
    any trailing slash). Used to locate the model dir from cfg.checkpoint."""
    var p = path
    var pb = p.as_bytes()
    var end = p.byte_length()
    while end > 0 and pb[end - 1] == 0x2F:  # strip trailing '/'
        end -= 1
    var cut = end
    while cut > 0 and pb[cut - 1] != 0x2F:
        cut -= 1
    if cut <= 0:
        raise Error(String("cannot derive parent dir from: ") + path)
    var out = List[UInt8]()
    for i in range(cut - 1):  # drop the separating '/'
        out.append(pb[i])
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


def _load_image(path: String, ctx: DeviceContext) raises -> Tensor:
    var st = SafeTensors.open(path)
    var info = st.tensor_info(String("image"))
    var bytes = st.tensor_bytes(String("image"))
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    var image = Tensor.from_view(tv, ctx)
    var sh = image.shape()
    if len(sh) != 4 or sh[0] != 1 or sh[1] != 3:
        raise Error("zimage_prepare: staged image must be [1,3,H,W]")
    if sh[2] % 64 != 0 or sh[3] % 64 != 0:
        raise Error("zimage_prepare: staged image dims must be 64-aligned")
    if image.dtype() != STDtype.BF16:
        raise Error("zimage_prepare: staged image must be BF16; do not stage Z-Image in F32")
    return image^


def _staged_image_dims(path: String) raises -> Tuple[Int, Int]:
    """(H, W) of the staged `image` tensor, header-only (no pixel read)."""
    var st = SafeTensors.open(path)
    var info = st.tensor_info(String("image"))
    if len(info.shape) != 4:
        raise Error("zimage_prepare: staged image must be rank-4")
    return (info.shape[2], info.shape[3])


def _encode_text_and_write(
    latent: Tensor,
    cap_path: String,
    out_path: String,
    tok: Qwen3Tokenizer,
    qenc: Qwen3Encoder,
    ctx: DeviceContext,
) raises:
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


def main() raises:
    var ctx = DeviceContext()
    var a = argv()
    if len(a) < 2 or not String(a[1]).endswith(String(".json")):
        raise Error(
            "usage: zimage_prepare <config.json>  "
            "(config must set dataset_path + cache_dir; run zimage_stage first)"
        )
    var cfg = read_model_config(String(a[1]))
    var stage_dir = ot_stage_dir_from_train_config(cfg, String(""))
    var cache_dir = ot_cache_dir_from_train_config(cfg, String(""))
    if cache_dir == String(""):
        raise Error("zimage_prepare: config has no cache_dir/dataset_cache_dir")
    if cfg.checkpoint == String(""):
        raise Error("zimage_prepare: config has no checkpoint dir (needed to locate VAE/text encoder)")
    var model_root = _parent_dir(cfg.checkpoint)
    var vae_dir = model_root + String("/vae")
    var text_encoder_dir = model_root + String("/text_encoder")
    var tok_json = model_root + String("/tokenizer/tokenizer.json")

    print("=== Z-Image prepare: staged image -> latent+text cache ===")
    print("  config:", String(a[1]))
    print("  stage: ", stage_dir)
    print("  cache: ", cache_dir)
    print("  model: ", model_root)

    var files = _stage_files(stage_dir)
    if len(files) == 0:
        raise Error(
            String("zimage_prepare: no staged image safetensors in ")
            + stage_dir
            + String(". Run zimage_stage <config.json> first.")
        )
    _ = sys_system(String("mkdir -p ") + cache_dir)
    _ = sys_system(String("rm -f ") + cache_dir + String("/*.safetensors"))
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

    print("[load] Qwen3 text encoder", text_encoder_dir)
    var tok = Qwen3Tokenizer(tok_json)
    var qenc = Qwen3Encoder.load(text_encoder_dir, Qwen3Config.zimage(), ctx)

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
                " image", IH_BI, "x", IW_BI, " (", n_match, "samples )", vae_dir,
            )
            var vae_bi = ZImageVaeEncoder[LH_BI, LW_BI].load(vae_dir, ctx)
            for i in range(len(files)):
                if hs[i] == IH_BI and ws[i] == IW_BI:
                    var name = files[i]
                    var stem = _drop_suffix(name, String(".safetensors").byte_length())
                    print("-- sample", i + 1, "/", len(files), name,
                          " bucket lat", LH_BI, "x", LW_BI)
                    var image = _load_image(stage_dir + String("/") + name, ctx)
                    var latent = vae_bi.encode_mean(image, ctx)
                    var lsh = latent.shape()
                    if lsh[1] != 16 or lsh[2] != LH_BI or lsh[3] != LW_BI:
                        raise Error("zimage_prepare: VAE latent shape wrong")
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
                String("zimage_prepare: staged image ") + files[i]
                + String(" (") + String(hs[i]) + String("x") + String(ws[i])
                + String(") is outside the comptime 512px ladder — fail loud")
            )
    print("PASS: wrote", len(files), "Z-Image cache samples to", cache_dir)

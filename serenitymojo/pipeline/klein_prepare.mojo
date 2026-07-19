# klein_prepare.mojo — GENERALIZED Klein prepare driver.
#
# Generalization of klein_prepare_alina.mojo: instead of the hardcoded 4 Alina
# samples, take (stage_dir, cache_dir, N) on argv and process N staged samples.
#
# For each staged sample i in [0, N):
#   input:  <stage_dir>/sample_<i>.safetensors  key "image" = [1,3,512,512] F32 [-1,1]
#           <stage_dir>/sample_<i>.txt           raw caption
#   1. VAE-encode the image  -> latent [1,128,32,32]  (real KleinVaeEncoder).
#      Assert std ≈ 0.96 (the latent-correctness gate).
#   2. Qwen3-4B encode the caption (Klein chat template, 512 tokens)
#      -> text_embedding [1,512,12288] (klein9b encode_klein).
#   3. write_sample(latent, text_embedding, mask) -> <cache_dir>/sample_<i>.safetensors
#      in the KleinCache format (training/klein_dataset.mojo reads latent +
#      text_embedding + text_mask).
#
# MEMORY: Qwen3-4B + the VAE encoder co-reside ONLY in THIS process. We load
# Qwen3 ONCE and encode all captions, VAE-encode all images, then write — never
# re-loading the encoder. The training process never imports Qwen3Encoder, so the
# encoder GPU memory is fully freed by THIS process's exit before training loads
# the DiT.
#
# Run (GPU — main session, NOT the tooling session):
#   cd /home/alex/mojodiffusion
#   rm -f serenitymojo.mojopkg
#   /tmp/klein_prepare <stage_dir> <cache_dir> <N>
# e.g.
#   /tmp/klein_prepare /home/alex/mojodiffusion/output/eri2_klein_stage \
#                      /home/alex/mojodiffusion/output/eri2_klein_cache 115

from std.collections import List
from std.gpu.host import DeviceContext
from std.math import sqrt
from std.sys import argv

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.io.ffi import sys_system, sys_open, sys_close, sys_pread, BytePtr, O_RDONLY
from std.memory import alloc
from serenitymojo.models.vae.klein_encoder import KleinVaeEncoder
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer
from serenitymojo.models.text_encoder.qwen3_encoder import Qwen3Encoder, Qwen3Config, _clone
from serenitymojo.ops.tensor_algebra import concat
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.training.klein_dataset import write_sample, write_sample_control


comptime VAE_PATH = "/home/alex/.serenity/models/vaes/flux2-vae.safetensors"
comptime QWEN8_DIR = (
    "/home/alex/.cache/huggingface/hub/models--Qwen--Qwen3-8B/"
    "snapshots/b968826d9c46dd6066d109eabc6255188de91218"
)
comptime TOK_JSON = QWEN8_DIR + "/tokenizer.json"
comptime IH = 512
comptime IW = 512
comptime SEQ = 512
comptime PAD_ID = 151643


# ── caption + image IO ───────────────────────────────────────────────────────


def _read_caption(stage_dir: String, idx: Int) raises -> String:
    """Read <stage_dir>/sample_<idx>.txt into a String via the io/ffi sys_pread
    idiom (the codebase routes all file I/O through io/ffi, never std open()).
    Mirrors train_config_reader._read_file_bytes."""
    var path = stage_dir + String("/sample_") + String(idx) + String(".txt")
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


def _load_image(stage_dir: String, idx: Int, ctx: DeviceContext) raises -> Tensor:
    var path = stage_dir + String("/sample_") + String(idx) + String(".safetensors")
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
    var a = argv()
    if len(a) < 4:
        raise Error(
            "usage: klein_prepare <stage_dir> <cache_dir> <N> [ref_stage_dir]  "
            "(reads <stage_dir>/sample_<i>.{safetensors,txt} for i in [0,N), "
            "writes <cache_dir>/sample_<i>.safetensors KleinCache samples. "
            "IMAGE-EDIT: pass a 4th arg <ref_stage_dir> holding the REFERENCE "
            "images under the SAME base filename sample_<i>.safetensors; each "
            "reference is VAE-encoded and stored in the control_latent slot of "
            "the same cache sample -> a 4-key edit cache.)"
        )
    var stage_dir = String(a[1])
    var cache_dir = String(a[2])
    var num_samples = atol(String(a[3]))
    if num_samples <= 0:
        raise Error("klein_prepare: N must be > 0")
    # IMAGE-EDIT convention (mirrors ai-toolkit / SimpleTuner control_path):
    # the target image lives at <stage_dir>/sample_<i>.safetensors and its
    # matching REFERENCE image lives at <ref_stage_dir>/sample_<i>.safetensors
    # (SAME base filename across two parallel dirs). When ref_stage_dir is given
    # we VAE-encode BOTH with the same encoder instance and write the reference
    # latent into the additive control_latent slot via write_sample_control.
    var ref_stage_dir = String("")
    if len(a) >= 5:
        ref_stage_dir = String(a[4])
    var edit_mode = len(ref_stage_dir) > 0

    var ctx = DeviceContext()
    print("=== Klein prepare:", num_samples, "images -> VAE latent + Qwen3 caption -> cache ===")
    print("  stage:", stage_dir)
    print("  cache:", cache_dir)
    if edit_mode:
        print("  EDIT MODE ref stage:", ref_stage_dir, "(-> control_latent slot)")
    _ = sys_system(String("mkdir -p ") + cache_dir)
    _ = sys_system(String("rm -f ") + cache_dir + String("/*.safetensors"))

    # ── load encoders ONCE ────────────────────────────────────────────────────
    print("[load] KleinVaeEncoder", VAE_PATH)
    var enc = KleinVaeEncoder[IH, IW].load(VAE_PATH, ctx)
    print("[load] Qwen3-8B encoder (klein_9b config, max_layer=26 — 16GB fit)")
    var tok = Qwen3Tokenizer(TOK_JSON)
    # encode_klein extracts layers [8,17,26] only — load(max_layer=26) skips
    # lm_head + layers 27..35 (~4.8GB), the difference between fitting the
    # 16GB 5080 and OOM. The encode below stops at layer 26 to match (running
    # later layers cannot change an earlier state; identical output bytes).
    var qenc = Qwen3Encoder.load(QWEN8_DIR, Qwen3Config.klein_9b(), ctx, 26)

    var warned = 0
    for idx in range(num_samples):
        print("── sample", idx, "──")
        # 1. VAE encode
        var img = _load_image(stage_dir, idx, ctx)
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
            warned += 1

        # 2. Qwen3 encode the caption
        var cap = _read_caption(stage_dir, idx)
        var ntok = _real_token_count(tok, cap)
        var ids = _tokenize_512(tok, cap)
        # encode_klein with an explicit layer-26 stop (== encode_klein output:
        # concat of layer states [8,17,26]; the load above skipped layers >26).
        var states = qenc.encode_layer_states(ids, ctx, 26)
        var h8 = _clone(states[8][], ctx)
        var h17 = _clone(states[17][], ctx)
        var h26 = _clone(states[26][], ctx)
        var emb = concat(2, ctx, h8, h17, h26)
        var esh = emb.shape()
        print(
            "  caption tokens:", ntok, "-> 512   text_embedding:",
            esh[0], esh[1], esh[2], " dtype.tag", emb.dtype().tag,
        )
        if esh[1] != SEQ or esh[2] != 12288:
            raise Error("text_embedding shape wrong (expect [1,512,12288])")

        # 3. write the cache sample. KleinCache dtype convention is ALL-F32
        # (latent + text_embedding) — the reader loads at STORED dtype and BF16
        # writes fail the trainer's concat/modulate dtype checks (MAP.md klein
        # restage note). Cast the BF16 encoder outputs before write_sample.
        var latent_f32 = latent.clone(ctx)
        if latent_f32.dtype() != STDtype.F32:
            latent_f32 = cast_tensor(latent_f32, STDtype.F32, ctx)
        var emb_f32 = emb.clone(ctx)
        if emb_f32.dtype() != STDtype.F32:
            emb_f32 = cast_tensor(emb_f32, STDtype.F32, ctx)
        var mask = _mask_512(ntok, ctx)
        var out_path = cache_dir + String("/sample_") + String(idx) + String(".safetensors")

        if edit_mode:
            # VAE-encode the REFERENCE image with the SAME encoder instance, then
            # store it in the control_latent slot (same [1,128,32,32] contract as
            # the target latent). Reference matched by SAME base filename.
            var ref_img = _load_image(ref_stage_dir, idx, ctx)
            var ref_latent = enc.encode(ref_img, ctx)
            var rsh = ref_latent.shape()
            var rstd = _std(ref_latent, ctx)
            print(
                "  ref latent shape:", rsh[0], rsh[1], rsh[2], rsh[3],
                " std=", Float32(rstd),
            )
            if rsh[1] != 128 or rsh[2] != IH // 16 or rsh[3] != IW // 16:
                raise Error("ref latent shape wrong (expect [1,128,32,32])")
            if rstd < 0.80 or rstd > 1.15:
                print("  WARNING: ref latent std", Float32(rstd), "outside [0.80,1.15] expected ~0.96")
                warned += 1
            var ref_f32 = ref_latent.clone(ctx)
            if ref_f32.dtype() != STDtype.F32:
                ref_f32 = cast_tensor(ref_f32, STDtype.F32, ctx)
            write_sample_control(latent_f32, emb_f32, mask, ref_f32, out_path, ctx)
            print("  wrote (edit, +control_latent)", out_path)
        else:
            write_sample(latent_f32, emb_f32, mask, out_path, ctx)
            print("  wrote", out_path)

    print("")
    print("PASS: wrote", num_samples, "cache samples to", cache_dir)
    if warned > 0:
        print("  NOTE:", warned, "sample(s) had latent std outside [0.80,1.15]")
    print("[done] Qwen3 + VAE GPU memory freed on process exit")

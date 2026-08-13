# ideogram4_prepare.mojo — torch-free Ideogram-4 LoRA-training cache prepare.
#
# The torch-side cache producer is gone. This regenerates the indexed
# safetensors cache the Ideogram4CacheReader streams, directly from the staged
# dir, using the GATED serenitymojo Mojo encoders (same VAE + Qwen3-VL 13-tap
# text path the inference/sampler stack is parity-verified against). Structure
# mirrors serenitymojo/pipeline/klein_prepare.mojo and
# serenitymojo/models/krea2/krea2_prepare_cache.mojo: load encoders ONCE, loop
# samples, write a single-file cache, process exit frees the encoder GPU memory
# before the trainer loads the DiT.
#
# ── INPUT (stage dir, produced by the ideogram4 stager) ──────────────────────
#   <stage_dir>/images.safetensors   image.<i>  [1,3,512,512] F32, range [-1,1]
#   <stage_dir>/caption.<i>.txt      RAW caption prose (no chat template)
#   (<stage_dir>/prompt.<i>.txt is the SAME text pre-wrapped in the chat
#    template — VERIFIED byte-equal to _render_chat_prompt(caption.<i>); we
#    render from the raw caption so the tool owns the exact template, matching
#    Ideogram4SampleResident._render_chat_prompt.)
#
# ── OUTPUT (single-file cache.safetensors; Ideogram4CacheReader contract) ─────
#   clean.<i>          [1,128,32,32] F32   packed normalized VAE latent
#   llm.<i>            [1,256,53248] BF16  Qwen3-VL 13-tap, PAD ROWS ZEROED
#   text_len.<i>       [1]           F32   natural (pre-pad) token count
#   llm_uncond         [1,256,53248] BF16  empty-prompt (template-only) encode
#   text_len_uncond    [1]           F32   uncond natural token count (== 8)
#   (noise/noisy/t_flow are NOT written — the reader synthesizes them per step.)
#
# ── CONTRACT EVIDENCE (from the surviving eri2 cache + the reader/sampler) ────
#   * dataLoader/Ideogram4CacheReader.mojo _discover: keys clean.<i>/llm.<i>,
#     optional text_len.<i>, optional llm_uncond/text_len_uncond; _validate_*
#     require clean [1,128,GH,GW] and llm [1,NT,53248] with NT/GH/GW from the
#     trainer instantiation train_ideogram4_lora_from_cache[NT=256,GH=32,GW=32]
#     (Ideogram4LiveTrainer.mojo:57-59,334).
#   * PACKED_CHANNELS=128, TEXT_FEATURE_DIM=53248 (Ideogram4Sampler.mojo:24-25).
#   * Tokenizer + template + PAD id + NT-pad are the sampler's canonical path
#     (Ideogram4SampleResident.mojo:77-87,111-143): Qwen3Tokenizer(I4_TOK_JSON),
#     _render_chat_prompt = "<|im_start|>user\n"+cap+"<|im_end|>\n<|im_start|>assistant\n",
#     pad with I4_PAD_ID=151643 to NT, and ZERO the pad rows of the features.
#     MEASURED in the surviving cache: llm.0 rows [text_len.0=177 .. 256) are
#     exactly 0.0; text_len ranges 106..256 (2 samples hit 256 = TRUNCATED, not
#     rejected — so we truncate long captions to NT), uncond text_len == 8.
#   * VAE image is fed BF16 (models/vae/parity/ideogram4_vae_encode_probe.mojo:27
#     casts image to BF16 before encode_ideogram4_latents; gate cos 0.9999586).
#     latent_shift/latent_scale [128] F32 come from the fx latent-norm fixture.
#   * clean latent sanity: MEASURED std ~1.03 (surviving clean.0 std 1.0338,
#     mean -0.053, range [-4.69, 5.63]); we warn if std is outside [0.85,1.20].
#
# ── BUILD (tooling session; -O2, cshim rpath for the TE sdpa; NVIDIA GPU) ─────
#   cd /home/alex/mojodiffusion && \
#   pixi run mojo build --optimization-level 2 -I . -I /home/alex/MOJO-libs \
#     -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     -Xlinker -rpath -Xlinker /home/alex/mojodiffusion/serenitymojo/ops/cshim/lib \
#     serenitymojo/pipeline/ideogram4_prepare.mojo -o /tmp/ideogram4_prepare
#
# ── RUN (GPU — main session, NOT the tooling session) ────────────────────────
#   Write to a NEW path first so the surviving eri2 cache stays intact as the
#   parity oracle; diff, then replace it only once parity is confirmed.
#   LD_LIBRARY_PATH=/home/alex/mojodiffusion/.pixi/envs/default/lib \
#     /tmp/ideogram4_prepare \
#       /home/alex/serenity-trainer/output/eri2_ideogram4_staged \
#       /home/alex/serenity-trainer/output/eri2_ideogram4_cache_mojo.safetensors 115
#   # PARITY: per-key vs the surviving eri2_ideogram4_cache.safetensors —
#   #   clean.<i>/llm.<i> cosine >= 0.999 (VAE + 13-tap TE are gated to cos
#   #   0.9999 vs torch, so >= 0.999 end-to-end), text_len.<i> exact-equal,
#   #   llm pad rows [text_len,256) exactly 0.0, uncond text_len == 8.
#
# Mojo 1.0.0b1.

from max.gpu.host import DeviceContext
from std.math import sqrt
from std.memory import ArcPointer, alloc
from std.sys import argv

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, BytePtr, O_RDONLY
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import mul
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer
from serenitymojo.models.vae.ldm_encoder import (
    load_ideogram4_vae_encoder,
    encode_ideogram4_latents,
)
from serenitymojo.models.text_encoder.ideogram_qwen3vl import (
    load_ideogram_qwen3vl,
    encode_ideogram_taps,
)
from serenitymojo.models.text_encoder.qwen3_encoder import Qwen3Encoder


comptime TArc = ArcPointer[Tensor]

comptime I4_VAE = "/home/alex/.serenity/models/ideogram-4-fp8/vae/diffusion_pytorch_model.safetensors"
comptime I4_LATENTNORM = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/ideogram4_fx_latentnorm.safetensors"
comptime I4_TE = "/home/alex/.serenity/models/ideogram-4-fp8/text_encoder/model.safetensors"
comptime I4_TOK_JSON = "/home/alex/.serenity/models/ideogram-4-fp8/tokenizer/tokenizer.json"
comptime I4_PAD_ID = 151643

comptime NT = 256          # giger 512px cache bucket (Ideogram4LiveTrainer.mojo:57)
comptime LH = 64           # 512px image / 8 = VAE-latent edge; packed grid = LH/2
comptime LW = 64
comptime FEAT_DIM = 53248
comptime PACKED_CH = 128
comptime GRID = 32         # LH/2 = LW/2 = 32


# ── file IO (io/ffi sys_pread idiom — the codebase never uses std open() for
#    cache files; mirrors train_config_reader._read_file_bytes) ───────────────
def _read_text_file(path: String) raises -> String:
    var fd = sys_open(path, O_RDONLY, Int32(0))
    if fd < 0:
        raise Error(String("ideogram4_prepare: file not found: ") + path)
    var bytes = List[UInt8]()
    comptime CHUNK = 65536
    var buf = alloc[UInt8](CHUNK)
    var offset = 0
    while True:
        var nread = sys_pread(fd, BytePtr(unsafe_from_address=Int(buf)), CHUNK, offset)
        if nread < 0:
            buf.free()
            _ = sys_close(fd)
            raise Error(String("ideogram4_prepare: read error: ") + path)
        if nread == 0:
            break
        for i in range(nread):
            bytes.append(buf[i])
        offset += nread
        if nread < CHUNK:
            break
    buf.free()
    _ = sys_close(fd)
    # strip trailing newline(s); caption.<i>.txt has none, but be defensive so a
    # stray \n never perturbs the tokenization.
    while len(bytes) > 0 and (bytes[len(bytes) - 1] == 10 or bytes[len(bytes) - 1] == 13):
        _ = bytes.pop()
    return String(unsafe_from_utf8=bytes)


# Ideogram-4 chat template — byte-identical to
# Ideogram4SampleResident._render_chat_prompt (proven == prompt.<i>.txt and,
# for the empty prompt, == uncond.txt).
def _render_chat_prompt(prompt: String) -> String:
    return (
        String("<|im_start|>user\n")
        + prompt
        + String("<|im_end|>\n<|im_start|>assistant\n")
    )


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


# ids -> 13-tap [1,NT,53248] BF16 with pad rows [real_len, NT) zeroed. `ids` is
# the natural token list; if len > NT it is truncated to NT (real_len already
# clamped to NT by the caller), matching the producer's truncation behavior.
def _encode_padded(
    enc: Qwen3Encoder, var ids: List[Int], real_len: Int, ctx: DeviceContext
) raises -> Tensor:
    if len(ids) > NT:
        var trunc = List[Int]()
        for i in range(NT):
            trunc.append(ids[i])
        ids = trunc^
    while len(ids) < NT:
        ids.append(I4_PAD_ID)

    var feats = encode_ideogram_taps(enc, ids, ctx)   # [1,NT,FEAT_DIM] BF16
    if real_len < NT:
        var mask_host = List[Float32]()
        for j in range(NT):
            if j < real_len:
                mask_host.append(Float32(1.0))
            else:
                mask_host.append(Float32(0.0))
        var mask_f32 = Tensor.from_host(mask_host^, [1, NT, 1], STDtype.F32, ctx)
        var mask = cast_tensor(mask_f32, STDtype.BF16, ctx)   # [1,NT,1] BF16
        return cast_tensor(mul(feats, mask, ctx), STDtype.BF16, ctx)
    return cast_tensor(feats, STDtype.BF16, ctx)


def _text_len_tensor(real_len: Int, ctx: DeviceContext) raises -> Tensor:
    var h = List[Float32]()
    h.append(Float32(real_len))
    return Tensor.from_host(h^, [1], STDtype.F32, ctx)


def main() raises:
    var a = argv()
    if len(a) < 4:
        raise Error(
            "usage: ideogram4_prepare <stage_dir> <cache_out.safetensors> <N>"
            "  (reads <stage_dir>/images.safetensors image.<i> [1,3,512,512] +"
            " <stage_dir>/caption.<i>.txt for i in [0,N); writes the indexed"
            " Ideogram4 cache clean/llm/text_len.<i> + llm_uncond/text_len_uncond)"
        )
    var stage_dir = String(a[1])
    var out_path = String(a[2])
    var n = atol(String(a[3]))
    if n <= 0:
        raise Error("ideogram4_prepare: N must be > 0")

    var ctx = DeviceContext()
    print("=== Ideogram4 prepare:", n, "samples -> VAE latent + Qwen3-VL 13-tap -> cache ===")
    print("  stage:", stage_dir)
    print("  cache:", out_path)

    # ── load encoders ONCE (VAE small, then the ~8.7GB fp8 TE -> BF16) ─────────
    print("[load] Ideogram4 VAE encoder", I4_VAE)
    var venc = load_ideogram4_vae_encoder[LH, LW](String(I4_VAE), ctx)
    var ln = ShardedSafeTensors.open(String(I4_LATENTNORM))
    var shift = Tensor.from_view(ln.tensor_view(String("latent_shift")), ctx)  # [128] F32
    var scale = Tensor.from_view(ln.tensor_view(String("latent_scale")), ctx)  # [128] F32
    print("[load] Qwen3-VL 13-tap text encoder", I4_TE)
    var enc = load_ideogram_qwen3vl(String(I4_TE), ctx)
    var tok = Qwen3Tokenizer(String(I4_TOK_JSON))

    var imgs = ShardedSafeTensors.open(stage_dir + String("/images.safetensors"))

    var names = List[String]()
    var tensors = List[TArc]()
    var warned = 0

    for i in range(n):
        print("── sample", i, "──")

        # 1. VAE encode: image.<i> [1,3,512,512] F32 -> BF16 -> packed normalized
        #    latent [1,128,32,32], stored F32 (matches surviving clean.<i>).
        var img_f32 = Tensor.from_view(imgs.tensor_view(String("image.") + String(i)), ctx)
        var img = cast_tensor(img_f32, STDtype.BF16, ctx)
        var clean_raw = encode_ideogram4_latents[LH, LW](venc, img, shift, scale, ctx)
        var clean = cast_tensor(clean_raw, STDtype.F32, ctx)
        var csh = clean.shape()
        if (
            len(csh) != 4
            or csh[0] != 1
            or csh[1] != PACKED_CH
            or csh[2] != GRID
            or csh[3] != GRID
        ):
            raise Error(
                String("ideogram4_prepare: clean.") + String(i)
                + " shape wrong (expect [1,128,32,32])"
            )
        var cstd = _std(clean, ctx)
        if cstd < 0.85 or cstd > 1.20:
            print("  WARNING: clean std", Float32(cstd), "outside [0.85,1.20] expected ~1.03")
            warned += 1

        # 2. Qwen3-VL 13-tap: raw caption -> chat template -> ids -> [1,256,53248]
        #    BF16 with pad rows zeroed. text_len = natural token count (clamped NT).
        var rendered = _render_chat_prompt(
            _read_text_file(stage_dir + String("/caption.") + String(i) + String(".txt"))
        )
        var ids = tok.encode(rendered)
        var real_len = len(ids)
        if real_len > NT:
            real_len = NT
        var llm = _encode_padded(enc, ids^, real_len, ctx)
        var lsh = llm.shape()
        if len(lsh) != 3 or lsh[0] != 1 or lsh[1] != NT or lsh[2] != FEAT_DIM:
            raise Error(
                String("ideogram4_prepare: llm.") + String(i)
                + " shape wrong (expect [1,256,53248])"
            )
        print("  clean std", Float32(cstd), " | text_len", real_len, "/", NT)

        names.append(String("clean.") + String(i))
        tensors.append(TArc(clean^))
        names.append(String("llm.") + String(i))
        tensors.append(TArc(llm^))
        names.append(String("text_len.") + String(i))
        tensors.append(TArc(_text_len_tensor(real_len, ctx)))

    # ── uncond (caption dropout): the empty-prompt template-only encode. This is
    #    _render_chat_prompt("") == uncond.txt (proven byte-equal, 8 tokens). The
    #    reader treats it as OPTIONAL, but the surviving cache always carried it. ─
    print("── uncond (empty-prompt) ──")
    var u_ids = tok.encode(_render_chat_prompt(String("")))
    var u_real = len(u_ids)
    if u_real > NT:
        u_real = NT
    var u_llm = _encode_padded(enc, u_ids^, u_real, ctx)
    names.append(String("llm_uncond"))
    tensors.append(TArc(u_llm^))
    names.append(String("text_len_uncond"))
    tensors.append(TArc(_text_len_tensor(u_real, ctx)))
    print("  uncond text_len", u_real, "/", NT)

    save_safetensors(names, tensors, out_path, ctx)
    print("")
    print("PASS: wrote", n, "samples (+uncond) to", out_path)
    if warned > 0:
        print("  NOTE:", warned, "sample(s) had clean std outside [0.85,1.20]")
    print("[done] VAE + Qwen3-VL GPU memory freed on process exit")

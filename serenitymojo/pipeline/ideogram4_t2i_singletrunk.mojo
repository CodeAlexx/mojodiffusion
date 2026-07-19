# pipeline/ideogram4_t2i_singletrunk.mojo — GATE 2 for ideogram4 FlowEdit (task #25).
#
# Minimal 1024x1024 text->image with SINGLE-TRUNK uncond: the uncond pass runs
# through the ONE resident cond transformer with zeroed text features (image-only
# S=NIMG), instead of loading the 8.7GB unconditional_transformer — the accepted
# 16GB-fit precedent from Serenity Ideogram4SampleResident.mojo:143-152 (trainer
# inline sampling). ~9GB resident DiT + activations fits the RTX 5080.
#
# Differences vs pipeline/ideogram4_generate.mojo (the dual-trunk 24GB path):
#   * prompt read from a FILE (structured JSON caption), tokenized LIVE
#     (Qwen3Tokenizer + chat template), padded to the FIXED 1024-token window
#     with PAD 151643, and the pad feature rows ZEROED (ideogram4_prepare.mojo
#     convention, NOT the backend's unzeroed pad).
#   * TE encoded FIRST, mempool trimmed, THEN the single DiT trunk loads
#     (serve/ideogram4_backend.mojo IPHASE_ENCODE -> trim -> IPHASE_LOAD pattern).
#   * uncond forward uses cond weights + zero llm + image-only seq (S=4096).
#   * decode: whole-image if >=14GiB free else 3x3 tiled (backend policy).
#
# Schedule/CFG: V4_DEFAULT_20 exactly as ideogram4_generate.mojo — logit-normal
# (res-shifted mean, std=1.75), Euler z += v*(s-t), gw=7.0 with 3.0 on the last
# 2 steps.
#
# BUILD:
#   cd /home/alex/mojodiffusion && \
#   pixi run mojo build --optimization-level 2 -I . -I /home/alex/MOJO-libs \
#     -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     -Xlinker -rpath -Xlinker /home/alex/mojodiffusion/serenitymojo/ops/cshim/lib \
#     serenitymojo/pipeline/ideogram4_t2i_singletrunk.mojo -o /tmp/i4_t2i_single
# RUN:
#   LD_LIBRARY_PATH=/home/alex/mojodiffusion/.pixi/envs/default/lib \
#     /tmp/i4_t2i_single <prompt.json> <out.png> [--steps 20] [--seed 0] [--cfg 7.0]
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu.host import DeviceContext
from std.memory import ArcPointer, alloc
from std.sys import argv
from std.time import perf_counter_ns

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, BytePtr, O_RDONLY
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import (
    mul, add, mul_scalar, reshape, permute, slice, concat,
)
from serenitymojo.offload.vmm_cuda import cu_mempool_trim_current, cu_mem_get_info
from serenitymojo.image.png import save_png
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer
from serenitymojo.models.dit.ideogram4_resident import (
    Ideogram4Weights, ideogram4_forward_r, ideogram4_build_masks,
)
from serenitymojo.models.dit.ideogram4_mrope import build_ideogram4_mrope
from serenitymojo.models.vae.ldm_decoder import load_ideogram4_vae_decoder
from serenitymojo.models.vae.ideogram4_tiled_decode import ideogram4_tiled_decode
from serenitymojo.models.text_encoder.ideogram_qwen3vl_streamed import (
    encode_ideogram_taps_streamed,
)
from serenitymojo.sampling.ideogram4_schedule import (
    ideogram4_logitnormal, ideogram4_schedule_mean, make_step_intervals,
)

comptime TArc = ArcPointer[Tensor]

comptime COND = "/home/alex/.serenity/models/ideogram-4-fp8/transformer/diffusion_pytorch_model.safetensors"
comptime TE = "/home/alex/.serenity/models/ideogram-4-fp8/text_encoder/model.safetensors"
comptime TOK_JSON = "/home/alex/.serenity/models/ideogram-4-fp8/tokenizer/tokenizer.json"
comptime VAE = "/home/alex/.serenity/models/ideogram-4-fp8/vae/diffusion_pytorch_model.safetensors"
comptime LATENT_NORM = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/ideogram4_fx_latentnorm.safetensors"

comptime PAD_ID = 151643
comptime IMG_OFFSET = 65536
comptime TEXT_TOKENS = 1024      # fixed text window (backend TEXT_TOKENS)
comptime FEAT_DIM = 53248
comptime GH = 64                 # 1024 px / 16
comptime GW = 64
comptime NIMG = GH * GW          # 4096
comptime TOTAL = TEXT_TOKENS + NIMG  # 5120
comptime LAYERS = 34
comptime HEADS = 18
comptime HEAD_DIM = 256
comptime HIDDEN = 4608


def _read_text_file(path: String) raises -> String:
    var fd = sys_open(path, O_RDONLY, Int32(0))
    if fd < 0:
        raise Error(String("i4_t2i_single: file not found: ") + path)
    var bytes = List[UInt8]()
    comptime CHUNK = 65536
    var buf = alloc[UInt8](CHUNK)
    var offset = 0
    while True:
        var nread = sys_pread(fd, BytePtr(unsafe_from_address=Int(buf)), CHUNK, offset)
        if nread < 0:
            buf.free()
            _ = sys_close(fd)
            raise Error(String("i4_t2i_single: read error: ") + path)
        if nread == 0:
            break
        for i in range(nread):
            bytes.append(buf[i])
        offset += nread
        if nread < CHUNK:
            break
    buf.free()
    _ = sys_close(fd)
    while len(bytes) > 0 and (bytes[len(bytes) - 1] == 10 or bytes[len(bytes) - 1] == 13):
        _ = bytes.pop()
    return String(unsafe_from_utf8=bytes)


def _render_chat_prompt(prompt: String) -> String:
    return (
        String("<|im_start|>user\n") + prompt
        + String("<|im_end|>\n<|im_start|>assistant\n")
    )


# Encode N prompts -> N x [1,TEXT_TOKENS,53248] BF16 with pad rows ZEROED
# (ideogram4_prepare convention). Uses the 16GB LAYER-STREAMED TE (the resident
# ~15.1GB BF16 encoder OOMs at seq 1024 on the RTX 5080 — measured; the streamed
# variant is parity-gated cos 0.99998 vs the resident oracle). One disk pass
# covers ALL prompts; peak TE VRAM ~2GB.
def _encode_prompts(
    prompts: List[String], names: List[String], ctx: DeviceContext
) raises -> List[TArc]:
    var tok = Qwen3Tokenizer(String(TOK_JSON))
    var ids_list = List[List[Int]]()
    var real_lens = List[Int]()
    for i in range(len(prompts)):
        var ids = tok.encode(_render_chat_prompt(prompts[i]))
        var real_len = len(ids)
        if real_len > TEXT_TOKENS:
            raise Error(
                String("i4_t2i_single: ") + names[i] + " tokenized to "
                + String(real_len) + " tokens > fixed window " + String(TEXT_TOKENS)
            )
        for _ in range(TEXT_TOKENS - real_len):
            ids.append(PAD_ID)
        print("[t2i-single]", names[i], "tokens =", real_len, "/", TEXT_TOKENS)
        ids_list.append(ids^)
        real_lens.append(real_len)
    var feats = encode_ideogram_taps_streamed(String(TE), ids_list^, ctx)
    var out = List[TArc]()
    for i in range(len(feats)):
        var mask_host = List[Float32]()
        for j in range(TEXT_TOKENS):
            mask_host.append(Float32(1.0) if j < real_lens[i] else Float32(0.0))
        var mask_f32 = Tensor.from_host(mask_host^, [1, TEXT_TOKENS, 1], STDtype.F32, ctx)
        var mask = cast_tensor(mask_f32, STDtype.BF16, ctx)
        out.append(TArc(cast_tensor(mul(feats[i][], mask, ctx), STDtype.BF16, ctx)))
    return out^


# pos/ind for [1024-text][4096-img] cond seq + image-only uncond seq.
def _build_inputs(ctx: DeviceContext) raises -> List[TArc]:
    var pos = List[Float32]()
    var ind = List[Float32]()
    var npos = List[Float32]()
    var nind = List[Float32]()
    for l in range(TEXT_TOKENS):
        pos.append(Float32(l)); pos.append(Float32(l)); pos.append(Float32(l))
        ind.append(3.0)   # LLM_TOKEN_INDICATOR
    for h in range(GH):
        for w in range(GW):
            var t0 = Float32(IMG_OFFSET)
            var hh = Float32(IMG_OFFSET + h)
            var ww = Float32(IMG_OFFSET + w)
            pos.append(t0); pos.append(hh); pos.append(ww)
            npos.append(t0); npos.append(hh); npos.append(ww)
            ind.append(2.0)   # OUTPUT_IMAGE_INDICATOR
            nind.append(2.0)
    var out = List[TArc]()
    out.append(TArc(Tensor.from_host(pos^, [1, TOTAL, 3], STDtype.F32, ctx)))
    out.append(TArc(Tensor.from_host(ind^, [1, TOTAL], STDtype.F32, ctx)))
    out.append(TArc(Tensor.from_host(npos^, [1, NIMG, 3], STDtype.F32, ctx)))
    out.append(TArc(Tensor.from_host(nind^, [1, NIMG], STDtype.F32, ctx)))
    return out^


def _zeros_bf16(var shape: List[Int], n: Int, ctx: DeviceContext) raises -> Tensor:
    var h = List[Float32](capacity=n)
    for _ in range(n):
        h.append(0.0)
    return Tensor.from_host(h^, shape^, STDtype.BF16, ctx)


# The single-trunk CFG denoise. Loads the ONE cond transformer inside so it
# frees on return (before VAE decode). Returns denoised z [1,NIMG,128] F32.
def _denoise(
    text_features: Tensor,   # [1,TEXT_TOKENS,53248] BF16 (pad rows zeroed)
    steps: Int, seed: UInt64, gw_main: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    var inp = _build_inputs(ctx)

    # llm_full = [text ; zeros(NIMG)]; uncond llm = zeros(NIMG) (image-only).
    var img_zeros = _zeros_bf16([1, NIMG, FEAT_DIM], NIMG * FEAT_DIM, ctx)
    var llm = concat(1, ctx, text_features, img_zeros)      # [1,TOTAL,53248]
    var neg_llm = _zeros_bf16([1, NIMG, FEAT_DIM], NIMG * FEAT_DIM, ctx)

    var sec = [24, 20, 20]
    var cs = build_ideogram4_mrope(inp[0][], HEAD_DIM, sec, Float32(5000000.0), ctx, STDtype.BF16)
    var ncs = build_ideogram4_mrope(inp[2][], HEAD_DIM, sec, Float32(5000000.0), ctx, STDtype.BF16)

    print("[t2i-single] loading SINGLE resident fp8 trunk (cond only)...")
    var w = Ideogram4Weights.load(ShardedSafeTensors.open(String(COND)), ctx)
    var cond_masks = ideogram4_build_masks(inp[1][], ctx)
    var uncond_masks = ideogram4_build_masks(inp[3][], ctx)
    ctx.synchronize()
    var mem0 = cu_mem_get_info()
    print("[t2i-single] after DiT load: free =",
          Float32(Float64(mem0.free_bytes) / 1073741824.0), "GiB")

    var z = randn([1, NIMG, 128], seed, STDtype.F32, ctx)
    var zpad_h = List[Float32](capacity=TEXT_TOKENS * 128)
    for _ in range(TEXT_TOKENS * 128):
        zpad_h.append(0.0)
    var text_zpad = Tensor.from_host(zpad_h^, [1, TEXT_TOKENS, 128], STDtype.F32, ctx)

    var mean = ideogram4_schedule_mean(1024, 1024, 0.0)
    var si = make_step_intervals(steps)
    var min_free = mem0.free_bytes
    var total_bytes = mem0.total_bytes
    var step_s_sum = 0.0
    var step_n = 0

    for step in range(steps - 1, -1, -1):
        ctx.synchronize()
        var t0 = Int(perf_counter_ns())
        var t_val = ideogram4_logitnormal(Float64(si[step + 1]), mean, 1.75)
        var s_val = ideogram4_logitnormal(Float64(si[step]), mean, 1.75)
        var gw = Float32(3.0) if step < 2 else gw_main
        var t = Tensor.from_host([t_val], [1], STDtype.F32, ctx)
        var pos_z = cast_tensor(concat(1, ctx, text_zpad, z), STDtype.BF16, ctx)
        var cout = ideogram4_forward_r[TOTAL](
            w, pos_z, llm, t, cond_masks, cs[0], cs[1],
            LAYERS, HEADS, HEAD_DIM, HIDDEN, ctx,
        )
        var pos_v = slice(cout, 1, TEXT_TOKENS, NIMG, ctx)
        # SINGLE-TRUNK uncond: same weights, zero text, image-only sequence.
        var t2 = Tensor.from_host([t_val], [1], STDtype.F32, ctx)
        var z_bf = cast_tensor(z, STDtype.BF16, ctx)
        var nout = ideogram4_forward_r[NIMG](
            w, z_bf, neg_llm, t2, uncond_masks, ncs[0], ncs[1],
            LAYERS, HEADS, HEAD_DIM, HIDDEN, ctx,
        )
        var v = add(mul_scalar(pos_v, gw, ctx), mul_scalar(nout, Float32(1.0) - gw, ctx), ctx)
        z = add(z, mul_scalar(v, s_val - t_val, ctx), ctx)
        ctx.synchronize()
        var t1 = Int(perf_counter_ns())
        var mem = cu_mem_get_info()
        if mem.free_bytes < min_free:
            min_free = mem.free_bytes
        step_s_sum += Float64(t1 - t0) / 1e9
        step_n += 1
        print("  step", step, "gw", gw, "t", t_val, "s", s_val,
              " STEP =", Float32(Float64(t1 - t0) / 1e9), "s  free =",
              Float32(Float64(mem.free_bytes) / 1073741824.0), "GiB")

    print("[t2i-single] denoise avg s/step =", Float32(step_s_sum / Float64(step_n)),
          " (", step_n, "steps, 2 forwards each)")
    print("[t2i-single] PEAK VRAM (denoise) =",
          Float32(Float64(total_bytes - min_free) / 1073741824.0), "GiB used / ",
          Float32(Float64(total_bytes) / 1073741824.0), "GiB")
    return z^


# denorm + unpatch + decode (whole if >=14GiB free else tiled) + save.
def decode_tokens_to_png(z: Tensor, out_png: String, ctx: DeviceContext) raises:
    var ln = ShardedSafeTensors.open(String(LATENT_NORM))
    var scale = reshape(Tensor.from_view(ln.tensor_view("latent_scale"), ctx), [1, 1, 128], ctx)
    var shift = reshape(Tensor.from_view(ln.tensor_view("latent_shift"), ctx), [1, 1, 128], ctx)
    var zd = add(mul(z, scale, ctx), shift, ctx)
    var z6 = reshape(zd, [1, GH, GW, 2, 2, 32], ctx)
    var zp = permute(z6, [0, 5, 1, 3, 2, 4], ctx)
    var latent = reshape(zp, [1, 32, 2 * GH, 2 * GW], ctx)
    var latent_bf = cast_tensor(latent, STDtype.BF16, ctx)
    ctx.synchronize()
    cu_mempool_trim_current(0)
    var mem = cu_mem_get_info()
    var free_gib = Float64(mem.free_bytes) / 1073741824.0
    var img: Tensor
    if mem.free_bytes > 14 * 1024 * 1024 * 1024:
        print("[t2i-single] WHOLE-image decode (free =", Float32(free_gib), "GiB)")
        var dec = load_ideogram4_vae_decoder[2 * GH, 2 * GW](String(VAE), ctx)
        img = dec.decode(latent_bf, ctx)
    else:
        print("[t2i-single] TILED 3x3 decode (free =", Float32(free_gib), "GiB)")
        img = ideogram4_tiled_decode[2 * GH, 2 * GW](latent_bf, String(VAE), ctx)
    save_png(img, out_png, ctx)
    print("[t2i-single] wrote", out_png)


def main() raises:
    var a = argv()
    if len(a) < 3:
        raise Error(
            "usage: i4_t2i_single <prompt.json> <out.png> [--steps 20] [--seed 0] [--cfg 7.0]"
        )
    var prompt_path = String(a[1])
    var out_png = String(a[2])
    var steps = 20
    var seed = UInt64(0)
    var gw_main = Float32(7.0)
    for i in range(len(a)):
        if a[i] == String("--steps") and i + 1 < len(a):
            steps = Int(String(a[i + 1]))
        if a[i] == String("--seed") and i + 1 < len(a):
            seed = UInt64(Int(String(a[i + 1])))
        if a[i] == String("--cfg") and i + 1 < len(a):
            gw_main = Float32(Float64(String(a[i + 1])))

    var ctx = DeviceContext()
    print("[t2i-single] 1024x1024 single-trunk CFG  steps =", steps,
          " seed =", seed, " cfg =", gw_main)

    var prompts = List[String]()
    prompts.append(_read_text_file(prompt_path))
    var names = List[String]()
    names.append(String("PROMPT"))
    var feats = _encode_prompts(prompts^, names^, ctx)
    ctx.synchronize()
    cu_mempool_trim_current(0)   # CRITICAL: TE must NOT co-reside with the DiT
    var mem = cu_mem_get_info()
    print("[t2i-single] TE freed+trimmed: free =",
          Float32(Float64(mem.free_bytes) / 1073741824.0), "GiB")

    var z = _denoise(feats[0][], steps, seed, gw_main, ctx)
    ctx.synchronize()
    cu_mempool_trim_current(0)   # DiT freed by _denoise scope
    decode_tokens_to_png(z, out_png, ctx)

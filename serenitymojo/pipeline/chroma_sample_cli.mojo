# serenitymojo/pipeline/chroma_sample_cli.mojo
#
# UI-driven CLI adapter for Chroma 1024x1024 text→image generation.
# Copies the qwenimage_sample_cli.mojo pattern: argv parsing, _select_prompt /
# _load_prompt_json, and a main() that bridges the sample-prompt JSON to the
# proven chroma_pipeline_1024_multistep.mojo generate path.
#
# Contract (the UI bridge calls it exactly this way):
#
#   chroma_sample_cli  <config.json>  <lora|->  <sample_prompts.json>  <prompt_id>  <out.png>
#
#   argv[1]  config JSON path (model dirs are comptime constants; this argument
#            is ACCEPTED BUT IGNORED TODAY — edit the comptime paths at the top
#            of this file to override, or use the config to generate a per-prompt
#            sidecar and point caps_pos/caps_neg at it).
#
#   argv[2]  LoRA safetensors path, or "-"/"base"/"" for base model.
#            Chroma has no LoRA hook today; value ACCEPTED AND IGNORED.
#
#   argv[3]  sample_prompts JSON (serenity.sample_prompts.v1 schema).
#            Read with `read_sample_prompt_config`.
#
#   argv[4]  Prompt id/label to select from the JSON, or "" for the first entry.
#
#   argv[5]  Output PNG path.  Written via save_png(…, ValueRange.SIGNED).
#
# ──────────────────────────────────────────────────────────────────────────────
# Request fields honored vs fixed:
#
#   HONORED at runtime:
#     • prompt    — T5 embeddings path read from caps_pos in the sample prompt
#                   entry (see "Text conditioning" below).
#     • negative  — T5 embeddings path read from caps_neg in the sample prompt
#                   entry.
#
#   FIXED at comptime (from chroma_pipeline_1024_multistep.mojo):
#     • steps  = NUM_STEPS  (30)
#     • cfg    = GUIDANCE   (4.0)
#     • seed   = SEED       (UInt64(42))
#     • width  = 1024  (LH*8 = 128*8)
#     • height = 1024  (LW*8 = 128*8)
#
#   The Chroma DiT attention shape (N_IMG, N_TXT, S) and the sigma schedule
#   (build_flux1_sigma_schedule) are comptime constants.  Steps/cfg/seed/resolution
#   require a recompile to change.  When the runner gains runtime dispatch,
#   remove them from the fixed list and thread from `req_prompt` below.
#
# ──────────────────────────────────────────────────────────────────────────────
# Text conditioning: pre-encoded T5 sidecar route.
#
#   Chroma uses T5-XXL as its text encoder.  The SentencePiece T5 tokenizer is
#   not yet in-tree as pure Mojo (as of 2026-06-08).  Instead, the adapter
#   follows the same pattern as anima_sample_cli.mojo: the sample_prompts JSON
#   entry carries:
#
#     caps.positive  — path to a safetensors file with key "t5_cond"
#                      shape [1, 512, 4096] BF16 (or any dtype castable to BF16)
#     caps.negative  — path to a safetensors file with key "t5_uncond"
#                      shape [1, 512, 4096] BF16
#
#   This is exactly the same format as the standalone runner's
#   chroma_embeddings.safetensors. One file can carry both keys (positive
#   and negative pointing to the same file), or they can be split.
#
#   To generate sidecars: run `t5_embed_chroma.py` (a small Python helper that
#   calls HF T5TokenizerFast + T5EncoderModel and saves the [1,512,4096]
#   embeddings).  The UI bridge must set precache_required: false in the JSON so
#   validation is skipped for the text fields (the runner reads caps directly).
#
# ──────────────────────────────────────────────────────────────────────────────
# Generate path: REAL, not a stub.
#   Calls the full chroma_pipeline_1024_multistep.mojo generate chain:
#     T5 sidecar load → ChromaShared load → BlockLoader → RoPE tables →
#     noise init → 30-step CFG Euler loop → unpack → FLUX VAE decode → PNG.
#   The only delta vs the standalone runner: t5_cond and t5_uncond come from
#   the per-prompt sidecar paths in caps_pos/caps_neg, and the output PNG is
#   argv[5] rather than the comptime OUT_PNG constant.
#
# Build:
#   cd /home/alex/mojodiffusion && pixi run mojo build -I . \
#     -Xlinker -lm -Xlinker -lcuda \
#     serenitymojo/pipeline/chroma_sample_cli.mojo \
#     -o /tmp/chroma_sample_cli

from std.sys import argv
from std.gpu.host import DeviceContext
from std.math import cos as fcos, exp as fexp, log as flog, sin as fsin, sqrt
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.models.dit.chroma_contract import (
    CHROMA_SINGLE_DIT_CHECKPOINT,
    CHROMA_DIT_APPROX_IN,
    CHROMA_DIT_APPROX_HIDDEN,
    CHROMA_DIT_APPROX_LAYERS,
    CHROMA_DIT_HIDDEN,
    CHROMA_DIT_HEADS,
    CHROMA_DIT_HEAD_DIM,
    CHROMA_DIT_DOUBLE_BLOCKS,
    CHROMA_DIT_SINGLE_BLOCKS,
    CHROMA_DIT_MOD_INDEX,
    CHROMA_IMAGE_TOKENS,
    CHROMA_T5_SEQ_LEN,
    CHROMA_LATENT_H,
    CHROMA_LATENT_W,
    CHROMA_LATENT_CHANNELS,
    CHROMA_PACK_PATCH,
    CHROMA_PATCH_GRID_H,
    CHROMA_PATCH_GRID_W,
    CHROMA_PATCH_VECTOR_DIM,
)
from serenitymojo.models.dit.flux1_dit import build_flux1_rope_tables
from serenitymojo.models.vae.ldm_decoder import load_flux1_ldm_decoder
from serenitymojo.offload.block_loader import BlockLoader, Block, unload_block
from serenitymojo.ops.activations import gelu, silu
from serenitymojo.ops.attention import sdpa_nomask
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.elementwise import modulate, residual_gate
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import layer_norm, rms_norm
from serenitymojo.ops.random import randn
from serenitymojo.ops.rope import rope_interleaved
from serenitymojo.ops.tensor_algebra import add, concat, mul_scalar, permute, reshape, slice
from serenitymojo.sampling.flux1_dev import build_flux1_sigma_schedule
from serenitymojo.image.png import save_png, ValueRange
from serenitymojo.training.sample_prompt_config import (
    SamplePrompt, SamplePromptConfig, read_sample_prompt_config,
)


# ── Model paths (comptime; edit to override or add config-file support) ────────
comptime CHROMA_CKPT = CHROMA_SINGLE_DIT_CHECKPOINT
comptime VAE_PATH = "/home/alex/.serenity/models/vaes/ae.safetensors"

# ── Sampler constants (comptime-fixed today; see header) ─────────────────────
comptime NUM_STEPS = 30
comptime GUIDANCE = Float32(4.0)
comptime SEED = UInt64(42)

# ── Geometry constants (from chroma_contract) ─────────────────────────────────
comptime LH = CHROMA_LATENT_H          # 128
comptime LW = CHROMA_LATENT_W          # 128
comptime LC = CHROMA_LATENT_CHANNELS   # 16
comptime PACK = CHROMA_PACK_PATCH      # 2
comptime PGH = CHROMA_PATCH_GRID_H     # 64
comptime PGW = CHROMA_PATCH_GRID_W     # 64
comptime N_IMG = CHROMA_IMAGE_TOKENS   # 4096
comptime N_TXT = CHROMA_T5_SEQ_LEN    # 512
comptime S = N_IMG + N_TXT             # 4608

comptime N_DBL = CHROMA_DIT_DOUBLE_BLOCKS  # 19
comptime N_SGL = CHROMA_DIT_SINGLE_BLOCKS  # 38
comptime MOD_IDX = CHROMA_DIT_MOD_INDEX    # 344
comptime HIDDEN = CHROMA_DIT_HIDDEN        # 3072
comptime HEADS = CHROMA_DIT_HEADS          # 24
comptime HEAD_DIM = CHROMA_DIT_HEAD_DIM    # 128

comptime MOD_SGL_OFF = 0
comptime MOD_DBL_IMG_OFF = 3 * N_SGL
comptime MOD_DBL_TXT_OFF = MOD_DBL_IMG_OFF + 6 * N_DBL


# ── Helpers (verbatim from chroma_pipeline_1024_multistep.mojo) ────────────────

def _clone(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var dev = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    ctx.enqueue_copy(dst_buf=dev, src_buf=x.buf)
    ctx.synchronize()
    return Tensor(dev^, x.shape(), x.dtype())


def _linear_b(x: Tensor, w: Tensor, b: Tensor, ctx: DeviceContext) raises -> Tensor:
    return linear(x, w, Optional[Tensor](_clone(b, ctx)), ctx)


struct ChromaShared(Movable):
    var weights: List[ArcPointer[Tensor]]
    var name_to_idx: Dict[String, Int]

    def __init__(out self, var weights: List[ArcPointer[Tensor]], var name_to_idx: Dict[String, Int]):
        self.weights = weights^
        self.name_to_idx = name_to_idx^

    @staticmethod
    def load(path: String, ctx: DeviceContext) raises -> ChromaShared:
        var sharded = ShardedSafeTensors.open(path)
        var weights = List[ArcPointer[Tensor]]()
        var name_to_idx = Dict[String, Int]()
        for ref nm in sharded.names():
            var is_shared = (
                nm.startswith("distilled_guidance_layer.")
                or nm == "x_embedder.weight"
                or nm == "x_embedder.bias"
                or nm == "context_embedder.weight"
                or nm == "context_embedder.bias"
                or nm == "proj_out.weight"
                or nm == "proj_out.bias"
            )
            if not is_shared:
                continue
            var tv = sharded.tensor_view(nm)
            var t = Tensor.from_view(tv, ctx)
            var idx = len(weights)
            weights.append(ArcPointer(t^))
            name_to_idx[nm] = idx
        return ChromaShared(weights^, name_to_idx^)

    def _w(self, name: String) raises -> ref [self.weights] Tensor:
        if name not in self.name_to_idx:
            raise Error(String("ChromaShared missing weight: ") + name)
        var idx = self.name_to_idx[name]
        return self.weights[idx][]


def _bw(ref block: Block, name: String) raises -> ref [block] Tensor:
    if name not in block:
        raise Error(String("Block missing weight: ") + name)
    return block[name][]


def _ones_vec(d: Int, dtype: STDtype, ctx: DeviceContext) raises -> Tensor:
    var vals = List[Float32]()
    for _ in range(d):
        vals.append(1.0)
    var sh = List[Int]()
    sh.append(d)
    return Tensor.from_host(vals, sh^, dtype, ctx)


def _zeros_vec(d: Int, dtype: STDtype, ctx: DeviceContext) raises -> Tensor:
    var vals = List[Float32]()
    for _ in range(d):
        vals.append(0.0)
    var sh = List[Int]()
    sh.append(d)
    return Tensor.from_host(vals, sh^, dtype, ctx)


def _layer_norm_no_affine(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var shape = x.shape()
    var d = shape[len(shape) - 1]
    var ones = _ones_vec(d, x.dtype(), ctx)
    var zeros = _zeros_vec(d, x.dtype(), ctx)
    return layer_norm(x, ones, zeros, Float32(1.0e-6), ctx)


def _modulate_pre(x: Tensor, shift: Tensor, scale: Tensor, ctx: DeviceContext) raises -> Tensor:
    var normed = _layer_norm_no_affine(x, ctx)
    return modulate(normed, scale, shift, ctx)


def _pooled_row(pooled_temb: Tensor, row: Int, ctx: DeviceContext) raises -> Tensor:
    var part = slice(pooled_temb, 1, row, 1, ctx)
    var sh = List[Int]()
    sh.append(HIDDEN)
    return reshape(part, sh^, ctx)


def _to_bshd(x: Tensor, n: Int, ctx: DeviceContext) raises -> Tensor:
    var sh = List[Int]()
    sh.append(1)
    sh.append(n)
    sh.append(HEADS)
    sh.append(HEAD_DIM)
    return reshape(x, sh^, ctx)


def _from_bshd(x: Tensor, n: Int, ctx: DeviceContext) raises -> Tensor:
    var sh = List[Int]()
    sh.append(1)
    sh.append(n)
    sh.append(HIDDEN)
    return reshape(x, sh^, ctx)


def _sinusoid_values(t: Float32, dim: Int) raises -> List[Float32]:
    if dim % 2 != 0:
        raise Error("_sinusoid_values: dim must be even")
    var half = dim // 2
    var vals = List[Float32]()
    var neg_ln_mp = -flog(Float32(10000.0))
    for i in range(half):
        var freq = fexp(neg_ln_mp * (Float32(i) / Float32(half)))
        vals.append(fcos(t * freq))
    for i in range(half):
        var freq = fexp(neg_ln_mp * (Float32(i) / Float32(half)))
        vals.append(fsin(t * freq))
    return vals^


def _mod_proj_values(out_dim: Int, dim: Int) raises -> List[Float32]:
    if dim % 2 != 0:
        raise Error("_mod_proj_values: dim must be even")
    var half = dim // 2
    var vals = List[Float32]()
    var neg_ln_mp = -flog(Float32(10000.0))
    for row in range(out_dim):
        var t = Float32(row) * Float32(1000.0)
        for i in range(half):
            var freq = fexp(neg_ln_mp * (Float32(i) / Float32(half)))
            vals.append(fcos(t * freq))
        for i in range(half):
            var freq = fexp(neg_ln_mp * (Float32(i) / Float32(half)))
            vals.append(fsin(t * freq))
    return vals^


def _build_approximator_input(timestep: Float32, ctx: DeviceContext) raises -> Tensor:
    comptime NUM_CH = CHROMA_DIT_APPROX_IN // 4  # 16
    var ts_proj = _sinusoid_values(timestep * Float32(1000.0), NUM_CH)
    var guid_proj = _sinusoid_values(Float32(0.0), NUM_CH)
    var mod_proj = _mod_proj_values(MOD_IDX, 2 * NUM_CH)

    var vals = List[Float32]()
    for row in range(MOD_IDX):
        for i in range(NUM_CH):
            vals.append(ts_proj[i])
        for i in range(NUM_CH):
            vals.append(guid_proj[i])
        for i in range(2 * NUM_CH):
            vals.append(mod_proj[row * (2 * NUM_CH) + i])

    var sh = List[Int]()
    sh.append(1)
    sh.append(MOD_IDX)
    sh.append(CHROMA_DIT_APPROX_IN)
    return Tensor.from_host(vals, sh^, STDtype.BF16, ctx)


def _approximator_forward(ref shared: ChromaShared, x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var h = _linear_b(
        x,
        shared._w("distilled_guidance_layer.in_proj.weight"),
        shared._w("distilled_guidance_layer.in_proj.bias"),
        ctx,
    )
    for i in range(CHROMA_DIT_APPROX_LAYERS):
        var norm_key = String("distilled_guidance_layer.norms.") + String(i) + ".weight"
        var n = rms_norm(h, shared._w(norm_key), Float32(1.0e-6), ctx)
        var h1 = _linear_b(
            n,
            shared._w(String("distilled_guidance_layer.layers.") + String(i) + ".linear_1.weight"),
            shared._w(String("distilled_guidance_layer.layers.") + String(i) + ".linear_1.bias"),
            ctx,
        )
        var a = silu(h1, ctx)
        var h2 = _linear_b(
            a,
            shared._w(String("distilled_guidance_layer.layers.") + String(i) + ".linear_2.weight"),
            shared._w(String("distilled_guidance_layer.layers.") + String(i) + ".linear_2.bias"),
            ctx,
        )
        h = add(h, h2, ctx)
    return _linear_b(
        h,
        shared._w("distilled_guidance_layer.out_proj.weight"),
        shared._w("distilled_guidance_layer.out_proj.bias"),
        ctx,
    )


def _double_block(
    block_idx: Int,
    img: Tensor,
    txt: Tensor,
    img_mod_slice: Tensor,
    txt_mod_slice: Tensor,
    rope_cos: Tensor,
    rope_sin: Tensor,
    ref loader_block: Block,
    ctx: DeviceContext,
) raises -> Tuple[Tensor, Tensor]:
    var p = String("transformer_blocks.") + String(block_idx)

    var img_shift1 = _pooled_row(img_mod_slice, 0, ctx)
    var img_scale1 = _pooled_row(img_mod_slice, 1, ctx)
    var img_gate1  = _pooled_row(img_mod_slice, 2, ctx)
    var img_shift2 = _pooled_row(img_mod_slice, 3, ctx)
    var img_scale2 = _pooled_row(img_mod_slice, 4, ctx)
    var img_gate2  = _pooled_row(img_mod_slice, 5, ctx)

    var txt_shift1 = _pooled_row(txt_mod_slice, 0, ctx)
    var txt_scale1 = _pooled_row(txt_mod_slice, 1, ctx)
    var txt_gate1  = _pooled_row(txt_mod_slice, 2, ctx)
    var txt_shift2 = _pooled_row(txt_mod_slice, 3, ctx)
    var txt_scale2 = _pooled_row(txt_mod_slice, 4, ctx)
    var txt_gate2  = _pooled_row(txt_mod_slice, 5, ctx)

    var img_norm = _modulate_pre(img, img_shift1, img_scale1, ctx)
    var txt_norm = _modulate_pre(txt, txt_shift1, txt_scale1, ctx)

    var img_q = _to_bshd(
        _linear_b(img_norm, _bw(loader_block, p + ".attn.to_q.weight"), _bw(loader_block, p + ".attn.to_q.bias"), ctx),
        N_IMG, ctx
    )
    var img_k = _to_bshd(
        _linear_b(img_norm, _bw(loader_block, p + ".attn.to_k.weight"), _bw(loader_block, p + ".attn.to_k.bias"), ctx),
        N_IMG, ctx
    )
    var img_v = _to_bshd(
        _linear_b(img_norm, _bw(loader_block, p + ".attn.to_v.weight"), _bw(loader_block, p + ".attn.to_v.bias"), ctx),
        N_IMG, ctx
    )
    var txt_q = _to_bshd(
        _linear_b(txt_norm, _bw(loader_block, p + ".attn.add_q_proj.weight"), _bw(loader_block, p + ".attn.add_q_proj.bias"), ctx),
        N_TXT, ctx
    )
    var txt_k = _to_bshd(
        _linear_b(txt_norm, _bw(loader_block, p + ".attn.add_k_proj.weight"), _bw(loader_block, p + ".attn.add_k_proj.bias"), ctx),
        N_TXT, ctx
    )
    var txt_v = _to_bshd(
        _linear_b(txt_norm, _bw(loader_block, p + ".attn.add_v_proj.weight"), _bw(loader_block, p + ".attn.add_v_proj.bias"), ctx),
        N_TXT, ctx
    )

    img_q = rms_norm(img_q, _bw(loader_block, p + ".attn.norm_q.weight"), Float32(1.0e-6), ctx)
    img_k = rms_norm(img_k, _bw(loader_block, p + ".attn.norm_k.weight"), Float32(1.0e-6), ctx)
    txt_q = rms_norm(txt_q, _bw(loader_block, p + ".attn.norm_added_q.weight"), Float32(1.0e-6), ctx)
    txt_k = rms_norm(txt_k, _bw(loader_block, p + ".attn.norm_added_k.weight"), Float32(1.0e-6), ctx)

    var q = concat(1, ctx, txt_q, img_q)
    var k = concat(1, ctx, txt_k, img_k)
    var v = concat(1, ctx, txt_v, img_v)
    q = rope_interleaved(q, rope_cos, rope_sin, ctx)
    k = rope_interleaved(k, rope_cos, rope_sin, ctx)
    var att = sdpa_nomask[1, S, HEADS, HEAD_DIM](
        q, k, v, Float32(1.0) / sqrt(Float32(HEAD_DIM)), ctx
    )

    var txt_att_bshd = slice(att, 1, 0, N_TXT, ctx)
    var img_att_bshd = slice(att, 1, N_TXT, N_IMG, ctx)
    var img_att = _from_bshd(img_att_bshd, N_IMG, ctx)
    var txt_att = _from_bshd(txt_att_bshd, N_TXT, ctx)

    var img_o = _linear_b(img_att, _bw(loader_block, p + ".attn.to_out.0.weight"), _bw(loader_block, p + ".attn.to_out.0.bias"), ctx)
    var txt_o = _linear_b(txt_att, _bw(loader_block, p + ".attn.to_add_out.weight"), _bw(loader_block, p + ".attn.to_add_out.bias"), ctx)

    var img_r = residual_gate(img, img_gate1, img_o, ctx)
    var txt_r = residual_gate(txt, txt_gate1, txt_o, ctx)

    var img_ff_in = _modulate_pre(img_r, img_shift2, img_scale2, ctx)
    var img_ff = _linear_b(img_ff_in, _bw(loader_block, p + ".ff.net.0.proj.weight"), _bw(loader_block, p + ".ff.net.0.proj.bias"), ctx)
    img_ff = gelu(img_ff, ctx)
    img_ff = _linear_b(img_ff, _bw(loader_block, p + ".ff.net.2.weight"), _bw(loader_block, p + ".ff.net.2.bias"), ctx)
    var img_final = residual_gate(img_r, img_gate2, img_ff, ctx)

    var txt_ff_in = _modulate_pre(txt_r, txt_shift2, txt_scale2, ctx)
    var txt_ff = _linear_b(txt_ff_in, _bw(loader_block, p + ".ff_context.net.0.proj.weight"), _bw(loader_block, p + ".ff_context.net.0.proj.bias"), ctx)
    txt_ff = gelu(txt_ff, ctx)
    txt_ff = _linear_b(txt_ff, _bw(loader_block, p + ".ff_context.net.2.weight"), _bw(loader_block, p + ".ff_context.net.2.bias"), ctx)
    var txt_final = residual_gate(txt_r, txt_gate2, txt_ff, ctx)

    return (img_final^, txt_final^)


def _single_block(
    block_idx: Int,
    x: Tensor,
    single_mod_slice: Tensor,
    rope_cos: Tensor,
    rope_sin: Tensor,
    ref loader_block: Block,
    ctx: DeviceContext,
) raises -> Tensor:
    var p = String("single_transformer_blocks.") + String(block_idx)

    var shift = _pooled_row(single_mod_slice, 0, ctx)
    var scale = _pooled_row(single_mod_slice, 1, ctx)
    var gate  = _pooled_row(single_mod_slice, 2, ctx)

    var x_norm = _modulate_pre(x, shift, scale, ctx)

    var q = _to_bshd(
        _linear_b(x_norm, _bw(loader_block, p + ".attn.to_q.weight"), _bw(loader_block, p + ".attn.to_q.bias"), ctx),
        S, ctx
    )
    var k = _to_bshd(
        _linear_b(x_norm, _bw(loader_block, p + ".attn.to_k.weight"), _bw(loader_block, p + ".attn.to_k.bias"), ctx),
        S, ctx
    )
    var v = _to_bshd(
        _linear_b(x_norm, _bw(loader_block, p + ".attn.to_v.weight"), _bw(loader_block, p + ".attn.to_v.bias"), ctx),
        S, ctx
    )

    q = rms_norm(q, _bw(loader_block, p + ".attn.norm_q.weight"), Float32(1.0e-6), ctx)
    k = rms_norm(k, _bw(loader_block, p + ".attn.norm_k.weight"), Float32(1.0e-6), ctx)

    q = rope_interleaved(q, rope_cos, rope_sin, ctx)
    k = rope_interleaved(k, rope_cos, rope_sin, ctx)

    var att = sdpa_nomask[1, S, HEADS, HEAD_DIM](
        q, k, v, Float32(1.0) / sqrt(Float32(HEAD_DIM)), ctx
    )
    var att_flat = _from_bshd(att, S, ctx)

    var mlp = _linear_b(x_norm, _bw(loader_block, p + ".proj_mlp.weight"), _bw(loader_block, p + ".proj_mlp.bias"), ctx)
    mlp = gelu(mlp, ctx)

    var cat_out = concat(2, ctx, att_flat, mlp)
    var out = _linear_b(cat_out, _bw(loader_block, p + ".proj_out.weight"), _bw(loader_block, p + ".proj_out.bias"), ctx)
    return residual_gate(x, gate, out, ctx)


def _chroma_forward(
    shared: ChromaShared,
    loader: BlockLoader,
    img_packed: Tensor,
    txt: Tensor,
    timestep: Float32,
    rope_cos: Tensor,
    rope_sin: Tensor,
    ctx: DeviceContext,
) raises -> Tensor:
    var approx_in = _build_approximator_input(timestep, ctx)
    var pooled_temb = _approximator_forward(shared, approx_in, ctx)

    var img = _linear_b(
        img_packed,
        shared._w("x_embedder.weight"),
        shared._w("x_embedder.bias"),
        ctx,
    )
    var x_txt = _linear_b(
        txt,
        shared._w("context_embedder.weight"),
        shared._w("context_embedder.bias"),
        ctx,
    )

    for i in range(N_DBL):
        var prefix = String("transformer_blocks.") + String(i)
        loader.prefetch_block(prefix)
        var block = loader.load_block(prefix, ctx)

        var img_mod_s = slice(pooled_temb, 1, MOD_DBL_IMG_OFF + 6 * i, 6, ctx)
        var txt_mod_s = slice(pooled_temb, 1, MOD_DBL_TXT_OFF + 6 * i, 6, ctx)

        var res = _double_block(i, img, x_txt, img_mod_s, txt_mod_s, rope_cos, rope_sin, block, ctx)
        img = _clone(res[0], ctx)
        x_txt = _clone(res[1], ctx)

        unload_block(block^)

    var x = concat(1, ctx, x_txt, img)

    for i in range(N_SGL):
        var prefix = String("single_transformer_blocks.") + String(i)
        loader.prefetch_block(prefix)
        var block = loader.load_block(prefix, ctx)

        var sgl_mod_s = slice(pooled_temb, 1, MOD_SGL_OFF + 3 * i, 3, ctx)

        x = _single_block(i, x, sgl_mod_s, rope_cos, rope_sin, block, ctx)

        unload_block(block^)

    var img_out = slice(x, 1, N_TXT, N_IMG, ctx)

    var norm_shift = _pooled_row(pooled_temb, MOD_IDX - 2, ctx)
    var norm_scale = _pooled_row(pooled_temb, MOD_IDX - 1, ctx)
    var normed = _layer_norm_no_affine(img_out, ctx)
    var modulated = modulate(normed, norm_scale, norm_shift, ctx)

    return _linear_b(
        modulated,
        shared._w("proj_out.weight"),
        shared._w("proj_out.bias"),
        ctx,
    )


def _pack_latent(z: Tensor, ctx: DeviceContext) raises -> Tensor:
    var s6 = List[Int]()
    s6.append(1)
    s6.append(LC)
    s6.append(PGH)
    s6.append(PACK)
    s6.append(PGW)
    s6.append(PACK)
    var t6 = reshape(z, s6^, ctx)
    var perm = List[Int]()
    perm.append(0)
    perm.append(2)
    perm.append(4)
    perm.append(1)
    perm.append(3)
    perm.append(5)
    var tp = permute(t6, perm^, ctx)
    var sp = List[Int]()
    sp.append(1)
    sp.append(N_IMG)
    sp.append(LC * PACK * PACK)
    return reshape(tp, sp^, ctx)


def _unpack_latent(packed: Tensor, ctx: DeviceContext) raises -> Tensor:
    var s6 = List[Int]()
    s6.append(1)
    s6.append(PGH)
    s6.append(PGW)
    s6.append(LC)
    s6.append(PACK)
    s6.append(PACK)
    var t6 = reshape(packed, s6^, ctx)
    var perm = List[Int]()
    perm.append(0)
    perm.append(3)
    perm.append(1)
    perm.append(4)
    perm.append(2)
    perm.append(5)
    var tp = permute(t6, perm^, ctx)
    var sp = List[Int]()
    sp.append(1)
    sp.append(LC)
    sp.append(LH)
    sp.append(LW)
    return reshape(tp, sp^, ctx)


# ── Load T5 embeddings from a per-prompt sidecar ──────────────────────────────
# The sidecar is a safetensors file with key "t5_cond" and/or "t5_uncond"
# at shape [1, 512, 4096] (any dtype; will be cast to BF16).
# If both keys live in the same file, pass the same path for both.
# For the negative/uncond side: also try key "t5_cond" as a fallback (so a
# single-key file can serve as either side).
def _load_t5_embedding(path: String, key: String, ctx: DeviceContext) raises -> Tensor:
    var st = ShardedSafeTensors.open(path)
    var t = Tensor.from_view(st.tensor_view(key), ctx)
    return cast_tensor(t, STDtype.BF16, ctx)


def _load_t5_embedding_with_fallback(
    path: String, primary_key: String, fallback_key: String, ctx: DeviceContext
) raises -> Tensor:
    var st = ShardedSafeTensors.open(path)
    # Try primary key first.
    var names = st.names()
    var has_primary = False
    for ref nm in names:
        if nm == primary_key:
            has_primary = True
    if has_primary:
        var t = Tensor.from_view(st.tensor_view(primary_key), ctx)
        return cast_tensor(t, STDtype.BF16, ctx)
    # Fall back to alternative key.
    var t = Tensor.from_view(st.tensor_view(fallback_key), ctx)
    return cast_tensor(t, STDtype.BF16, ctx)


# ── Prompt selection helpers (verbatim pattern from qwenimage_sample_cli.mojo) ─

def _select_prompt(sample_cfg: SamplePromptConfig, wanted: String) raises -> SamplePrompt:
    if len(sample_cfg.prompts) == 0:
        raise Error("chroma_sample_cli: sample prompt JSON has no prompts")
    if wanted == String(""):
        return sample_cfg.prompts[0].copy()
    for i in range(len(sample_cfg.prompts)):
        if sample_cfg.prompts[i].label == wanted:
            return sample_cfg.prompts[i].copy()
    raise Error(String("chroma_sample_cli: prompt id not found: ") + wanted)


def _load_prompt_json(
    path: String, wanted: String,
    mut caps_pos: String, mut caps_neg: String,
) raises:
    var sample_cfg = read_sample_prompt_config(path)
    var p = _select_prompt(sample_cfg, wanted)
    if p.frames != 1:
        raise Error("chroma_sample_cli: only image prompts (frames=1) are supported")
    if p.caps_pos == String(""):
        raise Error(
            String("chroma_sample_cli: prompt '") + p.label
            + String("' has no caps.positive — Chroma needs a pre-encoded T5 sidecar")
            + String(" (set precache_required: false and provide caps.positive/caps.negative)")
        )
    caps_pos = p.caps_pos.copy()
    # caps_neg is optional: if absent, use the same sidecar as pos (uncond key
    # inside it, or fall back to pos embedding as no-op uncond at inference).
    if p.caps_neg != String(""):
        caps_neg = p.caps_neg.copy()
    else:
        caps_neg = p.caps_pos.copy()
        print(
            "  [info] caps.negative not set; using caps.positive sidecar for uncond too.",
            "  Check that the sidecar has a 't5_uncond' key, or supply a separate neg sidecar.",
        )
    # steps/cfg/seed/width/height are comptime-fixed today; log what the JSON
    # requested so the caller knows what was ignored.
    print(
        "  [info] sample prompt requests steps=", p.steps, "cfg=", p.cfg,
        "seed=", p.seed, "size=", p.width, "x", p.height,
        "-> all ignored (comptime fixed); caps_pos + caps_neg honored.",
    )
    print("  [info] prompt text (for reference):", p.prompt)
    if p.negative != String(""):
        print("  [info] negative text (for reference):", p.negative)


# ── Main entry ──────────────────────────────────────────────────────────────

def main() raises:
    var a = argv()
    if len(a) < 6:
        print(
            "usage: chroma_sample_cli <config.json> <lora|-> <sample_prompts.json>"
            " <prompt_id> <out.png>"
        )
        print("  argv[1] config   — accepted, ignored (model dirs are comptime)")
        print("  argv[2] lora     — accepted, ignored (no LoRA support yet)")
        print("  argv[3] prompts  — serenity.sample_prompts.v1 JSON")
        print("  argv[4] id       — prompt label, or '' for first")
        print("  argv[5] out.png  — output image path")
        print("")
        print("  The sample prompt must supply caps.positive (path to a safetensors")
        print("  file with key 't5_cond' [1,512,4096]) and optionally caps.negative")
        print("  (same format, key 't5_uncond').  Set precache_required: false.")
        raise Error("chroma_sample_cli: need exactly 5 arguments")

    # argv[1]: config — accepted, not used today.
    var _config_path = String(a[1])

    # argv[2]: lora path or sentinel; accepted, not used today.
    var lora_raw = String(a[2])
    var _lora_path = String("")
    if lora_raw != String("-") and lora_raw != String("base") and lora_raw != String(""):
        _lora_path = lora_raw
        print("[lora] path provided but ignored (Chroma LoRA not wired yet):", _lora_path)

    # argv[3]: sample prompts JSON
    var prompts_json = String(a[3])

    # argv[4]: prompt id
    var prompt_id = String(a[4])

    # argv[5]: output PNG
    var out_png = String(a[5])

    # Load caps paths from JSON.
    var caps_pos = String("")
    var caps_neg = String("")
    _load_prompt_json(prompts_json, prompt_id, caps_pos, caps_neg)

    print("=== Chroma sample CLI ===")
    print("  prompts:", prompts_json, " id:", prompt_id)
    print("  output:", out_png)
    print("  caps_pos:", caps_pos)
    print("  caps_neg:", caps_neg)

    var ctx = DeviceContext()

    # Stage 1: Load T5 embeddings from per-prompt sidecars.
    print("\n--- Stage 1: Load T5 embeddings from sidecars ---")
    var t5_cond = _load_t5_embedding(caps_pos, String("t5_cond"), ctx)
    var cs = t5_cond.shape()
    print("  t5_cond:", cs[0], cs[1], cs[2])
    # For uncond: try "t5_uncond" first; fall back to "t5_cond" (same-file case).
    var t5_uncond = _load_t5_embedding_with_fallback(
        caps_neg, String("t5_uncond"), String("t5_cond"), ctx
    )
    var us = t5_uncond.shape()
    print("  t5_uncond:", us[0], us[1], us[2])

    # Stage 2: Load shared Chroma weights.
    print("\n--- Stage 2: Load Chroma shared weights ---")
    var shared = ChromaShared.load(String(CHROMA_CKPT), ctx)
    print("  Shared weights loaded:", len(shared.weights), "tensors")

    # Stage 3: Open block loader.
    print("\n--- Stage 3: Open block loader ---")
    var loader = BlockLoader.open(String(CHROMA_CKPT))
    print("  Block loader ready")

    # Stage 4: Build RoPE tables.
    print("\n--- Stage 4: Build RoPE tables ---")
    var rope = build_flux1_rope_tables[N_IMG, N_TXT, HEADS, HEAD_DIM](
        PGH, PGW, ctx, STDtype.BF16
    )
    var rope_cos = _clone(rope[0], ctx)
    var rope_sin = _clone(rope[1], ctx)
    print("  RoPE cos shape:", rope_cos.shape()[0], rope_cos.shape()[1])

    # Stage 5: Init noise latent.
    print("\n--- Stage 5: Init noise latent [1,", LC, ",", LH, ",", LW, "] ---")
    var noise_shape = List[Int]()
    noise_shape.append(1)
    noise_shape.append(LC)
    noise_shape.append(LH)
    noise_shape.append(LW)
    var noise_nchw = randn(noise_shape^, SEED, STDtype.BF16, ctx)
    var img_packed = _pack_latent(noise_nchw, ctx)
    var psh = img_packed.shape()
    print("  Packed latent:", psh[0], psh[1], psh[2])

    # Stage 6: Build sigma schedule.
    print("\n--- Stage 6: Build schedule (", NUM_STEPS, "steps) ---")
    var sigmas = build_flux1_sigma_schedule(NUM_STEPS, N_IMG)
    print("  sigma[0]=", sigmas[0], " sigma[-1]=", sigmas[NUM_STEPS])

    # Stage 7: CFG Euler denoise loop.
    # This is the KEY delta from the standalone runner: t5_cond/t5_uncond come
    # from the per-prompt sidecars above, not the EMBEDDINGS_PATH comptime.
    print("\n--- Stage 7: CFG Euler denoise (", NUM_STEPS, "steps, CFG", GUIDANCE, ") ---")
    for step in range(NUM_STEPS):
        var t_curr = sigmas[step]
        var t_next = sigmas[step + 1]
        var dt = t_next - t_curr

        var pred_cond = _chroma_forward(
            shared, loader, img_packed, t5_cond, t_curr, rope_cos, rope_sin, ctx
        )
        var pred_uncond = _chroma_forward(
            shared, loader, img_packed, t5_uncond, t_curr, rope_cos, rope_sin, ctx
        )

        # CFG: pred = uncond + guidance * (cond - uncond)
        var neg_uncond = mul_scalar(pred_uncond, Float32(-1.0), ctx)
        var diff = add(pred_cond, neg_uncond, ctx)
        var scaled_diff = mul_scalar(diff, GUIDANCE, ctx)
        var pred = add(pred_uncond, scaled_diff, ctx)

        # Euler step: x = x + dt * pred
        var step_delta = mul_scalar(pred, dt, ctx)
        img_packed = add(img_packed, step_delta, ctx)

        if step == 0 or (step + 1) % 5 == 0 or step + 1 == NUM_STEPS:
            print("  step", step + 1, "/", NUM_STEPS, " t=", t_curr, " dt=", dt)

    # Stage 8: Unpack + VAE decode.
    print("\n--- Stage 8: Unpack + VAE decode ---")
    var latent = _unpack_latent(img_packed, ctx)
    var lsh = latent.shape()
    print("  Unpacked latent:", lsh[0], lsh[1], lsh[2], lsh[3])

    # Drop DiT weights before loading VAE.
    var _sink_shared = shared^
    var _sink_loader = loader^
    _ = _sink_shared
    _ = _sink_loader
    print("  DiT weights dropped")

    var latent_f32 = cast_tensor(latent, STDtype.F32, ctx)
    var vae = load_flux1_ldm_decoder[LH, LW](String(VAE_PATH), ctx)
    print("  VAE loaded")

    var rgb = vae.decode(latent_f32, ctx)
    var rsh = rgb.shape()
    print("  Decoded RGB:", rsh[0], rsh[1], rsh[2], rsh[3])

    # Stage 9: Save PNG.
    print("\n--- Stage 9: Save PNG ---")
    save_png(rgb, out_png, ctx, ValueRange.SIGNED)
    print("[done] saved:", out_png)

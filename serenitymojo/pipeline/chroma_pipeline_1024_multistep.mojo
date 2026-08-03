# chroma_pipeline_1024_multistep.mojo — Chroma 1024x1024 30-step CFG pipeline.
#
# Faithful port of chroma_gen.rs (inference-flame):
#   1. Load cached T5 sidecar (cond + uncond) — skip T5 port.
#   2. Init noise latent [1, 16, 128, 128] BF16 (Box-Muller, seed=42).
#   3. 30-step shifted rectified-flow CFG Euler loop:
#      - FLUX get_schedule (mu linear 0.5..1.15, shift=True)
#      - approximator forward (distilled_guidance_layer) once per step
#      - Full 19 double + 38 single blocks via BlockLoader
#      - CFG: uncond + guidance * (cond - uncond), guidance=4.0
#      - Euler: x_next = x + dt * pred
#   4. Unpack latent, decode via FLUX VAE (ae.safetensors).
#   5. Save output/chroma_1024_30step.png.
#
# Text conditioning: sidecar route (no T5 port required).
#   Embeddings path: output/chroma_embeddings.safetensors
#   Keys: t5_cond [1,512,4096], t5_uncond [1,512,4096]
#
# Modulation layout (from chroma_dit.rs, transformer_chroma.py:560-570):
#   mod_index_length = 3*38 + 6*2*19 + 2 = 114 + 228 + 2 = 344
#   single block i:  pooled_temb[:, 3*i : 3*i+3]
#   double img i:    pooled_temb[:, 114 + 6*i : 114 + 6*i + 6]
#   double txt i:    pooled_temb[:, 114+114 + 6*i : 114+114 + 6*i + 6]
#   norm_out:        pooled_temb[:, -2:]
#
# Build:
#   pixi run mojo build -I . -Xlinker -lm \
#     serenitymojo/pipeline/chroma_pipeline_1024_multistep.mojo \
#     -o /tmp/chroma_1024 && /tmp/chroma_1024

from std.collections import Optional
from std.gpu.host import DeviceContext
from std.math import cos as fcos, exp as fexp, log as flog, sin as fsin, sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
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
from serenitymojo.models.chroma.chroma_lora_overlay import ChromaLoraOverlay
from serenitymojo.models.vae.ldm_decoder import load_flux1_ldm_decoder
from serenitymojo.offload.block_loader import BlockLoader, Block, unload_block
from serenitymojo.ops.activations import gelu, silu
from serenitymojo.ops.attention import sdpa_qwen_flash_padmask
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.elementwise import modulate, residual_gate
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import layer_norm, rms_norm
from serenitymojo.ops.random import randn
from serenitymojo.ops.rope import rope_interleaved
from serenitymojo.ops.tensor_algebra import add, concat, mul_scalar, permute, reshape, slice
from serenitymojo.sampling.flux1_dev import build_flux1_sigma_schedule
from serenitymojo.image.png import save_png, ValueRange
from std.memory import ArcPointer


# ── Configuration ──────────────────────────────────────────────────────────────
comptime EMBEDDINGS_PATH = "output/chroma_embeddings.safetensors"
comptime CHROMA_CKPT = CHROMA_SINGLE_DIT_CHECKPOINT
comptime VAE_PATH = "models/vae/flux.safetensors"
comptime OUT_PNG = "output/chroma_1024_30step.png"

comptime NUM_STEPS = 30
comptime GUIDANCE = Float32(4.0)
comptime SEED = UInt64(42)

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

# Modulation layout offsets (matches chroma_dit.rs forward_inner)
comptime MOD_SGL_OFF = 0                   # single block i: [3*i, 3*i+3)
comptime MOD_DBL_IMG_OFF = 3 * N_SGL      # double img: [114 + 6*i, ...)
comptime MOD_DBL_TXT_OFF = MOD_DBL_IMG_OFF + 6 * N_DBL  # double txt: [228 + ..., ...)


# ── Stats helper ───────────────────────────────────────────────────────────────
def _stats(name: String, t: Tensor, ctx: DeviceContext) raises:
    var h = t.to_host(ctx)
    var n = len(h)
    if n == 0:
        print("  [stat]", name, "EMPTY")
        return
    var s: Float64 = 0.0
    var amax: Float64 = 0.0
    for i in range(n):
        var v = Float64(h[i])
        s += v
        var av = v if v >= 0.0 else -v
        if av > amax:
            amax = av
    print("  [stat]", name, "mean=", Float32(s / Float64(n)), "absmax=", Float32(amax))


# ── Read a 1-element F32 real-text-length sidecar key (MJ-1048) ──────────────
# The chroma_encode.rs sidecar carries "cond_real_len"/"uncond_real_len" (the
# unpadded T5 token count) alongside t5_cond/t5_uncond. Old sidecars written
# before that key existed default to the full N_TXT (512) — printed LOUD, since
# that means the ~490 T5 pad rows are NOT masked (the pre-fix behavior).
def _sidecar_real_len(
    ref st: ShardedSafeTensors, key: String, ctx: DeviceContext
) raises -> Int:
    var has_key = False
    for ref nm in st.names():
        if nm == key:
            has_key = True
    if not has_key:
        print(
            "  [WARN] sidecar has no key", key,
            "— OLD sidecar without the T5 real length; defaulting real_txt_len to",
            N_TXT, "(T5 pad rows will NOT be masked; re-run chroma_encode.rs to fix)",
        )
        return N_TXT
    var h = Tensor.from_view(st.tensor_view(key), ctx).to_host(ctx)
    if len(h) < 1:
        print("  [WARN] sidecar key", key, "is empty; defaulting real_txt_len to", N_TXT)
        return N_TXT
    var rl = Int(h[0])
    if rl < 1 or rl > N_TXT:
        print(
            "  [WARN] sidecar", key, "=", rl, "out of range [1,", N_TXT,
            "]; defaulting real_txt_len to", N_TXT,
        )
        return N_TXT
    print("  [info] sidecar", key, "=", rl, "of", N_TXT)
    return rl


# ── Deep-copy a Tensor (needed for bias operands) ────────────────────────────
def _clone(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var dev = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    ctx.enqueue_copy(dst_buf=dev, src_buf=x.buf)
    ctx.synchronize()
    return Tensor(dev^, x.shape(), x.dtype())


# ── Linear helper with cloned bias ───────────────────────────────────────────
def _linear_b(x: Tensor, w: Tensor, b: Tensor, ctx: DeviceContext) raises -> Tensor:
    return linear(x, w, Optional[Tensor](_clone(b, ctx)), ctx)


def _linear_chroma_lora(
    x: Tensor,
    w: Tensor,
    b: Tensor,
    base_name: String,
    lora: Optional[ArcPointer[ChromaLoraOverlay]],
    ctx: DeviceContext,
) raises -> Tensor:
    """Creator-equivalent runtime patch: base linear + scale*B(A(x)).

    The no-LoRA arm calls the original `_linear_b` and returns it untouched.
    """
    var base = _linear_b(x, w, b, ctx)
    if lora and lora.value()[].has(base_name):
        var delta = lora.value()[].delta(x, base_name, ctx)
        return add(base, delta, ctx)
    return base^


# ── Shared weight loading (approximator + x_embedder + context_embedder + proj_out) ──
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


# ── Block weight lookup ────────────────────────────────────────────────────────
def _bw(ref block: Block, name: String) raises -> ref [block] Tensor:
    if name not in block:
        raise Error(String("Block missing weight: ") + name)
    return block[name][]


# ── Ones/zeros vectors (for no-affine LayerNorm) ──────────────────────────────
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


# ── LayerNorm no-affine ────────────────────────────────────────────────────────
def _layer_norm_no_affine(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var shape = x.shape()
    var d = shape[len(shape) - 1]
    var ones = _ones_vec(d, x.dtype(), ctx)
    var zeros = _zeros_vec(d, x.dtype(), ctx)
    return layer_norm(x, ones, zeros, Float32(1.0e-6), ctx)


# ── modulate_pre: LayerNorm(x) * (1 + scale) + shift ─────────────────────────
# scale/shift are [1, HIDDEN], x is [1, N, HIDDEN]
def _modulate_pre(x: Tensor, shift: Tensor, scale: Tensor, ctx: DeviceContext) raises -> Tensor:
    var normed = _layer_norm_no_affine(x, ctx)
    return modulate(normed, scale, shift, ctx)


# ── Slice a row from a 3D tensor [..., N, HIDDEN] -> [HIDDEN] (1D) ───────────
# modulate() expects scale/shift to be [D] (rank-1).
def _pooled_row(pooled_temb: Tensor, row: Int, ctx: DeviceContext) raises -> Tensor:
    var part = slice(pooled_temb, 1, row, 1, ctx)
    var sh = List[Int]()
    sh.append(HIDDEN)
    return reshape(part, sh^, ctx)


# ── Reshape [1, N, HIDDEN] -> [1, N, HEADS, HEAD_DIM] (BSHD for sdpa_nomask) ─
# sdpa_nomask expects [B, S, H, Dh] — no permute, just split the last dim.
def _to_bshd(x: Tensor, n: Int, ctx: DeviceContext) raises -> Tensor:
    var sh = List[Int]()
    sh.append(1)
    sh.append(n)
    sh.append(HEADS)
    sh.append(HEAD_DIM)
    return reshape(x, sh^, ctx)


# ── Reshape [1, N, HEADS, HEAD_DIM] -> [1, N, HIDDEN] ────────────────────────
def _from_bshd(x: Tensor, n: Int, ctx: DeviceContext) raises -> Tensor:
    var sh = List[Int]()
    sh.append(1)
    sh.append(n)
    sh.append(HIDDEN)
    return reshape(x, sh^, ctx)


# ── Sinusoidal embedding (flip_sin_to_cos=True, FLUX/Chroma style) ────────────
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


# ── mod_proj values (positional embedding over mod_index rows) ─────────────────
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


# ── Build approximator input [1, MOD_IDX, APPROX_IN] ─────────────────────────
# Matches chroma_dit.rs approximator_input():
#   num_channels = approximator_in_channels // 4 = 16
#   timesteps_proj = timestep_embedding(t*1000, num_channels)  -> [16]
#   guidance_proj  = timestep_embedding(0.0, num_channels)      -> [16]
#   tg = cat(timesteps_proj, guidance_proj)                     -> [32]
#   mod_proj = positional_embedding(row*1000, 2*num_channels)   -> [32] per row
#   input_vec = cat(tg, mod_proj)                               -> [64] per row
#   shape [1, MOD_IDX, 64]
def _build_approximator_input(timestep: Float32, ctx: DeviceContext) raises -> Tensor:
    comptime NUM_CH = CHROMA_DIT_APPROX_IN // 4  # 16
    var ts_proj = _sinusoid_values(timestep * Float32(1000.0), NUM_CH)
    var guid_proj = _sinusoid_values(Float32(0.0), NUM_CH)
    var mod_proj = _mod_proj_values(MOD_IDX, 2 * NUM_CH)  # [MOD_IDX * 32]

    var vals = List[Float32]()
    for row in range(MOD_IDX):
        # tg = [ts_proj | guid_proj] = 32 values
        for i in range(NUM_CH):
            vals.append(ts_proj[i])
        for i in range(NUM_CH):
            vals.append(guid_proj[i])
        # mod_proj row = 32 values
        for i in range(2 * NUM_CH):
            vals.append(mod_proj[row * (2 * NUM_CH) + i])

    var sh = List[Int]()
    sh.append(1)
    sh.append(MOD_IDX)
    sh.append(CHROMA_DIT_APPROX_IN)
    return Tensor.from_host(vals, sh^, STDtype.BF16, ctx)


# ── Run distilled_guidance_layer (approximator) ───────────────────────────────
# Returns [1, MOD_IDX, HIDDEN] pooled_temb.
def _approximator_forward(ref shared: ChromaShared, x: Tensor, ctx: DeviceContext) raises -> Tensor:
    # in_proj
    var h = _linear_b(
        x,
        shared._w("distilled_guidance_layer.in_proj.weight"),
        shared._w("distilled_guidance_layer.in_proj.bias"),
        ctx,
    )
    # 5 residual blocks
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
    # out_proj
    return _linear_b(
        h,
        shared._w("distilled_guidance_layer.out_proj.weight"),
        shared._w("distilled_guidance_layer.out_proj.bias"),
        ctx,
    )


# ── Double block forward ───────────────────────────────────────────────────────
# img [1, N_IMG, HIDDEN], txt [1, N_TXT, HIDDEN]
# img_mod_slice [1, 6, HIDDEN], txt_mod_slice [1, 6, HIDDEN]  (slices of pooled_temb)
# Returns (new_img, new_txt)
def _double_block_shape[N_IMG_: Int](
    block_idx: Int,
    img: Tensor,
    txt: Tensor,
    img_mod_slice: Tensor,
    txt_mod_slice: Tensor,
    rope_cos: Tensor,
    rope_sin: Tensor,
    real_txt_len: Int,
    ref loader_block: Block,
    ctx: DeviceContext,
    lora: Optional[ArcPointer[ChromaLoraOverlay]] = None,
) raises -> Tuple[Tensor, Tensor]:
    comptime S_ = N_IMG_ + N_TXT
    var p = String("transformer_blocks.") + String(block_idx)

    # Slice modulation params [1,6,HIDDEN] -> 6 x [1,1,HIDDEN] -> [1,HIDDEN]
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

    # 1. Modulate
    var img_norm = _modulate_pre(img, img_shift1, img_scale1, ctx)
    var txt_norm = _modulate_pre(txt, txt_shift1, txt_scale1, ctx)

    # 2. QKV projections
    var img_q = _to_bshd(
        _linear_chroma_lora(
            img_norm,
            _bw(loader_block, p + ".attn.to_q.weight"),
            _bw(loader_block, p + ".attn.to_q.bias"),
            p + ".attn.to_q.weight",
            lora,
            ctx,
        ),
        N_IMG_, ctx
    )
    var img_k = _to_bshd(
        _linear_chroma_lora(
            img_norm,
            _bw(loader_block, p + ".attn.to_k.weight"),
            _bw(loader_block, p + ".attn.to_k.bias"),
            p + ".attn.to_k.weight",
            lora,
            ctx,
        ),
        N_IMG_, ctx
    )
    var img_v = _to_bshd(
        _linear_chroma_lora(
            img_norm,
            _bw(loader_block, p + ".attn.to_v.weight"),
            _bw(loader_block, p + ".attn.to_v.bias"),
            p + ".attn.to_v.weight",
            lora,
            ctx,
        ),
        N_IMG_, ctx
    )
    var txt_q = _to_bshd(
        _linear_chroma_lora(
            txt_norm,
            _bw(loader_block, p + ".attn.add_q_proj.weight"),
            _bw(loader_block, p + ".attn.add_q_proj.bias"),
            p + ".attn.add_q_proj.weight",
            lora,
            ctx,
        ),
        N_TXT, ctx
    )
    var txt_k = _to_bshd(
        _linear_chroma_lora(
            txt_norm,
            _bw(loader_block, p + ".attn.add_k_proj.weight"),
            _bw(loader_block, p + ".attn.add_k_proj.bias"),
            p + ".attn.add_k_proj.weight",
            lora,
            ctx,
        ),
        N_TXT, ctx
    )
    var txt_v = _to_bshd(
        _linear_chroma_lora(
            txt_norm,
            _bw(loader_block, p + ".attn.add_v_proj.weight"),
            _bw(loader_block, p + ".attn.add_v_proj.bias"),
            p + ".attn.add_v_proj.weight",
            lora,
            ctx,
        ),
        N_TXT, ctx
    )

    # 3. QK RMSNorm
    img_q = rms_norm(img_q, _bw(loader_block, p + ".attn.norm_q.weight"), Float32(1.0e-6), ctx)
    img_k = rms_norm(img_k, _bw(loader_block, p + ".attn.norm_k.weight"), Float32(1.0e-6), ctx)
    txt_q = rms_norm(txt_q, _bw(loader_block, p + ".attn.norm_added_q.weight"), Float32(1.0e-6), ctx)
    txt_k = rms_norm(txt_k, _bw(loader_block, p + ".attn.norm_added_k.weight"), Float32(1.0e-6), ctx)

    # 4. Concat txt+img on seq dim (axis 1), apply RoPE, SDPA
    # Note: Rust concat order is [txt, img] -> matching chroma_dit.mojo staging
    # q/k/v are [1, N, HEADS, HEAD_DIM] so concat on axis 1 -> [1, S, HEADS, HEAD_DIM]
    var q = concat(1, ctx, txt_q, img_q)  # [1, S, HEADS, HEAD_DIM]
    var k = concat(1, ctx, txt_k, img_k)
    var v = concat(1, ctx, txt_v, img_v)
    q = rope_interleaved(q, rope_cos, rope_sin, ctx)
    k = rope_interleaved(k, rope_cos, rope_sin, ctx)
    # MJ-1048: mask the T5 pad-row KEY columns [real_txt_len, N_TXT) in the joint
    # [txt512, img] attention (inference-flame chroma masks them; the old
    # sdpa_nomask path attended to all 512 T5 rows incl. ~490 pad). N_TXT=512 is
    # the comptime text-token count; scale is the same 1/sqrt(HEAD_DIM) the
    # sdpa_nomask path used, so real_txt_len==512 reproduces the old output.
    var att = sdpa_qwen_flash_padmask[1, S_, HEADS, HEAD_DIM, N_TXT](
        q, k, v, real_txt_len, Float32(1.0) / sqrt(Float32(HEAD_DIM)), ctx
    )

    # 5. Split back [1, S, HEADS, HEAD_DIM] -> txt [..N_TXT..], img [..N_IMG..]
    var txt_att_bshd = slice(att, 1, 0, N_TXT, ctx)
    var img_att_bshd = slice(att, 1, N_TXT, N_IMG_, ctx)
    var img_att = _from_bshd(img_att_bshd, N_IMG_, ctx)
    var txt_att = _from_bshd(txt_att_bshd, N_TXT, ctx)

    # 6. Output projections
    var img_o = _linear_chroma_lora(
        img_att,
        _bw(loader_block, p + ".attn.to_out.0.weight"),
        _bw(loader_block, p + ".attn.to_out.0.bias"),
        p + ".attn.to_out.0.weight",
        lora,
        ctx,
    )
    var txt_o = _linear_chroma_lora(
        txt_att,
        _bw(loader_block, p + ".attn.to_add_out.weight"),
        _bw(loader_block, p + ".attn.to_add_out.bias"),
        p + ".attn.to_add_out.weight",
        lora,
        ctx,
    )

    # 7. Gated residuals (attention)
    var img_r = residual_gate(img, img_gate1, img_o, ctx)
    var txt_r = residual_gate(txt, txt_gate1, txt_o, ctx)

    # 8. FFN: modulate -> linear -> gelu -> linear -> gated residual
    var img_ff_in = _modulate_pre(img_r, img_shift2, img_scale2, ctx)
    var img_ff = _linear_chroma_lora(
        img_ff_in,
        _bw(loader_block, p + ".ff.net.0.proj.weight"),
        _bw(loader_block, p + ".ff.net.0.proj.bias"),
        p + ".ff.net.0.proj.weight",
        lora,
        ctx,
    )
    img_ff = gelu(img_ff, ctx)
    img_ff = _linear_chroma_lora(
        img_ff,
        _bw(loader_block, p + ".ff.net.2.weight"),
        _bw(loader_block, p + ".ff.net.2.bias"),
        p + ".ff.net.2.weight",
        lora,
        ctx,
    )
    var img_final = residual_gate(img_r, img_gate2, img_ff, ctx)

    var txt_ff_in = _modulate_pre(txt_r, txt_shift2, txt_scale2, ctx)
    var txt_ff = _linear_chroma_lora(
        txt_ff_in,
        _bw(loader_block, p + ".ff_context.net.0.proj.weight"),
        _bw(loader_block, p + ".ff_context.net.0.proj.bias"),
        p + ".ff_context.net.0.proj.weight",
        lora,
        ctx,
    )
    txt_ff = gelu(txt_ff, ctx)
    txt_ff = _linear_chroma_lora(
        txt_ff,
        _bw(loader_block, p + ".ff_context.net.2.weight"),
        _bw(loader_block, p + ".ff_context.net.2.bias"),
        p + ".ff_context.net.2.weight",
        lora,
        ctx,
    )
    var txt_final = residual_gate(txt_r, txt_gate2, txt_ff, ctx)

    return (img_final^, txt_final^)


def _double_block(
    block_idx: Int,
    img: Tensor,
    txt: Tensor,
    img_mod_slice: Tensor,
    txt_mod_slice: Tensor,
    rope_cos: Tensor,
    rope_sin: Tensor,
    real_txt_len: Int,
    ref loader_block: Block,
    ctx: DeviceContext,
) raises -> Tuple[Tensor, Tensor]:
    """Compatibility wrapper for the proven fixed 1024-square pipeline."""
    return _double_block_shape[N_IMG](
        block_idx, img, txt, img_mod_slice, txt_mod_slice, rope_cos, rope_sin,
        real_txt_len, loader_block, ctx,
    )


# ── Single block forward ───────────────────────────────────────────────────────
# x [1, S, HIDDEN] (cat of [txt, img])
# single_mod_slice [1, 3, HIDDEN]
# Returns updated x [1, S, HIDDEN]
def _single_block_shape[N_IMG_: Int](
    block_idx: Int,
    x: Tensor,
    single_mod_slice: Tensor,
    rope_cos: Tensor,
    rope_sin: Tensor,
    real_txt_len: Int,
    ref loader_block: Block,
    ctx: DeviceContext,
    lora: Optional[ArcPointer[ChromaLoraOverlay]] = None,
) raises -> Tensor:
    comptime S_ = N_IMG_ + N_TXT
    var p = String("single_transformer_blocks.") + String(block_idx)

    # Modulation params
    var shift = _pooled_row(single_mod_slice, 0, ctx)
    var scale = _pooled_row(single_mod_slice, 1, ctx)
    var gate  = _pooled_row(single_mod_slice, 2, ctx)

    # 1. Modulate
    var x_norm = _modulate_pre(x, shift, scale, ctx)

    # 2. QKV projections -> [1, HEADS, S, HEAD_DIM]
    var q = _to_bshd(
        _linear_chroma_lora(
            x_norm,
            _bw(loader_block, p + ".attn.to_q.weight"),
            _bw(loader_block, p + ".attn.to_q.bias"),
            p + ".attn.to_q.weight",
            lora,
            ctx,
        ),
        S_, ctx
    )
    var k = _to_bshd(
        _linear_chroma_lora(
            x_norm,
            _bw(loader_block, p + ".attn.to_k.weight"),
            _bw(loader_block, p + ".attn.to_k.bias"),
            p + ".attn.to_k.weight",
            lora,
            ctx,
        ),
        S_, ctx
    )
    var v = _to_bshd(
        _linear_chroma_lora(
            x_norm,
            _bw(loader_block, p + ".attn.to_v.weight"),
            _bw(loader_block, p + ".attn.to_v.bias"),
            p + ".attn.to_v.weight",
            lora,
            ctx,
        ),
        S_, ctx
    )

    # 3. QK RMSNorm
    q = rms_norm(q, _bw(loader_block, p + ".attn.norm_q.weight"), Float32(1.0e-6), ctx)
    k = rms_norm(k, _bw(loader_block, p + ".attn.norm_k.weight"), Float32(1.0e-6), ctx)

    # 4. RoPE
    q = rope_interleaved(q, rope_cos, rope_sin, ctx)
    k = rope_interleaved(k, rope_cos, rope_sin, ctx)

    # 5. SDPA — same [txt512, img] joint layout as the double block, so mask the
    # T5 pad-row key columns [real_txt_len, N_TXT) here too (MJ-1048).
    var att = sdpa_qwen_flash_padmask[1, S_, HEADS, HEAD_DIM, N_TXT](
        q, k, v, real_txt_len, Float32(1.0) / sqrt(Float32(HEAD_DIM)), ctx
    )
    var att_flat = _from_bshd(att, S_, ctx)

    # 6. MLP path
    var mlp = _linear_chroma_lora(
        x_norm,
        _bw(loader_block, p + ".proj_mlp.weight"),
        _bw(loader_block, p + ".proj_mlp.bias"),
        p + ".proj_mlp.weight",
        lora,
        ctx,
    )
    mlp = gelu(mlp, ctx)

    # 7. Concat [att | mlp] -> proj_out -> gated residual
    var cat_out = concat(2, ctx, att_flat, mlp)
    var out = _linear_chroma_lora(
        cat_out,
        _bw(loader_block, p + ".proj_out.weight"),
        _bw(loader_block, p + ".proj_out.bias"),
        p + ".proj_out.weight",
        lora,
        ctx,
    )
    return residual_gate(x, gate, out, ctx)


def _single_block(
    block_idx: Int,
    x: Tensor,
    single_mod_slice: Tensor,
    rope_cos: Tensor,
    rope_sin: Tensor,
    real_txt_len: Int,
    ref loader_block: Block,
    ctx: DeviceContext,
) raises -> Tensor:
    """Compatibility wrapper for the proven fixed 1024-square pipeline."""
    return _single_block_shape[N_IMG](
        block_idx, x, single_mod_slice, rope_cos, rope_sin, real_txt_len,
        loader_block, ctx,
    )


# ── Full Chroma DiT forward ────────────────────────────────────────────────────
# img_packed [1, N_IMG, 64] BF16 packed latent
# txt        [1, N_TXT, 4096] BF16 T5 hidden states
# timestep   Float32 in [0, 1] (sigma)
# Returns    [1, N_IMG, 64] predicted velocity
def _chroma_forward(
    shared: ChromaShared,
    loader: BlockLoader,
    img_packed: Tensor,
    txt: Tensor,
    timestep: Float32,
    rope_cos: Tensor,
    rope_sin: Tensor,
    real_txt_len: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    # 1. Approximator: build per-step modulation table [1, 344, 3072]
    var approx_in = _build_approximator_input(timestep, ctx)
    var pooled_temb = _approximator_forward(shared, approx_in, ctx)

    # 2. Input projections
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

    # 3. Double blocks (19) — stream one at a time
    for i in range(N_DBL):
        var prefix = String("transformer_blocks.") + String(i)
        loader.prefetch_block(prefix)
        var block = loader.load_block(prefix, ctx)

        # Slice mod params: img [114+6i : 114+6i+6], txt [228+6i : 228+6i+6]
        var img_mod_s = slice(pooled_temb, 1, MOD_DBL_IMG_OFF + 6 * i, 6, ctx)  # [1, 6, HIDDEN]
        var txt_mod_s = slice(pooled_temb, 1, MOD_DBL_TXT_OFF + 6 * i, 6, ctx)

        var res = _double_block(i, img, x_txt, img_mod_s, txt_mod_s, rope_cos, rope_sin, real_txt_len, block, ctx)
        # Borrow tuple elements into _clone (def params are borrowed in Mojo).
        img = _clone(res[0], ctx)
        x_txt = _clone(res[1], ctx)

        unload_block(block^)

    # 4. Concat [txt | img] -> [1, S, HIDDEN] for single blocks
    var x = concat(1, ctx, x_txt, img)

    # 5. Single blocks (38) — stream one at a time
    for i in range(N_SGL):
        var prefix = String("single_transformer_blocks.") + String(i)
        loader.prefetch_block(prefix)
        var block = loader.load_block(prefix, ctx)

        # Slice mod params: [3*i : 3*i+3]
        var sgl_mod_s = slice(pooled_temb, 1, MOD_SGL_OFF + 3 * i, 3, ctx)  # [1, 3, HIDDEN]

        x = _single_block(i, x, sgl_mod_s, rope_cos, rope_sin, real_txt_len, block, ctx)

        unload_block(block^)

    # 6. Extract img portion [1, N_IMG, HIDDEN], apply norm_out + proj_out
    # x is [txt | img], txt first (N_TXT rows), img after
    var img_out = slice(x, 1, N_TXT, N_IMG, ctx)

    # norm_out: last 2 rows of pooled_temb -> shift, scale
    var norm_shift = _pooled_row(pooled_temb, MOD_IDX - 2, ctx)
    var norm_scale = _pooled_row(pooled_temb, MOD_IDX - 1, ctx)
    var normed = _layer_norm_no_affine(img_out, ctx)
    var modulated = modulate(normed, norm_scale, norm_shift, ctx)

    # proj_out -> [1, N_IMG, 64]
    return _linear_b(
        modulated,
        shared._w("proj_out.weight"),
        shared._w("proj_out.bias"),
        ctx,
    )


# ── Pack [1,16,LH,LW] -> [1, N_IMG, 64] ─────────────────────────────────────
def _pack_latent_shape[LH_: Int, LW_: Int](
    z: Tensor, ctx: DeviceContext
) raises -> Tensor:
    comptime PGH_ = LH_ // PACK
    comptime PGW_ = LW_ // PACK
    comptime N_IMG_ = PGH_ * PGW_
    # reshape [1,16,h2,2,w2,2] -> permute [0,2,4,1,3,5] -> reshape [1,h2*w2,64]
    var s6 = List[Int]()
    s6.append(1)
    s6.append(LC)
    s6.append(PGH_)
    s6.append(PACK)
    s6.append(PGW_)
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
    sp.append(N_IMG_)
    sp.append(LC * PACK * PACK)
    return reshape(tp, sp^, ctx)


def _pack_latent(z: Tensor, ctx: DeviceContext) raises -> Tensor:
    """Compatibility wrapper for the proven fixed 1024-square pipeline."""
    return _pack_latent_shape[LH, LW](z, ctx)


# ── Unpack [1, N_IMG, 64] -> [1,16,LH,LW] ────────────────────────────────────
def _unpack_latent_shape[LH_: Int, LW_: Int](
    packed: Tensor, ctx: DeviceContext
) raises -> Tensor:
    comptime PGH_ = LH_ // PACK
    comptime PGW_ = LW_ // PACK
    # reshape [1,h2,w2,16,2,2] -> permute [0,3,1,4,2,5] -> reshape [1,16,2*h2,2*w2]
    var s6 = List[Int]()
    s6.append(1)
    s6.append(PGH_)
    s6.append(PGW_)
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
    sp.append(LH_)
    sp.append(LW_)
    return reshape(tp, sp^, ctx)


def _unpack_latent(packed: Tensor, ctx: DeviceContext) raises -> Tensor:
    """Compatibility wrapper for the proven fixed 1024-square pipeline."""
    return _unpack_latent_shape[LH, LW](packed, ctx)


# ── Main ──────────────────────────────────────────────────────────────────────
def _legacy_disk_streaming_main_disabled() raises:
    var ctx = DeviceContext()
    print("============================================================")
    print("Chroma 1024x1024 — 30-step shifted flow CFG 4.0 seed 42")
    print("  Checkpoint:", CHROMA_CKPT)
    print("  Embeddings:", EMBEDDINGS_PATH)
    print("  VAE:", VAE_PATH)
    print("  Output:", OUT_PNG)
    print("============================================================")

    # Stage 1: Load T5 embeddings sidecar
    print("\n--- Stage 1: Load T5 sidecar ---")
    var emb_st = ShardedSafeTensors.open(String(EMBEDDINGS_PATH))
    var t5_cond = cast_tensor(
        Tensor.from_view(emb_st.tensor_view(String("t5_cond")), ctx),
        STDtype.BF16, ctx
    )
    var t5_uncond = cast_tensor(
        Tensor.from_view(emb_st.tensor_view(String("t5_uncond")), ctx),
        STDtype.BF16, ctx
    )
    var cs = t5_cond.shape()
    print("  t5_cond:", cs[0], cs[1], cs[2], "dtype:", t5_cond.dtype().name())
    _stats("t5_cond", t5_cond, ctx)
    _stats("t5_uncond", t5_uncond, ctx)

    # MJ-1048: real (unpadded) T5 lengths, used to mask the T5 pad-row keys in the
    # DiT joint attention (see _sidecar_real_len / _double_block / _single_block).
    var cond_real_len = _sidecar_real_len(emb_st, String("cond_real_len"), ctx)
    var uncond_real_len = _sidecar_real_len(emb_st, String("uncond_real_len"), ctx)

    # Stage 2: Load shared Chroma weights (approximator + embedders + proj_out)
    print("\n--- Stage 2: Load Chroma shared weights ---")
    var shared = ChromaShared.load(String(CHROMA_CKPT), ctx)
    print("  Shared weights loaded:", len(shared.weights), "tensors")

    # Stage 3: Open block loader for streaming double/single blocks
    print("\n--- Stage 3: Open block loader ---")
    var loader = BlockLoader.open(String(CHROMA_CKPT))
    print("  Block loader ready")

    # Stage 4: Build RoPE tables [S*HEADS, HEAD_DIM/2] — same for all steps
    print("\n--- Stage 4: Build RoPE tables ---")
    var rope = build_flux1_rope_tables[N_IMG, N_TXT, HEADS, HEAD_DIM](
        PGH, PGW, ctx, STDtype.BF16
    )
    # Tuple element indexing cannot move Tensor values; borrow into _clone.
    var rope_cos = _clone(rope[0], ctx)
    var rope_sin = _clone(rope[1], ctx)
    print("  RoPE cos shape:", rope_cos.shape()[0], rope_cos.shape()[1])

    # Stage 5: Init noise latent
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
    _stats("init_latent", img_packed, ctx)

    # Stage 6: Build sigma schedule
    print("\n--- Stage 6: Build schedule (", NUM_STEPS, "steps) ---")
    var sigmas = build_flux1_sigma_schedule(NUM_STEPS, N_IMG)
    print("  sigma[0]=", sigmas[0], " sigma[-1]=", sigmas[NUM_STEPS])

    # Stage 7: CFG Euler denoise loop
    print("\n--- Stage 7: CFG Euler denoise ---")
    for step in range(NUM_STEPS):
        var t_curr = sigmas[step]
        var t_next = sigmas[step + 1]
        var dt = t_next - t_curr  # negative (descending schedule)

        # Conditioned forward
        var pred_cond = _chroma_forward(
            shared, loader, img_packed, t5_cond, t_curr, rope_cos, rope_sin, cond_real_len, ctx
        )
        # Unconditioned forward
        var pred_uncond = _chroma_forward(
            shared, loader, img_packed, t5_uncond, t_curr, rope_cos, rope_sin, uncond_real_len, ctx
        )

        # CFG: pred = uncond + guidance * (cond - uncond)
        # diff = cond - uncond
        var neg_uncond = mul_scalar(pred_uncond, Float32(-1.0), ctx)
        var diff = add(pred_cond, neg_uncond, ctx)
        var scaled_diff = mul_scalar(diff, GUIDANCE, ctx)
        var pred = add(pred_uncond, scaled_diff, ctx)

        # Euler step: x = x + dt * pred
        var step_delta = mul_scalar(pred, dt, ctx)
        img_packed = add(img_packed, step_delta, ctx)

        if step == 0 or (step + 1) % 5 == 0 or step + 1 == NUM_STEPS:
            print("  step", step + 1, "/", NUM_STEPS, " t=", t_curr, " dt=", dt)
            _stats("x", img_packed, ctx)

    print("\n--- Stage 8: Unpack + VAE decode ---")
    var latent = _unpack_latent(img_packed, ctx)
    var lsh = latent.shape()
    print("  Unpacked latent:", lsh[0], lsh[1], lsh[2], lsh[3])
    _stats("latent", latent, ctx)

    # Drop DiT weights before loading VAE by transferring them to sink bindings.
    var _sink_shared = shared^
    var _sink_loader = loader^
    _ = _sink_shared
    _ = _sink_loader
    print("  DiT weights dropped")

    # The current Flux VAE decoder path is F32-weight/file-format only; this is
    # the remaining decode-boundary cast until that VAE path is dtype-generic.
    var latent_f32 = cast_tensor(latent, STDtype.F32, ctx)

    # Load FLUX VAE decoder (ae.safetensors, 16ch, scale=0.3611, shift=0.1159)
    var vae = load_flux1_ldm_decoder[LH, LW](String(VAE_PATH), ctx)
    print("  VAE loaded")

    var rgb = vae.decode(latent_f32, ctx)
    var rsh = rgb.shape()
    print("  Decoded RGB:", rsh[0], rsh[1], rsh[2], rsh[3])
    _stats("rgb", rgb, ctx)

    print("\n--- Stage 9: Save PNG ---")
    save_png(rgb, String(OUT_PNG), ctx, ValueRange.SIGNED)
    print("  Saved:", OUT_PNG)
    print("\n============================================================")
    print("DONE:", OUT_PNG)
    print("============================================================")


def main() raises:
    raise Error(
        "disabled: this legacy Chroma executable reads checkpoint weights "
        "during denoise; use the fail-closed memory-resident product worker"
    )

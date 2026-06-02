# lens_pipeline_1024_multistep.mojo — Microsoft Lens 1024x1024 full pipeline.
#
# End-to-end: cached GPT-OSS text features -> Lens DiT denoise -> FLUX.2 VAE -> PNG.
# Mirrors lens_infer.rs (--use-cached-features path, zeroed text features).
#
# Text conditioning: captures_text_smoke/ hidden_layer_{05,11,17,23}.safetensors
#   Each [1,64,2880] BF16. txt_norm.{0..3} RMSNorm (eps=1e-5) -> concat [1,64,11520]
#   -> txt_in -> [1,64,1536]. Both cond and uncond are zeros; CFG is identity.
#
# Oracle: captures/reference_image.png (zeroed text, 20 steps, cfg=5.0, seed=42).
#
# Architecture: 48 dual-stream MMDiT blocks (lens_dit.rs), streaming via BlockLoader.

from std.gpu.host import DeviceContext
from std.math import sqrt, exp as fexp, cos as fcos, sin as fsin, log as flog, pow as fpow
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.offload.block_loader import BlockLoader, Block, unload_block
from serenitymojo.models.vae.klein_decoder import KleinVaeDecoder
from serenitymojo.ops.activations import silu, swiglu
from serenitymojo.ops.attention import sdpa_nomask
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.elementwise import modulate, residual_gate
from serenitymojo.ops.embeddings import timestep_embedding
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import rms_norm, layer_norm
from serenitymojo.ops.random import randn
from serenitymojo.ops.rope import rope_interleaved
from serenitymojo.ops.tensor_algebra import add, sub, mul_scalar, div, reshape, permute, slice, concat
from serenitymojo.sampling.lens_flowmatch import LensFlowMatchScheduler, lens_euler_step
from serenitymojo.image.png import save_png, ValueRange


# ── Paths ───────────────────────────────────────────────────────────────────
comptime TRANSFORMER_DIR = "/home/alex/.serenity/models/microsoft_lens/transformer"
comptime FLUX2_VAE_PATH  = "/home/alex/.serenity/models/vaes/flux2-vae.safetensors"
comptime TEXT_SMOKE_DIR  = "/home/alex/EriDiffusion/inference-flame/lens/parity/captures_text_smoke"
comptime HIDDEN_05 = TEXT_SMOKE_DIR + "/hidden_layer_05.safetensors"
comptime HIDDEN_11 = TEXT_SMOKE_DIR + "/hidden_layer_11.safetensors"
comptime HIDDEN_17 = TEXT_SMOKE_DIR + "/hidden_layer_17.safetensors"
comptime HIDDEN_23 = TEXT_SMOKE_DIR + "/hidden_layer_23.safetensors"
comptime OUT_PATH = "/home/alex/mojodiffusion/output/lens_1024_20step.png"

# ── Dimensions ──────────────────────────────────────────────────────────────
comptime LH = 64
comptime LW = 64
comptime N_IMG = LH * LW          # 4096
comptime N_TXT = 64
comptime S = N_IMG + N_TXT        # 4160

comptime DIM = 1536
comptime NUM_HEADS = 24
comptime HEAD_DIM = 64
comptime ROPE_HALF = 32           # sum(axes_dim)/2 = (8+28+28)/2
comptime MLP_HIDDEN = 4096
comptime NUM_LAYERS = 48
comptime IN_CH = 128

comptime ENC_HIDDEN = 2880
comptime N_LAYERS_ENC = 4
comptime TXT_IN_DIM = ENC_HIDDEN * N_LAYERS_ENC  # 11520
comptime TEMB_DIM = 256
comptime ROPE_TABLE_ROWS = 4096

comptime AXES_FRAME_HALF = 4
comptime AXES_H_HALF = 14
comptime AXES_W_HALF = 14

# ── Sampler ──────────────────────────────────────────────────────────────────
comptime NUM_STEPS = 20
comptime CFG_SCALE = Float32(5.0)
comptime SEED = UInt64(42)

comptime BLOCK_NORM_EPS = Float32(1.0e-6)
comptime QK_NORM_EPS    = Float32(1.0e-5)
comptime TXT_NORM_EPS   = Float32(1.0e-5)
comptime FINAL_LN_EPS   = Float32(1.0e-6)


# ── Stats helper ─────────────────────────────────────────────────────────────
def _stats(name: String, t: Tensor, ctx: DeviceContext) raises:
    var h = t.to_host(ctx)
    var n = len(h)
    var s = 0.0
    var s2 = 0.0
    var amax = 0.0
    for i in range(n):
        var v = Float64(h[i])
        s += v
        s2 += v * v
        var av = v if v >= 0.0 else -v
        if av > amax:
            amax = av
    var mean = s / Float64(n)
    var var_ = s2 / Float64(n) - mean * mean
    if var_ < 0.0:
        var_ = 0.0
    print(
        "  [stat]", name,
        "mean=", Float32(mean), "std=", Float32(sqrt(var_)),
        "absmax=", Float32(amax), "n=", n,
    )


def _clone(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var dev = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    ctx.enqueue_copy(dst_buf=dev, src_buf=x.buf)
    ctx.synchronize()
    return Tensor(dev^, x.shape(), x.dtype())


def _ones_bf16(n: Int, ctx: DeviceContext) raises -> Tensor:
    var vals = List[Float32]()
    for _ in range(n):
        vals.append(1.0)
    var sh = List[Int]()
    sh.append(n)
    return cast_tensor(Tensor.from_host(vals, sh^, STDtype.F32, ctx), STDtype.BF16, ctx)


def _zeros_bf16(n: Int, ctx: DeviceContext) raises -> Tensor:
    var vals = List[Float32]()
    for _ in range(n):
        vals.append(0.0)
    var sh = List[Int]()
    sh.append(n)
    return cast_tensor(Tensor.from_host(vals, sh^, STDtype.F32, ctx), STDtype.BF16, ctx)


# Extract one [DIM] chunk from adaln output [1, 6*DIM]
def _adaln_chunk(mod_out: Tensor, idx: Int, ctx: DeviceContext) raises -> Tensor:
    var part = slice(mod_out, 1, idx * DIM, DIM, ctx)
    var sh = List[Int]()
    sh.append(DIM)
    return reshape(part, sh^, ctx)


# [1, S_, DIM] -> [1, S_, NUM_HEADS, HEAD_DIM]
def _to_bshd[S_: Int](x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var sh = List[Int]()
    sh.append(1)
    sh.append(S_)
    sh.append(NUM_HEADS)
    sh.append(HEAD_DIM)
    return reshape(x, sh^, ctx)


# [1, S_, NUM_HEADS, HEAD_DIM] -> [1, S_, DIM]
def _from_bshd[S_: Int](x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var sh = List[Int]()
    sh.append(1)
    sh.append(S_)
    sh.append(DIM)
    return reshape(x, sh^, ctx)


# ── RoPE table storage ────────────────────────────────────────────────────────
@fieldwise_init
struct LensRopeTables(Movable):
    """3-axis Lens RoPE tables, tiled across heads for direct rope_interleaved use.

    img_cos/img_sin: [N_IMG * NUM_HEADS, ROPE_HALF] BF16
    txt_cos/txt_sin: [N_TXT * NUM_HEADS, ROPE_HALF] BF16
    """
    var img_cos: Tensor
    var img_sin: Tensor
    var txt_cos: Tensor
    var txt_sin: Tensor


# ── Block output ──────────────────────────────────────────────────────────────
@fieldwise_init
struct LensBlockOut(Movable):
    var img: Tensor
    var txt: Tensor


# ── Build 3-axis Lens RoPE tables ────────────────────────────────────────────
# Mirrors LensEmbedRope in lens_dit.rs: axes=[8,28,28], theta=10000, scale_rope=True.
# Builds img [N_IMG, 32] and txt [N_TXT, 32], then tiles over heads -> [N*H, 32].
def build_lens_rope_tables(ctx: DeviceContext) raises -> LensRopeTables:
    print("[rope] building 3-axis Lens RoPE tables")
    # -- Build pos and neg host tables [4096, ROPE_HALF] --
    var pos_cos_host = List[Float32]()
    var pos_sin_host = List[Float32]()
    var neg_cos_host = List[Float32]()
    var neg_sin_host = List[Float32]()
    for _ in range(ROPE_TABLE_ROWS * ROPE_HALF):
        pos_cos_host.append(0.0)
        pos_sin_host.append(0.0)
        neg_cos_host.append(0.0)
        neg_sin_host.append(0.0)

    var axes = List[Int]()
    axes.append(8)   # frame
    axes.append(28)  # H
    axes.append(28)  # W
    var halfs = List[Int]()
    halfs.append(AXES_FRAME_HALF)
    halfs.append(AXES_H_HALF)
    halfs.append(AXES_W_HALF)

    var col_offset = 0
    for axis in range(3):
        var d = axes[axis]
        var half = halfs[axis]
        var base = List[Float64]()
        for k in range(half):
            var exp_ = Float64(2 * k) / Float64(d)
            base.append(1.0 / fpow(10000.0, exp_))
        for row in range(ROPE_TABLE_ROWS):
            var pos_n = Float64(row)
            var neg_n = -(Float64(ROPE_TABLE_ROWS) - Float64(row))
            for k in range(half):
                var dst = row * ROPE_HALF + col_offset + k
                var arg_pos = pos_n * base[k]
                var arg_neg = neg_n * base[k]
                pos_cos_host[dst] = Float32(fcos(arg_pos))
                pos_sin_host[dst] = Float32(fsin(arg_pos))
                neg_cos_host[dst] = Float32(fcos(arg_neg))
                neg_sin_host[dst] = Float32(fsin(arg_neg))
        col_offset += half

    # -- Build img [LH*LW, ROPE_HALF] --
    var h_lo = LH // 2   # 32
    var h_hi = LH - h_lo # 32
    var w_lo = LW // 2
    var w_hi = LW - w_lo

    # height rows: cat(neg[-(h-h/2):], pos[:h/2])
    var height_cos = List[Float32]()
    var height_sin = List[Float32]()
    for _ in range(LH * AXES_H_HALF):
        height_cos.append(0.0)
        height_sin.append(0.0)
    for i in range(h_hi):
        var src_row = ROPE_TABLE_ROWS - h_hi + i
        for k in range(AXES_H_HALF):
            var src = src_row * ROPE_HALF + AXES_FRAME_HALF + k
            height_cos[i * AXES_H_HALF + k] = neg_cos_host[src]
            height_sin[i * AXES_H_HALF + k] = neg_sin_host[src]
    for i in range(h_lo):
        var src_row = i
        for k in range(AXES_H_HALF):
            var src = src_row * ROPE_HALF + AXES_FRAME_HALF + k
            height_cos[(h_hi + i) * AXES_H_HALF + k] = pos_cos_host[src]
            height_sin[(h_hi + i) * AXES_H_HALF + k] = pos_sin_host[src]

    # width rows
    var width_cos = List[Float32]()
    var width_sin = List[Float32]()
    for _ in range(LW * AXES_W_HALF):
        width_cos.append(0.0)
        width_sin.append(0.0)
    for i in range(w_hi):
        var src_row = ROPE_TABLE_ROWS - w_hi + i
        for k in range(AXES_W_HALF):
            var src = src_row * ROPE_HALF + AXES_FRAME_HALF + AXES_H_HALF + k
            width_cos[i * AXES_W_HALF + k] = neg_cos_host[src]
            width_sin[i * AXES_W_HALF + k] = neg_sin_host[src]
    for i in range(w_lo):
        var src_row = i
        for k in range(AXES_W_HALF):
            var src = src_row * ROPE_HALF + AXES_FRAME_HALF + AXES_H_HALF + k
            width_cos[(w_hi + i) * AXES_W_HALF + k] = pos_cos_host[src]
            width_sin[(w_hi + i) * AXES_W_HALF + k] = pos_sin_host[src]

    # img_cos/sin [N_IMG, ROPE_HALF] then tile to [N_IMG*H, ROPE_HALF]
    var img_cos_host = List[Float32]()
    var img_sin_host = List[Float32]()
    for _ in range(N_IMG * ROPE_HALF):
        img_cos_host.append(0.0)
        img_sin_host.append(0.0)
    for yy in range(LH):
        for xx in range(LW):
            var dst_row = (yy * LW + xx) * ROPE_HALF
            for k in range(AXES_FRAME_HALF):
                var src = 0 * ROPE_HALF + k
                img_cos_host[dst_row + k] = pos_cos_host[src]
                img_sin_host[dst_row + k] = pos_sin_host[src]
            for k in range(AXES_H_HALF):
                img_cos_host[dst_row + AXES_FRAME_HALF + k] = height_cos[yy * AXES_H_HALF + k]
                img_sin_host[dst_row + AXES_FRAME_HALF + k] = height_sin[yy * AXES_H_HALF + k]
            for k in range(AXES_W_HALF):
                img_cos_host[dst_row + AXES_FRAME_HALF + AXES_H_HALF + k] = width_cos[xx * AXES_W_HALF + k]
                img_sin_host[dst_row + AXES_FRAME_HALF + AXES_H_HALF + k] = width_sin[xx * AXES_W_HALF + k]

    # txt [N_TXT, ROPE_HALF]: pos[max_vid_index : max_vid_index + N_TXT]
    # max_vid_index = max(LH/2, LW/2) = 32
    comptime MAX_VID_IDX = LH // 2  # 32
    var txt_cos_host = List[Float32]()
    var txt_sin_host = List[Float32]()
    for _ in range(N_TXT * ROPE_HALF):
        txt_cos_host.append(0.0)
        txt_sin_host.append(0.0)
    for i in range(N_TXT):
        var src_row = MAX_VID_IDX + i
        for k in range(ROPE_HALF):
            var src = src_row * ROPE_HALF + k
            txt_cos_host[i * ROPE_HALF + k] = pos_cos_host[src]
            txt_sin_host[i * ROPE_HALF + k] = pos_sin_host[src]

    # Tile over heads: [S, 32] -> [S*H, 32] where cos[s*H+h] = cos[s]
    var img_cos_tiled = List[Float32]()
    var img_sin_tiled = List[Float32]()
    for i in range(N_IMG):
        for _ in range(NUM_HEADS):
            for k in range(ROPE_HALF):
                img_cos_tiled.append(img_cos_host[i * ROPE_HALF + k])
                img_sin_tiled.append(img_sin_host[i * ROPE_HALF + k])

    var txt_cos_tiled = List[Float32]()
    var txt_sin_tiled = List[Float32]()
    for i in range(N_TXT):
        for _ in range(NUM_HEADS):
            for k in range(ROPE_HALF):
                txt_cos_tiled.append(txt_cos_host[i * ROPE_HALF + k])
                txt_sin_tiled.append(txt_sin_host[i * ROPE_HALF + k])

    # Upload to GPU as BF16
    var ic_sh = List[Int]()
    ic_sh.append(N_IMG * NUM_HEADS)
    ic_sh.append(ROPE_HALF)
    var tc_sh = List[Int]()
    tc_sh.append(N_TXT * NUM_HEADS)
    tc_sh.append(ROPE_HALF)

    var ic = cast_tensor(Tensor.from_host(img_cos_tiled, ic_sh.copy(), STDtype.F32, ctx), STDtype.BF16, ctx)
    var is_ = cast_tensor(Tensor.from_host(img_sin_tiled, ic_sh.copy(), STDtype.F32, ctx), STDtype.BF16, ctx)
    var tc = cast_tensor(Tensor.from_host(txt_cos_tiled, tc_sh.copy(), STDtype.F32, ctx), STDtype.BF16, ctx)
    var ts = cast_tensor(Tensor.from_host(txt_sin_tiled, tc_sh.copy(), STDtype.F32, ctx), STDtype.BF16, ctx)

    print("[rope] img tables:", N_IMG * NUM_HEADS, "x", ROPE_HALF)
    print("[rope] txt tables:", N_TXT * NUM_HEADS, "x", ROPE_HALF)
    return LensRopeTables(ic^, is_^, tc^, ts^)


# Apply interleaved RoPE to [1, S_, H, HEAD_DIM]
# cos/sin: [S_*H, ROPE_HALF] (pre-tiled over heads)
def _apply_rope[S_: Int](
    x: Tensor, cos_tiled: Tensor, sin_tiled: Tensor, ctx: DeviceContext
) raises -> Tensor:
    # Flatten to [S_*H, HEAD_DIM]
    var flat_sh = List[Int]()
    flat_sh.append(S_ * NUM_HEADS)
    flat_sh.append(HEAD_DIM)
    var x_flat = reshape(x, flat_sh^, ctx)
    var roped = rope_interleaved(x_flat, cos_tiled, sin_tiled, ctx)
    # Reshape back to [1, S_, H, HEAD_DIM]
    var bshd_sh = List[Int]()
    bshd_sh.append(1)
    bshd_sh.append(S_)
    bshd_sh.append(NUM_HEADS)
    bshd_sh.append(HEAD_DIM)
    return reshape(roped, bshd_sh^, ctx)


# ── Resident weights (everything except the 48 transformer blocks) ────────────
@fieldwise_init
struct LensResident(Movable):
    var img_in_w: Tensor
    var img_in_b: Tensor
    var txt_in_w: Tensor
    var txt_in_b: Tensor
    var txt_norm0_w: Tensor
    var txt_norm1_w: Tensor
    var txt_norm2_w: Tensor
    var txt_norm3_w: Tensor
    var temb_lin1_w: Tensor
    var temb_lin1_b: Tensor
    var temb_lin2_w: Tensor
    var temb_lin2_b: Tensor
    var norm_out_w: Tensor
    var norm_out_b: Tensor
    var proj_out_w: Tensor
    var proj_out_b: Tensor

    @staticmethod
    def load(ctx: DeviceContext) raises -> LensResident:
        var st = ShardedSafeTensors.open(String(TRANSFORMER_DIR))

        var img_in_w   = Tensor.from_view(st.tensor_view(String("img_in.weight")), ctx)
        var img_in_b   = Tensor.from_view(st.tensor_view(String("img_in.bias")), ctx)
        var txt_in_w   = Tensor.from_view(st.tensor_view(String("txt_in.weight")), ctx)
        var txt_in_b   = Tensor.from_view(st.tensor_view(String("txt_in.bias")), ctx)
        var tn0        = Tensor.from_view(st.tensor_view(String("txt_norm.0.weight")), ctx)
        var tn1        = Tensor.from_view(st.tensor_view(String("txt_norm.1.weight")), ctx)
        var tn2        = Tensor.from_view(st.tensor_view(String("txt_norm.2.weight")), ctx)
        var tn3        = Tensor.from_view(st.tensor_view(String("txt_norm.3.weight")), ctx)
        var tl1w       = Tensor.from_view(st.tensor_view(String("time_text_embed.timestep_embedder.linear_1.weight")), ctx)
        var tl1b       = Tensor.from_view(st.tensor_view(String("time_text_embed.timestep_embedder.linear_1.bias")), ctx)
        var tl2w       = Tensor.from_view(st.tensor_view(String("time_text_embed.timestep_embedder.linear_2.weight")), ctx)
        var tl2b       = Tensor.from_view(st.tensor_view(String("time_text_embed.timestep_embedder.linear_2.bias")), ctx)
        var now_w      = Tensor.from_view(st.tensor_view(String("norm_out.linear.weight")), ctx)
        var now_b      = Tensor.from_view(st.tensor_view(String("norm_out.linear.bias")), ctx)
        var proj_out_w = Tensor.from_view(st.tensor_view(String("proj_out.weight")), ctx)
        var proj_out_b = Tensor.from_view(st.tensor_view(String("proj_out.bias")), ctx)

        return LensResident(
            img_in_w^, img_in_b^,
            txt_in_w^, txt_in_b^,
            tn0^, tn1^, tn2^, tn3^,
            tl1w^, tl1b^, tl2w^, tl2b^,
            now_w^, now_b^,
            proj_out_w^, proj_out_b^,
        )


# ── Text conditioning: cached hidden states -> projected [1,N_TXT,DIM] ───────
def build_text_cond(resident: LensResident, ctx: DeviceContext) raises -> Tensor:
    print("[text] loading 4 cached GPT-OSS hidden layers")
    var st05 = ShardedSafeTensors.open(String(HIDDEN_05))
    var st11 = ShardedSafeTensors.open(String(HIDDEN_11))
    var st17 = ShardedSafeTensors.open(String(HIDDEN_17))
    var st23 = ShardedSafeTensors.open(String(HIDDEN_23))
    var h05 = Tensor.from_view(st05.tensor_view(String("tensor")), ctx)
    var h11 = Tensor.from_view(st11.tensor_view(String("tensor")), ctx)
    var h17 = Tensor.from_view(st17.tensor_view(String("tensor")), ctx)
    var h23 = Tensor.from_view(st23.tensor_view(String("tensor")), ctx)
    # Each [1, 64, 2880] BF16 (or F32 - auto-loaded). Cast weights to match.
    var tn0 = cast_tensor(resident.txt_norm0_w, h05.dtype(), ctx)
    var tn1 = cast_tensor(resident.txt_norm1_w, h11.dtype(), ctx)
    var tn2 = cast_tensor(resident.txt_norm2_w, h17.dtype(), ctx)
    var tn3 = cast_tensor(resident.txt_norm3_w, h23.dtype(), ctx)
    var n05 = rms_norm(h05, tn0, TXT_NORM_EPS, ctx)
    var n11 = rms_norm(h11, tn1, TXT_NORM_EPS, ctx)
    var n17 = rms_norm(h17, tn2, TXT_NORM_EPS, ctx)
    var n23 = rms_norm(h23, tn3, TXT_NORM_EPS, ctx)
    # Concat to [1, 64, 11520]
    var cat4 = concat(2, ctx, n05, n11, n17, n23)
    # Cast txt_in weights to match cat4 dtype, then project
    var cat4_dt = cat4.dtype()
    var tin_w = cast_tensor(resident.txt_in_w, cat4_dt, ctx)
    var tin_b = cast_tensor(resident.txt_in_b, cat4_dt, ctx)
    var e = linear(cat4, tin_w, Optional[Tensor](tin_b^), ctx)  # [1,64,1536]
    _stats("txt_cond", e, ctx)
    return e^


# ── Initial noise ─────────────────────────────────────────────────────────────
def initial_noise(ctx: DeviceContext) raises -> Tensor:
    var sh = List[Int]()
    sh.append(1)
    sh.append(N_IMG)
    sh.append(IN_CH)
    return randn(sh^, SEED, STDtype.BF16, ctx)  # [1, 4096, 128] BF16


# ── Timestep embedding -> temb [1, DIM] ──────────────────────────────────────
# Mirrors lens_dit.rs timestep_embedding: cos-first, scale=1000 internally,
# max_period=10000. Mojo timestep_embedding IS cos-first (z-image order).
# Pre-scale sigma * 1000 since Mojo's timestep_embedding doesn't scale internally.
def make_temb(sigma: Float32, resident: LensResident, ctx: DeviceContext) raises -> Tensor:
    var tvals = List[Float32]()
    tvals.append(sigma * 1000.0)
    var tsh = List[Int]()
    tsh.append(1)
    var t = Tensor.from_host(tvals, tsh^, STDtype.F32, ctx)
    var proj = timestep_embedding(t, TEMB_DIM, ctx)  # [1, 256] F32
    var proj_bf16 = cast_tensor(proj, STDtype.BF16, ctx)
    var l1w = cast_tensor(resident.temb_lin1_w, STDtype.BF16, ctx)
    var l1b = cast_tensor(resident.temb_lin1_b, STDtype.BF16, ctx)
    var l2w = cast_tensor(resident.temb_lin2_w, STDtype.BF16, ctx)
    var l2b = cast_tensor(resident.temb_lin2_b, STDtype.BF16, ctx)
    var h1 = linear(proj_bf16, l1w, Optional[Tensor](l1b^), ctx)  # [1, 1536]
    var h2 = silu(h1, ctx)
    var temb = linear(h2, l2w, Optional[Tensor](l2b^), ctx)  # [1, 1536] BF16
    return temb^


# ── Single block forward (updates img_h and txt_e in-place via reassignment) ──
# Mirrors LensDiTBlock.forward in lens_dit.rs.
def lens_block_forward(
    mut img_h: Tensor,   # [1, N_IMG, DIM] BF16 — replaced with block output
    mut txt_e: Tensor,   # [1, N_TXT, DIM] BF16 — replaced with block output
    temb: Tensor,        # [1, DIM] BF16
    blk: Block,
    prefix: String,
    rope: LensRopeTables,
    ctx: DeviceContext,
) raises:
    var p = prefix + "."

    # ── 1) Modulation (silu(temb) -> mod linear -> 6 chunks) ──────────────
    var temb_act = silu(temb, ctx)

    var img_mod_w = cast_tensor(blk[p + "img_mod.1.weight"][], STDtype.BF16, ctx)
    var img_mod_b = cast_tensor(blk[p + "img_mod.1.bias"][], STDtype.BF16, ctx)
    var img_mod = linear(temb_act, img_mod_w, Optional[Tensor](img_mod_b^), ctx) # [1, 9216]

    var txt_mod_w = cast_tensor(blk[p + "txt_mod.1.weight"][], STDtype.BF16, ctx)
    var txt_mod_b = cast_tensor(blk[p + "txt_mod.1.bias"][], STDtype.BF16, ctx)
    var txt_mod = linear(temb_act, txt_mod_w, Optional[Tensor](txt_mod_b^), ctx) # [1, 9216]

    var img_shift1 = _adaln_chunk(img_mod, 0, ctx)
    var img_scale1 = _adaln_chunk(img_mod, 1, ctx)
    var img_gate1  = _adaln_chunk(img_mod, 2, ctx)
    var img_shift2 = _adaln_chunk(img_mod, 3, ctx)
    var img_scale2 = _adaln_chunk(img_mod, 4, ctx)
    var img_gate2  = _adaln_chunk(img_mod, 5, ctx)

    var txt_shift1 = _adaln_chunk(txt_mod, 0, ctx)
    var txt_scale1 = _adaln_chunk(txt_mod, 1, ctx)
    var txt_gate1  = _adaln_chunk(txt_mod, 2, ctx)
    var txt_shift2 = _adaln_chunk(txt_mod, 3, ctx)
    var txt_scale2 = _adaln_chunk(txt_mod, 4, ctx)
    var txt_gate2  = _adaln_chunk(txt_mod, 5, ctx)

    # ── 2) Norm1 + modulate ────────────────────────────────────────────────
    var img_n1w = cast_tensor(blk[p + "img_norm1.weight"][], STDtype.BF16, ctx)
    var img_n1  = rms_norm(img_h, img_n1w, BLOCK_NORM_EPS, ctx)
    var img_m1  = modulate(img_n1, img_scale1, img_shift1, ctx)

    var txt_n1w = cast_tensor(blk[p + "txt_norm1.weight"][], STDtype.BF16, ctx)
    var txt_n1  = rms_norm(txt_e, txt_n1w, BLOCK_NORM_EPS, ctx)
    var txt_m1  = modulate(txt_n1, txt_scale1, txt_shift1, ctx)

    # ── 3) QKV projections ─────────────────────────────────────────────────
    var iqkv_w = cast_tensor(blk[p + "attn.img_qkv.weight"][], STDtype.BF16, ctx)
    var iqkv_b = cast_tensor(blk[p + "attn.img_qkv.bias"][], STDtype.BF16, ctx)
    var img_qkv = linear(img_m1, iqkv_w, Optional[Tensor](iqkv_b^), ctx)  # [1,N_IMG,3*DIM]

    var tqkv_w = cast_tensor(blk[p + "attn.txt_qkv.weight"][], STDtype.BF16, ctx)
    var tqkv_b = cast_tensor(blk[p + "attn.txt_qkv.bias"][], STDtype.BF16, ctx)
    var txt_qkv = linear(txt_m1, tqkv_w, Optional[Tensor](tqkv_b^), ctx)  # [1,N_TXT,3*DIM]

    # Split QKV via slice on last dim
    var img_q_flat = slice(img_qkv, 2, 0,     DIM, ctx)
    var img_k_flat = slice(img_qkv, 2, DIM,   DIM, ctx)
    var img_v_flat = slice(img_qkv, 2, 2*DIM, DIM, ctx)
    var txt_q_flat = slice(txt_qkv, 2, 0,     DIM, ctx)
    var txt_k_flat = slice(txt_qkv, 2, DIM,   DIM, ctx)
    var txt_v_flat = slice(txt_qkv, 2, 2*DIM, DIM, ctx)

    # Reshape to [1, S, H, Dh]
    var img_q = _to_bshd[N_IMG](img_q_flat, ctx)
    var img_k = _to_bshd[N_IMG](img_k_flat, ctx)
    var img_v = _to_bshd[N_IMG](img_v_flat, ctx)
    var txt_q = _to_bshd[N_TXT](txt_q_flat, ctx)
    var txt_k = _to_bshd[N_TXT](txt_k_flat, ctx)
    var txt_v = _to_bshd[N_TXT](txt_v_flat, ctx)

    # ── 4) Per-head QK RMSNorm (eps=1e-5) ─────────────────────────────────
    var nq  = cast_tensor(blk[p + "attn.norm_q.weight"][], STDtype.BF16, ctx)
    var nk  = cast_tensor(blk[p + "attn.norm_k.weight"][], STDtype.BF16, ctx)
    var naq = cast_tensor(blk[p + "attn.norm_added_q.weight"][], STDtype.BF16, ctx)
    var nak = cast_tensor(blk[p + "attn.norm_added_k.weight"][], STDtype.BF16, ctx)
    img_q = rms_norm(img_q, nq,  QK_NORM_EPS, ctx)
    img_k = rms_norm(img_k, nk,  QK_NORM_EPS, ctx)
    txt_q = rms_norm(txt_q, naq, QK_NORM_EPS, ctx)
    txt_k = rms_norm(txt_k, nak, QK_NORM_EPS, ctx)

    # ── 5) RoPE (interleaved-pair, pre-tiled tables) ───────────────────────
    img_q = _apply_rope[N_IMG](img_q, rope.img_cos, rope.img_sin, ctx)
    img_k = _apply_rope[N_IMG](img_k, rope.img_cos, rope.img_sin, ctx)
    txt_q = _apply_rope[N_TXT](txt_q, rope.txt_cos, rope.txt_sin, ctx)
    txt_k = _apply_rope[N_TXT](txt_k, rope.txt_cos, rope.txt_sin, ctx)

    # ── 6) Joint SDPA (image first, then text) ─────────────────────────────
    var q_joint = concat(1, ctx, img_q, txt_q)  # [1, S=4160, H, Dh]
    var k_joint = concat(1, ctx, img_k, txt_k)
    var v_joint = concat(1, ctx, img_v, txt_v)

    var scale = Float32(1.0) / sqrt(Float32(HEAD_DIM))
    var attn = sdpa_nomask[1, S, NUM_HEADS, HEAD_DIM](q_joint, k_joint, v_joint, scale, ctx)

    var attn_flat = _from_bshd[S](attn, ctx)  # [1, S, DIM]
    var img_attn  = slice(attn_flat, 1, 0,     N_IMG, ctx)
    var txt_attn  = slice(attn_flat, 1, N_IMG, N_TXT, ctx)

    # Output projections
    var io_w = cast_tensor(blk[p + "attn.to_out.0.weight"][], STDtype.BF16, ctx)
    var io_b = cast_tensor(blk[p + "attn.to_out.0.bias"][], STDtype.BF16, ctx)
    var img_attn_proj = linear(img_attn, io_w, Optional[Tensor](io_b^), ctx)

    var to_w = cast_tensor(blk[p + "attn.to_add_out.weight"][], STDtype.BF16, ctx)
    var to_b = cast_tensor(blk[p + "attn.to_add_out.bias"][], STDtype.BF16, ctx)
    var txt_attn_proj = linear(txt_attn, to_w, Optional[Tensor](to_b^), ctx)

    # ── 7) Gate1 residual ──────────────────────────────────────────────────
    var img_h2 = residual_gate(img_h, img_gate1, img_attn_proj, ctx)
    var txt_e2 = residual_gate(txt_e, txt_gate1, txt_attn_proj, ctx)

    # ── 8) Norm2 + modulate2 + SwiGLU MLP + gate2 residual ────────────────
    var img_n2w = cast_tensor(blk[p + "img_norm2.weight"][], STDtype.BF16, ctx)
    var img_n2  = rms_norm(img_h2, img_n2w, BLOCK_NORM_EPS, ctx)
    var img_m2  = modulate(img_n2, img_scale2, img_shift2, ctx)

    var iw1 = cast_tensor(blk[p + "img_mlp.w1.weight"][], STDtype.BF16, ctx)
    var iw2 = cast_tensor(blk[p + "img_mlp.w2.weight"][], STDtype.BF16, ctx)
    var iw3 = cast_tensor(blk[p + "img_mlp.w3.weight"][], STDtype.BF16, ctx)
    var ig  = linear(img_m2, iw1, None, ctx)
    var iu  = linear(img_m2, iw3, None, ctx)
    var ia  = swiglu(ig, iu, ctx)
    var imo = linear(ia, iw2, None, ctx)
    var img_h3 = residual_gate(img_h2, img_gate2, imo, ctx)

    var txt_n2w = cast_tensor(blk[p + "txt_norm2.weight"][], STDtype.BF16, ctx)
    var txt_n2  = rms_norm(txt_e2, txt_n2w, BLOCK_NORM_EPS, ctx)
    var txt_m2  = modulate(txt_n2, txt_scale2, txt_shift2, ctx)

    var tw1 = cast_tensor(blk[p + "txt_mlp.w1.weight"][], STDtype.BF16, ctx)
    var tw2 = cast_tensor(blk[p + "txt_mlp.w2.weight"][], STDtype.BF16, ctx)
    var tw3 = cast_tensor(blk[p + "txt_mlp.w3.weight"][], STDtype.BF16, ctx)
    var tg  = linear(txt_m2, tw1, None, ctx)
    var tu  = linear(txt_m2, tw3, None, ctx)
    var ta  = swiglu(tg, tu, ctx)
    var tmo = linear(ta, tw2, None, ctx)
    var txt_e3 = residual_gate(txt_e2, txt_gate2, tmo, ctx)

    img_h = img_h3^
    txt_e = txt_e3^


# ── Final AdaLayerNormContinuous ──────────────────────────────────────────────
# lens_dit.rs:976-995: scale,shift=chunk(linear(silu(temb)),2,-1); normed=layernorm(x); out=normed*(1+scale)+shift
# chunk gives scale=first half, shift=second half.
def final_norm_proj(
    h: Tensor,  # [1, N_IMG, DIM]
    temb: Tensor,
    resident: LensResident,
    ctx: DeviceContext,
) raises -> Tensor:
    var temb_act = silu(temb, ctx)
    var nw = cast_tensor(resident.norm_out_w, STDtype.BF16, ctx)
    var nb = cast_tensor(resident.norm_out_b, STDtype.BF16, ctx)
    var mod_params = linear(temb_act, nw, Optional[Tensor](nb^), ctx) # [1, 2*DIM]
    var scale_1d = slice(mod_params, 1, 0,   DIM, ctx)  # [1, DIM]
    var shift_1d = slice(mod_params, 1, DIM, DIM, ctx)  # [1, DIM]
    var dim_sh = List[Int]()
    dim_sh.append(DIM)
    var scale = reshape(scale_1d, dim_sh.copy(), ctx)  # [DIM]
    var shift = reshape(shift_1d, dim_sh.copy(), ctx)
    # layer_norm (no affine): use ones/zeros
    var ln_ones  = _ones_bf16(DIM, ctx)
    var ln_zeros = _zeros_bf16(DIM, ctx)
    var normed = layer_norm(h, ln_ones, ln_zeros, FINAL_LN_EPS, ctx)
    var out = modulate(normed, scale, shift, ctx)  # [1, N_IMG, DIM]
    # proj_out [1, N_IMG, DIM] -> [1, N_IMG, IN_CH=128]
    var pw = cast_tensor(resident.proj_out_w, STDtype.BF16, ctx)
    var pb = cast_tensor(resident.proj_out_b, STDtype.BF16, ctx)
    return linear(out, pw, Optional[Tensor](pb^), ctx)


# ── Single DiT forward pass ──────────────────────────────────────────────────
def lens_forward(
    latents: Tensor,   # [1, N_IMG, 128] BF16
    txt_cond: Tensor,  # [1, N_TXT, DIM] BF16
    sigma: Float32,
    resident: LensResident,
    loader: BlockLoader,
    rope: LensRopeTables,
    ctx: DeviceContext,
) raises -> Tensor:
    # Project image patches: [1, N_IMG, 128] -> [1, N_IMG, DIM]
    var iiw = cast_tensor(resident.img_in_w, STDtype.BF16, ctx)
    var iib = cast_tensor(resident.img_in_b, STDtype.BF16, ctx)
    var h = linear(latents, iiw, Optional[Tensor](iib^), ctx)

    # Clone text cond for this step
    var e = _clone(txt_cond, ctx)

    # Timestep embedding
    var temb = make_temb(sigma, resident, ctx)  # [1, DIM]

    # 48-block streaming loop
    for i in range(NUM_LAYERS):
        var prefix = String("transformer_blocks.") + String(i)
        loader.prefetch_block(prefix)
        var blk = loader.load_block(prefix, ctx)
        lens_block_forward(h, e, temb, blk, prefix, rope, ctx)
        unload_block(blk^)
        if i % 8 == 0:
            print("    block", i + 1, "/", NUM_LAYERS)

    # Final norm + proj_out
    return final_norm_proj(h, temb, resident, ctx)  # [1, N_IMG, 128]


# ── CFG norm-rescale pair ─────────────────────────────────────────────────────
# Mirrors lens_infer.rs `cfg_norm_rescale_pair` (pipeline.py:502-511):
#   comb = uncond + cfg_scale * (cond - uncond)
#   cond_norm = ||cond||_2 per token over last dim (keepdim)
#   comb_norm = ||comb||_2 per token over last dim, clamped to 1e-12
#   noise_pred = comb * (cond_norm / comb_norm)
#
# Both cond and uncond are [1, N_IMG, IN_CH=128] BF16.
# Implemented host-side: N_IMG * IN_CH = 524_288 floats is small (~1 MB).
def cfg_norm_rescale_pair(
    cond: Tensor,    # [1, N_IMG, 128]
    uncond: Tensor,  # [1, N_IMG, 128]
    cfg_scale: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    var cond_h  = cond.to_host(ctx)
    var uncond_h = uncond.to_host(ctx)
    var n_tok = N_IMG
    var n_ch  = IN_CH
    var total = n_tok * n_ch

    # comb = uncond + cfg_scale * (cond - uncond)
    # cond_norm[i] = sqrt(sum_j cond[i,j]^2)
    # comb_norm[i] = max(sqrt(sum_j comb[i,j]^2), 1e-12)
    var out = List[Float32]()
    for i in range(total):
        out.append(0.0)

    # First pass: compute comb and both norms per token
    for tok in range(n_tok):
        var cond_sq_sum  = Float64(0.0)
        var comb_sq_sum  = Float64(0.0)
        var base = tok * n_ch
        # Temp buffer for comb row
        var comb_row = List[Float32]()
        for j in range(n_ch):
            comb_row.append(0.0)
        for j in range(n_ch):
            var c  = Float64(cond_h[base + j])
            var u  = Float64(uncond_h[base + j])
            var cm = u + Float64(cfg_scale) * (c - u)
            comb_row[j] = Float32(cm)
            cond_sq_sum  += c * c
            comb_sq_sum  += cm * cm
        var cond_norm = Float64(0.0)
        if cond_sq_sum > 0.0:
            cond_norm = cond_sq_sum ** 0.5
        var comb_norm = comb_sq_sum ** 0.5
        if comb_norm < 1.0e-12:
            comb_norm = 1.0e-12
        var scale = cond_norm / comb_norm
        for j in range(n_ch):
            out[base + j] = Float32(Float64(comb_row[j]) * scale)

    var sh = List[Int]()
    sh.append(1)
    sh.append(N_IMG)
    sh.append(IN_CH)
    return Tensor.from_host(out, sh^, STDtype.BF16, ctx)


# ── Denoise loop ──────────────────────────────────────────────────────────────
# Two-forward CFG per lens_infer.rs:
#   noise_cond  = lens_forward(latents, txt_cond,   sigma)
#   noise_uncond = lens_forward(latents, txt_uncond, sigma)  [or reuse if identical]
#   noise_pred  = cfg_norm_rescale_pair(noise_cond, noise_uncond, CFG_SCALE)
#
# When cond == uncond (zeroed text smoke path), the second forward is skipped
# because rescale degenerates to identity (scale = 1.0 everywhere).
# Pass `txt_uncond` as the same tensor to trigger the fast-path.
def denoise(
    resident: LensResident,
    txt_cond: Tensor,
    txt_uncond: Tensor,
    rope: LensRopeTables,
    ctx: DeviceContext,
) raises -> Tensor:
    var sched = LensFlowMatchScheduler.for_resolution(1024, 1024, NUM_STEPS)
    var sigmas = sched.sigmas()
    # Determine if cond == uncond (zeroed smoke path): compare shapes as a proxy
    # (both are the same [1,N_TXT,DIM] zeroed tensor in the smoke run).
    # We use pointer equality would require unsafe; instead check byte-count match
    # and compare first + last elements via to_host.
    var cond_h_chk  = txt_cond.to_host(ctx)
    var uncond_h_chk = txt_uncond.to_host(ctx)
    var same_cond = True
    var n_chk = len(cond_h_chk)
    if n_chk != len(uncond_h_chk):
        same_cond = False
    else:
        for ii in range(n_chk):
            if cond_h_chk[ii] != uncond_h_chk[ii]:
                same_cond = False
                break
    print("[denoise]", NUM_STEPS, "steps, cfg=", CFG_SCALE, "seed=", SEED)
    print("  mu=", sched.mu, "sigma[0]=", sigmas[0], "sigma[-1]=", sigmas[NUM_STEPS - 1])
    if same_cond:
        print("  [cfg] cond == uncond (zeroed smoke path): single forward, CFG identity")
    else:
        print("  [cfg] real prompt: two-forward CFG with norm-rescale")

    var loader = BlockLoader.open(String(TRANSFORMER_DIR))
    var latents = initial_noise(ctx)
    _stats("init_noise", latents, ctx)

    for step in range(NUM_STEPS):
        var sigma_curr = sigmas[step]
        var sigma_next = sched.sigma_next(step)
        print("  step", step + 1, "/", NUM_STEPS,
              "sigma", sigma_curr, "->", sigma_next)

        var noise_cond = lens_forward(
            latents, txt_cond, sigma_curr,
            resident, loader, rope, ctx,
        )
        var noise_pred: Tensor
        if same_cond:
            # Fast path: cond == uncond -> CFG is identity (scale = 1.0 everywhere)
            noise_pred = noise_cond^
        else:
            # Two-forward CFG: run uncond forward then norm-rescale
            var noise_uncond = lens_forward(
                latents, txt_uncond, sigma_curr,
                resident, loader, rope, ctx,
            )
            noise_pred = cfg_norm_rescale_pair(noise_cond, noise_uncond, CFG_SCALE, ctx)
        latents = lens_euler_step(latents, noise_pred, sigma_curr, sigma_next, ctx)

    _stats("final_latent", latents, ctx)
    return latents^


# ── VAE decode ────────────────────────────────────────────────────────────────
def vae_decode(latents: Tensor, ctx: DeviceContext) raises -> Tensor:
    # [1, N_IMG=4096, 128] BF16 -> [1, 128, 64, 64] F32 -> decode -> [1,3,1024,1024]
    var lat_f32 = cast_tensor(latents, STDtype.F32, ctx)
    var nhwc_sh = List[Int]()
    nhwc_sh.append(1)
    nhwc_sh.append(LH)
    nhwc_sh.append(LW)
    nhwc_sh.append(IN_CH)
    var nhwc = reshape(lat_f32, nhwc_sh^, ctx)
    var perm = List[Int]()
    perm.append(0)
    perm.append(3)
    perm.append(1)
    perm.append(2)
    var nchw = permute(nhwc, perm^, ctx)  # [1, 128, 64, 64]
    print("[vae] loading flux2-vae and decoding")
    var vae = KleinVaeDecoder[LH, LW].load(String(FLUX2_VAE_PATH), ctx)
    return vae.decode(nchw, ctx)  # [1, 3, 1024, 1024]


def main() raises:
    var ctx = DeviceContext()
    print("=== Lens 1024x1024 multistep pipeline ===")
    print("  output:", String(OUT_PATH))

    # Load resident weights
    print("[weights] loading Lens resident weights")
    var resident = LensResident.load(ctx)
    print("  resident weights loaded")

    # Build text conditioning (cached GPT-OSS hidden states).
    # For the zeroed smoke path both cond and uncond are the same zeroed projection.
    # The denoise() loop detects this and uses the CFG fast-path (single forward).
    var txt_cond = build_text_cond(resident, ctx)
    # Uncond: clone the same zeroed projection (smoke path: identical to cond).
    # For a real-prompt run, replace txt_uncond with a separate negative-prompt projection.
    var txt_uncond = _clone(txt_cond, ctx)

    # Build RoPE tables once
    var rope = build_lens_rope_tables(ctx)

    # Denoise (two-forward CFG wired; fast-paths to single forward for zeroed smoke)
    var latents = denoise(resident, txt_cond, txt_uncond, rope, ctx)

    # VAE decode
    var img = vae_decode(latents, ctx)
    var sh = img.shape()
    print("  image shape:", sh[0], sh[1], sh[2], sh[3])
    _stats("image", img, ctx)

    # Save PNG
    save_png(img, String(OUT_PATH), ctx, ValueRange.SIGNED)
    print("[done] saved", String(OUT_PATH))

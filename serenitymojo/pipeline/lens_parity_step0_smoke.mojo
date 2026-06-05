# lens_parity_step0_smoke.mojo — RNG-independent one-step DiT parity vs Python.
#
# The captures in lens/parity/captures/ contain (shape [2, N_IMG, 128] BF16):
#   hidden_states_pre_step_00.safetensors  key="hs"
#   noise_pred_step_00.safetensors         key="noise"
#
# Both halves of the CFG batch are identical (zeroed text features). We take
# [0:1] from each → [1, 4096, 128], run one Mojo DiT forward with sigma=1.0
# and zeroed text conditioning, then compute cos similarity vs the Python
# oracle. This is completely RNG-independent: the initial latent comes from
# Python's captured hidden state, not from our randn.
#
# Target: cos >= 0.999 (matches the Rust parity bar).

from std.gpu.host import DeviceContext
from std.math import sqrt, exp as fexp, cos as fcos, sin as fsin, pow as fpow

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.offload.block_loader import BlockLoader, Block, unload_block
from serenitymojo.ops.activations import silu, swiglu
from serenitymojo.ops.attention import sdpa_nomask
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.elementwise import modulate, residual_gate
from serenitymojo.ops.embeddings import timestep_embedding
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import rms_norm, layer_norm
from serenitymojo.ops.rope import rope_interleaved
from serenitymojo.ops.tensor_algebra import add, mul_scalar, reshape, permute, slice, concat
from serenitymojo.sampling.lens_flowmatch import LensFlowMatchScheduler

# ── Paths ────────────────────────────────────────────────────────────────────
comptime TRANSFORMER_DIR = "/home/alex/.serenity/models/microsoft_lens/transformer"
comptime TEXT_SMOKE_DIR  = "/home/alex/EriDiffusion/inference-flame/lens/parity/captures_text_smoke"
comptime CAPTURES_DIR    = "/home/alex/EriDiffusion/inference-flame/lens/parity/captures"
comptime HIDDEN_05 = TEXT_SMOKE_DIR + "/hidden_layer_05.safetensors"
comptime HIDDEN_11 = TEXT_SMOKE_DIR + "/hidden_layer_11.safetensors"
comptime HIDDEN_17 = TEXT_SMOKE_DIR + "/hidden_layer_17.safetensors"
comptime HIDDEN_23 = TEXT_SMOKE_DIR + "/hidden_layer_23.safetensors"
comptime HS_STEP0_PATH    = CAPTURES_DIR + "/hidden_states_pre_step_00.safetensors"
comptime NOISE_STEP0_PATH = CAPTURES_DIR + "/noise_pred_step_00.safetensors"

# ── Dimensions ───────────────────────────────────────────────────────────────
comptime LH = 64
comptime LW = 64
comptime N_IMG = LH * LW       # 4096
# The Python oracle masks ALL 256 text tokens (mask=0 = all padding).
# Image-only attention (N_TXT=0) is MATHEMATICALLY EQUIVALENT:
#   - softmax with mask=-inf on text positions = softmax with no text positions
#   - V-contribution from text = 0 either way
# So we test with N_TXT=0 to match the Python oracle's computation exactly,
# without needing masked SDPA in Mojo.
comptime N_TXT = 0
comptime S = N_IMG  # 4096 — image-only attention

comptime DIM = 1536
comptime NUM_HEADS = 24
comptime HEAD_DIM = 64
comptime ROPE_HALF = 32
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

comptime BLOCK_NORM_EPS = Float32(1.0e-6)
comptime QK_NORM_EPS    = Float32(1.0e-5)
comptime TXT_NORM_EPS   = Float32(1.0e-5)
comptime FINAL_LN_EPS   = Float32(1.0e-6)

# sigma at step 0 — matches capture_metadata.json sigmas[0]=1.0
comptime SIGMA_STEP0 = Float32(1.0)


# ── Stats ─────────────────────────────────────────────────────────────────────
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


def _adaln_chunk(mod_out: Tensor, idx: Int, ctx: DeviceContext) raises -> Tensor:
    var part = slice(mod_out, 1, idx * DIM, DIM, ctx)
    var sh = List[Int]()
    sh.append(DIM)
    return reshape(part, sh^, ctx)


def _to_bshd[S_: Int](x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var sh = List[Int]()
    sh.append(1)
    sh.append(S_)
    sh.append(NUM_HEADS)
    sh.append(HEAD_DIM)
    return reshape(x, sh^, ctx)


def _from_bshd[S_: Int](x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var sh = List[Int]()
    sh.append(1)
    sh.append(S_)
    sh.append(DIM)
    return reshape(x, sh^, ctx)


# ── RoPE ──────────────────────────────────────────────────────────────────────
@fieldwise_init
struct LensRopeTables(Movable):
    var img_cos: Tensor
    var img_sin: Tensor
    var txt_cos: Tensor
    var txt_sin: Tensor


def _apply_rope[S_: Int](
    x: Tensor, cos_tiled: Tensor, sin_tiled: Tensor, ctx: DeviceContext
) raises -> Tensor:
    var flat_sh = List[Int]()
    flat_sh.append(S_ * NUM_HEADS)
    flat_sh.append(HEAD_DIM)
    var x_flat = reshape(x, flat_sh^, ctx)
    var roped = rope_interleaved(x_flat, cos_tiled, sin_tiled, ctx)
    var bshd_sh = List[Int]()
    bshd_sh.append(1)
    bshd_sh.append(S_)
    bshd_sh.append(NUM_HEADS)
    bshd_sh.append(HEAD_DIM)
    return reshape(roped, bshd_sh^, ctx)


def build_lens_rope_tables(ctx: DeviceContext) raises -> LensRopeTables:
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
    axes.append(8)
    axes.append(28)
    axes.append(28)
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

    var h_lo = LH // 2
    var h_hi = LH - h_lo
    var w_lo = LW // 2
    var w_hi = LW - w_lo

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

    comptime MAX_VID_IDX = LH // 2
    var txt_cos_host = List[Float32]()
    var txt_sin_host = List[Float32]()
    for _ in range(N_TXT * ROPE_HALF):
        txt_cos_host.append(0.0)
        txt_sin_host.append(0.0)
    for i in range(N_TXT):
        var src_row = MAX_VID_IDX + i
        for k in range(ROPE_HALF):
            txt_cos_host[i * ROPE_HALF + k] = pos_cos_host[src_row * ROPE_HALF + k]
            txt_sin_host[i * ROPE_HALF + k] = pos_sin_host[src_row * ROPE_HALF + k]

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
    return LensRopeTables(ic^, is_^, tc^, ts^)


# ── Resident weights ──────────────────────────────────────────────────────────
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


# ── Build zeroed text cond (N_TXT=256 zeros, matching Python oracle) ───────────
# The Python oracle used fake_text_seq_len=256: four layers of [1,256,2880] zeros
# passed through rms_norm(w) + concat + txt_in. rms_norm on zeros = zeros (regardless
# of w), so cat4 = zeros [1,256,11520] and txt_in(zeros) = bias only.
# We build this exactly: zeros [1, N_TXT, TXT_IN_DIM] → linear with txt_in_w/b.
# This is equivalent to txt_in applied to zero input = bias broadcast to [1,N_TXT,DIM].
def build_text_cond(resident: LensResident, ctx: DeviceContext) raises -> Tensor:
    # Zeros [1, N_TXT, TXT_IN_DIM=11520] BF16
    var n_elem = N_TXT * TXT_IN_DIM
    var zeros = List[Float32]()
    for _ in range(n_elem):
        zeros.append(0.0)
    var sh = List[Int]()
    sh.append(1)
    sh.append(N_TXT)
    sh.append(TXT_IN_DIM)
    var cat4 = cast_tensor(
        Tensor.from_host(zeros, sh^, STDtype.F32, ctx), STDtype.BF16, ctx
    )
    var tin_w = cast_tensor(resident.txt_in_w, STDtype.BF16, ctx)
    var tin_b = cast_tensor(resident.txt_in_b, STDtype.BF16, ctx)
    return linear(cat4, tin_w, Optional[Tensor](tin_b^), ctx)  # [1, N_TXT, DIM]


# ── Timestep embedding ─────────────────────────────────────────────────────────
def make_temb(sigma: Float32, resident: LensResident, ctx: DeviceContext) raises -> Tensor:
    var tvals = List[Float32]()
    tvals.append(sigma * 1000.0)
    var tsh = List[Int]()
    tsh.append(1)
    var t = Tensor.from_host(tvals, tsh^, STDtype.F32, ctx)
    var proj_bf16 = timestep_embedding(
        t, TEMB_DIM, ctx, Float32(10000.0), STDtype.BF16
    )
    var l1w = cast_tensor(resident.temb_lin1_w, STDtype.BF16, ctx)
    var l1b = cast_tensor(resident.temb_lin1_b, STDtype.BF16, ctx)
    var l2w = cast_tensor(resident.temb_lin2_w, STDtype.BF16, ctx)
    var l2b = cast_tensor(resident.temb_lin2_b, STDtype.BF16, ctx)
    var h1 = linear(proj_bf16, l1w, Optional[Tensor](l1b^), ctx)
    var h2 = silu(h1, ctx)
    return linear(h2, l2w, Optional[Tensor](l2b^), ctx)


# ── Single block forward (image-only path — N_TXT=0) ──────────────────────────
# When N_TXT=0, we skip all text processing and run image-only self-attention.
# This matches the Python oracle which masks all text tokens (mask=0 for all
# 256 positions → mathematically equivalent to no text tokens at all).
def lens_block_forward_img_only(
    mut img_h: Tensor,   # [1, N_IMG, DIM] BF16
    temb: Tensor,        # [1, DIM] BF16
    blk: Block,
    prefix: String,
    rope: LensRopeTables,
    ctx: DeviceContext,
) raises:
    var p = prefix + "."
    var temb_act = silu(temb, ctx)

    # Image modulation (6 chunks)
    var img_mod_w = cast_tensor(blk[p + "img_mod.1.weight"][], STDtype.BF16, ctx)
    var img_mod_b = cast_tensor(blk[p + "img_mod.1.bias"][], STDtype.BF16, ctx)
    var img_mod = linear(temb_act, img_mod_w, Optional[Tensor](img_mod_b^), ctx)

    var img_shift1 = _adaln_chunk(img_mod, 0, ctx)
    var img_scale1 = _adaln_chunk(img_mod, 1, ctx)
    var img_gate1  = _adaln_chunk(img_mod, 2, ctx)
    var img_shift2 = _adaln_chunk(img_mod, 3, ctx)
    var img_scale2 = _adaln_chunk(img_mod, 4, ctx)
    var img_gate2  = _adaln_chunk(img_mod, 5, ctx)

    # RMSNorm1 + modulate
    var img_n1w = cast_tensor(blk[p + "img_norm1.weight"][], STDtype.BF16, ctx)
    var img_n1  = rms_norm(img_h, img_n1w, BLOCK_NORM_EPS, ctx)
    var img_m1  = modulate(img_n1, img_scale1, img_shift1, ctx)

    # QKV projection → image-only
    var iqkv_w = cast_tensor(blk[p + "attn.img_qkv.weight"][], STDtype.BF16, ctx)
    var iqkv_b = cast_tensor(blk[p + "attn.img_qkv.bias"][], STDtype.BF16, ctx)
    var img_qkv = linear(img_m1, iqkv_w, Optional[Tensor](iqkv_b^), ctx)  # [1,N_IMG,3*DIM]

    var img_q_flat = slice(img_qkv, 2, 0,     DIM, ctx)
    var img_k_flat = slice(img_qkv, 2, DIM,   DIM, ctx)
    var img_v_flat = slice(img_qkv, 2, 2*DIM, DIM, ctx)

    var img_q = _to_bshd[N_IMG](img_q_flat, ctx)
    var img_k = _to_bshd[N_IMG](img_k_flat, ctx)
    var img_v = _to_bshd[N_IMG](img_v_flat, ctx)

    # QK RMSNorm
    var nq = cast_tensor(blk[p + "attn.norm_q.weight"][], STDtype.BF16, ctx)
    var nk = cast_tensor(blk[p + "attn.norm_k.weight"][], STDtype.BF16, ctx)
    img_q = rms_norm(img_q, nq, QK_NORM_EPS, ctx)
    img_k = rms_norm(img_k, nk, QK_NORM_EPS, ctx)

    # RoPE (image only)
    img_q = _apply_rope[N_IMG](img_q, rope.img_cos, rope.img_sin, ctx)
    img_k = _apply_rope[N_IMG](img_k, rope.img_cos, rope.img_sin, ctx)

    # Image-only self-attention: S = N_IMG (no text tokens)
    var scale = Float32(1.0) / sqrt(Float32(HEAD_DIM))
    var attn = sdpa_nomask[1, N_IMG, NUM_HEADS, HEAD_DIM](img_q, img_k, img_v, scale, ctx)

    # Output projection (image only)
    var attn_flat = _from_bshd[N_IMG](attn, ctx)  # [1, N_IMG, DIM]
    var io_w = cast_tensor(blk[p + "attn.to_out.0.weight"][], STDtype.BF16, ctx)
    var io_b = cast_tensor(blk[p + "attn.to_out.0.bias"][], STDtype.BF16, ctx)
    var img_attn_proj = linear(attn_flat, io_w, Optional[Tensor](io_b^), ctx)

    # Gate1 residual
    var img_h2 = residual_gate(img_h, img_gate1, img_attn_proj, ctx)

    # RMSNorm2 + modulate2 + SwiGLU MLP + gate2
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

    img_h = img_h3^


# ── Final norm + proj ─────────────────────────────────────────────────────────
def final_norm_proj(
    h: Tensor,
    temb: Tensor,
    resident: LensResident,
    ctx: DeviceContext,
) raises -> Tensor:
    var temb_act = silu(temb, ctx)
    var nw = cast_tensor(resident.norm_out_w, STDtype.BF16, ctx)
    var nb = cast_tensor(resident.norm_out_b, STDtype.BF16, ctx)
    var mod_params = linear(temb_act, nw, Optional[Tensor](nb^), ctx)
    var scale_1d = slice(mod_params, 1, 0,   DIM, ctx)
    var shift_1d = slice(mod_params, 1, DIM, DIM, ctx)
    var dim_sh = List[Int]()
    dim_sh.append(DIM)
    var scale = reshape(scale_1d, dim_sh.copy(), ctx)
    var shift = reshape(shift_1d, dim_sh.copy(), ctx)
    var ln_ones  = _ones_bf16(DIM, ctx)
    var ln_zeros = _zeros_bf16(DIM, ctx)
    var normed = layer_norm(h, ln_ones, ln_zeros, FINAL_LN_EPS, ctx)
    var out = modulate(normed, scale, shift, ctx)
    var pw = cast_tensor(resident.proj_out_w, STDtype.BF16, ctx)
    var pb = cast_tensor(resident.proj_out_b, STDtype.BF16, ctx)
    return linear(out, pw, Optional[Tensor](pb^), ctx)


# ── One DiT forward (image-only path, N_TXT=0) ────────────────────────────────
# Matches Python oracle which masks all text tokens: image-only self-attention
# is mathematically equivalent to joint attention with mask=0 on all text.
def lens_forward_one_step(
    latents: Tensor,
    sigma: Float32,
    resident: LensResident,
    loader: BlockLoader,
    rope: LensRopeTables,
    ctx: DeviceContext,
) raises -> Tensor:
    var iiw = cast_tensor(resident.img_in_w, STDtype.BF16, ctx)
    var iib = cast_tensor(resident.img_in_b, STDtype.BF16, ctx)
    var h = linear(latents, iiw, Optional[Tensor](iib^), ctx)  # [1, N_IMG, DIM]

    var temb = make_temb(sigma, resident, ctx)

    for i in range(NUM_LAYERS):
        var prefix = String("transformer_blocks.") + String(i)
        loader.prefetch_block(prefix)
        var blk = loader.load_block(prefix, ctx)
        lens_block_forward_img_only(h, temb, blk, prefix, rope, ctx)
        unload_block(blk^)
        if i % 8 == 0:
            print("    block", i + 1, "/", NUM_LAYERS)

    return final_norm_proj(h, temb, resident, ctx)


# ── Cosine similarity (host-side) ─────────────────────────────────────────────
def cosine_sim(a: List[Float32], b: List[Float32]) raises -> Float64:
    var n = len(a)
    if n != len(b):
        raise Error("cosine_sim: length mismatch")
    var dot = Float64(0.0)
    var sa2 = Float64(0.0)
    var sb2 = Float64(0.0)
    for i in range(n):
        var av = Float64(a[i])
        var bv = Float64(b[i])
        dot += av * bv
        sa2 += av * av
        sb2 += bv * bv
    var denom = (sa2 * sb2) ** 0.5
    if denom <= 0.0:
        return Float64(0.0)
    return dot / denom


def max_abs_diff(a: List[Float32], b: List[Float32]) raises -> Float32:
    var n = len(a)
    if n != len(b):
        raise Error("max_abs_diff: length mismatch")
    var mx = Float32(0.0)
    for i in range(n):
        var d = a[i] - b[i]
        if d < 0.0:
            d = -d
        if d > mx:
            mx = d
    return mx


def mean_abs_diff(a: List[Float32], b: List[Float32]) raises -> Float32:
    var n = len(a)
    if n != len(b):
        raise Error("mean_abs_diff: length mismatch")
    var s = Float64(0.0)
    for i in range(n):
        var d = a[i] - b[i]
        if d < 0.0:
            d = -d
        s += Float64(d)
    return Float32(s / Float64(n))


# ── Main: one-step parity test ─────────────────────────────────────────────────
def main() raises:
    var ctx = DeviceContext()
    print("=== Lens DiT one-step parity (Mojo vs Python captures) ===")
    print("  hs_step0 :", String(HS_STEP0_PATH))
    print("  noise_ref:", String(NOISE_STEP0_PATH))
    print("  sigma_0  :", SIGMA_STEP0)

    # ── Load Python-captured hidden state (initial latents for step 0) ──────
    print("[load] hidden_states_pre_step_00 (shape [2,4096,128] BF16, taking [0:1])")
    var st_hs = ShardedSafeTensors.open(String(HS_STEP0_PATH))
    var hs_full = Tensor.from_view(st_hs.tensor_view(String("hs")), ctx)  # [2,4096,128]
    var latents = slice(hs_full, 0, 0, 1, ctx)  # [1,4096,128]
    _stats("latents_from_capture", latents, ctx)

    # ── Load Python oracle noise_pred (step 0) ────────────────────────────
    print("[load] noise_pred_step_00 (shape [2,4096,128] BF16, taking [0:1])")
    var st_np = ShardedSafeTensors.open(String(NOISE_STEP0_PATH))
    var np_full = Tensor.from_view(st_np.tensor_view(String("noise")), ctx)  # [2,4096,128]
    var noise_ref = slice(np_full, 0, 0, 1, ctx)  # [1,4096,128]
    _stats("noise_ref", noise_ref, ctx)

    # ── Load resident weights ─────────────────────────────────────────────
    print("[weights] loading Lens resident weights")
    var resident = LensResident.load(ctx)

    # ── Build RoPE tables (image-only: no text tables needed) ─────────────
    print("[rope] building RoPE tables")
    var rope = build_lens_rope_tables(ctx)

    # ── Run ONE DiT forward (image-only, N_TXT=0) ─────────────────────────
    # Matches Python oracle: text is fully masked → image tokens self-attend only.
    print("[dit] running image-only DiT forward at sigma=", SIGMA_STEP0)
    var loader = BlockLoader.open(String(TRANSFORMER_DIR))
    var noise_mojo = lens_forward_one_step(
        latents, SIGMA_STEP0,
        resident, loader, rope, ctx,
    )
    _stats("noise_mojo", noise_mojo, ctx)

    # ── Compute parity metrics ────────────────────────────────────────────
    print("[parity] computing cosine similarity vs Python oracle")
    var mojo_h = noise_mojo.to_host(ctx)
    var ref_h  = noise_ref.to_host(ctx)
    var cos  = cosine_sim(mojo_h, ref_h)
    var maxd = max_abs_diff(mojo_h, ref_h)
    var meand = mean_abs_diff(mojo_h, ref_h)

    print()
    print("============================================================")
    print("LENS DiT STEP-0 PARITY (Mojo vs Python)")
    print("  cos      =", Float32(cos))
    print("  max_abs  =", maxd)
    print("  mean_abs =", meand)
    var verdict = String("FAIL")
    if cos >= 0.999:
        verdict = String("PASS")
    print("  verdict  =", verdict, "(target cos >= 0.999)")
    print("============================================================")

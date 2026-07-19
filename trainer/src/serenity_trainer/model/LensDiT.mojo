# LensDiT.mojo — the Lens DiT forward (LensTransformer2DModel), COPIED into the
# serenity_trainer namespace from serenitymojo's WORKING Lens pipeline
# (pipeline/lens_pipeline_1024_multistep.mojo: lens_forward, lens_block_forward,
# LensResident, LensRopeTables, build_lens_rope_tables, make_temb,
# final_norm_proj). The math/structure/order are unchanged; only the namespace and
# the train/infer split are Lens-port flavored.
#
# BORROW boundary: serenitymojo.{tensor,io,ops,offload} are imported UNCHANGED
# (foundation tier). The DiT forward itself is COPIED here (NOT a serenitymojo
# import) per the port's borrow rule. LoRA adapters come from
# serenity_trainer.module.LoRAModule.
#
# FORWARD MATH verified against lens/transformer.py (LensTransformer2DModel):
#   - LensTransformerBlock.forward (transformer.py:392-419): img_mod/txt_mod =
#     Linear(SiLU(temb)).chunk(2) -> two (shift,scale,gate) triples each
#     (_modulate :360-363: x*(1+scale)+shift, gate). RMSNorm img_norm1/txt_norm1
#     (rms_norm=True, eps 1e-6).
#   - LensJointAttention.forward (transformer.py:240-300): fused img_qkv/txt_qkv
#     Linear(+bias) -> split q,k,v; QK RMSNorm norm_q/norm_k (img) +
#     norm_added_q/norm_added_k (txt), dim_head=64 eps 1e-6; complex RoPE
#     (apply_rotary_emb_lens :74-86); joint cat([img,txt]) SDPA (scale 1/sqrt(Dh));
#     to_out.0 (img) / to_add_out (txt).
#   - gate1 residual; img_norm2/txt_norm2 + modulate2; GateMLP SwiGLU
#     (w2(silu(w1(x))*w3(x)), hidden=int(dim/3*8)=4096); gate2 residual.
#   - AdaLayerNormContinuous norm_out (transformer.py:485-487, elementwise_affine
#     =False, eps 1e-6): chunk(Linear(SiLU(temb)),2) -> scale,shift; LayerNorm(x);
#     x*(1+scale)+shift) ; proj_out Linear(1536 -> patch^2*out_ch = 128).
#
# RoPE NOTE: the borrowed build_lens_rope_tables encodes apply_rotary_emb_lens's
# complex rotation as interleaved-pair (cos,sin) tables for rope_interleaved, with
# the 3-axis (frame=8,H=28,W=28) split, theta=10000, scale_rope=True
# (LensEmbedRope, transformer.py:104-180). It is specialized to LH=LW=64 (1024px),
# the verified path.
#
# MASK NOTE: transformer.py ALWAYS builds an additive joint mask and threads it
# into every block's SDPA (forward :501-503 -> block.attn :346-351 ->
# F.scaled_dot_product_attention(..., attn_mask=attention_mask) :276-277). The
# mask is _build_joint_attention_mask (transformer.py:536-554): image positions
# always valid, text positions per encoder_hidden_states_mask, -inf on padded.
#   - INFER fast path (lens_block_forward_infer / lens_forward_full_infer) keeps
#     sdpa_nomask: it is numerically EXACT only for the verified bs=1 / full
#     (unpadded) text 1024 case, where the joint mask is all-zero.
#   - TRAIN path (lens_block_forward_lora / lens_forward_full_lora) consumes the
#     per-sample text mask and uses the masked sdpa so padded text tokens cannot
#     leak into the joint softmax. _build_joint_attention_mask (below) materializes
#     the additive [1,NUM_HEADS,S,S] mask once per forward (constant across query
#     rows + heads; serenitymojo.ops.sdpa needs the full [B,H,S,S] form), then all
#     48 blocks reuse it.
#
# TRAIN/INFER SPLIT (the Z-Image lesson):
#   - lens_forward_full_infer : NO activation saving, NO LoRA `down` clones; the
#     sampler path (torch.no_grad()).
#   - lens_forward_full_lora  : applies LoRA deltas on the trained projections AND
#     saves the per-block stream inputs (recompute checkpoints) for the
#     hand-chained backward (a later slice consumes the LensStackForward bundle).
#
# DTYPE: BF16 storage in/out, F32 only inside foundation kernels.

from std.math import sqrt, exp as fexp, cos as fcos, sin as fsin, pow as fpow
from std.memory import ArcPointer
from std.gpu.host import DeviceContext
from os import getenv

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.offload.block_loader import BlockLoader, Block, unload_block
from serenitymojo.ops.activations import silu, swiglu
from serenitymojo.ops.attention import sdpa_nomask, sdpa, sdpa_chunked
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.elementwise import modulate, residual_gate
from serenitymojo.ops.embeddings import timestep_embedding
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import rms_norm, layer_norm
from serenitymojo.ops.rope import rope_interleaved
from serenitymojo.ops.tensor_algebra import (
    add, sub, mul_scalar, div, reshape, permute, slice, concat,
)

from serenity_trainer.module.LoRAModule import LoraAdapter, make_lora_adapter
from serenity_trainer.modelLoader.LensModelLoader import LensWeights


comptime TArc = ArcPointer[Tensor]
comptime LArc = ArcPointer[LoraAdapter]


# ── Dimensions (checkpoint config.json; 1024px verified path) ─────────────────
comptime LH = 64
comptime LW = 64
comptime N_IMG = LH * LW          # 4096
comptime N_TXT = 64
comptime S = N_IMG + N_TXT        # 4160

comptime DIM = 1536
comptime NUM_HEADS = 24
comptime HEAD_DIM = 64
comptime ROPE_HALF = 32           # sum(axes_dim)/2 = (8+28+28)/2
comptime MLP_HIDDEN = 4096        # int(dim/3*8)
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
comptime QK_NORM_EPS    = Float32(1.0e-6)   # attn.norm_q/k/added (RMSNorm dim_head, eps=1e-6: transformer.py:295 block default eps=1e-6 -> :306 attn eps -> :210-213 RMSNorm; model builds blocks w/o override :424-432)
comptime TXT_NORM_EPS   = Float32(1.0e-5)   # GPT-OSS per-layer RMSNorm(enc_hidden_dim, eps=1e-5) — distinct from QK norm
comptime FINAL_LN_EPS   = Float32(1.0e-6)

# ── LoRA slot layout (10 trained Linears per LensTransformerBlock) ────────────
# 0 attn.img_qkv | 1 attn.txt_qkv | 2 attn.to_out.0 | 3 attn.to_add_out
# 4 img_mlp.w1   | 5 img_mlp.w3   | 6 img_mlp.w2     | 7 txt_mlp.w1
# 8 txt_mlp.w3   | 9 txt_mlp.w2
comptime LORA_IMG_QKV = 0
comptime LORA_TXT_QKV = 1
comptime LORA_TO_OUT = 2
comptime LORA_TO_ADD_OUT = 3
comptime LORA_IMG_W1 = 4
comptime LORA_IMG_W3 = 5
comptime LORA_IMG_W2 = 6
comptime LORA_TXT_W1 = 7
comptime LORA_TXT_W3 = 8
comptime LORA_TXT_W2 = 9
comptime LORA_SLOTS_PER_BLOCK = 10


# ── VRAM diagnostic (env-gated; LENS_MEM_DEBUG=1) ─────────────────────────────
# get_memory_info() returns (free, total) bytes; prints USED MiB at a tag. Reads
# the env each call so it stays a no-op (no GPU sync) unless explicitly enabled.
def _mem_dbg(ctx: DeviceContext, tag: String) raises:
    if getenv("LENS_MEM_DEBUG") != String("1"):
        return
    var mi = ctx.get_memory_info()
    var free = Float64(Int(mi[0]))
    var total = Float64(Int(mi[1]))
    var used = total - free
    print("[mem]", tag, " used =", Int(used / (1024.0 * 1024.0)), "MiB  free =",
          Int(free / (1024.0 * 1024.0)), "MiB")


# ── helpers (copied from the pipeline) ─────────────────────────────────────────
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
    return Tensor.from_host(vals, sh^, STDtype.BF16, ctx)


def _zeros_bf16(n: Int, ctx: DeviceContext) raises -> Tensor:
    var vals = List[Float32]()
    for _ in range(n):
        vals.append(0.0)
    var sh = List[Int]()
    sh.append(n)
    return Tensor.from_host(vals, sh^, STDtype.BF16, ctx)


def _adaln_chunk(mod_out: Tensor, idx: Int, ctx: DeviceContext) raises -> Tensor:
    var part = slice(mod_out, 1, idx * DIM, DIM, ctx)
    var sh = List[Int]()
    sh.append(DIM)
    return reshape(part, sh^, ctx)


def _to_bshd[S_: Int](x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var sh = List[Int]()
    sh.append(1); sh.append(S_); sh.append(NUM_HEADS); sh.append(HEAD_DIM)
    return reshape(x, sh^, ctx)


def _from_bshd[S_: Int](x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var sh = List[Int]()
    sh.append(1); sh.append(S_); sh.append(DIM)
    return reshape(x, sh^, ctx)


# ── RoPE table storage (copied) ───────────────────────────────────────────────
@fieldwise_init
struct LensRopeTables(Movable):
    var img_cos: Tensor
    var img_sin: Tensor
    var txt_cos: Tensor
    var txt_sin: Tensor


# Build 3-axis Lens RoPE tables (copied from the working pipeline; the
# interleaved-(cos,sin) encoding of apply_rotary_emb_lens, axes=[8,28,28],
# theta=10000, scale_rope=True). LensEmbedRope (transformer.py:104-184).
#
# PARAMETRIC (Finding 2): generalized over the latent grid (LH_, LW_) and the text
# length N_TXT_ so the trainer can build correct-size tables for variable
# resolution / variable (cropped, pruned) caption length. Mirrors
# LensEmbedRope.forward (transformer.py:128-164): img freqs from
# _compute_video_freqs(frame=1, height=LH_, width=LW_) and
# txt_freqs = pos_freqs[max_vid_index : max_vid_index + text_seq_len] with
# max_vid_index = max(height//2, width//2) under scale_rope (transformer.py:157).
# The 1024px verified path is build_lens_rope_tables_1024 == [64, 64, 64].
def build_lens_rope_tables[
    LH_: Int, LW_: Int, N_TXT_: Int
](ctx: DeviceContext) raises -> LensRopeTables:
    comptime N_IMG_ = LH_ * LW_
    comptime MAX_VID_IDX = max(LH_ // 2, LW_ // 2)  # transformer.py:157
    constrained[
        MAX_VID_IDX + N_TXT_ <= ROPE_TABLE_ROWS,
        "Lens RoPE: max_vid_index + text_seq_len exceeds the 4096 freq table rows",
    ]()
    constrained[
        LH_ <= ROPE_TABLE_ROWS and LW_ <= ROPE_TABLE_ROWS,
        "Lens RoPE: latent grid dim exceeds the 4096 freq table rows",
    ]()

    var pos_cos_host = List[Float32]()
    var pos_sin_host = List[Float32]()
    var neg_cos_host = List[Float32]()
    var neg_sin_host = List[Float32]()
    for _ in range(ROPE_TABLE_ROWS * ROPE_HALF):
        pos_cos_host.append(0.0); pos_sin_host.append(0.0)
        neg_cos_host.append(0.0); neg_sin_host.append(0.0)

    var axes = List[Int]()
    axes.append(8); axes.append(28); axes.append(28)
    var halfs = List[Int]()
    halfs.append(AXES_FRAME_HALF); halfs.append(AXES_H_HALF); halfs.append(AXES_W_HALF)

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

    var h_lo = LH_ // 2
    var h_hi = LH_ - h_lo
    var w_lo = LW_ // 2
    var w_hi = LW_ - w_lo

    var height_cos = List[Float32]()
    var height_sin = List[Float32]()
    for _ in range(LH_ * AXES_H_HALF):
        height_cos.append(0.0); height_sin.append(0.0)
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
    for _ in range(LW_ * AXES_W_HALF):
        width_cos.append(0.0); width_sin.append(0.0)
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
    for _ in range(N_IMG_ * ROPE_HALF):
        img_cos_host.append(0.0); img_sin_host.append(0.0)
    for yy in range(LH_):
        for xx in range(LW_):
            var dst_row = (yy * LW_ + xx) * ROPE_HALF
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

    var txt_cos_host = List[Float32]()
    var txt_sin_host = List[Float32]()
    for _ in range(N_TXT_ * ROPE_HALF):
        txt_cos_host.append(0.0); txt_sin_host.append(0.0)
    for i in range(N_TXT_):
        var src_row = MAX_VID_IDX + i
        for k in range(ROPE_HALF):
            var src = src_row * ROPE_HALF + k
            txt_cos_host[i * ROPE_HALF + k] = pos_cos_host[src]
            txt_sin_host[i * ROPE_HALF + k] = pos_sin_host[src]

    var img_cos_tiled = List[Float32]()
    var img_sin_tiled = List[Float32]()
    for i in range(N_IMG_):
        for _ in range(NUM_HEADS):
            for k in range(ROPE_HALF):
                img_cos_tiled.append(img_cos_host[i * ROPE_HALF + k])
                img_sin_tiled.append(img_sin_host[i * ROPE_HALF + k])

    var txt_cos_tiled = List[Float32]()
    var txt_sin_tiled = List[Float32]()
    for i in range(N_TXT_):
        for _ in range(NUM_HEADS):
            for k in range(ROPE_HALF):
                txt_cos_tiled.append(txt_cos_host[i * ROPE_HALF + k])
                txt_sin_tiled.append(txt_sin_host[i * ROPE_HALF + k])

    var ic_sh = List[Int]()
    ic_sh.append(N_IMG_ * NUM_HEADS); ic_sh.append(ROPE_HALF)
    var tc_sh = List[Int]()
    tc_sh.append(N_TXT_ * NUM_HEADS); tc_sh.append(ROPE_HALF)

    var ic = Tensor.from_host(img_cos_tiled, ic_sh.copy(), STDtype.BF16, ctx)
    var is_ = Tensor.from_host(img_sin_tiled, ic_sh.copy(), STDtype.BF16, ctx)
    var tc = Tensor.from_host(txt_cos_tiled, tc_sh.copy(), STDtype.BF16, ctx)
    var ts = Tensor.from_host(txt_sin_tiled, tc_sh.copy(), STDtype.BF16, ctx)
    return LensRopeTables(ic^, is_^, tc^, ts^)


# 1024px verified fast path: the original specialization (LH=LW=64, N_TXT=64).
def build_lens_rope_tables_1024(ctx: DeviceContext) raises -> LensRopeTables:
    return build_lens_rope_tables[LH, LW, N_TXT](ctx)


def _apply_rope[S_: Int](
    x: Tensor, cos_tiled: Tensor, sin_tiled: Tensor, ctx: DeviceContext
) raises -> Tensor:
    var flat_sh = List[Int]()
    flat_sh.append(S_ * NUM_HEADS); flat_sh.append(HEAD_DIM)
    var x_flat = reshape(x, flat_sh^, ctx)
    var roped = rope_interleaved(x_flat, cos_tiled, sin_tiled, ctx)
    var bshd_sh = List[Int]()
    bshd_sh.append(1); bshd_sh.append(S_); bshd_sh.append(NUM_HEADS); bshd_sh.append(HEAD_DIM)
    return reshape(roped, bshd_sh^, ctx)


# ── Additive joint attention mask (Finding 1) ─────────────────────────────────
# 1:1 port of LensTransformer2DModel._build_joint_attention_mask
# (transformer.py:536-554):
#   joint    = cat([img_ones(img_len), text_mask], dim=1)   # bool, True = valid
#   additive = zeros_like(joint, f32); additive[~joint] = -inf
#   return additive[:, None, None, :]                        # [B,1,1,img+txt]
# Image tokens are ALWAYS valid; text positions follow text_mask. PyTorch broadcasts
# the [B,1,1,N] additive bias over query rows and heads inside SDPA. serenitymojo's
# sdpa needs the FULL [B,H,S,S] additive tensor, so we materialize the broadcast:
# every (head, query-row) gets the same per-key bias row. Built ONCE per forward and
# reused by all 48 blocks. `text_mask` is host-side [N_TXT_] (1.0 = valid, 0.0 =
# padded). NEG large-magnitude stands in for -inf (exp→0 in the F32 softmax; image
# tokens keep every row from being fully masked, so no NaN). BF16 to match q/k/v.
comptime LENS_MASK_NEG = Float32(-1.0e30)


def _build_joint_attention_mask[
    N_IMG_: Int, N_TXT_: Int
](text_mask: List[Float32], ctx: DeviceContext) raises -> Tensor:
    comptime S_ = N_IMG_ + N_TXT_
    if len(text_mask) != N_TXT_:
        raise Error(
            "Lens joint mask: text_mask length does not match N_TXT_"
        )
    # Per-key additive bias row [S_]: image keys 0, text keys 0 if valid else NEG.
    var key_bias = List[Float32]()
    for _ in range(N_IMG_):
        key_bias.append(Float32(0.0))
    for j in range(N_TXT_):
        if text_mask[j] != Float32(0.0):
            key_bias.append(Float32(0.0))
        else:
            key_bias.append(LENS_MASK_NEG)
    # Broadcast to [1, NUM_HEADS, S_, S_] (same bias for every head + query row).
    var flat = List[Float32]()
    for _ in range(NUM_HEADS):
        for _ in range(S_):
            for j in range(S_):
                flat.append(key_bias[j])
    var sh = List[Int]()
    sh.append(1); sh.append(NUM_HEADS); sh.append(S_); sh.append(S_)
    return Tensor.from_host(flat, sh^, STDtype.BF16, ctx)


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
    def load(transformer_dir: String, ctx: DeviceContext) raises -> LensResident:
        var st = ShardedSafeTensors.open(transformer_dir)
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
            img_in_w^, img_in_b^, txt_in_w^, txt_in_b^,
            tn0^, tn1^, tn2^, tn3^,
            tl1w^, tl1b^, tl2w^, tl2b^,
            now_w^, now_b^, proj_out_w^, proj_out_b^,
        )


# ── Text conditioning: 4 per-layer features -> projected [1,N_TXT,DIM] ─────────
# transformer.py:639-646: normed=[txt_norm[i](feat[i])]; cat(dim=-1); txt_in(cat).
# Per-layer RMSNorm(enc_hidden_dim, eps 1e-5) -> concat -> Linear(11520->1536).
def build_text_cond_from_feats(
    resident: LensResident, feats: List[TArc], ctx: DeviceContext
) raises -> Tensor:
    var tn0 = cast_tensor(resident.txt_norm0_w, feats[0][].dtype(), ctx)
    var tn1 = cast_tensor(resident.txt_norm1_w, feats[1][].dtype(), ctx)
    var tn2 = cast_tensor(resident.txt_norm2_w, feats[2][].dtype(), ctx)
    var tn3 = cast_tensor(resident.txt_norm3_w, feats[3][].dtype(), ctx)
    var n05 = rms_norm(feats[0][], tn0, TXT_NORM_EPS, ctx)
    var n11 = rms_norm(feats[1][], tn1, TXT_NORM_EPS, ctx)
    var n17 = rms_norm(feats[2][], tn2, TXT_NORM_EPS, ctx)
    var n23 = rms_norm(feats[3][], tn3, TXT_NORM_EPS, ctx)
    var cat4 = concat(2, ctx, n05, n11, n17, n23)
    var cat4_dt = cat4.dtype()
    var tin_w = cast_tensor(resident.txt_in_w, cat4_dt, ctx)
    var tin_b = cast_tensor(resident.txt_in_b, cat4_dt, ctx)
    return linear(cat4, tin_w, Optional[Tensor](tin_b^), ctx)  # [1,N_TXT,1536]


# ── Timestep embedding -> temb [1, DIM] ───────────────────────────────────────
# LensTimestepProjEmbeddings (transformer.py:88-101): Timesteps(256,
# flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000) -> TimestepEmbedder
# (Linear -> SiLU -> Linear). Mojo timestep_embedding is cos-first; pre-scale by
# 1000 (scale=1000) since it does not scale internally.
def make_temb(sigma: Float32, resident: LensResident, ctx: DeviceContext) raises -> Tensor:
    var tvals = List[Float32]()
    tvals.append(sigma * 1000.0)
    var tsh = List[Int]()
    tsh.append(1)
    var t = Tensor.from_host(tvals, tsh^, STDtype.F32, ctx)
    var proj_bf16 = timestep_embedding(t, TEMB_DIM, ctx, Float32(10000.0), STDtype.BF16)
    var h1 = _linw_b(proj_bf16, resident.temb_lin1_w, resident.temb_lin1_b, ctx)
    var h2 = silu(h1, ctx)
    var temb = _linw_b(h2, resident.temb_lin2_w, resident.temb_lin2_b, ctx)
    return temb^


# ── LoRA delta: scale * ((x @ Aᵀ) @ Bᵀ). x [.,in], a [rank,in], b [out,rank]. ──
def _lora_delta(x: Tensor, ad: LoraAdapter, ctx: DeviceContext) raises -> Tensor:
    var down = linear(x, ad.a, None, ctx)          # [., rank]
    var up = linear(down, ad.b, None, ctx)         # [., out]
    return mul_scalar(up, ad.scale(), ctx)


# ── Resident-BF16 weight consumption (edv2-Klein fast path) ───────────────────
# The block/resident weight stores are already BF16 (LensWeights/LensResident cast
# at LOAD; LensModelLoader.mojo:100). The previous code re-cast EVERY weight to BF16
# on EVERY block (cast_tensor defaults synchronize=True → a host stall per call,
# thousands per denoise). These helpers consume an already-BF16 weight DIRECTLY (no
# copy, no host sync — matches edv2 klein's native-bf16 fused block), and fall back
# to a NO-SYNC cast only when the source is genuinely F32 (e.g. the streaming
# load_block path on an F32 checkpoint). Numerically bit-identical to the old
# cast-then-GEMM for both dtypes (a BF16→BF16 cast was a pure copy; the F32→BF16
# cast value is unchanged — only the per-op `ctx.synchronize()` is removed). The
# BF16-storage / F32-compute boundary inside `linear`/`rms_norm` is untouched.
def _linw(x: Tensor, w_raw: Tensor, ctx: DeviceContext) raises -> Tensor:
    """No-bias linear consuming a resident weight (BF16 direct, F32 no-sync cast)."""
    if w_raw.dtype() == STDtype.BF16:
        return linear(x, w_raw, None, ctx)
    var wc = cast_tensor(w_raw, STDtype.BF16, ctx, False)
    return linear(x, wc, None, ctx)


def _linw_b(
    x: Tensor, w_raw: Tensor, b_raw: Tensor, ctx: DeviceContext
) raises -> Tensor:
    """Biased linear consuming resident weight+bias. The bias must be OWNED for the
    `Optional[Tensor]` arg, so it is a no-sync BF16 copy (tiny vs the weight); the
    weight itself is consumed in place when already BF16 (no copy, no sync)."""
    var bias = cast_tensor(b_raw, STDtype.BF16, ctx, False)
    if w_raw.dtype() == STDtype.BF16:
        return linear(x, w_raw, Optional[Tensor](bias^), ctx)
    var wc = cast_tensor(w_raw, STDtype.BF16, ctx, False)
    return linear(x, wc, Optional[Tensor](bias^), ctx)


# ══════════════════════════════════════════════════════════════════════════════
# Single block forward — INFERENCE (no activation save). Copied from the working
# pipeline lens_block_forward (which is already activation-free). Updates img_h /
# txt_e in place via reassignment. LensTransformerBlock.forward (transformer.py).
# ══════════════════════════════════════════════════════════════════════════════
def lens_block_forward_infer(
    mut img_h: Tensor,
    mut txt_e: Tensor,
    temb: Tensor,
    blk: Block,
    prefix: String,
    rope: LensRopeTables,
    ctx: DeviceContext,
) raises:
    var p = prefix + "."
    var temb_act = silu(temb, ctx)

    var img_mod = _linw_b(temb_act, blk[p + "img_mod.1.weight"][], blk[p + "img_mod.1.bias"][], ctx)
    var txt_mod = _linw_b(temb_act, blk[p + "txt_mod.1.weight"][], blk[p + "txt_mod.1.bias"][], ctx)

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

    var img_n1  = rms_norm(img_h, blk[p + "img_norm1.weight"][], BLOCK_NORM_EPS, ctx)
    var img_m1  = modulate(img_n1, img_scale1, img_shift1, ctx)
    var txt_n1  = rms_norm(txt_e, blk[p + "txt_norm1.weight"][], BLOCK_NORM_EPS, ctx)
    var txt_m1  = modulate(txt_n1, txt_scale1, txt_shift1, ctx)

    var img_qkv = _linw_b(img_m1, blk[p + "attn.img_qkv.weight"][], blk[p + "attn.img_qkv.bias"][], ctx)
    var txt_qkv = _linw_b(txt_m1, blk[p + "attn.txt_qkv.weight"][], blk[p + "attn.txt_qkv.bias"][], ctx)

    var img_q_flat = slice(img_qkv, 2, 0,     DIM, ctx)
    var img_k_flat = slice(img_qkv, 2, DIM,   DIM, ctx)
    var img_v_flat = slice(img_qkv, 2, 2*DIM, DIM, ctx)
    var txt_q_flat = slice(txt_qkv, 2, 0,     DIM, ctx)
    var txt_k_flat = slice(txt_qkv, 2, DIM,   DIM, ctx)
    var txt_v_flat = slice(txt_qkv, 2, 2*DIM, DIM, ctx)

    var img_q = _to_bshd[N_IMG](img_q_flat, ctx)
    var img_k = _to_bshd[N_IMG](img_k_flat, ctx)
    var img_v = _to_bshd[N_IMG](img_v_flat, ctx)
    var txt_q = _to_bshd[N_TXT](txt_q_flat, ctx)
    var txt_k = _to_bshd[N_TXT](txt_k_flat, ctx)
    var txt_v = _to_bshd[N_TXT](txt_v_flat, ctx)

    img_q = rms_norm(img_q, blk[p + "attn.norm_q.weight"][],       QK_NORM_EPS, ctx)
    img_k = rms_norm(img_k, blk[p + "attn.norm_k.weight"][],       QK_NORM_EPS, ctx)
    txt_q = rms_norm(txt_q, blk[p + "attn.norm_added_q.weight"][], QK_NORM_EPS, ctx)
    txt_k = rms_norm(txt_k, blk[p + "attn.norm_added_k.weight"][], QK_NORM_EPS, ctx)

    img_q = _apply_rope[N_IMG](img_q, rope.img_cos, rope.img_sin, ctx)
    img_k = _apply_rope[N_IMG](img_k, rope.img_cos, rope.img_sin, ctx)
    txt_q = _apply_rope[N_TXT](txt_q, rope.txt_cos, rope.txt_sin, ctx)
    txt_k = _apply_rope[N_TXT](txt_k, rope.txt_cos, rope.txt_sin, ctx)

    var q_joint = concat(1, ctx, img_q, txt_q)
    var k_joint = concat(1, ctx, img_k, txt_k)
    var v_joint = concat(1, ctx, img_v, txt_v)
    var scale = Float32(1.0) / sqrt(Float32(HEAD_DIM))
    var attn = sdpa_nomask[1, S, NUM_HEADS, HEAD_DIM](q_joint, k_joint, v_joint, scale, ctx)

    var attn_flat = _from_bshd[S](attn, ctx)
    var img_attn  = slice(attn_flat, 1, 0,     N_IMG, ctx)
    var txt_attn  = slice(attn_flat, 1, N_IMG, N_TXT, ctx)

    var img_attn_proj = _linw_b(img_attn, blk[p + "attn.to_out.0.weight"][], blk[p + "attn.to_out.0.bias"][], ctx)
    var txt_attn_proj = _linw_b(txt_attn, blk[p + "attn.to_add_out.weight"][], blk[p + "attn.to_add_out.bias"][], ctx)

    var img_h2 = residual_gate(img_h, img_gate1, img_attn_proj, ctx)
    var txt_e2 = residual_gate(txt_e, txt_gate1, txt_attn_proj, ctx)

    var img_n2  = rms_norm(img_h2, blk[p + "img_norm2.weight"][], BLOCK_NORM_EPS, ctx)
    var img_m2  = modulate(img_n2, img_scale2, img_shift2, ctx)
    var ig  = _linw(img_m2, blk[p + "img_mlp.w1.weight"][], ctx)
    var iu  = _linw(img_m2, blk[p + "img_mlp.w3.weight"][], ctx)
    var ia  = swiglu(ig, iu, ctx)
    var imo = _linw(ia, blk[p + "img_mlp.w2.weight"][], ctx)
    var img_h3 = residual_gate(img_h2, img_gate2, imo, ctx)

    var txt_n2  = rms_norm(txt_e2, blk[p + "txt_norm2.weight"][], BLOCK_NORM_EPS, ctx)
    var txt_m2  = modulate(txt_n2, txt_scale2, txt_shift2, ctx)
    var tg  = _linw(txt_m2, blk[p + "txt_mlp.w1.weight"][], ctx)
    var tu  = _linw(txt_m2, blk[p + "txt_mlp.w3.weight"][], ctx)
    var ta  = swiglu(tg, tu, ctx)
    var tmo = _linw(ta, blk[p + "txt_mlp.w2.weight"][], ctx)
    var txt_e3 = residual_gate(txt_e2, txt_gate2, tmo, ctx)

    img_h = img_h3^
    txt_e = txt_e3^


# ══════════════════════════════════════════════════════════════════════════════
# Single block forward — LoRA TRAIN variant. SAME math as the infer block, but
# additively overlays the 10 trained-projection LoRA deltas. Activation saving for
# the hand-chained backward is done at the STACK level (per-block stream inputs as
# recompute checkpoints, the §8.3 contract used by the Z-Image vertical), so this
# block does not build a per-tensor acts bundle. `loras` = 10 LArc in slot order.
# ══════════════════════════════════════════════════════════════════════════════
def lens_block_forward_lora[N_IMG_: Int, N_TXT_: Int](
    mut img_h: Tensor,
    mut txt_e: Tensor,
    temb: Tensor,
    blk: Block,
    prefix: String,
    rope: LensRopeTables,
    loras: List[LArc],
    mask: Tensor,
    ctx: DeviceContext,
) raises:
    comptime S_ = N_IMG_ + N_TXT_
    var p = prefix + "."
    var temb_act = silu(temb, ctx)

    var img_mod = _linw_b(temb_act, blk[p + "img_mod.1.weight"][], blk[p + "img_mod.1.bias"][], ctx)
    var txt_mod = _linw_b(temb_act, blk[p + "txt_mod.1.weight"][], blk[p + "txt_mod.1.bias"][], ctx)

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

    var img_n1  = rms_norm(img_h, blk[p + "img_norm1.weight"][], BLOCK_NORM_EPS, ctx)
    var img_m1  = modulate(img_n1, img_scale1, img_shift1, ctx)
    var txt_n1  = rms_norm(txt_e, blk[p + "txt_norm1.weight"][], BLOCK_NORM_EPS, ctx)
    var txt_m1  = modulate(txt_n1, txt_scale1, txt_shift1, ctx)

    var img_qkv = _linw_b(img_m1, blk[p + "attn.img_qkv.weight"][], blk[p + "attn.img_qkv.bias"][], ctx)
    img_qkv = add(img_qkv, _lora_delta(img_m1, loras[LORA_IMG_QKV][], ctx), ctx)
    var txt_qkv = _linw_b(txt_m1, blk[p + "attn.txt_qkv.weight"][], blk[p + "attn.txt_qkv.bias"][], ctx)
    txt_qkv = add(txt_qkv, _lora_delta(txt_m1, loras[LORA_TXT_QKV][], ctx), ctx)

    var img_q_flat = slice(img_qkv, 2, 0,     DIM, ctx)
    var img_k_flat = slice(img_qkv, 2, DIM,   DIM, ctx)
    var img_v_flat = slice(img_qkv, 2, 2*DIM, DIM, ctx)
    var txt_q_flat = slice(txt_qkv, 2, 0,     DIM, ctx)
    var txt_k_flat = slice(txt_qkv, 2, DIM,   DIM, ctx)
    var txt_v_flat = slice(txt_qkv, 2, 2*DIM, DIM, ctx)

    var img_q = _to_bshd[N_IMG_](img_q_flat, ctx)
    var img_k = _to_bshd[N_IMG_](img_k_flat, ctx)
    var img_v = _to_bshd[N_IMG_](img_v_flat, ctx)
    var txt_q = _to_bshd[N_TXT_](txt_q_flat, ctx)
    var txt_k = _to_bshd[N_TXT_](txt_k_flat, ctx)
    var txt_v = _to_bshd[N_TXT_](txt_v_flat, ctx)

    img_q = rms_norm(img_q, blk[p + "attn.norm_q.weight"][],       QK_NORM_EPS, ctx)
    img_k = rms_norm(img_k, blk[p + "attn.norm_k.weight"][],       QK_NORM_EPS, ctx)
    txt_q = rms_norm(txt_q, blk[p + "attn.norm_added_q.weight"][], QK_NORM_EPS, ctx)
    txt_k = rms_norm(txt_k, blk[p + "attn.norm_added_k.weight"][], QK_NORM_EPS, ctx)

    img_q = _apply_rope[N_IMG_](img_q, rope.img_cos, rope.img_sin, ctx)
    img_k = _apply_rope[N_IMG_](img_k, rope.img_cos, rope.img_sin, ctx)
    txt_q = _apply_rope[N_TXT_](txt_q, rope.txt_cos, rope.txt_sin, ctx)
    txt_k = _apply_rope[N_TXT_](txt_k, rope.txt_cos, rope.txt_sin, ctx)

    var q_joint = concat(1, ctx, img_q, txt_q)
    var k_joint = concat(1, ctx, img_k, txt_k)
    var v_joint = concat(1, ctx, img_v, txt_v)
    _mem_dbg(ctx, prefix + " after-qkv (joint q/k/v built)")
    var scale = Float32(1.0) / sqrt(Float32(HEAD_DIM))
    # MASKED SDPA (Finding 1): joint attention with the additive text-padding mask
    # (transformer.py:276-277 F.scaled_dot_product_attention(..., attn_mask=mask)).
    # sdpa_chunked: head-chunked math SDPA — bit-identical to sdpa but the F32
    # [B*H,S,S] scores slab is never materialized (one [S,S] head buffer is reused),
    # so the per-block attention peak drops from O(H*S^2) to O(S^2). See attention.mojo.
    var attn = sdpa_chunked[1, S_, NUM_HEADS, HEAD_DIM](q_joint, k_joint, v_joint, mask, scale, ctx)
    _mem_dbg(ctx, prefix + " after-sdpa")

    var attn_flat = _from_bshd[S_](attn, ctx)
    var img_attn  = slice(attn_flat, 1, 0,      N_IMG_, ctx)
    var txt_attn  = slice(attn_flat, 1, N_IMG_, N_TXT_, ctx)

    var img_attn_proj = _linw_b(img_attn, blk[p + "attn.to_out.0.weight"][], blk[p + "attn.to_out.0.bias"][], ctx)
    img_attn_proj = add(img_attn_proj, _lora_delta(img_attn, loras[LORA_TO_OUT][], ctx), ctx)
    var txt_attn_proj = _linw_b(txt_attn, blk[p + "attn.to_add_out.weight"][], blk[p + "attn.to_add_out.bias"][], ctx)
    txt_attn_proj = add(txt_attn_proj, _lora_delta(txt_attn, loras[LORA_TO_ADD_OUT][], ctx), ctx)

    var img_h2 = residual_gate(img_h, img_gate1, img_attn_proj, ctx)
    var txt_e2 = residual_gate(txt_e, txt_gate1, txt_attn_proj, ctx)

    var img_n2  = rms_norm(img_h2, blk[p + "img_norm2.weight"][], BLOCK_NORM_EPS, ctx)
    var img_m2  = modulate(img_n2, img_scale2, img_shift2, ctx)
    var ig  = _linw(img_m2, blk[p + "img_mlp.w1.weight"][], ctx)
    ig = add(ig, _lora_delta(img_m2, loras[LORA_IMG_W1][], ctx), ctx)
    var iu  = _linw(img_m2, blk[p + "img_mlp.w3.weight"][], ctx)
    iu = add(iu, _lora_delta(img_m2, loras[LORA_IMG_W3][], ctx), ctx)
    var ia  = swiglu(ig, iu, ctx)
    var imo = _linw(ia, blk[p + "img_mlp.w2.weight"][], ctx)
    imo = add(imo, _lora_delta(ia, loras[LORA_IMG_W2][], ctx), ctx)
    var img_h3 = residual_gate(img_h2, img_gate2, imo, ctx)

    var txt_n2  = rms_norm(txt_e2, blk[p + "txt_norm2.weight"][], BLOCK_NORM_EPS, ctx)
    var txt_m2  = modulate(txt_n2, txt_scale2, txt_shift2, ctx)
    var tg  = _linw(txt_m2, blk[p + "txt_mlp.w1.weight"][], ctx)
    tg = add(tg, _lora_delta(txt_m2, loras[LORA_TXT_W1][], ctx), ctx)
    var tu  = _linw(txt_m2, blk[p + "txt_mlp.w3.weight"][], ctx)
    tu = add(tu, _lora_delta(txt_m2, loras[LORA_TXT_W3][], ctx), ctx)
    var ta  = swiglu(tg, tu, ctx)
    var tmo = _linw(ta, blk[p + "txt_mlp.w2.weight"][], ctx)
    tmo = add(tmo, _lora_delta(ta, loras[LORA_TXT_W2][], ctx), ctx)
    var txt_e3 = residual_gate(txt_e2, txt_gate2, tmo, ctx)

    img_h = img_h3^
    txt_e = txt_e3^
    _mem_dbg(ctx, prefix + " after-mlp (block end)")


# ── Final AdaLayerNormContinuous + proj_out (transformer.py:485-487, 514). ─────
# scale,shift = chunk(Linear(SiLU(temb)),2,-1); normed = LayerNorm(x, no affine);
# out = normed*(1+scale)+shift; proj_out(out). chunk -> scale=first, shift=second.
def final_norm_proj(
    h: Tensor, temb: Tensor, resident: LensResident, ctx: DeviceContext
) raises -> Tensor:
    var temb_act = silu(temb, ctx)
    var mod_params = _linw_b(temb_act, resident.norm_out_w, resident.norm_out_b, ctx)
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
    return _linw_b(out, resident.proj_out_w, resident.proj_out_b, ctx)   # [1,N_IMG,128]


# ══════════════════════════════════════════════════════════════════════════════
# TOP-LEVEL FORWARD — INFERENCE (no activations, no LoRA `down`). The sampler path.
# Copied from the working pipeline lens_forward. `txt_cond` is the already-projected
# [1,N_TXT,DIM] (build_text_cond_from_feats); the 48 blocks are streamed.
# ══════════════════════════════════════════════════════════════════════════════
def lens_forward_full_infer(
    latents: Tensor,        # [1, N_IMG, 128] BF16
    txt_cond: Tensor,       # [1, N_TXT, DIM] BF16
    sigma: Float32,
    resident: LensResident,
    loader: BlockLoader,
    rope: LensRopeTables,
    ctx: DeviceContext,
) raises -> Tensor:
    var h = _linw_b(latents, resident.img_in_w, resident.img_in_b, ctx)
    var e = _clone(txt_cond, ctx)
    var temb = make_temb(sigma, resident, ctx)
    for i in range(NUM_LAYERS):
        var prefix = String("transformer_blocks.") + String(i)
        loader.prefetch_block(prefix)
        var blk = loader.load_block(prefix, ctx)
        lens_block_forward_infer(h, e, temb, blk, prefix, rope, ctx)
        unload_block(blk^)
    return final_norm_proj(h, temb, resident, ctx)


# ══════════════════════════════════════════════════════════════════════════════
# TOP-LEVEL FORWARD — LoRA TRAIN. Applies LoRA deltas on the trained projections
# AND saves the per-block stream inputs (img_h, txt_e BEFORE each block) as the
# recompute checkpoints the hand-chained backward (a later slice) consumes. Returns
# the LensStackForward bundle (velocity [1,N_IMG,128] + the 48 (img,txt) inputs +
# temb). `loras` is block-major (block b slot s at index b*10+s).
# ══════════════════════════════════════════════════════════════════════════════
struct LensStackForward(Movable):
    var velocity: Tensor          # [1, N_IMG, 128] BF16 (proj_out)
    var img_inputs: List[TArc]    # [NUM_LAYERS] block img-stream input checkpoints
    var txt_inputs: List[TArc]    # [NUM_LAYERS] block txt-stream input checkpoints
    var temb: Tensor              # [1, DIM] BF16 (shared modulation source)

    def __init__(
        out self,
        var velocity: Tensor, var img_inputs: List[TArc],
        var txt_inputs: List[TArc], var temb: Tensor,
    ):
        self.velocity = velocity^
        self.img_inputs = img_inputs^
        self.txt_inputs = txt_inputs^
        self.temb = temb^


def _loras_for_block(loras: List[LArc], b: Int) -> List[LArc]:
    var base = b * LORA_SLOTS_PER_BLOCK
    var out = List[LArc]()
    for s in range(LORA_SLOTS_PER_BLOCK):
        out.append(loras[base + s])    # ArcPointer copy = shared ref
    return out^


def lens_forward_full_lora[N_IMG_: Int, N_TXT_: Int](
    latents: Tensor,        # [1, N_IMG_, 128] BF16
    txt_cond: Tensor,       # [1, N_TXT_, DIM] BF16
    text_mask: List[Float32],  # [N_TXT_] host (1.0 = valid, 0.0 = padded)
    sigma: Float32,
    resident: LensResident,
    loader: BlockLoader,
    rope: LensRopeTables,
    loras: List[LArc],      # [NUM_LAYERS * 10] block-major
    ctx: DeviceContext,
) raises -> LensStackForward:
    var h = _linw_b(latents, resident.img_in_w, resident.img_in_b, ctx)
    var e = _clone(txt_cond, ctx)
    var temb = make_temb(sigma, resident, ctx)

    # Build the additive joint attention mask ONCE (transformer.py:501-503 builds it
    # before the block loop and threads the SAME mask into every block).
    var mask = _build_joint_attention_mask[N_IMG_, N_TXT_](text_mask, ctx)

    var img_inputs = List[TArc]()
    var txt_inputs = List[TArc]()
    for i in range(NUM_LAYERS):
        # CHECKPOINT: save the stream inputs to this block (backward recompute).
        img_inputs.append(TArc(_clone(h, ctx)))
        txt_inputs.append(TArc(_clone(e, ctx)))
        var prefix = String("transformer_blocks.") + String(i)
        loader.prefetch_block(prefix)
        var blk = loader.load_block(prefix, ctx)
        var bl = _loras_for_block(loras, i)
        lens_block_forward_lora[N_IMG_, N_TXT_](h, e, temb, blk, prefix, rope, bl, mask, ctx)
        unload_block(blk^)

    var velocity = final_norm_proj(h, temb, resident, ctx)
    return LensStackForward(velocity^, img_inputs^, txt_inputs^, temb^)


# ══════════════════════════════════════════════════════════════════════════════
# SMOKE-FACING IN-MEMORY FORWARD + COLD-START LoRA SET (SLICE-A parity contract).
#
# These satisfy the lens_forward_parity_smoke contract (in-memory LensWeights store
# + a B=0 LoRA overlay) WITHOUT a second disk read or a BlockLoader. The math is
# UNCHANGED: it reuses build_text_cond_from_feats / make_temb / build_lens_rope_tables
# / _build_joint_attention_mask / lens_block_forward_lora / final_norm_proj. With
# the LoRA B factors all 0 the per-Linear deltas are exactly 0, so the forward is
# identical to the no-LoRA base (the "identity overlay" the smoke documents).
# ══════════════════════════════════════════════════════════════════════════════

# Integer sqrt (comptime): the square latent grid side from the image-token count
# (S_IMG = LH*LW, square verified path → LH = LW = isqrt(S_IMG)).
fn _isqrt(n: Int) -> Int:
    var r = 1
    while (r + 1) * (r + 1) <= n:
        r += 1
    return r


# (in_features, out_features) of the wrapped Linear at LoRA slot `slot` (matches
# lensLoraTargets / lens_backward._block_slot_dims).
def _lora_slot_dims(slot: Int) raises -> Tuple[Int, Int]:
    if slot == LORA_IMG_QKV or slot == LORA_TXT_QKV:
        return (DIM, 3 * DIM)
    if slot == LORA_TO_OUT or slot == LORA_TO_ADD_OUT:
        return (DIM, DIM)
    if slot == LORA_IMG_W1 or slot == LORA_TXT_W1:
        return (DIM, MLP_HIDDEN)
    if slot == LORA_IMG_W3 or slot == LORA_TXT_W3:
        return (DIM, MLP_HIDDEN)
    if slot == LORA_IMG_W2 or slot == LORA_TXT_W2:
        return (MLP_HIDDEN, DIM)
    raise Error(String("_lora_slot_dims: bad slot ") + String(slot))


# Cold-start LoRA overlay set (480 = 48*10, block-major slot-minor; A~kaiming-ish,
# B=0 → identity overlay at init). Tensor-backed adapters (module.LoRAModule), the
# form lens_block_forward_lora consumes.
def build_lens_lora_set(rank: Int, alpha: Float32, ctx: DeviceContext) raises -> List[LArc]:
    var out = List[LArc]()
    var seed = UInt64(0x1234ABCD)
    for _b in range(NUM_LAYERS):
        for slot in range(LORA_SLOTS_PER_BLOCK):
            var dims = _lora_slot_dims(slot)
            seed = seed * UInt64(6364136223846793005) + UInt64(1)
            out.append(LArc(make_lora_adapter(dims[0], dims[1], rank, alpha, seed, ctx)))
    return out^


# Materialize a LensResident from the in-memory LensWeights store (clones the 16
# resident tensors; no disk read).
def _resident_from_weights(weights: LensWeights, ctx: DeviceContext) raises -> LensResident:
    return LensResident(
        _clone(weights.get(String("img_in.weight")), ctx),
        _clone(weights.get(String("img_in.bias")), ctx),
        _clone(weights.get(String("txt_in.weight")), ctx),
        _clone(weights.get(String("txt_in.bias")), ctx),
        _clone(weights.get(String("txt_norm.0.weight")), ctx),
        _clone(weights.get(String("txt_norm.1.weight")), ctx),
        _clone(weights.get(String("txt_norm.2.weight")), ctx),
        _clone(weights.get(String("txt_norm.3.weight")), ctx),
        _clone(weights.get(String("time_text_embed.timestep_embedder.linear_1.weight")), ctx),
        _clone(weights.get(String("time_text_embed.timestep_embedder.linear_1.bias")), ctx),
        _clone(weights.get(String("time_text_embed.timestep_embedder.linear_2.weight")), ctx),
        _clone(weights.get(String("time_text_embed.timestep_embedder.linear_2.bias")), ctx),
        _clone(weights.get(String("norm_out.linear.weight")), ctx),
        _clone(weights.get(String("norm_out.linear.bias")), ctx),
        _clone(weights.get(String("proj_out.weight")), ctx),
        _clone(weights.get(String("proj_out.bias")), ctx),
    )


# Build an in-memory Block (name→Tensor) for one transformer block by SHARING the
# refcounted handles already held by LensWeights (no copy; the block forward casts
# each to BF16 into fresh tensors and never mutates the Block).
def _block_from_weights(weights: LensWeights, prefix: String, ctx: DeviceContext) raises -> Block:
    _ = ctx
    var blk = Block()
    var p = prefix + "."
    for ref e in weights.name_to_idx.items():
        if e.key.startswith(p):
            blk[e.key] = weights.weights[e.value]
    return blk^


# In-memory parity forward: hidden[1,S_IMG,128] + 4 per-layer txt feats[1,S_TXT,2880]
# + mask[1,S_TXT] + timestep (AS-IS) → flow [1,S_IMG,128]. `loras` is the 480-adapter
# overlay (B=0 ⇒ identity). Square latent grid LH=LW=isqrt(S_IMG).
def lens_forward_full_infer[S_IMG: Int, S_TXT: Int](
    hidden: Tensor,
    txt0: Tensor, txt1: Tensor, txt2: Tensor, txt3: Tensor,
    mask: Tensor,
    timestep: Float32,
    weights: LensWeights,
    loras: List[LArc],
    ctx: DeviceContext,
) raises -> Tensor:
    comptime LHv = _isqrt(S_IMG)
    comptime LWv = LHv

    var resident = _resident_from_weights(weights, ctx)

    # per-layer text features → projected txt_cond [1,S_TXT,DIM].
    var feats = List[TArc]()
    feats.append(TArc(_clone(txt0, ctx)))
    feats.append(TArc(_clone(txt1, ctx)))
    feats.append(TArc(_clone(txt2, ctx)))
    feats.append(TArc(_clone(txt3, ctx)))
    var txt_cond = build_text_cond_from_feats(resident, feats, ctx)

    # host attention mask (1.0 = valid) → additive joint mask [1,H,S,S].
    var mask_h = mask.to_host(ctx)
    var mask_list = List[Float32]()
    for j in range(S_TXT):
        mask_list.append(mask_h[j])
    var jmask = _build_joint_attention_mask[S_IMG, S_TXT](mask_list, ctx)
    _mem_dbg(ctx, "after jmask build")

    var rope = build_lens_rope_tables[LHv, LWv, S_TXT](ctx)
    _mem_dbg(ctx, "after rope build")

    var h = _linw_b(hidden, resident.img_in_w, resident.img_in_b, ctx)
    var e = _clone(txt_cond, ctx)
    var temb = make_temb(timestep, resident, ctx)
    _mem_dbg(ctx, "before block loop (img_in done)")

    for i in range(NUM_LAYERS):
        var prefix = String("transformer_blocks.") + String(i)
        var blk = _block_from_weights(weights, prefix, ctx)
        var bl = _loras_for_block(loras, i)
        lens_block_forward_lora[S_IMG, S_TXT](h, e, temb, blk, prefix, rope, bl, jmask, ctx)
        unload_block(blk^)

    return final_norm_proj(h, temb, resident, ctx)

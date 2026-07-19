# models/dit/boogu_dit.mojo — Boogu-Image DiT input embedders (pure Mojo, inference).
#
# SCOPE: embedders ONLY (Chunk C1). NO rope, NO attention, NO transformer blocks,
# NO patchify reshape — those are later chunks. This file mirrors the input-side
# of `Lumina2CombinedTimestepCaptionEmbedding` + the `x_embedder` Linear, read
# line-by-line from the reference (NOT inferred):
#   /home/alex/Boogu-Image/boogu/models/transformers/block_lumina2.py:177-219
#   /home/alex/Boogu-Image/boogu/models/embeddings.py:24-77 (TimestepEmbedding)
#   /home/alex/Boogu-Image/boogu/models/transformers/transformer_boogu.py:840-855
#
# Config (Boogu-Image-0.1-Base): hidden_size=3360, patch_size=2, in_channels=16,
# instruction_feat_dim=4096, frequency_embedding_size=256,
# time_embed_dim=min(3360,1024)=1024, timestep_scale=1000.0, norm_eps=1e-5.
#
# ── x_embedder (transformer_boogu.py:840) ───────────────────────────────────
#   nn.Linear(patch*patch*in_channels = 64, hidden_size = 3360, bias=True).
#   Input is ALREADY-patchified tokens [B, L_img, 64]; output [B, L_img, 3360].
#   == ops/linear.linear(tokens, x_embedder.weight, x_embedder.bias).
#
# ── time_caption_embed.forward (block_lumina2.py:210-219) ───────────────────
#   timestep_proj = Timesteps(num_channels=256, flip_sin_to_cos=True,
#                             downscale_freq_shift=0.0, scale=1000.0)(timestep)
#       = diffusers get_timestep_embedding: half=128;
#         freq_i = exp(-ln(10000)*i/(half - 0)) = exp(-ln(10000)*i/half);
#         emb = scale * (timestep * freq); flip_sin_to_cos=True => cat([cos,sin]).
#       COS-first, denom = half (downscale_freq_shift=0). Output [B,256].
#   time_embed = timestep_embedder(timestep_proj)
#       = TimestepEmbedding(256->1024): Linear(256,1024) -> SiLU -> Linear(1024,1024).
#       Output [B,1024].
#   caption_embed = caption_embedder(instruction_hidden_states)
#       = Sequential(RMSNorm(4096, eps=1e-5), Linear(4096,3360, bias=True)).
#       Input [B,L,4096] -> output [B,L,3360].
#
# PARITY NOTE on the sinusoid (settled by reading diffusers get_timestep_embedding
# line-by-line): serenitymojo `timestep_embedding` is COS-first with denom = dim/2
# and computes `cos/sin(t * freq)`. diffusers computes `cos/sin(scale * t * freq)`.
# Since `scale * t * freq == (scale*t) * freq`, pre-scaling the timestep by 1000
# BEFORE the embedding is mathematically IDENTICAL. We pass `timestep*1000` into
# the reused `t_embedder`/`timestep_embedding`. downscale_freq_shift=0 => the
# serenitymojo denom (half) matches exactly. (See report.)
#
# REUSES foundation ops: ops/linear.linear, ops/norm.rms_norm,
# ops/embeddings.{timestep_embedding, t_embedder}. Weights loaded via
# io/sharded.ShardedSafeTensors + Tensor.from_view (BF16 preserved) — same pattern
# as models/dit/ideogram4_dit.mojo. Nothing reimplemented.
#
# Mojo 1.0.0b1, NVIDIA GPU, inference-only, GPU-only.

from std.gpu.host import DeviceContext
from std.gpu import global_idx
from std.memory import ArcPointer
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import rms_norm, layer_norm_no_affine
from serenitymojo.ops.embeddings import timestep_embedding, t_embedder
from serenitymojo.ops.tensor_algebra import (
    mul_scalar,
    mul,
    add,
    add_scalar,
    reshape,
    slice,
    concat,
)
from serenitymojo.ops.rope_tables import build_multiaxis_rope_tables
from serenitymojo.ops.rope import rope_interleaved
from serenitymojo.ops.activations import silu, swiglu
from serenitymojo.ops.unary import tanh_op
from serenitymojo.ops.attention import sdpa_nomask
from serenitymojo.ops.gqa_backward import repeat_kv_f32


# Boogu-Image-0.1-Base embedder config constants.
comptime BOOGU_HIDDEN_SIZE = 3360
comptime BOOGU_PATCH_SIZE = 2
comptime BOOGU_IN_CHANNELS = 16
comptime BOOGU_X_EMBED_IN = 64           # patch*patch*in_channels = 2*2*16
comptime BOOGU_INSTRUCTION_FEAT_DIM = 4096
comptime BOOGU_FREQ_EMBED_SIZE = 256     # frequency_embedding_size
comptime BOOGU_TIME_EMBED_DIM = 1024     # min(hidden_size, 1024)
comptime BOOGU_TIMESTEP_SCALE = Float32(1000.0)
comptime BOOGU_NORM_EPS = Float32(1.0e-5)
comptime BOOGU_TIME_MAX_PERIOD = Float32(10000.0)


# ── weight load helper (BF16 preserved, mirrors ideogram4 load_w_bf16) ───────
def _load_w(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    """Load a BF16 checkpoint tensor preserving dtype (Linear weight/bias, RMSNorm gamma)."""
    return Tensor.from_view(st.tensor_view(name), ctx)


# ── BooguEmbedders: holds the x_embedder + time_caption_embed weights ────────
# Tensor is Movable-not-Copyable, so each weight is owned directly and forward
# methods return fresh tensors (with `^`). Weights are loaded once at construction.
@fieldwise_init
struct BooguEmbedders(Movable):
    # x_embedder: nn.Linear(64 -> 3360, bias=True).
    var x_embedder_weight: Tensor          # [3360, 64]
    var x_embedder_bias: Tensor            # [3360]
    # time_caption_embed.timestep_embedder: TimestepEmbedding(256 -> 1024).
    var ts_linear_1_weight: Tensor         # [1024, 256]
    var ts_linear_1_bias: Tensor           # [1024]
    var ts_linear_2_weight: Tensor         # [1024, 1024]
    var ts_linear_2_bias: Tensor           # [1024]
    # time_caption_embed.caption_embedder: Sequential(RMSNorm(4096), Linear(4096->3360)).
    var caption_norm_weight: Tensor        # [4096]  (RMSNorm gamma)
    var caption_linear_weight: Tensor      # [3360, 4096]
    var caption_linear_bias: Tensor        # [3360]

    @staticmethod
    def load(transformer_dir: String, ctx: DeviceContext) raises -> BooguEmbedders:
        """Load all embedder weights from the Boogu transformer dir via ShardedSafeTensors.

        transformer_dir: path to .../Boogu-Image-0.1-Base/transformer (3-shard
        safetensors with diffusion_pytorch_model.safetensors.index.json).
        """
        var st = ShardedSafeTensors.open(transformer_dir)
        var xw = _load_w(st, "x_embedder.weight", ctx)
        var xb = _load_w(st, "x_embedder.bias", ctx)
        var t1w = _load_w(st, "time_caption_embed.timestep_embedder.linear_1.weight", ctx)
        var t1b = _load_w(st, "time_caption_embed.timestep_embedder.linear_1.bias", ctx)
        var t2w = _load_w(st, "time_caption_embed.timestep_embedder.linear_2.weight", ctx)
        var t2b = _load_w(st, "time_caption_embed.timestep_embedder.linear_2.bias", ctx)
        var cnw = _load_w(st, "time_caption_embed.caption_embedder.0.weight", ctx)
        var clw = _load_w(st, "time_caption_embed.caption_embedder.1.weight", ctx)
        var clb = _load_w(st, "time_caption_embed.caption_embedder.1.bias", ctx)
        return BooguEmbedders(
            xw^, xb^, t1w^, t1b^, t2w^, t2b^, cnw^, clw^, clb^
        )

    # ── x_embed: already-patchified tokens [B,L_img,64] -> [B,L_img,3360] ────
    # transformer_boogu.py:840 nn.Linear(64 -> 3360, bias=True).
    def x_embed(self, tokens: Tensor, ctx: DeviceContext) raises -> Tensor:
        """x_embedder Linear over already-patchified tokens. tokens: [B,L_img,64]."""
        return linear(
            tokens,
            self.x_embedder_weight,
            Optional[Tensor](self.x_embedder_bias.clone(ctx)),
            ctx,
        )

    # ── time_caption_embed: (time_embed [B,1024], caption_embed [B,L,3360]) ──
    # block_lumina2.py:210-219 forward.
    def time_caption_embed(
        self,
        timestep: Tensor,        # [B] scalar timesteps, F32
        instruction_feats: Tensor,  # [B,L,4096] BF16
        ctx: DeviceContext,
    ) raises -> Tuple[Tensor, Tensor]:
        """Returns (time_embed [B,1024], caption_embed [B,L,3360]).

        time path == diffusers Timesteps(256, flip_sin_to_cos=True,
        downscale_freq_shift=0, scale=1000) -> TimestepEmbedding(256->1024).
        Implemented by pre-scaling the timestep by 1000 (mathematically identical
        to diffusers' post-freq scale) then reusing the COS-first
        ops/embeddings.t_embedder (timestep_embedding -> Linear -> SiLU -> Linear).
        caption path == RMSNorm(4096, eps=1e-5) -> Linear(4096->3360, bias=True).
        """
        # scale=1000 applied to the timestep BEFORE the sinusoid (== diffusers).
        var t_scaled = mul_scalar(timestep, BOOGU_TIMESTEP_SCALE, ctx)
        var time_embed = t_embedder(
            t_scaled,
            BOOGU_FREQ_EMBED_SIZE,
            self.ts_linear_1_weight,
            Optional[Tensor](self.ts_linear_1_bias.clone(ctx)),
            self.ts_linear_2_weight,
            Optional[Tensor](self.ts_linear_2_bias.clone(ctx)),
            ctx,
            BOOGU_TIME_MAX_PERIOD,
        )  # [B, 1024]

        # caption_embedder: RMSNorm(eps=1e-5) -> Linear(4096->3360, bias=True).
        var cn = rms_norm(
            instruction_feats, self.caption_norm_weight, BOOGU_NORM_EPS, ctx
        )
        var caption_embed = linear(
            cn,
            self.caption_linear_weight,
            Optional[Tensor](self.caption_linear_bias.clone(ctx)),
            ctx,
        )  # [B, L, 3360]

        return (time_embed^, caption_embed^)


# ── Chunk C2: 3-axis RoPE cos/sin tables (T2I, no ref images, batch=1) ────────
# Mirrors BooguImageDoubleStreamRotaryPosEmbed.forward (rope.py:266-448) for the
# no-ref T2I batch=1 path, read line-by-line (NOT inferred):
#   /home/alex/Boogu-Image/boogu/models/transformers/rope.py
#
# Config (Boogu-Image-0.1-Base): axes_dim=[40,40,40] (sum=120=head_dim),
# axes_lens=[2048,1664,1664], theta=10000, patch_size=2.  half = 120/2 = 60.
#
# POSITION-ID rule (rope.py forward, no-ref branch; rope.py:304-361):
#   - caption token t in [0,cap_len): position_ids[t] = (t,t,t)   (rope.py:304-306
#     `repeat(arange(cap_seq_len), "l -> l 3")` => same index on all 3 axes).
#   - image token k in [0,img_len): h = k // W_tok, w = k % W_tok (row-major,
#     rope.py:347-356 row_ids/col_ids flattened H-major); position_ids =
#     (pe_shift, h, w) where pe_shift = cap_len (rope.py:308,359-361). axis0 is the
#     constant caption-length shift, axis1 = row id h, axis2 = col id w.
#   joint seq_len = cap_len + img_len (no ref => sum(ref_img_len)=0).
#
# The per-axis rope freqs are diffusers get_1d_rotary_pos_embed(dim=40, theta=10000)
# (rope.py:51 get_freqs_cis -> get_1d_rotary_pos_embed), then per-axis gathered by
# the integer position and concatenated axis0|axis1|axis2 (rope.py:250-264
# _get_freqs_cis: `torch.cat(result, dim=-1)`). That is EXACTLY what
# ops/rope_tables.build_multiaxis_rope_tables produces: for axis a, half_a=20,
# angle = pos * theta^(-i/20), i in [0,20); cos/sin blocks concatenated in axis
# order => width 60. So this function ONLY builds the token-major positions array
# [seq*3] and calls the foundation op. NO rope freq/trig math here.
#
# WHY cos/sin (not complex): Boogu's apply_rotary_emb(use_real=False) does
# view_as_complex(x reshape (...,60,2)) * freqs_cis = INTERLEAVED adjacent-pair
# rope (embeddings.py:126-133). serenitymojo applies it via ops/rope.rope_interleaved
# using cos/sin tables [seq,60]. The oracle dumps freqs_cis (complex e^{iθ}):
# real = cos θ, imag = sin θ. So the produced cos table == oracle real part,
# sin table == oracle imag part. (This function does NOT apply rope — only builds
# the tables; the orchestrator gates cos↔real, sin↔imag.)

comptime BOOGU_ROPE_THETA = Float32(10000.0)
comptime BOOGU_ROPE_AXIS_DIM = 40        # per-axis full rotary dim (x3 axes)
comptime BOOGU_ROPE_HALF = 60            # sum(axes_dim)/2 = (40+40+40)/2


def build_boogu_rope_tables(
    cap_len: Int, h_tok: Int, w_tok: Int, ctx: DeviceContext
) raises -> Tuple[Tensor, Tensor]:
    """Build the joint 3-axis RoPE cos/sin tables for the Boogu T2I no-ref case.

    Args:
        cap_len: number of caption tokens (l_effective_cap_len, batch=1).
        h_tok:   image token rows  = H_lat / patch_size (= H_lat // 2).
        w_tok:   image token cols  = W_lat / patch_size (= W_lat // 2).
        ctx:     device context.

    Returns:
        (cos, sin), each `[seq_len, 60]` F32, where seq_len = cap_len + h_tok*w_tok.
        Rows [0, cap_len) are the CAPTION tables; rows [cap_len, seq_len) are the
        IMAGE tables (combined_img == img since no ref). Feed straight into
        ops/rope.rope_interleaved with q/k of shape [..., head_dim=120].

    Slice offsets (for the segment-sliced returns rope.py produces):
        cap tables = rows [0, cap_len)
        img tables = rows [cap_len, seq_len)   (== combined_img, no ref)
    """
    var img_len = h_tok * w_tok
    var seq_len = cap_len + img_len

    # Token-major positions [seq_len*3] F32: index t*3 + a = token t pos on axis a.
    var positions = List[Float32]()
    # caption tokens t in [0,cap_len): (t,t,t).  (rope.py:304-306)
    for t in range(cap_len):
        var ft = Float32(t)
        positions.append(ft)  # axis0
        positions.append(ft)  # axis1
        positions.append(ft)  # axis2
    # image tokens k in [0,img_len): (cap_len, h, w), h=k//w_tok, w=k%w_tok.
    # (rope.py:308 pe_shift=cap_len; 347-356 row-major H,W flatten; 359-361 assign)
    var pe_shift = Float32(cap_len)
    for k in range(img_len):
        var h = k // w_tok
        var w = k % w_tok
        positions.append(pe_shift)     # axis0 = constant caption-length shift
        positions.append(Float32(h))   # axis1 = row id
        positions.append(Float32(w))   # axis2 = col id

    var pos_t = Tensor.from_host(positions, [seq_len * 3], STDtype.F32, ctx)

    # REUSE the foundation op (no rope freq/trig math reimplemented here). axes_dim
    # per axis is the FULL rotary dim (40); the op halves it internally to 20.
    # Forward its (cos [seq,60], sin [seq,60]) tuple directly.
    return build_multiaxis_rope_tables(
        pos_t,
        [BOOGU_ROPE_AXIS_DIM, BOOGU_ROPE_AXIS_DIM, BOOGU_ROPE_AXIS_DIM],
        BOOGU_ROPE_THETA,
        ctx,
        STDtype.F32,
    )


# ── Chunk C3: single BooguImageTransformerBlock (single/double-stream block) ──
# Mirrors `BooguImageTransformerBlock.forward` (NON-taylorseer branch), read
# line-by-line (NOT inferred):
#   /home/alex/Boogu-Image/boogu/models/transformers/transformer_boogu.py:266-373
#     modulation=True  (339-360): LuminaRMSNormZero -> attn -> tanh-gated residual
#                                  -> SwiGLU FF -> tanh-gated residual.
#     modulation=False (361-371): plain RMSNorm pre/post, NO temb, NO gates.
#   /home/alex/Boogu-Image/boogu/models/attention_processor.py:1163-1275
#     BooguImageAttnProcessor.__call__ (the self-attn used here).
#   /home/alex/Boogu-Image/boogu/models/transformers/block_lumina2.py
#     LuminaRMSNormZero (39-71), LuminaFeedForward (125-174), components.swiglu.
#
# Config (Boogu-Image-0.1-Base single_stream_layers): hidden=3360, heads=28,
# kv_heads=7, head_dim=120 (GQA repeat n_rep=4), norm_eps=1e-5,
# SwiGLU inner=13568, attn.scale = 1/sqrt(120).
#
# ── LuminaRMSNormZero(hidden, temb), modulation=True (block_lumina2.py:68-71) ──
#   emb = norm1.linear(silu(temb))                       [B,1024] -> [B,4*3360]
#   (scale_msa, gate_msa, scale_mlp, gate_mlp) = emb.chunk(4, dim=1) IN ORDER
#   x = norm1.norm(hidden) * (1 + scale_msa[:,None,:])   RMSNorm(eps=1e-5)
#   returns (x, gate_msa, scale_mlp, gate_mlp).
#
# ── Self-attn (BooguImageAttnProcessor, attention_processor.py:1186-1273) ─────
#   q=to_q(x) [.,.,3360] -> view[B,seq,28,120]; k=to_k(x)/v=to_v(x) -> [B,seq,7,120].
#   per-head qk RMSNorm over head_dim (norm_q/norm_k, eps=1e-5).
#   rope (apply_rotary_emb use_real=False, interleaved adjacent-pair) on q AND k,
#     freqs_cis [seq,60] (cos=real, sin=imag) broadcast over heads.
#   repeat_interleave kv n_rep=4 -> k,v [B,seq,28,120].  (line 1260-1261)
#   sdpa scale = attn.scale = 1/sqrt(120); base_sequence_length None; mask all-True
#     (b=1, full attention) => sdpa_nomask. reshape [B,seq,3360];
#   out = to_out[0](x) (NO bias).  (to_out[1] dropout = identity at inference.)
#
# ── Block residuals, modulation=True (transformer_boogu.py:352-360) ───────────
#   hidden += tanh(gate_msa)[:,None,:] * norm2(attn_out)               (norm2=RMSNorm)
#   mlp_in  = ffn_norm1(hidden) * (1 + scale_mlp[:,None,:])
#   mlp_out = feed_forward(mlp_in)                                     (SwiGLU)
#   hidden += tanh(gate_mlp)[:,None,:] * ffn_norm2(mlp_out)
#
# ── SwiGLU LuminaFeedForward (block_lumina2.py:171-174 + components.swiglu) ────
#   h1=linear_1(x) [3360->13568], h3=linear_3(x), swiglu = silu(h1)*h3 (NO bias),
#   out = linear_2(swiglu) [13568->3360].
#
# ── modulation=False (context_refiner, transformer_boogu.py:362-371) ──────────
#   norm_hidden = norm1(hidden)                                        (plain RMSNorm)
#   attn_out    = attn(norm_hidden)
#   hidden     += norm2(attn_out)
#   mlp_out     = feed_forward(ffn_norm1(hidden))
#   hidden     += ffn_norm2(mlp_out)
#
# REUSES foundation ops ONLY: ops/linear.linear, ops/norm.rms_norm,
# ops/activations.{silu, swiglu}, ops/unary.tanh_op,
# ops/tensor_algebra.{mul, add, add_scalar, slice, reshape},
# ops/rope.rope_interleaved, ops/gqa_backward.repeat_kv_f32,
# ops/attention.sdpa_nomask. Nothing reimplemented.

comptime BOOGU_HEADS = 28
comptime BOOGU_KV_HEADS = 7
comptime BOOGU_HEAD_DIM = 120
comptime BOOGU_GQA_NREP = 4              # heads / kv_heads = 28 / 7
comptime BOOGU_FFN_INNER = 13568         # SwiGLU intermediate
comptime BOOGU_ROPE_PAIRS = 60           # head_dim / 2 (rope table width)


# Expand the [seq,60] joint rope table to [seq*heads, 60] so rope_interleaved
# (which flattens leading dims of q[1,seq,H,Dh] to rows = seq*H, row r = s*H+h)
# reads cos_table[s] for every head. repeat_kv_f32 broadcasts a single kv "head"
# (the table) across n_rep=heads dst heads: dst row s*heads+h reads src head 0.
def _expand_rope_table_per_head(
    table: Tensor, seq: Int, heads: Int, ctx: DeviceContext
) raises -> Tensor:
    """[seq,60] F32 -> [seq*heads,60] F32 (each seq row repeated `heads` times)."""
    var t4 = reshape(table, [1, seq, 1, BOOGU_ROPE_PAIRS], ctx)
    var expanded = repeat_kv_f32(t4, seq, 1, heads, BOOGU_ROPE_PAIRS, ctx)
    return reshape(expanded, [seq * heads, BOOGU_ROPE_PAIRS], ctx)


# ── BooguBlock: one transformer block (single_stream / noise_refiner /
#    context_refiner) loaded by key prefix. Tensor is Movable-not-Copyable so each
#    weight is owned directly; forward returns a fresh tensor. ──────────────────
@fieldwise_init
struct BooguBlock(Movable):
    var modulation: Bool
    # AdaLN (modulation=True only): norm1 = LuminaRMSNormZero.
    #   norm1.linear: Linear(1024 -> 4*3360=13440, bias=True).
    #   norm1.norm:   RMSNorm(3360) gamma.
    var norm1_lin_w: Tensor            # [13440, 1024]  (modulation=True)
    var norm1_lin_b: Tensor            # [13440]        (modulation=True)
    var norm1_norm_w: Tensor           # [3360]         (modulation=True: norm1.norm.weight;
                                       #  modulation=False: norm1.weight plain RMSNorm)
    # Plain RMSNorm gammas shared by both modes.
    var norm2_w: Tensor                # [3360]
    var ffn_norm1_w: Tensor            # [3360]
    var ffn_norm2_w: Tensor            # [3360]
    # Attention.
    var to_q_w: Tensor                 # [3360, 3360]
    var to_k_w: Tensor                 # [840, 3360]
    var to_v_w: Tensor                 # [840, 3360]
    var to_out_w: Tensor               # [3360, 3360]  (to_out.0.weight, NO bias)
    var norm_q_w: Tensor               # [120]
    var norm_k_w: Tensor               # [120]
    # SwiGLU feed-forward.
    var ff_w1: Tensor                  # [13568, 3360]  (linear_1, NO bias)
    var ff_w2: Tensor                  # [3360, 13568]  (linear_2, NO bias)
    var ff_w3: Tensor                  # [13568, 3360]  (linear_3, NO bias)

    @staticmethod
    def load(
        st: ShardedSafeTensors, prefix: String, modulation: Bool, ctx: DeviceContext
    ) raises -> BooguBlock:
        """Load one block's weights by key prefix (e.g. "single_stream_layers.0").

        modulation=True  -> norm1 is LuminaRMSNormZero (norm1.linear.* +
                            norm1.norm.weight). modulation=False -> norm1 is a
                            plain RMSNorm (norm1.weight); the AdaLN linear is
                            absent, so for that case the loader expects the
                            caller to use a modulation=False block (probe targets
                            a modulation=True block).
        """
        var n1_lin_w: Tensor
        var n1_lin_b: Tensor
        var n1_norm_w: Tensor
        if modulation:
            n1_lin_w = _load_w(st, prefix + ".norm1.linear.weight", ctx)
            n1_lin_b = _load_w(st, prefix + ".norm1.linear.bias", ctx)
            n1_norm_w = _load_w(st, prefix + ".norm1.norm.weight", ctx)
        else:
            # Plain RMSNorm: reuse the same field for norm1.weight; the linear
            # fields are present but unused (filled with the plain gamma to keep
            # the struct total — they are never read when modulation=False).
            n1_norm_w = _load_w(st, prefix + ".norm1.weight", ctx)
            n1_lin_w = n1_norm_w.clone(ctx)
            n1_lin_b = n1_norm_w.clone(ctx)
        return BooguBlock(
            modulation,
            n1_lin_w^,
            n1_lin_b^,
            n1_norm_w^,
            _load_w(st, prefix + ".norm2.weight", ctx),
            _load_w(st, prefix + ".ffn_norm1.weight", ctx),
            _load_w(st, prefix + ".ffn_norm2.weight", ctx),
            _load_w(st, prefix + ".attn.to_q.weight", ctx),
            _load_w(st, prefix + ".attn.to_k.weight", ctx),
            _load_w(st, prefix + ".attn.to_v.weight", ctx),
            _load_w(st, prefix + ".attn.to_out.0.weight", ctx),
            _load_w(st, prefix + ".attn.norm_q.weight", ctx),
            _load_w(st, prefix + ".attn.norm_k.weight", ctx),
            _load_w(st, prefix + ".feed_forward.linear_1.weight", ctx),
            _load_w(st, prefix + ".feed_forward.linear_2.weight", ctx),
            _load_w(st, prefix + ".feed_forward.linear_3.weight", ctx),
        )

    # ── self-attention (BooguImageAttnProcessor, full attention, b=1) ─────────
    # x: [1,seq,3360]; cos/sin: [seq,60] F32 joint rope tables. Returns [1,seq,3360].
    def _attn[
        S: Int
    ](self, x: Tensor, cos: Tensor, sin: Tensor, ctx: DeviceContext) raises -> Tensor:
        # q/k/v projections (no bias).
        var q = linear(x, self.to_q_w, None, ctx)              # [1,seq,3360]
        var k = linear(x, self.to_k_w, None, ctx)              # [1,seq,840]
        var v = linear(x, self.to_v_w, None, ctx)              # [1,seq,840]
        # view into heads.
        q = reshape(q, [1, S, BOOGU_HEADS, BOOGU_HEAD_DIM], ctx)
        k = reshape(k, [1, S, BOOGU_KV_HEADS, BOOGU_HEAD_DIM], ctx)
        v = reshape(v, [1, S, BOOGU_KV_HEADS, BOOGU_HEAD_DIM], ctx)
        # per-head qk RMSNorm over head_dim (eps=1e-5).
        q = rms_norm(q, self.norm_q_w, BOOGU_NORM_EPS, ctx)
        k = rms_norm(k, self.norm_k_w, BOOGU_NORM_EPS, ctx)
        # interleaved rope on q AND k. rope_interleaved flattens leading dims to
        # rows; q is [1,seq,heads,Dh] (rows=seq*heads), k is [1,seq,kv_heads,Dh]
        # (rows=seq*kv_heads). Expand the joint [seq,60] table per head-count so
        # row s*H+h reads table[s].
        var cos_q = _expand_rope_table_per_head(cos, S, BOOGU_HEADS, ctx)
        var sin_q = _expand_rope_table_per_head(sin, S, BOOGU_HEADS, ctx)
        var cos_k = _expand_rope_table_per_head(cos, S, BOOGU_KV_HEADS, ctx)
        var sin_k = _expand_rope_table_per_head(sin, S, BOOGU_KV_HEADS, ctx)
        q = rope_interleaved(q, cos_q, sin_q, ctx)
        k = rope_interleaved(k, cos_k, sin_k, ctx)
        # GQA repeat kv n_rep=4 -> [1,seq,28,120] (repeat_interleave over heads).
        k = repeat_kv_f32(k, S, BOOGU_KV_HEADS, BOOGU_GQA_NREP, BOOGU_HEAD_DIM, ctx)
        v = repeat_kv_f32(v, S, BOOGU_KV_HEADS, BOOGU_GQA_NREP, BOOGU_HEAD_DIM, ctx)
        # full attention (mask all-True, b=1) => sdpa_nomask. scale = 1/sqrt(120).
        var scale = Float32(1.0 / (Float32(BOOGU_HEAD_DIM) ** 0.5))
        var attn = sdpa_nomask[1, S, BOOGU_HEADS, BOOGU_HEAD_DIM](q, k, v, scale, ctx)
        var merged = reshape(attn, [1, S, BOOGU_HIDDEN_SIZE], ctx)
        return linear(merged, self.to_out_w, None, ctx)        # to_out.0 (no bias)

    # ── SwiGLU feed-forward (LuminaFeedForward) ───────────────────────────────
    def _feed_forward(self, x: Tensor, ctx: DeviceContext) raises -> Tensor:
        var h1 = linear(x, self.ff_w1, None, ctx)              # linear_1
        var h3 = linear(x, self.ff_w3, None, ctx)              # linear_3
        var act = swiglu(h1, h3, ctx)                          # silu(h1)*h3
        return linear(act, self.ff_w2, None, ctx)              # linear_2

    # ── forward: temb ignored when modulation=False. ─────────────────────────
    # hidden: [1,seq,3360]; temb: [1,1024]; cos/sin: [seq,60] F32 joint tables.
    def forward[
        S: Int
    ](
        self,
        hidden: Tensor,
        temb: Tensor,
        cos: Tensor,
        sin: Tensor,
        ctx: DeviceContext,
    ) raises -> Tensor:
        if self.modulation:
            # ── LuminaRMSNormZero(hidden, temb) ──────────────────────────────
            var emb = linear(
                silu(temb, ctx),
                self.norm1_lin_w,
                Optional[Tensor](self.norm1_lin_b.clone(ctx)),
                ctx,
            )  # [1, 4*3360]
            # chunk(4, dim=1): (scale_msa, gate_msa, scale_mlp, gate_mlp).
            var scale_msa = slice(emb, 1, 0 * BOOGU_HIDDEN_SIZE, BOOGU_HIDDEN_SIZE, ctx)
            var gate_msa = slice(emb, 1, 1 * BOOGU_HIDDEN_SIZE, BOOGU_HIDDEN_SIZE, ctx)
            var scale_mlp = slice(emb, 1, 2 * BOOGU_HIDDEN_SIZE, BOOGU_HIDDEN_SIZE, ctx)
            var gate_mlp = slice(emb, 1, 3 * BOOGU_HIDDEN_SIZE, BOOGU_HIDDEN_SIZE, ctx)
            # broadcast as [1,1,3360] over the seq dim.
            var scale_msa_b = reshape(scale_msa, [1, 1, BOOGU_HIDDEN_SIZE], ctx)
            var gate_msa_b = reshape(
                tanh_op(gate_msa, ctx), [1, 1, BOOGU_HIDDEN_SIZE], ctx
            )
            var scale_mlp_b = reshape(scale_mlp, [1, 1, BOOGU_HIDDEN_SIZE], ctx)
            var gate_mlp_b = reshape(
                tanh_op(gate_mlp, ctx), [1, 1, BOOGU_HIDDEN_SIZE], ctx
            )
            # norm_hidden = norm1.norm(hidden) * (1 + scale_msa).
            var nh = rms_norm(hidden, self.norm1_norm_w, BOOGU_NORM_EPS, ctx)
            var norm_hidden = mul(nh, add_scalar(scale_msa_b, Float32(1.0), ctx), ctx)
            # attn_out = attn(norm_hidden, norm_hidden).
            var attn_out = self._attn[S](norm_hidden, cos, sin, ctx)
            # hidden = hidden + tanh(gate_msa) * norm2(attn_out).
            var n2 = rms_norm(attn_out, self.norm2_w, BOOGU_NORM_EPS, ctx)
            var hidden1 = add(hidden, mul(gate_msa_b, n2, ctx), ctx)
            # mlp_in = ffn_norm1(hidden) * (1 + scale_mlp).
            var fn1 = rms_norm(hidden1, self.ffn_norm1_w, BOOGU_NORM_EPS, ctx)
            var mlp_in = mul(fn1, add_scalar(scale_mlp_b, Float32(1.0), ctx), ctx)
            var mlp_out = self._feed_forward(mlp_in, ctx)
            # hidden = hidden + tanh(gate_mlp) * ffn_norm2(mlp_out).
            var fn2 = rms_norm(mlp_out, self.ffn_norm2_w, BOOGU_NORM_EPS, ctx)
            return add(hidden1, mul(gate_mlp_b, fn2, ctx), ctx)
        else:
            # context_refiner: plain RMSNorm pre/post, NO temb, NO gates.
            var norm_hidden = rms_norm(hidden, self.norm1_norm_w, BOOGU_NORM_EPS, ctx)
            var attn_out = self._attn[S](norm_hidden, cos, sin, ctx)
            var n2 = rms_norm(attn_out, self.norm2_w, BOOGU_NORM_EPS, ctx)
            var hidden1 = add(hidden, n2, ctx)
            var fn1 = rms_norm(hidden1, self.ffn_norm1_w, BOOGU_NORM_EPS, ctx)
            var mlp_out = self._feed_forward(fn1, ctx)
            var fn2 = rms_norm(mlp_out, self.ffn_norm2_w, BOOGU_NORM_EPS, ctx)
            return add(hidden1, fn2, ctx)


# ── Chunk C4: single BooguImageDoubleStreamTransformerBlock (modulation=True) ──
# Mirrors `BooguImageDoubleStreamTransformerBlock.forward` modulation=True branch,
# read line-by-line (NOT inferred):
#   /home/alex/Boogu-Image/boogu/models/transformers/transformer_boogu.py:558-683
#     (596-683: modulate both streams -> joint attn -> split -> img self-attn ->
#      tanh-gated residuals -> SwiGLU MLPs).
#   /home/alex/Boogu-Image/boogu/models/attention_processor.py:505-877
#     BooguImageDoubleStreamSelfAttnProcessor (separate per-stream q/k/v projections,
#     concat instruct-first, qk-norm, rope, repeat_kv, sdpa, split, per-stream out
#     projections, re-concat, to_out.0).
#   /home/alex/Boogu-Image/boogu/models/attention_processor.py:1163-1275
#     BooguImageAttnProcessor.__call__ (the img self-attn — structurally identical
#     to BooguBlock._attn above; reused as a method here).
#   /home/alex/Boogu-Image/boogu/models/transformers/block_lumina2.py:39-71
#     LuminaRMSNormZero (chunk(4): scale_msa, gate_msa, scale_mlp, gate_mlp;
#     out = norm(x)*(1+scale_msa)).
#
# T2I no-ref, batch=1: L_instruct=16, L_img=256, joint seq=272, hidden=3360,
# heads=28, kv_heads=7, head_dim=120, scale=1/sqrt(120), SwiGLU inner=13568.
#
# ── Modulations (5 LuminaRMSNormZero, each its own AdaLN linear) ───────────────
#   img_norm1(img,temb)      -> (img_norm1_out, img_gate_msa, img_scale_mlp, img_gate_mlp)
#   img_norm2(img,temb)      -> (img_norm2_out, img_shift_mlp, _, _)   (out chunk0;
#                                shift = chunk1, i.e. the "gate_msa" slot reused)
#   img_norm3(img,temb)      -> (img_norm3_out, img_gate_self, _, _)   (gate_self = chunk1)
#   instruct_norm1(ins,temb) -> (instruct_norm1_out, instruct_gate_msa,
#                                instruct_scale_mlp, instruct_gate_mlp)
#   instruct_norm2(ins,temb) -> (instruct_norm2_out, instruct_shift_mlp, _, _)
#   LuminaRMSNormZero(x,temb): emb=linear(silu(temb)).chunk(4) =
#     (scale_msa, gate_msa, scale_mlp, gate_mlp); out = rms_norm(x)*(1+scale_msa).
#   So chunk0 = scale_msa (multiplies the norm), chunk1 = gate_msa. For norm2 the
#   downstream code reads its chunk1 as the MLP SHIFT (img_shift_mlp/instruct_shift_mlp);
#   for norm3 chunk1 is the self-attn gate (img_gate_self). chunk0 is ALWAYS the
#   in-norm scale_msa (already folded into *Out).
#
# ── Joint attention (the NEW piece) ───────────────────────────────────────────
#   img_q  = linear(img_norm1_out, img_to_q)    [1,256,3360]; img_k/v [1,256,840]
#   ins_q  = linear(instruct_norm1_out, instruct_to_q) [1,16,3360]; ins_k/v [1,16,840]
#   query  = cat([ins_q, img_q], dim=1) [1,272,3360]  (INSTRUCT-FIRST)
#   key    = cat([ins_k, img_k]) [1,272,840]; value = cat([ins_v, img_v]) [1,272,840]
#   reshape q[1,272,28,120] k/v[1,272,7,120]; qk RMSNorm(norm_q/norm_k, eps 1e-5);
#   rope_interleaved q&k with JOINT rope (272 rows); repeat_kv x4; sdpa scale 1/sqrt(120);
#   reshape[1,272,3360]; split instruct-first: ins_h=[:, :16], img_h=[:, 16:272];
#   ins_proj=linear(ins_h, instruct_out); img_proj=linear(img_h, img_out);
#   merged=cat([ins_proj, img_proj]) [1,272,3360]; joint_out=linear(merged, to_out.0) (no bias).
#
# ── Img self-attn ─ structurally identical to BooguBlock._attn, on img_norm3_out
#   [1,256,3360] with img_self_attn.* weights and the COMBINED-IMG rope = joint rope
#   rows [cap_len:seq] (rope.py:427-437; ref_img_len=0 => combined_img == img segment).
#
# ── Residuals (transformer_boogu.py:651-683; tanh on GATES, (1+) on SCALES) ─────
#   img += tanh(img_gate_msa)  * img_attn_norm(img_attn_out)        (img_attn_out=joint[:,16:272])
#   img += tanh(img_gate_self) * img_self_attn_norm(img_self_attn_out)
#   img_mlp_in = (1 + img_scale_mlp) * img_norm2_out + img_shift_mlp
#   img += tanh(img_gate_mlp)  * img_ffn_norm2(img_feed_forward(img_ffn_norm1(img_mlp_in)))
#   ins += tanh(instruct_gate_msa) * instruct_attn_norm(instruct_attn_out)  (joint[:, :16])
#   ins_mlp_in = (1 + instruct_scale_mlp) * instruct_norm2_out + instruct_shift_mlp
#   ins += tanh(instruct_gate_mlp) * instruct_ffn_norm2(instruct_feed_forward(instruct_ffn_norm1(ins_mlp_in)))
#
# REUSES the C3 helpers (_expand_rope_table_per_head, build_boogu_rope_tables) and
# foundation ops ONLY (linear, rms_norm, silu, swiglu, tanh_op, mul/add/add_scalar/
# slice/reshape/concat, rope_interleaved, repeat_kv_f32, sdpa_nomask). Nothing
# reimplemented.

comptime BOOGU_QKV_KV_DIM = 840          # head_dim * kv_heads = 120 * 7


# Apply one LuminaRMSNormZero: returns (out = rms_norm(x)*(1+scale_msa), gate/shift
# = chunk1, scale_mlp = chunk2, gate_mlp = chunk3). The AdaLN linear is
# Linear(1024 -> 4*3360, bias). x: [1,S,3360], temb: [1,1024].
# chunk order (block_lumina2.py:69) = (scale_msa, gate_msa, scale_mlp, gate_mlp).
def _lumina_rms_norm_zero(
    x: Tensor,
    temb: Tensor,
    lin_w: Tensor,
    lin_b: Tensor,
    norm_w: Tensor,
    ctx: DeviceContext,
) raises -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Returns (out, chunk1, chunk2, chunk3) where out = rms_norm(x)*(1+scale_msa).

    chunk0 = scale_msa (folded into out). chunk1/2/3 are the raw gate_msa /
    scale_mlp / gate_mlp slices [1,1,3360] (un-tanh'd, un-(1+)'d — callers apply
    tanh on gates and (1+) on scales). Mirrors LuminaRMSNormZero.forward.
    """
    var emb = linear(
        silu(temb, ctx), lin_w, Optional[Tensor](lin_b.clone(ctx)), ctx
    )  # [1, 4*3360]
    var scale_msa = slice(emb, 1, 0 * BOOGU_HIDDEN_SIZE, BOOGU_HIDDEN_SIZE, ctx)
    var chunk1 = slice(emb, 1, 1 * BOOGU_HIDDEN_SIZE, BOOGU_HIDDEN_SIZE, ctx)
    var chunk2 = slice(emb, 1, 2 * BOOGU_HIDDEN_SIZE, BOOGU_HIDDEN_SIZE, ctx)
    var chunk3 = slice(emb, 1, 3 * BOOGU_HIDDEN_SIZE, BOOGU_HIDDEN_SIZE, ctx)
    var scale_msa_b = reshape(scale_msa, [1, 1, BOOGU_HIDDEN_SIZE], ctx)
    var nh = rms_norm(x, norm_w, BOOGU_NORM_EPS, ctx)
    var out = mul(nh, add_scalar(scale_msa_b, Float32(1.0), ctx), ctx)
    var c1 = reshape(chunk1, [1, 1, BOOGU_HIDDEN_SIZE], ctx)
    var c2 = reshape(chunk2, [1, 1, BOOGU_HIDDEN_SIZE], ctx)
    var c3 = reshape(chunk3, [1, 1, BOOGU_HIDDEN_SIZE], ctx)
    return (out^, c1^, c2^, c3^)


@fieldwise_init
struct BooguDoubleStreamBlock(Movable):
    # ── 5 LuminaRMSNormZero AdaLN blocks (each linear 1024->13440 + norm gamma) ──
    var img_n1_lin_w: Tensor          # [13440, 1024]
    var img_n1_lin_b: Tensor          # [13440]
    var img_n1_norm_w: Tensor         # [3360]
    var img_n2_lin_w: Tensor
    var img_n2_lin_b: Tensor
    var img_n2_norm_w: Tensor
    var img_n3_lin_w: Tensor
    var img_n3_lin_b: Tensor
    var img_n3_norm_w: Tensor
    var ins_n1_lin_w: Tensor
    var ins_n1_lin_b: Tensor
    var ins_n1_norm_w: Tensor
    var ins_n2_lin_w: Tensor
    var ins_n2_lin_b: Tensor
    var ins_n2_norm_w: Tensor
    # ── plain RMSNorm gammas (post-attn / pre+post MLP), per stream ──
    var img_attn_norm_w: Tensor       # [3360]
    var img_self_attn_norm_w: Tensor
    var img_ffn_norm1_w: Tensor
    var img_ffn_norm2_w: Tensor
    var ins_attn_norm_w: Tensor
    var ins_ffn_norm1_w: Tensor
    var ins_ffn_norm2_w: Tensor
    # ── joint attention processor (separate per-stream projections, NO bias) ──
    var ji_img_to_q: Tensor           # [3360, 3360]
    var ji_img_to_k: Tensor           # [840, 3360]
    var ji_img_to_v: Tensor           # [840, 3360]
    var ji_ins_to_q: Tensor           # [3360, 3360]
    var ji_ins_to_k: Tensor           # [840, 3360]
    var ji_ins_to_v: Tensor           # [840, 3360]
    var ji_img_out: Tensor            # [3360, 3360]
    var ji_ins_out: Tensor            # [3360, 3360]
    var ji_to_out: Tensor             # [3360, 3360]  (to_out.0, no bias)
    var ji_norm_q: Tensor             # [120]
    var ji_norm_k: Tensor             # [120]
    # ── img self-attn (BooguImageAttnProcessor, NO bias) ──
    var sa_to_q: Tensor               # [3360, 3360]
    var sa_to_k: Tensor               # [840, 3360]
    var sa_to_v: Tensor               # [840, 3360]
    var sa_to_out: Tensor             # [3360, 3360]  (to_out.0, no bias)
    var sa_norm_q: Tensor             # [120]
    var sa_norm_k: Tensor             # [120]
    # ── SwiGLU feed-forwards (per stream, NO bias) ──
    var img_ff_w1: Tensor             # [13568, 3360]
    var img_ff_w2: Tensor             # [3360, 13568]
    var img_ff_w3: Tensor             # [13568, 3360]
    var ins_ff_w1: Tensor
    var ins_ff_w2: Tensor
    var ins_ff_w3: Tensor

    @staticmethod
    def load(
        st: ShardedSafeTensors, prefix: String, ctx: DeviceContext
    ) raises -> BooguDoubleStreamBlock:
        """Load one double-stream block by key prefix (e.g. "double_stream_layers.0")."""
        var p = prefix + "."
        var jp = p + "img_instruct_attn.processor."
        return BooguDoubleStreamBlock(
            _load_w(st, p + "img_norm1.linear.weight", ctx),
            _load_w(st, p + "img_norm1.linear.bias", ctx),
            _load_w(st, p + "img_norm1.norm.weight", ctx),
            _load_w(st, p + "img_norm2.linear.weight", ctx),
            _load_w(st, p + "img_norm2.linear.bias", ctx),
            _load_w(st, p + "img_norm2.norm.weight", ctx),
            _load_w(st, p + "img_norm3.linear.weight", ctx),
            _load_w(st, p + "img_norm3.linear.bias", ctx),
            _load_w(st, p + "img_norm3.norm.weight", ctx),
            _load_w(st, p + "instruct_norm1.linear.weight", ctx),
            _load_w(st, p + "instruct_norm1.linear.bias", ctx),
            _load_w(st, p + "instruct_norm1.norm.weight", ctx),
            _load_w(st, p + "instruct_norm2.linear.weight", ctx),
            _load_w(st, p + "instruct_norm2.linear.bias", ctx),
            _load_w(st, p + "instruct_norm2.norm.weight", ctx),
            _load_w(st, p + "img_attn_norm.weight", ctx),
            _load_w(st, p + "img_self_attn_norm.weight", ctx),
            _load_w(st, p + "img_ffn_norm1.weight", ctx),
            _load_w(st, p + "img_ffn_norm2.weight", ctx),
            _load_w(st, p + "instruct_attn_norm.weight", ctx),
            _load_w(st, p + "instruct_ffn_norm1.weight", ctx),
            _load_w(st, p + "instruct_ffn_norm2.weight", ctx),
            _load_w(st, jp + "img_to_q.weight", ctx),
            _load_w(st, jp + "img_to_k.weight", ctx),
            _load_w(st, jp + "img_to_v.weight", ctx),
            _load_w(st, jp + "instruct_to_q.weight", ctx),
            _load_w(st, jp + "instruct_to_k.weight", ctx),
            _load_w(st, jp + "instruct_to_v.weight", ctx),
            _load_w(st, jp + "img_out.weight", ctx),
            _load_w(st, jp + "instruct_out.weight", ctx),
            _load_w(st, p + "img_instruct_attn.to_out.0.weight", ctx),
            _load_w(st, p + "img_instruct_attn.norm_q.weight", ctx),
            _load_w(st, p + "img_instruct_attn.norm_k.weight", ctx),
            _load_w(st, p + "img_self_attn.to_q.weight", ctx),
            _load_w(st, p + "img_self_attn.to_k.weight", ctx),
            _load_w(st, p + "img_self_attn.to_v.weight", ctx),
            _load_w(st, p + "img_self_attn.to_out.0.weight", ctx),
            _load_w(st, p + "img_self_attn.norm_q.weight", ctx),
            _load_w(st, p + "img_self_attn.norm_k.weight", ctx),
            _load_w(st, p + "img_feed_forward.linear_1.weight", ctx),
            _load_w(st, p + "img_feed_forward.linear_2.weight", ctx),
            _load_w(st, p + "img_feed_forward.linear_3.weight", ctx),
            _load_w(st, p + "instruct_feed_forward.linear_1.weight", ctx),
            _load_w(st, p + "instruct_feed_forward.linear_2.weight", ctx),
            _load_w(st, p + "instruct_feed_forward.linear_3.weight", ctx),
        )

    # ── joint attention processor (BooguImageDoubleStreamSelfAttnProcessor) ─────
    # img_norm1_out: [1,L_IMG,3360]; instruct_norm1_out: [1,L_INSTRUCT,3360];
    # joint cos/sin: [JOINT,60] F32 (JOINT = L_INSTRUCT + L_IMG). Returns the FULL
    # joint output [1,JOINT,3360] (instruct-first). Caller slices the two segments.
    def _joint_attn[
        L_INSTRUCT: Int, L_IMG: Int, JOINT: Int
    ](
        self,
        img_norm1_out: Tensor,
        instruct_norm1_out: Tensor,
        cos: Tensor,
        sin: Tensor,
        ctx: DeviceContext,
    ) raises -> Tensor:
        # separate per-stream q/k/v projections (no bias).
        var img_q = linear(img_norm1_out, self.ji_img_to_q, None, ctx)   # [1,L_IMG,3360]
        var img_k = linear(img_norm1_out, self.ji_img_to_k, None, ctx)   # [1,L_IMG,840]
        var img_v = linear(img_norm1_out, self.ji_img_to_v, None, ctx)
        var ins_q = linear(instruct_norm1_out, self.ji_ins_to_q, None, ctx)  # [1,L_INS,3360]
        var ins_k = linear(instruct_norm1_out, self.ji_ins_to_k, None, ctx)
        var ins_v = linear(instruct_norm1_out, self.ji_ins_to_v, None, ctx)
        # concat INSTRUCT-FIRST along seq (b=1: _concat is exactly this).
        var q = concat(1, ctx, ins_q, img_q)   # [1,JOINT,3360]
        var k = concat(1, ctx, ins_k, img_k)   # [1,JOINT,840]
        var v = concat(1, ctx, ins_v, img_v)   # [1,JOINT,840]
        # view into heads.
        q = reshape(q, [1, JOINT, BOOGU_HEADS, BOOGU_HEAD_DIM], ctx)
        k = reshape(k, [1, JOINT, BOOGU_KV_HEADS, BOOGU_HEAD_DIM], ctx)
        v = reshape(v, [1, JOINT, BOOGU_KV_HEADS, BOOGU_HEAD_DIM], ctx)
        # per-head qk RMSNorm over head_dim (eps=1e-5).
        q = rms_norm(q, self.ji_norm_q, BOOGU_NORM_EPS, ctx)
        k = rms_norm(k, self.ji_norm_k, BOOGU_NORM_EPS, ctx)
        # interleaved rope on q AND k with the JOINT table.
        var cos_q = _expand_rope_table_per_head(cos, JOINT, BOOGU_HEADS, ctx)
        var sin_q = _expand_rope_table_per_head(sin, JOINT, BOOGU_HEADS, ctx)
        var cos_k = _expand_rope_table_per_head(cos, JOINT, BOOGU_KV_HEADS, ctx)
        var sin_k = _expand_rope_table_per_head(sin, JOINT, BOOGU_KV_HEADS, ctx)
        q = rope_interleaved(q, cos_q, sin_q, ctx)
        k = rope_interleaved(k, cos_k, sin_k, ctx)
        # GQA repeat kv n_rep=4 -> [1,JOINT,28,120].
        k = repeat_kv_f32(k, JOINT, BOOGU_KV_HEADS, BOOGU_GQA_NREP, BOOGU_HEAD_DIM, ctx)
        v = repeat_kv_f32(v, JOINT, BOOGU_KV_HEADS, BOOGU_GQA_NREP, BOOGU_HEAD_DIM, ctx)
        # full attention (mask all-True, b=1) => sdpa_nomask. scale = 1/sqrt(120).
        var scale = Float32(1.0 / (Float32(BOOGU_HEAD_DIM) ** 0.5))
        var attn = sdpa_nomask[1, JOINT, BOOGU_HEADS, BOOGU_HEAD_DIM](q, k, v, scale, ctx)
        var merged = reshape(attn, [1, JOINT, BOOGU_HIDDEN_SIZE], ctx)
        # split instruct-first, per-stream out projections, re-concat instruct-first.
        var ins_h = slice(merged, 1, 0, L_INSTRUCT, ctx)        # [1,L_INS,3360]
        var img_h = slice(merged, 1, L_INSTRUCT, L_IMG, ctx)    # [1,L_IMG,3360]
        var ins_proj = linear(ins_h, self.ji_ins_out, None, ctx)
        var img_proj = linear(img_h, self.ji_img_out, None, ctx)
        var remerged = concat(1, ctx, ins_proj, img_proj)       # [1,JOINT,3360]
        # final joint output projection (to_out.0, no bias; to_out.1 dropout=identity).
        return linear(remerged, self.ji_to_out, None, ctx)

    # ── img self-attn (BooguImageAttnProcessor — identical to BooguBlock._attn) ──
    # x: [1,L_IMG,3360]; combined-img cos/sin: [L_IMG,60] F32. Returns [1,L_IMG,3360].
    def _img_self_attn[
        L_IMG: Int
    ](self, x: Tensor, cos: Tensor, sin: Tensor, ctx: DeviceContext) raises -> Tensor:
        var q = linear(x, self.sa_to_q, None, ctx)              # [1,L_IMG,3360]
        var k = linear(x, self.sa_to_k, None, ctx)              # [1,L_IMG,840]
        var v = linear(x, self.sa_to_v, None, ctx)              # [1,L_IMG,840]
        q = reshape(q, [1, L_IMG, BOOGU_HEADS, BOOGU_HEAD_DIM], ctx)
        k = reshape(k, [1, L_IMG, BOOGU_KV_HEADS, BOOGU_HEAD_DIM], ctx)
        v = reshape(v, [1, L_IMG, BOOGU_KV_HEADS, BOOGU_HEAD_DIM], ctx)
        q = rms_norm(q, self.sa_norm_q, BOOGU_NORM_EPS, ctx)
        k = rms_norm(k, self.sa_norm_k, BOOGU_NORM_EPS, ctx)
        var cos_q = _expand_rope_table_per_head(cos, L_IMG, BOOGU_HEADS, ctx)
        var sin_q = _expand_rope_table_per_head(sin, L_IMG, BOOGU_HEADS, ctx)
        var cos_k = _expand_rope_table_per_head(cos, L_IMG, BOOGU_KV_HEADS, ctx)
        var sin_k = _expand_rope_table_per_head(sin, L_IMG, BOOGU_KV_HEADS, ctx)
        q = rope_interleaved(q, cos_q, sin_q, ctx)
        k = rope_interleaved(k, cos_k, sin_k, ctx)
        k = repeat_kv_f32(k, L_IMG, BOOGU_KV_HEADS, BOOGU_GQA_NREP, BOOGU_HEAD_DIM, ctx)
        v = repeat_kv_f32(v, L_IMG, BOOGU_KV_HEADS, BOOGU_GQA_NREP, BOOGU_HEAD_DIM, ctx)
        var scale = Float32(1.0 / (Float32(BOOGU_HEAD_DIM) ** 0.5))
        var attn = sdpa_nomask[1, L_IMG, BOOGU_HEADS, BOOGU_HEAD_DIM](q, k, v, scale, ctx)
        var merged = reshape(attn, [1, L_IMG, BOOGU_HIDDEN_SIZE], ctx)
        return linear(merged, self.sa_to_out, None, ctx)        # to_out.0 (no bias)

    # ── img SwiGLU feed-forward (LuminaFeedForward) ───────────────────────────
    def _img_feed_forward(self, x: Tensor, ctx: DeviceContext) raises -> Tensor:
        var h1 = linear(x, self.img_ff_w1, None, ctx)
        var h3 = linear(x, self.img_ff_w3, None, ctx)
        var act = swiglu(h1, h3, ctx)
        return linear(act, self.img_ff_w2, None, ctx)

    # ── instruct SwiGLU feed-forward ──────────────────────────────────────────
    def _ins_feed_forward(self, x: Tensor, ctx: DeviceContext) raises -> Tensor:
        var h1 = linear(x, self.ins_ff_w1, None, ctx)
        var h3 = linear(x, self.ins_ff_w3, None, ctx)
        var act = swiglu(h1, h3, ctx)
        return linear(act, self.ins_ff_w2, None, ctx)

    # ── forward (modulation=True). img: [1,L_IMG,3360]; instruct: [1,L_INSTRUCT,3360];
    #    temb: [1,1024]. Builds the joint rope (JOINT=L_INSTRUCT+L_IMG rows) from
    #    cap_len=L_INSTRUCT, h_tok=w_tok=sqrt(L_IMG), and the combined-img rope as the
    #    joint rows [L_INSTRUCT:JOINT]. Returns (img_out, instruct_out). ───────────
    def forward[
        L_INSTRUCT: Int, L_IMG: Int
    ](
        self,
        img: Tensor,
        instruct: Tensor,
        temb: Tensor,
        h_tok: Int,
        w_tok: Int,
        ctx: DeviceContext,
    ) raises -> Tuple[Tensor, Tensor]:
        comptime JOINT = L_INSTRUCT + L_IMG
        # ── rope tables: joint (JOINT rows) and combined-img (rows [L_INSTRUCT:JOINT]).
        # Tensor is Movable-not-Copyable; clone out of the returned tuple (no `^` out
        # of a tuple subscript).
        var joint_rope = build_boogu_rope_tables(L_INSTRUCT, h_tok, w_tok, ctx)
        var cos_j = joint_rope[0].clone(ctx)
        var sin_j = joint_rope[1].clone(ctx)
        # combined-img rope == img segment of the joint table (rope.py:427-437,
        # ref_img_len=0 => combined_img is exactly the img rows).
        var cos_img = slice(cos_j, 0, L_INSTRUCT, L_IMG, ctx)   # [L_IMG,60]
        var sin_img = slice(sin_j, 0, L_INSTRUCT, L_IMG, ctx)

        # ── Step 1: modulation for both streams (5 LuminaRMSNormZero). ──
        var m_n1 = _lumina_rms_norm_zero(
            img, temb, self.img_n1_lin_w, self.img_n1_lin_b, self.img_n1_norm_w, ctx
        )
        var img_norm1_out = m_n1[0].clone(ctx)
        var img_gate_msa = m_n1[1].clone(ctx)    # chunk1
        var img_scale_mlp = m_n1[2].clone(ctx)   # chunk2
        var img_gate_mlp = m_n1[3].clone(ctx)    # chunk3

        var m_n2 = _lumina_rms_norm_zero(
            img, temb, self.img_n2_lin_w, self.img_n2_lin_b, self.img_n2_norm_w, ctx
        )
        var img_norm2_out = m_n2[0].clone(ctx)
        var img_shift_mlp = m_n2[1].clone(ctx)   # chunk1 read as MLP shift

        var m_n3 = _lumina_rms_norm_zero(
            img, temb, self.img_n3_lin_w, self.img_n3_lin_b, self.img_n3_norm_w, ctx
        )
        var img_norm3_out = m_n3[0].clone(ctx)
        var img_gate_self = m_n3[1].clone(ctx)   # chunk1 read as self-attn gate

        var mi_n1 = _lumina_rms_norm_zero(
            instruct, temb, self.ins_n1_lin_w, self.ins_n1_lin_b, self.ins_n1_norm_w, ctx
        )
        var instruct_norm1_out = mi_n1[0].clone(ctx)
        var instruct_gate_msa = mi_n1[1].clone(ctx)
        var instruct_scale_mlp = mi_n1[2].clone(ctx)
        var instruct_gate_mlp = mi_n1[3].clone(ctx)

        var mi_n2 = _lumina_rms_norm_zero(
            instruct, temb, self.ins_n2_lin_w, self.ins_n2_lin_b, self.ins_n2_norm_w, ctx
        )
        var instruct_norm2_out = mi_n2[0].clone(ctx)
        var instruct_shift_mlp = mi_n2[1].clone(ctx)

        # ── Step 2: joint attention on [instruct + img], then split. ──
        var joint_attn_out = self._joint_attn[L_INSTRUCT, L_IMG, JOINT](
            img_norm1_out, instruct_norm1_out, cos_j, sin_j, ctx
        )  # [1,JOINT,3360]
        var instruct_attn_out = slice(joint_attn_out, 1, 0, L_INSTRUCT, ctx)
        var img_attn_out = slice(joint_attn_out, 1, L_INSTRUCT, L_IMG, ctx)

        # ── Step 3: image self-attention (combined-img rope). ──
        var img_self_attn_out = self._img_self_attn[L_IMG](
            img_norm3_out, cos_img, sin_img, ctx
        )

        # ── Step 4: residual updates (tanh on gates, (1+) on scales). ──
        # img += tanh(img_gate_msa) * img_attn_norm(img_attn_out)
        var img_a = rms_norm(img_attn_out, self.img_attn_norm_w, BOOGU_NORM_EPS, ctx)
        var img1 = add(img, mul(tanh_op(img_gate_msa, ctx), img_a, ctx), ctx)
        # img += tanh(img_gate_self) * img_self_attn_norm(img_self_attn_out)
        var img_s = rms_norm(
            img_self_attn_out, self.img_self_attn_norm_w, BOOGU_NORM_EPS, ctx
        )
        var img2 = add(img1, mul(tanh_op(img_gate_self, ctx), img_s, ctx), ctx)
        # img_mlp_in = (1 + img_scale_mlp) * img_norm2_out + img_shift_mlp
        var img_mlp_in = add(
            mul(add_scalar(img_scale_mlp, Float32(1.0), ctx), img_norm2_out, ctx),
            img_shift_mlp,
            ctx,
        )
        var img_mlp_out = self._img_feed_forward(
            rms_norm(img_mlp_in, self.img_ffn_norm1_w, BOOGU_NORM_EPS, ctx), ctx
        )
        var img_mlp_n = rms_norm(img_mlp_out, self.img_ffn_norm2_w, BOOGU_NORM_EPS, ctx)
        var img_out = add(img2, mul(tanh_op(img_gate_mlp, ctx), img_mlp_n, ctx), ctx)

        # instruct += tanh(instruct_gate_msa) * instruct_attn_norm(instruct_attn_out)
        var ins_a = rms_norm(
            instruct_attn_out, self.ins_attn_norm_w, BOOGU_NORM_EPS, ctx
        )
        var ins1 = add(instruct, mul(tanh_op(instruct_gate_msa, ctx), ins_a, ctx), ctx)
        # instruct_mlp_in = (1 + instruct_scale_mlp) * instruct_norm2_out + instruct_shift_mlp
        var ins_mlp_in = add(
            mul(
                add_scalar(instruct_scale_mlp, Float32(1.0), ctx),
                instruct_norm2_out,
                ctx,
            ),
            instruct_shift_mlp,
            ctx,
        )
        var ins_mlp_out = self._ins_feed_forward(
            rms_norm(ins_mlp_in, self.ins_ffn_norm1_w, BOOGU_NORM_EPS, ctx), ctx
        )
        var ins_mlp_n = rms_norm(ins_mlp_out, self.ins_ffn_norm2_w, BOOGU_NORM_EPS, ctx)
        var ins_out = add(ins1, mul(tanh_op(instruct_gate_mlp, ctx), ins_mlp_n, ctx), ctx)

        return (img_out^, ins_out^)


# ── Chunk C5: output norm (LuminaLayerNormContinuous) + 2D unpatchify ─────────
# Mirrors the Boogu output projection + reshape-to-image, read line-by-line
# (NOT inferred):
#   /home/alex/Boogu-Image/boogu/models/transformers/block_lumina2.py:74-122
#     LuminaLayerNormContinuous.forward (elementwise_affine=False here):
#       emb   = linear_1(silu(conditioning).to(x.dtype))      [1,1024] -> [1,3360]
#       x     = norm(x) * (1 + emb)[:, None, :]   norm=LayerNorm(3360, eps=1e-6,
#                                                  elementwise_affine=False) => pure
#                                                  standardization, NO learnable params.
#       x     = linear_2(x)                       [1,seq,3360] -> [1,seq,64]
#   /home/alex/Boogu-Image/boogu/models/transformers/transformer_boogu.py:937-944
#     norm_out construction: embedding_dim=3360, conditioning_embedding_dim=1024,
#     elementwise_affine=False, eps=1e-6, bias=True, out_dim=patch*patch*out_ch=64.
#   /home/alex/Boogu-Image/boogu/models/transformers/transformer_boogu.py:1576-1592
#     unpatchify of the IMAGE token rows:
#       rearrange(img_tokens, "(h w) (p1 p2 c) -> c (h p1) (w p2)",
#                 h=H//p, w=W//p, p1=p, p2=p)
#
# norm_out config (Boogu-Image-0.1-Base): embedding_dim=3360, cond_dim=1024,
# eps=1e-6 (NOT the 1e-5 used by the RMSNorms elsewhere), out_dim=64=2*2*16.
#
# REUSES foundation ops ONLY: ops/activations.silu, ops/linear.linear,
# ops/norm.layer_norm_no_affine (LayerNorm elementwise_affine=False:
# (x-mean)/sqrt(var+eps), F32-accumulated, no gamma/beta — confirmed
# ops/norm.mojo:782), ops/tensor_algebra.{mul, add_scalar, reshape}. The norm
# itself has NO weights. Nothing reimplemented for norm_out.
#
# unpatchify is NOT the existing ops/layout.unpatchify: that op's within-patch
# flatten is f=(c*p+ph)*p+pw (channel SLOWEST). Boogu's "(p1 p2 c)" is channel
# FASTEST, with output index (c, h*p+p1, w*p+p2). Different layout => a dedicated
# 2D kernel is hand-rolled below (gather, no reduction => no F32 accumulation).

comptime BOOGU_NORM_OUT_EPS = Float32(1.0e-6)   # eps=1e-6 (NOT 1e-5)
comptime BOOGU_OUT_CHANNELS = 16                 # out_channels (== in_channels)
comptime BOOGU_OUT_DIM = 64                      # patch*patch*out_ch = 2*2*16
comptime _UNPATCH_BLOCK = 256
comptime _UNPATCH_DYN1 = Layout.row_major(-1)


# ── BooguNormOut: LuminaLayerNormContinuous(elementwise_affine=False). ────────
# Tensor is Movable-not-Copyable so each weight is owned directly; forward
# returns a fresh tensor.
@fieldwise_init
struct BooguNormOut(Movable):
    # linear_1: nn.Linear(1024 -> 3360, bias=True).
    var linear_1_weight: Tensor        # [3360, 1024]
    var linear_1_bias: Tensor          # [3360]
    # linear_2: nn.Linear(3360 -> 64, bias=True).
    var linear_2_weight: Tensor        # [64, 3360]
    var linear_2_bias: Tensor          # [64]

    @staticmethod
    def load(
        st: ShardedSafeTensors, prefix: String, ctx: DeviceContext
    ) raises -> BooguNormOut:
        """Load norm_out weights by key prefix (e.g. "norm_out"). The LayerNorm
        has no learnable params (elementwise_affine=False)."""
        var p = prefix + "."
        return BooguNormOut(
            _load_w(st, p + "linear_1.weight", ctx),
            _load_w(st, p + "linear_1.bias", ctx),
            _load_w(st, p + "linear_2.weight", ctx),
            _load_w(st, p + "linear_2.bias", ctx),
        )

    # ── forward: x [1,seq,3360], temb [1,1024] -> [1,seq,64]. ─────────────────
    # block_lumina2.py:115-120:
    #   emb = linear_1(silu(temb))          (.to(x.dtype) is a no-op: both BF16)
    #   x   = layer_norm_no_affine(x, eps=1e-6) * (1 + emb)[:, None, :]
    #   out = linear_2(x)
    def forward(
        self, x: Tensor, temb: Tensor, ctx: DeviceContext
    ) raises -> Tensor:
        # scale = linear_1(silu(temb))  -> [1,3360]
        var scale = linear(
            silu(temb, ctx),
            self.linear_1_weight,
            Optional[Tensor](self.linear_1_bias.clone(ctx)),
            ctx,
        )  # [1, 3360]
        # broadcast (1+scale) over the seq dim as [1,1,3360].
        var scale_b = reshape(scale, [1, 1, BOOGU_HIDDEN_SIZE], ctx)
        # LayerNorm(elementwise_affine=False, eps=1e-6) — pure standardization.
        var xn = layer_norm_no_affine(x, BOOGU_NORM_OUT_EPS, ctx)
        var xs = mul(xn, add_scalar(scale_b, Float32(1.0), ctx), ctx)
        # out = linear_2(x)  -> [1,seq,64]
        return linear(
            xs,
            self.linear_2_weight,
            Optional[Tensor](self.linear_2_bias.clone(ctx)),
            ctx,
        )


# ── 2D unpatchify kernel (Boogu "(h w) (p1 p2 c) -> c (h p1) (w p2)"). ────────
# One thread per OUTPUT element. Output [C, H_out, W_out] where H_out=h_tok*p,
# W_out=w_tok*p, p=patch_size (=2 for Boogu). Decode the output flat index
# (row-major over [C, H_out, W_out]) into (c, oh, ow); recover
#   h  = oh // p,  p1 = oh % p        (output H is "(h p1)" => h outer, p1 inner)
#   w  = ow // p,  p2 = ow % p        (output W is "(w p2)" => w outer, p2 inner)
# token row    = h*w_tok + w          ("(h w)" => h outer, w inner; row-major)
# within-token = (p1*p + p2)*C + c    ("(p1 p2 c)" => p1 outer, p2 mid, c FASTEST)
# read in[token, within] = in_flat[token*F + within], F = p*p*C.
def _boogu_unpatch_kernel_f32(
    inp: LayoutTensor[DType.float32, _UNPATCH_DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _UNPATCH_DYN1, MutAnyOrigin],
    C: Int, H_OUT: Int, W_OUT: Int, p: Int, w_tok: Int, F: Int,
):
    var idx = Int(global_idx.x)
    var total = C * H_OUT * W_OUT
    if idx < total:
        var ow = idx % W_OUT
        var rem = idx // W_OUT
        var oh = rem % H_OUT
        var c = rem // H_OUT
        var h = oh // p
        var p1 = oh % p
        var w = ow // p
        var p2 = ow % p
        var token = h * w_tok + w
        var within = (p1 * p + p2) * C + c
        var in_off = token * F + within
        o[idx] = rebind[o.element_type](rebind[Scalar[DType.float32]](inp[in_off]))


def _boogu_unpatch_kernel_bf16(
    inp: LayoutTensor[DType.bfloat16, _UNPATCH_DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _UNPATCH_DYN1, MutAnyOrigin],
    C: Int, H_OUT: Int, W_OUT: Int, p: Int, w_tok: Int, F: Int,
):
    var idx = Int(global_idx.x)
    var total = C * H_OUT * W_OUT
    if idx < total:
        var ow = idx % W_OUT
        var rem = idx // W_OUT
        var oh = rem % H_OUT
        var c = rem // H_OUT
        var h = oh // p
        var p1 = oh % p
        var w = ow // p
        var p2 = ow % p
        var token = h * w_tok + w
        var within = (p1 * p + p2) * C + c
        var in_off = token * F + within
        o[idx] = rebind[o.element_type](rebind[Scalar[DType.bfloat16]](inp[in_off]))


def _boogu_unpatch_kernel_f16(
    inp: LayoutTensor[DType.float16, _UNPATCH_DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float16, _UNPATCH_DYN1, MutAnyOrigin],
    C: Int, H_OUT: Int, W_OUT: Int, p: Int, w_tok: Int, F: Int,
):
    var idx = Int(global_idx.x)
    var total = C * H_OUT * W_OUT
    if idx < total:
        var ow = idx % W_OUT
        var rem = idx // W_OUT
        var oh = rem % H_OUT
        var c = rem // H_OUT
        var h = oh // p
        var p1 = oh % p
        var w = ow // p
        var p2 = ow % p
        var token = h * w_tok + w
        var within = (p1 * p + p2) * C + c
        var in_off = token * F + within
        o[idx] = rebind[o.element_type](rebind[Scalar[DType.float16]](inp[in_off]))


# ── boogu_unpatchify: image tokens [img_len, 64] -> [16, h_tok*2, w_tok*2]. ───
# Exact inverse of the C6 patchify `c (h p1)(w p2) -> (h w)(p1 p2 c)`.
def boogu_unpatchify(
    img_tokens: Tensor, h_tok: Int, w_tok: Int, ctx: DeviceContext
) raises -> Tensor:
    """Boogu 2D unpatchify: rearrange "(h w) (p1 p2 c) -> c (h p1) (w p2)".

    Args:
        img_tokens: [img_len, 64] where img_len = h_tok*w_tok and
                    64 = p*p*out_ch = 2*2*16 (channel FASTEST within a token).
        h_tok:      image token rows  = H_lat / patch_size.
        w_tok:      image token cols  = W_lat / patch_size.
        ctx:        device context.

    Returns:
        [out_ch=16, H_lat=h_tok*2, W_lat=w_tok*2] in img_tokens' dtype.
    """
    var ishape = img_tokens.shape()
    if len(ishape) != 2:
        raise Error("boogu_unpatchify: img_tokens must be rank-2 [img_len, 64]")
    var img_len = ishape[0]
    var Fdim = ishape[1]
    var p = BOOGU_PATCH_SIZE
    var C = BOOGU_OUT_CHANNELS
    if Fdim != p * p * C:
        raise Error("boogu_unpatchify: last dim != patch*patch*out_ch (64)")
    if img_len != h_tok * w_tok:
        raise Error("boogu_unpatchify: img_len != h_tok*w_tok")
    var H_OUT = h_tok * p
    var W_OUT = w_tok * p
    var total = C * H_OUT * W_OUT

    var dt = img_tokens.dtype().to_mojo_dtype()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](img_tokens.nbytes())
    var rl = RuntimeLayout[_UNPATCH_DYN1].row_major(IndexList[1](total))
    var grid = (total + _UNPATCH_BLOCK - 1) // _UNPATCH_BLOCK
    if dt == DType.float32:
        var I = LayoutTensor[DType.float32, _UNPATCH_DYN1, MutAnyOrigin](
            img_tokens.buf.unsafe_ptr().bitcast[Float32](), rl
        )
        var O = LayoutTensor[DType.float32, _UNPATCH_DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), rl
        )
        ctx.enqueue_function[_boogu_unpatch_kernel_f32, _boogu_unpatch_kernel_f32](
            I, O, C, H_OUT, W_OUT, p, w_tok, Fdim, grid_dim=grid, block_dim=_UNPATCH_BLOCK
        )
    elif dt == DType.bfloat16:
        var I = LayoutTensor[DType.bfloat16, _UNPATCH_DYN1, MutAnyOrigin](
            img_tokens.buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        var O = LayoutTensor[DType.bfloat16, _UNPATCH_DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        ctx.enqueue_function[_boogu_unpatch_kernel_bf16, _boogu_unpatch_kernel_bf16](
            I, O, C, H_OUT, W_OUT, p, w_tok, Fdim, grid_dim=grid, block_dim=_UNPATCH_BLOCK
        )
    else:  # float16
        var I = LayoutTensor[DType.float16, _UNPATCH_DYN1, MutAnyOrigin](
            img_tokens.buf.unsafe_ptr().bitcast[Float16](), rl
        )
        var O = LayoutTensor[DType.float16, _UNPATCH_DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), rl
        )
        ctx.enqueue_function[_boogu_unpatch_kernel_f16, _boogu_unpatch_kernel_f16](
            I, O, C, H_OUT, W_OUT, p, w_tok, Fdim, grid_dim=grid, block_dim=_UNPATCH_BLOCK
        )
    ctx.synchronize()
    return Tensor(out_buf^, [C, H_OUT, W_OUT], img_tokens.dtype())


# ── Chunk C6: full Boogu DiT forward (T2I, no ref, batch=1) ───────────────────
# WIRES the verified C1–C5 chunks + the patchify. Mirrors
# `BooguImageTransformer2DModel.forward` (transformer_boogu.py:1253-1607), the
# no-ref T2I batch=1 path with enable_teacache / enable_taylorseer == False,
# read line-by-line (NOT inferred). Helper reference functions:
#   flat_and_pad_to_seq            (1096-1196) — the patchify (no-ref branch).
#   img_patch_embed_and_refine     (987-1094)  — no-ref branch == x_embed +
#                                                noise_refiner; combined_img == img.
#   preprocess_instruction_hidden_states (1215-1251) — tensor input => identity
#                                                (num_layers=1, reduce mean of 1).
#
# Forward (cap_len=16, latent [1,16,32,32] => h_tok=w_tok=16, img_len=256,
# joint seq=272, hidden=3360):
#   1. instruction_feats = input [1,16,4096] (mean of 1 layer = identity).
#   2. (temb [1,1024], caption [1,16,3360]) = embedders.time_caption_embed(ts, instr).
#   3. tokens [1,256,64] = patchify(latent)  (`c (h p1)(w p2)->(h w)(p1 p2 c)`,
#      channel FASTEST — exact inverse of boogu_unpatchify).
#   4. joint rope [272,60] = build_boogu_rope_tables(16,16,16). Segments via dim0
#      slice: cap = rows[0:16]; img/combined = rows[16:272]; joint = rows[0:272].
#   5. context_refiner.{0,1} (BooguBlock modulation=False) on caption, cap rope.
#   6. x = x_embed(tokens); noise_refiner.{0,1} (modulation=True) on x, img rope.
#      combined_img = x (no ref to prepend).
#   7. 8 double_stream_layers (BooguDoubleStreamBlock): builds joint+combined-img
#      rope internally from cap_len=16, h_tok=w_tok=16.
#   8. fuse: joint = concat(dim=1, [instruct(16), img(256)])  (INSTRUCT-FIRST).
#   9. 32 single_stream_layers (BooguBlock modulation=True) on joint, joint rope.
#  10. y = norm_out.forward(joint, temb)  -> [1,272,64].
#  11. img token rows = y[:, 16:272, :] -> [256,64]; vel = boogu_unpatchify(...)
#      -> [16,32,32]; return [1,16,32,32].
#
# REUSES the C1–C5 structs + foundation ops ONLY. The ONLY new math is the
# patchify (boogu_patchify) — the exact inverse of the C5 unpatchify kernel.
#
# Config (Boogu-Image-0.1-Base): 2 context_refiner, 2 noise_refiner,
# 8 double_stream_layers, 32 single_stream_layers, embedders, norm_out.

comptime BOOGU_N_CONTEXT_REFINER = 2
comptime BOOGU_N_NOISE_REFINER = 2
comptime BOOGU_N_DOUBLE_STREAM = 8
comptime BOOGU_N_SINGLE_STREAM = 32


# ── 2D patchify kernel (Boogu "c (h p1)(w p2) -> (h w)(p1 p2 c)"). ────────────
# The EXACT inverse of _boogu_unpatch_kernel_*. One thread per OUTPUT (token)
# element. Output [img_len, F] where img_len=h_tok*w_tok, F=p*p*C. Decode the
# output flat index (row-major over [img_len, F]) into (token, within); recover
#   h  = token // w_tok,  w  = token % w_tok        ("(h w)" => h outer, w inner)
#   p1 = (within // C) // p,  p2 = (within // C) % p, c = within % C
#                                                     ("(p1 p2 c)" => c FASTEST)
#   oh = h*p + p1,  ow = w*p + p2
# read in[c, oh, ow] = in_flat[(c*H_IN + oh)*W_IN + ow].
def _boogu_patch_kernel_f32(
    inp: LayoutTensor[DType.float32, _UNPATCH_DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _UNPATCH_DYN1, MutAnyOrigin],
    C: Int, H_IN: Int, W_IN: Int, p: Int, w_tok: Int, F: Int,
):
    var idx = Int(global_idx.x)
    var img_len = (H_IN // p) * (W_IN // p)
    var total = img_len * F
    if idx < total:
        var within = idx % F
        var token = idx // F
        var h = token // w_tok
        var w = token % w_tok
        var c = within % C
        var pp = within // C
        var p1 = pp // p
        var p2 = pp % p
        var oh = h * p + p1
        var ow = w * p + p2
        var in_off = (c * H_IN + oh) * W_IN + ow
        o[idx] = rebind[o.element_type](rebind[Scalar[DType.float32]](inp[in_off]))


def _boogu_patch_kernel_bf16(
    inp: LayoutTensor[DType.bfloat16, _UNPATCH_DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _UNPATCH_DYN1, MutAnyOrigin],
    C: Int, H_IN: Int, W_IN: Int, p: Int, w_tok: Int, F: Int,
):
    var idx = Int(global_idx.x)
    var img_len = (H_IN // p) * (W_IN // p)
    var total = img_len * F
    if idx < total:
        var within = idx % F
        var token = idx // F
        var h = token // w_tok
        var w = token % w_tok
        var c = within % C
        var pp = within // C
        var p1 = pp // p
        var p2 = pp % p
        var oh = h * p + p1
        var ow = w * p + p2
        var in_off = (c * H_IN + oh) * W_IN + ow
        o[idx] = rebind[o.element_type](rebind[Scalar[DType.bfloat16]](inp[in_off]))


def _boogu_patch_kernel_f16(
    inp: LayoutTensor[DType.float16, _UNPATCH_DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float16, _UNPATCH_DYN1, MutAnyOrigin],
    C: Int, H_IN: Int, W_IN: Int, p: Int, w_tok: Int, F: Int,
):
    var idx = Int(global_idx.x)
    var img_len = (H_IN // p) * (W_IN // p)
    var total = img_len * F
    if idx < total:
        var within = idx % F
        var token = idx // F
        var h = token // w_tok
        var w = token % w_tok
        var c = within % C
        var pp = within // C
        var p1 = pp // p
        var p2 = pp % p
        var oh = h * p + p1
        var ow = w * p + p2
        var in_off = (c * H_IN + oh) * W_IN + ow
        o[idx] = rebind[o.element_type](rebind[Scalar[DType.float16]](inp[in_off]))


# ── boogu_patchify: latent [in_ch=16, H, W] -> tokens [h_tok*w_tok, 64]. ──────
# The exact inverse of boogu_unpatchify (`c (h p1)(w p2) -> (h w)(p1 p2 c)`,
# channel FASTEST within the 64-wide token). Accepts the latent as rank-3
# [C,H,W] or rank-4 [1,C,H,W] (batch=1). Returns [h_tok*w_tok, p*p*C] (rank-2,
# the x_embedder input before the batch dim is restored by the caller).
def boogu_patchify(
    latent: Tensor, h_tok: Int, w_tok: Int, ctx: DeviceContext
) raises -> Tensor:
    """Boogu 2D patchify: rearrange "c (h p1)(w p2) -> (h w)(p1 p2 c)".

    Args:
        latent: [C=16, H, W] or [1, C=16, H, W] (batch=1). H=h_tok*2, W=w_tok*2.
        h_tok:  image token rows = H / patch_size (= H // 2).
        w_tok:  image token cols = W / patch_size (= W // 2).
        ctx:    device context.

    Returns:
        [img_len=h_tok*w_tok, F=p*p*C=64] in latent's dtype (channel FASTEST in F).
    """
    var lshape = latent.shape()
    var C: Int
    var H_IN: Int
    var W_IN: Int
    if len(lshape) == 4:
        if lshape[0] != 1:
            raise Error("boogu_patchify: batch dim must be 1")
        C = lshape[1]
        H_IN = lshape[2]
        W_IN = lshape[3]
    elif len(lshape) == 3:
        C = lshape[0]
        H_IN = lshape[1]
        W_IN = lshape[2]
    else:
        raise Error("boogu_patchify: latent must be rank-3 [C,H,W] or rank-4 [1,C,H,W]")
    if C != BOOGU_OUT_CHANNELS:
        raise Error("boogu_patchify: in_channels != 16")
    var p = BOOGU_PATCH_SIZE
    if H_IN != h_tok * p or W_IN != w_tok * p:
        raise Error("boogu_patchify: H/W != h_tok*2 / w_tok*2")
    var img_len = h_tok * w_tok
    var Fdim = p * p * C                                 # 64
    var total = img_len * Fdim

    var dt = latent.dtype().to_mojo_dtype()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](latent.nbytes())
    var rl = RuntimeLayout[_UNPATCH_DYN1].row_major(IndexList[1](total))
    var grid = (total + _UNPATCH_BLOCK - 1) // _UNPATCH_BLOCK
    if dt == DType.float32:
        var I = LayoutTensor[DType.float32, _UNPATCH_DYN1, MutAnyOrigin](
            latent.buf.unsafe_ptr().bitcast[Float32](), rl
        )
        var O = LayoutTensor[DType.float32, _UNPATCH_DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), rl
        )
        ctx.enqueue_function[_boogu_patch_kernel_f32, _boogu_patch_kernel_f32](
            I, O, C, H_IN, W_IN, p, w_tok, Fdim, grid_dim=grid, block_dim=_UNPATCH_BLOCK
        )
    elif dt == DType.bfloat16:
        var I = LayoutTensor[DType.bfloat16, _UNPATCH_DYN1, MutAnyOrigin](
            latent.buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        var O = LayoutTensor[DType.bfloat16, _UNPATCH_DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        ctx.enqueue_function[_boogu_patch_kernel_bf16, _boogu_patch_kernel_bf16](
            I, O, C, H_IN, W_IN, p, w_tok, Fdim, grid_dim=grid, block_dim=_UNPATCH_BLOCK
        )
    else:  # float16
        var I = LayoutTensor[DType.float16, _UNPATCH_DYN1, MutAnyOrigin](
            latent.buf.unsafe_ptr().bitcast[Float16](), rl
        )
        var O = LayoutTensor[DType.float16, _UNPATCH_DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), rl
        )
        ctx.enqueue_function[_boogu_patch_kernel_f16, _boogu_patch_kernel_f16](
            I, O, C, H_IN, W_IN, p, w_tok, Fdim, grid_dim=grid, block_dim=_UNPATCH_BLOCK
        )
    ctx.synchronize()
    return Tensor(out_buf^, [img_len, Fdim], latent.dtype())


# ── BooguDiT: the full transformer. Loads ALL blocks + embedders + norm_out
#    RESIDENT (≈20GB bf16 on a 24GB GPU). Tensor is Movable-not-Copyable so the
#    block lists own their tensors directly. ────────────────────────────────────
# Tensor (and thus BooguBlock / BooguDoubleStreamBlock) is Movable-not-Copyable,
# so it cannot go directly in a `List` (List requires Copyable). Wrap each block
# in ArcPointer (reference-counted, Copyable) — the established serenitymojo
# many-block pattern (flux1/anima/chroma DiTs). The block itself is never copied.
struct BooguDiT(Movable):
    """Boogu DiT with STREAMING block load.

    Holds only the small embedders + norm_out resident; each of the 46 transformer
    blocks (2 context_refiner + 2 noise_refiner + 8 double_stream + 32 single_stream)
    is loaded ON DEMAND inside `forward` and FREED at the end of its loop iteration.
    Bounds resident GPU to ~1 block + the SDPA scratch — essential at 1024 (joint
    seq ~4141 => a ~1.9GB F32 scores buffer per attention that cannot coexist with a
    20GB resident DiT on 24GB). Block weights are bf16 (uploaded, not re-converted),
    so per-step re-load is a cheap H2D copy, NOT the F32-reconvert trap."""

    var embedders: BooguEmbedders
    var norm_out: BooguNormOut
    var dir: String

    def __init__(
        out self,
        var embedders: BooguEmbedders,
        var norm_out: BooguNormOut,
        var dir: String,
    ):
        self.embedders = embedders^
        self.norm_out = norm_out^
        self.dir = dir^

    @staticmethod
    def load(transformer_dir: String, ctx: DeviceContext) raises -> BooguDiT:
        """Load ONLY the embedders + norm_out resident. The 46 transformer blocks
        are streamed per-layer inside `forward` (see struct doc)."""
        var embedders = BooguEmbedders.load(transformer_dir, ctx)
        var st = ShardedSafeTensors.open(transformer_dir)
        var norm_out = BooguNormOut.load(st, "norm_out", ctx)
        return BooguDiT(embedders^, norm_out^, transformer_dir)

    # ── full forward (T2I, no ref, batch=1). latent [1,16,32,32], timestep [1],
    #    instruction_feats [1,16,4096] -> velocity [1,16,32,32]. ──────────────
    def forward[
        CAP_LEN: Int, H_TOK: Int, W_TOK: Int
    ](
        self,
        latent: Tensor,
        timestep: Tensor,
        instruction_feats: Tensor,
        ctx: DeviceContext,
    ) raises -> Tensor:
        comptime IMG_LEN = H_TOK * W_TOK
        comptime JOINT = CAP_LEN + IMG_LEN

        # ── 1+2. embedders: (temb [1,1024], caption [1,CAP_LEN,3360]). ──
        # preprocess_instruction_hidden_states with a tensor input is identity.
        var tc = self.embedders.time_caption_embed(timestep, instruction_feats, ctx)
        var temb = tc[0].clone(ctx)                      # [1,1024]
        var caption = tc[1].clone(ctx)                   # [1,CAP_LEN,3360]

        # ── 3. patchify latent -> tokens [IMG_LEN,64] -> [1,IMG_LEN,64]. ──
        var tokens2 = boogu_patchify(latent, H_TOK, W_TOK, ctx)   # [IMG_LEN,64]
        var tokens = reshape(tokens2, [1, IMG_LEN, BOOGU_X_EMBED_IN], ctx)

        # ── 4. joint rope [JOINT,60]; slice the cap / img segments along dim0. ──
        var jt = build_boogu_rope_tables(CAP_LEN, H_TOK, W_TOK, ctx)
        var cos_j = jt[0].clone(ctx)                     # [JOINT,60]
        var sin_j = jt[1].clone(ctx)
        var cap_cos = slice(cos_j, 0, 0, CAP_LEN, ctx)   # rows [0:CAP_LEN]
        var cap_sin = slice(sin_j, 0, 0, CAP_LEN, ctx)
        var img_cos = slice(cos_j, 0, CAP_LEN, IMG_LEN, ctx)  # rows [CAP_LEN:JOINT]
        var img_sin = slice(sin_j, 0, CAP_LEN, IMG_LEN, ctx)

        # STREAMING: reopen the (mmap'd) safetensors once; each block below is
        # loaded just before use and dropped at the end of its loop iteration.
        var st = ShardedSafeTensors.open(self.dir)

        # ── 5. context_refiner ×2 (modulation=False; temb unused, cap rope). ──
        for i in range(BOOGU_N_CONTEXT_REFINER):
            var blk = BooguBlock.load(st, "context_refiner." + String(i), False, ctx)
            caption = blk.forward[CAP_LEN](caption, temb, cap_cos, cap_sin, ctx)

        # ── 6. x_embed + noise_refiner ×2 (modulation=True, img rope). ──
        var x = self.embedders.x_embed(tokens, ctx)      # [1,IMG_LEN,3360]
        for i in range(BOOGU_N_NOISE_REFINER):
            var blk = BooguBlock.load(st, "noise_refiner." + String(i), True, ctx)
            x = blk.forward[IMG_LEN](x, temb, img_cos, img_sin, ctx)
        # combined_img == x (no ref to prepend).

        # ── 7. 8 double_stream_layers (instruct=caption, img=x). ──
        for i in range(BOOGU_N_DOUBLE_STREAM):
            var blk = BooguDoubleStreamBlock.load(
                st, "double_stream_layers." + String(i), ctx
            )
            var ds = blk.forward[CAP_LEN, IMG_LEN](
                x, caption, temb, H_TOK, W_TOK, ctx
            )
            x = ds[0].clone(ctx)                         # img_out
            caption = ds[1].clone(ctx)                   # instruct_out

        # ── 8. fuse: joint = concat(dim=1, [instruct(CAP), img(IMG)]). ──
        var joint = concat(1, ctx, caption, x)           # [1,JOINT,3360]

        # ── 9. 32 single_stream_layers (modulation=True, joint rope). ──
        for i in range(BOOGU_N_SINGLE_STREAM):
            var blk = BooguBlock.load(st, "single_stream_layers." + String(i), True, ctx)
            joint = blk.forward[JOINT](joint, temb, cos_j, sin_j, ctx)

        # ── 10. norm_out -> [1,JOINT,64]. ──
        var y = self.norm_out.forward(joint, temb, ctx)  # [1,JOINT,64]

        # ── 11. extract img token rows [CAP_LEN:JOINT] + unpatchify. ──
        var y_img = slice(y, 1, CAP_LEN, IMG_LEN, ctx)   # [1,IMG_LEN,64]
        var y_img2 = reshape(y_img, [IMG_LEN, BOOGU_OUT_DIM], ctx)  # [IMG_LEN,64]
        var vel = boogu_unpatchify(y_img2, H_TOK, W_TOK, ctx)      # [16,H,W]
        var vshape = vel.shape()
        return reshape(vel, [1, vshape[0], vshape[1], vshape[2]], ctx)

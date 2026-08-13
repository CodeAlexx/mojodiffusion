# models/lingbotvideo/backbone.mojo — LingBot-Video attention forward (Chunk A1).
#
# Mirrors LingBotVideoAttention.forward (transformer_lingbot_video.py :211-319) +
# apply_rotary_emb (:115-120), the T2I / BASE transformer path (no packed_indices,
# no parallel_config, bidirectional SDPA, no mask):
#
#   q = to_q(x); k = to_k(x); v = to_v(x)      # Linear 2048->2048 NO bias, bf16
#   q,k,v -> [B,S,16,128]                        # unflatten heads
#   q = apply_rope(norm_q(q), rope)              # RMSNorm(128) THEN interleaved RoPE
#   k = apply_rope(norm_k(k), rope)              # v is NOT normed and NOT roped
#   out = SDPA(q,k,v)                            # bidirectional, no mask, 1/sqrt(128)
#   return to_out(out.reshape(B,S,2048))         # Linear 2048->2048 WITH bias, bf16
#
# apply_rotary_emb is INTERLEAVED complex RoPE (pairs (x[2i],x[2i+1]) rotated by
# the per-position angle) == ops/rope.rope_interleaved. The complex freqs table is
# per-position (shared across heads); we tile it [S,hd/2] -> [S*H,hd/2] so each of
# the H head-rows for a token reuses that token's angle (like nucleus _rope_per_head).
#
# DTYPE CONTRACT: x + to_q/k/v/out Linears bf16; norm_q/norm_k RMSNorm run F32
# (F32 accumulate; f32 gamma) with bf16 output; RoPE math F32 with F32 cos/sin.
# All ops here reuse serenitymojo/ops (linear/rms_norm/rope_interleaved/sdpa_nomask)
# which already honor these boundaries.
#
# Mojo 1.0.0b1, NVIDIA GPU. INFERENCE ONLY.

from std.math import sqrt, tanh
from std.memory import ArcPointer
from max.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.linear import linear, linear_bias
from serenitymojo.ops.norm import rms_norm, layer_norm_no_affine
from serenitymojo.ops.rope import rope_interleaved
from serenitymojo.ops.attention import sdpa_nomask, sdpa_nomask_tiled
from serenitymojo.ops.attention_flash import sdpa_flash_train_fwd
from serenitymojo.ops.tensor_algebra import reshape, add
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.activations import silu
from serenitymojo.ops.embeddings import timestep_embedding
from serenitymojo.ops.adaln import adaln_modulate, adaln_gate_residual, rms_norm_bf16_dev
from serenitymojo.ops.fp8 import load_fp8_expert_dequant, fp8_e4m3_dequant_perexpert_to_bf16
from serenitymojo.ops.mxfp4 import load_mxfp4_expert_dequant, mxfp4_dequant_to_bf16
from serenitymojo.ops.rope_tables import build_multiaxis_rope_tables
from serenitymojo.models.lingbotvideo.moe import lingbot_video_moe


@fieldwise_init
struct LingBotAttnConfig(Copyable, Movable):
    """LingBotVideoAttention hyperparameters (base transformer)."""

    var hidden: Int      # 2048
    var heads: Int       # 16
    var head_dim: Int    # 128
    var eps: Float32     # 1e-6

    @staticmethod
    def default() -> Self:
        return Self(2048, 16, 128, Float32(1e-6))


# Tile a per-position RoPE table [S, half] into the per-head-row layout
# [S*heads, half] rope_interleaved expects: row (token*heads + head) reuses the
# token's angle (the reference broadcasts freqs over the head axis).
def _tile_rope_perhead[
    S: Int
](tbl: Tensor, heads: Int, half: Int, ctx: DeviceContext) raises -> Tensor:
    var host = tbl.to_host(ctx)  # S * half (F32)
    var out = List[Float32]()
    out.resize(S * heads * half, Float32(0.0))
    for t in range(S):
        for h in range(heads):
            var dst = (t * heads + h) * half
            var src = t * half
            for i in range(half):
                out[dst + i] = host[src + i]
    var sh = [S * heads, half]
    return Tensor.from_host(out, sh^, STDtype.F32, ctx)


# Min comptime S for the cuDNN flash SDPA path (matches dense._FLASH_MIN_S). Below
# it the math-mode sdpa_nomask is used, so T2I probes (S < 4096) never reference the
# cshim symbol and run under plain `mojo run`. At/above (T2V) attention dispatches to
# the flame cuDNN flash shim — those runs must link -lserenity_cudnn_sdpa.
comptime _FLASH_MIN_S = 4096


def lingbot_attention[
    S: Int,
    USE_TILED: Bool = False,
](
    x: Tensor,          # [1, S, 2048] bf16
    w_q: Tensor,        # [2048, 2048] bf16
    w_k: Tensor,        # [2048, 2048] bf16
    w_v: Tensor,        # [2048, 2048] bf16
    w_o: Tensor,        # [2048, 2048] bf16
    b_o: Tensor,        # [2048] bf16
    norm_q_w: Tensor,   # [128] f32
    norm_k_w: Tensor,   # [128] f32
    cos: Tensor,        # [S, 64] f32 (freqs_cis.real)
    sin: Tensor,        # [S, 64] f32 (freqs_cis.imag)
    cfg: LingBotAttnConfig,
    ctx: DeviceContext,
) raises -> Tensor:
    """One LingBotVideoAttention block forward. Returns [1, S, 2048] bf16."""
    var h = cfg.heads
    var hd = cfg.head_dim
    var half = hd // 2

    # ── q/k/v projections (bf16, no bias) ────────────────────────────────────
    var q = linear(x, w_q, None, ctx)  # [1, S, 2048]
    var k = linear(x, w_k, None, ctx)
    var v = linear(x, w_v, None, ctx)

    # Reshape to per-head rows [S*H, hd] (row order token-major then head, which
    # matches the reference unflatten(2,(H,hd)) row-major memory).
    var q_rows = reshape(q, [S * h, hd], ctx)
    var k_rows = reshape(k, [S * h, hd], ctx)

    # ── RMSNorm over head_dim (F32 accumulate; bf16 out), q and k only ───────
    var q_n = rms_norm(q_rows, norm_q_w, cfg.eps, ctx)
    var k_n = rms_norm(k_rows, norm_k_w, cfg.eps, ctx)

    # ── interleaved RoPE on normed q/k (F32 tables, bf16 activations) ────────
    var cos_h = _tile_rope_perhead[S](cos, h, half, ctx)
    var sin_h = _tile_rope_perhead[S](sin, h, half, ctx)
    var q_r = rope_interleaved(q_n, cos_h, sin_h, ctx)
    var k_r = rope_interleaved(k_n, cos_h, sin_h, ctx)

    # ── SDPA (bidirectional, no mask, scale 1/sqrt(head_dim)) ────────────────
    var q4 = reshape(q_r, [1, S, h, hd], ctx)
    var k4 = reshape(k_r, [1, S, h, hd], ctx)
    var v4 = reshape(v, [1, S, h, hd], ctx)
    var scale = Float32(1.0) / Float32(sqrt(Float64(hd)))
    # SDPA dispatch (bidirectional, no mask):
    #   * S >= _FLASH_MIN_S (T2V, ~9-48K tokens): cuDNN flash shim (dense-proven;
    #     needs the cshim link flags at run time). Flash is comptime-eliminated for
    #     T2I (S < 4096) so those probes never reference the cshim symbol.
    #   * else: plain sdpa_nomask (small S), or the tiled online-softmax path
    #     (USE_TILED) which streams K/V and never materializes the [16,S,S] scores.
    var attn: Tensor
    comptime if S >= _FLASH_MIN_S:
        var fwd = sdpa_flash_train_fwd[1, S, 16, 128](q4, k4, v4, scale, ctx)
        attn = fwd.o.clone(ctx)  # own the bytes before the flash scratch drops
    else:
        @parameter
        if USE_TILED:
            attn = sdpa_nomask_tiled[1, S, 16, 128](q4, k4, v4, scale, ctx)
        else:
            attn = sdpa_nomask[1, S, 16, 128](q4, k4, v4, scale, ctx)

    # ── output projection (bf16, WITH bias) ──────────────────────────────────
    var attn_2d = reshape(attn, [S, cfg.hidden], ctx)  # [S, 2048]
    var out = linear_bias(attn_2d, w_o, b_o, ctx)      # [S, 2048]
    return reshape(out, [1, S, cfg.hidden], ctx)       # [1, S, 2048]


# ─────────────────────────────────────────────────────────────────────────────
# Chunk A3 — full LingBotVideoBlock (AdaLN 6-way modulation + attn + MoE +
# two gated residuals). Mirrors LingBotVideoBlock.forward
# (transformer_lingbot_video.py :921-965) EXACTLY:
#
#   mod = temb6.view(B,S,12288) + scale_shift_table            # FP32
#   shift_msa,scale_msa,gate_msa,shift_mlp,scale_mlp,gate_mlp = mod.chunk(6,-1)
#   gate_* = tanh(gate_*);  scale_* = 1.0 + scale_*            # FP32
#   attn_in = (norm1(x)*scale_msa + shift_msa).to(bf16)
#   x = x + (gate_msa * norm_post_attn(attn(attn_in,rope))).to(x.dtype)
#   ffn_in = (norm2(x)*scale_mlp + shift_mlp).to(bf16)
#   x = x + (gate_mlp * norm_post_ffn(ffn(ffn_in))).to(x.dtype)
#
# DTYPE CONTRACT (the #1 accumulation trap):
#   temb6 / scale_shift_table / mod / all 6 modulation tensors are FP32. The four
#   block RMSNorms (norm1/norm2/norm_post_attn/norm_post_ffn) mirror
#   LingBotVideoRMSNorm: upcast bf16 x to F32, F32-accumulate reduction with the
#   F32 gamma, then `.to(bf16)` (rounds to bf16). The modulation arithmetic
#   `(norm(x)*scale + shift)` and `(gate*normed)` are done in FP32 (host-side
#   here — every value is an F32) and cast to bf16 ONLY at the `.to(bf16)` /
#   `.to(x.dtype)` boundary. x stays bf16 across both residuals.


# Mirror LingBotVideoRMSNorm(x) for a bf16 x with an F32 gamma: upcast x -> F32,
# RMSNorm (F32 accumulate, F32 gamma), then `.to(bf16)`. Returns the bf16-rounded
# normed values as a host F32 list (so the following FP32 modulation reads the
# exact rounded values the reference `.to(input_dtype)` produced).
def _rms_norm_bf16_host(
    x: Tensor, gamma_f32: Tensor, eps: Float32, ctx: DeviceContext
) raises -> List[Float32]:
    var xf = cast_tensor(x, STDtype.F32, ctx)         # bf16 -> f32 (lossless)
    var nf = rms_norm(xf, gamma_f32, eps, ctx)        # F32 out, F32 gamma
    var nb = cast_tensor(nf, STDtype.BF16, ctx)       # .to(input_dtype) == bf16
    return nb.to_host(ctx)                            # bf16-rounded values, as F32


def lingbot_video_block[
    S: Int
](
    x_in: Tensor,               # [1, S, 2048] bf16
    temb_row: Tensor,           # [1, 12288] f32 (single modulation row — timestep
                                #   is per-batch so all S token rows are identical;
                                #   the device AdaLN kernels broadcast it)
    scale_shift_table: Tensor,  # [1, 12288] f32
    w_q: Tensor,                # attn: [2048,2048] bf16
    w_k: Tensor,
    w_v: Tensor,
    w_o: Tensor,
    b_o: Tensor,                # [2048] bf16
    norm_q_w: Tensor,           # [128] f32
    norm_k_w: Tensor,           # [128] f32
    cos: Tensor,                # [S, 64] f32
    sin: Tensor,                # [S, 64] f32
    norm1_w: Tensor,            # block norms: [2048] f32
    norm2_w: Tensor,
    norm_post_attn_w: Tensor,
    norm_post_ffn_w: Tensor,
    router_weight: Tensor,      # MoE: [128,2048] f32
    e_score_correction_bias: Tensor,  # [128] f32
    w1: Tensor,                 # [128,768,2048] bf16 (gate)
    w3: Tensor,                 # [128,768,2048] bf16 (up)
    w2: Tensor,                 # [128,2048,768] bf16 (down)
    shared_gate: Tensor,        # [768,2048] bf16
    shared_up: Tensor,          # [768,2048] bf16
    shared_down: Tensor,        # [2048,768] bf16
    cfg: LingBotAttnConfig,
    ctx: DeviceContext,
) raises -> Tensor:
    """One full LingBotVideoBlock forward. Returns [1, S, 2048] bf16.

    AdaLN is done on-device (ops/adaln — same F32 math + F32→bf16 boundaries as the
    old host path, but no S*12288 download / ~250M host element-ops per block; this
    is what makes T2V-scale S feasible). Chunk offsets: shift/scale/gate_msa =
    0/1/2*hdim, shift/scale/gate_mlp = 3/4/5*hdim."""
    var hdim = cfg.hidden       # 2048

    # ── attention sub-block ──────────────────────────────────────────────────
    # attn_in = (norm1(x)*scale_msa + shift_msa).to(bf16)
    var n1 = rms_norm_bf16_dev(x_in, norm1_w, cfg.eps, ctx)
    var attn_in = adaln_modulate(
        n1, temb_row, scale_shift_table, S, hdim, 0 * hdim, 1 * hdim, [1, S, hdim], ctx
    )
    var attn_out = lingbot_attention[S](
        attn_in, w_q, w_k, w_v, w_o, b_o, norm_q_w, norm_k_w, cos, sin, cfg, ctx
    )  # [1, S, 2048] bf16

    # x = x + (gate_msa * norm_post_attn(attn_out)).to(x.dtype)
    var npa = rms_norm_bf16_dev(attn_out, norm_post_attn_w, cfg.eps, ctx)
    var x = adaln_gate_residual(
        x_in, npa, temb_row, scale_shift_table, S, hdim, 2 * hdim, [1, S, hdim], ctx
    )

    # ── MoE sub-block ────────────────────────────────────────────────────────
    # ffn_in = (norm2(x)*scale_mlp + shift_mlp).to(bf16).
    var n2 = rms_norm_bf16_dev(x, norm2_w, cfg.eps, ctx)
    var ffn_in_bf16 = adaln_modulate(
        n2, temb_row, scale_shift_table, S, hdim, 3 * hdim, 4 * hdim, [S, hdim], ctx
    )
    # router GEMM uses tokens.float() where tokens IS the bf16 ffn_in.
    var ffn_in_f32 = cast_tensor(ffn_in_bf16, STDtype.F32, ctx)

    var ffn_out = lingbot_video_moe(
        ffn_in_bf16, ffn_in_f32, router_weight, e_score_correction_bias,
        w1, w3, w2, shared_gate, shared_up, shared_down, ctx,
    )  # [S, 2048] bf16

    # x = x + (gate_mlp * norm_post_ffn(ffn_out)).to(x.dtype).
    var npf = rms_norm_bf16_dev(ffn_out, norm_post_ffn_w, cfg.eps, ctx)
    return adaln_gate_residual(
        x, npf, temb_row, scale_shift_table, S, hdim, 5 * hdim, [1, S, hdim], ctx
    )


# ═════════════════════════════════════════════════════════════════════════════
# Chunk B1 — the transformer PRE-BLOCK path: patchify + patch_embed +
# text_embed + time->temb6 + the 3D RoPE-table BUILD (positions + cos/sin).
# Mirrors transformer_lingbot_video.py forward (:1119-1266) + make_joint_position_ids
# (:164-180) + LingBotVideoRotaryEmbedding (:123-161) + LingBotVideoTextEmbedder
# (:197-208). CLOSES skeptic FRAGILE-2 (A1/A3 consumed a precomputed rope table;
# here we BUILD it against the oracle freqs_cos/sin + pos_ids).
#
# DTYPE: this pre-block path is effectively FP32 (norms, time, rope, and the
# oracle's synthetic Linears are all FP32). We keep everything FP32 end-to-end;
# patch/text Linears produce the joint that (in production) feeds the bf16 blocks.
#
# patchify (:1119-1124): latent (B,C,T,H,W) -> reshape (B,C,gt,pF,gh,pH,gw,pW) ->
#   permute (0,2,4,6,3,5,7,1) -> reshape (B, gt*gh*gw, pF*pH*pW*C). Token order
#   (f,h,w) with f slowest; feature order (pf,ph,pw,c) with CHANNEL FASTEST. This
#   is the OPPOSITE within-patch order from ops/patchify3d (c slowest), so we
#   hand-roll the gather here (host-side; the whole pre-block is FP32 anyway).
# ═════════════════════════════════════════════════════════════════════════════


def lingbot_patchify(
    latent: Tensor,   # [1, C, T, H, W] f32
    patch_f: Int,     # pF (1)
    patch_h: Int,     # pH (2)
    patch_w: Int,     # pW (2)
    ctx: DeviceContext,
) raises -> Tensor:
    """LingBot-Video patchify: [1,C,T,H,W] -> [gt*gh*gw, pF*pH*pW*C] (f32).

    Token order (f,h,w) f-slowest; within-patch feature order (pf,ph,pw,c) with
    CHANNEL FASTEST — matches the reference permute (0,2,4,6,3,5,7,1) + reshape,
    NOT ops/patchify3d's c-slowest layout. Host-side gather (FP32 pre-block)."""
    var sh = latent.shape()
    if len(sh) != 5:
        raise Error("lingbot_patchify: latent must be rank-5 [1,C,T,H,W]")
    var C = sh[1]
    var T = sh[2]
    var Hh = sh[3]
    var Ww = sh[4]
    if T % patch_f != 0 or Hh % patch_h != 0 or Ww % patch_w != 0:
        raise Error("lingbot_patchify: patch sizes must divide T,H,W")
    var gt = T // patch_f
    var gh = Hh // patch_h
    var gw = Ww // patch_w
    var n_video = gt * gh * gw
    var pd = patch_f * patch_h * patch_w * C  # feature dim

    var src = latent.to_host(ctx)  # [C*T*H*W] (B=1) f32
    var out = List[Float32]()
    out.resize(n_video * pd, Float32(0.0))
    for f in range(gt):
        for h in range(gh):
            for w in range(gw):
                var patch = (f * gh + h) * gw + w
                # feature order (pf, ph, pw, c) — c fastest
                for pfi in range(patch_f):
                    for phi in range(patch_h):
                        for pwi in range(patch_w):
                            for ci in range(C):
                                var feat = (
                                    ((pfi * patch_h + phi) * patch_w + pwi) * C + ci
                                )
                                var src_t = f * patch_f + pfi
                                var src_h = h * patch_h + phi
                                var src_w = w * patch_w + pwi
                                var in_off = (
                                    ((ci * T + src_t) * Hh + src_h) * Ww + src_w
                                )
                                out[patch * pd + feat] = src[in_off]
    var oshape = List[Int]()
    oshape.append(n_video)
    oshape.append(pd)
    return Tensor.from_host(out^, oshape^, STDtype.F32, ctx)


def lingbot_patch_embed(
    patch_tokens: Tensor,   # [n_video, 64] f32
    pe_w: Tensor,           # [2048, 64] f32 (patch_embedder.weight)
    pe_b: Tensor,           # [2048] f32
    ctx: DeviceContext,
) raises -> Tensor:
    """patch_embedder: Linear(64->2048, bias). Returns [n_video, 2048] f32."""
    return linear_bias(patch_tokens, pe_w, pe_b, ctx)


def lingbot_text_embed(
    text_embeds: Tensor,   # [1, L, 2560] f32
    norm_w: Tensor,        # [2560] f32
    lin1_w: Tensor,        # [2048, 2560] f32
    lin1_b: Tensor,        # [2048] f32
    lin2_w: Tensor,        # [2048, 2048] f32
    lin2_b: Tensor,        # [2048] f32
    eps: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    """LingBotVideoTextEmbedder: RMSNorm(2560,eps1e-6) -> Linear-SiLU-Linear.
    Returns [1, L, 2048] f32 (text tokens)."""
    var n = rms_norm(text_embeds, norm_w, eps, ctx)   # [1,L,2560] f32
    var h1 = linear_bias(n, lin1_w, lin1_b, ctx)      # [1,L,2048]
    var h1a = silu(h1, ctx)
    return linear_bias(h1a, lin2_w, lin2_b, ctx)      # [1,L,2048]


# ── 3D RoPE BUILD: positions (make_joint_position_ids) + per-axis complex freqs ──
# make_joint_position_ids (:164-180): VIDEO tokens = meshgrid(t,h,w, indexing=ij)
#   flattened (f,h,w) with t-axis = arange(gt)+(text_len+1), h=arange(gh),
#   w=arange(gw); TEXT tokens = (arange(text_len)+1, 0, 0). Order [video; text].
# freqs (:123-161): per-axis a (dim d_a in (32,48,48)):
#   inv_freq = 1/theta^(arange(0,d_a,2)/d_a) == theta^(-i/half_a);
#   angle = pos[:,a] (x) inv_freq; cos/sin; concat 3 axes -> (S,64). theta=256.
# We build positions token-major [S*3] and call build_multiaxis_rope_tables
# (axes_dims=[32,48,48]) which produces EXACTLY this concat/inv_freq layout.
def lingbot_build_rope(
    grid_t: Int,
    grid_h: Int,
    grid_w: Int,
    text_len: Int,
    theta: Float32,
    ctx: DeviceContext,
) raises -> Tuple[Tensor, Tensor, Tensor]:
    """Build (freqs_cos, freqs_sin, pos_ids) for the joint [video; text] sequence.

    freqs_cos/freqs_sin: [S, 64] f32 (S = grid_t*grid_h*grid_w + text_len).
    pos_ids: [S, 3] f32 (integer grid positions cast to f32; for gating)."""
    var n_video = grid_t * grid_h * grid_w
    var S = n_video + text_len
    var pos = List[Float32]()
    pos.resize(S * 3, Float32(0.0))
    # VIDEO tokens: meshgrid(t,h,w) indexing='ij' flatten (f,h,w); t += text_len+1
    var toff = Float32(text_len + 1)
    var idx = 0
    for f in range(grid_t):
        for h in range(grid_h):
            for w in range(grid_w):
                pos[idx * 3 + 0] = Float32(f) + toff
                pos[idx * 3 + 1] = Float32(h)
                pos[idx * 3 + 2] = Float32(w)
                idx += 1
    # TEXT tokens: (arange(text_len)+1, 0, 0)
    for i in range(text_len):
        pos[idx * 3 + 0] = Float32(i + 1)
        pos[idx * 3 + 1] = Float32(0)
        pos[idx * 3 + 2] = Float32(0)
        idx += 1

    var pshape = List[Int]()
    pshape.append(S)
    pshape.append(3)
    var positions = Tensor.from_host(pos.copy(), pshape^, STDtype.F32, ctx)

    var axes = List[Int]()
    axes.append(32)
    axes.append(48)
    axes.append(48)
    var cs = build_multiaxis_rope_tables(positions, axes, theta, ctx, STDtype.F32)
    # Rebuild owned tensors (tuple elements can't be moved out by subscript).
    var half = 16 + 24 + 24  # axes_half sum == head_dim/2 == 64
    var cos_host = cs[0].to_host(ctx)
    var sin_host = cs[1].to_host(ctx)
    var cshape = List[Int]()
    cshape.append(S)
    cshape.append(half)
    var sshape = List[Int]()
    sshape.append(S)
    sshape.append(half)
    var cos_t = Tensor.from_host(cos_host^, cshape^, STDtype.F32, ctx)
    var sin_t = Tensor.from_host(sin_host^, sshape^, STDtype.F32, ctx)

    # pos_ids tensor [S,3] for gating (same host data).
    var idshape = List[Int]()
    idshape.append(S)
    idshape.append(3)
    var pos_ids = Tensor.from_host(pos^, idshape^, STDtype.F32, ctx)
    return (cos_t^, sin_t^, pos_ids^)


# ── time -> temb6 (:1236-1266) ──────────────────────────────────────────────
# time_proj = Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0) ==
#   ops/embeddings.timestep_embedding (COS-first, max_period=10000).
# time_embedder = TimestepEmbedding(256->2048, act silu, bias) -> t_emb (B,2048).
# temb_input = t_emb.expand(B,S,2048); temb6 = time_modulation(temb_input) where
# time_modulation = SiLU -> Linear(2048->12288). Output temb6 (S,12288).
def lingbot_time_embed(
    timestep: Tensor,      # [B] f32
    time_lin1_w: Tensor,   # [2048, 256] f32
    time_lin1_b: Tensor,   # [2048] f32
    time_lin2_w: Tensor,   # [2048, 2048] f32
    time_lin2_b: Tensor,   # [2048] f32
    time_mod_w: Tensor,    # [12288, 2048] f32
    time_mod_b: Tensor,    # [12288] f32
    seq_len: Int,          # S
    ctx: DeviceContext,
) raises -> Tuple[Tensor, Tensor]:
    """Returns (t_emb [B,2048], temb6 [S,12288]) f32 (B==1)."""
    var emb = timestep_embedding(timestep, 256, ctx, Float32(10000.0), STDtype.F32)  # [B,256]
    var h1 = linear_bias(emb, time_lin1_w, time_lin1_b, ctx)   # [B,2048]
    var h1a = silu(h1, ctx)
    var t_emb = linear_bias(h1a, time_lin2_w, time_lin2_b, ctx)  # [B,2048]

    # temb_input = t_emb.expand(S, 2048) (B==1) -> host tile.
    var te = t_emb.to_host(ctx)  # 2048
    var d = len(te)
    var exp_host = List[Float32]()
    exp_host.resize(seq_len * d, Float32(0.0))
    for t in range(seq_len):
        for j in range(d):
            exp_host[t * d + j] = te[j]
    var eshape = List[Int]()
    eshape.append(seq_len)
    eshape.append(d)
    var temb_input = Tensor.from_host(exp_host^, eshape^, STDtype.F32, ctx)
    # time_modulation = SiLU -> Linear(2048->12288).
    var mod_in = silu(temb_input, ctx)
    var temb6 = linear_bias(mod_in, time_mod_w, time_mod_b, ctx)  # [S,12288]
    return (t_emb^, temb6^)


# ═════════════════════════════════════════════════════════════════════════════
# Chunk B2 — FULL 48-layer backbone: compose B1 (embeddings + rope + time) +
# lingbot_video_block × depth (STREAMED weights, one block ~1.2GB at a time) +
# the POST head (norm_out LayerNorm no-affine -> final modulation -> proj_out ->
# unpatchify) -> velocity. Mirrors oracle_b2_backbone.py EXACTLY.
#
# WEIGHT STREAMING (16GB constraint): the 60GB model must NOT reside. We load
# each block's 20 weights via `ShardedSafeTensors.from_view` (mmap H2D at the
# STORED dtype — bf16 bulk / f32 norms+router+scale_shift), run the block, then
# drop the weights (Arc refcount) and trim the async pool before the next block.
# Proven streaming template: models/nldiffusion/backbone_stack.mojo.
#
# DTYPE (mirror production / the oracle):
#   * embeddings: patchify FP32 -> bf16 patch_embed (bf16 W+b); text_embedder
#     RMSNorm (f32 gamma, bf16 in/out) -> bf16 Linear-SiLU-Linear. joint is bf16.
#   * time path is pure FP32 (time_embedder + time_modulation f32) -> temb6 FP32.
#   * blocks: A3-gated bf16/f32 contract, unchanged.
#   * POST: norm_out LayerNorm no-affine (bf16 in -> f32-accum -> bf16 out);
#     final modulation (SiLU->Linear 2048->4096) FP32; final_hidden = LN*(1+scale)
#     + shift in FP32; proj_out casts to bf16 (bf16 W+b) -> bf16 [S,64]; take the
#     first n_video rows; unpatchify (inverse of lingbot_patchify).
# ═════════════════════════════════════════════════════════════════════════════

comptime _TArc = ArcPointer[Tensor]


def _lw(model: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> _TArc:
    """Load a weight at its STORED dtype (bf16 bulk / f32 norms) via from_view —
    NO force-cast; this is exactly the production dtype A1/A2/A3 gated against."""
    return _TArc(Tensor.from_view(model.tensor_view(name), ctx))


def _lwe(model: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> _TArc:
    """Load an EXPERT weight (blocks.i.ffn.experts.{w1,w2,w3}) as bf16, dequanting
    on load if the ckpt is quantized. Three ckpt flavors, detected per tensor:
      * mxfp4 (Phase-2, transformer_mxfp4/): `<name>_blocks`+`<name>_scales` present
        → mxfp4_dequant (experts ~54GB→~15GB).
      * fp8   (Phase-1, transformer_fp8/):  `<name>` is F8_E4M3 → per-expert dequant
        (~54GB→~27GB).
      * bf16  (original):                    stored-dtype `from_view`.
    grouped_expert_ffn is unchanged — it always receives bf16."""
    if model.has_tensor(name + "_blocks"):
        return _TArc(load_mxfp4_expert_dequant(model, name, ctx))
    if model.tensor_info(name).dtype == STDtype.F8_E4M3:
        return _TArc(load_fp8_expert_dequant(model, name, ctx))
    return _TArc(Tensor.from_view(model.tensor_view(name), ctx))


@fieldwise_init
struct LingBotBlockW(Movable):
    """One block's 20 streamed weights (stored dtypes). Dropping it frees the
    ~1.2GB of GPU buffers — the per-block streaming unit."""

    var scale_shift_table: _TArc      # [1,12288] f32
    var w_q: _TArc                    # [2048,2048] bf16
    var w_k: _TArc
    var w_v: _TArc
    var w_o: _TArc
    var b_o: _TArc                    # [2048] bf16
    var norm_q_w: _TArc               # [128] f32
    var norm_k_w: _TArc
    var norm1_w: _TArc                # [2048] f32
    var norm2_w: _TArc
    var norm_post_attn_w: _TArc
    var norm_post_ffn_w: _TArc
    var router_weight: _TArc          # [128,2048] f32
    var e_score_correction_bias: _TArc  # [128] f32
    var w1: _TArc                     # experts gate [128,768,2048] bf16
    var w3: _TArc                     # experts up   [128,768,2048] bf16
    var w2: _TArc                     # experts down [128,2048,768] bf16
    var shared_gate: _TArc            # [768,2048] bf16
    var shared_up: _TArc
    var shared_down: _TArc            # [2048,768] bf16


def _load_block_weights(
    model: ShardedSafeTensors, i: Int, ctx: DeviceContext
) raises -> LingBotBlockW:
    """Stream block-i's 20 weights from the shards (STORED dtypes). Field order
    packs experts as (w1=gate, w3=up, w2=down) so the block call receives them
    in its (w1, w3, w2) parameter order."""
    var p = String("blocks.") + String(i) + String(".")
    return LingBotBlockW(
        _lw(model, p + "scale_shift_table", ctx),
        _lw(model, p + "attn.to_q.weight", ctx),
        _lw(model, p + "attn.to_k.weight", ctx),
        _lw(model, p + "attn.to_v.weight", ctx),
        _lw(model, p + "attn.to_out.weight", ctx),
        _lw(model, p + "attn.to_out.bias", ctx),
        _lw(model, p + "attn.norm_q.weight", ctx),
        _lw(model, p + "attn.norm_k.weight", ctx),
        _lw(model, p + "norm1.weight", ctx),
        _lw(model, p + "norm2.weight", ctx),
        _lw(model, p + "norm_post_attn.weight", ctx),
        _lw(model, p + "norm_post_ffn.weight", ctx),
        _lw(model, p + "ffn.router.weight", ctx),
        _lw(model, p + "ffn.router.e_score_correction_bias", ctx),
        _lwe(model, p + "ffn.experts.w1", ctx),
        _lwe(model, p + "ffn.experts.w3", ctx),
        _lwe(model, p + "ffn.experts.w2", ctx),
        _lw(model, p + "ffn.shared_experts.gate_proj.weight", ctx),
        _lw(model, p + "ffn.shared_experts.up_proj.weight", ctx),
        _lw(model, p + "ffn.shared_experts.down_proj.weight", ctx),
    )


# ═════════════════════════════════════════════════════════════════════════════
# PHASE-2 HYBRID RESIDENCY (mxfp4 experts). Keep the first N blocks' PACKED mxfp4
# experts + their small non-expert weights RESIDENT on the GPU (loaded once), and
# stream the remaining depth-N blocks per forward (mxfp4 = 4× less I/O than bf16).
# Each resident block's experts are ~321MB packed (vs 1.2GB bf16), so ~35-40 blocks
# fit the 5080's free VRAM — the streaming disk read (the ~90s/step bottleneck)
# is avoided for the resident majority. Per forward a resident block dequants its 3
# packed experts into transient bf16 (pooled, freed each block) and hands a normal
# `LingBotBlockW` to `lingbot_video_block` — the block math is UNCHANGED.
# ═════════════════════════════════════════════════════════════════════════════

def _lwu(model: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> _TArc:
    """Load a U8 tensor (mxfp4 `_blocks`/`_scales`) RAW-resident (dtype preserved;
    U8 is not a compute dtype so `from_view` would reject it)."""
    return _TArc(Tensor.from_view_raw(model.tensor_view(name), ctx))


@fieldwise_init
struct LingBotResidentBlockW(Movable, Copyable):
    """One block resident on the GPU: 14 non-expert weights + 3 shared-expert
    weights (bf16/f32) + the 3 MoE experts as PACKED mxfp4 (blocks+scales U8).
    `.copy()` is a refcount bump (weights already on GPU), not a re-upload."""
    var scale_shift_table: _TArc
    var w_q: _TArc
    var w_k: _TArc
    var w_v: _TArc
    var w_o: _TArc
    var b_o: _TArc
    var norm_q_w: _TArc
    var norm_k_w: _TArc
    var norm1_w: _TArc
    var norm2_w: _TArc
    var norm_post_attn_w: _TArc
    var norm_post_ffn_w: _TArc
    var router_weight: _TArc
    var e_score_correction_bias: _TArc
    var w1_blk: _TArc    # mxfp4 U8 [E,M,G,16]
    var w1_sc: _TArc     # mxfp4 U8 [E,M,G]
    var w3_blk: _TArc
    var w3_sc: _TArc
    var w2_fp8: _TArc    # fp8 F8_E4M3 [E,H,F]  (w2/down-proj kept fp8 in the hybrid ckpt)
    var w2_scale: _TArc  # fp8 F32 [E]
    var shared_gate: _TArc
    var shared_up: _TArc
    var shared_down: _TArc


def _load_resident_block(
    model: ShardedSafeTensors, i: Int, ctx: DeviceContext
) raises -> LingBotResidentBlockW:
    """Load block-i RESIDENT for the ACCEPTED hybrid ckpt (transformer_mxfp4_w2fp8):
    non-expert weights bf16/f32, w1/w3 as packed mxfp4 (`_blocks`/`_scales`), and w2
    as fp8 (`w2` F8_E4M3 + `w2_scale` F32) — matching the `_lwe` streaming dispatch."""
    var p = String("blocks.") + String(i) + String(".")
    var ep = p + "ffn.experts."
    return LingBotResidentBlockW(
        _lw(model, p + "scale_shift_table", ctx),
        _lw(model, p + "attn.to_q.weight", ctx),
        _lw(model, p + "attn.to_k.weight", ctx),
        _lw(model, p + "attn.to_v.weight", ctx),
        _lw(model, p + "attn.to_out.weight", ctx),
        _lw(model, p + "attn.to_out.bias", ctx),
        _lw(model, p + "attn.norm_q.weight", ctx),
        _lw(model, p + "attn.norm_k.weight", ctx),
        _lw(model, p + "norm1.weight", ctx),
        _lw(model, p + "norm2.weight", ctx),
        _lw(model, p + "norm_post_attn.weight", ctx),
        _lw(model, p + "norm_post_ffn.weight", ctx),
        _lw(model, p + "ffn.router.weight", ctx),
        _lw(model, p + "ffn.router.e_score_correction_bias", ctx),
        _lwu(model, ep + "w1_blocks", ctx),
        _lwu(model, ep + "w1_scales", ctx),
        _lwu(model, ep + "w3_blocks", ctx),
        _lwu(model, ep + "w3_scales", ctx),
        _lwu(model, ep + "w2", ctx),          # F8_E4M3 raw (dtype preserved)
        _lw(model, ep + "w2_scale", ctx),     # F32 per-expert scale
        _lw(model, p + "ffn.shared_experts.gate_proj.weight", ctx),
        _lw(model, p + "ffn.shared_experts.up_proj.weight", ctx),
        _lw(model, p + "ffn.shared_experts.down_proj.weight", ctx),
    )


def _assemble_resident_block(
    rb: LingBotResidentBlockW, ctx: DeviceContext
) raises -> LingBotBlockW:
    """Dequant a resident block's experts → transient bf16 (pooled), and assemble a
    `LingBotBlockW` reusing the resident non-expert Arcs. w1/w3 dequant from mxfp4,
    w2 from fp8 (the hybrid ckpt). The bf16 experts free when the returned struct is
    dropped; the resident packed/fp8 weights persist."""
    var w1 = mxfp4_dequant_to_bf16(rb.w1_blk[], rb.w1_sc[], ctx)
    var w3 = mxfp4_dequant_to_bf16(rb.w3_blk[], rb.w3_sc[], ctx)
    var w2 = fp8_e4m3_dequant_perexpert_to_bf16(rb.w2_fp8[], rb.w2_scale[], ctx)
    return LingBotBlockW(
        rb.scale_shift_table, rb.w_q, rb.w_k, rb.w_v, rb.w_o, rb.b_o,
        rb.norm_q_w, rb.norm_k_w, rb.norm1_w, rb.norm2_w,
        rb.norm_post_attn_w, rb.norm_post_ffn_w,
        rb.router_weight, rb.e_score_correction_bias,
        _TArc(w1^), _TArc(w3^), _TArc(w2^),
        rb.shared_gate, rb.shared_up, rb.shared_down,
    )


@fieldwise_init
struct LingBotResidentStore(Movable):
    """The first `n_resident` blocks held resident on the GPU (packed mxfp4). Build
    ONCE (before the denoise loop) and pass to every `lingbot_backbone` call so the
    residency survives across steps. n_resident=0 ⇒ pure streaming (Phase-1 behavior)."""
    var blocks: List[LingBotResidentBlockW]
    var n_resident: Int

    @staticmethod
    def empty() -> LingBotResidentStore:
        return LingBotResidentStore(List[LingBotResidentBlockW](), 0)

    @staticmethod
    def load(
        model: ShardedSafeTensors, n_resident: Int, ctx: DeviceContext
    ) raises -> LingBotResidentStore:
        var blocks = List[LingBotResidentBlockW]()
        for i in range(n_resident):
            blocks.append(_load_resident_block(model, i, ctx))
        return LingBotResidentStore(blocks^, n_resident)


def _snapshot_bf16(t: Tensor, ctx: DeviceContext) raises -> Tensor:
    """Own a bf16 copy of `t` (host round-trip; identity for bf16 values) so a
    per-block tap survives the `joint = out^` reassignment."""
    return Tensor.from_host(t.to_host(ctx), t.shape().copy(), STDtype.BF16, ctx)


@fieldwise_init
struct LingBotBackboneOut(Movable):
    """B2 outputs: the pre-block joint (bf16), the requested per-block taps
    (bf16, in tap_indices order), and the final velocity (f32)."""

    var joint_preblock: _TArc     # [1,S,2048] bf16
    var taps: List[_TArc]         # each [1,S,2048] bf16
    var velocity: _TArc           # [1,Cout,T,H,W] f32


def lingbot_backbone[
    S: Int
](
    latent: Tensor,          # [1,C,T,H,W] f32
    timestep: Tensor,        # [B] f32
    text_embeds: Tensor,     # [1,text_len,2560] f32 (bf16 values)
    model: ShardedSafeTensors,
    grid_t: Int,
    grid_h: Int,
    grid_w: Int,
    text_len: Int,
    patch_f: Int,
    patch_h: Int,
    patch_w: Int,
    out_channels: Int,
    depth: Int,
    theta: Float32,
    tap_indices: List[Int],
    cfg: LingBotAttnConfig,
    store: LingBotResidentStore,
    ctx: DeviceContext,
) raises -> LingBotBackboneOut:
    """Full LingBot-Video transformer forward. Blocks < store.n_resident use the
    GPU-resident packed-mxfp4 experts (no disk stream); the rest stream per block.
    Pass `LingBotResidentStore.empty()` for pure streaming (Phase-1 behavior)."""
    var hdim = cfg.hidden       # 2048
    var eps = cfg.eps
    var n_video = grid_t * grid_h * grid_w

    # ── embedding + post weights (small; load once, stored dtypes) ────────────
    var pe_w = _lw(model, String("patch_embedder.weight"), ctx)         # bf16
    var pe_b = _lw(model, String("patch_embedder.bias"), ctx)           # bf16
    var t_norm_w = _lw(model, String("text_embedder.norm.weight"), ctx)   # f32
    var t_l1_w = _lw(model, String("text_embedder.linear_1.weight"), ctx) # bf16
    var t_l1_b = _lw(model, String("text_embedder.linear_1.bias"), ctx)   # bf16
    var t_l2_w = _lw(model, String("text_embedder.linear_2.weight"), ctx) # bf16
    var t_l2_b = _lw(model, String("text_embedder.linear_2.bias"), ctx)   # bf16
    var ti_l1_w = _lw(model, String("time_embedder.linear_1.weight"), ctx)  # f32
    var ti_l1_b = _lw(model, String("time_embedder.linear_1.bias"), ctx)    # f32
    var ti_l2_w = _lw(model, String("time_embedder.linear_2.weight"), ctx)  # f32
    var ti_l2_b = _lw(model, String("time_embedder.linear_2.bias"), ctx)    # f32
    var tmod_w = _lw(model, String("time_modulation.1.weight"), ctx)   # f32
    var tmod_b = _lw(model, String("time_modulation.1.bias"), ctx)     # f32
    var nom_w = _lw(model, String("norm_out_modulation.1.weight"), ctx)  # f32 [4096,2048]
    var nom_b = _lw(model, String("norm_out_modulation.1.bias"), ctx)    # f32 [4096]
    var proj_w = _lw(model, String("proj_out.weight"), ctx)   # bf16 [64,2048]
    var proj_b = _lw(model, String("proj_out.bias"), ctx)     # bf16 [64]

    # ── 1. embeddings -> joint (bf16) ─────────────────────────────────────────
    var pt_f32 = lingbot_patchify(latent, patch_f, patch_h, patch_w, ctx)  # [n_video,pd] f32
    var pt_bf16 = cast_tensor(pt_f32, STDtype.BF16, ctx)
    var x = lingbot_patch_embed(pt_bf16, pe_w[], pe_b[], ctx)   # [n_video,2048] bf16

    # text_embedder: RMSNorm (f32 gamma, bf16 in/out) -> bf16 Linear-SiLU-Linear.
    var te_bf16 = cast_tensor(text_embeds, STDtype.BF16, ctx)   # [1,L,2560] bf16
    var te_f32 = cast_tensor(te_bf16, STDtype.F32, ctx)         # bf16-rounded values as f32
    var tn_f32 = rms_norm(te_f32, t_norm_w[], eps, ctx)         # [1,L,2560] f32
    var tn_bf16 = cast_tensor(tn_f32, STDtype.BF16, ctx)        # .to(bf16)
    var th1 = linear_bias(tn_bf16, t_l1_w[], t_l1_b[], ctx)     # [1,L,2048] bf16
    var th1a = silu(th1, ctx)
    var text = linear_bias(th1a, t_l2_w[], t_l2_b[], ctx)       # [1,L,2048] bf16

    # joint = cat([x, text], dim=1) -> [1,S,2048] bf16 (host concat; identity).
    var x_host = x.to_host(ctx)          # n_video*2048 (f32 of bf16)
    var text_host = text.to_host(ctx)    # L*2048
    var joint_host = List[Float32]()
    joint_host.resize(S * hdim, Float32(0.0))
    for k in range(n_video * hdim):
        joint_host[k] = x_host[k]
    for k in range(text_len * hdim):
        joint_host[n_video * hdim + k] = text_host[k]
    var joint = Tensor.from_host(joint_host, [1, S, hdim], STDtype.BF16, ctx)
    var joint_pre = _snapshot_bf16(joint, ctx)

    # ── 2. rope tables + time -> temb6 / t_emb ────────────────────────────────
    var rope = lingbot_build_rope(grid_t, grid_h, grid_w, text_len, theta, ctx)
    # seq_len=1 → the second return is the SINGLE modulation row [1,12288]
    # (temb6 rows are all identical); the device AdaLN kernels broadcast it, so we
    # skip the S×12288 expand entirely (essential at T2V-scale S).
    var te = lingbot_time_embed(
        timestep, ti_l1_w[], ti_l1_b[], ti_l2_w[], ti_l2_b[],
        tmod_w[], tmod_b[], 1, ctx,
    )  # (t_emb [1,2048] f32, temb_row [1,12288] f32)

    # ── 3. stream the `depth` blocks ──────────────────────────────────────────
    var taps = List[_TArc]()
    for i in range(depth):
        var w = (
            _assemble_resident_block(store.blocks[i], ctx)
            if i < store.n_resident
            else _load_block_weights(model, i, ctx)
        )
        var out = lingbot_video_block[S](
            joint, te[1], w.scale_shift_table[],
            w.w_q[], w.w_k[], w.w_v[], w.w_o[], w.b_o[],
            w.norm_q_w[], w.norm_k_w[], rope[0], rope[1],
            w.norm1_w[], w.norm2_w[], w.norm_post_attn_w[], w.norm_post_ffn_w[],
            w.router_weight[], w.e_score_correction_bias[],
            w.w1[], w.w3[], w.w2[], w.shared_gate[], w.shared_up[], w.shared_down[],
            cfg, ctx,
        )  # [1,S,2048] bf16
        for k in range(len(tap_indices)):
            if tap_indices[k] == i:
                taps.append(_TArc(_snapshot_bf16(out, ctx)))
        joint = out^
        # Drop this block's ~1.2GB of GPU weights (RAII) before the next load;
        # every block allocates identical-sized buffers, so the async device
        # pool REUSES the freed slabs next iteration (no growth, no OOM). A hard
        # driver trim (cuMemPoolTrimTo) is unavailable under JIT `mojo run`.
        _ = w^
        ctx.synchronize()

    # ── 4. POST head ──────────────────────────────────────────────────────────
    # temb_input = t_emb.expand(S,2048); final_mod = SiLU->Linear(2048->4096).
    var t_emb_host = te[0].to_host(ctx)   # 2048
    var temb_in_host = List[Float32]()
    temb_in_host.resize(S * hdim, Float32(0.0))
    for t in range(S):
        for j in range(hdim):
            temb_in_host[t * hdim + j] = t_emb_host[j]
    var temb_in = Tensor.from_host(temb_in_host, [S, hdim], STDtype.F32, ctx)
    var mod_in = silu(temb_in, ctx)
    var final_mod = linear_bias(mod_in, nom_w[], nom_b[], ctx)  # [S,4096] f32
    var fm_host = final_mod.to_host(ctx)   # S*4096 (shift | scale)

    # final_hidden = norm_out(joint) * (1+scale) + shift (LN bf16 out, FP32 mod).
    var ln = layer_norm_no_affine(joint, eps, ctx)  # [1,S,2048] bf16
    var ln_host = ln.to_host(ctx)          # S*2048 (f32 of bf16)
    var twoh = 2 * hdim
    var fh = List[Float32]()
    fh.resize(S * hdim, Float32(0.0))
    for t in range(S):
        for j in range(hdim):
            var shift = fm_host[t * twoh + j]
            var scale = fm_host[t * twoh + hdim + j]
            fh[t * hdim + j] = ln_host[t * hdim + j] * (Float32(1.0) + scale) + shift
    var fh_bf16 = Tensor.from_host(fh, [S, hdim], STDtype.BF16, ctx)  # .to(bf16)
    var projected = linear_bias(fh_bf16, proj_w[], proj_b[], ctx)     # [S,64] bf16
    var proj_host = projected.to_host(ctx)   # S*64

    # unpatchify: take first n_video tokens; reshape (B,gt,gh,gw,pF,pH,pW,Cout),
    # permute (0,7,1,4,2,5,3,6) -> (B,Cout,T,H,W). Inverse of lingbot_patchify.
    var Tt = grid_t * patch_f
    var Hh = grid_h * patch_h
    var Ww = grid_w * patch_w
    var pd = patch_f * patch_h * patch_w * out_channels
    var vel = List[Float32]()
    vel.resize(out_channels * Tt * Hh * Ww, Float32(0.0))
    for f in range(grid_t):
        for h in range(grid_h):
            for w in range(grid_w):
                var token = (f * grid_h + h) * grid_w + w
                for pfi in range(patch_f):
                    for phi in range(patch_h):
                        for pwi in range(patch_w):
                            for ci in range(out_channels):
                                var feat = (
                                    ((pfi * patch_h + phi) * patch_w + pwi)
                                    * out_channels + ci
                                )
                                var tt = f * patch_f + pfi
                                var hh = h * patch_h + phi
                                var ww = w * patch_w + pwi
                                var out_off = ((ci * Tt + tt) * Hh + hh) * Ww + ww
                                vel[out_off] = proj_host[token * pd + feat]
    var velocity = Tensor.from_host(
        vel, [1, out_channels, Tt, Hh, Ww], STDtype.F32, ctx
    )

    return LingBotBackboneOut(_TArc(joint_pre^), taps^, _TArc(velocity^))

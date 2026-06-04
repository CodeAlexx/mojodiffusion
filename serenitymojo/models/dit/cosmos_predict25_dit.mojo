# models/dit/cosmos_predict25_dit.mojo — Cosmos-Predict2.5-2B DiT (MiniTrainDIT),
# pure Mojo + MAX. Inference-only, GPU-only.
#
# Reference (read LINE BY LINE, READ-ONLY):
#   /home/alex/EriDiffusion/inference-flame/src/models/cosmos_predict25_dit.rs
#   Oracle: cosmos_predict2/_src/predict2/networks/minimal_v4_dit.py (V2_2B_NET)
#   Per-layer captures: inference-flame/ports/cosmos-predict25-2b/parity/captures/
#
# Variant: COSMOS_V2_2B production (the shipped 4.1GB post-trained checkpoint,
#   cosmos_predict25_2b_dit.safetensors). Confirmed against the checkpoint keys:
#   model_channels=2048, num_blocks=28, num_heads=16, head_dim=128,
#   patch_spatial=2, patch_temporal=1, in_channels=16, out_channels=16,
#   adaln_lora_dim=256, crossattn (text) k/v in=1024.
#   PRODUCTION OVERRIDES (verified — ckpt has NO extra_pos_embedder.* keys, HAS
#   crossattn_proj.0.{weight[1024,100352],bias}, x_embedder.proj.1.weight[2048,72]):
#     lvg_wrapper=true            -> patch_in = (16+1+1)*2*2 = 72
#     concat_padding_mask=true
#     use_crossattn_projection=true  (100352 -> 1024 Linear+bias, before blocks)
#     extra_per_block_abs_pos_emb=false  (no learnable abs pos emb)
#     rope_enable_fps_modulation=false   (integer positions)
#     rope NTK ratios: h=3.0 w=3.0 t=1.0  -> PER-AXIS theta (see below)
#
# ── 3D RoPE (VideoRopePosition3DEmb, minimal_v4_dit.py:730-795) ───────────────
# GPT-NeoX HALF-SPLIT layout (NOT interleaved). head_dim=128 axis split:
#   dim_h = head_dim/6*2 = 42 ; dim_w = 42 ; dim_t = head_dim-2*dim_h = 44.
# Per-axis NTK theta:  theta_a = 10000 * ratio_a^(dim_a/(dim_a-2)).
#   t: 10000*1^(44/42)=10000 ; h=w: 10000*3^(42/40).
# Table row (t*H*W+h*W+w) = concat[ cos(t_ang), cos(h_ang), cos(w_ang) ] over the
# FIRST half (dim_*/2 each), summing to head_dim/2=64. rope_halfsplit pairs index
# d with d+64 sharing the same angle (cosmos cat([t,h,w]*2)). REQUIRES the
# per-axis-theta builder ops/rope_tables.build_multiaxis_rope_tables_per_axis
# (the scalar builder cannot express ratio 3.0). cos/sin are F32 -> cast to q/k
# dtype at the apply site (BF16 RoPE precision trap).
#
# ── Block (Block.forward, minimal_v4_dit.py:1257-1382) ───────────────────────
# Residual stream lives in F32 (cosmos magnitudes reach >30k by block 27 in BF16);
# only the sub-block forwards (attn/cross/mlp) run BF16. Three sub-blocks, each:
#   modulation = (adaln_modulation_<sub>(silu(emb)) + adaln_lora_B_T_3D).chunk(3)
#               -> (shift, scale, gate)
#   y = sub( LN_no_affine(x_bf16)*(1+scale) + shift )  ; x_f32 += (gate*y)_f32
# Self-attn: Q/K/V Linear(no bias); per-head RMSNorm(eps 1e-6) on Q,K (NOT V);
#   half-split 3D RoPE on Q,K; SDPA; output_proj. Cross-attn: Q from x, K/V from
#   text_context (1024->2048); per-head RMSNorm Q,K; NO RoPE; output_proj (text-
#   only, no k_img branch for V2_2B). MLP (GPT2FeedForward): Linear->GELU->Linear
#   (Python uses EXACT-erf GELU; serenity gelu is tanh-approx — ~0.02% ceiling).
#
# ── Forward (MiniTrainDIT.forward + LVG wrapper) ─────────────────────────────
# 1. LVG concat: x = cat([x, cond_video_mask|zeros], C)          (lvg_wrapper)
# 2. padding-mask concat: x = cat([x, padding_mask|zeros], C)    (concat_padding_mask)
# 3. crossattn_proj(text) 100352->1024 (Linear+bias)
# 4. patchify [B,C,T,H,W] -> [B,Tp,Hp,Wp,72]; x_embedder Linear -> [.., 2048]
# 5. timestep: sinusoidal(cos-first) -> MLP(linear1,silu,linear2[->6144]); the
#    sinusoidal sample IS emb_B_T_D (use_adaln_lora), linear2 IS adaln_lora_B_T_3D;
#    t_embedding_norm = RMSNorm on emb only.
# 6. build rope ONCE; 28 blocks; FinalLayer (LN_no_affine + 2-chunk adaln + Linear
#    [2048->64]); cosmos unpatchify (p1,p2,t',c) -> [B,16,T,H,W].
#
# DTYPE: bf16 weights+input, F32 accumulate. Mojo 1.0.0b1, NVIDIA GPU.
#
# REUSE: ops/linear.linear, ops/norm.{rms_norm,layer_norm}, ops/attention.sdpa_nomask,
# ops/softmax.softmax_lastdim, ops/rope.rope_halfsplit,
# ops/rope_tables.build_multiaxis_rope_tables_per_axis, ops/activations.{silu,gelu},
# ops/cast.cast_tensor, ops/embeddings.timestep_embedding,
# ops/tensor_algebra.*, ops/patchify3d.patchify3d, ShardedSafeTensors.

from std.math import log
from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import rms_norm, layer_norm
from serenitymojo.ops.attention import sdpa_nomask, sdpa_nomask_tiled
from serenitymojo.ops.softmax import softmax_lastdim
from serenitymojo.ops.rope import rope_halfsplit
from serenitymojo.ops.rope_tables import build_multiaxis_rope_tables_per_axis
from serenitymojo.ops.activations import silu, gelu
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.embeddings import timestep_embedding
from serenitymojo.ops.patchify3d import patchify3d
from serenitymojo.ops.tensor_algebra import (
    add, mul, add_scalar, mul_scalar, slice, reshape, permute, transpose,
    concat as _ta_concat,
)


# ── Config ──────────────────────────────────────────────────────────────────
@fieldwise_init
struct CosmosConfig(Copyable, Movable, ImplicitlyCopyable):
    var num_blocks: Int
    var model_channels: Int
    var num_heads: Int
    var head_dim: Int
    var in_channels: Int
    var out_channels: Int
    var patch_spatial: Int
    var patch_temporal: Int
    var adaln_lora_dim: Int
    var crossattn_in: Int          # text K/V Linear in-dim (1024)
    var crossattn_proj_in: Int     # crossattn_proj in-dim (100352)
    var eps: Float32
    var rope_h_ratio: Float32
    var rope_w_ratio: Float32
    var rope_t_ratio: Float32

    @staticmethod
    def v2_2b_production() -> CosmosConfig:
        return CosmosConfig(
            num_blocks=28, model_channels=2048, num_heads=16, head_dim=128,
            in_channels=16, out_channels=16, patch_spatial=2, patch_temporal=1,
            adaln_lora_dim=256, crossattn_in=1024, crossattn_proj_in=100352,
            eps=1.0e-6, rope_h_ratio=3.0, rope_w_ratio=3.0, rope_t_ratio=1.0,
        )


# ── RoPE axis split + per-axis NTK theta (cosmos_predict25_dit.rs:744-793) ───
# dim_h = head_dim/6*2 ; dim_w = dim_h ; dim_t = head_dim-2*dim_h.
# Cosmos table order is (t, h, w) — temporal first.
def cosmos_rope_axes(head_dim: Int) -> List[Int]:
    var dim_h = head_dim // 6 * 2
    var dim_w = dim_h
    var dim_t = head_dim - 2 * dim_h
    var out = List[Int]()
    out.append(dim_t)
    out.append(dim_h)
    out.append(dim_w)
    return out^


# theta_a = 10000 * ratio_a^(dim_a/(dim_a-2)). Order (t,h,w) to match axes.
def cosmos_rope_thetas(
    head_dim: Int, t_ratio: Float32, h_ratio: Float32, w_ratio: Float32
) -> List[Float32]:
    var dim_h = head_dim // 6 * 2
    var dim_w = dim_h
    var dim_t = head_dim - 2 * dim_h
    var t_ntk = t_ratio ** (Float32(dim_t) / (Float32(dim_t) - 2.0))
    var h_ntk = h_ratio ** (Float32(dim_h) / (Float32(dim_h) - 2.0))
    var w_ntk = w_ratio ** (Float32(dim_w) / (Float32(dim_w) - 2.0))
    var out = List[Float32]()
    out.append(10000.0 * t_ntk)
    out.append(10000.0 * h_ntk)
    out.append(10000.0 * w_ntk)
    return out^


# Per-token (t,h,w) integer positions, token-major: index s*3+a = token s grid
# coord on axis a. Token order t-major then h then w (matches flatten_thw:
# token = ti*HP*WP + hi*WP + wi). Returns F32 [rows*3].
def cosmos_rope_positions(
    tp: Int, hp: Int, wp: Int, ctx: DeviceContext
) raises -> Tensor:
    var rows = tp * hp * wp
    var host = List[Float32]()
    for ti in range(tp):
        for hi in range(hp):
            for wi in range(wp):
                host.append(Float32(ti))
                host.append(Float32(hi))
                host.append(Float32(wi))
    var shp = List[Int]()
    shp.append(rows * 3)
    return Tensor.from_host(host^, shp^, STDtype.F32, ctx)


# Build half-split cos/sin tables [rows, head_dim/2] in q/k dtype for a grid.
def cosmos_build_rope(
    tp: Int, hp: Int, wp: Int, cfg: CosmosConfig, dtype: STDtype, ctx: DeviceContext
) raises -> Tuple[Tensor, Tensor]:
    var positions = cosmos_rope_positions(tp, hp, wp, ctx)
    var axes = cosmos_rope_axes(cfg.head_dim)
    var thetas = cosmos_rope_thetas(
        cfg.head_dim, cfg.rope_t_ratio, cfg.rope_h_ratio, cfg.rope_w_ratio
    )
    var cs = build_multiaxis_rope_tables_per_axis(positions, axes, thetas, ctx)
    var cos_d = cast_tensor(cs[0], dtype, ctx)
    var sin_d = cast_tensor(cs[1], dtype, ctx)
    return (cos_d^, sin_d^)


# ── small helpers ─────────────────────────────────────────────────────────────
def _lin_nobias(
    x: Tensor, w: Dict[String, ArcPointer[Tensor]], name: String, ctx: DeviceContext
) raises -> Tensor:
    return linear(x, w[name][], None, ctx)


# LN_no_affine(x) then *(1+scale)+shift. x BF16 [.., D]; scale/shift F32 [.., D]
# (broadcast over the seq/spatial leading dims via reshape at call site). Returns
# BF16 (cast at end). LayerNorm eps=1e-6, elementwise_affine=False.
def _ln_modulate(
    x: Tensor, scale_f32: Tensor, shift_f32: Tensor, D: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    var ones_h = List[Float32]()
    var zeros_h = List[Float32]()
    for _ in range(D):
        ones_h.append(1.0)
        zeros_h.append(0.0)
    var ws = List[Int]()
    ws.append(D)
    var ws2 = List[Int]()
    ws2.append(D)
    var ones = Tensor.from_host(ones_h^, ws^, x.dtype(), ctx)
    var zeros = Tensor.from_host(zeros_h^, ws2^, x.dtype(), ctx)
    var normed = layer_norm(x, ones, zeros, eps, ctx)
    var normed_f32 = cast_tensor(normed, STDtype.F32, ctx)
    var sc1 = add_scalar(scale_f32, 1.0, ctx)
    var prod = mul(normed_f32, sc1, ctx)
    var out_f32 = add(prod, shift_f32, ctx)
    return cast_tensor(out_f32, x.dtype(), ctx)


# Per-head RMSNorm over last dim for [N, H, DH] (flatten to [N*H, DH]).
def _qk_norm(x_nhd: Tensor, w: Tensor, N: Int, H: Int, DH: Int, eps: Float32,
             ctx: DeviceContext) raises -> Tensor:
    var fshp = List[Int]()
    fshp.append(N * H)
    fshp.append(DH)
    var flat = reshape(x_nhd, fshp^, ctx)
    var normed = rms_norm(flat, w, eps, ctx)
    var oshp = List[Int]()
    oshp.append(N)
    oshp.append(H)
    oshp.append(DH)
    return reshape(normed, oshp^, ctx)


# adaln_modulation_<sub>: (linear2(linear1(silu(emb))) + adaln_lora).chunk(3).
# emb [S, D] BF16; adaln_lora [S, 3D] BF16. Returns (shift,scale,gate) F32 [S,D].
def _adaln_chunk3(
    emb: Tensor, adaln_lora: Tensor, w: Dict[String, ArcPointer[Tensor]],
    sub: String, S: Int, D: Int, ctx: DeviceContext,
) raises -> Tuple[Tensor, Tensor, Tensor]:
    var pre = String("adaln_modulation_") + sub
    var h0 = silu(emb, ctx)
    var h1 = _lin_nobias(h0, w, pre + ".1.weight", ctx)   # [S,256]
    var h2 = _lin_nobias(h1, w, pre + ".2.weight", ctx)   # [S,3D]
    var summed = add(h2, adaln_lora, ctx)
    var summed_f32 = cast_tensor(summed, STDtype.F32, ctx)
    var shift = slice(summed_f32, 1, 0, D, ctx)
    var scale = slice(summed_f32, 1, D, D, ctx)
    var gate = slice(summed_f32, 1, 2 * D, D, ctx)
    return (shift^, scale^, gate^)


# Broadcast a [S,D] modulation over spatial: x is [S,D] tokens already flattened,
# so shift/scale are directly [S,D]. BUT cosmos modulation is [B,T,D] broadcast
# over (H,W). We keep the whole stream flattened to [N=Tp*Hp*Wp, D] and expand the
# per-T modulation to per-token by repeating each T-row Hp*Wp times.
def _expand_t_to_tokens(
    mod_td: Tensor, tp: Int, hpwp: Int, D: Int, ctx: DeviceContext
) raises -> Tensor:
    # mod_td: [tp, D] -> [tp, 1, D] -> broadcast [tp, hpwp, D] -> [tp*hpwp, D].
    var s3 = List[Int]()
    s3.append(tp)
    s3.append(1)
    s3.append(D)
    var t3 = reshape(mod_td, s3^, ctx)
    var n = tp * hpwp * D
    var zh = List[Float32]()
    for _ in range(n):
        zh.append(0.0)
    var zs = List[Int]()
    zs.append(tp)
    zs.append(hpwp)
    zs.append(D)
    var zeros = Tensor.from_host(zh^, zs^, mod_td.dtype(), ctx)
    var bc = add(t3, zeros, ctx)  # [tp,hpwp,D]
    var os = List[Int]()
    os.append(tp * hpwp)
    os.append(D)
    return reshape(bc, os^, ctx)


# ── Self-attention ─ x_seq [N, D] (N=Tp*Hp*Wp). Returns [N, D]. ──────────────
def cosmos_self_attention[N: Int, H: Int, DH: Int](
    x_seq: Tensor, cos: Tensor, sin: Tensor,
    w: Dict[String, ArcPointer[Tensor]], eps: Float32, ctx: DeviceContext,
) raises -> Tensor:
    var scale = 1.0 / Float32(DH) ** 0.5
    var q = _lin_nobias(x_seq, w, "self_attn.q_proj.weight", ctx)
    var k = _lin_nobias(x_seq, w, "self_attn.k_proj.weight", ctx)
    var v = _lin_nobias(x_seq, w, "self_attn.v_proj.weight", ctx)
    # [N, H*DH] -> [N, H, DH]
    var q3 = _to_nhd(q, N, H, DH, ctx)
    var k3 = _to_nhd(k, N, H, DH, ctx)
    var v3 = _to_nhd(v, N, H, DH, ctx)
    q3 = _qk_norm(q3, w["self_attn.q_norm.weight"][], N, H, DH, eps, ctx)
    k3 = _qk_norm(k3, w["self_attn.k_norm.weight"][], N, H, DH, eps, ctx)
    # half-split RoPE: q,k as [1,N,H,DH]; cos/sin expanded to [N*H, DH/2].
    var q4 = _to_bshd(q3, N, H, DH, ctx)
    var k4 = _to_bshd(k3, N, H, DH, ctx)
    var v4 = _to_bshd(v3, N, H, DH, ctx)
    var cos_e = _expand_rope_per_head(cos, N, H, DH // 2, ctx)
    var sin_e = _expand_rope_per_head(sin, N, H, DH // 2, ctx)
    q4 = rope_halfsplit(q4, cos_e, sin_e, ctx)
    k4 = rope_halfsplit(k4, cos_e, sin_e, ctx)
    # Tiled (online-softmax) SDPA: never materializes the [N,N] scores, so the
    # Dh=128 self-attn runs at large N (multi-block full-res) without OOM. The
    # online softmax is exact -> cos=1.0 vs the math-mode sdpa_nomask, so the
    # block-0 numeric gate (cos 0.99999605) is unchanged.
    var att = sdpa_nomask_tiled[1, N, H, DH](q4, k4, v4, scale, ctx)  # [1,N,H,DH]
    var att2 = _from_bshd(att, N, H * DH, ctx)                  # [N, H*DH]
    return _lin_nobias(att2, w, "self_attn.output_proj.weight", ctx)


# ── Cross-attention (text only) ─ q from x [N,D]; k/v from ctx [TXT,Cin] ─────
def cosmos_cross_attention[N: Int, TXT: Int, H: Int, DH: Int](
    x_seq: Tensor, text_ctx: Tensor,
    w: Dict[String, ArcPointer[Tensor]], eps: Float32, ctx: DeviceContext,
) raises -> Tensor:
    var scale = 1.0 / Float32(DH) ** 0.5
    var q = _lin_nobias(x_seq, w, "cross_attn.q_proj.weight", ctx)       # [N,H*DH]
    var k = _lin_nobias(text_ctx, w, "cross_attn.k_proj.weight", ctx)    # [TXT,H*DH]
    var v = _lin_nobias(text_ctx, w, "cross_attn.v_proj.weight", ctx)    # [TXT,H*DH]
    var q3 = _to_nhd(q, N, H, DH, ctx)
    var k3 = _to_nhd(k, TXT, H, DH, ctx)
    var v3 = _to_nhd(v, TXT, H, DH, ctx)
    q3 = _qk_norm(q3, w["cross_attn.q_norm.weight"][], N, H, DH, eps, ctx)
    k3 = _qk_norm(k3, w["cross_attn.k_norm.weight"][], TXT, H, DH, eps, ctx)
    # per-head matmul attention (q-len N, kv-len TXT). NO RoPE.
    var qh = _permute_102(q3, N, H, DH, ctx)     # [H,N,DH]
    var kh = _permute_102(k3, TXT, H, DH, ctx)   # [H,TXT,DH]
    var vh = _permute_102(v3, TXT, H, DH, ctx)   # [H,TXT,DH]
    var out_parts = List[ArcPointer[Tensor]]()
    for h in range(H):
        var qh_h = _row(qh, h, N, DH, ctx)       # [N,DH]
        var kh_h = _row(kh, h, TXT, DH, ctx)     # [TXT,DH]
        var vh_h = _row(vh, h, TXT, DH, ctx)     # [TXT,DH]
        var scores = linear(qh_h, kh_h, None, ctx)   # [N,TXT] = q @ kᵀ
        scores = mul_scalar(scores, scale, ctx)
        var p = softmax_lastdim(scores, ctx)         # [N,TXT]
        var v_t = transpose(vh_h, 0, 1, ctx)         # [DH,TXT]
        var out_h = linear(p, v_t, None, ctx)        # [N,DH]
        out_parts.append(ArcPointer(out_h^))
    var stacked = _stack0(out_parts, H, N, DH, ctx)  # [H,N,DH]
    var sh = _permute_102(stacked, H, N, DH, ctx)    # [N,H,DH]
    var out2 = _from_nhd(sh, N, H * DH, ctx)         # [N,H*DH]
    return _lin_nobias(out2, w, "cross_attn.output_proj.weight", ctx)


# ── MLP (GPT2FeedForward): Linear -> GELU -> Linear ───────────────────────────
def cosmos_mlp(
    x_seq: Tensor, w: Dict[String, ArcPointer[Tensor]], ctx: DeviceContext
) raises -> Tensor:
    var h = _lin_nobias(x_seq, w, "mlp.layer1.weight", ctx)
    h = gelu(h, ctx)
    return _lin_nobias(h, w, "mlp.layer2.weight", ctx)


# ── Block forward (CHUNK A) ───────────────────────────────────────────────────
# x_seq_f32: [N, D] F32 residual stream. emb [Tp, D] BF16; adaln_lora [Tp, 3D] BF16;
# text_ctx [TXT, Cin] BF16. cos/sin [N, DH/2] BF16. Returns [N, D] F32.
def cosmos_block_forward[N: Int, TXT: Int, H: Int, DH: Int](
    var x_seq_f32: Tensor,
    emb: Tensor,
    adaln_lora: Tensor,
    text_ctx: Tensor,
    cos: Tensor,
    sin: Tensor,
    w: Dict[String, ArcPointer[Tensor]],
    cfg: CosmosConfig,
    tp: Int,
    hpwp: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    var D = cfg.model_channels
    var eps = cfg.eps
    var bf = STDtype.BF16

    # --- Self-attn ---
    var sa = _adaln_chunk3(emb, adaln_lora, w, "self_attn", tp, D, ctx)
    var sa_sh = _expand_t_to_tokens(sa[0], tp, hpwp, D, ctx)  # [N,D] F32
    var sa_sc = _expand_t_to_tokens(sa[1], tp, hpwp, D, ctx)
    var sa_ga = _expand_t_to_tokens(sa[2], tp, hpwp, D, ctx)
    var x_bf = cast_tensor(x_seq_f32, bf, ctx)
    var x_mod = _ln_modulate(x_bf, sa_sc, sa_sh, D, eps, ctx)
    var attn = cosmos_self_attention[N, H, DH](x_mod, cos, sin, w, eps, ctx)
    x_seq_f32 = _gated_add(x_seq_f32, attn, sa_ga, ctx)

    # --- Cross-attn ---
    var ca = _adaln_chunk3(emb, adaln_lora, w, "cross_attn", tp, D, ctx)
    var ca_sh = _expand_t_to_tokens(ca[0], tp, hpwp, D, ctx)
    var ca_sc = _expand_t_to_tokens(ca[1], tp, hpwp, D, ctx)
    var ca_ga = _expand_t_to_tokens(ca[2], tp, hpwp, D, ctx)
    x_bf = cast_tensor(x_seq_f32, bf, ctx)
    x_mod = _ln_modulate(x_bf, ca_sc, ca_sh, D, eps, ctx)
    var cross = cosmos_cross_attention[N, TXT, H, DH](x_mod, text_ctx, w, eps, ctx)
    x_seq_f32 = _gated_add(x_seq_f32, cross, ca_ga, ctx)

    # --- MLP ---
    var ml = _adaln_chunk3(emb, adaln_lora, w, "mlp", tp, D, ctx)
    var ml_sh = _expand_t_to_tokens(ml[0], tp, hpwp, D, ctx)
    var ml_sc = _expand_t_to_tokens(ml[1], tp, hpwp, D, ctx)
    var ml_ga = _expand_t_to_tokens(ml[2], tp, hpwp, D, ctx)
    x_bf = cast_tensor(x_seq_f32, bf, ctx)
    x_mod = _ln_modulate(x_bf, ml_sc, ml_sh, D, eps, ctx)
    var mlp_out = cosmos_mlp(x_mod, w, ctx)
    x_seq_f32 = _gated_add(x_seq_f32, mlp_out, ml_ga, ctx)

    return x_seq_f32^


# gated residual in F32: x_f32 + gate_f32 * y(bf16->f32).
def _gated_add(x_f32: Tensor, y: Tensor, gate_f32: Tensor, ctx: DeviceContext) raises -> Tensor:
    var y_f32 = cast_tensor(y, STDtype.F32, ctx)
    var gy = mul(gate_f32, y_f32, ctx)
    return add(x_f32, gy, ctx)


# ── reshape / permute helpers ─────────────────────────────────────────────────
def _to_nhd(x: Tensor, N: Int, H: Int, DH: Int, ctx: DeviceContext) raises -> Tensor:
    var shp = List[Int]()
    shp.append(N)
    shp.append(H)
    shp.append(DH)
    return reshape(x, shp^, ctx)


def _from_nhd(x: Tensor, N: Int, HD: Int, ctx: DeviceContext) raises -> Tensor:
    var shp = List[Int]()
    shp.append(N)
    shp.append(HD)
    return reshape(x, shp^, ctx)


def _to_bshd(x: Tensor, N: Int, H: Int, DH: Int, ctx: DeviceContext) raises -> Tensor:
    var shp = List[Int]()
    shp.append(1)
    shp.append(N)
    shp.append(H)
    shp.append(DH)
    return reshape(x, shp^, ctx)


def _from_bshd(x: Tensor, N: Int, HD: Int, ctx: DeviceContext) raises -> Tensor:
    var shp = List[Int]()
    shp.append(N)
    shp.append(HD)
    return reshape(x, shp^, ctx)


def _permute_102(x: Tensor, A: Int, B: Int, C: Int, ctx: DeviceContext) raises -> Tensor:
    var perm = List[Int]()
    perm.append(1)
    perm.append(0)
    perm.append(2)
    return permute(x, perm, ctx)


def _row(x: Tensor, r: Int, M: Int, K: Int, ctx: DeviceContext) raises -> Tensor:
    var part = slice(x, 0, r, 1, ctx)  # [1,M,K]
    var shp = List[Int]()
    shp.append(M)
    shp.append(K)
    return reshape(part, shp^, ctx)


def _stack0(parts: List[ArcPointer[Tensor]], H: Int, N: Int, DH: Int, ctx: DeviceContext) raises -> Tensor:
    var acc = _row_to_1md(parts[0][], N, DH, ctx)
    for h in range(1, H):
        var nxt = _row_to_1md(parts[h][], N, DH, ctx)
        acc = _ta_concat(0, ctx, acc, nxt)
    return acc^


def _row_to_1md(x: Tensor, N: Int, DH: Int, ctx: DeviceContext) raises -> Tensor:
    var shp = List[Int]()
    shp.append(1)
    shp.append(N)
    shp.append(DH)
    return reshape(x, shp^, ctx)


# Expand [rows, half] -> [rows*H, half] (each token row repeated H times). q4 is
# [1,N,H,DH]; rope_halfsplit flattens leading dims to rows=N*H in token-major-then-
# head order, so flat index (t*H+head) must map to token t's table row.
def _expand_rope_per_head(
    tbl: Tensor, N: Int, H: Int, half: Int, ctx: DeviceContext
) raises -> Tensor:
    var s3 = List[Int]()
    s3.append(N)
    s3.append(1)
    s3.append(half)
    var t3 = reshape(tbl, s3^, ctx)
    var n = N * H * half
    var zh = List[Float32]()
    for _ in range(n):
        zh.append(0.0)
    var zs = List[Int]()
    zs.append(N)
    zs.append(H)
    zs.append(half)
    var zeros = Tensor.from_host(zh^, zs^, tbl.dtype(), ctx)
    var bc = add(t3, zeros, ctx)  # [N,H,half]
    var os = List[Int]()
    os.append(N * H)
    os.append(half)
    return reshape(bc, os^, ctx)


# ── Full model (CHUNK B) ──────────────────────────────────────────────────────
struct CosmosPredict25Dit(Movable):
    var weights: Dict[String, ArcPointer[Tensor]]
    var config: CosmosConfig

    def __init__(out self, var weights: Dict[String, ArcPointer[Tensor]], config: CosmosConfig):
        self.weights = weights^
        self.config = config

    @staticmethod
    def load(dir: String, cfg: CosmosConfig, ctx: DeviceContext) raises -> CosmosPredict25Dit:
        var st = ShardedSafeTensors.open(dir)
        var w = Dict[String, ArcPointer[Tensor]]()
        var names = st.names()
        for nm in names:
            var key = String(nm)
            # skip rope buffers + training accumulators + _extra_state.
            if key.startswith("pos_embedder.") or key.startswith("accum_"):
                continue
            if key.endswith("._extra_state"):
                continue
            var tv = st.tensor_view(key)
            w[key] = ArcPointer(Tensor.from_view(tv, ctx))
        return CosmosPredict25Dit(weights=w^, config=cfg)

    def _w(self, name: String) raises -> ref [self.weights] Tensor:
        return self.weights[name][]

    # Block weights sub-dict (keys stripped of "blocks.i." prefix).
    def _block_weights(self, i: Int, ctx: DeviceContext) raises -> Dict[String, ArcPointer[Tensor]]:
        var prefix = String("blocks.") + String(i) + "."
        var bw = Dict[String, ArcPointer[Tensor]]()
        var suffixes = [
            "adaln_modulation_self_attn.1.weight", "adaln_modulation_self_attn.2.weight",
            "adaln_modulation_cross_attn.1.weight", "adaln_modulation_cross_attn.2.weight",
            "adaln_modulation_mlp.1.weight", "adaln_modulation_mlp.2.weight",
            "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
            "self_attn.output_proj.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight",
            "cross_attn.q_proj.weight", "cross_attn.k_proj.weight", "cross_attn.v_proj.weight",
            "cross_attn.output_proj.weight", "cross_attn.q_norm.weight", "cross_attn.k_norm.weight",
            "mlp.layer1.weight", "mlp.layer2.weight",
        ]
        for sfx in suffixes:
            var s = String(sfx)
            bw[s] = ArcPointer(self.weights[prefix + s][].clone(ctx))
        return bw^

    # forward for one sample.
    #   x_lat   : [in_channels=16, T, H, W] bf16 latent (single sample, no batch dim)
    #   timestep: Float32 scalar
    #   text_emb: [TXTRAW, crossattn_proj_in=100352] bf16 raw text embeddings
    # Comptime: TG/HG/WG latent T/H/W; N=Tp*Hp*Wp; TXTRAW raw text len; TXT==TXTRAW
    #   (text len after proj, used as kv-len); H/DH heads/head_dim.
    # cond_mask/padding_mask default to zeros (image mode).
    def forward[
        TG: Int, HG: Int, WG: Int, N: Int, TXTRAW: Int, TXT: Int, H: Int, DH: Int
    ](
        self, x_lat: Tensor, timestep: Float32, text_emb: Tensor, ctx: DeviceContext,
    ) raises -> Tensor:
        var cfg = self.config
        var D = cfg.model_channels
        var out_c = cfg.out_channels
        var ps = cfg.patch_spatial
        var pt = cfg.patch_temporal
        var bf = STDtype.BF16

        var tp = TG // pt
        var hp = HG // ps
        var wp = WG // ps
        var hpwp = hp * wp

        # ── 1. LVG concat: x = cat([x, zeros], C=0) -> [17, T, H, W] ──
        # (image-mode: condition_video_input_mask = zeros [1,T,H,W])
        var lvg_zeros = _zeros4(1, TG, HG, WG, bf, ctx)
        var x_lvg = _ta_concat(0, ctx, x_lat, lvg_zeros)  # [17,T,H,W]

        # ── 2. padding-mask concat: cat([x, zeros], C=0) -> [18, T, H, W] ──
        var pad_zeros = _zeros4(1, TG, HG, WG, bf, ctx)
        var x_in = _ta_concat(0, ctx, x_lvg, pad_zeros)   # [18,T,H,W]

        # ── 3. crossattn_proj: text 100352 -> 1024 (Linear + bias) ──
        var text_proj = linear(text_emb, self._w("crossattn_proj.0.weight"),
                               Optional(self._w("crossattn_proj.0.bias").clone(ctx)), ctx)  # [TXT,1024]
        var text_ctx = cast_tensor(text_proj, bf, ctx)

        # ── 4. patchify [18,T,H,W] -> [N, 18*pt*ps*ps=72]; x_embedder Linear ──
        var patched = patchify3d(x_in, pt, ps, ps, ctx)  # [N, 72]
        var x_seq = _lin_nobias_t(patched, self._w("x_embedder.proj.1.weight"), ctx)  # [N, D]

        # ── 5. timestep conditioning ──
        # sinusoidal (cos-first) [Tp, D] is emb; MLP linear1->silu->linear2[->6144]
        # is adaln_lora. t_embedding_norm = RMSNorm on emb only.
        var t_host = List[Float32]()
        for _ in range(tp):
            t_host.append(timestep)
        var t_shape = List[Int]()
        t_shape.append(tp)
        var t_vec = Tensor.from_host(t_host^, t_shape^, STDtype.F32, ctx)
        var sin_emb = timestep_embedding(t_vec, D, ctx, 10000.0)  # [Tp,D] F32, cos-first
        var sample = cast_tensor(sin_emb, bf, ctx)               # emb_B_T_D (use_adaln_lora)
        var h1 = linear(sample, self._w("t_embedder.1.linear_1.weight"), None, ctx)  # [Tp,D]
        h1 = silu(h1, ctx)
        var adaln_lora = linear(h1, self._w("t_embedder.1.linear_2.weight"), None, ctx)  # [Tp,3D]
        var emb = rms_norm(sample, self._w("t_embedding_norm.weight"), cfg.eps, ctx)   # [Tp,D]

        # ── 6. RoPE tables ONCE (bf16, half-split) ──
        var cs = cosmos_build_rope(tp, hp, wp, cfg, bf, ctx)

        # residual stream in F32.
        var x_f32 = cast_tensor(x_seq, STDtype.F32, ctx)  # [N,D] F32

        # ── 7. 28 transformer blocks ──
        for i in range(cfg.num_blocks):
            var bw = self._block_weights(i, ctx)
            x_f32 = cosmos_block_forward[N, TXT, H, DH](
                x_f32^, emb, adaln_lora, text_ctx, cs[0], cs[1], bw, cfg, tp, hpwp, ctx
            )

        # ── 8. FinalLayer: LN_no_affine + 2-chunk adaln + Linear[D->64] ──
        var x_final_bf = cast_tensor(x_f32, bf, ctx)
        var fh0 = silu(emb, ctx)
        var fh1 = linear(fh0, self._w("final_layer.adaln_modulation.1.weight"), None, ctx)  # [Tp,256]
        var fh2 = linear(fh1, self._w("final_layer.adaln_modulation.2.weight"), None, ctx)  # [Tp,2D]
        var adaln_2d = slice(adaln_lora, 1, 0, 2 * D, ctx)  # first 2D of [Tp,3D]
        var fsum = add(fh2, adaln_2d, ctx)
        var fsum_f32 = cast_tensor(fsum, STDtype.F32, ctx)
        var f_shift_td = slice(fsum_f32, 1, 0, D, ctx)
        var f_scale_td = slice(fsum_f32, 1, D, D, ctx)
        var f_shift = _expand_t_to_tokens(f_shift_td, tp, hpwp, D, ctx)  # [N,D]
        var f_scale = _expand_t_to_tokens(f_scale_td, tp, hpwp, D, ctx)
        var x_mod = _ln_modulate(x_final_bf, f_scale, f_shift, D, cfg.eps, ctx)
        var head_out = linear(x_mod, self._w("final_layer.linear.weight"), None, ctx)  # [N, 64]

        # ── 9. cosmos unpatchify (p1,p2,t',c) -> [out_c, T, H, W] ──
        return cosmos_unpatchify(head_out, tp, hp, wp, out_c, ps, pt, ctx)


# linear with a directly-borrowed weight (no Dict) — bias-free.
def _lin_nobias_t(x: Tensor, w: Tensor, ctx: DeviceContext) raises -> Tensor:
    return linear(x, w, None, ctx)


def _zeros4(C: Int, F: Int, H: Int, W: Int, dt: STDtype, ctx: DeviceContext) raises -> Tensor:
    var n = C * F * H * W
    var zh = List[Float32]()
    for _ in range(n):
        zh.append(0.0)
    var shp = List[Int]()
    shp.append(C)
    shp.append(F)
    shp.append(H)
    shp.append(W)
    return Tensor.from_host(zh^, shp^, dt, ctx)


# Cosmos unpatchify (MiniTrainDIT.unpatchify, cosmos_predict25_dit.rs:1588-1616):
#   [N=Tp*Hp*Wp, O=ps*ps*pt*out_c] -> reshape [Tp,Hp,Wp,p1,p2,t',c]
#   -> permute to [c, Tp, t', Hp, p1, Wp, p2] -> reshape [out_c, Tp*pt, Hp*ps, Wp*ps].
# NOTE: trailing patch dim decomposes as (p1=h, p2=w, t', c) with c SLOWEST — this
# is the ASYMMETRIC layout vs patchify (c,r,m,n); it is faithful to the Python
# (FinalLayer's output Linear is trained to this layout). Input here is [N,O] with
# the leading [Tp,Hp,Wp] folded into N (token order ti*Hp*Wp+hi*Wp+wi).
def cosmos_unpatchify(
    x_no: Tensor, tp: Int, hp: Int, wp: Int, out_c: Int, ps: Int, pt: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    # V2_2B has patch_temporal == 1, so t' is a size-1 axis and the full
    # rank-7 permute reduces to rank 6 (permute supports max rank 6). General
    # rank-7 pt>1 is not needed for V2_2B (reported if a future variant needs it).
    if pt != 1:
        raise Error("cosmos_unpatchify: pt>1 needs rank-7 permute (not in V2_2B)")
    # [N, O] -> [Tp, Hp, Wp, p1, p2, c]   (t' folded out, size 1)
    var s6 = List[Int]()
    s6.append(tp)
    s6.append(hp)
    s6.append(wp)
    s6.append(ps)   # p1 (spatial h)
    s6.append(ps)   # p2 (spatial w)
    s6.append(out_c)
    var x6 = reshape(x_no, s6^, ctx)
    # axes: 0=Tp 1=Hp 2=Wp 3=p1 4=p2 5=c
    # target "c Tp (Hp p1) (Wp p2)" -> [c, Tp, Hp, p1, Wp, p2] = axes [5,0,1,3,2,4]
    var perm = List[Int]()
    perm.append(5)
    perm.append(0)
    perm.append(1)
    perm.append(3)
    perm.append(2)
    perm.append(4)
    var xp = permute(x6, perm, ctx)
    # reshape [out_c, Tp, Hp*ps, Wp*ps]
    var of = List[Int]()
    of.append(out_c)
    of.append(tp)
    of.append(hp * ps)
    of.append(wp * ps)
    return reshape(xp, of^, ctx)

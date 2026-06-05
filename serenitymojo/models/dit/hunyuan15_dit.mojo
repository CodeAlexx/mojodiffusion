# models/dit/hunyuan15_dit.mojo — HunyuanVideo-1.5 DiT (video transformer),
# pure Mojo + MAX. Inference-only, GPU-only.
#
# Reference (read LINE BY LINE, READ-ONLY):
#   /home/alex/EriDiffusion/inference-flame/src/models/hunyuan15_dit.rs
#   /home/alex/HunyuanVideo-1.5 (Python code repo) — architecture cross-check.
#
# Variant gated here: HunyuanVideo-1.5 480p T2V (Hunyuan15Config.default()):
#   num_double_blocks=54 (NO single-stream blocks), hidden_size=2048,
#   num_heads=16, head_dim=128, in_channels=out_channels=32, patch_size=(1,1,1),
#   rope_dim_list=[16,56,56] (sums to head_dim=128), rope_theta=256.0,
#   mlp_ratio=4.0, text_dim=3584 (text already refined to 2048 by Python Stage-1),
#   eps=1e-6.
#
# ── Block architecture: MMDiT-style DOUBLE-STREAM (hunyuan15_dit.rs:416-549) ──
# This is a FLUX/Klein-family double-stream block: separate img & txt streams,
# JOINT attention (concat [img,txt] along sequence -> SDPA -> split). There are
# NO single-stream blocks and NO cross-attention.
# Modulation is PER-VEC (one scalar timestep -> one [6*dim] modulation vector,
# broadcast over ALL tokens) — UNLIKE wan22 whose modulation is per-token.
#
# Per block, for each of {img, txt} the modulation vec is:
#   mods = Linear(SiLU(vec))[6*dim] -> chunk(6) = shift1,scale1,gate1,
#                                                  shift2,scale2,gate2
# img attn path:
#   x_n  = LN_no_affine(img)*(1+scale1) + shift1                 (AdaLN, F32)
#   q,k,v = Linear(x_n)  -> reshape [1,S,H,Dh]
#   q = RMSNorm(q, img_attn_q_norm); k = RMSNorm(k, img_attn_k_norm)  (per head)
#   q,k = halfsplit-RoPE(q,k)   (img tokens only; txt gets NO RoPE)
# txt attn path: mirror, but NO RoPE on q/k.
# joint attention: cat q=[img_q;txt_q] over seq, same k,v -> SDPA (no mask) ->
#   split back into img_attn (first img_len) and txt_attn (last txt_len).
# img gated residual + MLP:
#   img = img + Linear(img_attn, img_attn_proj) * gate1
#   y   = LN_no_affine(img)*(1+scale2)+shift2
#   y   = Linear(fc2, GELU_tanh(Linear(fc1, y)))
#   img = img + y * gate2
# txt mirror.
#
# Final layer (hunyuan15_dit.rs:387-409): AdaLN(shift,scale)=chunk2(Linear(SiLU(vec)));
#   out = Linear( LN_no_affine(img)*(1+scale)+shift ).
#
# DTYPE: bf16 weights+activations, F32-accumulate (matches the bf16-GPU oracle).
# The AdaLN (1+scale)*x+shift and gated residuals run in F32 then cast to bf16.
#
# REUSE (do NOT reimplement): ops/linear.linear, ops/norm.{rms_norm,layer_norm},
# ops/attention.sdpa_nomask, ops/rope.rope_halfsplit,
# ops/rope_tables.build_multiaxis_rope_tables, ops/activations.{silu,gelu},
# ops/embeddings.timestep_embedding, ops/cast.cast_tensor,
# ops/patchify3d.{patchify3d,unpatchify3d},
# ops/tensor_algebra.{add,mul,add_scalar,slice,reshape,permute,transpose,concat}.
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import rms_norm, layer_norm
from serenitymojo.ops.attention import sdpa_nomask
from serenitymojo.ops.rope import rope_halfsplit
from serenitymojo.ops.rope_tables import build_multiaxis_rope_tables
from serenitymojo.ops.activations import silu, gelu
from serenitymojo.ops.embeddings import timestep_embedding
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.patchify3d import patchify3d, unpatchify3d
from serenitymojo.ops.tensor_algebra import (
    add, mul, add_scalar, slice, reshape, permute, transpose,
    concat as _ta_concat,
)


# ── Config ──────────────────────────────────────────────────────────────────
@fieldwise_init
struct Hunyuan15Config(Copyable, Movable, ImplicitlyCopyable):
    var num_double_blocks: Int  # 54
    var hidden_size: Int        # 2048
    var num_heads: Int          # 16
    var head_dim: Int           # 128
    var in_channels: Int        # 32 (forward input pre-concat is 65)
    var out_channels: Int       # 32
    var patch_f: Int            # 1
    var patch_h: Int            # 1
    var patch_w: Int            # 1
    var rope_d0: Int            # 16
    var rope_d1: Int            # 56
    var rope_d2: Int            # 56
    var rope_theta: Float32     # 256.0
    var text_dim: Int           # 3584
    var eps: Float32            # 1e-6

    @staticmethod
    def default() -> Hunyuan15Config:
        return Hunyuan15Config(
            num_double_blocks=54, hidden_size=2048, num_heads=16, head_dim=128,
            in_channels=32, out_channels=32, patch_f=1, patch_h=1, patch_w=1,
            rope_d0=16, rope_d1=56, rope_d2=56, rope_theta=256.0, text_dim=3584,
            eps=1.0e-6,
        )

    def rope_axes(self) -> List[Int]:
        # rope_dim_list = [16, 56, 56]; sum == head_dim (128); halves sum to 64.
        var out = List[Int]()
        out.append(self.rope_d0)
        out.append(self.rope_d1)
        out.append(self.rope_d2)
        return out^


# ── RoPE table build (build_rope, hunyuan15_dit.rs:192-237) ──────────────────
# Token order F-major then H then W (token_idx = ti*th*tw + hi*tw + wi), matching
# patchify3d. Per-axis half-dims [8,28,28] concatenated -> [seq, 64] tables.
# Feed straight into rope_halfsplit (the Rust apply_rope uses rope_halfsplit_bf16).
def hunyuan15_rope_positions(
    tt: Int, th: Int, tw: Int, ctx: DeviceContext
) raises -> Tensor:
    var rows = tt * th * tw
    var host = List[Float32]()
    for ti in range(tt):
        for hi in range(th):
            for wi in range(tw):
                host.append(Float32(ti))
                host.append(Float32(hi))
                host.append(Float32(wi))
    var shp = List[Int]()
    shp.append(rows * 3)
    return Tensor.from_host(host^, shp^, STDtype.F32, ctx)


# Returns (cos, sin) each [img_seq, head_dim/2] in dtype `dt`.
def hunyuan15_build_rope(
    tt: Int, th: Int, tw: Int, cfg: Hunyuan15Config, dt: STDtype,
    ctx: DeviceContext,
) raises -> Tuple[Tensor, Tensor]:
    var positions = hunyuan15_rope_positions(tt, th, tw, ctx)
    var axes = cfg.rope_axes()
    return build_multiaxis_rope_tables(positions, axes, cfg.rope_theta, ctx, dt)


# ── linear(x, w[wname], bias=w[bname]) helper (clone bias into Optional) ──────
def _lin(
    x: Tensor, w: Dict[String, ArcPointer[Tensor]], wname: String, bname: String,
    ctx: DeviceContext,
) raises -> Tensor:
    return linear(x, w[wname][], Optional(w[bname][].clone(ctx)), ctx)


# ── LN-no-affine: layer_norm with weight=1, bias=0 (matches layer_norm_no_affine) ──
def _ln_no_affine(x: Tensor, dim: Int, eps: Float32, ctx: DeviceContext) raises -> Tensor:
    var ones_h = List[Float32]()
    var zeros_h = List[Float32]()
    for _ in range(dim):
        ones_h.append(1.0)
        zeros_h.append(0.0)
    var ws = List[Int]()
    ws.append(dim)
    var ws2 = List[Int]()
    ws2.append(dim)
    var ones = Tensor.from_host(ones_h^, ws^, x.dtype(), ctx)
    var zeros = Tensor.from_host(zeros_h^, ws2^, x.dtype(), ctx)
    return layer_norm(x, ones, zeros, eps, ctx)


# ── AdaLN: out(bf16) = LN_no_affine(x)*(1+scale) + shift ─────────────────────
# scale/shift are [1,dim] F32 (per-vec, broadcast over S). x is [1,S,dim] bf16.
def _modulate(
    x: Tensor, scale_f32: Tensor, shift_f32: Tensor, dim: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    var normed = _ln_no_affine(x, dim, eps, ctx)
    var normed_f32 = cast_tensor(normed, STDtype.F32, ctx)
    # scale/shift [1,dim] -> [1,1,dim] broadcast over S.
    var sc_shape = List[Int]()
    sc_shape.append(1)
    sc_shape.append(1)
    sc_shape.append(dim)
    var sc1 = add_scalar(reshape(scale_f32, sc_shape^, ctx), 1.0, ctx)
    var sh_shape = List[Int]()
    sh_shape.append(1)
    sh_shape.append(1)
    sh_shape.append(dim)
    var sh = reshape(shift_f32, sh_shape^, ctx)
    var prod = mul(normed_f32, sc1, ctx)
    var out_f32 = add(prod, sh, ctx)
    return cast_tensor(out_f32, x.dtype(), ctx)


# ── gated residual: out(bf16) = x + y * gate ; gate [1,dim] F32 broadcast ─────
def _gated_residual(
    x: Tensor, y: Tensor, gate_f32: Tensor, dim: Int, ctx: DeviceContext
) raises -> Tensor:
    var g_shape = List[Int]()
    g_shape.append(1)
    g_shape.append(1)
    g_shape.append(dim)
    var gate = reshape(gate_f32, g_shape^, ctx)
    var x_f32 = cast_tensor(x, STDtype.F32, ctx)
    var y_f32 = cast_tensor(y, STDtype.F32, ctx)
    var gy = mul(y_f32, gate, ctx)
    var res = add(x_f32, gy, ctx)
    return cast_tensor(res, x.dtype(), ctx)


# chunk(N, dim=1): index m of a [1, N*dim] tensor -> [1, dim] (F32).
def _chunk(mods: Tensor, m: Int, dim: Int, ctx: DeviceContext) raises -> Tensor:
    return slice(mods, 1, m * dim, dim, ctx)


# Expand a [S, half] RoPE table to [S*H, half] by repeating each token row H
# times CONTIGUOUSLY (token-major-then-head), so the flat index (t*H + head) in
# the BSHD x maps to token t's table row. (Same convention as wan22.)
def _expand_rope_per_head(
    tbl: Tensor, S: Int, H: Int, half: Int, ctx: DeviceContext
) raises -> Tensor:
    var t3_shape = List[Int]()
    t3_shape.append(S)
    t3_shape.append(1)
    t3_shape.append(half)
    var t3 = reshape(tbl, t3_shape^, ctx)  # [S,1,half]
    var n = S * H * half
    var zh = List[Float32]()
    for _ in range(n):
        zh.append(0.0)
    var zshape = List[Int]()
    zshape.append(S)
    zshape.append(H)
    zshape.append(half)
    var zeros = Tensor.from_host(zh^, zshape^, tbl.dtype(), ctx)
    var bc = add(t3, zeros, ctx)  # [S,H,half] (broadcast t3 over H)
    var out_shape = List[Int]()
    out_shape.append(S * H)
    out_shape.append(half)
    return reshape(bc, out_shape^, ctx)


# Apply halfsplit RoPE to a [1,S,H,Dh] BSHD tensor (img tokens). cos/sin are the
# already-per-head-expanded [S*H, Dh/2] tables. rope_halfsplit flattens x to
# rows = S*H (token-major-then-head) and pairs (x[i], x[i+Dh/2]).
def _apply_rope_bshd(
    x: Tensor, cos_e: Tensor, sin_e: Tensor, ctx: DeviceContext
) raises -> Tensor:
    return rope_halfsplit(x, cos_e, sin_e, ctx)


# ── Double-stream block (double_block_forward, hunyuan15_dit.rs:416-549) ──────
# S_IMG / S_TXT / H / DH are comptime (sdpa_nomask specializes on S=S_IMG+S_TXT).
struct Hunyuan15DoubleOut(Copyable, Movable):
    var img: ArcPointer[Tensor]
    var txt: ArcPointer[Tensor]

    def __init__(out self, var img: Tensor, var txt: Tensor):
        self.img = ArcPointer(img^)
        self.txt = ArcPointer(txt^)


def hunyuan15_double_block[
    S_IMG: Int, S_TXT: Int, H: Int, DH: Int
](
    img: Tensor,        # [1, S_IMG, dim] bf16
    txt: Tensor,        # [1, S_TXT, dim] bf16
    vec: Tensor,        # [1, dim] bf16 (timestep MLP output)
    cos_e: Tensor,      # [S_IMG*H, DH/2] (per-head-expanded img RoPE)
    sin_e: Tensor,
    w: Dict[String, ArcPointer[Tensor]],
    cfg: Hunyuan15Config,
    ctx: DeviceContext,
) raises -> Hunyuan15DoubleOut:
    comptime S = S_IMG + S_TXT
    var dim = cfg.hidden_size
    var eps = cfg.eps
    var scale = 1.0 / Float32(DH) ** 0.5

    # ── img_mod / txt_mod: Linear(SiLU(vec)) -> chunk(6) (F32) ──
    var vec_silu = silu(vec, ctx)
    var img_mods = cast_tensor(
        _lin(vec_silu, w, "img_mod.linear.weight", "img_mod.linear.bias", ctx),
        STDtype.F32, ctx,
    )  # [1, 6*dim]
    var txt_mods = cast_tensor(
        _lin(vec_silu, w, "txt_mod.linear.weight", "txt_mod.linear.bias", ctx),
        STDtype.F32, ctx,
    )

    var img_s1 = _chunk(img_mods, 0, dim, ctx)
    var img_sc1 = _chunk(img_mods, 1, dim, ctx)
    var img_g1 = _chunk(img_mods, 2, dim, ctx)
    var img_s2 = _chunk(img_mods, 3, dim, ctx)
    var img_sc2 = _chunk(img_mods, 4, dim, ctx)
    var img_g2 = _chunk(img_mods, 5, dim, ctx)

    var txt_s1 = _chunk(txt_mods, 0, dim, ctx)
    var txt_sc1 = _chunk(txt_mods, 1, dim, ctx)
    var txt_g1 = _chunk(txt_mods, 2, dim, ctx)
    var txt_s2 = _chunk(txt_mods, 3, dim, ctx)
    var txt_sc2 = _chunk(txt_mods, 4, dim, ctx)
    var txt_g2 = _chunk(txt_mods, 5, dim, ctx)

    # ── img norm + modulate -> Q/K/V ──
    var img_mod_in = _modulate(img, img_sc1, img_s1, dim, eps, ctx)
    var img_q = _lin(img_mod_in, w, "img_attn_q.weight", "img_attn_q.bias", ctx)
    var img_k = _lin(img_mod_in, w, "img_attn_k.weight", "img_attn_k.bias", ctx)
    var img_v = _lin(img_mod_in, w, "img_attn_v.weight", "img_attn_v.bias", ctx)
    img_q = _to_bshd(img_q, S_IMG, H, DH, ctx)
    img_k = _to_bshd(img_k, S_IMG, H, DH, ctx)
    img_v = _to_bshd(img_v, S_IMG, H, DH, ctx)
    # QK RMSNorm per head (last dim = DH).
    img_q = rms_norm(img_q, w["img_attn_q_norm.weight"][], eps, ctx)
    img_k = rms_norm(img_k, w["img_attn_k_norm.weight"][], eps, ctx)
    # RoPE on img Q/K (halfsplit).
    img_q = _apply_rope_bshd(img_q, cos_e, sin_e, ctx)
    img_k = _apply_rope_bshd(img_k, cos_e, sin_e, ctx)

    # ── txt norm + modulate -> Q/K/V (NO RoPE) ──
    var txt_mod_in = _modulate(txt, txt_sc1, txt_s1, dim, eps, ctx)
    var txt_q = _lin(txt_mod_in, w, "txt_attn_q.weight", "txt_attn_q.bias", ctx)
    var txt_k = _lin(txt_mod_in, w, "txt_attn_k.weight", "txt_attn_k.bias", ctx)
    var txt_v = _lin(txt_mod_in, w, "txt_attn_v.weight", "txt_attn_v.bias", ctx)
    txt_q = _to_bshd(txt_q, S_TXT, H, DH, ctx)
    txt_k = _to_bshd(txt_k, S_TXT, H, DH, ctx)
    txt_v = _to_bshd(txt_v, S_TXT, H, DH, ctx)
    txt_q = rms_norm(txt_q, w["txt_attn_q_norm.weight"][], eps, ctx)
    txt_k = rms_norm(txt_k, w["txt_attn_k_norm.weight"][], eps, ctx)

    # ── Joint attention: concat [img, txt] over sequence (dim 1 of BSHD) ──
    var q = _ta_concat(1, ctx, img_q, txt_q)  # [1, S, H, DH]
    var k = _ta_concat(1, ctx, img_k, txt_k)
    var v = _ta_concat(1, ctx, img_v, txt_v)
    var attn = sdpa_nomask[1, S, H, DH](q, k, v, scale, ctx)  # [1, S, H, DH]

    # Split back: img = first S_IMG, txt = last S_TXT (along seq dim 1).
    var img_attn4 = slice(attn, 1, 0, S_IMG, ctx)       # [1,S_IMG,H,DH]
    var txt_attn4 = slice(attn, 1, S_IMG, S_TXT, ctx)   # [1,S_TXT,H,DH]
    var img_attn = _from_bshd(img_attn4, S_IMG, dim, ctx)
    var txt_attn = _from_bshd(txt_attn4, S_TXT, dim, ctx)

    # ── img gated residual + MLP ──
    var img_proj = _lin(img_attn, w, "img_attn_proj.weight", "img_attn_proj.bias", ctx)
    var img1 = _gated_residual(img, img_proj, img_g1, dim, ctx)
    var img_mlp_in = _modulate(img1, img_sc2, img_s2, dim, eps, ctx)
    var img_mlp = _lin(img_mlp_in, w, "img_mlp.fc1.weight", "img_mlp.fc1.bias", ctx)
    img_mlp = gelu(img_mlp, ctx)
    img_mlp = _lin(img_mlp, w, "img_mlp.fc2.weight", "img_mlp.fc2.bias", ctx)
    var img_out = _gated_residual(img1, img_mlp, img_g2, dim, ctx)

    # ── txt gated residual + MLP ──
    var txt_proj = _lin(txt_attn, w, "txt_attn_proj.weight", "txt_attn_proj.bias", ctx)
    var txt1 = _gated_residual(txt, txt_proj, txt_g1, dim, ctx)
    var txt_mlp_in = _modulate(txt1, txt_sc2, txt_s2, dim, eps, ctx)
    var txt_mlp = _lin(txt_mlp_in, w, "txt_mlp.fc1.weight", "txt_mlp.fc1.bias", ctx)
    txt_mlp = gelu(txt_mlp, ctx)
    txt_mlp = _lin(txt_mlp, w, "txt_mlp.fc2.weight", "txt_mlp.fc2.bias", ctx)
    var txt_out = _gated_residual(txt1, txt_mlp, txt_g2, dim, ctx)

    return Hunyuan15DoubleOut(img_out^, txt_out^)


# ── helpers ───────────────────────────────────────────────────────────────────
# [1,S,dim] -> [1,S,H,DH]
def _to_bshd(x: Tensor, S: Int, H: Int, DH: Int, ctx: DeviceContext) raises -> Tensor:
    var shp = List[Int]()
    shp.append(1)
    shp.append(S)
    shp.append(H)
    shp.append(DH)
    return reshape(x, shp^, ctx)


# [1,S,H,DH] -> [1,S,dim]
def _from_bshd(x: Tensor, S: Int, dim: Int, ctx: DeviceContext) raises -> Tensor:
    var shp = List[Int]()
    shp.append(1)
    shp.append(S)
    shp.append(dim)
    return reshape(x, shp^, ctx)


# ── Full-stack forward (CHUNK B) ────────────────────────────────────────────
# Hunyuan15Dit: loads ALL weights resident and runs the complete double-stream
# forward for a single sample on a comptime grid. Reference: hunyuan15_dit.rs
# forward (309-410). Text is assumed already refined to hidden_size (Python
# Stage-1 SingleTokenRefiner), so txt_embeds is [1, S_TXT, hidden_size].
#
# Comptime params: TT/TH/TW = patched grid (== latent F/H/W since patch=(1,1,1));
# S_IMG = TT*TH*TW; S_TXT = text token count; H/DH = heads/head_dim.
#
# forward signature (single sample):
#   hidden_states: [1, C_in_concat, F, H, W] bf16 (already concat cond+mask = 65ch)
#   timestep:      Float32 scalar
#   txt_embeds:    [1, S_TXT, hidden_size] bf16 (pre-refined)
# returns          [1, out_channels, F, H, W] bf16 (velocity)

struct Hunyuan15Dit(Movable):
    var weights: Dict[String, ArcPointer[Tensor]]
    var config: Hunyuan15Config

    def __init__(out self, var weights: Dict[String, ArcPointer[Tensor]], config: Hunyuan15Config):
        self.weights = weights^
        self.config = config

    @staticmethod
    def load(dir: String, cfg: Hunyuan15Config, ctx: DeviceContext) raises -> Hunyuan15Dit:
        var st = ShardedSafeTensors.open(dir)
        var w = Dict[String, ArcPointer[Tensor]]()
        var names = st.names()
        for nm in names:
            var key = String(nm)
            var tv = st.tensor_view(key)
            w[key] = ArcPointer(Tensor.from_view(tv, ctx))
        return Hunyuan15Dit(weights=w^, config=cfg)

    def _w(self, name: String) raises -> ref [self.weights] Tensor:
        return self.weights[name][]

    # Block weights sub-dict for block i (keys stripped of "double_blocks.i.").
    def _block_weights(self, i: Int, ctx: DeviceContext) raises -> Dict[String, ArcPointer[Tensor]]:
        var prefix = String("double_blocks.") + String(i) + "."
        var bw = Dict[String, ArcPointer[Tensor]]()
        var suffixes = [
            "img_mod.linear.weight", "img_mod.linear.bias",
            "img_attn_q.weight", "img_attn_q.bias",
            "img_attn_k.weight", "img_attn_k.bias",
            "img_attn_v.weight", "img_attn_v.bias",
            "img_attn_q_norm.weight", "img_attn_k_norm.weight",
            "img_attn_proj.weight", "img_attn_proj.bias",
            "img_mlp.fc1.weight", "img_mlp.fc1.bias",
            "img_mlp.fc2.weight", "img_mlp.fc2.bias",
            "txt_mod.linear.weight", "txt_mod.linear.bias",
            "txt_attn_q.weight", "txt_attn_q.bias",
            "txt_attn_k.weight", "txt_attn_k.bias",
            "txt_attn_v.weight", "txt_attn_v.bias",
            "txt_attn_q_norm.weight", "txt_attn_k_norm.weight",
            "txt_attn_proj.weight", "txt_attn_proj.bias",
            "txt_mlp.fc1.weight", "txt_mlp.fc1.bias",
            "txt_mlp.fc2.weight", "txt_mlp.fc2.bias",
        ]
        for sfx in suffixes:
            var s = String(sfx)
            bw[s] = ArcPointer(self.weights[prefix + s][].clone(ctx))
        return bw^

    def forward[
        TT: Int, TH: Int, TW: Int, S_IMG: Int, S_TXT: Int, H: Int, DH: Int
    ](
        self, hidden_states: Tensor, timestep: Float32, txt_embeds: Tensor,
        ctx: DeviceContext,
    ) raises -> Tensor:
        var cfg = self.config
        var dim = cfg.hidden_size
        var bf = hidden_states.dtype()

        # ── img_in: Conv3d(C_in, dim, k=1, s=1) == patchify3d(p=1)+linear ──
        # hidden_states [1, C_in, F, H, W] -> drop batch -> [C_in, F, H, W].
        var hs_shape = List[Int]()
        hs_shape.append(hidden_states.shape()[1])  # C_in (e.g. 65)
        hs_shape.append(hidden_states.shape()[2])  # F
        hs_shape.append(hidden_states.shape()[3])  # H
        hs_shape.append(hidden_states.shape()[4])  # W
        var c_in = hidden_states.shape()[1]
        var hs_nobatch = reshape(hidden_states, hs_shape^, ctx)
        # patchify3d with patch=(1,1,1) -> [S_IMG, C_in*1*1*1] = [S_IMG, C_in].
        var patched = patchify3d(hs_nobatch, cfg.patch_f, cfg.patch_h, cfg.patch_w, ctx)
        var patch_dim = c_in * cfg.patch_f * cfg.patch_h * cfg.patch_w
        var pe_w_shape = List[Int]()
        pe_w_shape.append(dim)
        pe_w_shape.append(patch_dim)
        var pe_w = reshape(self._w("img_in.proj.weight"), pe_w_shape^, ctx)
        var img2d = linear(patched, pe_w,
                           Optional(self._w("img_in.proj.bias").clone(ctx)), ctx)  # [S_IMG, dim]
        var img_shape = List[Int]()
        img_shape.append(1)
        img_shape.append(S_IMG)
        img_shape.append(dim)
        var img = reshape(img2d, img_shape^, ctx)  # [1, S_IMG, dim]

        # ── time_in: timestep_embedding(256) -> Linear -> SiLU -> Linear ──
        var t_host = List[Float32]()
        t_host.append(timestep)
        var t_shape = List[Int]()
        t_shape.append(1)
        var t_vec = Tensor.from_host(t_host^, t_shape^, STDtype.F32, ctx)
        # freq_dim=256. NOTE: Rust uses cos-first/sin-second sinusoidal embedding
        # (timestep_embedding here is COS-first, matching).
        var t_emb_bf = timestep_embedding(t_vec, 256, ctx, 10000.0, bf)
        var vec = linear(t_emb_bf, self._w("time_in.mlp.0.weight"),
                         Optional(self._w("time_in.mlp.0.bias").clone(ctx)), ctx)
        vec = silu(vec, ctx)
        vec = linear(vec, self._w("time_in.mlp.2.weight"),
                     Optional(self._w("time_in.mlp.2.bias").clone(ctx)), ctx)  # [1, dim] bf16

        # ── txt already refined (Python Stage-1): [1, S_TXT, dim] ──
        var txt = txt_embeds.clone(ctx)

        # ── RoPE tables (bf16, halfsplit) for the (TT,TH,TW) grid, per-head ──
        var cs = hunyuan15_build_rope(TT, TH, TW, cfg, bf, ctx)
        var cos_e = _expand_rope_per_head(cs[0], S_IMG, H, DH // 2, ctx)
        var sin_e = _expand_rope_per_head(cs[1], S_IMG, H, DH // 2, ctx)

        # ── 54 double-stream blocks ──
        var img_s = ArcPointer(img^)
        var txt_s = ArcPointer(txt^)
        for i in range(cfg.num_double_blocks):
            var bw = self._block_weights(i, ctx)
            var pair = hunyuan15_double_block[S_IMG, S_TXT, H, DH](
                img_s[], txt_s[], vec, cos_e, sin_e, bw, cfg, ctx,
            )
            img_s = pair.img
            txt_s = pair.txt

        # ── Final layer: AdaLN(shift,scale)=chunk2(Linear(SiLU(vec))) -> Linear ──
        var fl_silu = silu(vec, ctx)
        var fl_mods = cast_tensor(
            linear(fl_silu, self._w("final_layer.adaLN_modulation.1.weight"),
                   Optional(self._w("final_layer.adaLN_modulation.1.bias").clone(ctx)), ctx),
            STDtype.F32, ctx,
        )  # [1, 2*dim]
        var fl_shift = _chunk(fl_mods, 0, dim, ctx)
        var fl_scale = _chunk(fl_mods, 1, dim, ctx)
        var fl_in = _modulate(img_s[], fl_scale, fl_shift, dim, cfg.eps, ctx)
        var out_tokens = linear(fl_in, self._w("final_layer.linear.weight"),
                                Optional(self._w("final_layer.linear.bias").clone(ctx)), ctx)
        # out_tokens: [1, S_IMG, out_channels*pf*ph*pw]

        # ── Unpatchify: [S_IMG, C_out*pf*ph*pw] -> [C_out, F, H, W] ──
        var ot2_shape = List[Int]()
        ot2_shape.append(S_IMG)
        ot2_shape.append(cfg.out_channels * cfg.patch_f * cfg.patch_h * cfg.patch_w)
        var out2d = reshape(out_tokens, ot2_shape^, ctx)
        return unpatchify3d(
            out2d, cfg.out_channels,
            TT * cfg.patch_f, TH * cfg.patch_h, TW * cfg.patch_w,
            cfg.patch_f, cfg.patch_h, cfg.patch_w, ctx,
        )

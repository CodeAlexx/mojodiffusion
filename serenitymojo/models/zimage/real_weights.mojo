# models/zimage/real_weights.mojo — REAL-weight embedder / modulation / rope /
# final-layer builders for the Z-Image (NextDiT) LoRA trainer.
#
# The verified Z-Image LoRA stack (zimage_stack_lora.mojo) is "deferred-embedder"
# by design (see zimage_stack.mojo header §SCOPE): it consumes PRECOMPUTED
# post-embedder tokens (x_seq, cap_seq), PER-BLOCK RAW modulation vectors
# (nr_mod[], main_mod[]), the final-layer f_scale, rope tables, and the final
# linear weights — and returns the grads into x_seq/cap_seq + the RAW mod-vec
# grads (the train loop backprops those into the embedders / adaLN MLPs at the
# step boundary; this milestone trains the in-block LoRA, so the embedder/adaLN
# weights stay FROZEN and we only need their FORWARD here).
#
# This module TRANSLATES the host math of models/dit/zimage_dit.mojo (the
# parity-verified inference oracle) line-for-line:
#   * _t_embedder      : sinusoidal(256, t*1000) -> mlp.0 -> silu -> mlp.2  [1,256]
#   * adaLN per block  : Linear(adaln) [4D] (RAW; block applies tanh/+1 itself)
#   * final f_scale    : diffusers uses 1 + Linear(silu(adaln)); our shared
#                        modulate() op applies the +1 internally, so pass raw.
#   * cap_embedder     : RMSNorm(2560) -> Linear -> [CAPLEN, D]
#   * x_embedder       : patchify(channel-minor) -> Linear -> [IMG_TOK, D]
#   * rope             : 3-axis (theta=256, axes 32/48/48) interleaved [S*H, Dh/2]
#
# All builders run on the GPU ops (linear / rms_norm / silu) then return host
# List[Float32] (the stack's carrier) or device tensors (rope / final_lin).
#
# Weight source: the DIFFUSERS transformer dir
#   /home/alex/.serenity/models/zimage_base/transformer
# (UNFUSED to_q/to_k/to_v + all_x_embedder.2-1 + all_final_layer.2-1). Confirmed
# 2026-06-01: the single-file z_image_base_bf16.safetensors is the FUSED comfy
# layout and is NOT used.
#
# Mojo 0.26.x+: def not fn; Tensor move-only; host List[Float32] carriers.

from std.collections import List, Optional
from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from std.math import exp as fexp, log as flog, cos as fcos, sin as fsin
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import rms_norm
from serenitymojo.ops.activations import silu
from serenitymojo.ops.tensor_algebra import reshape, permute, reshape_owned
from serenitymojo.models.zimage.weights import _load_device_preserve, _load_f32_device
from serenitymojo.models.zimage.block import ZImageModVecs


comptime TArc = ArcPointer[Tensor]


# ── small host helpers ───────────────────────────────────────────────────────
def _ones(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n):
        o.append(Float32(1.0))
    return o^


# Slice a host list [rows, cols] -> [rows, col0:col0+w] flat.
def _slice_cols(v: List[Float32], rows: Int, cols: Int, col0: Int, w: Int) -> List[Float32]:
    var o = List[Float32]()
    for r in range(rows):
        for c in range(w):
            o.append(v[r * cols + col0 + c])
    return o^


# ─────────────────────────────────────────────────────────────────────────────
# Holder for the frozen embedder / modulation / final-layer weights. Loaded ONCE.
# Builder-owned biased linears cast transient inputs to checkpoint weight dtype.
# The final projection remains F32 until the stack forward's host-F32 biased
# helper is converted outside this loader scope.
# ─────────────────────────────────────────────────────────────────────────────
struct ZImageRealAux(Movable):
    # t_embedder mlp
    var t_w0: TArc          # [1024, 256]
    var t_b0: TArc          # [1024]
    var t_w2: TArc          # [256, 1024]
    var t_b2: TArc          # [256]
    # cap_embedder
    var cap_norm: TArc      # [2560]
    var cap_lin_w: TArc     # [D, 2560]
    var cap_lin_b: TArc     # [D]
    # x_embedder
    var x_w: TArc           # [D, 64]
    var x_b: TArc           # [D]
    var x_pad_token: TArc   # [1, D]
    var cap_pad_token: TArc # [1, D]
    # per-block adaLN_modulation.0 (RAW [4D] = scale_msa|gate_msa|scale_mlp|gate_mlp)
    var nr_mod_w: List[TArc]    # num_nr   x [4D, 256]
    var nr_mod_b: List[TArc]    # num_nr   x [4D]
    var main_mod_w: List[TArc]  # num_main x [4D, 256]
    var main_mod_b: List[TArc]
    # final layer adaLN (SiLU -> Linear) + linear
    var final_mod_w: TArc   # [D, 256]
    var final_mod_b: TArc   # [D]
    var final_lin_w: TArc   # [out_ch=64, D]
    var final_lin_b: TArc   # [64]

    def __init__(
        out self,
        var t_w0: TArc, var t_b0: TArc, var t_w2: TArc, var t_b2: TArc,
        var cap_norm: TArc, var cap_lin_w: TArc, var cap_lin_b: TArc,
        var x_w: TArc, var x_b: TArc, var x_pad_token: TArc, var cap_pad_token: TArc,
        var nr_mod_w: List[TArc], var nr_mod_b: List[TArc],
        var main_mod_w: List[TArc], var main_mod_b: List[TArc],
        var final_mod_w: TArc, var final_mod_b: TArc,
        var final_lin_w: TArc, var final_lin_b: TArc,
    ):
        self.t_w0 = t_w0^
        self.t_b0 = t_b0^
        self.t_w2 = t_w2^
        self.t_b2 = t_b2^
        self.cap_norm = cap_norm^
        self.cap_lin_w = cap_lin_w^
        self.cap_lin_b = cap_lin_b^
        self.x_w = x_w^
        self.x_b = x_b^
        self.x_pad_token = x_pad_token^
        self.cap_pad_token = cap_pad_token^
        self.nr_mod_w = nr_mod_w^
        self.nr_mod_b = nr_mod_b^
        self.main_mod_w = main_mod_w^
        self.main_mod_b = main_mod_b^
        self.final_mod_w = final_mod_w^
        self.final_mod_b = final_mod_b^
        self.final_lin_w = final_lin_w^
        self.final_lin_b = final_lin_b^


# Load all frozen aux weights once. `num_nr`/`num_main` = how many blocks of each
# modulated stream we actually run (must match the block lists the trainer loads).
def load_zimage_real_aux(
    st: ShardedSafeTensors, num_nr: Int, num_main: Int, ctx: DeviceContext
) raises -> ZImageRealAux:
    var nr_mod_w = List[TArc]()
    var nr_mod_b = List[TArc]()
    for i in range(num_nr):
        var p = String("noise_refiner.") + String(i) + String(".adaLN_modulation.0")
        nr_mod_w.append(TArc(_load_device_preserve(st, p + String(".weight"), ctx)))
        nr_mod_b.append(TArc(_load_device_preserve(st, p + String(".bias"), ctx)))
    var main_mod_w = List[TArc]()
    var main_mod_b = List[TArc]()
    for i in range(num_main):
        var p = String("layers.") + String(i) + String(".adaLN_modulation.0")
        main_mod_w.append(TArc(_load_device_preserve(st, p + String(".weight"), ctx)))
        main_mod_b.append(TArc(_load_device_preserve(st, p + String(".bias"), ctx)))
    return ZImageRealAux(
        TArc(_load_device_preserve(st, String("t_embedder.mlp.0.weight"), ctx)),
        TArc(_load_device_preserve(st, String("t_embedder.mlp.0.bias"), ctx)),
        TArc(_load_device_preserve(st, String("t_embedder.mlp.2.weight"), ctx)),
        TArc(_load_device_preserve(st, String("t_embedder.mlp.2.bias"), ctx)),
        TArc(_load_device_preserve(st, String("cap_embedder.0.weight"), ctx)),
        TArc(_load_device_preserve(st, String("cap_embedder.1.weight"), ctx)),
        TArc(_load_device_preserve(st, String("cap_embedder.1.bias"), ctx)),
        TArc(_load_device_preserve(st, String("all_x_embedder.2-1.weight"), ctx)),
        TArc(_load_device_preserve(st, String("all_x_embedder.2-1.bias"), ctx)),
        TArc(_load_device_preserve(st, String("x_pad_token"), ctx)),
        TArc(_load_device_preserve(st, String("cap_pad_token"), ctx)),
        nr_mod_w^, nr_mod_b^, main_mod_w^, main_mod_b^,
        TArc(_load_device_preserve(st, String("all_final_layer.2-1.adaLN_modulation.1.weight"), ctx)),
        TArc(_load_device_preserve(st, String("all_final_layer.2-1.adaLN_modulation.1.bias"), ctx)),
        # BUG: the stack final projection still takes host-F32 x_out with a
        # biased linear helper outside this loader scope, so these two still
        # upcast BF16 checkpoints until that call site casts to weight dtype.
        TArc(_load_f32_device(st, String("all_final_layer.2-1.linear.weight"), ctx)),
        TArc(_load_f32_device(st, String("all_final_layer.2-1.linear.bias"), ctx)),
    )


# ── t_embedder: sinusoidal(adaln_dim, t*t_scale) -> mlp.0 -> silu -> mlp.2 ─────
# Returns device [1, 256] adaln embedding. Translates zimage_dit._t_embedder.
def build_adaln(
    aux: ZImageRealAux, t_val: Float32, adaln_dim: Int, t_scale: Float32, ctx: DeviceContext
) raises -> Tensor:
    var half = adaln_dim // 2
    var max_period = Float32(10000.0)
    var scaled = t_val * t_scale
    var emb = List[Float32]()
    var log_mp = flog(max_period)
    for i in range(half):
        var freq = fexp(-log_mp * Float32(i) / Float32(half))
        emb.append(fcos(scaled * freq))
    for i in range(half):
        var freq = fexp(-log_mp * Float32(i) / Float32(half))
        emb.append(fsin(scaled * freq))
    var t_freq = Tensor.from_host(emb, [1, adaln_dim], STDtype.F32, ctx)
    var t_in = cast_tensor(t_freq, aux.t_w0[].dtype(), ctx)
    var h = linear(t_in, aux.t_w0[], Optional[Tensor](aux.t_b0[].clone(ctx)), ctx)
    var ha = silu(h, ctx)
    return linear(ha, aux.t_w2[], Optional[Tensor](aux.t_b2[].clone(ctx)), ctx)  # [1,256]


# ── per-block RAW modvecs: Linear(adaln) -> [1,4D] -> chunk 4 (no tanh/+1) ─────
def build_block_modvecs(
    mod_w: Tensor, mod_b: Tensor, adaln: Tensor, D: Int, ctx: DeviceContext
) raises -> ZImageModVecs:
    var adaln_in = cast_tensor(adaln, mod_w.dtype(), ctx)
    var mod = linear(adaln_in, mod_w, Optional[Tensor](mod_b.clone(ctx)), ctx)  # [1,4D]
    var h = mod.to_host(ctx)
    return ZImageModVecs(
        _slice_cols(h, 1, 4 * D, 0 * D, D),   # scale_msa
        _slice_cols(h, 1, 4 * D, 1 * D, D),   # gate_msa
        _slice_cols(h, 1, 4 * D, 2 * D, D),   # scale_mlp
        _slice_cols(h, 1, 4 * D, 3 * D, D),   # gate_mlp
    )


# ── final-layer f_scale: raw Linear(silu(adaln)) as the stack's modulate scale.
# The stack uses modulate(ln, f_scale, 0) = (1 + f_scale) * ln.
# Diffusers final layer is (1 + Linear(silu(adaln))) * ln, so f_scale = raw.
def build_f_scale(
    aux: ZImageRealAux, adaln: Tensor, D: Int, ctx: DeviceContext
) raises -> List[Float32]:
    var c_silu = silu(adaln, ctx)
    var c_in = cast_tensor(c_silu, aux.final_mod_w[].dtype(), ctx)
    var raw = linear(c_in, aux.final_mod_w[], Optional[Tensor](aux.final_mod_b[].clone(ctx)), ctx)  # [1,D]
    var h = raw.to_host(ctx)
    var o = List[Float32]()
    for i in range(D):
        o.append(h[i])
    return o^


# ── cap_embedder: RMSNorm(cap_dim) -> Linear -> [CAPLEN, D] host ──────────────
# cap_feats device [CAPLEN, cap_dim]. Returns host [CAPLEN*D].
def build_cap_seq(
    aux: ZImageRealAux, cap_feats: Tensor, eps: Float32, ctx: DeviceContext
) raises -> List[Float32]:
    var cap_in = cast_tensor(cap_feats, aux.cap_norm[].dtype(), ctx)
    var normed = rms_norm(cap_in, aux.cap_norm[], eps, ctx)
    var normed_in = cast_tensor(normed, aux.cap_lin_w[].dtype(), ctx)
    var emb = linear(normed_in, aux.cap_lin_w[], Optional[Tensor](aux.cap_lin_b[].clone(ctx)), ctx)
    return emb.to_host(ctx)


# ── x_embedder: patchify(channel-minor) -> Linear -> [IMG_TOK, D] host ─────────
# latent device [1, C, H, W]. patch p. channel-MINOR within-patch flatten:
#   view [C, Ht, p, Wt, p] -> permute (Ht,Wt,p,p,C) -> reshape [Ht*Wt, p*p*C].
# Mirrors zimage_dit._patchify_zimage (diffusers (1,3,5,2,4,6,0) with F=pF=1).
def build_x_seq(
    aux: ZImageRealAux, latent: Tensor, C: Int, Hl: Int, Wl: Int, p: Int, ctx: DeviceContext
) raises -> List[Float32]:
    var ht = Hl // p
    var wt = Wl // p
    # view [C, Ht, p, Wt, p]
    var v5 = reshape(latent, [C, ht, p, wt, p], ctx)
    # permute -> [Ht, Wt, p, p, C]  (dims 1,3,2,4,0)
    var perm = List[Int]()
    perm.append(1); perm.append(3); perm.append(2); perm.append(4); perm.append(0)
    var pm = permute(v5, perm^, ctx)
    var patches = reshape_owned(pm^, [ht * wt, p * p * C])   # [IMG_TOK, 64]
    var patches_in = cast_tensor(patches, aux.x_w[].dtype(), ctx)
    var emb = linear(patches_in, aux.x_w[], Optional[Tensor](aux.x_b[].clone(ctx)), ctx)  # [IMG_TOK, D]
    return emb.to_host(ctx)


# ── rope tables: 3-axis interleaved [S*H, Dh/2]. Translates zimage_dit._build_rope.
# positions: per-token [p0,p1,p2]. axes (a0,a1,a2) sum/2 == Dh/2.
def build_rope(
    positions: List[List[Int]], H: Int, Dh: Int, theta: Float32,
    a0: Int, a1: Int, a2: Int, ctx: DeviceContext,
) raises -> Tuple[TArc, TArc]:
    var half = Dh // 2
    var log_theta = flog(theta)
    var axes = List[Int]()
    axes.append(a0); axes.append(a1); axes.append(a2)
    var cos_vals = List[Float32]()
    var sin_vals = List[Float32]()
    for t in range(len(positions)):
        var angles = List[Float32]()
        for a in range(3):
            var da = axes[a]
            var ha = da // 2
            var pos = Float32(positions[t][a])
            for i in range(ha):
                var inv_freq = fexp(-log_theta * Float32(2 * i) / Float32(da))
                angles.append(pos * inv_freq)
        for _head in range(H):
            for i in range(half):
                cos_vals.append(fcos(angles[i]))
                sin_vals.append(fsin(angles[i]))
    var rows = len(positions) * H
    var cos_t = TArc(Tensor.from_host(cos_vals, [rows, half], STDtype.F32, ctx))
    var sin_t = TArc(Tensor.from_host(sin_vals, [rows, half], STDtype.F32, ctx))
    return (cos_t^, sin_t^)


# ── rope positions for img / cap / unified ───────────────────────────────────
# cap tokens: real rows (i+1,0,0), pad rows (0,0,0). image tokens:
# (cap_padded+1, ih, iw), followed by image pad rows (0,0,0). unified = img ++ cap.
def build_positions(
    n_img: Int, ht: Int, wt: Int, cap_len: Int, valid_cap: Int
) -> Tuple[List[List[Int]], List[List[Int]]]:
    var real_cap = valid_cap
    if real_cap < 0 or real_cap > cap_len:
        real_cap = cap_len
    var cap_pos = List[List[Int]]()
    for i in range(cap_len):
        var pl = List[Int]()
        if i < real_cap:
            pl.append(i + 1); pl.append(0); pl.append(0)
        else:
            pl.append(0); pl.append(0); pl.append(0)
        cap_pos.append(pl^)
    var x0 = cap_len + 1
    var x_pos = List[List[Int]]()
    for ih in range(ht):
        for iw in range(wt):
            var pl = List[Int]()
            pl.append(x0); pl.append(ih); pl.append(iw)
            x_pos.append(pl^)
    while len(x_pos) < n_img:
        var pl = List[Int]()
        pl.append(0); pl.append(0); pl.append(0)
        x_pos.append(pl^)
    return (x_pos^, cap_pos^)

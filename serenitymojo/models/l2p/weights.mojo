# models/l2p/weights.mojo — Real safetensors -> L2P training block weights.
#
# L2P checkpoint layout (model-1k-merge.safetensors, 545 tensors, confirmed
# 2026-06-03 header scan):
#   all_x_embedder.16-1.{weight,bias}         patchify16 input proj   [3840,768]
#   cap_embedder.0.weight                      RMSNorm scale            [2560]
#   cap_embedder.1.{weight,bias}               caption linear           [3840,2560]
#   t_embedder.mlp.{0,2}.{weight,bias}         timestep MLP             [1024,256],[256,1024]
#   noise_refiner.{i}.adaLN_modulation.0.*     per-block adaLN          [15360,256]
#   noise_refiner.{i}.attention.{to_q,...}.*   modulated main blocks    [3840,3840]
#   context_refiner.{i}.attention.{to_q,...}.* unmodulated refiner      [3840,3840]
#   layers.{i}.attention.{to_q,...}.*          main blocks (BF16/F32)   [3840,3840]
#   x_pad_token                                learned pad token        [1,3840]
#   cap_pad_token                              learned pad token        [1,3840]
#   local_decoder.*                            MicroDiffusion head (FROZEN, never trained)
#
# Weight-dtype quirk (confirmed from header):
#   layers.0..4   = BF16   layers.5..24 = F32   layers.25..29 = BF16
#   noise_refiner / context_refiner = BF16
# The loader preserves checkpoint dtype for large projection matrices (same
# mixed-dtype contract as load_zimage_block_weights_prefixed_mixed) and upcast
# small norm scale vectors to F32 (rms_norm kernel contract).
#
# SCOPE: Real single-file SafeTensors (NOT sharded). L2P is one file.
# The L2P block key layout is IDENTICAL to the base Z-Image layout, so we
# REUSE ZImageBlockWeights and the existing loader (load_zimage_block_weights_prefixed_mixed)
# from models/zimage/weights.mojo — but that loader takes a ShardedSafeTensors.
# This module wraps a SafeTensors (single-file) in a lightweight shim and
# provides builders for the L2P aux (embedders + adaLN + pad tokens).
#
# NO final-layer linear: L2P has no all_final_layer. The local_decoder
# ConvNet head operates on the last transformer hidden state and is frozen
# (not a LoRA target). The trainer receives x_seq [N_IMG, D] directly from
# the stack and passes it to the local_decoder separately.
#
# Mojo 0.26.x+ / 1.0.0b1: def not fn; no fn; Tensor move-only; host List[Float32].

from std.collections import List, Optional
from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from std.math import exp as fexp, log as flog, cos as fcos, sin as fsin
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import rms_norm
from serenitymojo.ops.activations import silu
from serenitymojo.ops.tensor_algebra import reshape, permute, reshape_owned
from serenitymojo.models.zimage.weights import ZImageBlockWeights
from serenitymojo.models.zimage.block import ZImageModVecs


comptime TArc = ArcPointer[Tensor]


# ── raw loaders for a single-file SafeTensors ─────────────────────────────────

def _load_l2p_device_preserve(
    st: SafeTensors, name: String, ctx: DeviceContext
) raises -> Tensor:
    """Load tensor preserving checkpoint dtype (BF16 or F32)."""
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    return Tensor.from_view(tv, ctx)


def _load_l2p_f32_device(
    st: SafeTensors, name: String, ctx: DeviceContext
) raises -> Tensor:
    """Load tensor and upcast to F32 (for small norm/scale vectors)."""
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    var t = Tensor.from_view(tv, ctx)
    return cast_tensor(t, STDtype.F32, ctx)


# ── block weight loader ───────────────────────────────────────────────────────

def load_l2p_block_weights_prefixed(
    st: SafeTensors, prefix: String, ctx: DeviceContext
) raises -> ZImageBlockWeights:
    """Load one Z-Image block from the L2P single-file checkpoint.

    Reuses ZImageBlockWeights (identical block structure to Z-Image base).
    Mixed dtype: large projections stay in checkpoint dtype (BF16 or F32);
    small norm scale vectors are upcast to F32 (rms_norm kernel contract).
    Mirrors load_zimage_block_weights_prefixed_mixed but reads SafeTensors
    (single file) instead of ShardedSafeTensors.
    """
    var ap = prefix + String(".attention")
    var fp = prefix + String(".feed_forward")
    return ZImageBlockWeights(
        TArc(_load_l2p_f32_device(st, prefix + String(".attention_norm1.weight"), ctx)),
        TArc(_load_l2p_device_preserve(st, ap + String(".to_q.weight"), ctx)),
        TArc(_load_l2p_device_preserve(st, ap + String(".to_k.weight"), ctx)),
        TArc(_load_l2p_device_preserve(st, ap + String(".to_v.weight"), ctx)),
        TArc(_load_l2p_device_preserve(st, ap + String(".to_out.0.weight"), ctx)),
        TArc(_load_l2p_f32_device(st, ap + String(".norm_q.weight"), ctx)),
        TArc(_load_l2p_f32_device(st, ap + String(".norm_k.weight"), ctx)),
        TArc(_load_l2p_f32_device(st, prefix + String(".attention_norm2.weight"), ctx)),
        TArc(_load_l2p_f32_device(st, prefix + String(".ffn_norm1.weight"), ctx)),
        TArc(_load_l2p_device_preserve(st, fp + String(".w1.weight"), ctx)),
        TArc(_load_l2p_device_preserve(st, fp + String(".w3.weight"), ctx)),
        TArc(_load_l2p_device_preserve(st, fp + String(".w2.weight"), ctx)),
        TArc(_load_l2p_f32_device(st, prefix + String(".ffn_norm2.weight"), ctx)),
    )


# ── frozen auxiliary weights (embedders + adaLN per block + pad tokens) ───────

struct L2PRealAux(Movable):
    """Frozen pre-block weights for L2P training loop.

    All builders run on GPU ops (linear / rms_norm / silu) and return host
    List[Float32] or device Tensors for per-step reuse.  Mirrors ZImageRealAux
    from models/zimage/real_weights.mojo but adapted for the L2P checkpoint
    layout (patchify16 input proj key: all_x_embedder.16-1.*, timestep MLP
    kept identical key structure; NO all_final_layer in L2P).
    """
    # t_embedder MLP (checkpoint dtype; builders cast transient inputs)
    var t_w0: TArc          # [1024, 256]
    var t_b0: TArc          # [1024]
    var t_w2: TArc          # [256, 1024]
    var t_b2: TArc          # [256]
    # cap_embedder (RMSNorm + linear)
    var cap_norm: TArc      # [2560]
    var cap_lin_w: TArc     # [3840, 2560]
    var cap_lin_b: TArc     # [3840]
    # patchify16 input proj (all_x_embedder.16-1.*): [3840,768]
    var x_w: TArc           # [3840, 768]
    var x_b: TArc           # [3840]
    # learned pad tokens
    var x_pad_token: TArc   # [1, 3840]
    var cap_pad_token: TArc # [1, 3840]
    # per-block adaLN_modulation.0 (noise_refiner only; context_refiner has none)
    var nr_mod_w: List[TArc]    # num_nr x [15360, 256]
    var nr_mod_b: List[TArc]    # num_nr x [15360]
    var main_mod_w: List[TArc]  # num_main x [15360, 256]
    var main_mod_b: List[TArc]  # num_main x [15360]

    def __init__(
        out self,
        var t_w0: TArc, var t_b0: TArc, var t_w2: TArc, var t_b2: TArc,
        var cap_norm: TArc, var cap_lin_w: TArc, var cap_lin_b: TArc,
        var x_w: TArc, var x_b: TArc,
        var x_pad_token: TArc, var cap_pad_token: TArc,
        var nr_mod_w: List[TArc], var nr_mod_b: List[TArc],
        var main_mod_w: List[TArc], var main_mod_b: List[TArc],
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


def load_l2p_real_aux(
    st: SafeTensors, num_nr: Int, num_main: Int, ctx: DeviceContext
) raises -> L2PRealAux:
    """Load all frozen L2P auxiliary weights from a single-file SafeTensors.

    Key differences from load_zimage_real_aux:
      * x_embedder key = 'all_x_embedder.16-1.*' (patch16 proj, dim 768 not 64)
      * adaLN weight shape = [15360, 256] (4*3840 vs 4*3840; same but for 4D)
      * context_refiner has NO adaLN_modulation keys (not loaded here)
      * NO all_final_layer (local_decoder is a ConvNet head, loaded separately)
    Auxiliary tensors preserve checkpoint dtype; the builder functions below cast
    transient F32 cache/schedule inputs to the relevant weight dtype before
    biased linears or norm ops.
    """
    var nr_mod_w = List[TArc]()
    var nr_mod_b = List[TArc]()
    for i in range(num_nr):
        var p = String("noise_refiner.") + String(i) + String(".adaLN_modulation.0")
        nr_mod_w.append(TArc(_load_l2p_device_preserve(st, p + String(".weight"), ctx)))
        nr_mod_b.append(TArc(_load_l2p_device_preserve(st, p + String(".bias"), ctx)))
    var main_mod_w = List[TArc]()
    var main_mod_b = List[TArc]()
    for i in range(num_main):
        var p = String("layers.") + String(i) + String(".adaLN_modulation.0")
        main_mod_w.append(TArc(_load_l2p_device_preserve(st, p + String(".weight"), ctx)))
        main_mod_b.append(TArc(_load_l2p_device_preserve(st, p + String(".bias"), ctx)))
    return L2PRealAux(
        TArc(_load_l2p_device_preserve(st, String("t_embedder.mlp.0.weight"), ctx)),
        TArc(_load_l2p_device_preserve(st, String("t_embedder.mlp.0.bias"), ctx)),
        TArc(_load_l2p_device_preserve(st, String("t_embedder.mlp.2.weight"), ctx)),
        TArc(_load_l2p_device_preserve(st, String("t_embedder.mlp.2.bias"), ctx)),
        TArc(_load_l2p_device_preserve(st, String("cap_embedder.0.weight"), ctx)),
        TArc(_load_l2p_device_preserve(st, String("cap_embedder.1.weight"), ctx)),
        TArc(_load_l2p_device_preserve(st, String("cap_embedder.1.bias"), ctx)),
        TArc(_load_l2p_device_preserve(st, String("all_x_embedder.16-1.weight"), ctx)),
        TArc(_load_l2p_device_preserve(st, String("all_x_embedder.16-1.bias"), ctx)),
        TArc(_load_l2p_device_preserve(st, String("x_pad_token"), ctx)),
        TArc(_load_l2p_device_preserve(st, String("cap_pad_token"), ctx)),
        nr_mod_w^, nr_mod_b^,
        main_mod_w^, main_mod_b^,
    )


# ── small host helpers (mirrors real_weights.mojo) ────────────────────────────

def _ones_l2p(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n):
        o.append(Float32(1.0))
    return o^


def _slice_cols_l2p(
    v: List[Float32], rows: Int, cols: Int, col0: Int, w: Int
) -> List[Float32]:
    var o = List[Float32]()
    for r in range(rows):
        for c in range(w):
            o.append(v[r * cols + col0 + c])
    return o^


# ── t_embedder: sinusoidal -> mlp.0 -> silu -> mlp.2 ─────────────────────────
# L2P timestep MLP dim = 256 (ZIMAGE_L2P_TIMESTEP_DIM in contract.mojo).
# Identical math to build_adaln in real_weights.mojo; adaln_dim fixed to 256.
def build_l2p_adaln(
    aux: L2PRealAux, t_val: Float32, t_scale: Float32, ctx: DeviceContext
) raises -> Tensor:
    """Compute adaln embedding [1, 256] for a given t_val (flow-matching sigma)."""
    var adaln_dim = 256  # ZIMAGE_L2P_TIMESTEP_DIM
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
    return linear(ha, aux.t_w2[], Optional[Tensor](aux.t_b2[].clone(ctx)), ctx)


# ── per-block RAW modvecs: Linear(adaln) [1,15360] -> chunk 4 x [1,D] ─────────
# L2P adaLN_modulation weight is [15360, 256] (15360 = 4 * D = 4 * 3840).
# Returns ZImageModVecs (scale_msa, gate_msa, scale_mlp, gate_mlp) each [D].
# The block applies tanh(gate) and (1+scale) internally (not here).
def build_l2p_block_modvecs(
    mod_w: Tensor, mod_b: Tensor, adaln: Tensor, D: Int, ctx: DeviceContext
) raises -> ZImageModVecs:
    """Build RAW 4-chunk modvec for one L2P block. Identical to build_block_modvecs."""
    var adaln_in = cast_tensor(adaln, mod_w.dtype(), ctx)
    var mod = linear(adaln_in, mod_w, Optional[Tensor](mod_b.clone(ctx)), ctx)
    var h = mod.to_host(ctx)
    return ZImageModVecs(
        _slice_cols_l2p(h, 1, 4 * D, 0 * D, D),
        _slice_cols_l2p(h, 1, 4 * D, 1 * D, D),
        _slice_cols_l2p(h, 1, 4 * D, 2 * D, D),
        _slice_cols_l2p(h, 1, 4 * D, 3 * D, D),
    )


# ── cap_embedder: RMSNorm(2560) -> Linear -> [CAPLEN, D] host ─────────────────
def build_l2p_cap_seq(
    aux: L2PRealAux, cap_feats: Tensor, eps: Float32, ctx: DeviceContext
) raises -> List[Float32]:
    var cap_in = cast_tensor(cap_feats, aux.cap_norm[].dtype(), ctx)
    var normed = rms_norm(cap_in, aux.cap_norm[], eps, ctx)
    var normed_in = cast_tensor(normed, aux.cap_lin_w[].dtype(), ctx)
    var emb = linear(normed_in, aux.cap_lin_w[], Optional[Tensor](aux.cap_lin_b[].clone(ctx)), ctx)
    return emb.to_host(ctx)


# ── patchify16 x_embedder: pixel [1,3,H,W] -> patch16 -> Linear -> [IMG_TOK, D]
# patch_size = 16, patch_vector_dim = 16*16*3 = 768.
# Channel-minor within-patch flatten: view [3, Ht, 16, Wt, 16] -> permute
# (Ht, Wt, 16, 16, 3) -> reshape [Ht*Wt, 768] -> Linear -> [Ht*Wt, D].
# Mirrors ZImageL2PDiTPreBlockGate.pixel_embed in models/dit/zimage_l2p_dit.mojo
# but operating on F32 pixel data (training cache stores F32 pixels after
# normalization; the inference path uses BF16).
def build_l2p_x_seq(
    aux: L2PRealAux, pixels: Tensor, H: Int, W: Int, ctx: DeviceContext
) raises -> List[Float32]:
    """Patchify16 [1,3,H,W] F32 pixels and embed to [Ht*Wt, D] host."""
    var P = 16
    var ht = H // P
    var wt = W // P
    # view [3, ht, 16, wt, 16]
    var v5 = reshape(pixels, [3, ht, P, wt, P], ctx)
    # permute -> [ht, wt, 16, 16, 3]  (axes 1, 3, 2, 4, 0)
    var perm = List[Int]()
    perm.append(1); perm.append(3); perm.append(2); perm.append(4); perm.append(0)
    var pm = permute(v5, perm^, ctx)
    # reshape [ht*wt, 768]
    var patches = reshape_owned(pm^, [ht * wt, P * P * 3])
    # linear -> [ht*wt, D]
    var patches_in = cast_tensor(patches, aux.x_w[].dtype(), ctx)
    var emb = linear(patches_in, aux.x_w[], Optional[Tensor](aux.x_b[].clone(ctx)), ctx)
    return emb.to_host(ctx)


# ── rope tables: 3-axis interleaved [S*H, Dh/2]. Identical to build_rope. ─────
def build_l2p_rope(
    positions: List[List[Int]], H: Int, Dh: Int, theta: Float32,
    a0: Int, a1: Int, a2: Int, ctx: DeviceContext,
) raises -> Tuple[TArc, TArc]:
    var half = Dh // 2
    var log_theta = flog(theta)
    var axes = [a0, a1, a2]
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


# ── rope positions for img / cap / unified. Mirrors build_positions. ──────────
def build_l2p_positions(
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

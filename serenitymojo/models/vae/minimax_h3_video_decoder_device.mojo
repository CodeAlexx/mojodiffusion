# serenitymojo/models/vae/minimax_h3_video_decoder_device.mojo
#
# MiniMax-H3 video ViT decoder — DEVICE path, authority = the CREATOR SOURCE
# (`ViT3DDecoder`/`TransformerBlock`/`Attention` in the released
# `video_vae/vae_vit.py` + `base_module.py` + `attention.py`), NOT the
# diffusers PR rewrite.
#
# `models/minimax_h3/video_decoder.mojo` (unit 11) stays untouched: it is
# gated against `MiniMaxH3VideoViTDecoder3d` (the diffusers rewrite) and is
# correct FOR THAT reference. This file is a SEPARATE port whose oracle is
# the actual released checkpoint's module tree (2026-08-02 audit, read
# vae_vit.py/base_module.py/attention.py/func.py in full).
#
# WHAT WAS VERIFIED FAITHFUL in the diffusers-oracle port and is CARRIED OVER
# here unchanged (same audit; the pure-math pieces are IMPORTED from
# video_decoder.mojo, not copied, since re-deriving already-gated trig would
# be pure risk for no benefit): the rope inv_freq formula (step = 6/dim,
# n_dim=3), angle_scale=2*pi, tile(2) duplication, rotate-half convention,
# PARTIAL rotation (rot_dim=48 of head_dim=64, the rest passed through
# untouched) — `video_rope_inv_freq`/`video_position_grid`/`video_rope_table`
# (func.py create_token_ids:34-37, base_module.py RotaryEmbeddingND:157-196).
# Also carried over conceptually (re-expressed as Tensor ops, not copied):
# token layout [patch tokens | register_tokens | ONE zero suffix] (vae_vit.py
# :309-317); unweighted q/k RMSNorm (qk_norm_affine:false, attention.py
# :68-74); weighted RMSNorm norm1/norm2 (norm_type:"rms_norm", norm_affine:
# true, base_module.py:222-233); unconditional LayerNorm for norm_out
# (vae_vit.py:280, NOT gated by norm_type); zero-init learned per-channel
# scale1/scale2 residual scaling, not adaLN (base_module.py:244-245,259-260);
# no block-causal mask for FL2VA (`t_causal=causal_decoder=false`, vae_vit.py
# :327); the unpatchify index math `token*patch_dim + ((c*pt+it)*ps+ih)*ps+iw`
# (func.py _unpack_tensors_3d:77-93 — same (C,pt,ph,pw) row-major layout our
# EXISTING host oracle's unpatch loop already assumed correctly).
#
# WHAT IS DIFFERENT FROM `video_decoder.mojo` (the actual fixes; see the
# 2026-08-02 audit sent to team-lead):
#   1. FFN gate/value order. Released `FeedForward._forward_impl`
#      (base_module.py:95-105): `gate, value = w1(x).chunk(2,-1); out =
#      w2(silu(gate) * value)` — GATE IS THE FIRST HALF. The diffusers-oracle
#      port assumed the opposite (value first). This file implements the
#      release's order directly — see `_swiglu_ff` below.
#   2. Attention is ONE FUSED per-head-interleaved `to_qkv` projection
#      (attention.py:80, `nn.Linear(embed_dim, attn_inner_dim*3)`, then
#      `qkv.view(B,N,heads,3*dim_head)` BEFORE chunking — i.e. row layout
#      `[head0: q(dim_head) k(dim_head) v(dim_head), head1: ...]`), not three
#      separate to_q/to_k/to_v linears. This file SPLITS the fused weight
#      into three at LOAD TIME, reusing `models/minimax_h3/loader.mojo`'s
#      `minimax_h3_deinterleave_qkv` + `minimax_h3_split_qkv` — the SAME
#      transform already gated at max_abs 0.0 for the DiT's fused qkv, which
#      is the identical per-head-interleaved shape. Reused, not reimplemented.
#   3. Native key names: `decoder.x_embedder` (not `proj_in`), `attn.to_qkv`
#      (split at load into synthetic `attn.to_q`/`to_k`/`to_v` — see #2),
#      `attn.to_out` (a bare Linear, not `to_out.0`), `ff.w1`/`ff.w2` (not
#      `ff.net.0.proj`/`ff.net.2`). `decoder.register_tokens`/`norm_out`/
#      `proj_out` and top-level `post_quant_conv` already matched.
#
# SEQUENCE LENGTH IS COMPTIME. `ops/attention.sdpa_nomask[B,S,H,Dh]` requires
# S at compile time (mojo-port skill gotcha), so this decoder is generic over
# `S = latent_T*latent_H*latent_W + 1 + num_register_tokens` — monomorphized
# per input shape, same convention `models/vae/wan22_decoder.mojo` uses for
# `[LH,LW]`.
#
# SCOPE: `t_causal=True` (a block-causal attention mask over the temporal
# blocks) is NOT implemented — FL2VA's released config sets
# `causal_decoder: false`, so the forward here is unmasked full attention
# only, matching that partition. A caller passing `config.t_causal=True`
# gets a loud error, not silently-wrong (unmasked) output. Whether ref2va/
# t2va need the mask is UNVERIFIED (their configs weren't read).
#
# NOT IN SCOPE (pipeline-level, not this module — see the audit report):
# ImageNet pixel normalize/denormalize (normalize.py) and the clip_length=17/
# token_drop=3 temporal chunk+blend loop (klvae.py encode_temporal/
# decode_temporal). This file decodes ONE volume in a single pass.
#
# REUSE, not reimplemented:
#   models/dit/minimax_h3_dit.mojo         (none directly; see loader below)
#   models/minimax_h3/loader.mojo          minimax_h3_deinterleave_qkv,
#                                           minimax_h3_split_qkv (qkv reorder,
#                                           already gated for the DiT's fused
#                                           qkv — same per-head-interleaved
#                                           shape)
#   models/dit/minimax_h3_dit... (rope)    video_decoder.mojo's
#                                           video_rope_inv_freq /
#                                           video_position_grid /
#                                           video_rope_table (PURE MATH, no
#                                           weight dependency, unedited)
#   ops/linear.mojo                        linear_bias
#   ops/activations.mojo                   silu
#   ops/norm.mojo                          rms_norm, layer_norm
#   ops/attention.mojo                     sdpa_nomask
#   ops/tensor_algebra.mojo                slice, concat, mul, add,
#                                           mul_scalar, reshape, reshape_owned,
#                                           zeros_device
#   io/sharded.mojo                        ShardedSafeTensors
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.collections import Dict, List
from max.gpu.host import DeviceContext
from std.gpu import global_idx
from std.memory import ArcPointer
from std.math import sqrt
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.tensor import Tensor
from serenitymojo.ops.activations import silu
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.linear import linear_bias
from serenitymojo.ops.norm import rms_norm, layer_norm
from serenitymojo.ops.attention import sdpa_nomask_infer
from serenitymojo.ops.attention_flash import sdpa_flash_infer_fwd_dynamic
from serenitymojo.ops.tensor_algebra import (
    add, concat, mul, mul_scalar, reshape, reshape_owned, slice, zeros_device,
)
from serenitymojo.models.minimax_h3.loader import (
    minimax_h3_deinterleave_qkv, minimax_h3_split_qkv,
)
from serenitymojo.models.minimax_h3.video_decoder import (
    video_position_grid, video_rope_inv_freq, video_rope_table,
)

comptime TArc = ArcPointer[Tensor]
comptime _DYN1 = Layout.row_major(-1)
comptime _UNPACK_BLOCK = 256


@fieldwise_init
struct MiniMaxH3VideoDecoderDeviceConfig(Copyable, Movable):
    """`vit_decoder_kwargs` from `video_vae/source/config.json`, plus the
    parent `AutoencoderKLLegacy` fields it needs (patch_size=vae_ratio=16,
    patch_size_t=vae_ratio_t=4, in_channels=z_channels=24,
    t_causal=causal_decoder). `num_register_tokens` (4) and `norm_eps`
    (1e-5) are NOT listed in vit_decoder_kwargs, so they fall back to
    `ViT3DDecoder`'s own class defaults (base_module.py/vae_vit.py) —
    UNVERIFIED against a real header, flagged here rather than silently
    assumed correct."""

    var latent_channels: Int   # in_channels = z_channels = embed_dim (24; equal for this release)
    var out_channels: Int      # 3
    var num_layers: Int        # 36
    var heads: Int             # 32
    var dim_head: Int          # 64
    var num_register_tokens: Int  # 4 — ViT3DDecoder class default, UNVERIFIED
    var patch_size: Int        # 16 (= vae_ratio)
    var patch_size_t: Int      # 4  (= vae_ratio_t)
    var rope_theta: Float64    # 100.0
    var rope_dim_ratio: Float64  # 0.75
    var norm_eps: Float32      # 1e-5 — TransformerBlock/Attention class default, UNVERIFIED
    var t_causal: Bool         # causal_decoder; false for FL2VA
    var ffn_mult: Int          # 4 — FeedForward class default (mult not overridden)

    def dim(self) -> Int:
        return self.heads * self.dim_head


def minimax_h3_video_released_decoder_config() -> MiniMaxH3VideoDecoderDeviceConfig:
    """FL2VA `video_vae/source/config.json` `vit_decoder_kwargs` + parent
    fields, verbatim where the JSON states a value."""
    return MiniMaxH3VideoDecoderDeviceConfig(
        24, 3, 36, 32, 64, 4, 16, 4, Float64(100.0), Float64(0.75),
        Float32(1.0e-5), False, 4,
    )


struct MiniMaxH3VideoDecoderDevice(Movable):
    var weights: List[TArc]
    var name_to_idx: Dict[String, Int]
    var config: MiniMaxH3VideoDecoderDeviceConfig

    def __init__(
        out self,
        var weights: List[TArc],
        var name_to_idx: Dict[String, Int],
        config: MiniMaxH3VideoDecoderDeviceConfig,
    ):
        self.weights = weights^
        self.name_to_idx = name_to_idx^
        self.config = config.copy()

    def _w(self, name: String) raises -> ref [self.weights[0]] Tensor:
        if name not in self.name_to_idx:
            raise Error(
                String("MiniMax-H3 video decoder device: missing weight ") + name
            )
        return self.weights[self.name_to_idx[name]][]

    def has(self, name: String) -> Bool:
        return name in self.name_to_idx


# ─────────────────────────────────────────────────────────────────────────────
# Weight loading. Native keys straight through, EXCEPT the fused `to_qkv`,
# which is split into three synthetic `to_q`/`to_k`/`to_v` entries at load
# time — see this file's header item #2. Tiny weights (attn_inner_dim*3 x
# dim, at most ~12.6 MB per block at the released size) — a host round-trip
# per block, 36 times total, is not a hot path.
# ─────────────────────────────────────────────────────────────────────────────
def _block_prefix(layer: Int) -> String:
    return String("decoder.transformer_blocks.") + String(layer)


def minimax_h3_video_decoder_native_key_names(
    config: MiniMaxH3VideoDecoderDeviceConfig,
) raises -> List[String]:
    """Every NATIVE (on-disk) tensor this decoder needs — i.e. `to_qkv`
    fused, not the synthetic split keys `load` derives from it."""
    var names = List[String]()
    names.append("decoder.x_embedder.weight")
    names.append("decoder.x_embedder.bias")
    names.append("decoder.register_tokens")
    for layer in range(config.num_layers):
        var p = _block_prefix(layer)
        names.append(p + ".norm1.weight")
        names.append(p + ".attn.to_qkv.weight")
        names.append(p + ".attn.to_qkv.bias")
        names.append(p + ".attn.to_out.weight")
        names.append(p + ".attn.to_out.bias")
        names.append(p + ".scale1")
        names.append(p + ".norm2.weight")
        names.append(p + ".ff.w1.weight")
        names.append(p + ".ff.w1.bias")
        names.append(p + ".ff.w2.weight")
        names.append(p + ".ff.w2.bias")
        names.append(p + ".scale2")
    names.append("decoder.norm_out.weight")
    names.append("decoder.norm_out.bias")
    names.append("decoder.proj_out.weight")
    names.append("decoder.proj_out.bias")
    names.append("post_quant_conv.weight")
    names.append("post_quant_conv.bias")
    return names^


def _split_qkv_at_load(
    shards: ShardedSafeTensors, prefix: String,
    heads: Int, dim_head: Int, embed_dim: Int,
    dtype: STDtype, ctx: DeviceContext,
    mut weights: List[TArc], mut name_to_idx: Dict[String, Int],
) raises:
    """De-interleave + split the fused `attn.to_qkv.{weight,bias}` into
    synthetic `attn.to_q/to_k/to_v.{weight,bias}` entries, via the SAME
    host-List reorder `models/minimax_h3/loader.mojo` already uses (and
    gates at max_abs 0.0) for the DiT's fused qkv — the row layout is
    identical (per-head-interleaved `[head0: q k v, head1: ...]`), so the
    function is reused verbatim, not reimplemented."""
    var w_view = shards.tensor_view(prefix + ".attn.to_qkv.weight")
    var b_view = shards.tensor_view(prefix + ".attn.to_qkv.bias")
    var w_tensor = Tensor.from_view_as_f32(w_view, ctx)
    var b_tensor = Tensor.from_view_as_f32(b_view, ctx)
    var w_host = w_tensor.to_host(ctx)
    var b_host = b_tensor.to_host(ctx)

    var w_reordered = minimax_h3_deinterleave_qkv(w_host, heads, dim_head, embed_dim)
    # Bias is per-output-row (no in_features axis) — de-interleave with
    # in_features=1, one "row" per scalar.
    var b_reordered = minimax_h3_deinterleave_qkv(b_host, heads, dim_head, 1)

    var part_names = List[String]()
    part_names.append("q")
    part_names.append("k")
    part_names.append("v")
    for part in range(3):
        var w_part = minimax_h3_split_qkv(w_reordered, heads, dim_head, embed_dim, part)
        var b_part = minimax_h3_split_qkv(b_reordered, heads, dim_head, 1, part)
        var w_key = prefix + ".attn.to_" + part_names[part] + ".weight"
        var b_key = prefix + ".attn.to_" + part_names[part] + ".bias"
        var wt = Tensor.from_host(w_part, [heads * dim_head, embed_dim], dtype, ctx)
        var bt = Tensor.from_host(b_part, [heads * dim_head], dtype, ctx)
        name_to_idx[w_key] = len(weights)
        weights.append(TArc(wt^))
        name_to_idx[b_key] = len(weights)
        weights.append(TArc(bt^))


def minimax_h3_video_decoder_device_load(
    dir: String, config: MiniMaxH3VideoDecoderDeviceConfig, ctx: DeviceContext,
) raises -> MiniMaxH3VideoDecoderDevice:
    """Preflight every NATIVE tensor, then load: most stream straight
    through via `Tensor.from_view`; each block's fused `to_qkv` is split
    into three at load time (see `_split_qkv_at_load`)."""
    var shards = ShardedSafeTensors.open(dir)
    var native_names = minimax_h3_video_decoder_native_key_names(config)
    for i in range(len(native_names)):
        if native_names[i] not in shards.name_to_shard:
            raise Error(
                String("MiniMax-H3 video decoder device: missing weight ")
                + native_names[i] + " in " + dir
            )
    var embed_dim = config.dim()
    var weights = List[TArc]()
    var name_to_idx = Dict[String, Int]()
    var qkv_dtype = shards.tensor_info(_block_prefix(0) + ".attn.to_qkv.weight").dtype
    for i in range(len(native_names)):
        var name = native_names[i]
        if name.find(".attn.to_qkv.") >= 0:
            continue  # handled per-block below, once per block not per key
        var t = Tensor.from_view(shards.tensor_view(name), ctx)
        # BF16-resident: the checkpoint stores F32 (9.9 GiB resident); flash
        # decode is gated at 66 dB vs the F32 math reference and every op in
        # this file is dtype-polymorphic, so narrow once at load (~5 GiB,
        # BF16 GEMMs, native-BF16 flash path — no per-call cast round-trip).
        if t.dtype() == STDtype.F32:
            t = cast_tensor(t, STDtype.BF16, ctx)
        name_to_idx[name] = len(weights)
        weights.append(TArc(t^))
    var resident_dtype = STDtype.BF16 if qkv_dtype == STDtype.F32 else qkv_dtype
    for layer in range(config.num_layers):
        _split_qkv_at_load(
            shards, _block_prefix(layer), config.heads, config.dim_head,
            embed_dim, resident_dtype, ctx, weights, name_to_idx,
        )
    return MiniMaxH3VideoDecoderDevice(weights^, name_to_idx^, config)


# ─────────────────────────────────────────────────────────────────────────────
# FFN — gate = FIRST half, value = SECOND half (the release's order; see
# this file's header item #1). `w1`/`w2`, not `net.0.proj`/`net.2`.
# ─────────────────────────────────────────────────────────────────────────────
def _swiglu_ff(
    x: Tensor, prefix: String, config: MiniMaxH3VideoDecoderDeviceConfig,
    decoder: MiniMaxH3VideoDecoderDevice, ctx: DeviceContext,
) raises -> Tensor:
    var dim = config.dim()
    var inner = dim * config.ffn_mult
    var projected = linear_bias(
        x, decoder._w(prefix + ".ff.w1.weight"), decoder._w(prefix + ".ff.w1.bias"), ctx,
    )
    var rank = len(projected.shape())
    var gate = slice(projected, rank - 1, 0, inner, ctx)
    var value = slice(projected, rank - 1, inner, inner, ctx)
    var gated = mul(silu(gate, ctx), value, ctx)
    return linear_bias(
        gated, decoder._w(prefix + ".ff.w2.weight"), decoder._w(prefix + ".ff.w2.bias"), ctx,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Rope: build the cos/sin table via the ALREADY-VERIFIED pure-math functions
# in video_decoder.mojo (unedited), upload once, apply per q/k via
# slice/concat/mul/add (no new kernel — partial rotation, rot_dim < head_dim,
# the rest passed through untouched).
# ─────────────────────────────────────────────────────────────────────────────
@fieldwise_init
struct MiniMaxH3VideoDecoderRopeTables(Movable):
    var cos: Tensor
    var sin: Tensor
    var rotary_dim: Int


def _build_rope_tables(
    num_frames: Int, height: Int, width: Int, num_suffix: Int,
    config: MiniMaxH3VideoDecoderDeviceConfig, ctx: DeviceContext,
) raises -> MiniMaxH3VideoDecoderRopeTables:
    var position_ids = video_position_grid(num_frames, height, width, num_suffix)
    var rows = num_frames * height * width + num_suffix
    var inv_freq = video_rope_inv_freq(
        Int(Float64(config.dim_head) * config.rope_dim_ratio), config.rope_theta,
    )
    var rope = video_rope_table(position_ids, rows, inv_freq)
    var cos_t = Tensor.from_host(rope.cos, [rows, rope.rotary_dim], STDtype.F32, ctx)
    var sin_t = Tensor.from_host(rope.sin, [rows, rope.rotary_dim], STDtype.F32, ctx)
    return MiniMaxH3VideoDecoderRopeTables(cos_t^, sin_t^, rope.rotary_dim)


def _apply_rope(
    x: Tensor,  # [1, S, heads, head_dim]
    cos_t: Tensor, sin_t: Tensor,  # [S, rot_dim] F32
    rot_dim: Int, ctx: DeviceContext,
) raises -> Tensor:
    var s = x.shape()
    var seq = s[1]
    var head_dim = s[3]
    var half = rot_dim // 2

    var x_rot = slice(x, 3, 0, rot_dim, ctx)
    var x1 = slice(x_rot, 3, 0, half, ctx)
    var x2 = slice(x_rot, 3, half, half, ctx)
    var neg_x2 = mul_scalar(x2, Float32(-1.0), ctx)
    var rotated = concat(3, ctx, neg_x2, x1)

    var cos_b = cast_tensor(reshape(cos_t, [1, seq, 1, rot_dim], ctx), x.dtype(), ctx)
    var sin_b = cast_tensor(reshape(sin_t, [1, seq, 1, rot_dim], ctx), x.dtype(), ctx)
    var term1 = mul(x_rot, cos_b, ctx)
    var term2 = mul(rotated, sin_b, ctx)
    var out_rot = add(term1, term2, ctx)

    if rot_dim == head_dim:
        return out_rot^
    var x_pass = slice(x, 3, rot_dim, head_dim - rot_dim, ctx)
    return concat(3, ctx, out_rot, x_pass)


# ─────────────────────────────────────────────────────────────────────────────
# Attention: split-at-load to_q/to_k/to_v (see header #2), unweighted q/k
# RMSNorm, partial rope, full (unmasked) attention.
# ─────────────────────────────────────────────────────────────────────────────
def _vit_attention[S: Int, H: Int, Dh: Int](
    x: Tensor, prefix: String, config: MiniMaxH3VideoDecoderDeviceConfig,
    decoder: MiniMaxH3VideoDecoderDevice,
    cos_t: Tensor, sin_t: Tensor, rot_dim: Int, qk_ones: Tensor,
    ctx: DeviceContext,
) raises -> Tensor:
    if config.heads != H or config.dim_head != Dh:
        raise Error(
            "MiniMax-H3 video decoder device: comptime H/Dh do not match config"
        )
    var heads = H
    var head_dim = Dh
    var q = linear_bias(
        x, decoder._w(prefix + ".attn.to_q.weight"),
        decoder._w(prefix + ".attn.to_q.bias"), ctx,
    )
    var k = linear_bias(
        x, decoder._w(prefix + ".attn.to_k.weight"),
        decoder._w(prefix + ".attn.to_k.bias"), ctx,
    )
    var v = linear_bias(
        x, decoder._w(prefix + ".attn.to_v.weight"),
        decoder._w(prefix + ".attn.to_v.bias"), ctx,
    )
    q = reshape(q, [1, S, heads, head_dim], ctx)
    k = reshape(k, [1, S, heads, head_dim], ctx)
    v = reshape(v, [1, S, heads, head_dim], ctx)

    q = rms_norm(q, qk_ones, config.norm_eps, ctx)
    k = rms_norm(k, qk_ones, config.norm_eps, ctx)

    q = _apply_rope(q, cos_t, sin_t, rot_dim, ctx)
    k = _apply_rope(k, cos_t, sin_t, rot_dim, ctx)

    var scale = Float32(1.0) / sqrt(Float32(head_dim))
    var attn = sdpa_nomask_infer[1, S, H, Dh](q, k, v, scale, ctx)
    var flat = reshape_owned(attn^, [1, S, heads * head_dim])
    return linear_bias(
        flat, decoder._w(prefix + ".attn.to_out.weight"),
        decoder._w(prefix + ".attn.to_out.bias"), ctx,
    )


def _vit_block_forward[S: Int, H: Int, Dh: Int](
    var hidden: Tensor, prefix: String, config: MiniMaxH3VideoDecoderDeviceConfig,
    decoder: MiniMaxH3VideoDecoderDevice,
    cos_t: Tensor, sin_t: Tensor, rot_dim: Int, qk_ones: Tensor,
    ctx: DeviceContext,
) raises -> Tensor:
    """`hidden = hidden + attn(norm1(hidden))*scale1;
       hidden = hidden + ff(norm2(hidden))*scale2` (base_module.py
    TransformerBlock.forward:262-282)."""
    var normed = rms_norm(hidden, decoder._w(prefix + ".norm1.weight"), config.norm_eps, ctx)
    var attn_out = _vit_attention[S, H, Dh](normed, prefix, config, decoder, cos_t, sin_t, rot_dim, qk_ones, ctx)
    hidden = add(hidden, mul(attn_out, decoder._w(prefix + ".scale1"), ctx), ctx)

    var normed2 = rms_norm(hidden, decoder._w(prefix + ".norm2.weight"), config.norm_eps, ctx)
    var ff_out = _swiglu_ff(normed2, prefix, config, decoder, ctx)
    hidden = add(hidden, mul(ff_out, decoder._w(prefix + ".scale2"), ctx), ctx)
    return hidden^


# ─────────────────────────────────────────────────────────────────────────────
# Unpatchify — one gather kernel (rank-8 permute exceeds ops/tensor_algebra's
# _MAXRANK=6, so this is a small dedicated kernel rather than a
# reshape+permute composition). Index math mirrors func.py
# _unpack_tensors_3d/(C,pt,ph,pw) row-major, the SAME formula
# `video_decoder.mojo`'s host oracle unpatch loop already uses correctly —
# just re-expressed as one-thread-per-output-element instead of a host loop.
# ─────────────────────────────────────────────────────────────────────────────
def _unpack_patches_kernel_f32(
    src: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    dst: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    t_dim_w: Int32, h_dim_w: Int32, w_dim_w: Int32, c_dim_w: Int32, pt_w: Int32, ps_w: Int32, total_w: Int64,
):
    var t_dim = Int(t_dim_w)
    var h_dim = Int(h_dim_w)
    var w_dim = Int(w_dim_w)
    var c_dim = Int(c_dim_w)
    var pt = Int(pt_w)
    var ps = Int(ps_w)
    var total = Int(total_w)
    var idx = Int(global_idx.x)
    if idx >= total:
        return
    var out_h = h_dim * ps
    var out_w = w_dim * ps
    var c = idx % c_dim
    var rest = idx // c_dim
    var dw = rest % out_w
    rest //= out_w
    var dh = rest % out_h
    rest //= out_h
    var dt = rest
    var t = dt // pt
    var it = dt % pt
    var h = dh // ps
    var ih = dh % ps
    var w = dw // ps
    var iw = dw % ps
    var token = (t * h_dim + h) * w_dim + w
    var patch_dim = c_dim * pt * ps * ps
    var source = token * patch_dim + ((c * pt + it) * ps + ih) * ps + iw
    dst[idx] = src[source]


def _unpack_patches_kernel_bf16(
    src: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    dst: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    t_dim_w: Int32, h_dim_w: Int32, w_dim_w: Int32, c_dim_w: Int32, pt_w: Int32, ps_w: Int32, total_w: Int64,
):
    var t_dim = Int(t_dim_w)
    var h_dim = Int(h_dim_w)
    var w_dim = Int(w_dim_w)
    var c_dim = Int(c_dim_w)
    var pt = Int(pt_w)
    var ps = Int(ps_w)
    var total = Int(total_w)
    var idx = Int(global_idx.x)
    if idx >= total:
        return
    var out_h = h_dim * ps
    var out_w = w_dim * ps
    var c = idx % c_dim
    var rest = idx // c_dim
    var dw = rest % out_w
    rest //= out_w
    var dh = rest % out_h
    rest //= out_h
    var dt = rest
    var t = dt // pt
    var it = dt % pt
    var h = dh // ps
    var ih = dh % ps
    var w = dw // ps
    var iw = dw % ps
    var token = (t * h_dim + h) * w_dim + w
    var patch_dim = c_dim * pt * ps * ps
    var source = token * patch_dim + ((c * pt + it) * ps + ih) * ps + iw
    dst[idx] = src[source]


def _unpack_patches(
    patch_tokens: Tensor,  # [1, num_tokens, patch_dim]
    out_channels: Int, pt: Int, ps: Int, t_dim: Int, h_dim: Int, w_dim: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    var out_t = t_dim * pt
    var out_h = h_dim * ps
    var out_w = w_dim * ps
    var total = out_t * out_h * out_w * out_channels
    var dt_ = patch_tokens.dtype().to_mojo_dtype()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](
        total * patch_tokens.dtype().byte_size()
    )
    var src_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](patch_tokens.numel()))
    var dst_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](total))
    var grid = (total + _UNPACK_BLOCK - 1) // _UNPACK_BLOCK
    if dt_ == DType.float32:
        var S_ = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(patch_tokens.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=src_rl,
    )
        var D = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=dst_rl,
    )
        ctx.enqueue_function[_unpack_patches_kernel_f32](
            S_, D, Int32(t_dim), Int32(h_dim), Int32(w_dim), Int32(out_channels), Int32(pt), Int32(ps), Int64(total),
            grid_dim=grid, block_dim=_UNPACK_BLOCK,
        )
    elif dt_ == DType.bfloat16:
        var S_ = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(patch_tokens.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=src_rl,
    )
        var D = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=dst_rl,
    )
        ctx.enqueue_function[_unpack_patches_kernel_bf16](
            S_, D, Int32(t_dim), Int32(h_dim), Int32(w_dim), Int32(out_channels), Int32(pt), Int32(ps), Int64(total),
            grid_dim=grid, block_dim=_UNPACK_BLOCK,
        )
    else:
        raise Error("MiniMax-H3 video decoder device: unpack only supports F32/BF16")
    return Tensor(
        out_buf^, [1, out_t, out_h, out_w, out_channels], patch_tokens.dtype()
    )


# ─────────────────────────────────────────────────────────────────────────────
# Top-level forward.
# ─────────────────────────────────────────────────────────────────────────────
def _vit_attention_dyn(
    x: Tensor,  # [B, S, dim]
    prefix: String, config: MiniMaxH3VideoDecoderDeviceConfig,
    decoder: MiniMaxH3VideoDecoderDevice,
    cos_t: Tensor, sin_t: Tensor, rot_dim: Int, qk_ones: Tensor,
    ctx: DeviceContext,
) raises -> Tensor:
    """Runtime-shape twin of `_vit_attention` for the BATCHED tile path:
    attention runs through the cuDNN dynamic entry (runtime B/S), so one
    call covers every tile of a clip instead of one launch storm per tile."""
    var xs = x.shape()
    var Bt = xs[0]
    var St = xs[1]
    var heads = config.heads
    var head_dim = config.dim_head
    var q = linear_bias(
        x, decoder._w(prefix + ".attn.to_q.weight"),
        decoder._w(prefix + ".attn.to_q.bias"), ctx,
    )
    var k = linear_bias(
        x, decoder._w(prefix + ".attn.to_k.weight"),
        decoder._w(prefix + ".attn.to_k.bias"), ctx,
    )
    var v = linear_bias(
        x, decoder._w(prefix + ".attn.to_v.weight"),
        decoder._w(prefix + ".attn.to_v.bias"), ctx,
    )
    q = reshape(q, [Bt, St, heads, head_dim], ctx)
    k = reshape(k, [Bt, St, heads, head_dim], ctx)
    v = reshape(v, [Bt, St, heads, head_dim], ctx)
    q = rms_norm(q, qk_ones, config.norm_eps, ctx)
    k = rms_norm(k, qk_ones, config.norm_eps, ctx)
    q = _apply_rope(q, cos_t, sin_t, rot_dim, ctx)
    k = _apply_rope(k, cos_t, sin_t, rot_dim, ctx)
    var scale = Float32(1.0) / sqrt(Float32(head_dim))
    var attn = sdpa_flash_infer_fwd_dynamic(q, k, v, scale, ctx)
    var flat = reshape_owned(attn^, [Bt, St, heads * head_dim])
    return linear_bias(
        flat, decoder._w(prefix + ".attn.to_out.weight"),
        decoder._w(prefix + ".attn.to_out.bias"), ctx,
    )


def _vit_block_forward_dyn(
    var hidden: Tensor, prefix: String,
    config: MiniMaxH3VideoDecoderDeviceConfig,
    decoder: MiniMaxH3VideoDecoderDevice,
    cos_t: Tensor, sin_t: Tensor, rot_dim: Int, qk_ones: Tensor,
    ctx: DeviceContext,
) raises -> Tensor:
    var normed = rms_norm(hidden, decoder._w(prefix + ".norm1.weight"), config.norm_eps, ctx)
    var attn_out = _vit_attention_dyn(normed, prefix, config, decoder, cos_t, sin_t, rot_dim, qk_ones, ctx)
    hidden = add(hidden, mul(attn_out, decoder._w(prefix + ".scale1"), ctx), ctx)
    var normed2 = rms_norm(hidden, decoder._w(prefix + ".norm2.weight"), config.norm_eps, ctx)
    var ff_out = _swiglu_ff(normed2, prefix, config, decoder, ctx)
    hidden = add(hidden, mul(ff_out, decoder._w(prefix + ".scale2"), ctx), ctx)
    return hidden^


def minimax_h3_video_decode_device_batched(
    decoder: MiniMaxH3VideoDecoderDevice,
    latents: Tensor,  # [B, T, H, W, latent_channels] NDHWC — B stacked tiles
    ctx: DeviceContext,
) raises -> List[Tensor]:
    """Decode B same-shape latent tiles in ONE pass through the ViT.

    The spatially tiled product decode previously ran one full
    `minimax_h3_video_decode_device` per tile — at 1344x768x243 that is
    ~270 calls whose GEMM content is seconds but whose per-call launch/alloc
    overhead measured ~98% of a 449 s decode. Every non-attention op here is
    leading-dim agnostic, and attention runs the cuDNN dynamic entry with
    runtime B, so the whole clip's tiles cost one launch chain. Returns the
    per-tile decoded volumes in batch order (the blend/stitch consumer is
    unchanged). BF16-resident weights required (the shipped configuration)."""
    var config = decoder.config.copy()
    if config.t_causal:
        raise Error(
            "MiniMax-H3 video decoder device (batched): t_causal=True not"
            " implemented"
        )
    var ls = latents.shape()
    var Bt = ls[0]
    var lt = ls[1]
    var lh = ls[2]
    var lw = ls[3]
    var num_tokens = lt * lh * lw
    var num_suffix = 1 + config.num_register_tokens
    var S = num_tokens + num_suffix
    var dim = config.dim()

    var pqw_shape = decoder._w("post_quant_conv.weight").shape()
    var pqw2d = reshape(decoder._w("post_quant_conv.weight"), [pqw_shape[0], pqw_shape[1]], ctx)
    if pqw2d.dtype() != STDtype.BF16:
        raise Error(
            "MiniMax-H3 batched decode requires the BF16-resident decoder"
        )
    var tokens_in = reshape(latents, [Bt, num_tokens, config.latent_channels], ctx)
    if tokens_in.dtype() != STDtype.BF16:
        tokens_in = cast_tensor(tokens_in, STDtype.BF16, ctx)
    var post_quant = linear_bias(
        tokens_in, pqw2d, decoder._w("post_quant_conv.bias"), ctx,
    )
    var hidden = linear_bias(
        post_quant, decoder._w("decoder.x_embedder.weight"),
        decoder._w("decoder.x_embedder.bias"), ctx,
    )  # [B, num_tokens, dim]

    # Suffix rows: shared register tokens replicated per batch element + the
    # one zero token. Tiny tensors — a B-long concat chain is fine.
    var regs_b = decoder._w("decoder.register_tokens").clone(ctx)
    for _ in range(Bt - 1):
        regs_b = concat(0, ctx, regs_b, decoder._w("decoder.register_tokens"))
    var zero_tok = zeros_device([Bt, 1, dim], hidden.dtype(), ctx)
    var h1 = concat(1, ctx, hidden, regs_b)
    var full = concat(1, ctx, h1, zero_tok)  # [B, S, dim]

    var rope = _build_rope_tables(lt, lh, lw, num_suffix, config, ctx)
    var ones_host = List[Float32](capacity=config.dim_head)
    for _ in range(config.dim_head):
        ones_host.append(Float32(1.0))
    var qk_ones = Tensor.from_host(ones_host, [config.dim_head], STDtype.BF16, ctx)

    var current = full^
    for layer in range(config.num_layers):
        current = _vit_block_forward_dyn(
            current^, _block_prefix(layer), config, decoder,
            rope.cos, rope.sin, rope.rotary_dim, qk_ones, ctx
        )

    var normed_out = layer_norm(
        current, decoder._w("decoder.norm_out.weight"),
        decoder._w("decoder.norm_out.bias"), config.norm_eps, ctx,
    )
    var projected = linear_bias(
        normed_out, decoder._w("decoder.proj_out.weight"),
        decoder._w("decoder.proj_out.bias"), ctx,
    )  # [B, S, patch_dim]

    var out = List[Tensor]()
    var pd = projected.shape()[2]
    for b in range(Bt):
        var one = slice(projected, 0, b, 1, ctx)              # [1, S, pd]
        var patch_tokens = slice(one, 1, 0, num_tokens, ctx)  # [1, ntok, pd]
        var vol = _unpack_patches(
            patch_tokens, config.out_channels, config.patch_size_t,
            config.patch_size, lt, lh, lw, ctx,
        )
        if vol.dtype() != STDtype.F32:
            vol = cast_tensor(vol, STDtype.F32, ctx)
        out.append(vol^)
    _ = pd
    return out^


def minimax_h3_video_decode_device[S: Int, H: Int, Dh: Int](
    decoder: MiniMaxH3VideoDecoderDevice,
    latents: Tensor,  # [1, T, H, W, latent_channels] NDHWC (sampled latent)
    ctx: DeviceContext,
) raises -> Tensor:
    """`post_quant_conv` (1x1x1, plain — no causal/reflect padding, kernel=1
    needs none) then the ViT decoder. `latents` is the SAMPLED latent
    (embed_dim-wide; for the released config embed_dim==z_channels==24, so
    `post_quant_conv` is a same-width 1x1x1 conv — a real learned transform,
    not an identity). `H`/`Dh` are comptime twins of `config.heads`/
    `config.dim_head` (`ops/attention.sdpa_nomask` requires the head count
    and head dim at compile time) — checked against the runtime config
    inside `_vit_attention`."""
    var config = decoder.config.copy()
    if config.t_causal:
        raise Error(
            "MiniMax-H3 video decoder device: t_causal=True (block-causal"
            " attention mask) is not implemented — only verified for FL2VA"
            " (causal_decoder=false); see this file's SCOPE note"
        )
    var ls = latents.shape()
    var lt = ls[1]
    var lh = ls[2]
    var lw = ls[3]
    var num_tokens = lt * lh * lw
    var num_suffix = 1 + config.num_register_tokens
    if num_tokens + num_suffix != S:
        raise Error(
            "MiniMax-H3 video decoder device: S does not match"
            " latent_T*H*W + 1 + num_register_tokens"
        )
    var dim = config.dim()

    # kernel=1 Conv3d, NDHWC, treated as a per-voxel linear: flatten to
    # [1, num_tokens, embed_dim] then `linear_bias` with the conv weight
    # reshaped [z_ch, embed_dim] (the trailing 1x1x1 kernel dims are unit).
    var pqw_shape = decoder._w("post_quant_conv.weight").shape()
    var pqw2d = reshape(decoder._w("post_quant_conv.weight"), [pqw_shape[0], pqw_shape[1]], ctx)
    var tokens_in = reshape(latents, [1, num_tokens, config.latent_channels], ctx)
    # Weights are BF16-resident (see load); bridge the caller's latents dtype
    # at the entry and restore it at the exit so the pipeline contract
    # (F32 latents in, F32 pixels out) is unchanged.
    var resident_dtype = pqw2d.dtype()
    if tokens_in.dtype() != resident_dtype:
        tokens_in = cast_tensor(tokens_in, resident_dtype, ctx)
    var post_quant = linear_bias(
        tokens_in, pqw2d, decoder._w("post_quant_conv.bias"), ctx,
    )  # [1,num_tokens,z_ch]

    var hidden = linear_bias(
        post_quant, decoder._w("decoder.x_embedder.weight"),
        decoder._w("decoder.x_embedder.bias"), ctx,
    )  # [1, num_tokens, dim]

    var zero_tok = zeros_device([1, 1, dim], hidden.dtype(), ctx)
    var h1 = concat(1, ctx, hidden, decoder._w("decoder.register_tokens"))
    var full = concat(1, ctx, h1, zero_tok)  # [1, S, dim]

    var rope = _build_rope_tables(lt, lh, lw, num_suffix, config, ctx)

    # One [Dh] all-ones gain for the unweighted q/k RMSNorm, built ONCE per
    # decode call — previously rebuilt (a sync host upload) twice per block.
    var ones_host = List[Float32](capacity=config.dim_head)
    for _ in range(config.dim_head):
        ones_host.append(Float32(1.0))
    var qk_ones = Tensor.from_host(ones_host, [config.dim_head], resident_dtype, ctx)

    var current = full^
    for layer in range(config.num_layers):
        current = _vit_block_forward[S, H, Dh](
            current^, _block_prefix(layer), config, decoder, rope.cos, rope.sin, rope.rotary_dim, qk_ones, ctx
        )

    var normed_out = layer_norm(
        current, decoder._w("decoder.norm_out.weight"),
        decoder._w("decoder.norm_out.bias"), config.norm_eps, ctx,
    )
    var projected = linear_bias(
        normed_out, decoder._w("decoder.proj_out.weight"),
        decoder._w("decoder.proj_out.bias"), ctx,
    )  # [1, S, patch_dim]

    var patch_tokens = slice(projected, 1, 0, num_tokens, ctx)
    var out = _unpack_patches(
        patch_tokens, config.out_channels, config.patch_size_t, config.patch_size,
        lt, lh, lw, ctx,
    )
    if out.dtype() != latents.dtype():
        out = cast_tensor(out, latents.dtype(), ctx)
    return out^

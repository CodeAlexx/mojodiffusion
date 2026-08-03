# serenitymojo/models/vae/minimax_h3_video_encoder_device.mojo
#
# MiniMax-H3 video encoder — DEVICE path, authority = the CREATOR SOURCE
# (`EncoderFCN3D` in the released `video_vae/vae_cnn.py`, verified against
# `video_vae/source/config.json`), NOT the diffusers PR rewrite.
#
# `models/minimax_h3/video_encoder.mojo` (unit 13) stays untouched: it is
# gated against `MiniMaxH3VideoEncoder3d` (the diffusers rewrite) and is
# correct FOR THAT reference. This file is a SEPARATE port whose oracle is
# the actual released checkpoint's module tree, confirmed key-for-key by
# reading klvae.py/vae_cnn.py/conv.py/norm.py in full (2026-08-02 audit) and
# by `minimax_h3_video_vae.py`'s `load_state_dict(strict=True)`, which proves
# there is no hidden key-conversion step anywhere in the release bundle.
#
# WHAT WAS VERIFIED FAITHFUL in the diffusers-oracle port and is CARRIED OVER
# here unchanged (same audit): channel progression 128,128/128,256/256,256/
# 256,512/512,512/512,1024 (ch=128, ch_mult=[1,2,2,4,4,8]); symmetric REFLECT
# spatial pad ±1 on H/W for every kernel-3 conv (conv.py BaseConv3d.
# _apply_padding:62-73, `padding_mode:"reflect"`); CAUSAL temporal pad, ALL on
# the front, amount = kernel-1 (conv.py:40-51, `padding[0]*2` when causal);
# the downsampler's ASYMMETRIC bottom/right reflect pad of 1 BEFORE a
# zero-spatial-pad conv (vae_cnn.py Downsample3D.forward:65-80); per-FRAME
# GroupNorm (`use_t_isolated_gn: true` in the released config selects
# `TemporalIsolatedSpatialParallelGroupNorm`, norm.py:245-252 — folds depth
# into batch, groups=32, eps=1e-6); the resnet block order norm1->silu->
# conv1->norm2->silu->conv2, +shortcut only on channel change (vae_cnn.py
# ResnetBlock3D.forward:154-174).
#
# WHAT IS DIFFERENT FROM `video_encoder.mojo` (the actual fix):
#   * NATIVE key names, not diffusers spellings:
#       encoder.down.{lvl}.block.{i}.{norm1,conv1,norm2,conv2,nin_shortcut}
#       encoder.down.{lvl}.downsample.conv
#     (vs the diffusers oracle's down_blocks/resnets/conv_shortcut/
#     downsamplers.0.conv). Leaf suffixes (norm1.weight, conv1.weight, ...)
#     are unchanged — only the path segments differ.
#   * Runs on the GPU as `Tensor` ops (`conv3d_fcqrs_cudnn`, `group_norm`,
#     `silu`), not host `List[Float32]` loops.
#
# LAYOUT CONTRACT: NDHWC throughout (`[N, D, H, W, C]`), matching this repo's
# house convention for 3D conv (`models/vae/conv3d.mojo` header) and
# `conv3d_fcqrs_cudnn`'s expected input. Conv weights are consumed RAW OIDHW
# (== FCQRS == `[Cout, Cin, Q, R, S]`, the literal PyTorch `nn.Conv3d.weight`
# on-disk layout) — zero host-side transpose, matching how
# `conv3d_fcqrs_cudnn` is already used for LTX2/Wan2.2/Qwen-Image VAEs
# (MAP.md `models/vae/conv3d.mojo` entry).
#
# PIXEL NORMALIZATION (ImageNet mean/std, `pixel_norm_type: imagenet`,
# normalize.py) is OUT OF SCOPE for this file: it is a pre-encode/post-decode
# transform on raw RGB, not part of the encoder/decoder network, and belongs
# in whatever pipeline calls this module. Same for the clip_length=17 /
# token_drop=3 temporal chunk+blend loop (klvae.py encode_temporal) — this
# file encodes ONE volume in a single pass, no chunking.
#
# REUSE, not reimplemented:
#   models/vae/conv3d.mojo        conv3d_fcqrs_cudnn (native OIDHW conv)
#   ops/norm.mojo                 group_norm (rank-4 NHWC)
#   ops/activations.mojo          silu
#   ops/tensor_algebra.mojo       slice, concat, zeros_device, reshape,
#                                 reshape_owned, add
#   io/sharded.mojo               ShardedSafeTensors
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.collections import Dict, List
from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.tensor import Tensor
from serenitymojo.ops.activations import silu
from serenitymojo.ops.norm import group_norm
from serenitymojo.ops.tensor_algebra import (
    add, concat, reshape, reshape_owned, slice, zeros_device,
)
from serenitymojo.models.vae.conv3d import conv3d_fcqrs_cudnn

comptime TArc = ArcPointer[Tensor]


@fieldwise_init
struct MiniMaxH3VideoEncoderDeviceConfig(Copyable, Movable):
    """Straight from `video_vae/source/config.json`. Field names match the
    ORIGINAL config's own spelling (ch, ch_mult, space_down, time_down,
    num_res_blocks), not diffusers' (block_out_channels, layers_per_block) —
    deliberate, so a config value can be checked against the JSON by eye."""

    var ch: Int
    var ch_mult: List[Int]
    var space_down: List[Int]
    var time_down: List[Int]
    var num_res_blocks: Int
    var in_channels: Int
    var z_channels: Int
    var norm_groups: Int
    var norm_eps: Float32

    def levels(self) -> Int:
        return len(self.ch_mult)

    def block_mid(self, level: Int) -> Int:
        return self.ch * self.ch_mult[level]

    def block_in(self, level: Int) -> Int:
        return self.block_mid(0) if level == 0 else self.block_mid(level - 1)


def minimax_h3_video_released_encoder_config() -> MiniMaxH3VideoEncoderDeviceConfig:
    """FL2VA `video_vae/source/config.json`, verbatim: ch=128,
    ch_mult=[1,2,2,4,4,8], space_down=[2,2,2,2,1,1], time_down=[1,2,2,1,1,1],
    num_res_blocks=2, in_channels=3, z_channels=24 (== embed_dim for this
    release), groups=32, eps=1e-6."""
    return MiniMaxH3VideoEncoderDeviceConfig(
        128,
        [1, 2, 2, 4, 4, 8],
        [2, 2, 2, 2, 1, 1],
        [1, 2, 2, 1, 1, 1],
        2, 3, 24, 32, Float32(1.0e-6),
    )


def minimax_h3_video_encoder_block_prefix(level: Int, index: Int) -> String:
    return (
        String("encoder.down.") + String(level) + ".block." + String(index)
    )


def minimax_h3_video_encoder_downsample_prefix(level: Int) -> String:
    return String("encoder.down.") + String(level) + ".downsample.conv"


def minimax_h3_video_encoder_key_names(
    config: MiniMaxH3VideoEncoderDeviceConfig,
) raises -> List[String]:
    """Every tensor `EncoderFCN3D` needs, in the NATIVE checkpoint namespace.
    `nin_shortcut` only exists when a block's in/out channel count differs;
    `downsample.conv` only exists when `space_down*time_down > 1` for that
    level — both mirror vae_cnn.py's own conditionals exactly (ResnetBlock3D
    :144-152, EncoderFCN3D.__init__:241-256)."""
    var names = List[String]()
    names.append("encoder.conv_in.weight")
    names.append("encoder.conv_in.bias")
    for level in range(config.levels()):
        var out_ch = config.block_mid(level)
        var in_ch = config.block_in(level)
        for i in range(config.num_res_blocks):
            var block_in = in_ch if i == 0 else out_ch
            var prefix = minimax_h3_video_encoder_block_prefix(level, i)
            names.append(prefix + ".norm1.weight")
            names.append(prefix + ".norm1.bias")
            names.append(prefix + ".conv1.weight")
            names.append(prefix + ".conv1.bias")
            names.append(prefix + ".norm2.weight")
            names.append(prefix + ".norm2.bias")
            names.append(prefix + ".conv2.weight")
            names.append(prefix + ".conv2.bias")
            if block_in != out_ch:
                names.append(prefix + ".nin_shortcut.weight")
                names.append(prefix + ".nin_shortcut.bias")
        if config.space_down[level] * config.time_down[level] > 1:
            var dprefix = minimax_h3_video_encoder_downsample_prefix(level)
            names.append(dprefix + ".weight")
            names.append(dprefix + ".bias")
    names.append("encoder.norm_out.weight")
    names.append("encoder.norm_out.bias")
    names.append("encoder.conv_out.weight")
    names.append("encoder.conv_out.bias")
    names.append("quant_conv.weight")
    names.append("quant_conv.bias")
    return names^


struct MiniMaxH3VideoEncoderDevice(Movable):
    var weights: List[TArc]
    var name_to_idx: Dict[String, Int]
    var config: MiniMaxH3VideoEncoderDeviceConfig

    def __init__(
        out self,
        var weights: List[TArc],
        var name_to_idx: Dict[String, Int],
        config: MiniMaxH3VideoEncoderDeviceConfig,
    ):
        self.weights = weights^
        self.name_to_idx = name_to_idx^
        self.config = config.copy()

    def _w(self, name: String) raises -> ref [self.weights] Tensor:
        if name not in self.name_to_idx:
            raise Error(
                String("MiniMax-H3 video encoder device: missing weight ") + name
            )
        return self.weights[self.name_to_idx[name]][]

    def _bias(self, name: String, ctx: DeviceContext) raises -> Tensor:
        return self._w(name).clone(ctx)

    def has(self, name: String) -> Bool:
        return name in self.name_to_idx

    @staticmethod
    def load(
        dir: String, config: MiniMaxH3VideoEncoderDeviceConfig, ctx: DeviceContext
    ) raises -> MiniMaxH3VideoEncoderDevice:
        """Preflight every expected tensor (presence only — shape/dtype
        mismatches surface as loud errors from the ops that consume them),
        then stream each into a device Tensor via `Tensor.from_view` (bf16 or
        f32, whatever the checkpoint stores)."""
        var shards = ShardedSafeTensors.open(dir)
        var names = minimax_h3_video_encoder_key_names(config)
        for i in range(len(names)):
            if names[i] not in shards.name_to_shard:
                raise Error(
                    String("MiniMax-H3 video encoder device: missing weight ")
                    + names[i] + " in " + dir
                )
        var weights = List[TArc]()
        var name_to_idx = Dict[String, Int]()
        for i in range(len(names)):
            var t = Tensor.from_view(shards.tensor_view(names[i]), ctx)
            name_to_idx[names[i]] = len(weights)
            weights.append(TArc(t^))
        return MiniMaxH3VideoEncoderDevice(weights^, name_to_idx^, config)


# ─────────────────────────────────────────────────────────────────────────────
# Padding — built entirely from `slice`/`concat`/`zeros_device` (no new GPU
# kernel): each is a small, already-gated D2D copy, composed.
# ─────────────────────────────────────────────────────────────────────────────
def _reflect_pad1_hw(var x: Tensor, pad: Int, ctx: DeviceContext) raises -> Tensor:
    """Symmetric REFLECT pad of `pad` on H and W (NDHWC axes 2,3). This model
    only ever uses pad 0 or 1 (kernel 1 or 3 convs), so only those are
    supported — anything else fails loud rather than guessing a formula."""
    if pad == 0:
        return x^
    if pad != 1:
        raise Error("MiniMax-H3 video encoder device: only reflect pad 0/1 supported")
    var s = x.shape()
    var h = s[2]
    var w = s[3]
    var top = slice(x, 2, 1, 1, ctx)
    var bottom = slice(x, 2, h - 2, 1, ctx)
    var xh = concat(2, ctx, top, x, bottom)
    var left = slice(xh, 3, 1, 1, ctx)
    var right = slice(xh, 3, w - 2, 1, ctx)
    return concat(3, ctx, left, xh, right)


def _reflect_pad_bottom_right1(var x: Tensor, ctx: DeviceContext) raises -> Tensor:
    """The downsampler's ASYMMETRIC pad: one reflected row/col appended at
    the bottom/right only (vae_cnn.py Downsample3D.forward:78, `F.pad(x,
    (0,1,0,1,0,0), mode="reflect")`)."""
    var s = x.shape()
    var h = s[2]
    var w = s[3]
    var bottom = slice(x, 2, h - 2, 1, ctx)
    var xh = concat(2, ctx, x, bottom)
    var right = slice(xh, 3, w - 2, 1, ctx)
    return concat(3, ctx, xh, right)


def _causal_zero_pad_t(var x: Tensor, pad: Int, ctx: DeviceContext) raises -> Tensor:
    """CAUSAL zero pad on D (NDHWC axis 1): `pad` empty frames prepended,
    none appended (conv.py:40-51, causal branch: all padding on the front,
    doubled — for a padding=1/kernel=3 conv that's exactly `pad=2` here)."""
    if pad == 0:
        return x^
    var s = x.shape()
    var zshape = s.copy()
    zshape[1] = pad
    var z = zeros_device(zshape^, x.dtype(), ctx)
    return concat(1, ctx, z, x)


def _causal_conv3d(
    x: Tensor, weight: Tensor, var bias: Tensor,
    spatial_pad: Int, temporal_pad: Int, stride_t: Int, stride_s: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    """Reflect the spatial axes (if `spatial_pad>0`), causally zero-pad time
    (if `temporal_pad>0`), then convolve with NO further padding — mirrors
    `models/minimax_h3/video_encoder.mojo`'s `causal_conv3d` (the ALREADY-
    VERIFIED host oracle math), on GPU Tensors via `conv3d_fcqrs_cudnn`."""
    var padded_hw = _reflect_pad1_hw(x.clone(ctx), spatial_pad, ctx)
    var padded = _causal_zero_pad_t(padded_hw^, temporal_pad, ctx)
    return conv3d_fcqrs_cudnn(
        padded, weight, Optional[Tensor](bias^),
        stride_t, stride_s, stride_s, 0, 0, 0, ctx,
    )


def _group_norm_per_frame(
    x: Tensor, num_groups: Int, weight: Tensor, bias: Tensor,
    eps: Float32, ctx: DeviceContext,
) raises -> Tensor:
    """GroupNorm with TIME FOLDED INTO BATCH — statistics per frame, matching
    `TemporalIsolatedSpatialParallelGroupNorm` (norm.py:245-252,
    `_merge_time_to_batch`/`_split_time_from_batch`). `reshape` merges
    [B,D,H,W,C] -> [B*D,H,W,C] (D2D copy, since `x` is borrowed); the
    unmerge back to 5-D is metadata-only (`reshape_owned`) since
    `group_norm`'s output is a fresh tensor this function owns."""
    var s = x.shape()
    var merged = reshape(x, [s[0] * s[1], s[2], s[3], s[4]], ctx)
    var normed = group_norm(merged, weight, bias, num_groups, eps, ctx)
    return reshape_owned(normed^, s.copy())


def _resnet_block_forward(
    x: Tensor, prefix: String, in_ch: Int, out_ch: Int,
    config: MiniMaxH3VideoEncoderDeviceConfig,
    encoder: MiniMaxH3VideoEncoderDevice, ctx: DeviceContext,
) raises -> Tensor:
    """norm1 -> silu -> conv1 -> norm2 -> silu -> conv2, +1x1x1 shortcut only
    on a channel change (vae_cnn.py ResnetBlock3D.forward:154-174)."""
    var h = _group_norm_per_frame(
        x, config.norm_groups, encoder._w(prefix + ".norm1.weight"),
        encoder._w(prefix + ".norm1.bias"), config.norm_eps, ctx,
    )
    h = silu(h, ctx)
    h = _causal_conv3d(
        h, encoder._w(prefix + ".conv1.weight"), encoder._bias(prefix + ".conv1.bias", ctx),
        1, 2, 1, 1, ctx,
    )
    h = _group_norm_per_frame(
        h, config.norm_groups, encoder._w(prefix + ".norm2.weight"),
        encoder._w(prefix + ".norm2.bias"), config.norm_eps, ctx,
    )
    h = silu(h, ctx)
    h = _causal_conv3d(
        h, encoder._w(prefix + ".conv2.weight"), encoder._bias(prefix + ".conv2.bias", ctx),
        1, 2, 1, 1, ctx,
    )
    var residual: Tensor
    if in_ch != out_ch:
        residual = _causal_conv3d(
            x, encoder._w(prefix + ".nin_shortcut.weight"),
            encoder._bias(prefix + ".nin_shortcut.bias", ctx),
            0, 0, 1, 1, ctx,
        )
    else:
        residual = x.clone(ctx)
    return add(h, residual, ctx)


def minimax_h3_video_encode_device(
    encoder: MiniMaxH3VideoEncoderDevice, pixels: Tensor, ctx: DeviceContext,
) raises -> Tensor:
    """`pixels`: `[1, T, H, W, in_channels]` NDHWC (already pixel-normalized
    by the caller if the pipeline wants ImageNet stats — see this file's
    header). Returns moments `[1, T', H', W', 2*z_channels]`, i.e.
    `quant_conv(encoder(pixels))`, exactly `EncoderFCN3D.forward` +
    `AutoencoderKLLegacy.encode`'s `quant_conv` step."""
    var config = encoder.config.copy()
    var h = _causal_conv3d(
        pixels, encoder._w("encoder.conv_in.weight"),
        encoder._bias("encoder.conv_in.bias", ctx), 1, 2, 1, 1, ctx,
    )
    for level in range(config.levels()):
        var out_ch = config.block_mid(level)
        var in_ch = config.block_in(level)
        for i in range(config.num_res_blocks):
            var block_in = in_ch if i == 0 else out_ch
            var prefix = minimax_h3_video_encoder_block_prefix(level, i)
            h = _resnet_block_forward(h, prefix, block_in, out_ch, config, encoder, ctx)
        if config.space_down[level] * config.time_down[level] > 1:
            var dprefix = minimax_h3_video_encoder_downsample_prefix(level)
            var pre: Tensor
            if config.space_down[level] == 2:
                pre = _reflect_pad_bottom_right1(h.clone(ctx), ctx)
            else:
                pre = h.clone(ctx)
            h = _causal_conv3d(
                pre, encoder._w(dprefix + ".weight"), encoder._bias(dprefix + ".bias", ctx),
                0, 2, config.time_down[level], config.space_down[level], ctx,
            )
    h = _group_norm_per_frame(
        h, config.norm_groups, encoder._w("encoder.norm_out.weight"),
        encoder._w("encoder.norm_out.bias"), config.norm_eps, ctx,
    )
    h = silu(h, ctx)
    h = _causal_conv3d(
        h, encoder._w("encoder.conv_out.weight"),
        encoder._bias("encoder.conv_out.bias", ctx), 1, 2, 1, 1, ctx,
    )
    return _causal_conv3d(
        h, encoder._w("quant_conv.weight"),
        encoder._bias("quant_conv.bias", ctx), 0, 0, 1, 1, ctx,
    )

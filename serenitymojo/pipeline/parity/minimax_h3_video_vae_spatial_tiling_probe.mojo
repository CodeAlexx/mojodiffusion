# pipeline/parity/minimax_h3_video_vae_spatial_tiling_probe.mojo —
# compile+run gate for pipeline/minimax_h3_video_vae_spatial_tiling.mojo.
#
#   1. `split_tiles` arithmetic, hand-traced: input_len=48, tile_size=32,
#      tile_overlap_min=4, vae_ratio=4 -> N=2, start=[0,16], overlap=[16]
#      (every intermediate number traced by hand against split_tiles's own
#      formula, not just "it ran").
#   2. `tiled_encode` SEAM TEST (H-axis only: W fits in one tile so only
#      the "new" axis is exercised). Verifies the blended seam differs from
#      an UNBLENDED encode of the same tile at the same position — both
#      computed on GPU, compared bit-for-bit.
#   3. `tiled_decode` SEAM TEST, same structure, on the decoder.
#
#   pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     -Xlinker -Lserenitymojo/ops/cshim/lib/cudnn_stubs -Xlinker -lcudnn \
#     pipeline/parity/minimax_h3_video_vae_spatial_tiling_probe.mojo \
#     -o /tmp/mmh3_tiling_probe
#   LD_LIBRARY_PATH=serenitymojo/ops/cshim/lib:serenitymojo/ops/cshim/lib/cudnn_stubs \
#     /tmp/mmh3_tiling_probe

from std.collections import List
from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.io.ffi import sys_remove
from serenitymojo.ops.tensor_algebra import slice

from serenitymojo.models.vae.minimax_h3_video_encoder_device import (
    MiniMaxH3VideoEncoderDevice, MiniMaxH3VideoEncoderDeviceConfig,
    minimax_h3_video_encode_device, minimax_h3_video_encoder_block_prefix,
    minimax_h3_video_encoder_downsample_prefix, minimax_h3_video_encoder_key_names,
)
from serenitymojo.models.vae.minimax_h3_video_decoder_device import (
    minimax_h3_video_decode_device, minimax_h3_video_decoder_device_load,
    minimax_h3_video_decoder_native_key_names, MiniMaxH3VideoDecoderDeviceConfig,
)
from serenitymojo.pipeline.minimax_h3_video_vae_spatial_tiling import (
    MiniMaxH3TilingConfig, minimax_h3_split_tiles, minimax_h3_video_tiled_decode,
    minimax_h3_video_tiled_encode,
)

comptime TArc = ArcPointer[Tensor]
comptime ENC_CKPT = "/tmp/minimax_h3_tiling_probe_enc.safetensors"
comptime DEC_CKPT = "/tmp/minimax_h3_tiling_probe_dec.safetensors"


def _pattern(seed: Int, n: Int) -> List[Float32]:
    var out = List[Float32](capacity=n)
    for i in range(n):
        var v = ((seed * 1103515245 + i * 12345 + 7) % 2003) - 1001
        out.append(Float32(v) * Float32(0.01))
    return out^


def _max_abs_diff(a: List[Float32], b: List[Float32]) raises -> Float32:
    if len(a) != len(b):
        raise Error("probe: _max_abs_diff length mismatch")
    var m = Float32(0.0)
    for i in range(len(a)):
        var d = a[i] - b[i]
        if d < Float32(0.0):
            d = -d
        if d > m:
            m = d
    return m


# ── phase 1: split_tiles arithmetic, hand-traced ──────────────────────────────
def _run_split_tiles_check() raises:
    var layout = minimax_h3_split_tiles(48, 32, 4, 4)
    print("  start_idx=", layout.start_idx, " lengths=", layout.lengths, " overlaps=", layout.overlaps)
    if layout.num_tiles() != 2:
        raise Error("probe: FAIL split_tiles tile count")
    if layout.start_idx != [0, 16]:
        raise Error("probe: FAIL split_tiles start_idx")
    if layout.lengths != [32, 32]:
        raise Error("probe: FAIL split_tiles lengths")
    if layout.overlaps != [16]:
        raise Error("probe: FAIL split_tiles overlaps")
    # single-tile fast path.
    var single = minimax_h3_split_tiles(20, 32, 4, 4)
    if single.num_tiles() != 1 or single.start_idx != [0] or single.lengths != [20]:
        raise Error("probe: FAIL split_tiles single-tile fast path")
    print("phase 1 (split_tiles arithmetic, hand-traced) PASS")


# ── phase 2: tiled_encode SEAM TEST (H-axis) ──────────────────────────────────
def _toy_encoder_config() -> MiniMaxH3VideoEncoderDeviceConfig:
    # single level, NO internal spatial/temporal downsampling (space_down=
    # time_down=1 -> vae_ratio=1) so latent H/W equal pixel H/W directly --
    # isolates the TILING seam logic from the encoder's own downsample math
    # (already gated separately in models/vae/parity/
    # minimax_h3_video_vae_device_probe.mojo).
    return MiniMaxH3VideoEncoderDeviceConfig(
        4, [1], [1], [1], 1, 2, 2, 2, Float32(1.0e-6),
    )


def _encoder_weight_shape(key: String, config: MiniMaxH3VideoEncoderDeviceConfig) raises -> List[Int]:
    if key == "encoder.conv_in.weight":
        return [config.block_mid(0), config.in_channels, 3, 3, 3]
    if key == "encoder.conv_in.bias":
        return [config.block_mid(0)]
    if key == "encoder.norm_out.weight" or key == "encoder.norm_out.bias":
        return [config.block_mid(config.levels() - 1)]
    if key == "encoder.conv_out.weight":
        return [2 * config.z_channels, config.block_mid(config.levels() - 1), 3, 3, 3]
    if key == "encoder.conv_out.bias":
        return [2 * config.z_channels]
    if key == "quant_conv.weight":
        return [2 * config.z_channels, 2 * config.z_channels, 1, 1, 1]
    if key == "quant_conv.bias":
        return [2 * config.z_channels]
    for level in range(config.levels()):
        var out_ch = config.block_mid(level)
        var in_ch = config.block_in(level)
        for i in range(config.num_res_blocks):
            var block_in = in_ch if i == 0 else out_ch
            var prefix = minimax_h3_video_encoder_block_prefix(level, i)
            if key == prefix + ".norm1.weight" or key == prefix + ".norm1.bias":
                return [block_in]
            if key == prefix + ".conv1.weight":
                return [out_ch, block_in, 3, 3, 3]
            if key == prefix + ".conv1.bias":
                return [out_ch]
            if key == prefix + ".norm2.weight" or key == prefix + ".norm2.bias":
                return [out_ch]
            if key == prefix + ".conv2.weight":
                return [out_ch, out_ch, 3, 3, 3]
            if key == prefix + ".conv2.bias":
                return [out_ch]
            if key == prefix + ".nin_shortcut.weight":
                return [out_ch, block_in, 1, 1, 1]
            if key == prefix + ".nin_shortcut.bias":
                return [out_ch]
        var dprefix = minimax_h3_video_encoder_downsample_prefix(level)
        if key == dprefix + ".weight":
            return [out_ch, out_ch, 3, 3, 3]
        if key == dprefix + ".bias":
            return [out_ch]
    raise Error(String("probe: no shape rule for encoder key ") + key)


def _write_encoder_checkpoint(config: MiniMaxH3VideoEncoderDeviceConfig, ctx: DeviceContext) raises:
    var names = minimax_h3_video_encoder_key_names(config)
    var tensors = List[TArc]()
    var seed = 500
    for i in range(len(names)):
        seed += 1
        var shape = _encoder_weight_shape(names[i], config)
        var n = 1
        for d in range(len(shape)):
            n *= shape[d]
        var host = _pattern(seed, n)
        tensors.append(TArc(Tensor.from_host(host, shape^, STDtype.BF16, ctx)))
    save_safetensors(names, tensors, String(ENC_CKPT), ctx)


def _run_tiled_encode_seam_test(ctx: DeviceContext) raises:
    var config = _toy_encoder_config()
    _write_encoder_checkpoint(config, ctx)
    var encoder = MiniMaxH3VideoEncoderDevice.load(String(ENC_CKPT), config, ctx)
    var tiling = MiniMaxH3TilingConfig(4, 1, 4, 1, 1)  # vae_ratio=1, matches this toy encoder

    # [1, T=2, H=6, W=4, C=2] -- H splits into 2 tiles ([0,4),[2,6), overlap
    # 2), W fits in one tile (4<=tile_size 4) so only the H seam is live.
    var pixels = Tensor.from_host(
        _pattern(900, 1 * 2 * 6 * 4 * 2), [1, 2, 6, 4, 2], STDtype.BF16, ctx
    )
    var moments = minimax_h3_video_tiled_encode(encoder, pixels, tiling, ctx)
    print("  tiled_encode output shape:", moments.shape())
    if moments.shape() != [1, 2, 6, 4, 4]:
        raise Error("probe: FAIL tiled_encode output shape (vae_ratio=1 -> H,W unchanged)")

    # UNBLENDED reference: encode tile 1 (pixel rows [2,6)) directly, take
    # its own first 2 rows (the region that got blended in the tiled path).
    var tile1_pixels = slice(pixels, 2, 2, 4, ctx)
    var tile1_moments = minimax_h3_video_encode_device(encoder, tile1_pixels, ctx)
    var tile1_unblended_seam = slice(tile1_moments, 2, 0, 2, ctx)

    # The tiled output's rows [2,4) are exactly that seam region, blended.
    var tiled_seam = slice(moments, 2, 2, 2, ctx)

    var diff = _max_abs_diff(tile1_unblended_seam.to_host(ctx), tiled_seam.to_host(ctx))
    print("  encode seam max_abs diff (blended vs unblended tile1):", diff)
    if diff == Float32(0.0):
        raise Error("probe: FAIL tiled_encode seam is bit-identical to an unblended tile -- no real blend")

    _ = sys_remove(String(ENC_CKPT))
    print("phase 2 (tiled_encode SEAM TEST, H-axis) PASS")


# ── phase 3: tiled_decode SEAM TEST (H-axis) ──────────────────────────────────
def _toy_decoder_config() -> MiniMaxH3VideoDecoderDeviceConfig:
    return MiniMaxH3VideoDecoderDeviceConfig(
        2, 2, 1, 2, 8, 1, 1, 1, Float64(100.0), Float64(0.75), Float32(1.0e-5), False, 4,
    )


def _decoder_weight_shape(key: String, config: MiniMaxH3VideoDecoderDeviceConfig) raises -> List[Int]:
    var dim = config.dim()
    if key == "decoder.x_embedder.weight":
        return [dim, config.latent_channels]
    if key == "decoder.x_embedder.bias":
        return [dim]
    if key == "decoder.register_tokens":
        return [1, config.num_register_tokens, dim]
    if key == "decoder.norm_out.weight" or key == "decoder.norm_out.bias":
        return [dim]
    if key == "decoder.proj_out.weight":
        var patch_dim = config.out_channels * config.patch_size_t * config.patch_size * config.patch_size
        return [patch_dim, dim]
    if key == "decoder.proj_out.bias":
        var patch_dim2 = config.out_channels * config.patch_size_t * config.patch_size * config.patch_size
        return [patch_dim2]
    if key == "post_quant_conv.weight":
        return [config.latent_channels, config.latent_channels, 1, 1, 1]
    if key == "post_quant_conv.bias":
        return [config.latent_channels]
    for layer in range(config.num_layers):
        var p = String("decoder.transformer_blocks.") + String(layer)
        if key == p + ".norm1.weight":
            return [dim]
        if key == p + ".attn.to_qkv.weight":
            return [3 * dim, dim]
        if key == p + ".attn.to_qkv.bias":
            return [3 * dim]
        if key == p + ".attn.to_out.weight":
            return [dim, dim]
        if key == p + ".attn.to_out.bias":
            return [dim]
        if key == p + ".scale1" or key == p + ".scale2" or key == p + ".norm2.weight":
            return [dim]
        if key == p + ".ff.w1.weight":
            return [2 * dim * config.ffn_mult, dim]
        if key == p + ".ff.w1.bias":
            return [2 * dim * config.ffn_mult]
        if key == p + ".ff.w2.weight":
            return [dim, dim * config.ffn_mult]
        if key == p + ".ff.w2.bias":
            return [dim]
    raise Error(String("probe: no shape rule for decoder key ") + key)


def _write_decoder_checkpoint(config: MiniMaxH3VideoDecoderDeviceConfig, ctx: DeviceContext) raises:
    var names = minimax_h3_video_decoder_native_key_names(config)
    var tensors = List[TArc]()
    var seed = 600
    for i in range(len(names)):
        seed += 1
        var shape = _decoder_weight_shape(names[i], config)
        var n = 1
        for d in range(len(shape)):
            n *= shape[d]
        var host = _pattern(seed, n)
        tensors.append(TArc(Tensor.from_host(host, shape^, STDtype.BF16, ctx)))
    save_safetensors(names, tensors, String(DEC_CKPT), ctx)


def _run_tiled_decode_seam_test(ctx: DeviceContext) raises:
    var config = _toy_decoder_config()
    _write_decoder_checkpoint(config, ctx)
    var decoder = minimax_h3_video_decoder_device_load(String(DEC_CKPT), config, ctx)
    var tiling = MiniMaxH3TilingConfig(4, 1, 4, 1, 1)  # vae_ratio=1 (patch_size=1 toy decoder)

    # latent [1, T=1, H=6, W=4, latent_channels=2] -- same H-splits-into-2 layout.
    var latents = Tensor.from_host(
        _pattern(1000, 1 * 1 * 6 * 4 * 2), [1, 1, 6, 4, 2], STDtype.BF16, ctx
    )
    var pixels = minimax_h3_video_tiled_decode[4, 4, 2, 8, 2, 1](decoder, latents, tiling, ctx)
    print("  tiled_decode output shape:", pixels.shape())
    if pixels.shape() != [1, 1, 6, 4, 2]:
        raise Error("probe: FAIL tiled_decode output shape (patch_size=1 -> H,W unchanged)")

    var tile1_latent = slice(latents, 2, 2, 4, ctx)
    var tile1_pixels = minimax_h3_video_decode_device[18, 2, 8](decoder, tile1_latent, ctx)
    var tile1_unblended_seam = slice(tile1_pixels, 2, 0, 2, ctx)
    var tiled_seam = slice(pixels, 2, 2, 2, ctx)

    var diff = _max_abs_diff(tile1_unblended_seam.to_host(ctx), tiled_seam.to_host(ctx))
    print("  decode seam max_abs diff (blended vs unblended tile1):", diff)
    if diff == Float32(0.0):
        raise Error("probe: FAIL tiled_decode seam is bit-identical to an unblended tile -- no real blend")

    _ = sys_remove(String(DEC_CKPT))
    print("phase 3 (tiled_decode SEAM TEST, H-axis) PASS")


def main() raises:
    var ctx = DeviceContext()
    print("== phase 1: split_tiles arithmetic ==")
    _run_split_tiles_check()
    print("== phase 2: tiled_encode SEAM TEST ==")
    _run_tiled_encode_seam_test(ctx)
    print("== phase 3: tiled_decode SEAM TEST ==")
    _run_tiled_decode_seam_test(ctx)
    print("minimax_h3_video_vae_spatial_tiling_probe PASS")

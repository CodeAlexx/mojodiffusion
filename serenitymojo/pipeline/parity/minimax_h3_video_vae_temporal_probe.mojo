# pipeline/parity/minimax_h3_video_vae_temporal_probe.mojo — compile+run gate
# for pipeline/minimax_h3_video_vae_temporal.mojo.
#
# Per house rule ([[no-cpu-parity-for-gpu]]), all numeric comparisons here
# are GPU-vs-GPU (values read back via `.to_host()` only for the final
# comparison, never computed on the CPU as a "reference").
#
#   1. Config arithmetic (REAL FL2VA numbers, clip_length=17/vae_ratio_t=4/
#      token_drop=3): tokens_chunk_size=5, frame_pre_padding=3,
#      token_overlap=2, frame_overlap=5 — hand-derived from setup_forward's
#      formulas, asserted exactly.
#   2. `_blend_frames` in isolation: two small known tensors, full overlap,
#      output checked against the hand-computed linear-interpolation values.
#   3. `minimax_h3_video_encode_temporal`, toy scale: a 10-frame input with
#      clip_length=7 pads to 14 (2 chunks), each chunk's causal-downsample-2
#      encode gives 4 latent tokens (8 total), token_drop=1 trims to 7 —
#      output token count asserted exactly.
#   4. THE SEAM TEST (what team-lead asked for): `minimax_h3_video_decode_
#      temporal`, toy scale, 2 chunks. Verifies (a) the total output frame
#      count matches a hand-computed value (19 raw frames before the
#      pad-frame trim, 14 after — every intermediate number in
#      `_decode_temporal_pad_frames` traced by hand) and (b) the blended
#      seam is a REAL blend, not a silent bypass: the first `frame_overlap`
#      frames of chunk 2's contribution differ from what an UNBLENDED
#      decode of chunk 2 alone would have produced at those same positions
#      (both computed on GPU, compared bit-for-bit — a discriminating test,
#      not a tautology).
#
#   pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     -Xlinker -Lserenitymojo/ops/cshim/lib/cudnn_stubs -Xlinker -lcudnn \
#     pipeline/parity/minimax_h3_video_vae_temporal_probe.mojo \
#     -o /tmp/mmh3_temporal_probe
#   LD_LIBRARY_PATH=serenitymojo/ops/cshim/lib:serenitymojo/ops/cshim/lib/cudnn_stubs \
#     /tmp/mmh3_temporal_probe
#
# (needs the cuDNN shim — the per-volume decode this layer calls uses
# conv3d_fcqrs_cudnn's BF16 path; plain `mojo run` cannot link it.)

from std.collections import List
from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.io.ffi import sys_remove
from serenitymojo.ops.tensor_algebra import slice

from serenitymojo.models.vae.minimax_h3_video_encoder_device import (
    MiniMaxH3VideoEncoderDevice, MiniMaxH3VideoEncoderDeviceConfig,
    minimax_h3_video_encoder_block_prefix, minimax_h3_video_encoder_downsample_prefix,
    minimax_h3_video_encoder_key_names,
)
from serenitymojo.models.vae.minimax_h3_video_decoder_device import (
    minimax_h3_video_decode_device, minimax_h3_video_decoder_device_load,
    minimax_h3_video_decoder_native_key_names, MiniMaxH3VideoDecoderDeviceConfig,
)
from serenitymojo.pipeline.minimax_h3_video_vae_temporal import (
    MiniMaxH3VideoTemporalConfig, _blend_frames, _decode_temporal_pad_frames,
    minimax_h3_video_decode_temporal, minimax_h3_video_encode_temporal,
    minimax_h3_video_released_temporal_config,
)

comptime TArc = ArcPointer[Tensor]
comptime ENC_CKPT = "/tmp/minimax_h3_temporal_probe_enc.safetensors"
comptime DEC_CKPT = "/tmp/minimax_h3_temporal_probe_dec.safetensors"


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


# ── phase 1: config arithmetic (real FL2VA numbers) ────────────────────────────
def _run_config_arithmetic_check() raises:
    var tc = minimax_h3_video_released_temporal_config()
    print("  clip_length=", tc.clip_length, " vae_ratio_t=", tc.vae_ratio_t, " token_drop=", tc.token_drop)
    print("  tokens_chunk_size=", tc.tokens_chunk_size(), " frame_pre_padding=", tc.frame_pre_padding(),
        " token_overlap=", tc.token_overlap(), " frame_overlap=", tc.frame_overlap())
    if tc.tokens_chunk_size() != 5:
        raise Error("probe: FAIL tokens_chunk_size")
    if tc.frame_pre_padding() != 3:
        raise Error("probe: FAIL frame_pre_padding")
    if tc.token_overlap() != 2:
        raise Error("probe: FAIL token_overlap")
    if tc.frame_overlap() != 5:
        raise Error("probe: FAIL frame_overlap")
    if tc.tokens_per_clip() != 7:
        raise Error("probe: FAIL tokens_per_clip")
    print("phase 1 (config arithmetic, real FL2VA numbers) PASS")


# ── phase 2: _blend_frames isolated check ───────────────────────────────────────
def _run_blend_check(ctx: DeviceContext) raises:
    var a = Tensor.from_host(
        [Float32(10.0), Float32(20.0), Float32(30.0)], [1, 3, 1, 1, 1], STDtype.F32, ctx
    )
    var b = Tensor.from_host(
        [Float32(100.0), Float32(200.0), Float32(300.0)], [1, 3, 1, 1, 1], STDtype.F32, ctx
    )
    var blended = _blend_frames(a, b, 3, ctx)
    if blended.shape() != [1, 3, 1, 1, 1]:
        raise Error("probe: FAIL blend shape")
    var host = blended.to_host(ctx)
    # i=0: wa=1,wb=0 -> 10.  i=1: wa=2/3,wb=1/3 -> 20*2/3+200/3=80.  i=2: wa=1/3,wb=2/3 -> 30/3+600/3=210.
    var expected = List[Float32]()
    expected.append(Float32(10.0))
    expected.append(Float32(80.0))
    expected.append(Float32(210.0))
    var diff = _max_abs_diff(host, expected)
    print("  blend max_abs vs hand-computed:", diff)
    if diff > Float32(0.01):
        raise Error("probe: FAIL blend does not match hand-computed linear interpolation")
    print("phase 2 (_blend_frames isolated check) PASS")


# ── phase 3: encode_temporal, toy scale ─────────────────────────────────────────
def _toy_encoder_config() -> MiniMaxH3VideoEncoderDeviceConfig:
    # space_down paired with time_down at 2 -- the real released config never
    # fires a downsample with space_down=1 (levels with time_down>1 always
    # also have space_down=2; the two levels with space_down=1 have
    # time_down=1 too, so their downsample branch never fires at all). The
    # downsample conv's OWN declared spatial padding is unconditionally 0
    # (vae_cnn.py Downsample3D.conv: `padding=(1,0,0)`), so a space_down=1
    # downsample would shrink H/W by kernel-1 with no padding -- a real
    # limitation of the ORIGINAL architecture for a combination the release
    # never uses, not something to route around here.
    return MiniMaxH3VideoEncoderDeviceConfig(
        4, [1], [2], [2], 1, 2, 3, 2, Float32(1.0e-6),
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
    var seed = 300
    for i in range(len(names)):
        seed += 1
        var shape = _encoder_weight_shape(names[i], config)
        var n = 1
        for d in range(len(shape)):
            n *= shape[d]
        var host = _pattern(seed, n)
        tensors.append(TArc(Tensor.from_host(host, shape^, STDtype.BF16, ctx)))
    save_safetensors(names, tensors, String(ENC_CKPT), ctx)


def _run_encode_temporal_smoke(ctx: DeviceContext) raises:
    var econfig = _toy_encoder_config()
    _write_encoder_checkpoint(econfig, ctx)
    var encoder = MiniMaxH3VideoEncoderDevice.load(String(ENC_CKPT), econfig, ctx)
    var tconfig = MiniMaxH3VideoTemporalConfig(7, 2, 1)  # toy clip_length/vae_ratio_t/token_drop

    # T_raw=10, H=W=4 -- after the one 2x spatial downsample level, H=W=2,
    # still >=2 (the reflect-pad1 minimum) for encoder.conv_out's own
    # kernel-3 reflect pad afterward.
    var pixels = Tensor.from_host(
        _pattern(700, 1 * 10 * 4 * 4 * 2), [1, 10, 4, 4, 2], STDtype.BF16, ctx
    )
    var moments = minimax_h3_video_encode_temporal(encoder, pixels, tconfig, ctx)
    print("  encode_temporal output shape:", moments.shape())
    # T_raw=10 pads to 14 (2 chunks of 7); each chunk -> ceil(7/2)=4 latent
    # tokens (causal stride-2 downsample); 2*4=8 tokens total; token_drop=1
    # trims to 7.
    if moments.shape()[1] != 7:
        raise Error("probe: FAIL encode_temporal token count")
    _ = sys_remove(String(ENC_CKPT))
    print("phase 3 (encode_temporal, toy scale) PASS")


# ── phase 4: decode_temporal SEAM TEST, toy scale ───────────────────────────────
def _toy_decoder_config() -> MiniMaxH3VideoDecoderDeviceConfig:
    # patch_size_t MUST equal the temporal config's vae_ratio_t (2) --
    # architectural invariant (vit_kwargs.setdefault("patch_size_t", vae_ratio_t)).
    return MiniMaxH3VideoDecoderDeviceConfig(
        3, 2, 1, 2, 8, 1, 1, 2, Float64(100.0), Float64(0.75), Float32(1.0e-5), False, 4,
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
    var seed = 400
    for i in range(len(names)):
        seed += 1
        var shape = _decoder_weight_shape(names[i], config)
        var n = 1
        for d in range(len(shape)):
            n *= shape[d]
        var host = _pattern(seed, n)
        tensors.append(TArc(Tensor.from_host(host, shape^, STDtype.BF16, ctx)))
    save_safetensors(names, tensors, String(DEC_CKPT), ctx)


def _run_decode_temporal_seam_test(ctx: DeviceContext) raises:
    var dconfig = _toy_decoder_config()
    _write_decoder_checkpoint(dconfig, ctx)
    var decoder = minimax_h3_video_decoder_device_load(String(DEC_CKPT), dconfig, ctx)
    var tconfig = MiniMaxH3VideoTemporalConfig(7, 2, 1)

    # latent_t=8, LATENT_H=LATENT_W=1 -> num_chunks=2, pad_tokens=3 (hand-derived).
    var latents = Tensor.from_host(
        _pattern(800, 1 * 8 * 1 * 1 * 3), [1, 8, 1, 1, 3], STDtype.BF16, ctx
    )

    var dec = minimax_h3_video_decode_temporal[1, 1, 2, 8, 2, 7](decoder, latents, tconfig, ctx)
    print("  decode_temporal output shape:", dec.shape())
    # Hand-derived: chunk0(7) + chunk1_blended(7) + final_tail(5) = 19 raw,
    # minus pad_frames=5 (traced through _decode_temporal_pad_frames by hand
    # for z_len_after_pad=11, pad_tokens=3) = 14.
    if dec.shape() != [1, 14, 1, 1, 2]:
        raise Error("probe: FAIL decode_temporal output shape/frame-count mismatch")

    var pad_frames = _decode_temporal_pad_frames(tconfig, 11, 3)
    print("  hand-traced pad_frames:", pad_frames, " (expect 5)")
    if pad_frames != 5:
        raise Error("probe: FAIL _decode_temporal_pad_frames arithmetic")

    # SEAM DISCRIMINATION: chunk1's clip_z is z[4:11] (7 tokens). Decode it
    # DIRECTLY (unblended) and compare its first `frame_overlap`=5 frames
    # (after the same frame_pre_padding=1 trim) against decode_temporal's
    # OWN output at the corresponding position (frames [7:12) of `dec`,
    # since chunk0 contributed frames [0:7) unblended). If the temporal
    # layer's blending were a silent no-op, these two would match exactly;
    # a real blend must differ (except in the vanishingly unlikely case the
    # two source segments are already identical, ruled out by construction
    # since dec_overlap comes from chunk0 and this comes from chunk1).
    var clip1_z = Tensor.from_host(
        _pattern(800, 1 * 8 * 1 * 1 * 3), [1, 8, 1, 1, 3], STDtype.BF16, ctx
    )
    var clip1_tokens = slice(clip1_z, 1, 4, 7, ctx)
    var clip1_dec = minimax_h3_video_decode_device[9, 2, 8](decoder, clip1_tokens, ctx)
    var clip1_seg = slice(clip1_dec, 1, 0, 8, ctx)          # j=0 segment before trim
    var clip1_trimmed = slice(clip1_seg, 1, 1, 7, ctx)      # frame_pre_padding=1 trim -> 7 frames
    var clip1_unblended_seam = slice(clip1_trimmed, 1, 0, 5, ctx)  # first frame_overlap=5

    var dec_seam = slice(dec, 1, 7, 5, ctx)  # dec's frames [7:12), chunk1's blended contribution start

    var unblended_host = clip1_unblended_seam.to_host(ctx)
    var blended_host = dec_seam.to_host(ctx)
    var seam_diff = _max_abs_diff(unblended_host, blended_host)
    print("  seam max_abs diff (blended vs unblended chunk1):", seam_diff)
    if seam_diff == Float32(0.0):
        raise Error(
            "probe: FAIL the seam is bit-identical to an unblended decode --"
            " blending is not actually happening"
        )

    _ = sys_remove(String(DEC_CKPT))
    print("phase 4 (decode_temporal SEAM TEST, toy scale) PASS")


def main() raises:
    var ctx = DeviceContext()
    print("== phase 1: config arithmetic (real FL2VA numbers) ==")
    _run_config_arithmetic_check()
    print("== phase 2: _blend_frames isolated check ==")
    _run_blend_check(ctx)
    print("== phase 3: encode_temporal (toy scale) ==")
    _run_encode_temporal_smoke(ctx)
    print("== phase 4: decode_temporal SEAM TEST (toy scale) ==")
    _run_decode_temporal_seam_test(ctx)
    print("minimax_h3_video_vae_temporal_probe PASS")

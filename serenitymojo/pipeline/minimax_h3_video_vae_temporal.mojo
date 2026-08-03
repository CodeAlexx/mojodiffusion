# serenitymojo/pipeline/minimax_h3_video_vae_temporal.mojo
#
# MiniMax-H3 video VAE — TEMPORAL CHUNK + BLEND layer, the piece that turns
# "encode/decode ONE volume" (models/vae/minimax_h3_video_{encoder,decoder}_
# device.mojo) into "encode/decode a real video". H3's production shape is
# 121-124 frames; `clip_length=17` means a real generation is ~8-10 clips,
# not one volume — without this layer the corrected VAE can only ever
# produce a single clip.
#
# Authority: klvae.py `AutoencoderKL.encode_temporal`/`decode_temporal`
# (:461-788), `blend`/`split_tiles` (:192-250), `_decode_temporal_pad_frames`/
# `_decode_temporal_output_frame_plan` (:514-569). Read in full.
#
# SCOPE: this ports the case FL2VA's OWN `video_vae/config.json` actually
# uses — `isolated_first_frame`/`isolated_last_frame`/`isolated_key_frame`
# are NOT in that file's `load_kwargs` (only clip_length, token_drop,
# {encoder,decoder,parallel}_tiling, tile_size/overlap, {encoder,decoder}_
# parallel, chunk_dim are), so `setup_forward`'s `kwargs.get(..., False)`
# leaves all three False for this release. With all three False,
# `encode_temporal`'s `offset_frame` is always 0 and `decode_temporal`'s
# `z_head`/`z_tail` are always `None` — the isolated-frame branches in the
# reference (prepend/append a separately-encoded first/last frame) are DEAD
# CODE for FL2VA and are NOT ported here. If a later partition's config sets
# any of the three, this module will need those branches added — it will
# not silently produce wrong output for that case, because it does not
# expose those flags as toggles at all (there is nothing to misconfigure).
#
# This also ported the NON-STREAMING decode path (`decode_temporal`'s
# `torch.cat(dec_list, ...)` branch, not `_decode_temporal_streaming`'s
# preallocated-buffer write) — same final tensor, `_resolve_temporal_stream_
# cat()` is a memory-locality optimization for the reference's own hardware,
# not a numerical difference. Holding ~num_chunks decoded clips in a List
# before the final concat is the resulting memory trade; revisit if a real
# 8-10-chunk run OOMs.
#
# KEY SIMPLIFICATION THAT MADE THIS TRACTABLE: every temporal clip decode
# consumes EXACTLY `tokens_chunk_size + token_overlap` latent tokens (proven
# below), so `ops/attention.sdpa_nomask`'s compile-time S requirement only
# needs ONE value, not one per chunk. Proof: after alignment-padding, the
# padded latent length is `(num_chunks+1) * tokens_chunk_size`; chunk `i`'s
# window ends at `(i+1)*tokens_chunk_size + token_overlap`, and since
# `token_overlap < tokens_chunk_size` by construction (it is a remainder),
# that end index is `< (num_chunks+1)*tokens_chunk_size` for every
# `i <= num_chunks-1` — no chunk is ever truncated at the padded boundary.
#
# NOT PORTED (see the audit report to team-lead): SPATIAL TILING
# (tiled_encode/tiled_decode, klvae.py :297-429) — the reference's
# `_adaptive_encode`/`_adaptive_decode` call tiled_encode/tiled_decode
# INSIDE each temporal clip's encode/decode when `encoder_tiling`/
# `decoder_tiling` are set, and FL2VA's `video_vae/config.json` sets BOTH
# to 1 (on) with `vae_tile_size: 256` PIXELS — meaning tiling is NOT merely
# large-canvas insurance for this release, it is the release's own default
# even at moderate resolutions. This module calls the per-volume device
# functions DIRECTLY at the `_adaptive_encode`/`_adaptive_decode` call
# sites (`_encode_one_clip`/`_decode_one_clip` below) specifically so
# spatial tiling can be inserted there later without touching the temporal
# logic — same layering the reference itself uses.
#
# ALSO NOT PORTED: `DiagonalGaussianDistribution.sample()`. This module's
# encode path returns MOMENTS (2*z_channels wide), matching
# `encode_temporal`'s own contract exactly (`encode_base` samples ONCE,
# AFTER temporal concatenation, from `encode_temporal`'s output — token_drop
# trims the moments sequence, not a sampled latent). Sampling needs a
# device Gaussian-noise draw; wiring it to whatever RNG the DiT's own
# noise-init already uses is a separate, small follow-up.
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.collections import List
from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.ops.tensor_algebra import add, concat, mul, slice
from serenitymojo.models.vae.minimax_h3_video_encoder_device import (
    MiniMaxH3VideoEncoderDevice, minimax_h3_video_encode_device,
)
from serenitymojo.models.vae.minimax_h3_video_decoder_device import (
    MiniMaxH3VideoDecoderDevice, minimax_h3_video_decode_device,
)

comptime TArc = ArcPointer[Tensor]


@fieldwise_init
struct MiniMaxH3VideoTemporalConfig(Copyable, Movable):
    """`video_vae/config.json`'s temporal fields, verbatim: clip_length=17,
    token_drop=3. `vae_ratio_t` is the encoder's temporal compression
    (product of `time_down`, [1,2,2,1,1,1] -> 4 for the released config —
    NOT re-derived here, passed explicitly so this module has no hidden
    coupling to the encoder config's exact field layout)."""

    var clip_length: Int
    var vae_ratio_t: Int
    var token_drop: Int

    def tokens_chunk_size(self) -> Int:
        """`ceil(clip_length / vae_ratio_t)` (setup_forward:104)."""
        return (self.clip_length + self.vae_ratio_t - 1) // self.vae_ratio_t

    def frame_pre_padding(self) -> Int:
        """`(-clip_length) % vae_ratio_t` (setup_forward:103), computed via
        POSITIVE-operand modulo only — Mojo's `%` sign convention for
        negative operands is not something this file depends on."""
        var r = self.clip_length % self.vae_ratio_t
        if r == 0:
            return 0
        return self.vae_ratio_t - r

    def token_overlap(self) -> Int:
        """`(-token_drop) % tokens_chunk_size` (setup_forward:105), same
        positive-modulo construction."""
        var tcs = self.tokens_chunk_size()
        var r = self.token_drop % tcs
        if r == 0:
            return 0
        return tcs - r

    def frame_overlap(self) -> Int:
        """`max(token_overlap*vae_ratio_t - frame_pre_padding, 0)`
        (setup_forward:106)."""
        var v = self.token_overlap() * self.vae_ratio_t - self.frame_pre_padding()
        return v if v > 0 else 0

    def tokens_per_clip(self) -> Int:
        return self.tokens_chunk_size() + self.token_overlap()


def minimax_h3_video_released_temporal_config() -> MiniMaxH3VideoTemporalConfig:
    """FL2VA `video_vae/config.json`: clip_length=17, token_drop=3.
    vae_ratio_t=4 = product(time_down=[1,2,2,1,1,1]) from
    `video_vae/source/config.json` — matches
    `minimax_h3_video_released_encoder_config()`'s time_down list."""
    return MiniMaxH3VideoTemporalConfig(17, 4, 3)


# ─────────────────────────────────────────────────────────────────────────────
# blend — linear cross-fade over the LAST `extent` frames of `a` and the
# FIRST `extent` frames of `b` (temporal axis 1, NDHWC), `b`'s remaining
# frames appended untouched. Matches klvae.py `AutoencoderKL.blend`
# (:220-250, `dim=-3` there is this same frame axis).
# ─────────────────────────────────────────────────────────────────────────────
def _blend_frames(a: Tensor, b: Tensor, blend_extent: Int, ctx: DeviceContext) raises -> Tensor:
    var ash = a.shape()
    var bsh = b.shape()
    var extent = blend_extent
    if ash[1] < extent:
        extent = ash[1]
    if bsh[1] < extent:
        extent = bsh[1]
    if extent <= 0:
        return b.clone(ctx)

    var a_overlap = slice(a, 1, ash[1] - extent, extent, ctx)
    var b_overlap = slice(b, 1, 0, extent, ctx)

    var wb_host = List[Float32](capacity=extent)
    var wa_host = List[Float32](capacity=extent)
    for i in range(extent):
        var t = Float32(i) / Float32(extent)
        wb_host.append(t)
        wa_host.append(Float32(1.0) - t)
    var wb = Tensor.from_host(wb_host, [1, extent, 1, 1, 1], a.dtype(), ctx)
    var wa = Tensor.from_host(wa_host, [1, extent, 1, 1, 1], a.dtype(), ctx)

    var blended = add(mul(a_overlap, wa, ctx), mul(b_overlap, wb, ctx), ctx)
    if extent < bsh[1]:
        var b_rest = slice(b, 1, extent, bsh[1] - extent, ctx)
        return concat(1, ctx, blended, b_rest)
    return blended^


# ─────────────────────────────────────────────────────────────────────────────
# ENCODE.
# ─────────────────────────────────────────────────────────────────────────────
def minimax_h3_video_encode_temporal(
    encoder: MiniMaxH3VideoEncoderDevice,
    pixels: Tensor,  # [1, T_raw, H, W, in_channels] NDHWC, ImageNet-normalized
    tconfig: MiniMaxH3VideoTemporalConfig,
    ctx: DeviceContext,
) raises -> Tensor:
    """Returns MOMENTS `[1, num_chunks*tokens_chunk_size - token_drop, H',
    W', 2*z_channels]` — the caller samples (DiagonalGaussianDistribution),
    this module does not (see this file's header). Mirrors
    `encode_temporal` (:461-512) for FL2VA's config (all isolated_* flags
    False, so `offset_frame` is always 0)."""
    var s = pixels.shape()
    var t_raw = s[1]
    var clip_length = tconfig.clip_length

    var x: Tensor
    var remainder = t_raw % clip_length
    if remainder != 0:
        var pad_size = clip_length - remainder
        var last = slice(pixels, 1, t_raw - 1, 1, ctx)
        var padded = pixels.clone(ctx)
        for _ in range(pad_size):
            padded = concat(1, ctx, padded, last)
        x = padded^
    else:
        x = pixels.clone(ctx)

    var t_total = x.shape()[1]
    var num_chunks = t_total // clip_length
    if num_chunks <= 0:
        raise Error("MiniMax-H3 video temporal encode: num_chunks <= 0")

    var z_parts = List[TArc]()
    for i in range(num_chunks):
        var clip_x = slice(x, 1, i * clip_length, clip_length, ctx)
        var moments = minimax_h3_video_encode_device(encoder, clip_x, ctx)
        z_parts.append(TArc(moments^))

    var z_cat = z_parts[0][].clone(ctx)
    for i in range(1, num_chunks):
        z_cat = concat(1, ctx, z_cat, z_parts[i][])

    if tconfig.token_drop > 0:
        var keep = z_cat.shape()[1] - tconfig.token_drop
        if keep <= 0:
            raise Error("MiniMax-H3 video temporal encode: token_drop >= total tokens")
        z_cat = slice(z_cat, 1, 0, keep, ctx)
    return z_cat^


# ─────────────────────────────────────────────────────────────────────────────
# DECODE.
# ─────────────────────────────────────────────────────────────────────────────
def _decode_temporal_pad_frames(
    tconfig: MiniMaxH3VideoTemporalConfig, z_len_after_pad: Int, pad_tokens: Int,
) raises -> Int:
    """`_decode_temporal_pad_frames` (:514-529): how many DECODED frames the
    `pad_tokens` alignment tokens correspond to, so they can be trimmed off
    the end of the reconstruction."""
    if pad_tokens <= 0:
        return 0
    var intra_tail = tconfig.clip_length % tconfig.vae_ratio_t
    if intra_tail == 0:
        return pad_tokens * tconfig.vae_ratio_t
    var z_len_before_pad = z_len_after_pad - pad_tokens
    var tokens_chunk_size = tconfig.tokens_chunk_size()
    var total = 0
    for k in range(pad_tokens):
        if (z_len_before_pad + k) % tokens_chunk_size == 0:
            total += intra_tail
        else:
            total += tconfig.vae_ratio_t
    return total


def minimax_h3_video_decode_temporal[
    LATENT_H: Int, LATENT_W: Int, HEADS: Int, DIM_HEAD: Int, NUM_SUFFIX: Int,
    TOKENS_PER_CLIP: Int,
](
    decoder: MiniMaxH3VideoDecoderDevice,
    latents: Tensor,  # [1, latent_T, LATENT_H, LATENT_W, latent_channels] NDHWC, SAMPLED
    tconfig: MiniMaxH3VideoTemporalConfig,
    ctx: DeviceContext,
) raises -> Tensor:
    """Mirrors `decode_temporal`'s NON-STREAMING branch (:721-788) for
    FL2VA's config (`z_head`/`z_tail` always None). `TOKENS_PER_CLIP` is the
    comptime twin of `tconfig.tokens_per_clip()` — required because
    `ops/attention.sdpa_nomask`'s sequence length is compile-time; see this
    file's header for why ONE value suffices for every chunk. `NUM_SUFFIX`
    is `1 + decoder.config.num_register_tokens`, `HEADS`/`DIM_HEAD` are
    `decoder.config.heads`/`dim_head` — all four checked against the
    runtime config below, same pattern
    `minimax_h3_video_decode_device` uses for its own H/Dh."""
    if decoder.config.heads != HEADS or decoder.config.dim_head != DIM_HEAD:
        raise Error("MiniMax-H3 video temporal decode: comptime HEADS/DIM_HEAD mismatch")
    if decoder.config.num_register_tokens + 1 != NUM_SUFFIX:
        raise Error("MiniMax-H3 video temporal decode: comptime NUM_SUFFIX mismatch")
    if tconfig.tokens_per_clip() != TOKENS_PER_CLIP:
        raise Error("MiniMax-H3 video temporal decode: comptime TOKENS_PER_CLIP mismatch")

    comptime S_CLIP = TOKENS_PER_CLIP * LATENT_H * LATENT_W + NUM_SUFFIX

    var tokens_chunk_size = tconfig.tokens_chunk_size()
    var frame_pre_padding = tconfig.frame_pre_padding()
    var frame_overlap = tconfig.frame_overlap()
    var vae_ratio_t = tconfig.vae_ratio_t
    var token_drop = tconfig.token_drop

    var latent_t = latents.shape()[1]
    var pseudo_total_tokens = latent_t + token_drop
    var remainder = pseudo_total_tokens % tokens_chunk_size
    var pad_tokens = 0
    if remainder != 0:
        pad_tokens = tokens_chunk_size - remainder
        pseudo_total_tokens += pad_tokens
    var pseudo_num_chunks = pseudo_total_tokens // tokens_chunk_size
    var num_chunks = pseudo_num_chunks - (1 if token_drop > 0 else 0)
    if num_chunks <= 0:
        raise Error("MiniMax-H3 video temporal decode: computed non-positive num_chunks")

    var z: Tensor
    if pad_tokens > 0:
        var last_tok = slice(latents, 1, latent_t - 1, 1, ctx)
        var padded = latents.clone(ctx)
        for _ in range(pad_tokens):
            padded = concat(1, ctx, padded, last_tok)
        z = padded^
    else:
        z = latents.clone(ctx)
    var z_len_after_pad = latent_t + pad_tokens

    var split_count = 2 if token_drop > 0 else 1
    var chunk_dec = tokens_chunk_size * vae_ratio_t

    var dec_parts = List[TArc]()
    var dec_overlap = Optional[Tensor](None)

    for i in range(num_chunks):
        var t_start = i * tokens_chunk_size
        var clip_z = slice(z, 1, t_start, TOKENS_PER_CLIP, ctx)
        var clip_dec = minimax_h3_video_decode_device[S_CLIP, HEADS, DIM_HEAD](
            decoder, clip_z, ctx
        )
        var clip_frames = clip_dec.shape()[1]

        for j in range(split_count):
            var f_start = j * chunk_dec
            if f_start >= clip_frames:
                continue
            var f_end = f_start + chunk_dec
            if f_end > clip_frames:
                f_end = clip_frames
            var seg_len = f_end - f_start
            var seg = slice(clip_dec, 1, f_start, seg_len, ctx)
            var trimmed_len = seg.shape()[1] - frame_pre_padding
            if trimmed_len <= 0:
                continue
            var trimmed = slice(seg, 1, frame_pre_padding, trimmed_len, ctx)

            if j == 0:
                var out_chunk: Tensor
                if dec_overlap:
                    out_chunk = _blend_frames(dec_overlap.value(), trimmed, frame_overlap, ctx)
                else:
                    out_chunk = trimmed^
                dec_parts.append(TArc(out_chunk^))
            else:
                dec_overlap = Optional[Tensor](trimmed^)

    if dec_overlap:
        dec_parts.append(TArc(dec_overlap.value().clone(ctx)))

    if len(dec_parts) == 0:
        raise Error("MiniMax-H3 video temporal decode: produced zero output chunks")
    var dec = dec_parts[0][].clone(ctx)
    for i in range(1, len(dec_parts)):
        dec = concat(1, ctx, dec, dec_parts[i][])

    var pad_frames = _decode_temporal_pad_frames(tconfig, z_len_after_pad, pad_tokens)
    if pad_frames > 0:
        var keep = dec.shape()[1] - pad_frames
        if keep <= 0:
            raise Error("MiniMax-H3 video temporal decode: pad_frames >= total decoded frames")
        dec = slice(dec, 1, 0, keep, ctx)
    return dec^

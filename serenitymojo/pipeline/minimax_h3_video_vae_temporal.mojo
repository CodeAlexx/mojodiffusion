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
# SPATIAL TILING (tiled_encode/tiled_decode, klvae.py :297-429; ported in
# pipeline/minimax_h3_video_vae_spatial_tiling.mojo) is WIRED IN at exactly
# the reference's own `_adaptive_encode`/`_adaptive_decode` call sites: this
# module's per-clip step calls `minimax_h3_video_tiled_encode`/`_tiled_
# decode`, not the untiled per-volume functions directly. FL2VA's
# `video_vae/config.json` sets `encoder_tiling`/`decoder_tiling` to 1 (on)
# by default with `vae_tile_size: 256` PIXELS — tiling is NOT merely
# large-canvas insurance for this release, it is the release's own default
# even at moderate resolutions (confirmed: at 1344x768 both axes exceed 256).
#
# COMPOSITION SAFETY (tiles-within-chunks): spatial tiling fully resolves
# every H/W seam INSIDE one `tiled_encode`/`tiled_decode` call (same input/
# output contract as the untiled per-volume functions), so by the time THIS
# module's temporal blend runs, it is operating on a complete, already
# spatially-reconstructed per-clip volume — the temporal blend never sees a
# partial tile, and the two blends are resolved at different times on
# disjoint axes. Measured, not just argued: see the composition gate in
# pipeline/parity/minimax_h3_video_vae_composition_probe.mojo, which checks
# a volume needing BOTH multiple chunks AND multiple tiles simultaneously,
# including the corner where a spatial seam meets a temporal seam, against
# a direct untiled/unchunked decode of the same volume (no oracle needed —
# tiled+chunked must match the direct path to blend precision).
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
from max.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.ops.tensor_algebra import concat, slice
from serenitymojo.pipeline.minimax_h3_video_vae_blend import minimax_h3_video_blend
from serenitymojo.pipeline.minimax_h3_video_vae_spatial_tiling import (
    MiniMaxH3TilingConfig, minimax_h3_video_tiled_decode, minimax_h3_video_tiled_encode,
)
from serenitymojo.models.vae.minimax_h3_video_encoder_device import (
    MiniMaxH3VideoEncoderDevice,
)
from serenitymojo.models.vae.minimax_h3_video_decoder_device import (
    MiniMaxH3VideoDecoderDevice,
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
# ENCODE.
# ─────────────────────────────────────────────────────────────────────────────
def minimax_h3_video_encode_temporal(
    encoder: MiniMaxH3VideoEncoderDevice,
    pixels: Tensor,  # [1, T_raw, H, W, in_channels] NDHWC, ImageNet-normalized
    tconfig: MiniMaxH3VideoTemporalConfig,
    tiling: MiniMaxH3TilingConfig,
    ctx: DeviceContext,
) raises -> Tensor:
    """Returns MOMENTS `[1, num_chunks*tokens_chunk_size - token_drop, H',
    W', 2*z_channels]` — the caller samples (DiagonalGaussianDistribution),
    this module does not (see this file's header). Mirrors
    `encode_temporal` (:461-512) for FL2VA's config (all isolated_* flags
    False, so `offset_frame` is always 0). Each clip's encode goes through
    `minimax_h3_video_tiled_encode` (`_adaptive_encode`'s own composition —
    tiling fully resolves spatially before this function ever concatenates
    across the temporal axis)."""
    var s = pixels.shape()
    var t_raw = s[1]
    var clip_length = tconfig.clip_length

    var x: Tensor
    var remainder = t_raw % clip_length
    if remainder != 0:
        var pad_size = clip_length - remainder
        # Build the pad block FIRST (doubling the single last frame), then
        # attach it in ONE concat. Byte-identical to the reference's
        # `x[:, :, -1:].repeat(1,1,pad,1,1)` + single `torch.cat`
        # (encode_temporal:466-467 — concat is a pure copy), but materially
        # different in allocation: the previous one-frame-at-a-time loop
        # allocated `pad_size` full-video-sized intermediates back to back
        # (~1.8 GiB each at the 1344x768 canvas) faster than the runtime
        # retires their frees, and a REAL 141-frame ref2va encode died
        # CUDA_ERROR_OUT_OF_MEMORY at pad iteration 10 with device VRAM
        # nearly idle (measured 2026-08-03, ref-encode gate + oom_bisect
        # probe). With the doubling, every intermediate is a pad-block of at
        # most `pad_size` frames, and only the final concat is video-sized.
        var last = slice(pixels, 1, t_raw - 1, 1, ctx)
        var pad = last^
        while pad.shape()[1] * 2 <= pad_size:
            pad = concat(1, ctx, pad, pad)
        var have = pad.shape()[1]
        if have < pad_size:
            var rest = slice(pad, 1, 0, pad_size - have, ctx)
            pad = concat(1, ctx, pad, rest)
        x = concat(1, ctx, pixels, pad)
    else:
        x = pixels.clone(ctx)

    var t_total = x.shape()[1]
    var num_chunks = t_total // clip_length
    if num_chunks <= 0:
        raise Error("MiniMax-H3 video temporal encode: num_chunks <= 0")

    var z_parts = List[TArc]()
    for i in range(num_chunks):
        var clip_x = slice(x, 1, i * clip_length, clip_length, ctx)
        var moments = minimax_h3_video_tiled_encode(encoder, clip_x, tiling, ctx)
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
    LATENT_TILE_H: Int, LATENT_TILE_W: Int, HEADS: Int, DIM_HEAD: Int, NUM_SUFFIX: Int,
    TOKENS_PER_CLIP: Int,
](
    decoder: MiniMaxH3VideoDecoderDevice,
    latents: Tensor,  # [1, latent_T, latent_H, latent_W, latent_channels] NDHWC, SAMPLED
    tconfig: MiniMaxH3VideoTemporalConfig,
    tiling: MiniMaxH3TilingConfig,
    ctx: DeviceContext,
) raises -> Tensor:
    """Mirrors `decode_temporal`'s NON-STREAMING branch (:721-788) for
    FL2VA's config (`z_head`/`z_tail` always None). Each clip's decode goes
    through `minimax_h3_video_tiled_decode` (`_adaptive_decode`'s own
    composition), so `latents`' own H/W (read at runtime below) need not
    equal a single tile — `LATENT_TILE_H`/`LATENT_TILE_W` are the comptime
    SPATIAL TILE size `minimax_h3_video_tiled_decode` will actually use
    (equal to the full per-clip canvas only when that canvas fits in one
    tile), required because `ops/attention.sdpa_nomask`'s sequence length
    is compile-time; see this file's header for why ONE value suffices for
    every chunk (temporal) and `minimax_h3_video_vae_spatial_tiling.mojo`'s
    header for why ONE value suffices for every tile (spatial). `NUM_SUFFIX`
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
        var clip_dec = minimax_h3_video_tiled_decode[
            LATENT_TILE_H, LATENT_TILE_W, HEADS, DIM_HEAD, NUM_SUFFIX, TOKENS_PER_CLIP
        ](decoder, clip_z, tiling, ctx)
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
                    out_chunk = minimax_h3_video_blend(dec_overlap.value(), trimmed, frame_overlap, 1, ctx)
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


# ─────────────────────────────────────────────────────────────────────────────
# STREAMING DECODE. Same math, same order, same values — the whole-video
# tensor is never materialized.
#
# WHY: `minimax_h3_video_decode_temporal` above accumulates every part in
# `dec_parts` and then folds them with a progressive `concat` (:337-339). That
# is O(n^2) in allocation churn AND it holds the ENTIRE decoded video on device
# twice at the fold's peak. Measured cost at the 14-second hero geometry (328
# frames, 960x544, F32): the pixel volume alone is 2.06 GiB, the progressive
# fold peaks near 4.2 GiB, and the pixel-denormalize adds another 2.06 GiB
# copy — on a card with ~2.3 GiB of headroom after the denoise loop.
#
# WHAT THE VENDOR DOES, AND WHY THIS GOES FURTHER. `_decode_temporal_streaming`
# (klvae.py:571-676) removes the CONCAT but not the VOLUME: it preallocates the
# full `[..., output_frames, ...]` output with `torch.empty` on the first write
# (:597-600) and `copy_`s each part into its slot (:605-607). Peak is therefore
# still one whole video plus one clip — better than the fold, but it does not
# stop growing with duration, and porting it would need a write-into-tensor-at-
# offset op this repo does not have (no `slice_assign`/`copy_into` in
# ops/tensor_algebra.mojo — checked).
#
# So this ports the vendor's LOOP STRUCTURE and its FRAME BOOKKEEPING exactly,
# but hands each finished part to the caller instead of writing it into a
# preallocated buffer. The caller (pipeline/minimax_h3_t2va.mojo) denormalizes
# and writes that part's PNGs, then drops it. Peak becomes ONE CLIP DECODE plus
# ONE PART — ~50 frames — INDEPENDENT OF VIDEO LENGTH. At the hero geometry
# that is ~0.31 GiB instead of ~4.2 GiB.
#
# BIT-EXACTNESS vs the non-streaming path is not an aspiration, it is
# structural: every part is produced by the identical sequence of slice /
# tiled-decode / blend calls in the identical order. Only the accumulation
# differs, and concat of the streamed parts must reproduce the folded tensor
# exactly. `minimax_h3_video_decode_temporal_streamed` below exists to let a
# gate assert precisely that (max_abs diff == 0), and
# pipeline/parity/minimax_h3_video_temporal_streaming_probe.mojo does.
#
# TRUNCATION. The vendor does NOT slice `pad_frames` off the tail in the
# streaming path the way the non-streaming path does (:341-346). It simply
# stops copying once the preallocated buffer is full (:602-609:
# `copy_frames = min(part_frames, max(0, remaining))`) and counts the remainder
# as `dropped_frames`. Since the pad frames are by construction at the END,
# truncating the write is exactly equivalent to trimming the tail. This port
# does the same, clamping each emitted part against the planned
# `output_frames`; a part that is fully truncated is reported as `None` rather
# than as a zero-length tensor. For every geometry the pipeline can actually
# produce (`latent_T = 5n+2` => `pseudo_total = 5n+5`, divisible by
# `tokens_chunk_size`) `pad_tokens` is 0 and this path never fires, but
# decode_only mode can hand over an arbitrary latent length, so it is
# implemented rather than assumed away.
#
# THE VENDOR'S OWN INVARIANT IS PORTED TOO (:668-674): after the last part,
# `finish()` checks `logical_frames == total_frames`, `dropped == pad_frames`
# and `emitted == output_frames`, and raises naming all six numbers otherwise.
# That is a real end-of-decode assertion on host integers, not a comment.
# ─────────────────────────────────────────────────────────────────────────────
@fieldwise_init
struct MiniMaxH3TemporalFramePlan(Copyable, Movable):
    """`_decode_temporal_output_frame_plan`'s three return values (:569), for
    the `z_head is None and z_tail is None` case FL2VA always takes (see this
    file's header SCOPE note)."""

    var total_frames: Int
    var pad_frames: Int
    var output_frames: Int


def minimax_h3_temporal_output_frame_plan(
    tconfig: MiniMaxH3VideoTemporalConfig,
    z_len_after_pad: Int,
    num_chunks: Int,
    pad_tokens: Int,
) raises -> MiniMaxH3TemporalFramePlan:
    """Port of `_decode_temporal_output_frame_plan` (:531-569). PURE HOST
    INTEGER ARITHMETIC — no tensor, no device, no weights — which is what lets
    it be gated on CPU against the vendor's own function while the GPU is
    busy."""
    var tcs = tconfig.tokens_chunk_size()
    var rt = tconfig.vae_ratio_t
    var chunk_dec = tcs * rt
    var split_count = 2 if tconfig.token_drop > 0 else 1
    var fpp = tconfig.frame_pre_padding()
    var token_overlap = tconfig.token_overlap()

    var total = 0
    var final_overlap = 0
    for i in range(num_chunks):
        var t_start = i * tcs
        var t_end = t_start + tcs + token_overlap
        # `min(idx, z_len)` on BOTH ends, exactly as the reference writes it
        # (:543) — that is what makes a past-the-end window contribute 0
        # rather than a negative length.
        var lo = t_start if t_start < z_len_after_pad else z_len_after_pad
        var hi = t_end if t_end < z_len_after_pad else z_len_after_pad
        var clip_token_len = hi - lo
        if clip_token_len < 0:
            clip_token_len = 0
        var clip_frame_len = clip_token_len * rt
        for j in range(split_count):
            var f_start = j * chunk_dec
            var f_end = f_start + chunk_dec
            if f_end > clip_frame_len:
                f_end = clip_frame_len
            var chunk_frames = f_end - f_start - fpp
            if chunk_frames < 0:
                chunk_frames = 0
            if j == 0:
                total += chunk_frames
            else:
                final_overlap = chunk_frames
    total += final_overlap

    var pad = _decode_temporal_pad_frames(tconfig, z_len_after_pad, pad_tokens)
    return MiniMaxH3TemporalFramePlan(total, pad, total - pad)


struct MiniMaxH3TemporalDecodeStream(Movable):
    """Emits `decode_temporal`'s output ONE PART AT A TIME. Construct, then
    call `next_part` while `has_next()`, then `finish()`.

    A "part" is one iteration's `j == 0` group (`clip_length` frames after the
    `frame_pre_padding` trim and the cross-fade with the previous clip's tail),
    plus — once, at the end — the final carried `dec_overlap`. That is exactly
    the sequence the non-streaming path appends to `dec_parts`, so
    concatenating every emitted part reproduces its output bit for bit."""

    var z: Tensor                       # alignment-padded latents
    var tconfig: MiniMaxH3VideoTemporalConfig
    var tiling: MiniMaxH3TilingConfig
    var num_chunks: Int
    var num_parts: Int
    var plan: MiniMaxH3TemporalFramePlan
    var part_index: Int
    var dec_overlap: Optional[Tensor]
    var emitted: Int                    # frames handed to the caller
    var logical: Int                    # frames produced before truncation
    var dropped: Int                    # frames truncated as alignment padding

    def __init__(
        out self,
        latents: Tensor,  # [1, latent_T, latent_H, latent_W, latent_channels] NDHWC
        tconfig: MiniMaxH3VideoTemporalConfig,
        tiling: MiniMaxH3TilingConfig,
        ctx: DeviceContext,
    ) raises:
        """Mirrors `decode_temporal`'s prologue (:678-713) — the same
        `pseudo_total_tokens` / `pad_tokens` / `num_chunks` arithmetic and the
        same last-token alignment padding — then plans the output."""
        var tcs = tconfig.tokens_chunk_size()
        var latent_t = latents.shape()[1]
        var pseudo_total_tokens = latent_t + tconfig.token_drop
        var remainder = pseudo_total_tokens % tcs
        var pad_tokens = 0
        if remainder != 0:
            pad_tokens = tcs - remainder
            pseudo_total_tokens += pad_tokens
        var pseudo_num_chunks = pseudo_total_tokens // tcs
        var num_chunks = pseudo_num_chunks - (1 if tconfig.token_drop > 0 else 0)
        if num_chunks <= 0:
            raise Error(
                "MiniMax-H3 video temporal stream: computed non-positive num_chunks"
            )

        if pad_tokens > 0:
            var last_tok = slice(latents, 1, latent_t - 1, 1, ctx)
            var padded = latents.clone(ctx)
            for _ in range(pad_tokens):
                padded = concat(1, ctx, padded, last_tok)
            self.z = padded^
        else:
            self.z = latents.clone(ctx)

        var z_len_after_pad = latent_t + pad_tokens
        self.plan = minimax_h3_temporal_output_frame_plan(
            tconfig, z_len_after_pad, num_chunks, pad_tokens
        )
        if self.plan.output_frames <= 0:
            raise Error(
                String("MiniMax-H3 video temporal stream: planned non-positive")
                + " output_frames=" + String(self.plan.output_frames)
                + " total_frames=" + String(self.plan.total_frames)
                + " pad_frames=" + String(self.plan.pad_frames)
            )

        self.tconfig = tconfig.copy()
        self.tiling = tiling.copy()
        self.num_chunks = num_chunks
        # One part per chunk (the j==0 group) plus the final carried overlap,
        # which only exists when token_drop > 0 creates a j==1 split.
        self.num_parts = num_chunks + (1 if tconfig.token_drop > 0 else 0)
        self.part_index = 0
        self.dec_overlap = Optional[Tensor](None)
        self.emitted = 0
        self.logical = 0
        self.dropped = 0

    def output_frames(self) -> Int:
        """How many frames the caller will receive in total."""
        return self.plan.output_frames

    def frames_emitted(self) -> Int:
        """Frames handed over so far — the global index of the next part's
        first frame, which is what a PNG-writing caller needs for numbering."""
        return self.emitted

    def has_next(self) -> Bool:
        return self.part_index < self.num_parts

    def _clamp_and_account(mut self, var part: Tensor, ctx: DeviceContext) raises -> Optional[Tensor]:
        """`write_part`'s accounting (:591-609) with the buffer write replaced
        by handing the part back. Returns None for a part that is entirely
        alignment padding."""
        var part_frames = part.shape()[1]
        if part_frames <= 0:
            return Optional[Tensor](None)
        self.logical += part_frames
        var remaining = self.plan.output_frames - self.emitted
        if remaining < 0:
            remaining = 0
        var copy_frames = part_frames if part_frames < remaining else remaining
        self.dropped += part_frames - copy_frames
        if copy_frames <= 0:
            return Optional[Tensor](None)
        self.emitted += copy_frames
        if copy_frames < part_frames:
            return Optional[Tensor](slice(part, 1, 0, copy_frames, ctx))
        return Optional[Tensor](part^)

    def next_part[
        LATENT_TILE_H: Int, LATENT_TILE_W: Int, HEADS: Int, DIM_HEAD: Int,
        NUM_SUFFIX: Int, TOKENS_PER_CLIP: Int,
    ](mut self, decoder: MiniMaxH3VideoDecoderDevice, ctx: DeviceContext) raises -> Optional[Tensor]:
        """Decode and return the next part. Comptime parameters are the same
        six `minimax_h3_video_decode_temporal` takes and carry the same
        meaning; they are re-checked against the runtime config here for the
        same reason."""
        if decoder.config.heads != HEADS or decoder.config.dim_head != DIM_HEAD:
            raise Error("MiniMax-H3 video temporal stream: comptime HEADS/DIM_HEAD mismatch")
        if decoder.config.num_register_tokens + 1 != NUM_SUFFIX:
            raise Error("MiniMax-H3 video temporal stream: comptime NUM_SUFFIX mismatch")
        if self.tconfig.tokens_per_clip() != TOKENS_PER_CLIP:
            raise Error("MiniMax-H3 video temporal stream: comptime TOKENS_PER_CLIP mismatch")
        if not self.has_next():
            raise Error("MiniMax-H3 video temporal stream: next_part past the end")

        var tcs = self.tconfig.tokens_chunk_size()
        var fpp = self.tconfig.frame_pre_padding()
        var frame_overlap = self.tconfig.frame_overlap()
        var chunk_dec = tcs * self.tconfig.vae_ratio_t
        var split_count = 2 if self.tconfig.token_drop > 0 else 1

        var i = self.part_index
        self.part_index += 1

        # The trailing part: the carried overlap, no decode (:657-660).
        if i >= self.num_chunks:
            if not self.dec_overlap:
                raise Error(
                    "MiniMax-H3 video temporal stream: final part expected a carried"
                    " overlap but none exists"
                )
            var tail = self.dec_overlap.value().clone(ctx)
            self.dec_overlap = Optional[Tensor](None)
            return self._clamp_and_account(tail^, ctx)

        var clip_z = slice(self.z, 1, i * tcs, TOKENS_PER_CLIP, ctx)
        var clip_dec = minimax_h3_video_tiled_decode[
            LATENT_TILE_H, LATENT_TILE_W, HEADS, DIM_HEAD, NUM_SUFFIX, TOKENS_PER_CLIP
        ](decoder, clip_z, self.tiling, ctx)
        var clip_frames = clip_dec.shape()[1]

        var out = Optional[Tensor](None)
        for j in range(split_count):
            var f_start = j * chunk_dec
            if f_start >= clip_frames:
                continue
            var f_end = f_start + chunk_dec
            if f_end > clip_frames:
                f_end = clip_frames
            var seg = slice(clip_dec, 1, f_start, f_end - f_start, ctx)
            var trimmed_len = seg.shape()[1] - fpp
            if trimmed_len <= 0:
                continue
            var trimmed = slice(seg, 1, fpp, trimmed_len, ctx)

            if j == 0:
                var group: Tensor
                if self.dec_overlap:
                    group = minimax_h3_video_blend(
                        self.dec_overlap.value(), trimmed, frame_overlap, 1, ctx
                    )
                    self.dec_overlap = Optional[Tensor](None)
                else:
                    group = trimmed^
                out = self._clamp_and_account(group^, ctx)
            else:
                self.dec_overlap = Optional[Tensor](trimmed^)
        return out^

    def finish(self) raises:
        """`_decode_temporal_streaming`'s closing invariant (:668-674), which
        is the whole reason that function can be trusted without re-reading its
        output: if the three counters do not match the plan, the frame
        bookkeeping diverged somewhere in the loop and the video is wrong."""
        if self.has_next():
            raise Error(
                String("MiniMax-H3 video temporal stream: finish() called with ")
                + String(self.num_parts - self.part_index) + " parts unconsumed"
            )
        if (
            self.logical != self.plan.total_frames
            or self.dropped != self.plan.pad_frames
            or self.emitted != self.plan.output_frames
        ):
            raise Error(
                String("MiniMax-H3 video temporal stream: frame plan mismatch:")
                + " logical_frames=" + String(self.logical)
                + " total_frames=" + String(self.plan.total_frames)
                + " dropped_frames=" + String(self.dropped)
                + " pad_frames=" + String(self.plan.pad_frames)
                + " emitted=" + String(self.emitted)
                + " output_frames=" + String(self.plan.output_frames)
            )


def minimax_h3_video_decode_temporal_streamed[
    LATENT_TILE_H: Int, LATENT_TILE_W: Int, HEADS: Int, DIM_HEAD: Int, NUM_SUFFIX: Int,
    TOKENS_PER_CLIP: Int,
](
    decoder: MiniMaxH3VideoDecoderDevice,
    latents: Tensor,
    tconfig: MiniMaxH3VideoTemporalConfig,
    tiling: MiniMaxH3TilingConfig,
    ctx: DeviceContext,
) raises -> Tensor:
    """Drives the stream and concatenates — SAME RESULT as
    `minimax_h3_video_decode_temporal`, same peak memory as it too, so this is
    NOT the memory win. It exists so a gate can assert the streaming loop and
    the folded loop agree bit for bit before the pipeline is allowed to trust
    the incremental path (see minimax_h3_video_temporal_streaming_probe.mojo).
    Real callers should drive `MiniMaxH3TemporalDecodeStream` directly and
    consume each part."""
    var stream = MiniMaxH3TemporalDecodeStream(latents, tconfig, tiling, ctx)
    var parts = List[TArc]()
    while stream.has_next():
        var p = stream.next_part[
            LATENT_TILE_H, LATENT_TILE_W, HEADS, DIM_HEAD, NUM_SUFFIX, TOKENS_PER_CLIP
        ](decoder, ctx)
        if p:
            parts.append(TArc(p.value().clone(ctx)))
    stream.finish()
    if len(parts) == 0:
        raise Error("MiniMax-H3 video temporal streamed decode: produced zero parts")
    var dec = parts[0][].clone(ctx)
    for i in range(1, len(parts)):
        dec = concat(1, ctx, dec, parts[i][])
    return dec^

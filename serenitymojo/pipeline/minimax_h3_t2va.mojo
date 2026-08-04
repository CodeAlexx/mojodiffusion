# serenitymojo/pipeline/minimax_h3_t2va.mojo — MiniMax-H3 text-to-video+audio,
# pure Mojo. Modeled on serenitymojo/pipeline/wan22_t2v.mojo (the house
# pattern for a device video pipeline): comptime geometry, preflight before
# `DeviceContext()`, a scoped denoise stage so resident weights die before
# decode, request/result JSON + per-stage timings.
#
# ── FLOW ──────────────────────────────────────────────────────────────────
#   prompt -> tokenize + streamed Qwen3-VL-32B encode (minimax_h3_conditioning)
#          -> hidden_states[50] [1,seq,5120]
#   packed-sequence geometry (dual video/audio schedule, position_ids,
#          token_tags) -> MM-RoPE tables -> AdaLN modulation cache (ONE
#          streamed pass over all 50 blocks' adaln_proj, replacing 24.29 GiB
#          of resident weight with ~0.92 GiB of precomputed rows)
#   N steps x [frontend embed -> 50 streamed blocks -> final layer -> Euler
#          step (video + audio, independently scheduled)]
#   -> video/audio patch-token latents -> AUDIO: unpack -> denormalize
#          (latents_mean/latents_std) -> host BigVGAN decode (per stereo
#          channel) -> audio.wav.  VIDEO: stubbed, see "VIDEO/AUDIO DECODE".
#
# ── THIS PASS'S SCOPE (team-lead instruction, second pass) ───────────────
# First pass: "structure + preflight + the denoise loop over real streamed
# blocks; report before wiring decode" — done, both decodes stubbed.
# THIS pass: "wire the audio decode... video stays stubbed, audio does not
# have to be" — the audio VAE is the ONLY model surface in this whole port
# verified against REAL released weights (12 checks, ~1e-6,
# minimax_h3_audio_real_weights_parity.mojo). Video remains STUBBED with a
# loud "not wired" error — see `_minimax_h3_decode_video_stub` below. The
# denoise loop is real (unchanged from pass 1): real streamed block loads
# via `minimax_h3_load_block_device`, a real modulation cache, real Euler
# steps through real `ops/tensor_algebra` GPU kernels.
#
# ── PATCH-TOKEN-SPACE ACCUMULATION (not latent-grid space) ────────────────
# The Euler-accumulating video/audio state is kept in PATCH-TOKEN row form
# the WHOLE denoise loop (`[Nv, video_patch_dim=96]` / `[Na, audio_latents_
# dim=32]`), not `[C,T,H,W]` / `[2,C,Ta]`. Three reasons: (1) that is
# EXACTLY the shape `minimax_h3_frontend_embed` consumes for `video_rows`/
# `audio_rows` and EXACTLY the shape `minimax_h3_final_layer` returns for
# `video_out`/`audio_out` — no patchify/unpatchify inside the step loop at
# all; (2) i.i.d. standard-normal noise sampled directly in patch-token
# space is distributionally identical to noise sampled in latent-grid space
# and then patchified (patchify is a pure bijective index permutation of
# i.i.d. Gaussian values) — a valid simplification, not a numerical
# shortcut, though it does mean a given seed produces DIFFERENT exact bytes
# than "sample in latent space, then patchify" would; (3) unpatchify/audio-
# unpack is then needed exactly ONCE, at the very end, not every step. Audio
# unpack REUSES `models/minimax_h3/rearrange.mojo::minimax_h3_unpack_audio`
# directly (host index arithmetic, no float math, explicitly authorized —
# team-lead: "import from them") rather than re-deriving it; video unpatchify
# is not reached this pass (video decode is stubbed before it would be
# needed).
#
# ── DUAL SCHEDULE / GLOBAL ADALN ADDRESSING (the part most likely to be
# gotten wrong by inspection alone) ──────────────────────────────────────
# `models/dit/minimax_h3_sampling.mojo::minimax_h3_build_sampling_row_
# timesteps` builds a LOCAL, per-call, value-deduplicated {values, indices}
# pair — correct for the oracle's "adaln_proj loaded fresh every forward"
# design, where a brand-new modulation tensor exists for exactly that one
# call. That is explicitly NOT this pipeline's design: `models/dit/
# minimax_h3_modcache.mojo` builds ONE modulation cache ONCE, up front, and
# every one of the `num_steps` forward calls addresses rows in that SAME
# cache. Its row layout must therefore be GLOBAL and FIXED before the loop
# starts, not re-derived (and re-deduplicated) locally every step.
#
# This pipeline uses the fp8_policy.mojo-documented layout directly: row
# `2*i` is step `i`'s VIDEO timestep (shared by video AND text rows — the
# oracle's own convention, packing.mojo: "text rows ... inherit the video
# timestep"), row `2*i+1` is step `i`'s AUDIO timestep. `distinct_timesteps
# = 2*num_steps` exactly, matching `models/minimax_h3/fp8_policy.mojo::
# minimax_h3_adaln_distinct_timesteps(steps, False, False)` for a
# conditioning-free t2va run. `_minimax_h3_global_timestep_row` below
# builds this per-row GLOBAL index directly from `geometry.token_tags` and
# the step index — deliberately NOT via `minimax_h3_build_sampling_row_
# timesteps` (that function's local dedup would simply be wrong here: it
# hands back an index into a { <=2 values } table that only exists for the
# DURATION of one call — step_index 0 or 1 from THAT table means nothing to
# the global cache). This is a NEW, pipeline-local index builder, not a
# reimplementation of anything already gated: it is pure row bookkeeping,
# no float arithmetic. A collapsed-to-1-distinct-value step (only step 0,
# where video and audio sigma both equal exactly 1.0 by construction — see
# minimax_h3_sampling.mojo's dual-schedule header) still gets 2 GLOBAL rows
# here, computed independently; adaln_proj is a deterministic function of
# temb, so computing the same row's value twice under two different global
# indices wastes at most 2 rows over the whole run, never wrong.
#
# `minimax_h3_final_layer`'s `timestep_indices` parameter uses this SAME
# global-row array directly (final AdaLN has no per-modality tag, per that
# function's own doc); `minimax_h3_block_forward`'s `adaln_indices` uses it
# one step removed, through the REUSED `minimax_h3_dit.minimax_h3_adaln_
# rows(global_row, token_tags)` (`timestep_index*3+tag`).
#
# ── NO CLASSIFIER-FREE GUIDANCE ─────────────────────────────────────────
# Grepped every H3 oracle/device file for "guidance"/"cfg"/"uncond": none
# exists anywhere in the model or conditioner code. One conditional forward
# per step, no negative prompt, no cond/uncond blend — unlike wan22's CFG
# loop, which this file's structure otherwise mirrors closely.
#
# ── FIXED PROMPT LENGTH (comptime, no attention mask — a real, currently
# UNFILLED gap, not silently papered over) ──────────────────────────────
# `minimax_h3_block_forward`'s self-attention (its own block-math header)
# and the token refiner's `sdpa_nomask` both run with NO mask over the FULL
# packed sequence — every row attends to every other row unconditionally.
# Unlike wan22's cross-attention-only text padding (padded KV rows no OTHER
# row's Q needs to ignore), H3 has ONE packed self-attention document: a
# padded/truncated text region would pollute EVERY row's attention, not
# just its own. Neither `sdpa_flash_infer_fwd` nor `sdpa_nomask` currently
# accept a mask in how this port calls them. So `TEXT_TOKENS` below is a
# COMPTIME budget the tokenized prompt's length must match EXACTLY (checked
# at runtime, rebuild-hint error otherwise) — mirroring wan22's own
# rebuild-on-geometry-change discipline, extended to prompt length. Wiring
# variable-length prompts without a rebuild needs masked attention added to
# both call sites first; that is out of this pass's scope and is reported,
# not silently assumed away.
#
# ── POSSIBLE PRECISION-BOUNDARY FINDING (flagged, NOT fixed — not this
# file's code) ───────────────────────────────────────────────────────────
# `models/dit/minimax_h3_frontend.mojo::minimax_h3_frontend_embed` casts
# the F32 video/audio patch-embed output down to BF16 via `ops/cast.
# cast_tensor` (its own "IMPLIED CAST BOUNDARY" note), not `ops/torch_bf16.
# torch_f32_to_bf16_rne`. That is exactly the F32->BF16 boundary this
# repo's DTYPE RULE (see models/dit/minimax_h3_sampling.mojo's header and
# this file's own `minimax_h3_prepare_model_input` reuse) requires the
# RNE-matching cast for, and it runs once per denoising step — the same
# "per-step rounding bias compounds and decorrelates from torch while every
# single-forward gate still reads cos ~0.9998" class of bug the NAVA port
# hit. `minimax_h3_frontend.mojo` is not this file's to edit (not listed
# among files I own), so this is reported to team-lead rather than patched
# here; the fix, if judged real, is a one-line swap at that file's two
# `cast_tensor(..._f32, STDtype.BF16, ctx)` call sites.
#
# ── VIDEO DECODE: STUBBED, NOT WIRED ──────────────────────────────────────
# Team-lead instruction, unchanged from pass 1: the existing device port(s)
# are unverified/being rebuilt against the creator's own source (FFN
# gate/value order, fused-qkv split). `models/vae/minimax_h3_video_decoder_
# device.mojo` exists on disk but its gate status is unknown to this file;
# it is deliberately NOT imported here. `_minimax_h3_decode_video_stub`
# below is the seam: it takes exactly the row-form tensor the denoise loop
# already produces and raises a named "not wired" error. Dropping in the
# corrected decoder later means adding one unpatchify call + one decoder
# call inside that function — this file's structure does not change.
#
# ── AUDIO DECODE: WIRED, HOST-SIDE (the model itself is host float32) ────
# `models/minimax_h3/audio_decoder.mojo` — `minimax_h3_audio_decode` and its
# staged siblings — is imported directly (not reimplemented; team-lead:
# "import from them", and this file only ever CALLS it, never edits it).
# It is entirely `List[Float32]` host arithmetic (its own header: "Host
# float32, batch 1"); no GPU/Tensor/DeviceContext anywhere inside it. This
# pipeline's boundary is therefore: `audio_state.to_host(ctx)` once, after
# the denoise loop, then everything through the waveform is host-side, then
# ONE upload (`Tensor.from_host`) to hand the finished stereo waveform to
# `serenitymojo/audio/wav.mojo::save_wav` (which does its own `to_host`
# internally — this file's upload is so `save_wav`'s existing [2,L] Tensor
# contract does not need a second, Tensor-accepting overload).
#
# STEREO: per the vendor's own README (team-lead) and confirmed structurally
# by `models/minimax_h3/rearrange.mojo::minimax_h3_unpack_audio`'s own
# layout (`[2, C, T]`, "one batch item per stereo channel" — that file's own
# header), H3's audio VAE runs the SAME mono decoder independently on each
# of 2 channel-major latent blocks and the two are recombined by the
# caller — it is not a stereo-aware model. `minimax_h3_audio_decode` is
# therefore called TWICE here, once per unpacked stereo item.
#
# DENORMALIZATION: the DiT's audio output lives in NORMALIZED latent space
# (the standard latent-diffusion convention — train/sample on whitened
# latents, denormalize only at the VAE boundary); `minimax_h3_audio_decode`
# expects the VAE's OWN latent space. `FL2VA/audio_vae/config.json`'s
# `latents_mean`/`latents_std` (32 values each, one per channel) are the
# fix: `latent = normalized*std + mean`, applied per-channel before decode.
# Hardcoded below from that real, landed config.json (verified 2026-08-02)
# rather than parsed at runtime — the SAME convention `minimax_h3_dit.mojo::
# minimax_h3_released_config` already uses for the transformer's config
# ("the vendor's own, not inferred"); no general JSON-value parser exists
# in this repo for arbitrary nested config keys, only the safetensors
# flat-header parser (`io/json_header.mojo`), which is a different format.
#
# ── PARTIAL MODE (`max_blocks`, 5th argv, default = full 50) ────────────────
# The transformer download is still in progress (8/13 shards as of this
# writing); full preflight walks all 50 blocks and correctly refuses to run
# until the last byte lands. `max_blocks < 50` is an EXPLICIT, LOUD bypass
# for exercising the rest of the chain end to end sooner:
#   * preflight only checks blocks `[0, max_blocks)` (via a `run_config`
#     copy of the released config with `num_layers` overridden — every
#     function that loops per-block already loops `range(config.num_
#     layers)`, so this needs no changes to any function's own code, only
#     which config value it is handed) — see PARTIAL-DOWNLOAD SHARD BYPASS
#     above for how the shard files themselves are opened.
#   * conditioning is STUBBED: the text_encoder needs its layer-50 weights,
#     which live in shard 11 (not downloaded), so a real encode is not
#     possible yet. A fixed-seed random `[1, TEXT_TOKENS, text_dim]` tensor
#     stands in — see `_minimax_h3_stub_conditioning` below. THE REAL PROMPT
#     TO WIRE ONCE THE TEXT ENCODER IS COMPLETE:
#     `output/minimax_h3_prompts/rain_ring_box_shot1.txt` (245 tokens,
#     tokenizes clean under H3's own processor) — not used tonight because
#     it does not match this binary's compiled TEXT_TOKENS budget (see file
#     header FIXED PROMPT LENGTH); wiring it needs H3_TEXT_TOKENS=245 and a
#     rebuild, same as any other prompt-length change.
#   * an UNMISSABLE banner prints (stdout, every phase, and result.json):
#     this is an N-of-50-layer run with STUBBED conditioning. IT IS NOT A
#     VALID MINIMAX-H3 GENERATION. It is a plumbing test — proving the
#     chain (tokenize/stub -> conditioning -> frontend embed -> packed
#     sequence -> modcache -> N real streamed blocks -> Euler steps ->
#     final layer -> audio latents -> denormalize -> BigVGAN -> audio.wav)
#     holds together end to end, which no isolated per-module gate can show.
# Audio decode, video stub, and every other piece of this file are
# UNCHANGED by `max_blocks` — the audio path is exercised for real even in
# a 2-of-50-block run, on whatever (meaningless, undertrained-by-omission)
# audio latents that run produces.
#
# argv: <prompt> <out_dir> [steps=30] [seed=0] [max_blocks=50]
#
# ── CHECKPOINT LAYOUT (matches every H3 device module's own defaults) ───
#   transformer:   .../MiniMax-H3/FL2VA/transformer    (61.73 GiB, 13 shards)
#   text_encoder:  .../MiniMax-H3/FL2VA/text_encoder    (62.13 GiB, 14 shards)
#   processor:     .../MiniMax-H3/FL2VA/processor       (tokenizer.json + config)
#   audio_vae:     .../MiniMax-H3/FL2VA/audio_vae/model.safetensors
#                  (single file, 605 MB, COMPLETE on disk — confirmed by
#                  `ls`, 2026-08-02 — unlike the other three checkpoints)
# As of this writing (team-lead status): transformer shards landing (6/13);
# text_encoder shards not yet down. Preflight below fails loudly and
# specifically on whichever is still incomplete — that is a VALID outcome
# of running this file today, not a bug in it. The audio_vae path is the
# real product location (`.../FL2VA/audio_vae/model.safetensors`), NOT the
# gate's own dev-time path (`/home/alex/Downloads/MiniMax-H3-audio_vae.
# safetensors`, checked: no longer present).
#
# LINKER: the block stack's `sdpa_flash_infer_fwd` needs the cuDNN SDPA
# shim; plain `mojo run -I .` fails at the first block's attention call
# with "Symbols not found: flame_cudnn_sdpa_bf16". Build with:
#   -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.sys import argv
from std.sys.defines import get_defined_int
from std.time import perf_counter_ns
from std.collections import Dict, List
from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.ops.patchify3d import unpatchify3d
from serenitymojo.image.png import save_png, ValueRange
from serenitymojo.models.vae.minimax_h3_video_decoder_device import (
    minimax_h3_video_released_decoder_config,
    minimax_h3_video_decoder_device_load,
    minimax_h3_video_decode_device,
)
from serenitymojo.pipeline.minimax_h3_video_vae_spatial_tiling import (
    minimax_h3_video_tiled_decode,
    minimax_h3_video_released_tiling_config,
)
from serenitymojo.pipeline.minimax_h3_video_vae_temporal import (
    minimax_h3_video_decode_temporal,
    minimax_h3_video_released_temporal_config,
    MiniMaxH3TemporalDecodeStream,
)
from serenitymojo.pipeline.minimax_h3_video_vae_pixel_norm import (
    minimax_h3_video_pixel_denormalize,
    minimax_h3_pixel_norm_constants,
)
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import sys_system
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import reshape, permute, slice, add, mul
from serenitymojo.serve.product_manifest import json_escape, json_bool, write_text_file
from serenitymojo.audio.wav import save_wav

from serenitymojo.models.dit.minimax_h3_dit import (
    MiniMaxH3DiTConfig,
    minimax_h3_released_config,
    minimax_h3_adaln_rows,
    minimax_h3_block_tensor_names,
    minimax_h3_expected_shape,
    minimax_h3_block_forward,
)
from serenitymojo.models.dit.minimax_h3_loader_device import (
    minimax_h3_load_block_device,
    minimax_h3_load_qkv_device,
    minimax_h3_load_fc1_device,
)
from serenitymojo.models.dit.minimax_h3_fp8_resident import (
    MiniMaxH3ResidentFp8,
    minimax_h3_build_resident_fp8,
    minimax_h3_resident_block_weights,
)
from serenitymojo.models.dit.minimax_h3_modcache import (
    MiniMaxH3ModCache,
    minimax_h3_check_modcache_weights,
    minimax_h3_build_modulation_cache,
)
from serenitymojo.models.dit.minimax_h3_rope import build_minimax_h3_rope_tables
from serenitymojo.models.dit.minimax_h3_frontend import (
    MiniMaxH3FrontendEmbed,
    MiniMaxH3FrontendOutput,
    minimax_h3_frontend_embed,
    minimax_h3_final_layer,
    minimax_h3_timestep_embedding,
)
from serenitymojo.models.text_encoder.minimax_h3_conditioning import (
    MiniMaxH3ConditioningOutput,
    minimax_h3_encode_conditioning,
)
from serenitymojo.models.dit.minimax_h3_sampling import (
    MiniMaxH3DualSchedule,
    MiniMaxH3SamplingGeometry,
    minimax_h3_build_sampling_geometry,
)
from serenitymojo.models.minimax_h3.rearrange import minimax_h3_unpack_audio
from serenitymojo.models.minimax_h3.audio_decoder import (
    MiniMaxH3AudioDecoderConfig,
    MiniMaxH3AudioWeights,
    minimax_h3_audio_decode,
)
from serenitymojo.models.minimax_h3_device.audio_decoder_device import (
    minimax_h3_audio_device_weights,
    minimax_h3_audio_decode_device,
)


# ── Checkpoint paths (matches every other H3 device module's own defaults) ──
comptime H3_ROOT = "/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA"
comptime TRANSFORMER_DIR = H3_ROOT + "/transformer"
comptime TEXT_ENCODER_DIR = H3_ROOT + "/text_encoder"
comptime PROCESSOR_DIR = H3_ROOT + "/processor"
comptime AUDIO_VAE_PATH = H3_ROOT + "/audio_vae/model.safetensors"
comptime VIDEO_VAE_DIR  = H3_ROOT + "/video_vae/source"
comptime AUDIO_SAMPLE_RATE = 32000

# 13 shards, `model-{i:05d}-of-00013.safetensors` — the released transformer's
# own layout (minimax_h3_loader_device.mojo header: "61.73 GiB bf16 across 13
# shards"; confirmed by the exact missing-file name this pipeline's own
# preflight has reported: "model-00002-of-00013.safetensors").
comptime TRANSFORMER_SHARD_COUNT = 13


# ═════════════════════════════════════════════════════════════════════════════
# PARTIAL-DOWNLOAD SHARD BYPASS. `ShardedSafeTensors.open(dir)` parses the
# index.json's FULL weight_map and eagerly opens EVERY shard file the index
# references — including ones not yet downloaded — and dies on the first
# missing one (io/sharded.mojo's own `open`). That is correct default
# behaviour for a real run, and is why `--max-blocks` (see file header
# "PARTIAL MODE") does not change the code path when it equals the full
# layer count. For a PARTIAL run it is exactly the wrong thing: it refuses
# to serve tensors from shards that ARE present just because SOME other
# shard isn't.
#
# The bypass: build a `ShardedSafeTensors` directly via its public
# constructor (`ShardedSafeTensors(shards, name_to_shard)`,
# io/sharded.mojo) over only the shard FILES that exist on disk right now,
# mapping each shard's own tensor names (read straight off its own header,
# not the index) to it. This is the SAME technique `models/dit/parity/
# minimax_h3_real_block_device_probe.mojo::_open_single_shard_no_index`
# already established for this exact checkpoint (one hardcoded shard);
# this generalizes it to "however many of the known 13 files happen to be
# present", tried newest-numbered-file-last so partial in-progress
# downloads (a shard file existing but only half-written) are the LAST
# thing this touches, not the first.
# ═════════════════════════════════════════════════════════════════════════════
def _h3_zero_pad5(n: Int) -> String:
    var s = String(n)
    while s.byte_length() < 5:
        s = String("0") + s
    return s


def _h3_shard_filename(index_1based: Int, total: Int) -> String:
    return (
        String("model-") + _h3_zero_pad5(index_1based) + String("-of-")
        + _h3_zero_pad5(total) + String(".safetensors")
    )


def _minimax_h3_open_partial_transformer_shards(
    dir: String,
) raises -> ShardedSafeTensors:
    var shards = List[ArcPointer[SafeTensors]]()
    var name_to_shard = Dict[String, Int]()
    var present_files = List[String]()
    for i in range(1, TRANSFORMER_SHARD_COUNT + 1):
        var fname = _h3_shard_filename(i, TRANSFORMER_SHARD_COUNT)
        var path = dir + String("/") + fname
        try:
            var st = SafeTensors.open(path)
            var names = st.names()
            var idx = len(shards)
            for j in range(len(names)):
                name_to_shard[names[j]] = idx
            shards.append(ArcPointer(st^))
            present_files.append(fname)
        except e:
            # Not yet downloaded (or a genuine open failure) — expected
            # during a partial download; this shard's tensors are simply
            # absent from `name_to_shard`, and anything that needs one
            # fails loudly, by name, at the point that actually reads it
            # (preflight, if it checks that tensor; the loader otherwise).
            pass
    if len(shards) == 0:
        raise Error(
            String("minimax_h3_t2va: no transformer shard files present at ")
            + dir
        )
    print("  preflight: partial-mode shard bypass — present files:", len(present_files), "of", TRANSFORMER_SHARD_COUNT)
    return ShardedSafeTensors(shards^, name_to_shard^)


def _minimax_h3_open_transformer_shards(dir: String, partial_mode: Bool) raises -> ShardedSafeTensors:
    """Single call site for `main()` so it only ever declares ONE `var
    transformer_shards` regardless of which path was taken — Mojo does not
    support conditionally initializing one outer `var` from two different
    branches, so the branch lives here instead."""
    if partial_mode:
        return _minimax_h3_open_partial_transformer_shards(dir)
    return ShardedSafeTensors.open(dir)


# ── Geometry (COMPTIME — rebuild to change; see file header "FIXED PROMPT
# LENGTH" for why TEXT_TOKENS is comptime too, not just video/audio). ────────
comptime HEIGHT = get_defined_int["H3_HEIGHT", 480]()
comptime WIDTH = get_defined_int["H3_WIDTH", 832]()
# FRAMES must already be of the form 17n+5 (video VAE chunking, packing.mojo
# minimax_h3_align_num_frames) — this file does not round up for you, it
# fails loudly instead (see _preflight_geometry).
comptime FRAMES = get_defined_int["H3_FRAMES", 22]()
comptime TEXT_TOKENS = get_defined_int["H3_TEXT_TOKENS", 32]()

comptime LATENT_H = HEIGHT // 16
comptime LATENT_W = WIDTH // 16
comptime PATCH_H = 2
comptime PATCH_W = 2
comptime ROWS_PER_FRAME = (LATENT_H // PATCH_H) * (LATENT_W // PATCH_W)

# Mirrors packing.mojo::minimax_h3_video_latent_num_frames's formula exactly
# (17n+5 -> 5n+2), inlined as comptime integer arithmetic — exact, no float
# involved, safe to duplicate (unlike the position-grid math this port
# deliberately never re-derives). NOT re-validated at runtime against the
# reused function; see file header — this pass follows wan22's own "geometry
# is comptime, trust the formula" convention rather than adding a runtime
# cross-check nothing else in this codebase's pipelines does either.
comptime NUM_LATENT_FRAMES = (FRAMES - 5) // 17 * 5 + 2

# Mirrors packing.mojo::minimax_h3_audio_latent_num_frames's formula
# (round(frames/24*40)). Uses Mojo's native `round` (half-away-from-zero),
# NOT packing.mojo's round-half-EVEN `_round_half_even` — a real, accepted
# divergence at exact .5 boundaries only (comptime arithmetic, not a device
# kernel, so this is a one-time build-time value, not a per-element risk).
comptime NUM_AUDIO_LATENTS = Int(round(Float64(FRAMES) / 24.0 * 40.0))

comptime NUM_VIDEO_ROWS = NUM_LATENT_FRAMES * ROWS_PER_FRAME
comptime NUM_AUDIO_ROWS = NUM_AUDIO_LATENTS * 2
comptime SEQ_LEN = TEXT_TOKENS + NUM_AUDIO_ROWS + NUM_VIDEO_ROWS

comptime H3_HEADS = 56
comptime H3_HEAD_DIM = 128

comptime DEFAULT_STEPS = 30
comptime DEFAULT_SEED = 0

# ── FP8-RESIDENT DENOISER (H3_FP8_RESIDENT, default OFF) ─────────────────────
# Build with `-D H3_FP8_RESIDENT=1` to quantize all 50 blocks' big GEMM
# weights ONCE (bf16 -> E4M3 + per-row F32 scale, on device, ~18 GiB
# resident) and dequant per block per step instead of re-streaming ~36 GiB of
# block weights from disk EVERY denoise step. The residency arithmetic is
# `models/minimax_h3/fp8_policy.mojo`'s (fits 24 GiB with spare, adaLN already
# evicted to the modcache); the runtime is `models/dit/minimax_h3_fp8_resident
# .mojo` (krea2 fp8-resident precedent, same gated encode/dequant kernels).
# Gated vs the bf16-streamed path by models/dit/parity/
# minimax_h3_fp8_resident_gate.mojo. Default OFF: a normal build takes the
# `@parameter else` branch below, which is the unmodified streaming loop.
comptime DIT_FP8_RESIDENT = get_defined_int["H3_FP8_RESIDENT", 0]()


# ═════════════════════════════════════════════════════════════════════════════
# PREFLIGHT — everything below runs BEFORE `DeviceContext()`. Missing shards,
# wrong dtype, bad geometry all fail loudly here with named errors, per the
# runtime-port rule (already the convention `MiniMaxH3DiTConfig.validate` and
# `minimax_h3_check_modcache_weights` follow — this file adds the ONE more
# check those two don't cover: the 50 main-stack blocks' tensors, checked by
# SHAPE/DTYPE against the shard INDEX only, no Tensor materialized, no ctx
# needed, mirroring modcache.mojo's own `_check_adaln_tensor` style).
# ═════════════════════════════════════════════════════════════════════════════
def _preflight_geometry() raises:
    if (FRAMES - 5) % 17 != 0:
        raise Error(
            String("minimax_h3_t2va: FRAMES=") + String(FRAMES)
            + " is not of the form 17n+5 (the video VAE's chunk size); edit"
            " H3_FRAMES and rebuild — e.g. 5, 22, 39, 56, ..."
        )
    if HEIGHT <= 0 or WIDTH <= 0 or HEIGHT % 32 != 0 or WIDTH % 32 != 0:
        raise Error(
            String("minimax_h3_t2va: H3_HEIGHT/H3_WIDTH must be positive")
            + " multiples of 32 (the canvas-rounding contract packing.mojo::"
            "minimax_h3_resolve_canvas_size enforces); got "
            + String(WIDTH) + "x" + String(HEIGHT)
        )
    if TEXT_TOKENS <= 0:
        raise Error("minimax_h3_t2va: H3_TEXT_TOKENS must be positive")
    if NUM_LATENT_FRAMES <= 0 or NUM_AUDIO_LATENTS <= 0:
        raise Error(
            "minimax_h3_t2va: derived NUM_LATENT_FRAMES/NUM_AUDIO_LATENTS"
            " is non-positive — FRAMES too small; rebuild with a larger"
            " H3_FRAMES"
        )


def _preflight_block_tensors(
    shards: ShardedSafeTensors, config: MiniMaxH3DiTConfig
) raises:
    """Every main-stack block's 8 tensors — presence, shape, dtype — checked
    against the shard INDEX (mmap'd header lookups only, no tensor bytes
    read, no DeviceContext needed). A partial download then fails with ONE
    clear message naming the first missing/malformed tensor, mirroring
    minimax_h3_modcache.mojo's minimax_h3_check_modcache_weights (called
    separately, for the adaLN tensors this function does not touch)."""
    for layer in range(config.num_layers):
        var names = minimax_h3_block_tensor_names(layer)
        for i in range(len(names)):
            ref name = names[i]
            if not shards.has_tensor(name):
                raise Error(
                    String("minimax_h3_t2va preflight: missing block tensor ")
                    + name
                )
            var info = shards.tensor_info(name)
            var want = minimax_h3_expected_shape(name, config)
            if len(info.shape) != len(want):
                raise Error(
                    String("minimax_h3_t2va preflight: rank mismatch for ")
                    + name
                )
            for d in range(len(want)):
                if info.shape[d] != want[d]:
                    raise Error(
                        String("minimax_h3_t2va preflight: shape mismatch for ")
                        + name
                    )
            if info.dtype != STDtype.BF16:
                raise Error(
                    String("minimax_h3_t2va preflight: ") + name
                    + " is not BF16"
                )


def _h3_fp32_frontend_keys() -> List[String]:
    var out = List[String]()
    out.append(String("video_patch_proj.weight"))
    out.append(String("video_patch_proj.bias"))
    out.append(String("audio_patch_proj.weight"))
    out.append(String("audio_patch_proj.bias"))
    out.append(String("time_embedder.proj_in.weight"))
    out.append(String("time_embedder.proj_in.bias"))
    out.append(String("time_embedder.proj_out.weight"))
    out.append(String("time_embedder.proj_out.bias"))
    out.append(String("final_layer.video_out.weight"))
    out.append(String("final_layer.video_out.bias"))
    out.append(String("final_layer.audio_out.weight"))
    out.append(String("final_layer.audio_out.bias"))
    return out^


def _preflight_frontend_tensors(
    shards: ShardedSafeTensors, config: MiniMaxH3DiTConfig
) raises:
    """The 12 fp32 dtype-trap tensors plus condition_proj/token_refiner/
    final_layer.norm — everything `_minimax_h3_load_frontend_weights` (below)
    will load. Same index-only-check discipline as `_preflight_block_
    tensors`."""
    var fp32_keys = _h3_fp32_frontend_keys()
    for i in range(len(fp32_keys)):
        var name = fp32_keys[i]
        if not shards.has_tensor(name):
            raise Error(
                String("minimax_h3_t2va preflight: missing frontend fp32 tensor ")
                + name
            )
        if shards.tensor_info(name).dtype != STDtype.F32:
            raise Error(
                String("minimax_h3_t2va preflight: ") + name + " is not F32"
                " (one of the checkpoint's 12 dtype-trap tensors)"
            )
    var bf16_names = List[String]()
    bf16_names.append(String("condition_proj.weight"))
    bf16_names.append(String("condition_proj.bias"))
    bf16_names.append(String("token_refiner.final_norm.weight"))
    bf16_names.append(String("final_layer.norm.weight"))
    for layer in range(config.token_refiner_num_layers):
        var p = String("token_refiner.blocks.") + String(layer)
        bf16_names.append(p + ".norm1.weight")
        bf16_names.append(p + ".attn.qkv_proj.weight")
        bf16_names.append(p + ".attn.q_norm.weight")
        bf16_names.append(p + ".attn.k_norm.weight")
        bf16_names.append(p + ".attn.out_proj.weight")
        bf16_names.append(p + ".norm2.weight")
        bf16_names.append(p + ".mlp.fc1.weight")
        bf16_names.append(p + ".mlp.fc2.weight")
    for i in range(len(bf16_names)):
        ref name = bf16_names[i]
        if not shards.has_tensor(name):
            raise Error(
                String("minimax_h3_t2va preflight: missing frontend tensor ")
                + name
            )
        if shards.tensor_info(name).dtype != STDtype.BF16:
            raise Error(
                String("minimax_h3_t2va preflight: ") + name + " is not BF16"
            )


# ═════════════════════════════════════════════════════════════════════════════
# Frontend weight loading — the small, resident-for-the-whole-run chunk
# (video/audio patch-proj, time_embedder, condition_proj, token_refiner,
# final_layer.norm/video_out/audio_out). NOT adaln_proj (minimax_h3_modcache's
# job) and NOT the 50 main blocks (streamed per layer, per step, in the
# denoise loop below). Reuses minimax_h3_load_qkv_device/minimax_h3_load_
# fc1_device (minimax_h3_loader_device.mojo) for the token refiner's fused
# tensors — the SAME de-interleave/swap rewrite the main blocks need, since
# the row-order transform depends only on (heads, head_dim, in_features), not
# which block "kind" is asking.
# ═════════════════════════════════════════════════════════════════════════════
def _minimax_h3_load_frontend_weights(
    shards: ShardedSafeTensors, config: MiniMaxH3DiTConfig, ctx: DeviceContext
) raises -> Dict[String, ArcPointer[Tensor]]:
    var w = Dict[String, ArcPointer[Tensor]]()
    var fp32_keys = _h3_fp32_frontend_keys()
    for i in range(len(fp32_keys)):
        var name = fp32_keys[i]
        w[name] = ArcPointer(Tensor.from_view(shards.tensor_view(name), ctx))

    w["condition_proj.weight"] = ArcPointer(
        Tensor.from_view(shards.tensor_view("condition_proj.weight"), ctx)
    )
    w["condition_proj.bias"] = ArcPointer(
        Tensor.from_view(shards.tensor_view("condition_proj.bias"), ctx)
    )

    var heads = config.num_attention_heads
    var head_dim = config.attention_head_dim
    var hidden = config.hidden_size
    var ffn = config.ffn_hidden_size
    for layer in range(config.token_refiner_num_layers):
        var p = String("token_refiner.blocks.") + String(layer)
        w[p + ".norm1.weight"] = ArcPointer(
            Tensor.from_view(shards.tensor_view(p + ".norm1.weight"), ctx)
        )
        var qkv_name = p + ".attn.qkv_proj.weight"
        w[qkv_name] = ArcPointer(
            minimax_h3_load_qkv_device(shards, qkv_name, heads, head_dim, hidden, ctx)
        )
        w[p + ".attn.q_norm.weight"] = ArcPointer(
            Tensor.from_view(shards.tensor_view(p + ".attn.q_norm.weight"), ctx)
        )
        w[p + ".attn.k_norm.weight"] = ArcPointer(
            Tensor.from_view(shards.tensor_view(p + ".attn.k_norm.weight"), ctx)
        )
        w[p + ".attn.out_proj.weight"] = ArcPointer(
            Tensor.from_view(shards.tensor_view(p + ".attn.out_proj.weight"), ctx)
        )
        w[p + ".norm2.weight"] = ArcPointer(
            Tensor.from_view(shards.tensor_view(p + ".norm2.weight"), ctx)
        )
        var fc1_name = p + ".mlp.fc1.weight"
        w[fc1_name] = ArcPointer(
            minimax_h3_load_fc1_device(shards, fc1_name, ffn, hidden, ctx)
        )
        w[p + ".mlp.fc2.weight"] = ArcPointer(
            Tensor.from_view(shards.tensor_view(p + ".mlp.fc2.weight"), ctx)
        )
    w["token_refiner.final_norm.weight"] = ArcPointer(
        Tensor.from_view(shards.tensor_view("token_refiner.final_norm.weight"), ctx)
    )
    w["final_layer.norm.weight"] = ArcPointer(
        Tensor.from_view(shards.tensor_view("final_layer.norm.weight"), ctx)
    )
    return w^


# ═════════════════════════════════════════════════════════════════════════════
# Global AdaLN row addressing — see file header "DUAL SCHEDULE / GLOBAL
# ADALN ADDRESSING".
# ═════════════════════════════════════════════════════════════════════════════
def _minimax_h3_global_timestep_row(
    token_tags: List[Int], step: Int
) raises -> List[Int]:
    """Row `r`'s index into the GLOBAL `2*num_steps`-row modulation cache for
    denoising step `step`: `2*step` for video/text rows (they share the
    video timestep — packing.mojo's own convention), `2*step + 1` for audio
    rows. Padding rows (tag < 0) get 0 — never read, since `minimax_h3_
    adaln_rows` short-circuits on the tag before touching this value; not
    reachable for a pure t2va layout (no padding), handled anyway."""
    var out = List[Int](capacity=len(token_tags))
    for i in range(len(token_tags)):
        var tag = token_tags[i]
        if tag == 2:
            out.append(2 * step + 1)
        elif tag == 0 or tag == 1:
            out.append(2 * step)
        else:
            out.append(0)
    return out^


# ═════════════════════════════════════════════════════════════════════════════
# Video decode seam — STUBBED. See file header "VIDEO DECODE".
# ═════════════════════════════════════════════════════════════════════════════
# 256 px tile / vae_ratio 16 = 16 latent cells per tile side.
comptime LATENT_TILE = 16

# ── TEMPORAL CHUNKING (H3_VAE_TEMPORAL, default OFF) ─────────────────────────
# The vendor NEVER decodes video as one volume: `klvae.py::decode_base` (:790-
# 797) routes every `use_3d_conv` decode through `decode_temporal` (:678), which
# splits the latent into `tokens_chunk_size + token_overlap` = 7-token clips and
# cross-fades them (`_adaptive_decode` per clip at :733 -> `tiled_decode` at
# :437-441, so TEMPORAL is the OUTER loop and SPATIAL TILING the inner one —
# exactly the nesting pipeline/minimax_h3_video_vae_temporal.mojo implements).
# There is no size threshold in the reference: even a 7-token latent goes
# through it, because the pre-padding trim and the tail split are part of the
# decode, not a large-input accommodation.
#
# MEASURED CONSEQUENCE OF THE DIRECT PATH (this file's default, 2026-08-03):
# it emits `4 * latent_T` frames instead of the requested FRAMES — shot 2 wrote
# 68 frames for H3_FRAMES=56, shot 3 wrote 48 for 39 (output/logs/h3_shots.log
# :73,:148). `decode_temporal` emits exactly `17n+5 = FRAMES`. The direct path
# also runs the ViT decoder at a temporal length it was never trained on
# (latent_T tokens rather than 7) and never trims `frame_pre_padding`.
#
# DEFAULT IS OFF ANYWAY, deliberately: the six-shot batch running as this was
# written builds its remaining per-shot binaries FROM THIS SOURCE on demand
# (output/logs/h3_shots_orchestrator.sh's `if [ ! -x "$BIN" ]` build step), so
# flipping the default mid-batch would silently change shots 5-7. Build with
# `-D H3_VAE_TEMPORAL=1` to get the vendor's composition. Recommendation once
# the GPU frees and the composition gate below passes: make 1 the default.
comptime VIDEO_DECODE_TEMPORAL = get_defined_int["H3_VAE_TEMPORAL", 0]()

# `video_vae/config.json`, mirrored as comptime integers because
# `minimax_h3_video_decode_temporal`'s per-clip sequence length is a compile-
# time parameter. These duplicate `minimax_h3_video_released_temporal_config()`
# and are CHECKED against it at runtime (that function raises on a
# `tokens_per_clip()` mismatch), so the duplication cannot drift silently.
comptime VAE_CLIP_LENGTH = 17
comptime VAE_RATIO_T = 4
comptime VAE_TOKEN_DROP = 3
comptime VAE_TOKENS_CHUNK = (VAE_CLIP_LENGTH + VAE_RATIO_T - 1) // VAE_RATIO_T
comptime VAE_TOKEN_OVERLAP = (
    VAE_TOKENS_CHUNK - VAE_TOKEN_DROP % VAE_TOKENS_CHUNK
) % VAE_TOKENS_CHUNK
comptime VAE_TOKENS_PER_CLIP = VAE_TOKENS_CHUNK + VAE_TOKEN_OVERLAP

# ── STREAMING DECODE (H3_VAE_STREAM_DECODE, default OFF) ─────────────────────
# IMPLIES temporal chunking (it IS the temporal loop, consumed one part at a
# time), so it does not need H3_VAE_TEMPORAL set as well and ignores it.
#
# WHAT IT CHANGES: the non-streaming decode builds the ENTIRE video as one
# device tensor, then denormalizes it (a second full copy), then writes PNGs.
# At the 14-second hero geometry that is a 2.06 GiB pixel volume, a fold that
# peaks near 4.2 GiB, and a 2.06 GiB denormalize copy — against roughly 2.3 GiB
# of headroom left after the denoise loop. This path decodes ONE temporal part,
# denormalizes THAT PART, writes its PNGs, and drops it, so the peak is one
# clip decode plus one part (~50 frames) NO MATTER HOW LONG THE VIDEO IS.
#
# Frame numbering and the resulting files are identical to the non-streaming
# path (shared `_write_rgb_frames`, running offset), so nothing downstream —
# ffmpeg's `frame_%05d.png`, the muxer, the sequence concat — changes.
#
# The vendor's own `_decode_temporal_streaming` (klvae.py:571-676) is the
# structural authority here, but note it preallocates the full output and
# writes into it (:597-607); it removes the fold, not the volume. This goes one
# step further because we can hand each part straight to the PNG writer. See
# the long header note in pipeline/minimax_h3_video_vae_temporal.mojo.
comptime VIDEO_DECODE_STREAM = get_defined_int["H3_VAE_STREAM_DECODE", 0]()


def num_video_rows_of(f: Int, h: Int, w: Int) -> Int:
    return f * (h // 2) * (w // 2)


def _write_rgb_frames(
    rgb: Tensor,          # [1, F, H, W, 3] denormalized, UNIT range
    out_dir: String,
    first_index: Int,     # global frame number of rgb's frame 0
    ctx: DeviceContext,
) raises -> Int:
    """Write one PNG per frame, numbered from `first_index`. Factored out of
    `_minimax_h3_decode_video` so the STREAMING decode path can call it once
    per temporal part with a running offset and never hold the whole video —
    the numbering is identical either way, which is what keeps
    `ffmpeg -i frame_%05d.png` working unchanged."""
    var ps = rgb.shape()
    var frames = ps[1]
    var height = ps[2]
    var width = ps[3]
    for f in range(frames):
        var one = slice(rgb, 1, f, 1, ctx)                       # [1,1,H,W,3]
        var hwc = reshape(one, [height, width, 3], ctx)
        var chw = permute(hwc, [2, 0, 1], ctx)                   # [3,H,W]
        var img = reshape(chw, [1, 3, height, width], ctx)
        var name = String(first_index + f)
        while len(name) < 5:
            name = String("0") + name
        save_png(img, out_dir + "/frame_" + name + ".png", ctx, ValueRange.UNIT)
    return frames


def _minimax_h3_decode_video(
    video_state: Tensor,   # [NUM_VIDEO_ROWS, video_patch_dim] F32, patch-token space
    latent_frames: Int,
    latent_h: Int,
    latent_w: Int,
    latent_channels: Int,  # config.latents_dim (24)
    vae_dir: String,
    out_dir: String,
    ctx: DeviceContext,
) raises -> Int:
    """Patch tokens -> latent grid -> ViT decoder -> denormalize -> PNG frames.
    Returns the number of frames written.

    WIRED 2026-08-03. The stub this replaces existed because our ORIGINAL video
    decoder was ported from the diffusers rewrite and targeted key names the
    released checkpoint does not contain (measured: zero occurrences of proj_in,
    to_q/to_k/to_v, to_out.0, ff.net.*, conv_shortcut across all 560 tensors) —
    it could not have loaded these weights at all. The rebuild under
    models/vae/ targets the NATIVE names and is gated against the vendor's own
    AutoencoderKLLegacy at cos 0.9999999978 (encoder) / 0.9999999999998
    (decoder), essentially F32 noise floor. That is why the stub can go.

    UNPATCHIFY READ ORDER: ops/patchify3d.unpatchify3d reads within-patch as
    (pf, ph, pw, c) — channel FASTEST — which is deliberately NOT the inverse of
    patchify3d's c-slowest order. That asymmetry is the trained convention of the
    model's output linear, documented in that op's own docstring; using
    patchify3d's order here would scramble every frame while producing a
    perfectly-shaped tensor."""
    # ROW ORDER FIX (measured against the vendor's packing.py, 2026-08-03):
    # H3 packs each video row CHANNEL-SLOWEST -- patchify_video_latents
    # permutes to (..., channels, pt, ph, pw) before the flatten, and
    # unpatchify_video_tokens reshapes rows as (c, pt, ph, pw). Our
    # ops/patchify3d.unpatchify3d reads within-patch CHANNEL-FASTEST
    # (pf, ph, pw, c) -- the wan/cosmos convention its docstring says is
    # model-specific. Decoding H3 rows without this reorder scrambles every
    # 2x2 latent patch, which the VAE renders as a 16-px periodic tile
    # artifact (observed on the first full run). Reorder (c,pt,ph,pw) ->
    # (pt,ph,pw,c) per row, then the shared unpatchify is correct.
    var rows5 = reshape(video_state, [num_video_rows_of(latent_frames, latent_h, latent_w), latent_channels, 1, 2, 2], ctx)
    var rows5p = permute(rows5, [0, 2, 3, 4, 1], ctx)
    var rows_cf = reshape(rows5p, [num_video_rows_of(latent_frames, latent_h, latent_w), latent_channels * 4], ctx)
    # [n_patches, C*pf*ph*pw] -> [C, F, H, W]
    var grid_cfhw = unpatchify3d(
        rows_cf, latent_channels, latent_frames, latent_h, latent_w,
        1, 2, 2, ctx,
    )
    # [C,F,H,W] -> [1,F,H,W,C] NDHWC, the house layout the device VAE expects
    var perm = [1, 2, 3, 0]
    var grid_fhwc = permute(grid_cfhw, perm^, ctx)
    var latents = reshape(
        grid_fhwc, [1, latent_frames, latent_h, latent_w, latent_channels], ctx
    )

    # ── VIDEO LATENT DENORMALIZATION. The DiT denoises in NORMALIZED latent
    # space; the VAE expects its OWN space. Measured in the diffusers PR
    # (pr15210.diff:1923,1929): encode `(mean - latents_mean)/latents_std`,
    # decode `z = z * latents_std + latents_mean` — the exact convention the
    # AUDIO path already applies. Skipping this feeds the decoder
    # out-of-distribution input and renders as fine mosaic texture with
    # surviving global structure (observed on the first two full runs).
    # Values hardcoded VERBATIM from the released
    # FL2VA/video_vae/config.json `latents_mean`/`latents_std` (24 per-channel
    # F32 values, verified 2026-08-03). Channel-last NDHWC broadcasts the
    # [24] tensors directly against the trailing axis.
    var vid_lat_mean = [Float32(0.858090341091156), Float32(-0.9606591463088989), Float32(1.0661640167236328), Float32(-0.5090325474739075), Float32(-0.2727581858634949), Float32(-1.3675414323806763), Float32(-0.2553254961967468), Float32(-0.26907554268836975), Float32(-0.5376840829849243), Float32(-0.0464097298681736), Float32(0.6657370328903198), Float32(0.19690127670764923), Float32(-0.5460608005523682), Float32(-0.4035342037677765), Float32(-0.23683024942874908), Float32(0.25928452610969543), Float32(-0.30133944749832153), Float32(0.211341992020607), Float32(-1.1206848621368408), Float32(0.3581933379173279), Float32(-0.04225143790245056), Float32(0.2604829967021942), Float32(0.22864092886447906), Float32(0.7056031823158264)]
    var vid_lat_std = [Float32(1.2223774194717407), Float32(1.2767263650894165), Float32(1.6831774711608887), Float32(1.7549455165863037), Float32(1.5636216402053833), Float32(2.194143533706665), Float32(0.9653137922286987), Float32(1.0569885969161987), Float32(0.841948926448822), Float32(0.7729952931404114), Float32(1.8955937623977661), Float32(0.946841835975647), Float32(0.7996809482574463), Float32(0.44988900423049927), Float32(0.7197399735450745), Float32(0.6936293244361877), Float32(2.961095094680786), Float32(2.7694199085235596), Float32(3.0496184825897217), Float32(2.1088054180145264), Float32(3.276226282119751), Float32(3.1627357006073), Float32(2.2816812992095947), Float32(2.6127843856811523)]
    var lat_mean_t = Tensor.from_host(vid_lat_mean, [24], latents.dtype(), ctx)
    var lat_std_t = Tensor.from_host(vid_lat_std, [24], latents.dtype(), ctx)
    latents = add(mul(latents, lat_std_t, ctx), lat_mean_t, ctx)

    var dcfg = minimax_h3_video_released_decoder_config()
    var decoder = minimax_h3_video_decoder_device_load(vae_dir, dcfg, ctx)
    # TILED, not the per-volume call. video_vae/config.json enables tiling BY
    # DEFAULT (vae_encoder_tiling/vae_decoder_tiling = 1) at vae_tile_size 256
    # PIXELS, and split_tiles only skips when the tile covers the whole input —
    # at 832x480 both axes exceed it. MEASURED: the untiled path OOMs, because
    # VIDEO_DECODE_S = latent_T*latent_H*latent_W + 5 = 10,925 ViT tokens at
    # 2048 wide does not fit 24 GiB across 32 heads. Tiling drops each call to
    # TOKENS_PER_CLIP*16*16 + 5 = 1,797.
    var tiling = minimax_h3_video_released_tiling_config()

    # ── STREAMING: decode -> denormalize -> write PNGs, one temporal part at a
    # time. The whole-video tensor is never built. Returns early; everything
    # below is the non-streaming path, unchanged.
    @parameter
    if VIDEO_DECODE_STREAM != 0:
        var stcfg = minimax_h3_video_released_temporal_config()
        var stream = MiniMaxH3TemporalDecodeStream(latents, stcfg, tiling, ctx)
        var planned = stream.output_frames()
        var norm = minimax_h3_pixel_norm_constants(String("imagenet"))
        var written = 0
        var last_h = 0
        var last_w = 0
        while stream.has_next():
            var part = stream.next_part[
                LATENT_TILE, LATENT_TILE, 32, 64, 5, VAE_TOKENS_PER_CLIP
            ](decoder, ctx)
            if part:
                var rgb_part = minimax_h3_video_pixel_denormalize(
                    part.value(), norm, ctx
                )
                var pshape = rgb_part.shape()
                last_h = pshape[2]
                last_w = pshape[3]
                written += _write_rgb_frames(rgb_part, out_dir, written, ctx)
        # Ports the vendor's closing invariant (:668-674) — if the frame
        # bookkeeping drifted anywhere in the loop this raises instead of
        # silently shipping a short or long video.
        stream.finish()
        if written != planned:
            raise Error(
                String("minimax_h3_t2va: streaming decode wrote ") + String(written)
                + " frames but planned " + String(planned)
            )
        print(
            "  wrote", written, "frames", last_h, "x", last_w, "to", out_dir,
            "(STREAMING temporal decode — whole-video tensor never materialized)",
        )
        return written

    var pixels: Tensor
    @parameter
    if VIDEO_DECODE_TEMPORAL != 0:
        # Vendor composition: temporal chunks OUTSIDE, spatial tiles INSIDE.
        # Per-clip ViT sequence is VAE_TOKENS_PER_CLIP*16*16 + 5 = 1797 tokens
        # REGARDLESS of total video length — that constant is what makes a
        # 14-second decode possible at all, where the direct path's
        # NUM_LATENT_FRAMES*16*16 + 5 grows without bound.
        var tcfg = minimax_h3_video_released_temporal_config()
        pixels = minimax_h3_video_decode_temporal[
            LATENT_TILE, LATENT_TILE, 32, 64, 5, VAE_TOKENS_PER_CLIP
        ](decoder, latents, tcfg, tiling, ctx)
    else:
        pixels = minimax_h3_video_tiled_decode[
            LATENT_TILE, LATENT_TILE, 32, 64, 5, NUM_LATENT_FRAMES
        ](decoder, latents, tiling, ctx)

    # ImageNet denormalize — the exact inverse of the pre-encode transform.
    # Neither original video unit had this; without it every frame is off by a
    # per-channel affine that reads as a colour bug and gets blamed on the DiT.
    var rgb = minimax_h3_video_pixel_denormalize(
        pixels, minimax_h3_pixel_norm_constants(String("imagenet")), ctx
    )

    # DELIBERATELY NOT refactored to call `_write_rgb_frames` (which is the
    # same loop with `first_index` added). Factoring it out is the obvious
    # cleanup and it was done, then REVERTED: it changed this binary — .text
    # +864 bytes, .rodata -16 (a deduplicated string) — and the six-shot batch
    # builds its remaining per-shot binaries FROM THIS SOURCE on demand, so the
    # default path has to stay byte-for-byte what shots 2-4 already ran. The
    # duplication buys a provable no-op for the in-flight batch and should be
    # collapsed the moment that batch is done.
    var ps = rgb.shape()
    var frames = ps[1]
    var height = ps[2]
    var width = ps[3]
    for f in range(frames):
        var one = slice(rgb, 1, f, 1, ctx)                       # [1,1,H,W,3]
        var hwc = reshape(one, [height, width, 3], ctx)
        var chw = permute(hwc, [2, 0, 1], ctx)                   # [3,H,W]
        var img = reshape(chw, [1, 3, height, width], ctx)
        var name = String(f)
        while len(name) < 5:
            name = String("0") + name
        save_png(img, out_dir + "/frame_" + name + ".png", ctx, ValueRange.UNIT)
    print("  wrote", frames, "frames", height, "x", width, "to", out_dir)
    return frames



# ═════════════════════════════════════════════════════════════════════════════
# Audio decode — WIRED. See file header "AUDIO DECODE: WIRED, HOST-SIDE".
# ═════════════════════════════════════════════════════════════════════════════

# The released BigVGAN decoder config, taken VERBATIM from the already-gated
# `minimax_h3_audio_real_weights_parity.mojo` (12/12 PASS at ~1e-6 against
# the real released weights) rather than re-derived — same values, same
# source of truth.
comptime AUDIO_LATENT_DIM = 2048
comptime AUDIO_DECODER_DIM = 1024


def _h3_audio_decoder_rates() -> List[Int]:
    return [5, 5, 2, 2, 2, 2, 2]


def _h3_audio_decoder_kernels() -> List[Int]:
    return [9, 9, 4, 4, 4, 4, 4]


def _h3_audio_resblock_kernels() -> List[Int]:
    return [3, 7, 11]


def _h3_audio_resblock_dilations() -> List[List[Int]]:
    var d: List[Int] = [1, 3, 5]
    return [d.copy(), d.copy(), d.copy()]


def _minimax_h3_audio_decoder_config(latent_channels: Int) -> MiniMaxH3AudioDecoderConfig:
    return MiniMaxH3AudioDecoderConfig(
        latent_channels,
        AUDIO_LATENT_DIM,
        AUDIO_DECODER_DIM,
        _h3_audio_decoder_rates(),
        _h3_audio_decoder_kernels(),
        _h3_audio_resblock_kernels(),
        _h3_audio_resblock_dilations(),
    )


# `latents_mean`/`latents_std` — hardcoded VERBATIM from the real, landed
# `FL2VA/audio_vae/config.json` (verified 2026-08-02; 32 values each, one
# per latent channel). See file header "AUDIO DECODE" for why this is
# hardcoded rather than parsed.
def _h3_audio_latents_mean() -> List[Float32]:
    return [
        Float32(-0.020211687488382354), Float32(0.3876466479950502),
        Float32(-0.04398279799186767), Float32(-0.28591514936373),
        Float32(0.08179686214561671), Float32(-0.35782641352446604),
        Float32(0.040623809960919084), Float32(-0.01552534501956604),
        Float32(-0.223362481667332), Float32(0.1821006842509091),
        Float32(0.2941778783780663), Float32(-0.07901167601970885),
        Float32(-0.056815072777201), Float32(-0.3699028221860095),
        Float32(-0.31616315591624855), Float32(0.5905951377425391),
        Float32(-0.052139568068853864), Float32(0.013673160263486295),
        Float32(-0.03691647864630577), Float32(0.09732660653298163),
        Float32(-0.3394662328788498), Float32(-0.30685677538541667),
        Float32(-0.24504598907458763), Float32(-0.034698524462007344),
        Float32(0.02868032184767538), Float32(-0.21217779266454084),
        Float32(-0.1678263169941987), Float32(0.3221287889040614),
        Float32(-0.1223055851554907), Float32(0.4356604928128464),
        Float32(-0.0502599202236253), Float32(0.3979258376211797),
    ]


def _h3_audio_latents_std() -> List[Float32]:
    return [
        Float32(1.6895524230479284), Float32(2.76263727217653),
        Float32(1.7945344281264435), Float32(1.6801681847309828),
        Float32(1.6390226546605453), Float32(2.7788298348882177),
        Float32(1.7659090095747236), Float32(1.6199757612137327),
        Float32(2.6336525640336896), Float32(1.8539356672817833),
        Float32(2.5056497896915633), Float32(1.811019237886178),
        Float32(1.9579657790720237), Float32(1.6685498243529284),
        Float32(1.4922469314453364), Float32(3.298670198067373),
        Float32(1.9491804496832168), Float32(1.8720003270431442),
        Float32(1.8334080103291832), Float32(1.6488070416529093),
        Float32(1.6176957696319716), Float32(1.9131449234774398),
        Float32(1.5695245398428617), Float32(1.6943659940415912),
        Float32(1.8318420762504692), Float32(1.5540637421583379),
        Float32(1.9344930328968526), Float32(1.599198216109855),
        Float32(1.718045989838149), Float32(1.6307219190837705),
        Float32(1.8661226051202384), Float32(1.5613768203168363),
    ]


def _minimax_h3_preflight_audio_vae(path: String) raises:
    """Index-only check (mmap header lookups, no tensor bytes read, no
    DeviceContext) — the audio decoder's own three structurally-load-bearing
    tensors: `dec_in_proj` (the latent-side entry point unit 10's own
    docstring calls out as "the only place a latent-side mistake can
    enter"), and `decoder.conv_pre`/`decoder.conv_post`'s weight_norm pair
    (every conv in this decoder is weight-normed except dec_in_proj — a
    missing `weight_g`/`weight_v` pair means `fold_weight_norm` fails deep
    inside the first stage instead of here)."""
    var st = SafeTensors.open(path)
    var required = [
        String("dec_in_proj.weight"), String("dec_in_proj.bias"),
        String("decoder.conv_pre.weight_g"), String("decoder.conv_pre.weight_v"),
        String("decoder.conv_pre.bias"),
        String("decoder.conv_post.weight_g"), String("decoder.conv_post.weight_v"),
    ]
    for i in range(len(required)):
        if not st.has_tensor(required[i]):
            raise Error(
                String("minimax_h3_t2va preflight: audio_vae missing tensor ")
                + required[i]
            )


def _minimax_h3_load_audio_vae_weights(path: String) raises -> MiniMaxH3AudioWeights:
    """Mirrors `minimax_h3_audio_real_weights_parity.mojo`'s own loading loop
    (same exclusion of `zero_k_bias`, a frozen zero buffer the decoder
    deliberately never reads) rather than a shared generic loader — this
    is a single small mmap-and-copy loop, not fragile numeric math, and
    every other H3 parity/pipeline file in this repo writes its own small
    version of it too."""
    var st = SafeTensors.open(path)
    var all_names = st.names()
    var names = List[String]()
    var values = List[List[Float32]]()
    for i in range(len(all_names)):
        ref n = all_names[i]
        if n.find("zero_k_bias") >= 0:
            continue
        var info = st.tensor_info(n)
        var bytes = st.tensor_bytes(n)
        var tv = from_parts(info.dtype, info.shape.copy(), bytes)
        if tv.dtype != STDtype.F32:
            raise Error(
                String("minimax_h3_t2va: audio_vae tensor ") + n + " is not F32"
            )
        var p = tv.data.unsafe_ptr().bitcast[Float32]()
        var v = List[Float32](capacity=tv.numel())
        for j in range(tv.numel()):
            v.append(p[j])
        names.append(String(n))
        values.append(v^)
    return MiniMaxH3AudioWeights(names^, values^)


def _minimax_h3_denormalize_audio_channel(
    latent: List[Float32], channels: Int, num_latents: Int,
    mean: List[Float32], std: List[Float32],
) raises -> List[Float32]:
    """`latent` is one stereo item's `[C, T]` channel-major block (as
    `minimax_h3_unpack_audio` produces): `latent = normalized*std + mean`,
    per channel, broadcast over T. See file header "DENORMALIZATION"."""
    if len(latent) != channels * num_latents:
        raise Error("minimax_h3_t2va: audio latent channel/length mismatch")
    var out = List[Float32](capacity=len(latent))
    for c in range(channels):
        var m = mean[c]
        var s = std[c]
        for t in range(num_latents):
            out.append(latent[c * num_latents + t] * s + m)
    return out^


def _minimax_h3_decode_audio(
    audio_state: Tensor,  # [NUM_AUDIO_ROWS, audio_latents_dim] F32, row space
    num_audio_latents: Int,
    audio_channels: Int,  # config.audio_latents_dim (32)
    out_dir: String,
    ctx: DeviceContext,
) raises -> Int:
    """Unpack -> denormalize -> host BigVGAN decode x2 (stereo, independent
    per channel per the vendor's own convention) -> interleave -> WAV.
    Returns the waveform length in samples (for the result JSON)."""
    var rows_host = audio_state.to_host(ctx)  # [Na*C] row-major: row*C+c
    var latents_2ct = minimax_h3_unpack_audio(rows_host, num_audio_latents, audio_channels)

    var block = audio_channels * num_audio_latents
    var ch0_normalized = List[Float32](capacity=block)
    var ch1_normalized = List[Float32](capacity=block)
    for i in range(block):
        ch0_normalized.append(latents_2ct[i])
        ch1_normalized.append(latents_2ct[block + i])

    var mean = _h3_audio_latents_mean()
    var std = _h3_audio_latents_std()
    if len(mean) != audio_channels or len(std) != audio_channels:
        raise Error(
            "minimax_h3_t2va: hardcoded latents_mean/latents_std length !="
            " config.audio_latents_dim — config.json drifted from what this"
            " file hardcoded; re-read FL2VA/audio_vae/config.json"
        )
    var ch0 = _minimax_h3_denormalize_audio_channel(ch0_normalized, audio_channels, num_audio_latents, mean, std)
    var ch1 = _minimax_h3_denormalize_audio_channel(ch1_normalized, audio_channels, num_audio_latents, mean, std)

    var dec_cfg = _minimax_h3_audio_decoder_config(audio_channels)
    var weights = _minimax_h3_load_audio_vae_weights(String(AUDIO_VAE_PATH))

    # Device BigVGAN (gate: models/minimax_h3_device/parity/, 11/11 vs host
    # oracle, e2e waveform cos 0.999999999994677 / max_abs 5.7e-6).
    var dev_weights = minimax_h3_audio_device_weights(weights, dec_cfg, ctx)
    var wave_l = minimax_h3_audio_decode_device(dev_weights, dec_cfg, ch0, num_audio_latents, ctx)
    var wave_r = minimax_h3_audio_decode_device(dev_weights, dec_cfg, ch1, num_audio_latents, ctx)
    if len(wave_l) != len(wave_r):
        raise Error("minimax_h3_t2va: L/R waveform length mismatch")

    var stereo_host = List[Float32](capacity=2 * len(wave_l))
    for i in range(len(wave_l)):
        stereo_host.append(wave_l[i])
    for i in range(len(wave_r)):
        stereo_host.append(wave_r[i])
    var stereo_shape: List[Int] = [2, len(wave_l)]
    var stereo_tensor = Tensor.from_host(stereo_host, stereo_shape^, STDtype.F32, ctx)

    var wav_path = out_dir + String("/audio.wav")
    save_wav(stereo_tensor, wav_path, AUDIO_SAMPLE_RATE, ctx)
    print("  wrote", wav_path, " (", len(wave_l), "samples/channel,",
          Float64(len(wave_l)) / Float64(AUDIO_SAMPLE_RATE), "s )")
    return len(wave_l)


# ═════════════════════════════════════════════════════════════════════════════
# PARTIAL MODE conditioning stand-in. See file header "PARTIAL MODE".
# ═════════════════════════════════════════════════════════════════════════════
def _minimax_h3_stub_conditioning(
    seed: UInt64, config: MiniMaxH3DiTConfig, ctx: DeviceContext
) raises -> MiniMaxH3ConditioningOutput:
    """Fixed-seed random `[1, TEXT_TOKENS, text_dim]` BF16 standing in for a
    real Qwen3-VL-32B encode — the text_encoder's layer-50 weights live in
    shard 11, not downloaded. Every row tagged TEXT (matches the real path's
    own convention: t2va has no keyframes, every text row is `packing.
    MINIMAX_H3_TEXT_TAG = 1`; models/text_encoder/minimax_h3_conditioning.
    mojo does the same unconditionally for every token). BF16 to match the
    real path's own dtype (that file's checkpoint is native bf16, no fp8;
    `condition_proj`, which consumes this, is bf16-native too — not one of
    the 12 fp32 dtype-trap tensors). Returns the SAME struct the real path
    returns (`[1,seq,5120]` embeds + token_tags) so `main()` treats both
    branches identically from here on — no conditional-var-init, no
    Tuple-move pitfalls."""
    var token_tags = List[Int](capacity=TEXT_TOKENS)
    for _ in range(TEXT_TOKENS):
        token_tags.append(1)
    var shape: List[Int] = [1, TEXT_TOKENS, config.text_dim]
    var embeds = randn(shape^, seed, STDtype.BF16, ctx)
    return MiniMaxH3ConditioningOutput(embeds^, token_tags^)


def _minimax_h3_get_conditioning(
    partial_mode: Bool, prompt: String, seed: UInt64,
    config: MiniMaxH3DiTConfig, ctx: DeviceContext,
) raises -> MiniMaxH3ConditioningOutput:
    """Single call site for `main()`, mirroring `_minimax_h3_open_
    transformer_shards`'s reason for existing: one `var cond` regardless of
    branch."""
    if partial_mode:
        return _minimax_h3_stub_conditioning(seed, config, ctx)
    return minimax_h3_encode_conditioning(
        String(PROCESSOR_DIR), String(TEXT_ENCODER_DIR), prompt, ctx
    )


# ═════════════════════════════════════════════════════════════════════════════
# main
# ═════════════════════════════════════════════════════════════════════════════
def main() raises:
    var args = argv()
    if len(args) < 3:
        print("usage: minimax_h3_t2va <prompt> <out_dir> [steps=30] [seed=0] [max_blocks=50]")
        print(
            "  compiled geometry:", WIDTH, "x", HEIGHT, ",", FRAMES, "frames,",
            "text_tokens=", TEXT_TOKENS, ", S=", SEQ_LEN,
        )
        print("  max_blocks < 50 is an explicit PARTIAL-MODE plumbing test — see file header")
        return

    var prompt = String(args[1])
    var out_dir = String(args[2])
    var steps = DEFAULT_STEPS
    if len(args) >= 4:
        steps = atol(String(args[3]))
    var seed = UInt64(DEFAULT_SEED)
    if len(args) >= 5:
        seed = UInt64(atol(String(args[4])))
    var max_blocks = 50
    if len(args) >= 6:
        max_blocks = atol(String(args[5]))

    # ── DECODE-ONLY MODE: argv[6] == "decode_only" reuses a prior run's saved
    # latents (out_dir/latents.safetensors) and runs ONLY the decode tail —
    # a decode-layer fix costs ~2 minutes instead of a 30-minute denoise.
    if len(args) >= 7 and String(args[6]) == "decode_only":
        var ctx2 = DeviceContext()
        print("  DECODE-ONLY: loading", out_dir + "/latents.safetensors")
        var lat_st = SafeTensors.open(out_dir + "/latents.safetensors")
        var vinfo = lat_st.tensor_info("video_state_rows")
        var video_rows = Tensor.from_view(
            from_parts(vinfo.dtype, vinfo.shape.copy(), lat_st.tensor_bytes("video_state_rows")), ctx2
        )
        var ainfo = lat_st.tensor_info("audio_state_rows")
        var audio_rows = Tensor.from_view(
            from_parts(ainfo.dtype, ainfo.shape.copy(), lat_st.tensor_bytes("audio_state_rows")), ctx2
        )
        print("  video rows", video_rows.shape(), " audio rows", audio_rows.shape())
        var config2 = minimax_h3_released_config()
        _ = _minimax_h3_decode_audio(
            audio_rows, NUM_AUDIO_LATENTS, config2.audio_latents_dim, out_dir, ctx2
        )
        var nframes = _minimax_h3_decode_video(
            video_rows, NUM_LATENT_FRAMES, LATENT_H, LATENT_W,
            config2.latents_dim, String(VIDEO_VAE_DIR), out_dir, ctx2,
        )
        print("  DECODE-ONLY done:", nframes, "frames +", out_dir + "/audio.wav")
        return

    if steps < 2:
        raise Error("minimax_h3_t2va: steps must be >= 2 (the schedule needs >= 2 sigmas)")
    if max_blocks < 1:
        raise Error("minimax_h3_t2va: max_blocks must be >= 1")

    print("=== MiniMax-H3 t2va ===")
    print("  prompt:", prompt)
    print(
        "  geometry:", WIDTH, "x", HEIGHT, ",", FRAMES, "frames -> latent [",
        NUM_LATENT_FRAMES, ",", LATENT_H, ",", LATENT_W, "], audio_latents=",
        NUM_AUDIO_LATENTS, ", text_tokens=", TEXT_TOKENS, ", S=", SEQ_LEN,
    )
    print("  steps=", steps, " seed=", seed, " max_blocks=", max_blocks)

    # ── PREFLIGHT (before DeviceContext) ──────────────────────────────────
    var t_preflight0 = perf_counter_ns()
    _preflight_geometry()
    var config = minimax_h3_released_config()
    config.validate()
    if max_blocks > config.num_layers:
        max_blocks = config.num_layers
    var partial_mode = max_blocks < config.num_layers
    var run_config = config  # implicit copy (Copyable/ImplicitlyCopyable)
    run_config.num_layers = max_blocks

    if partial_mode:
        print("")
        print("  ################################################################")
        print("  # PARTIAL MODE:", max_blocks, "of", config.num_layers, "transformer blocks.")
        print("  # Conditioning is STUBBED (fixed-seed random, NOT the real prompt).")
        print("  # THIS IS NOT A VALID MINIMAX-H3 GENERATION. It is a plumbing test:")
        print("  # tokenize/stub -> conditioning -> frontend -> packed sequence ->")
        print("  # modcache -> N real streamed blocks -> Euler -> final layer ->")
        print("  # audio latents -> denormalize -> BigVGAN -> audio.wav.")
        print("  ################################################################")
        print("")

    if partial_mode:
        print("  preflight: opening transformer shards (PARTIAL bypass):", String(TRANSFORMER_DIR))
    else:
        print("  preflight: opening transformer shards:", String(TRANSFORMER_DIR))
    var transformer_shards = _minimax_h3_open_transformer_shards(String(TRANSFORMER_DIR), partial_mode)
    print(
        "  preflight: ", transformer_shards.num_shards(), "shard(s), ",
        transformer_shards.num_tensors(), "tensors",
    )
    minimax_h3_check_modcache_weights(transformer_shards, run_config)
    _preflight_block_tensors(transformer_shards, run_config)
    _preflight_frontend_tensors(transformer_shards, config)
    if partial_mode:
        print("  preflight: transformer OK (adaLN + ", max_blocks, "of", config.num_layers, "blocks + frontend) [PARTIAL]")
    else:
        print("  preflight: transformer OK (adaLN + ", config.num_layers, "blocks + frontend)")

    if not partial_mode:
        print("  preflight: opening text_encoder shards:", String(TEXT_ENCODER_DIR))
        var text_encoder_shards = ShardedSafeTensors.open(String(TEXT_ENCODER_DIR))
        print(
            "  preflight: ", text_encoder_shards.num_shards(), "shard(s), ",
            text_encoder_shards.num_tensors(), "tensors",
        )
    else:
        print("  preflight: text_encoder SKIPPED (PARTIAL MODE — conditioning is stubbed)")

    print("  preflight: opening audio_vae:", String(AUDIO_VAE_PATH))
    _minimax_h3_preflight_audio_vae(String(AUDIO_VAE_PATH))
    print("  preflight: audio_vae OK")

    var t_preflight1 = perf_counter_ns()
    print("  preflight OK (", Float64(t_preflight1 - t_preflight0) / 1.0e6, "ms)")

    _ = sys_system(String("mkdir -p '") + out_dir + "'")

    var ctx = DeviceContext()

    # ── 1. Conditioning (real, or STUBBED in partial mode — see file header) ──
    var t_cond0 = perf_counter_ns()
    var cond = _minimax_h3_get_conditioning(partial_mode, prompt, seed, config, ctx)
    if len(cond.token_tags) != TEXT_TOKENS:
        raise Error(
            String("minimax_h3_t2va: prompt tokenized to ")
            + String(len(cond.token_tags)) + " tokens, but this binary is"
            " compiled for H3_TEXT_TOKENS=" + String(TEXT_TOKENS)
            + " (S is comptime — see file header FIXED PROMPT LENGTH); to run"
            " this prompt, edit H3_TEXT_TOKENS and rebuild"
        )
    var text_rows = reshape(cond.embeds, [TEXT_TOKENS, config.text_dim], ctx)
    var t_cond1 = perf_counter_ns()
    print(
        "  conditioning", "[STUBBED]" if partial_mode else "[real]", ": ",
        TEXT_TOKENS, " tokens (", Float64(t_cond1 - t_cond0) / 1.0e9, "s)",
    )

    # ── 2. Packed-sequence geometry (host scalar, this port's own reproduction) ──
    var no_anchors = List[Int]()
    var geometry = minimax_h3_build_sampling_geometry(
        cond.token_tags, NUM_LATENT_FRAMES, LATENT_H, LATENT_W,
        NUM_AUDIO_LATENTS, PATCH_H, PATCH_W, no_anchors,
    )
    if geometry.sequence_length != SEQ_LEN:
        raise Error(
            String("minimax_h3_t2va: geometry.sequence_length ")
            + String(geometry.sequence_length) + " != compiled S "
            + String(SEQ_LEN) + " — geometry drifted from the comptime"
            " constants; this indicates a bug in this file's derivation,"
            " not a normal rebuild-hint case"
        )
    if len(geometry.video_indices) != NUM_VIDEO_ROWS:
        raise Error("minimax_h3_t2va: video row count mismatch")
    if len(geometry.audio_indices) != NUM_AUDIO_ROWS:
        raise Error("minimax_h3_t2va: audio row count mismatch")

    # ── 3. MM-RoPE tables (device) ─────────────────────────────────────────
    var positions_f32 = List[Float32](capacity=len(geometry.position_ids))
    for i in range(len(geometry.position_ids)):
        positions_f32.append(Float32(geometry.position_ids[i]))
    var positions_shape: List[Int] = [SEQ_LEN * 3]
    var positions_tensor = Tensor.from_host(positions_f32, positions_shape^, STDtype.F32, ctx)
    # `rope` is kept alive and its cos/sin elements (rope[0]/rope[1]) are
    # referenced directly at each use site below rather than extracted into
    # owning `var`s — Tensor is Movable-not-Copyable, so `var cos = rope[0]`
    # would try to copy out of the Tuple; borrowed indexed access (repeated
    # `rope[0]`/`rope[1]` reads) works fine and is what every call site needs.
    var rope = build_minimax_h3_rope_tables(positions_tensor, ctx, config.rope_inv_freq_len)
    var rotary_dim = rope[0].shape()[1]
    print("  geometry + rope ready: S=", geometry.sequence_length, " rotary_dim=", rotary_dim)

    # ── 4. Dual schedule (video shift 12.0 / audio shift 3.0) ─────────────
    var schedule = MiniMaxH3DualSchedule()
    schedule.set_timesteps(steps)
    var num_steps = schedule.num_inference_steps()
    print("  schedule: ", num_steps, " model evaluations")

    # ── 5. AdaLN modulation cache — ONE streamed pass, built ONCE ──────────
    var t_mod0 = perf_counter_ns()
    var distinct_timesteps = 2 * num_steps
    var temb_timesteps = List[Float32](capacity=distinct_timesteps)
    for i in range(num_steps):
        temb_timesteps.append(schedule.video_timestep(i))
        temb_timesteps.append(schedule.audio_timestep(i))
    var temb_shape: List[Int] = [distinct_timesteps]
    var temb_timesteps_tensor = Tensor.from_host(temb_timesteps, temb_shape^, STDtype.F32, ctx)

    var frontend_w = _minimax_h3_load_frontend_weights(transformer_shards, config, ctx)
    var temb = minimax_h3_timestep_embedding(temb_timesteps_tensor, frontend_w, config, ctx)
    var modcache = minimax_h3_build_modulation_cache(transformer_shards, temb, run_config, ctx)
    ctx.synchronize()
    var t_mod1 = perf_counter_ns()
    print(
        "  modcache: ", distinct_timesteps, " rows, ",
        Float64(modcache.total_bytes()) / (1024.0 * 1024.0), " MiB (",
        Float64(t_mod1 - t_mod0) / 1.0e9, "s)",
    )

    # ── 5b. OPTIONAL fp8-resident base (H3_FP8_RESIDENT=1 builds only) ─────
    # One more streamed pass over the 50 blocks, quantizing each to E4M3 +
    # per-row scales on device; every denoise step below then dequants from
    # residency instead of re-reading ~36 GiB from disk. Declared
    # unconditionally (an empty Optional costs nothing); populated only under
    # the comptime flag, so a default build's denoise loop is untouched.
    var fp8_resident = Optional[MiniMaxH3ResidentFp8](None)

    @parameter
    if DIT_FP8_RESIDENT != 0:
        var t_q0 = perf_counter_ns()
        fp8_resident = Optional[MiniMaxH3ResidentFp8](
            minimax_h3_build_resident_fp8(
                transformer_shards, config, ctx, run_config.num_layers
            )
        )
        var t_q1 = perf_counter_ns()
        print(
            "  fp8-resident: ", run_config.num_layers, " blocks, ",
            Float64(fp8_resident.value().resident_bytes())
            / (1024.0 * 1024.0 * 1024.0),
            " GiB resident (", Float64(t_q1 - t_q0) / 1.0e9, "s one-time)",
        )

    # ── 6. Denoise loop — real streamed blocks, real Euler steps ───────────
    var t_denoise0 = perf_counter_ns()
    var video_shape: List[Int] = [NUM_VIDEO_ROWS, config.video_patch_dim()]
    var video_state = randn(video_shape^, seed, STDtype.F32, ctx)
    var audio_shape: List[Int] = [NUM_AUDIO_ROWS, config.audio_latents_dim]
    var audio_state = randn(audio_shape^, seed + 1, STDtype.F32, ctx)

    for i in range(num_steps):
        var t_step0 = perf_counter_ns()
        var video_ts = schedule.video_timestep(i)
        var audio_ts = schedule.audio_timestep(i)
        var global_row = _minimax_h3_global_timestep_row(geometry.token_tags, i)
        var block_adaln_indices = minimax_h3_adaln_rows(global_row, geometry.token_tags)

        # Frontend embed: video_ts as a placeholder scalar — this call's OWN
        # `temb` output is discarded (the precomputed modcache is the AdaLN
        # source of truth here, not a per-step recompute); only `.hidden`
        # (the bf16 packed sequence for the block stack) is used.
        var placeholder_ts_shape: List[Int] = [1]
        var placeholder_ts = Tensor.from_host([video_ts], placeholder_ts_shape^, STDtype.F32, ctx)
        # `embed.hidden` is read by BORROWED reference (via `reshape` below),
        # never moved out — MiniMaxH3FrontendEmbed has two Tensor fields
        # (hidden, temb) and Mojo does not allow destroying a struct whose
        # field was partially moved out from under it, so a borrowed read
        # (which leaves both fields intact) sidesteps that entirely. `.temb`
        # is never read (the precomputed modcache is the AdaLN source of
        # truth here, not a per-step recompute) and drops with `embed`
        # normally at the end of the step.
        var embed = minimax_h3_frontend_embed[TEXT_TOKENS, H3_HEADS, H3_HEAD_DIM](
            video_state, audio_state, text_rows, placeholder_ts,
            geometry.video_indices, geometry.audio_indices, geometry.text_indices,
            SEQ_LEN, frontend_w, config, ctx,
        )

        # RANK ADAPTER — a real seam bug, diagnosed 2026-08-02, between two
        # files this pipeline does not own (both models/dit/, neither
        # models/minimax_h3/): `minimax_h3_frontend.mojo::minimax_h3_
        # scatter_streams` builds `hidden` from `in_shape = [sequence_length,
        # hidden_size]` — RANK 2 — and `minimax_h3_frontend_embed`'s own
        # docstring calls that "ready for the stack". But `minimax_h3_dit.
        # mojo::minimax_h3_block_forward`'s docstring documents `x: [1, S,
        # hidden]` — RANK 3 — and its code is not merely documented that way:
        # `qkv_out = linear(attn_in, ...)` then `slice(qkv_out, 2, ...)`
        # hard-codes dim INDEX 2, which does not exist on a rank-2 tensor.
        # MEASURED: this is exactly where a real partial-block run raised
        # "slice: dim out of range", immediately after modcache, before any
        # per-block progress — i.e. inside the FIRST block_forward call, at
        # this exact slice. `minimax_h3_final_layer`'s own docstring, in
        # turn, wants `hidden: [S, hidden]` — RANK 2 — right back, matching
        # frontend_embed's convention, not block_forward's. So the two
        # non-block_forward functions agree with each other and disagree
        # with block_forward; the adapter belongs at THIS seam (the wiring
        # this file owns), not inside either upstream file — reshape in
        # immediately before the block loop, reshape back out immediately
        # after, and do not touch minimax_h3_frontend.mojo or minimax_h3_dit.
        # mojo to make this fit.
        var hidden3 = reshape(embed.hidden, [1, SEQ_LEN, config.hidden_size], ctx)

        for layer in range(run_config.num_layers):  # max_blocks (== 50 unless partial_mode)
            var block_w: Dict[String, ArcPointer[Tensor]]

            @parameter
            if DIT_FP8_RESIDENT != 0:
                # NO disk: dequant this block's E4M3 bytes back to bf16 on
                # device. Same Dict contract (shapes, dtype, transformed row
                # order, markers) — block_forward below is unchanged.
                block_w = minimax_h3_resident_block_weights(
                    fp8_resident.value(), layer, config, ctx
                )
            else:
                block_w = minimax_h3_load_block_device(transformer_shards, layer, config, ctx)
            hidden3 = minimax_h3_block_forward[SEQ_LEN, H3_HEADS, H3_HEAD_DIM](
                hidden3, block_w, layer, config, modcache.block_mod[layer][],
                block_adaln_indices, rope[0], rope[1], rotary_dim, ctx,
            )
            # `block_w` drops here: this layer's ~0.77 GiB bf16 is freed
            # before the next layer's load — resident footprint stays at one
            # block, never the full 61.73 GiB stack.

        var hidden2 = reshape(hidden3, [SEQ_LEN, config.hidden_size], ctx)
        var frontend_out = minimax_h3_final_layer(
            hidden2, modcache.final_mod[], global_row,
            geometry.video_indices, geometry.audio_indices,
            frontend_w, config, ctx,
        )

        video_state = schedule.step_video_device(frontend_out.video_out, video_ts, video_state, ctx)
        audio_state = schedule.step_audio_device(frontend_out.audio_out, audio_ts, audio_state, ctx)
        ctx.synchronize()
        var t_step1 = perf_counter_ns()
        print(
            "  phase=denoise step=", i + 1, " total=", num_steps,
            " video_t=", video_ts, " audio_t=", audio_ts,
            " (", Float64(t_step1 - t_step0) / 1.0e9, "s)",
        )

    # ── SAVE FINAL LATENTS. A decode-layer fix should cost a re-decode, not a
    # 30-minute denoise rerun (learned 2026-08-03: the row-order fix forced a
    # full rerun because the latent was gone). Row-space, pre-unpatchify.
    var lat_names = List[String]()
    lat_names.append(String("video_state_rows"))
    lat_names.append(String("audio_state_rows"))
    var lat_tensors = List[ArcPointer[Tensor]]()
    lat_tensors.append(ArcPointer[Tensor](slice(video_state, 0, 0, video_state.shape()[0], ctx)))
    lat_tensors.append(ArcPointer[Tensor](slice(audio_state, 0, 0, audio_state.shape()[0], ctx)))
    save_safetensors(lat_names, lat_tensors, out_dir + "/latents.safetensors", ctx)
    print("  saved final latents ->", out_dir + "/latents.safetensors")

    var t_denoise1 = perf_counter_ns()
    print("  denoise done (", Float64(t_denoise1 - t_denoise0) / 1.0e9, "s)")

    # ── 7. Audio decode (WIRED, real waveform) ─────────────────────────────
    var t_vae0 = perf_counter_ns()
    var audio_samples = _minimax_h3_decode_audio(
        audio_state, NUM_AUDIO_LATENTS, config.audio_latents_dim, out_dir, ctx
    )
    var t_vae1 = perf_counter_ns()
    print("  audio decode done (", Float64(t_vae1 - t_vae0) / 1.0e9, "s)")

    # ── 8. Result JSON — audio is a real artifact; video/mux are not ───────
    var result_body = String("{\n")
    result_body += String("  \"partial_mode\":") + json_bool(partial_mode) + String(",\n")
    result_body += String("  \"max_blocks\":") + String(max_blocks) + String(",\n")
    result_body += String("  \"total_blocks\":") + String(config.num_layers) + String(",\n")
    if partial_mode:
        result_body += String(
            "  \"WARNING\":\"THIS IS NOT A VALID MINIMAX-H3 GENERATION."
            " Partial block count + stubbed conditioning. Plumbing test only.\",\n"
        )
    result_body += String("  \"prompt\":\"") + json_escape(prompt) + String("\",\n")
    result_body += String("  \"conditioning\":\"") + (String("stubbed") if partial_mode else String("real")) + String("\",\n")
    result_body += String("  \"steps\":") + String(num_steps) + String(",\n")
    result_body += String("  \"seed\":") + String(seed) + String(",\n")
    result_body += String("  \"width\":") + String(WIDTH) + String(",\n")
    result_body += String("  \"height\":") + String(HEIGHT) + String(",\n")
    result_body += String("  \"frames\":") + String(FRAMES) + String(",\n")
    result_body += String("  \"sequence_length\":") + String(SEQ_LEN) + String(",\n")
    result_body += String("  \"audio_sample_rate\":") + String(AUDIO_SAMPLE_RATE) + String(",\n")
    result_body += String("  \"audio_samples_per_channel\":") + String(audio_samples) + String(",\n")
    result_body += String("  \"timings_ms\":{\n")
    result_body += String("    \"preflight\":") + String(Float64(t_preflight1 - t_preflight0) / 1.0e6) + String(",\n")
    result_body += String("    \"conditioning\":") + String(Float64(t_cond1 - t_cond0) / 1.0e6) + String(",\n")
    result_body += String("    \"modcache\":") + String(Float64(t_mod1 - t_mod0) / 1.0e6) + String(",\n")
    result_body += String("    \"denoise\":") + String(Float64(t_denoise1 - t_denoise0) / 1.0e6) + String(",\n")
    result_body += String("    \"audio_vae\":") + String(Float64(t_vae1 - t_vae0) / 1.0e6) + String("\n")
    result_body += String("  },\n")
    result_body += String("  \"artifacts\":{\"audio\":\"") + json_escape(out_dir + String("/audio.wav")) + String("\"},\n")
    result_body += String("  \"video_decode\":\"not_wired\",\n")
    result_body += String("  \"mux\":\"not_applicable_video_not_wired\"\n")
    result_body += String("}\n")
    write_text_file(out_dir + String("/result.json"), result_body)
    print("  wrote", out_dir + String("/result.json"))

    var t_vid0 = perf_counter_ns()
    var frames_written = _minimax_h3_decode_video(
        video_state, NUM_LATENT_FRAMES, LATENT_H, LATENT_W,
        config.latents_dim, String(VIDEO_VAE_DIR), out_dir, ctx,
    )
    var t_vid1 = perf_counter_ns()
    print("  video decode done (", Float64(t_vid1 - t_vid0) / 1.0e9, "s)")

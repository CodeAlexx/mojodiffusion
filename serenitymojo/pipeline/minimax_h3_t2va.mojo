# serenitymojo/pipeline/minimax_h3_t2va.mojo — MiniMax-H3 text-to-video+audio,
# pure Mojo. One product executable accepts the admitted resolution, frame
# count, FPS, and precision at runtime. Only the three attention sequence
# lengths remain AOT-specialized inside that executable. Preflight runs before
# `DeviceContext()`, and denoise/decode stay process-separated on a 24-GiB GPU.
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
# ── VARIABLE PROMPT LENGTH (exact rows, no DiT padding) ────────────────────────────
# The prompt is tokenized before GPU allocation and its exact row count is
# included in the runtime packed-sequence geometry. The language encoder may
# temporarily pad to a power-of-two causal-attention dispatch size, but slices
# those rows away before returning. The DiT therefore receives no synthetic
# text rows and needs no padding mask. This is the same runtime-length contract
# already used by the conditioned H3 pipelines.
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
#     tokenizes clean under H3's own processor). Product requests now resolve
#     this exact count at runtime; no prompt-specific rebuild is required.
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
from max.gpu.host import DeviceContext
from std.memory import alloc, ArcPointer
from std.ffi import external_call
from json.parser import loads as _json_loads

from serenitymojo.tensor import Tensor
from serenitymojo.ops.patchify3d import unpatchify3d
from serenitymojo.image.png import save_rgb24_video, ValueRange
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
from serenitymojo.io.ffi import sys_system, BytePtr
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.io.cap_cache import save_tensor_bin, load_tensor_bin
from serenitymojo.components.artifacts import shell_quote
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import (
    reshape, permute, slice, add, mul, mul_scalar, concat,
)
from serenitymojo.serve.product_manifest import json_escape, json_bool, write_text_file
from serenitymojo.audio.wav import save_wav

from serenitymojo.models.dit.minimax_h3_dit import (
    MiniMaxH3DiTConfig,
    MINIMAX_H3_ATTN_CUDNN,
    MINIMAX_H3_ATTN_SAGE_INT8,
    MINIMAX_H3_ATTN_SAGE_INT8_PV8,
    MINIMAX_H3_ATTN_SAGE_INT8_FAST,
    MINIMAX_H3_ATTN_COMFY_KITCHEN_INT8,
    MINIMAX_H3_SAGE_PV8_MIN_S,
    MINIMAX_H3_ATTN_SAGE_FAST_EXACT_PREFIX_BASE,
    MINIMAX_H3_ATTN_COMFY_KITCHEN_EXACT_PREFIX_BASE,
    MINIMAX_H3_ATTN_EVG_INT8,
    MINIMAX_H3_EVG_BUILD_ENABLED,
    minimax_h3_released_config,
    minimax_h3_adaln_rows,
    minimax_h3_block_tensor_names,
    minimax_h3_expected_shape,
    minimax_h3_block_forward,
    minimax_h3_block_forward_dynamic,
    minimax_h3_sage_exact_prefix_backend,
    minimax_h3_sage_fast_exact_prefix_backend,
    minimax_h3_comfy_kitchen_exact_prefix_backend,
)
from serenitymojo.ops.sage_attention_int8 import SageInt8Scratch
from serenitymojo.ops.comfy_kitchen_attention import (
    ComfyKitchenAttentionScratch,
)
from serenitymojo.ops.evg_attention_int8 import EVGH3RaggedLayout
from serenitymojo.models.minimax_h3.h3_lora_overlay import H3LoraOverlay
from serenitymojo.models.dit.minimax_h3_loader_device import (
    minimax_h3_load_block_device,
    minimax_h3_load_qkv_device,
    minimax_h3_load_fc1_device,
)
from serenitymojo.models.dit.minimax_h3_fp8_resident import (
    MINIMAX_H3_RESIDENT_INT8,
    MINIMAX_H3_RESIDENT_INT8_W8A8,
    MiniMaxH3ResidentFp8,
    minimax_h3_build_resident_fp8,
    minimax_h3_resident_block_weights,
    minimax_h3_resident_block_weights_groupwise,
    minimax_h3_resident_block_weights_w8a8,
)
from serenitymojo.models.dit.minimax_h3_modcache import (
    MiniMaxH3ModCache,
    minimax_h3_check_modcache_weights,
    minimax_h3_build_modulation_cache,
)
from serenitymojo.models.dit.minimax_h3_runtime_cache import (
    load_minimax_h3_modcache,
    load_minimax_h3_resident_cache,
    reload_minimax_h3_resident_w8a8_block,
    save_minimax_h3_modcache,
    save_minimax_h3_resident_cache,
)
from serenitymojo.models.dit.minimax_h3_step_cache import (
    MINIMAX_H3_CACHE_BACK_BLOCKS,
    MINIMAX_H3_CACHE_FRONT_BLOCKS,
    MiniMaxH3QuantizedActivation,
    MiniMaxH3StepCache,
    minimax_h3_cache_apply_residual_inplace,
    minimax_h3_cache_probe_rows,
    minimax_h3_cache_quantize_activation,
    minimax_h3_cache_should_reuse,
    minimax_h3_cache_probe_given_rows,
    minimax_h3_cache_store_middle_residual,
)
from serenitymojo.models.dit.minimax_h3_rope import build_minimax_h3_rope_tables
from serenitymojo.models.dit.minimax_h3_frontend import (
    MiniMaxH3FrontendEmbed,
    MiniMaxH3FrontendOutput,
    minimax_h3_frontend_embed,
    minimax_h3_frontend_embed_dynamic,
    minimax_h3_final_layer,
    minimax_h3_timestep_embedding,
)
from serenitymojo.models.text_encoder.minimax_h3_conditioning import (
    MINIMAX_H3_ENCODER_BF16,
    MINIMAX_H3_ENCODER_INT8,
    MiniMaxH3ConditioningOutput,
    minimax_h3_encode_conditioning,
    minimax_h3_tokenize_prompt,
)
from serenitymojo.models.dit.minimax_h3_sampling import (
    MiniMaxH3DualSchedule,
    MiniMaxH3SamplingGeometry,
    minimax_h3_build_sampling_geometry,
)
from serenitymojo.models.minimax_h3.motion_context import (
    MINIMAX_H3_MOTION_CONTEXT_FPS,
    minimax_h3_build_motion_context_geometry,
    minimax_h3_load_motion_context,
    minimax_h3_motion_context_audio_latents,
    minimax_h3_motion_context_steps,
    minimax_h3_preflight_motion_context,
    minimax_h3_save_motion_context_tail_at_pixel_frame,
)
from serenitymojo.models.minimax_h3.rearrange import minimax_h3_unpack_audio
from serenitymojo.models.minimax_h3.packing import MINIMAX_H3_TEXT_TAG
from serenitymojo.training.on_device_global_norm import on_device_grad_stats
from serenitymojo.models.minimax_h3.audio_decoder import (
    MiniMaxH3AudioDecoderConfig,
    MiniMaxH3AudioWeights,
    minimax_h3_audio_decode,
)
from serenitymojo.models.minimax_h3_device.audio_decoder_device import (
    MiniMaxH3AudioDeviceWeights,
    minimax_h3_audio_device_weights,
    minimax_h3_audio_decode_device,
)


# ── Checkpoint paths (matches every other H3 device module's own defaults) ──
comptime H3_ROOT = "/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA"
comptime TRANSFORMER_DIR = H3_ROOT + "/transformer"
comptime TRANSFORMER_INDEX = TRANSFORMER_DIR + "/model.safetensors.index.json"
comptime TEXT_ENCODER_DIR = H3_ROOT + "/text_encoder"
comptime PROCESSOR_DIR = H3_ROOT + "/processor"
comptime AUDIO_VAE_PATH = H3_ROOT + "/audio_vae/model.safetensors"
comptime VIDEO_VAE_DIR  = H3_ROOT + "/video_vae/source"
comptime AUDIO_SAMPLE_RATE = 32000
comptime RUNTIME_CACHE_DIR = H3_ROOT + "/serenity_runtime_cache_v1"
# The unified 24-GiB quality configuration keeps 41 blocks resident, but
# loads that prefix from the canonical 48-block groupwise cache. Keeping one
# cache artifact avoids duplicate 17+ GiB sidecars while leaving nine BF16
# tail blocks for the quality-preserving compute path.
comptime GROUPWISE_RUNTIME_CACHE_BLOCKS = 48
comptime PRODUCT_CONDITIONING_CACHE = (
    RUNTIME_CACHE_DIR
    + "/conditioning_ff21f1ebd1c73098_int8_bf16_output.bin"
)

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
def _h3_parse_f32(v: String) -> Float32:
    from std.memory import alloc
    var n = v.byte_length()
    var buf = alloc[UInt8](n + 1)
    var src = v.as_bytes()
    for i in range(n):
        buf[i] = src[i]
    buf[n] = 0
    var out = external_call["atof", Float64](
        BytePtr(unsafe_from_address=Int(buf))
    )
    buf.free()
    return Float32(out)


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


# ── Legacy direct-CLI defaults. Product requests override video geometry,
# and the real prompt overrides TEXT_TOKENS with its exact runtime token count.
comptime HEIGHT = get_defined_int["H3_HEIGHT", 480]()
comptime WIDTH = get_defined_int["H3_WIDTH", 832]()
# FRAMES must already be of the form 17n+5 (video VAE chunking, packing.mojo
# minimax_h3_align_num_frames) — this file does not round up for you, it
# fails loudly instead (see _preflight_geometry).
comptime FRAMES = get_defined_int["H3_FRAMES", 22]()
comptime TEXT_TOKENS = get_defined_int["H3_TEXT_TOKENS", 32]()
comptime FPS = 24
comptime MINIMAX_H3_TRAINED_MAX_FRAMES = 362
comptime MINIMAX_H3_SINGLE_PASS_MAX_FRAMES = 4323
# Direct-CLI guard. The server uses a slightly lower 107k estimate so real
# prompt rows retain headroom beneath this measured maximum-shape envelope.
comptime MINIMAX_H3_SINGLE_PASS_MAX_SEQUENCE = 109303
# The product path may request eight W8A8 blocks through the largest directly
# gated exact-attention sequence (S=37,951). The runner knows the real prompt
# length and owns the final safety cap.
comptime MINIMAX_H3_FAST_RESIDENT_EXACT_SEQUENCE_LIMIT = 38000

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

# ── Resident-denoiser direct-CLI defaults. Product requests select BF16,
# group-wise INT8, or W8A8 INT8-fast at runtime in one executable.
# Build with `-D H3_FP8_RESIDENT=1` to quantize all 50 blocks' big GEMM
# weights ONCE (bf16 -> 1 byte/param + class-groupwise scales, on device,
# resident; scheme = the store's default, class-groupwise INT8 since
# 2026-08-04 — E4M3 missed the 0.999 gate bar, see the store's header)
# and dequant per block per step instead of re-streaming ~36 GiB of
# block weights from disk EVERY denoise step. The residency arithmetic is
# `models/minimax_h3/fp8_policy.mojo`'s (fits 24 GiB with spare, adaLN already
# evicted to the modcache); the runtime is `models/dit/minimax_h3_fp8_resident
# .mojo` (krea2 fp8-resident precedent, same gated encode/dequant kernels).
# Gated vs the bf16-streamed path by models/dit/parity/
# minimax_h3_fp8_resident_gate.mojo. Default OFF: a normal build takes the
# `@parameter else` branch below, which is the unmodified streaming loop.
comptime DIT_FP8_RESIDENT = get_defined_int["H3_FP8_RESIDENT", 0]()
# A full 50-block store leaves insufficient activation headroom for long
# sequences (S=9145 at 512x320x175). Keep the switch independent from the
# on/off flag so a build may hold the first N blocks resident and stream the
# tail in BF16. Default remains 50 for backwards compatibility and parity
# gates; production long-video builds set this from measured VRAM headroom.
comptime DIT_RESIDENT_BLOCKS = get_defined_int["H3_RESIDENT_BLOCKS", 50]()


# ═════════════════════════════════════════════════════════════════════════════
# PREFLIGHT — everything below runs BEFORE `DeviceContext()`. Missing shards,
# wrong dtype, bad geometry all fail loudly here with named errors, per the
# runtime-port rule (already the convention `MiniMaxH3DiTConfig.validate` and
# `minimax_h3_check_modcache_weights` follow — this file adds the ONE more
# check those two don't cover: the 50 main-stack blocks' tensors, checked by
# SHAPE/DTYPE against the shard INDEX only, no Tensor materialized, no ctx
# needed, mirroring modcache.mojo's own `_check_adaln_tensor` style).
# ═════════════════════════════════════════════════════════════════════════════
def _preflight_geometry(
    width: Int,
    height: Int,
    frames: Int,
    text_tokens: Int,
    num_latent_frames: Int,
    num_audio_latents: Int,
) raises:
    if (frames - 5) % 17 != 0:
        raise Error(
            String("minimax_h3_t2va: frames=") + String(frames)
            + " is not of the form 17n+5 (the video VAE's chunk size);"
            " choose 5, 22, 39, 56, ..."
        )
    if height <= 0 or width <= 0 or height % 32 != 0 or width % 32 != 0:
        raise Error(
            String("minimax_h3_t2va: height/width must be positive")
            + " multiples of 32 (the canvas-rounding contract packing.mojo::"
            "minimax_h3_resolve_canvas_size enforces); got "
            + String(width) + "x" + String(height)
        )
    if text_tokens <= 0:
        raise Error("minimax_h3_t2va: text token count must be positive")
    if num_latent_frames <= 0 or num_audio_latents <= 0:
        raise Error(
            "minimax_h3_t2va: derived video/audio latent frame count is"
            " non-positive; choose a larger frame count"
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


def _minimax_h3_motion_context_global_timestep_row(
    geometry: MiniMaxH3SamplingGeometry, step: Int
) raises -> List[Int]:
    """Global 4-row modulation-cache addressing for latent continuation."""
    var out = List[Int](capacity=geometry.sequence_length)
    for _ in range(geometry.sequence_length):
        out.append(4 * step)
    for i in range(geometry.num_condition_video_rows):
        out[geometry.video_indices[i]] = 4 * step + 2
    for i in range(geometry.num_condition_audio_rows):
        out[geometry.audio_indices[i]] = 4 * step + 3
    for i in range(
        geometry.num_condition_audio_rows, len(geometry.audio_indices)
    ):
        out[geometry.audio_indices[i]] = 4 * step + 1
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
    """Append this part's frames to ONE raw RGB24 stream (`frames.rgb`),
    positioned by `first_index` — the STREAMING decode path calls this once
    per temporal part and never holds the whole video. Replaces the per-frame
    PNG writer: PNG spent ~370 ms/frame in CPU deflate (~65 s per 175-frame
    job, worse at 768p); the raw path is one GPU convert kernel + one PCIe
    crossing + one positional write per part, and NVENC muxes rawvideo
    directly."""
    var ps = rgb.shape()
    var frames = ps[1]
    var height = ps[2]
    var width = ps[3]
    var chw = permute(rgb, [0, 4, 1, 2, 3], ctx)                 # [1,3,F,H,W]
    save_rgb24_video(
        chw, out_dir + "/frames.rgb", ctx, ValueRange.UNIT, 0, 0,
        first_index * height * width * 3,
    )
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
    var perm: List[Int] = [1, 2, 3, 0]
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
    var vid_lat_mean: List[Float32] = [Float32(0.858090341091156), Float32(-0.9606591463088989), Float32(1.0661640167236328), Float32(-0.5090325474739075), Float32(-0.2727581858634949), Float32(-1.3675414323806763), Float32(-0.2553254961967468), Float32(-0.26907554268836975), Float32(-0.5376840829849243), Float32(-0.0464097298681736), Float32(0.6657370328903198), Float32(0.19690127670764923), Float32(-0.5460608005523682), Float32(-0.4035342037677765), Float32(-0.23683024942874908), Float32(0.25928452610969543), Float32(-0.30133944749832153), Float32(0.211341992020607), Float32(-1.1206848621368408), Float32(0.3581933379173279), Float32(-0.04225143790245056), Float32(0.2604829967021942), Float32(0.22864092886447906), Float32(0.7056031823158264)]
    var vid_lat_std: List[Float32] = [Float32(1.2223774194717407), Float32(1.2767263650894165), Float32(1.6831774711608887), Float32(1.7549455165863037), Float32(1.5636216402053833), Float32(2.194143533706665), Float32(0.9653137922286987), Float32(1.0569885969161987), Float32(0.841948926448822), Float32(0.7729952931404114), Float32(1.8955937623977661), Float32(0.946841835975647), Float32(0.7996809482574463), Float32(0.44988900423049927), Float32(0.7197399735450745), Float32(0.6936293244361877), Float32(2.961095094680786), Float32(2.7694199085235596), Float32(3.0496184825897217), Float32(2.1088054180145264), Float32(3.276226282119751), Float32(3.1627357006073), Float32(2.2816812992095947), Float32(2.6127843856811523)]
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
            # The released 256 px spatial tile becomes 16 latent cells only
            # on axes larger than 256 px.  At this proof's 320x192 geometry,
            # H is a single 192 px / 12-latent-cell tile while W is split into
            # 256 px / 16-cell tiles.  Keep the normal product specialization
            # unchanged and select the exact rectangular specialization only
            # for this admitted long-generation geometry.
            var part: Optional[Tensor]
            if latent_h == 12 and latent_w == 20:
                part = stream.next_part[
                    12, 16, 32, 64, 5, VAE_TOKENS_PER_CLIP
                ](decoder, ctx)
            elif latent_h == 14 and latent_w == 20:
                # 320x224 is the smallest landscape canvas which keeps the
                # Qwen vision preprocessor on its identity path.  H is one
                # 14-cell tile while W retains the released 16-cell tile.
                part = stream.next_part[
                    14, 16, 32, 64, 5, VAE_TOKENS_PER_CLIP
                ](decoder, ctx)
            else:
                part = stream.next_part[
                    LATENT_TILE, LATENT_TILE, 32, 64, 5,
                    VAE_TOKENS_PER_CLIP,
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

    # (The July six-shot byte-identical constraint that kept this loop
    # duplicated is over; both paths now share `_write_rgb_frames`.)
    var ps = rgb.shape()
    var frames = ps[1]
    var height = ps[2]
    var width = ps[3]
    _ = _write_rgb_frames(rgb, out_dir, 0, ctx)
    print("  wrote", frames, "frames", height, "x", width, "to", out_dir)
    return frames


def _minimax_h3_mux_av(
    out_dir: String,
    frames: Int,
    width: Int,
    height: Int,
    input_fps: Int,
    output_fps: Int,
    trim_start_frames: Int = 0,
) raises -> String:
    """Mux with NVENC, trimming a continuation overlap in picture and sound.

    Video input is the single raw RGB24 stream `_write_rgb_frames` produced
    (`frames.rgb`); the continuation trim is a byte-exact
    `-skip_initial_bytes` (frame size is constant), not a timestamp seek."""
    if trim_start_frames < 0:
        raise Error("MiniMax-H3 trim-start frames cannot be negative")
    var mp4 = out_dir + String("/video.mp4")
    var cmd = String("ffmpeg -v error -y -f rawvideo -pixel_format rgb24")
    cmd += String(" -video_size ") + String(width) + String("x") + String(height)
    cmd += String(" -framerate ") + String(input_fps)
    if trim_start_frames > 0:
        cmd += String(" -skip_initial_bytes ") + String(
            trim_start_frames * width * height * 3
        )
    cmd += String(" -i ") + shell_quote(out_dir + String("/frames.rgb"))
    if trim_start_frames > 0:
        var trim_seconds = (
            Float64(trim_start_frames) / Float64(input_fps)
        )
        cmd += String(" -ss ") + String(trim_seconds)
    cmd += String(" -i ") + shell_quote(out_dir + String("/audio.wav"))
    if output_fps != input_fps:
        cmd += String(" -vf fps=") + String(output_fps)
    cmd += String(" -frames:v ") + String(frames)
    cmd += String(" -c:v h264_nvenc -preset p7 -tune hq -rc vbr -cq 18")
    cmd += String(" -b:v 0 -pix_fmt yuv420p -af apad -c:a aac -shortest")
    cmd += String(" -movflags +faststart ") + shell_quote(mp4)
    if sys_system(cmd) != 0:
        raise Error("minimax_h3_t2va: GPU NVENC A/V mux failed")
    print("  muxed", mp4, "with h264_nvenc")
    return mp4^


def _minimax_h3_write_decode_result(
    out_dir: String,
    artifact: String,
    frames: Int,
    width: Int,
    height: Int,
    fps: Int,
) raises:
    var body = String("{\n")
    body += String("  \"schema\":\"serenity.minimax_h3.result.v1\",\n")
    body += String("  \"state\":\"done\",\n")
    body += String("  \"artifact_path\":\"") + json_escape(artifact) + String("\",\n")
    body += String("  \"width\":") + String(width) + String(",\n")
    body += String("  \"height\":") + String(height) + String(",\n")
    body += String("  \"frames\":") + String(frames) + String(",\n")
    body += String("  \"fps\":") + String(fps) + String(",\n")
    body += String("  \"audio\":true,\n")
    body += String("  \"video_encoder\":\"h264_nvenc\"\n")
    body += String("}\n")
    write_text_file(out_dir + String("/result.json"), body)



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


def _minimax_h3_decode_audio_channel_windowed(
    ref dev_weights: MiniMaxH3AudioDeviceWeights,
    config: MiniMaxH3AudioDecoderConfig,
    latents: List[Float32],
    num_latents: Int,
    channel_name: String,
    ctx: DeviceContext,
) raises -> List[Float32]:
    """Decode one continuous channel with exact-position halo windows.

    BigVGAN is fully convolutional, but a 180-second monolithic activation
    volume exceeds 24 GiB.  Each 30-second core is decoded with 64 latent
    samples (1.6 seconds) of real neighboring context on both sides; only the
    center samples belonging to that core are retained.  This is not a second
    generation or an audio stitch: every retained sample has its original
    global position and a halo wider than the decoder's convolutional
    receptive radius.  Global ends retain the model's native padding.
    """
    comptime CORE_LATENTS = 1200
    comptime HALO_LATENTS = 64

    if len(latents) != config.latent_channels * num_latents:
        raise Error("MiniMax-H3 windowed audio latent shape mismatch")

    var samples_per_latent = 1
    for i in range(len(config.decoder_rates)):
        samples_per_latent *= config.decoder_rates[i]
    var expected_samples = num_latents * samples_per_latent
    var out = List[Float32](capacity=expected_samples)
    var core_start = 0
    var window_index = 0
    while core_start < num_latents:
        var core_end = core_start + CORE_LATENTS
        if core_end > num_latents:
            core_end = num_latents
        var input_start = core_start - HALO_LATENTS
        if input_start < 0:
            input_start = 0
        var input_end = core_end + HALO_LATENTS
        if input_end > num_latents:
            input_end = num_latents
        var input_len = input_end - input_start

        var window = List[Float32](
            capacity=config.latent_channels * input_len
        )
        for c in range(config.latent_channels):
            for t in range(input_start, input_end):
                window.append(latents[c * num_latents + t])

        var decoded = minimax_h3_audio_decode_device(
            dev_weights, config, window, input_len, ctx
        )
        var keep_start = (core_start - input_start) * samples_per_latent
        var keep_len = (core_end - core_start) * samples_per_latent
        if keep_start < 0 or keep_start + keep_len > len(decoded):
            raise Error("MiniMax-H3 windowed audio crop exceeds decode output")
        for i in range(keep_len):
            out.append(decoded[keep_start + i])

        window_index += 1
        print(
            "  audio", channel_name, "window", window_index,
            "core_latents", core_start, "..", core_end,
            "input_latents", input_start, "..", input_end,
        )
        core_start = core_end

    if len(out) != expected_samples:
        raise Error(
            String("MiniMax-H3 windowed audio wrote ") + String(len(out))
            + " samples; expected " + String(expected_samples)
        )
    return out^


def _minimax_h3_decode_audio(
    audio_state: Tensor,  # [NUM_AUDIO_ROWS, audio_latents_dim] F32, row space
    num_audio_latents: Int,
    audio_channels: Int,  # config.audio_latents_dim (32)
    out_dir: String,
    ctx: DeviceContext,
) raises -> Int:
    """Unpack -> denormalize -> device BigVGAN decode x2 (stereo, independent
    per channel per the vendor's own convention) -> channel-major WAV.
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
    var wave_l = _minimax_h3_decode_audio_channel_windowed(
        dev_weights, dec_cfg, ch0, num_audio_latents, String("L"), ctx
    )
    var wave_r = _minimax_h3_decode_audio_channel_windowed(
        dev_weights, dec_cfg, ch1, num_audio_latents, String("R"), ctx
    )
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
    seed: UInt64,
    config: MiniMaxH3DiTConfig,
    text_tokens: Int,
    ctx: DeviceContext,
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
    var token_tags = List[Int](capacity=text_tokens)
    for _ in range(text_tokens):
        token_tags.append(1)
    var shape: List[Int] = [1, text_tokens, config.text_dim]
    var embeds = randn(shape^, seed, STDtype.BF16, ctx)
    return MiniMaxH3ConditioningOutput(embeds^, token_tags^)


def _minimax_h3_get_conditioning(
    partial_mode: Bool, prompt: String, seed: UInt64,
    config: MiniMaxH3DiTConfig, encoder_storage: Int, text_tokens: Int,
    ctx: DeviceContext,
) raises -> MiniMaxH3ConditioningOutput:
    """Single call site for `main()`, mirroring `_minimax_h3_open_
    transformer_shards`'s reason for existing: one `var cond` regardless of
    branch."""
    if partial_mode:
        return _minimax_h3_stub_conditioning(seed, config, text_tokens, ctx)
    return minimax_h3_encode_conditioning(
        String(PROCESSOR_DIR), String(TEXT_ENCODER_DIR), prompt, ctx,
        encoder_storage,
    )


def _minimax_h3_get_conditioning_cached(
    partial_mode: Bool,
    prompt: String,
    seed: UInt64,
    config: MiniMaxH3DiTConfig,
    encoder_storage: Int,
    cache_path: String,
    text_tokens: Int,
    ctx: DeviceContext,
) raises -> MiniMaxH3ConditioningOutput:
    """Load an explicitly request-keyed conditioning sidecar or encode once.

    The product server supplies a path whose filename includes the compiled
    prompt digest and encoder format.  Direct CLI callers get the original
    behavior unless they opt into a cache path themselves.
    """
    if not partial_mode and cache_path != String(""):
        try:
            var embeds = load_tensor_bin(cache_path, ctx)
            var shape = embeds.shape()
            if (
                embeds.dtype() != STDtype.BF16
                or len(shape) != 3
                or shape[0] != 1
                or shape[1] != text_tokens
                or shape[2] != config.text_dim
            ):
                raise Error("MiniMax-H3 conditioning cache shape/dtype mismatch")
            var tags = List[Int]()
            for _ in range(text_tokens):
                tags.append(MINIMAX_H3_TEXT_TAG)
            print("  conditioning cache: HIT", cache_path)
            return MiniMaxH3ConditioningOutput(embeds^, tags^)
        except e:
            print("  conditioning cache: MISS", String(e))

    var output = _minimax_h3_get_conditioning(
        partial_mode, prompt, seed, config, encoder_storage, text_tokens, ctx
    )
    if not partial_mode and cache_path != String(""):
        try:
            save_tensor_bin(output.embeds, cache_path, ctx)
            print("  conditioning cache: SAVED", cache_path)
        except e:
            # A cache write is an optimization failure, never permission to
            # discard the valid GPU conditioning already computed.
            print("  conditioning cache: SAVE FAILED", String(e))
    return output^


def _assert_finite_rows(
    name: String, tensor: Tensor, ctx: DeviceContext
) raises:
    """Scan an inference tensor on GPU and read back scalar summaries only."""
    var tensors = List[ArcPointer[Tensor]]()
    tensors.append(ArcPointer[Tensor](tensor.clone(ctx)))
    var stats = on_device_grad_stats(tensors, ctx)
    print(
        "  finite gate", name, "l2=", stats.grad_norm,
        "nonfinite=", stats.nonfinite_count,
    )
    if stats.nonfinite_count != 0:
        raise Error(
            String("minimax_h3_t2va: ") + name + String(" contains ")
            + String(stats.nonfinite_count) + String(" non-finite values")
        )


def _minimax_h3_get_modcache_cached(
    shards: ShardedSafeTensors,
    temb: Tensor,
    config: MiniMaxH3DiTConfig,
    steps: Int,
    distinct_timesteps: Int,
    cache_path: String,
    ctx: DeviceContext,
) raises -> MiniMaxH3ModCache:
    if cache_path != String(""):
        try:
            var cached = load_minimax_h3_modcache(
                cache_path,
                String(TRANSFORMER_INDEX),
                config,
                steps,
                distinct_timesteps,
                ctx,
            )
            print("  modcache: HIT", cache_path)
            return cached^
        except e:
            print("  modcache: MISS", String(e))
    var built = minimax_h3_build_modulation_cache(shards, temb, config, ctx)
    if cache_path != String(""):
        try:
            save_minimax_h3_modcache(
                built, String(TRANSFORMER_INDEX), cache_path, steps, ctx
            )
            print("  modcache: SAVED", cache_path)
        except e:
            print("  modcache: SAVE FAILED", String(e))
    return built^


def _minimax_h3_get_resident_cached(
    shards: ShardedSafeTensors,
    config: MiniMaxH3DiTConfig,
    resident_blocks: Int,
    resident_scheme: Int,
    cache_path: String,
    ctx: DeviceContext,
    cache_save_allowed: Bool = True,
    direct_groupwise: Bool = False,
) raises -> MiniMaxH3ResidentFp8:
    if cache_path != String(""):
        try:
            var cached = load_minimax_h3_resident_cache(
                cache_path,
                String(TRANSFORMER_INDEX),
                config,
                ctx,
                resident_blocks,
                0,
                resident_scheme,
                direct_groupwise,
            )
            print("  resident cache: HIT", cache_path)
            return cached^
        except e:
            print("  resident cache: MISS", String(e))
    var built = minimax_h3_build_resident_fp8(
        shards, config, ctx, resident_blocks, scheme=resident_scheme
    )
    if cache_path != String("") and cache_save_allowed:
        try:
            save_minimax_h3_resident_cache(
                built, String(TRANSFORMER_INDEX), cache_path, ctx
            )
            print("  resident cache: SAVED", cache_path)
        except e:
            print("  resident cache: SAVE FAILED", String(e))
    return built^


# One executable accepts every aligned product geometry and prompt length.
# Keep the accepted 241-row static refiner byte-identical; all other exact
# prompt lengths use the runtime refiner already shared with conditioned H3.
# Packed A/V/T sequence length remains runtime data through attention.
def _minimax_h3_model_eval_p[TEXT_S: Int](
    video_state: Tensor,
    audio_state: Tensor,
    text_rows: Tensor,
    placeholder_ts: Tensor,
    ref geometry: MiniMaxH3SamplingGeometry,
    ref frontend_w: Dict[String, ArcPointer[Tensor]],
    config: MiniMaxH3DiTConfig,
    run_config: MiniMaxH3DiTConfig,
    ref modcache: MiniMaxH3ModCache,
    global_row: List[Int],
    block_adaln_indices: List[Int],
    ref transformer_shards: ShardedSafeTensors,
    ref resident: Optional[MiniMaxH3ResidentFp8],
    mut reusable_w8a8_tail: Optional[MiniMaxH3ResidentFp8],
    use_resident: Bool,
    resident_scheme: Int,
    resident_cache_path: String,
    ref lora_overlay: Optional[H3LoraOverlay],
    cos: Tensor,
    sin: Tensor,
    rotary_dim: Int,
    attention_backend: Int,
    sage_scratch: Optional[SageInt8Scratch],
    comfy_kitchen_scratch: Optional[ComfyKitchenAttentionScratch],
    evg_layout: Optional[ArcPointer[EVGH3RaggedLayout]],
    step_index: Int,
    mut step_cache: MiniMaxH3StepCache,
    t2va_contiguous: Bool,
    ctx: DeviceContext,
) raises -> MiniMaxH3FrontendOutput:
    var sequence_length = geometry.sequence_length
    var embed: MiniMaxH3FrontendEmbed
    if text_rows.shape()[0] == TEXT_S:
        embed = minimax_h3_frontend_embed[TEXT_S, H3_HEADS, H3_HEAD_DIM](
            video_state,
            audio_state,
            text_rows,
            placeholder_ts,
            geometry.video_indices,
            geometry.audio_indices,
            geometry.text_indices,
            sequence_length,
            frontend_w,
            config,
            ctx,
            lora_overlay,
        )
    else:
        embed = minimax_h3_frontend_embed_dynamic[H3_HEADS, H3_HEAD_DIM](
            video_state,
            audio_state,
            text_rows,
            placeholder_ts,
            geometry.video_indices,
            geometry.audio_indices,
            geometry.text_indices,
            sequence_length,
            frontend_w,
            config,
            ctx,
            t2va_contiguous,
            lora_overlay,
        )
    var hidden3 = reshape(
        embed.hidden, [1, sequence_length, config.hidden_size], ctx
    )
    # At H3 Base's full-area long-video shapes, the frontend leaves several
    # GiB of completed projection/packing buffers pending behind the stream.
    # Fence once before block 0 so the allocator can reclaim those buffers;
    # the denoiser consumes `hidden3` only after all frontend work anyway.
    var low_headroom_bf16 = (
        not use_resident and sequence_length >= 60000
    )
    if low_headroom_bf16:
        ctx.synchronize()
    var cache_probe_before = List[Float32]()
    var cache_probe_audio_before = List[Float32]()
    var cache_audio_probe_ids = List[Int]()
    if step_cache.enabled:
        cache_probe_before = minimax_h3_cache_probe_rows(
            hidden3, sequence_length, config.hidden_size, ctx
        )
        # Audio veto: up to 16 evenly sampled audio rows held to a tighter
        # drift budget (audio degrades before video under approximation).
        var n_audio = len(geometry.audio_indices)
        if n_audio > 0:
            var want_rows = 16
            if n_audio < want_rows:
                want_rows = n_audio
            if want_rows == 1:
                cache_audio_probe_ids.append(geometry.audio_indices[0])
            else:
                for pi in range(want_rows):
                    cache_audio_probe_ids.append(
                        geometry.audio_indices[
                            (pi * (n_audio - 1)) // (want_rows - 1)
                        ]
                    )
            cache_probe_audio_before = minimax_h3_cache_probe_given_rows(
                hidden3, cache_audio_probe_ids, sequence_length,
                config.hidden_size, ctx,
            )
    var first_block_snapshot = Optional[MiniMaxH3QuantizedActivation](None)
    var reused_middle = False
    for layer in range(run_config.num_layers):
        if (
            step_cache.enabled
            and reused_middle
            and layer >= MINIMAX_H3_CACHE_FRONT_BLOCKS
            and layer
            < run_config.num_layers - MINIMAX_H3_CACHE_BACK_BLOCKS
        ):
            continue
        var block_w: Dict[String, ArcPointer[Tensor]]
        if use_resident:
            if layer < len(resident.value().blocks):
                if resident_scheme == MINIMAX_H3_RESIDENT_INT8_W8A8:
                    block_w = minimax_h3_resident_block_weights_w8a8(
                        resident.value(), layer, config, ctx
                    )
                elif resident_scheme == MINIMAX_H3_RESIDENT_INT8:
                    block_w = minimax_h3_resident_block_weights_groupwise(
                        resident.value(), layer, config, ctx
                    )
                else:
                    block_w = minimax_h3_resident_block_weights(
                        resident.value(), layer, config, ctx
                    )
            else:
                if (
                    resident_scheme == MINIMAX_H3_RESIDENT_INT8_W8A8
                    and resident_cache_path != String("")
                ):
                    reload_minimax_h3_resident_w8a8_block(
                        reusable_w8a8_tail.value(),
                        resident_cache_path,
                        String(TRANSFORMER_INDEX),
                        config,
                        layer,
                        ctx,
                    )
                    block_w = minimax_h3_resident_block_weights_w8a8(
                        reusable_w8a8_tail.value(), layer, config, ctx
                    )
                elif (
                    resident_scheme == MINIMAX_H3_RESIDENT_INT8
                    and resident_cache_path != String("")
                    and layer < GROUPWISE_RUNTIME_CACHE_BLOCKS
                    and len(resident.value().blocks) == 0
                ):
                    # Any zero-resident request needs the activation headroom
                    # of a streamed prefix. Load one already-quantized
                    # groupwise block at a time from the accepted 48-block
                    # cache; only the final two uncached blocks remain BF16.
                    var tail = load_minimax_h3_resident_cache(
                        resident_cache_path,
                        String(TRANSFORMER_INDEX),
                        config,
                        ctx,
                        1,
                        layer,
                        resident_scheme,
                        True,
                    )
                    block_w = minimax_h3_resident_block_weights_groupwise(
                        tail, layer, config, ctx
                    )
                else:
                    block_w = minimax_h3_load_block_device(
                        transformer_shards, layer, config, ctx
                    )
        else:
            block_w = minimax_h3_load_block_device(
                transformer_shards, layer, config, ctx
            )
        hidden3 = minimax_h3_block_forward_dynamic[
            H3_HEADS, H3_HEAD_DIM
        ](
            hidden3,
            block_w,
            layer,
            config,
            modcache.block_mod[layer][],
            block_adaln_indices,
            cos,
            sin,
            rotary_dim,
            ctx,
            attention_backend,
            sage_scratch,
            lora_overlay=lora_overlay,
            evg_layout=evg_layout,
            evg_step=step_index,
            comfy_kitchen_scratch=comfy_kitchen_scratch,
        )
        # The block's last-use temporaries are stream-ordered but large enough
        # at S>=60k that carrying them into the next streamed weight load can
        # exceed a 24 GiB card.  Synchronize only this BF16 low-headroom lane;
        # measured resident INT8 paths and ordinary BF16 shapes stay async.
        if low_headroom_bf16:
            ctx.synchronize()
        # A streamed cache block is owned by this per-layer Dict. Drop its
        # ArcPointers before loading the next block; otherwise long sequences
        # can overlap two weight blocks and lose several GiB of VRAM headroom.
        block_w.clear()
        if (
            step_cache.enabled
            and layer == MINIMAX_H3_CACHE_FRONT_BLOCKS - 1
        ):
            var cache_probe_after = minimax_h3_cache_probe_rows(
                hidden3, sequence_length, config.hidden_size, ctx
            )
            var cache_probe_audio_after = minimax_h3_cache_probe_given_rows(
                hidden3, cache_audio_probe_ids, sequence_length,
                config.hidden_size, ctx,
            )
            reused_middle = minimax_h3_cache_should_reuse(
                step_index, cache_probe_before, cache_probe_after, step_cache,
                cache_probe_audio_before, cache_probe_audio_after,
            )
            if reused_middle:
                minimax_h3_cache_apply_residual_inplace(
                    hidden3, step_cache, ctx
                )
                print(
                    "  cache-dit: step=", step_index + 1,
                    " action=reuse-middle residual_diff=",
                    step_cache.last_residual_diff,
                )
            elif step_cache.enabled:
                first_block_snapshot = Optional[MiniMaxH3QuantizedActivation](
                    minimax_h3_cache_quantize_activation(hidden3, ctx)
                )
                print(
                    "  cache-dit: step=", step_index + 1,
                    " action=full-refresh residual_diff=",
                    step_cache.last_residual_diff,
                )
        if (
            step_cache.enabled
            and not reused_middle
            and layer
            == run_config.num_layers - MINIMAX_H3_CACHE_BACK_BLOCKS - 1
        ):
            if not first_block_snapshot:
                raise Error("MiniMax-H3 Cache-DiT refresh has no front snapshot")
            minimax_h3_cache_store_middle_residual(
                hidden3, first_block_snapshot.value(), step_cache, ctx
            )
    var hidden2 = reshape(
        hidden3, [sequence_length, config.hidden_size], ctx
    )
    var frontend_out = minimax_h3_final_layer(
        hidden2,
        modcache.final_mod[],
        global_row,
        geometry.video_indices,
        geometry.audio_indices,
        frontend_w,
        config,
        ctx,
    )
    return frontend_out^


# ═════════════════════════════════════════════════════════════════════════════
# main
# ═════════════════════════════════════════════════════════════════════════════
def _job_main(raw_args: List[String]) raises:
    var args = List[String]()
    var runtime_width = WIDTH
    var runtime_height = HEIGHT
    var runtime_frames = FRAMES
    var output_frames = FRAMES
    var runtime_fps = FPS
    var output_fps = FPS
    var runtime_text_tokens = TEXT_TOKENS
    var use_resident = DIT_FP8_RESIDENT != 0
    var lora_path = String("")
    var lora_mult = Float32(1.0)
    var resident_blocks_requested = DIT_RESIDENT_BLOCKS
    var quant = String("int8") if use_resident else String("bf16")
    var attention_backend = MINIMAX_H3_ATTN_CUDNN
    var attention_backend_name = String("cudnn")
    var sage_exact_av_prefix = False
    var step_cache_enabled = False
    var step_cache_name = String("exact")
    var resident_scheme = MINIMAX_H3_RESIDENT_INT8
    var resident_backend_name = String("groupwise")
    var encoder_storage = MINIMAX_H3_ENCODER_BF16
    var encoder_storage_name = String("bf16")
    var defer_video_decode = False
    var runtime_cache = False
    var prepare_runtime_cache = False
    var validate_request = False
    var eval_start = 0
    var eval_stop = -1
    var motion_context_path = String("")
    var motion_context_frames = 22
    var trim_start_frames = 0
    var temporal_rope_scale = Float32(1.0)
    for i in range(len(raw_args)):
        var arg = String(raw_args[i])
        if arg.startswith("--width="):
            var fields = arg.split("=")
            if len(fields) != 2:
                raise Error("invalid --width flag")
            runtime_width = atol(String(fields[1]))
            continue
        if arg.startswith("--height="):
            var fields = arg.split("=")
            if len(fields) != 2:
                raise Error("invalid --height flag")
            runtime_height = atol(String(fields[1]))
            continue
        if arg.startswith("--frames="):
            var fields = arg.split("=")
            if len(fields) != 2:
                raise Error("invalid --frames flag")
            runtime_frames = atol(String(fields[1]))
            continue
        if arg.startswith("--output-frames="):
            var fields = arg.split("=")
            output_frames = atol(String(fields[1]))
            continue
        if arg.startswith("--fps="):
            var fields = arg.split("=")
            if len(fields) != 2:
                raise Error("invalid --fps flag")
            runtime_fps = atol(String(fields[1]))
            continue
        if arg.startswith("--output-fps="):
            var fields = arg.split("=")
            output_fps = atol(String(fields[1]))
            continue
        if arg.startswith("--resident-blocks="):
            var fields = arg.split("=")
            if len(fields) != 2:
                raise Error("invalid --resident-blocks flag")
            resident_blocks_requested = atol(String(fields[1]))
            continue
        if arg.startswith("--lora="):
            var fields = arg.split("=")
            if len(fields) != 2:
                raise Error("invalid --lora flag")
            lora_path = String(fields[1])
            continue
        if arg.startswith("--lora-mult="):
            var fields = arg.split("=")
            if len(fields) != 2:
                raise Error("invalid --lora-mult flag")
            lora_mult = _h3_parse_f32(String(fields[1]))
            continue
        if arg.startswith("--quant="):
            var fields = arg.split("=")
            if len(fields) != 2:
                raise Error("invalid --quant flag")
            quant = String(fields[1])
            continue
        if arg == String("--attention-backend=sage-int8"):
            attention_backend = MINIMAX_H3_ATTN_SAGE_INT8
            attention_backend_name = String("sage-int8")
            sage_exact_av_prefix = True
            continue
        if arg == String("--attention-backend=sage-int8-pv8"):
            attention_backend = MINIMAX_H3_ATTN_SAGE_INT8_PV8
            attention_backend_name = String("sage-int8-pv8")
            sage_exact_av_prefix = False
            continue
        if arg == String("--attention-backend=sage-int8-fast"):
            attention_backend = MINIMAX_H3_ATTN_SAGE_INT8_FAST
            attention_backend_name = String("sage-int8-fast")
            sage_exact_av_prefix = True
            continue
        if arg == String("--attention-backend=ck-int8"):
            attention_backend = MINIMAX_H3_ATTN_COMFY_KITCHEN_INT8
            attention_backend_name = String("ck-int8")
            sage_exact_av_prefix = True
            continue
        if arg == String("--attention-backend=evg-int8"):
            attention_backend = MINIMAX_H3_ATTN_EVG_INT8
            attention_backend_name = String("evg-int8-sm86")
            sage_exact_av_prefix = False
            continue
        if arg == String("--attention-backend=cudnn"):
            attention_backend = MINIMAX_H3_ATTN_CUDNN
            attention_backend_name = String("cudnn")
            sage_exact_av_prefix = False
            continue
        if arg.startswith("--attention-backend="):
            raise Error(
                String("unknown attention backend flag: ") + arg
                + String(
                    " (expected cudnn, sage-int8, sage-int8-pv8,"
                    " sage-int8-fast, ck-int8, or evg-int8)"
                )
            )
        if arg == String("--step-cache=high"):
            step_cache_enabled = True
            step_cache_name = String("high")
            continue
        if arg == String("--step-cache=exact"):
            step_cache_enabled = False
            step_cache_name = String("exact")
            continue
        if arg.startswith("--step-cache="):
            raise Error(
                String("unknown step cache flag: ") + arg
                + String(" (expected exact or high)")
            )
        if arg == String("--resident-backend=w8a8"):
            resident_scheme = MINIMAX_H3_RESIDENT_INT8_W8A8
            resident_backend_name = String("w8a8")
            continue
        if arg == String("--resident-backend=groupwise"):
            resident_scheme = MINIMAX_H3_RESIDENT_INT8
            resident_backend_name = String("groupwise")
            continue
        if arg.startswith("--resident-backend="):
            raise Error(
                String("unknown resident backend flag: ") + arg
                + String(" (expected groupwise or w8a8)")
            )
        if arg == String("--encoder-storage=int8"):
            encoder_storage = MINIMAX_H3_ENCODER_INT8
            encoder_storage_name = String("int8")
            continue
        if arg == String("--encoder-storage=bf16"):
            encoder_storage = MINIMAX_H3_ENCODER_BF16
            encoder_storage_name = String("bf16")
            continue
        if arg.startswith("--encoder-storage="):
            raise Error(
                String("unknown encoder storage flag: ") + arg
                + String(" (expected bf16 or int8)")
            )
        if arg == String("--defer-video-decode"):
            defer_video_decode = True
            continue
        if arg == String("--runtime-cache-exact-product-prompt"):
            runtime_cache = True
            continue
        if arg == String("--prepare-runtime-cache"):
            prepare_runtime_cache = True
            continue
        if arg == String("--validate-request"):
            validate_request = True
            continue
        if arg.startswith("--eval-start="):
            var fields = arg.split("=")
            if len(fields) != 2:
                raise Error("invalid --eval-start flag")
            eval_start = atol(String(fields[1]))
            continue
        if arg.startswith("--eval-stop="):
            var fields = arg.split("=")
            if len(fields) != 2:
                raise Error("invalid --eval-stop flag")
            eval_stop = atol(String(fields[1]))
            continue
        if arg.startswith("--motion-context="):
            var fields = arg.split("=")
            if len(fields) != 2 or String(fields[1]).byte_length() == 0:
                raise Error("invalid --motion-context flag")
            motion_context_path = String(fields[1])
            continue
        if arg.startswith("--motion-context-frames="):
            var fields = arg.split("=")
            if len(fields) != 2:
                raise Error("invalid --motion-context-frames flag")
            motion_context_frames = atol(String(fields[1]))
            continue
        if arg.startswith("--trim-start-frames="):
            var fields = arg.split("=")
            if len(fields) != 2:
                raise Error("invalid --trim-start-frames flag")
            trim_start_frames = atol(String(fields[1]))
            continue
        if arg.startswith("--temporal-rope-scale="):
            var fields = arg.split("=")
            if len(fields) != 2:
                raise Error("invalid --temporal-rope-scale flag")
            temporal_rope_scale = _h3_parse_f32(String(fields[1]))
            continue
        args.append(arg)
    var motion_context_enabled = motion_context_path != String("")
    if motion_context_enabled:
        _ = minimax_h3_motion_context_steps(motion_context_frames)
        if trim_start_frames == 0:
            trim_start_frames = motion_context_frames
    elif trim_start_frames != 0:
        raise Error("--trim-start-frames requires --motion-context")
    if (
        attention_backend == MINIMAX_H3_ATTN_EVG_INT8
        and motion_context_enabled
    ):
        raise Error(
            "MiniMax-H3 EVG attention currently requires contiguous"
            " text|audio|video T2VA packing; motion context is not admitted"
        )
    if (
        attention_backend == MINIMAX_H3_ATTN_EVG_INT8
        and not MINIMAX_H3_EVG_BUILD_ENABLED
    ):
        raise Error(
            "MiniMax-H3 EVG attention is not in this binary; rebuild with"
            " -D H3_EVG=1"
        )
    if quant == String("bf16"):
        use_resident = False
    elif quant == String("int8"):
        use_resident = True
        resident_scheme = MINIMAX_H3_RESIDENT_INT8
        resident_backend_name = String("groupwise")
    elif quant == String("int8-fast"):
        use_resident = True
        resident_scheme = MINIMAX_H3_RESIDENT_INT8_W8A8
        resident_backend_name = String("w8a8")
    else:
        raise Error(
            String("unknown --quant value: ") + quant
            + String(" (expected bf16, int8, or int8-fast)")
        )
    var _lora_overlay = Optional[H3LoraOverlay](None)
    if lora_path != String("") and use_resident and resident_blocks_requested == 0:
        raise Error("--lora with --resident-blocks=0 is unsupported (groupwise tail cache has no overlay)")
    if quant == String("bf16") and (
        attention_backend == MINIMAX_H3_ATTN_SAGE_INT8
        or attention_backend == MINIMAX_H3_ATTN_SAGE_INT8_PV8
        or attention_backend == MINIMAX_H3_ATTN_SAGE_INT8_FAST
        or attention_backend == MINIMAX_H3_ATTN_COMFY_KITCHEN_INT8
    ):
        raise Error(
            "MiniMax-H3 INT8 attention requires --quant=int8 or int8-fast;"
            " use --attention-backend=cudnn with --quant=bf16"
        )
    if len(args) < 3:
        print(
            "usage: minimax_h3_t2va <prompt> <out_dir> [steps=30] [seed=0]"
            " [max_blocks=50]"
            " [--attention-backend=cudnn|ck-int8|sage-int8|sage-int8-pv8|"
            "sage-int8-fast|evg-int8]"
            " [--step-cache=exact|high]"
            " [--resident-backend=groupwise|w8a8]"
            " [--quant=bf16|int8|int8-fast]"
            " [--width=N] [--height=N] [--frames=N] [--output-frames=N]"
            " [--fps=N] [--output-fps=N]"
            " [--resident-blocks=N]"
            " [--encoder-storage=bf16|int8]"
            " [--runtime-cache-exact-product-prompt]"
            " [--prepare-runtime-cache]"
            " [--validate-request]"
            " [--eval-start=N] [--eval-stop=N]"
            " [--motion-context=PATH]"
            " [--motion-context-frames=5|22|39]"
            " [--trim-start-frames=N]"
            " [--temporal-rope-scale=F]"
            " [--defer-video-decode]"
        )
        print(
            "  default geometry:", runtime_width, "x", runtime_height, ",",
            runtime_frames, "frames, text_tokens=", runtime_text_tokens,
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
    var decode_only_request = (
        len(args) >= 7
        and (
            String(args[6]) == "decode_only"
            or String(args[6]) == "decode_audio_only"
            or String(args[6]) == "decode_video_only"
        )
    )

    # Resolve exact text geometry before creating a DeviceContext or running
    # the 50-layer encoder. Unlike padding to a fixed DiT budget, using the
    # real token count adds no synthetic rows to packed self-attention.
    var prompt_token_ids = minimax_h3_tokenize_prompt(
        String(PROCESSOR_DIR), prompt
    )
    runtime_text_tokens = len(prompt_token_ids)
    if runtime_text_tokens == 0:
        raise Error("MiniMax-H3 prompt tokenized to zero ids")
    if runtime_text_tokens > 2048:
        raise Error(
            String("MiniMax-H3 prompt is ") + String(runtime_text_tokens)
            + " tokens; the text-encoder attention dispatch tops out at 2048"
        )

    if runtime_fps < 1 or runtime_fps > 120:
        raise Error("MiniMax-H3 internal FPS must be in [1,120]")
    if output_frames == FRAMES and runtime_frames != FRAMES:
        output_frames = runtime_frames
    if output_frames < 1:
        raise Error("MiniMax-H3 output frames must be positive")
    if output_fps < 1 or output_fps > 120:
        raise Error("MiniMax-H3 output FPS must be in [1,120]")
    if temporal_rope_scale <= Float32(0.0) \
            or temporal_rope_scale > Float32(1.0):
        raise Error("MiniMax-H3 temporal RoPE scale must be in (0,1]")
    if motion_context_enabled:
        if runtime_fps != MINIMAX_H3_MOTION_CONTEXT_FPS:
            raise Error("MiniMax-H3 motion context requires native 24 FPS")
        if trim_start_frames != motion_context_frames:
            raise Error(
                "MiniMax-H3 motion context must trim exactly its overlap"
            )
    var latent_h = runtime_height // 16
    var latent_w = runtime_width // 16
    var num_latent_frames = (runtime_frames - 5) // 17 * 5 + 2
    var num_audio_latents = Int(
        round(Float64(runtime_frames) / Float64(runtime_fps) * 40.0)
    )
    var rows_per_frame = (latent_h // PATCH_H) * (latent_w // PATCH_W)
    var num_video_rows = num_latent_frames * rows_per_frame
    var num_audio_rows = num_audio_latents * 2
    var num_condition_video_rows = 0
    var num_condition_audio_rows = 0
    if motion_context_enabled:
        num_condition_video_rows = (
            minimax_h3_motion_context_steps(motion_context_frames)
            * rows_per_frame
        )
        num_condition_audio_rows = (
            minimax_h3_motion_context_audio_latents(motion_context_frames) * 2
        )
    var sequence_length = (
        runtime_text_tokens + num_condition_video_rows
        + num_condition_audio_rows + num_audio_rows + num_video_rows
    )
    if (
        sequence_length > MINIMAX_H3_FAST_RESIDENT_EXACT_SEQUENCE_LIMIT
        and resident_blocks_requested > 0
    ):
        print(
            "  resident policy: exact S=", sequence_length,
            "exceeds", MINIMAX_H3_FAST_RESIDENT_EXACT_SEQUENCE_LIMIT,
            "; using streamed weights",
        )
        resident_blocks_requested = 0
    if sage_exact_av_prefix and not motion_context_enabled:
        if attention_backend == MINIMAX_H3_ATTN_COMFY_KITCHEN_INT8:
            attention_backend = minimax_h3_comfy_kitchen_exact_prefix_backend(
                runtime_text_tokens + num_audio_rows
            )
        elif attention_backend == MINIMAX_H3_ATTN_SAGE_INT8_FAST:
            attention_backend = minimax_h3_sage_fast_exact_prefix_backend(
                runtime_text_tokens + num_audio_rows
            )
        else:
            attention_backend = minimax_h3_sage_exact_prefix_backend(
                runtime_text_tokens + num_audio_rows
            )
    if runtime_width < 32 or runtime_width > 2048 \
            or runtime_height < 32 or runtime_height > 2048 \
            or runtime_width % 32 != 0 or runtime_height % 32 != 0:
        raise Error(
            "MiniMax-H3 width and height must each be in [32, 2048]"
            " and divisible by 32"
        )
    if runtime_frames < 5 or runtime_frames > MINIMAX_H3_SINGLE_PASS_MAX_FRAMES \
            or runtime_frames % 17 != 5:
        raise Error(
            "MiniMax-H3 internal frames must be in [5, 4323] and satisfy"
            " frames % 17 == 5"
        )
    if (
        trim_start_frames >= runtime_frames
        or Float64(runtime_frames - trim_start_frames) / Float64(runtime_fps)
            < Float64(output_frames) / Float64(output_fps)
    ):
        raise Error(
            "MiniMax-H3 internal frames do not cover the requested output"
            " after motion-context trimming"
        )
    # Decode-only loads already-denoised row-space latents and never allocates
    # the packed DiT sequence.  The long-context envelope protects denoising,
    # not the spatially tiled video VAE or windowed audio VAE.
    if not decode_only_request \
            and runtime_frames > MINIMAX_H3_TRAINED_MAX_FRAMES \
            and sequence_length > MINIMAX_H3_SINGLE_PASS_MAX_SEQUENCE:
        raise Error(
            String("MiniMax-H3 experimental single-pass long context S=")
            + String(sequence_length) + " exceeds the 109303-token 24-GB"
            " envelope; lower resolution or duration"
        )
    if validate_request:
        print(
            "MiniMax-H3 request valid:", runtime_width, "x", runtime_height,
            runtime_frames, "frames at", runtime_fps, "FPS, S=",
            sequence_length, "quant=", quant, "resident_blocks=",
            resident_blocks_requested,
        )
        return

    if runtime_cache and encoder_storage != MINIMAX_H3_ENCODER_INT8:
        raise Error(
            "MiniMax-H3 product runtime cache is keyed to the INT8 text"
            " encoder output; use --encoder-storage=int8"
        )
    if prepare_runtime_cache and not runtime_cache:
        raise Error(
            "--prepare-runtime-cache requires"
            " --runtime-cache-exact-product-prompt"
        )
    if step_cache_enabled and eval_start != 0:
        raise Error(
            "MiniMax-H3 High step cache cannot resume mid-denoise;"
            " start at evaluation 0"
        )
    if runtime_cache or use_resident:
        var mkdir_rc = sys_system(
            String("mkdir -p ") + shell_quote(String(RUNTIME_CACHE_DIR))
        )
        if mkdir_rc != 0:
            raise Error("MiniMax-H3 could not create the runtime cache directory")

    # ── DECODE-ONLY MODES reuse a prior run's saved row-space latents.
    # The split modes are load-bearing for unusually long generations: the
    # audio BigVGAN and video VAE must not share one 24 GiB CUDA process when
    # either decoder alone approaches the card's allocation ceiling.
    if decode_only_request:
        var decode_mode = String(args[6])
        var ctx2 = DeviceContext()
        print(" ", decode_mode, ": loading", out_dir + "/latents.safetensors")
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
        _assert_finite_rows("video_state_rows", video_rows, ctx2)
        _assert_finite_rows("audio_state_rows", audio_rows, ctx2)
        var config2 = minimax_h3_released_config()
        if decode_mode != "decode_video_only":
            _ = _minimax_h3_decode_audio(
                audio_rows, num_audio_latents, config2.audio_latents_dim,
                out_dir, ctx2,
            )
            if decode_mode == "decode_audio_only":
                print("  DECODE-AUDIO-ONLY done:", out_dir + "/audio.wav")
                return

        var nframes = 0
        if decode_mode != "decode_audio_only":
            nframes = _minimax_h3_decode_video(
                video_rows, num_latent_frames, latent_h, latent_w,
                config2.latents_dim, String(VIDEO_VAE_DIR), out_dir, ctx2,
            )
            if decode_mode == "decode_video_only":
                print(
                    "  DECODE-VIDEO-ONLY done:", nframes, "frames ->",
                    out_dir + "/frames.rgb",
                )
                return

        var artifact = _minimax_h3_mux_av(
            out_dir, output_frames, runtime_width, runtime_height,
            runtime_fps, output_fps, trim_start_frames,
        )
        _minimax_h3_write_decode_result(
            out_dir, artifact, output_frames, runtime_width, runtime_height,
            output_fps,
        )
        print(
            "  DECODE-ONLY done:", nframes, "frames +",
            out_dir + "/audio.wav +", artifact,
        )
        return

    if steps < 2:
        raise Error("minimax_h3_t2va: steps must be >= 2 (the schedule needs >= 2 sigmas)")
    if max_blocks < 1:
        raise Error("minimax_h3_t2va: max_blocks must be >= 1")

    print("=== MiniMax-H3 t2va ===")
    print("  prompt:", prompt)
    print(
        "  geometry:", runtime_width, "x", runtime_height, ",", runtime_frames,
        "frames -> latent [", num_latent_frames, ",", latent_h, ",", latent_w,
        "], audio_latents=", num_audio_latents, ", text_tokens=",
        runtime_text_tokens, ", S=", sequence_length,
    )
    print("  steps=", steps, " seed=", seed, " max_blocks=", max_blocks)
    print("  attention_backend=", attention_backend_name)
    print("  step_cache=", step_cache_name)
    print("  resident_backend=", resident_backend_name)
    print("  quant=", quant)
    print("  encoder_storage=", encoder_storage_name)
    print("  temporal_rope_scale=", temporal_rope_scale)

    # ── PREFLIGHT (before DeviceContext) ──────────────────────────────────
    var t_preflight0 = perf_counter_ns()
    _preflight_geometry(
        runtime_width, runtime_height, runtime_frames, runtime_text_tokens,
        num_latent_frames, num_audio_latents,
    )
    var config = minimax_h3_released_config()
    config.validate()
    if motion_context_enabled:
        minimax_h3_preflight_motion_context(
            motion_context_path,
            rows_per_frame,
            config.video_patch_dim(),
            config.audio_latents_dim,
            motion_context_frames,
        )
        print(
            "  preflight: native motion context", motion_context_frames,
            "frames from", motion_context_path,
        )
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

    # Text-encoder shards are NOT opened here: the old preflight parsed every
    # shard header only to print the count — on a conditioning-cache hit the
    # encoder is never read at all, and on a miss the encoder loader itself
    # fails loudly, by name, at the tensor that is actually missing.
    if partial_mode:
        print("  preflight: text_encoder SKIPPED (PARTIAL MODE — conditioning is stubbed)")

    print("  preflight: opening audio_vae:", String(AUDIO_VAE_PATH))
    _minimax_h3_preflight_audio_vae(String(AUDIO_VAE_PATH))
    print("  preflight: audio_vae OK")

    var t_preflight1 = perf_counter_ns()
    print("  preflight OK (", Float64(t_preflight1 - t_preflight0) / 1.0e6, "ms)")

    _ = sys_system(String("mkdir -p '") + out_dir + "'")

    var ctx = DeviceContext()
    if lora_path != String(""):
        print("  loading LoRA overlay:", lora_path, " mult:", lora_mult)
        _lora_overlay = Optional[H3LoraOverlay](
            H3LoraOverlay.load(lora_path, lora_mult, ctx)
        )
        print("  LoRA overlay adapters:", _lora_overlay.value().adapters)

    # ── 1. Conditioning (real, or STUBBED in partial mode — see file header) ──
    var t_cond0 = perf_counter_ns()
    var conditioning_cache = (
        String(PRODUCT_CONDITIONING_CACHE) if runtime_cache else String("")
    )
    var cond = _minimax_h3_get_conditioning_cached(
        partial_mode, prompt, seed, config, encoder_storage,
        conditioning_cache, runtime_text_tokens, ctx,
    )
    if len(cond.token_tags) != runtime_text_tokens:
        raise Error(
            String("minimax_h3_t2va: prompt tokenized to ")
            + String(len(cond.token_tags)) + " tokens after preflight resolved "
            + String(runtime_text_tokens) + "; tokenizer result drifted"
        )
    var text_rows = reshape(
        cond.embeds, [runtime_text_tokens, config.text_dim], ctx
    )
    var t_cond1 = perf_counter_ns()
    print(
        "  conditioning", "[STUBBED]" if partial_mode else "[real]", ": ",
        runtime_text_tokens, " tokens (",
        Float64(t_cond1 - t_cond0) / 1.0e9, "s)",
    )

    # Load the previous clip's compact latent tail directly onto the GPU.
    # Video condition noise is mixed once, then remains pinned for every
    # denoise step; audio remains bit-identical to the generated source tail.
    var condition_video_rows = Optional[Tensor](None)
    var condition_audio_rows = Optional[Tensor](None)
    var source_audio_overhang = Float64(0.0)
    if motion_context_enabled:
        var loaded_context = minimax_h3_load_motion_context(
            motion_context_path,
            rows_per_frame,
            config.video_patch_dim(),
            config.audio_latents_dim,
            motion_context_frames,
            runtime_width,
            runtime_height,
            ctx,
        )
        source_audio_overhang = loaded_context.source_audio_overhang
        var condition_noise = randn(
            loaded_context.video_rows.shape().copy(), seed + 102,
            STDtype.F32, ctx,
        )
        var scaled_video = mul_scalar(
            loaded_context.video_rows, Float32(0.999), ctx
        )
        var scaled_noise = mul_scalar(
            condition_noise, Float32(0.001), ctx
        )
        var noisy_video = add(scaled_video, scaled_noise, ctx)
        condition_video_rows = Optional[Tensor](noisy_video^)
        condition_audio_rows = Optional[Tensor](slice(
            loaded_context.audio_rows, 0, 0,
            loaded_context.audio_rows.shape()[0], ctx,
        ))
        print(
            "  motion context: pinned video rows=",
            condition_video_rows.value().shape()[0],
            " audio rows=", condition_audio_rows.value().shape()[0],
            " source_audio_overhang=", source_audio_overhang,
        )

    # ── 2. Packed-sequence geometry (host scalar, this port's own reproduction) ──
    var no_anchors = List[Int]()
    var geometry: MiniMaxH3SamplingGeometry
    if motion_context_enabled:
        geometry = minimax_h3_build_motion_context_geometry(
            cond.token_tags, num_latent_frames, latent_h, latent_w,
            num_audio_latents, PATCH_H, PATCH_W, motion_context_frames,
            source_audio_overhang,
        )
    else:
        geometry = minimax_h3_build_sampling_geometry(
            cond.token_tags, num_latent_frames, latent_h, latent_w,
            num_audio_latents, PATCH_H, PATCH_W, no_anchors,
        )
    if geometry.sequence_length != sequence_length:
        raise Error(
            String("minimax_h3_t2va: geometry.sequence_length ")
            + String(geometry.sequence_length) + " != compiled S "
            + String(sequence_length) + " — runtime geometry derivation"
            " drifted from the shared packing implementation"
        )
    if len(geometry.video_indices) \
            != num_condition_video_rows + num_video_rows:
        raise Error("minimax_h3_t2va: video row count mismatch")
    if len(geometry.audio_indices) \
            != num_condition_audio_rows + num_audio_rows:
        raise Error("minimax_h3_t2va: audio row count mismatch")

    # Experimental one-pass long-context position interpolation. H3 was
    # released for at most 15 seconds, so an unscaled 60-second target drives
    # media MM-RoPE four times beyond the trained temporal span and changes
    # every row through global self-attention. Keep text positions exact and
    # compress only media time around the canonical text/media boundary.
    # Scale 1.0 is the source-faithful default and is byte-identical to the
    # previous path; callers must opt in explicitly for long extrapolation.
    if temporal_rope_scale != Float32(1.0):
        var media_origin = Float64(runtime_text_tokens)
        var media_scale = Float64(temporal_rope_scale)
        for i in range(len(geometry.audio_indices)):
            var row = geometry.audio_indices[i]
            var offset = 3 * row
            geometry.position_ids[offset] = media_origin + (
                geometry.position_ids[offset] - media_origin
            ) * media_scale
        for i in range(len(geometry.video_indices)):
            var row = geometry.video_indices[i]
            var offset = 3 * row
            geometry.position_ids[offset] = media_origin + (
                geometry.position_ids[offset] - media_origin
            ) * media_scale
        print(
            "  temporal RoPE interpolation: media scale=",
            temporal_rope_scale, " origin=", runtime_text_tokens,
        )

    # ── 3. MM-RoPE tables (device) ─────────────────────────────────────────
    var positions_f32 = List[Float32](capacity=len(geometry.position_ids))
    for i in range(len(geometry.position_ids)):
        positions_f32.append(Float32(geometry.position_ids[i]))
    var positions_shape: List[Int] = [sequence_length * 3]
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
    if eval_stop < 0:
        eval_stop = num_steps
    if (
        eval_start < 0
        or eval_start >= num_steps
        or eval_stop <= eval_start
        or eval_stop > num_steps
    ):
        raise Error("MiniMax-H3 evaluation range is outside the schedule")
    print("  schedule: ", num_steps, " model evaluations")
    print("  evaluation range: [", eval_start, ",", eval_stop, ")")

    # ── 5. AdaLN modulation cache — ONE streamed pass, built ONCE ──────────
    var t_mod0 = perf_counter_ns()
    var distinct_timesteps = (
        4 * num_steps if motion_context_enabled else 2 * num_steps
    )
    var temb_timesteps = List[Float32](capacity=distinct_timesteps)
    for i in range(num_steps):
        var video_t = schedule.video_timestep(i)
        temb_timesteps.append(video_t)
        temb_timesteps.append(schedule.audio_timestep(i))
        if motion_context_enabled:
            temb_timesteps.append(
                video_t if video_t > Float32(0.999) else Float32(0.999)
            )
            temb_timesteps.append(Float32(1.0))
    var temb_shape: List[Int] = [distinct_timesteps]
    var temb_timesteps_tensor = Tensor.from_host(temb_timesteps, temb_shape^, STDtype.F32, ctx)

    var frontend_w = _minimax_h3_load_frontend_weights(transformer_shards, config, ctx)
    var temb = minimax_h3_timestep_embedding(temb_timesteps_tensor, frontend_w, config, ctx)
    var modcache_path = (
        String(RUNTIME_CACHE_DIR) + (
            String("/modcache_motion_context_steps_")
            if motion_context_enabled else String("/modcache_steps_")
        )
        + String(steps) + String("_blocks_")
        + String(run_config.num_layers) + String(".safetensors")
    )
    var modcache = _minimax_h3_get_modcache_cached(
        transformer_shards,
        temb,
        run_config,
        steps,
        distinct_timesteps,
        modcache_path,
        ctx,
    )
    ctx.synchronize()
    var t_mod1 = perf_counter_ns()
    print(
        "  modcache: ", distinct_timesteps, " rows, ",
        Float64(modcache.total_bytes()) / (1024.0 * 1024.0), " MiB (",
        Float64(t_mod1 - t_mod0) / 1.0e9, "s)",
    )

    # ── 5b. OPTIONAL fp8-resident base (H3_FP8_RESIDENT=1 builds only) ─────
    # One more streamed pass over the 50 blocks, quantizing each to E4M3 +
    # scheme-specific scales on device; every denoise step below then dequants from
    # residency instead of re-reading ~36 GiB from disk. Declared
    # unconditionally (an empty Optional costs nothing); populated only under
    # the comptime flag, so a default build's denoise loop is untouched.
    var fp8_resident = Optional[MiniMaxH3ResidentFp8](None)
    var reusable_w8a8_tail = Optional[MiniMaxH3ResidentFp8](None)
    var resident_cache_path = String("")

    if use_resident:
        var resident_blocks = resident_blocks_requested
        if resident_blocks < 0:
            raise Error("--resident-blocks must be non-negative")
        if resident_blocks > run_config.num_layers:
            resident_blocks = run_config.num_layers
        var t_q0 = perf_counter_ns()
        var resident_cache_save_allowed = True
        if resident_scheme == MINIMAX_H3_RESIDENT_INT8_W8A8:
            # The direct store can load a strict prefix from the full
            # 50-block cache. Production normally uses a measured prefix.
            resident_cache_path = (
                String(RUNTIME_CACHE_DIR)
                + String("/resident_w8a8_row_blocks_")
                + String(run_config.num_layers) + String(".safetensors")
            )
            resident_cache_save_allowed = (
                resident_blocks == run_config.num_layers
            )
        else:
            resident_cache_path = (
                String(RUNTIME_CACHE_DIR)
                + String("/resident_groupwise_q16_o64_fc132_fc264_blocks_")
                + String(GROUPWISE_RUNTIME_CACHE_BLOCKS)
                + String(".safetensors")
            )
            resident_cache_save_allowed = (
                resident_blocks == GROUPWISE_RUNTIME_CACHE_BLOCKS
            )
        # The base resident cache is adapter-independent. LoRA stays in its
        # own BF16 activation branch, so loading an adapter does not rebuild
        # or risk poisoning the shared INT8 cache.
        if _lora_overlay:
            print("  resident store: base cache + activation LoRA overlay")
        fp8_resident = Optional[MiniMaxH3ResidentFp8](
            _minimax_h3_get_resident_cached(
                transformer_shards,
                config,
                resident_blocks,
                resident_scheme,
                resident_cache_path,
                ctx,
                resident_cache_save_allowed,
                resident_scheme == MINIMAX_H3_RESIDENT_INT8,
            )
        )
        if (
            resident_scheme == MINIMAX_H3_RESIDENT_INT8_W8A8
            and resident_blocks < run_config.num_layers
        ):
            # Keep one fixed-shape W8A8 allocation and refill it in place for
            # the streamed tail. I2VA and Ref2VA already use this path; T2VA
            # must not allocate a fresh block store for every layer/step.
            reusable_w8a8_tail = Optional[MiniMaxH3ResidentFp8](
                load_minimax_h3_resident_cache(
                    resident_cache_path,
                    String(TRANSFORMER_INDEX),
                    config,
                    ctx,
                    1,
                    resident_blocks,
                    resident_scheme,
                )
            )
            print("  resident cache: reusable one-block W8A8 tail ready")
        var t_q1 = perf_counter_ns()
        print(
            "  fp8-resident: ", resident_blocks, " blocks + ",
            run_config.num_layers - resident_blocks, " streamed tail, ",
            Float64(fp8_resident.value().resident_bytes())
            / (1024.0 * 1024.0 * 1024.0),
            " GiB resident (", Float64(t_q1 - t_q0) / 1.0e9, "s one-time)",
        )

    if prepare_runtime_cache:
        print("  runtime cache preparation complete; denoise intentionally skipped")
        return

    # ── 6. Denoise loop — real streamed blocks, real Euler steps ───────────
    var t_denoise0 = perf_counter_ns()
    var video_shape: List[Int] = [num_video_rows, config.video_patch_dim()]
    var audio_shape: List[Int] = [num_audio_rows, config.audio_latents_dim]
    var video_state: Tensor
    var audio_state: Tensor
    if eval_start == 0:
        video_state = randn(video_shape^, seed, STDtype.F32, ctx)
        audio_state = randn(audio_shape^, seed + 1, STDtype.F32, ctx)
    else:
        var resume_path = out_dir + String("/resume_latents.safetensors")
        var resume_st = SafeTensors.open(resume_path)
        var vinfo = resume_st.tensor_info(String("video_state_rows"))
        var ainfo = resume_st.tensor_info(String("audio_state_rows"))
        if (
            vinfo.dtype != STDtype.F32
            or vinfo.shape != video_shape
            or ainfo.dtype != STDtype.F32
            or ainfo.shape != audio_shape
        ):
            raise Error("MiniMax-H3 resume latent dtype/shape mismatch")
        video_state = Tensor.from_view(
            from_parts(
                vinfo.dtype, vinfo.shape.copy(),
                resume_st.tensor_bytes(String("video_state_rows")),
            ),
            ctx,
        )
        audio_state = Tensor.from_view(
            from_parts(
                ainfo.dtype, ainfo.shape.copy(),
                resume_st.tensor_bytes(String("audio_state_rows")),
            ),
            ctx,
        )
        print(
            "  resumed latent rows at evaluation", eval_start,
            "from", resume_path,
        )

    # EVG keeps one immutable request geometry across all 50 layers and all
    # denoise evaluations. T2VA packs text|audio|video contiguously; the video
    # tail is a frame-major raster over the DiT patch grid.
    var evg_layout = Optional[ArcPointer[EVGH3RaggedLayout]](None)
    if attention_backend == MINIMAX_H3_ATTN_EVG_INT8:
        evg_layout = Optional[ArcPointer[EVGH3RaggedLayout]](
            ArcPointer(
                EVGH3RaggedLayout(
                    runtime_text_tokens + num_audio_rows,
                    num_latent_frames,
                    latent_h // PATCH_H,
                    latent_w // PATCH_W,
                    ctx,
                )
            )
        )
        ctx.synchronize()
        print(
            "  evg layout: prefix=",
            evg_layout.value()[].prefix_tokens,
            " video_blocks=", evg_layout.value()[].video_blocks,
            " packed_rows=", evg_layout.value()[].packed_rows,
        )

    # Sage attention runs the whole denoise through one preallocated scratch:
    # its per-call transient buffers otherwise churn ~1-2 GiB fifty times per
    # evaluation and intermittently OOM near the 24-GiB envelope
    # (video-0177). Allocated after the resident store so a geometry that
    # cannot hold both fails here, not minutes into denoise.
    var sage_scratch = Optional[SageInt8Scratch](None)
    if (
        attention_backend != MINIMAX_H3_ATTN_CUDNN
        and attention_backend != MINIMAX_H3_ATTN_EVG_INT8
        and attention_backend < MINIMAX_H3_ATTN_COMFY_KITCHEN_EXACT_PREFIX_BASE
        and attention_backend != MINIMAX_H3_ATTN_COMFY_KITCHEN_INT8
    ):
        sage_scratch = Optional[SageInt8Scratch](
            SageInt8Scratch(
                geometry.sequence_length, H3_HEADS, ctx,
                attention_backend == MINIMAX_H3_ATTN_SAGE_INT8_PV8 or (
                    (
                        attention_backend == MINIMAX_H3_ATTN_SAGE_INT8_FAST
                        or attention_backend
                            >= MINIMAX_H3_ATTN_SAGE_FAST_EXACT_PREFIX_BASE
                    )
                    and geometry.sequence_length >= MINIMAX_H3_SAGE_PV8_MIN_S
                ),
            )
        )
        ctx.synchronize()
        print(
            "  sage scratch: preallocated",
            Float64(sage_scratch.value().resident_bytes())
                / (1024.0 * 1024.0 * 1024.0),
            "GiB for S=", geometry.sequence_length,
        )
    var comfy_kitchen_scratch = Optional[ComfyKitchenAttentionScratch](None)
    if (
        attention_backend == MINIMAX_H3_ATTN_COMFY_KITCHEN_INT8
        or attention_backend
            >= MINIMAX_H3_ATTN_COMFY_KITCHEN_EXACT_PREFIX_BASE
    ):
        comfy_kitchen_scratch = Optional[ComfyKitchenAttentionScratch](
            ComfyKitchenAttentionScratch(
                geometry.sequence_length, H3_HEADS, ctx
            )
        )
        ctx.synchronize()
        print(
            "  ck scratch: preallocated",
            Float64(comfy_kitchen_scratch.value().resident_bytes())
                / (1024.0 * 1024.0 * 1024.0),
            "GiB for S=", geometry.sequence_length,
        )
    var step_cache = MiniMaxH3StepCache(step_cache_enabled, num_steps)
    for i in range(eval_start, eval_stop):
        var t_step0 = perf_counter_ns()
        var video_ts = schedule.video_timestep(i)
        var audio_ts = schedule.audio_timestep(i)
        var global_row = _minimax_h3_global_timestep_row(
            geometry.token_tags, i
        )
        if motion_context_enabled:
            global_row = _minimax_h3_motion_context_global_timestep_row(
                geometry, i
            )
        var block_adaln_indices = minimax_h3_adaln_rows(global_row, geometry.token_tags)

        var placeholder_ts_shape: List[Int] = [1]
        var placeholder_ts = Tensor.from_host([video_ts], placeholder_ts_shape^, STDtype.F32, ctx)
        if motion_context_enabled:
            var video_rows_combined = concat(
                0, ctx, condition_video_rows.value(), video_state
            )
            var audio_rows_combined = concat(
                0, ctx, condition_audio_rows.value(), audio_state
            )
            var frontend_out = _minimax_h3_model_eval_p[241](
                video_rows_combined, audio_rows_combined, text_rows,
                placeholder_ts, geometry, frontend_w, config, run_config,
                modcache, global_row, block_adaln_indices,
                transformer_shards, fp8_resident, reusable_w8a8_tail,
                use_resident, resident_scheme, resident_cache_path,
                _lora_overlay, rope[0],
                rope[1], rotary_dim, attention_backend, sage_scratch,
                comfy_kitchen_scratch,
                evg_layout, i,
                step_cache, False,
                ctx,
            )
            var target_video_out = slice(
                frontend_out.video_out, 0, num_condition_video_rows,
                num_video_rows, ctx,
            )
            var target_audio_out = slice(
                frontend_out.audio_out, 0, num_condition_audio_rows,
                num_audio_rows, ctx,
            )
            video_state = schedule.step_video_device(
                target_video_out, video_ts, video_state, ctx
            )
            audio_state = schedule.step_audio_device(
                target_audio_out, audio_ts, audio_state, ctx
            )
        else:
            var frontend_out = _minimax_h3_model_eval_p[241](
                video_state, audio_state, text_rows, placeholder_ts, geometry,
                frontend_w, config, run_config, modcache, global_row,
                block_adaln_indices, transformer_shards, fp8_resident,
                reusable_w8a8_tail,
                use_resident, resident_scheme, resident_cache_path,
                _lora_overlay, rope[0],
                rope[1], rotary_dim, attention_backend, sage_scratch,
                comfy_kitchen_scratch,
                evg_layout, i,
                step_cache, True,
                ctx,
            )
            video_state = schedule.step_video_device(
                frontend_out.video_out, video_ts, video_state, ctx
            )
            audio_state = schedule.step_audio_device(
                frontend_out.audio_out, audio_ts, audio_state, ctx
            )
        ctx.synchronize()
        var t_step1 = perf_counter_ns()
        print(
            "  phase=denoise step=", i + 1, " total=", num_steps,
            " video_t=", video_ts, " audio_t=", audio_ts,
            " (", Float64(t_step1 - t_step0) / 1.0e9, "s)",
        )

    # Quantized product paths fail at the latent boundary instead of surfacing
    # later as a misleading VAE/pixel error.  The scans and reductions run on
    # GPU; only scalar summaries cross back to the host.
    _assert_finite_rows("video_state_rows", video_state, ctx)
    _assert_finite_rows("audio_state_rows", audio_state, ctx)

    if eval_stop < num_steps:
        var resume_names = List[String]()
        resume_names.append(String("video_state_rows"))
        resume_names.append(String("audio_state_rows"))
        var resume_tensors = List[ArcPointer[Tensor]]()
        resume_tensors.append(ArcPointer[Tensor](
            slice(video_state, 0, 0, video_state.shape()[0], ctx)
        ))
        resume_tensors.append(ArcPointer[Tensor](
            slice(audio_state, 0, 0, audio_state.shape()[0], ctx)
        ))
        var resume_path = out_dir + String("/resume_latents.safetensors")
        save_safetensors(resume_names, resume_tensors, resume_path, ctx)
        print(
            "  saved resume latents at evaluation", eval_stop,
            "->", resume_path,
        )
        print("  partial denoise phase complete; decode intentionally skipped")
        return

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
    var continuation_end_pixel_frames = trim_start_frames + Int(round(
        Float64(output_frames) * Float64(runtime_fps)
        / Float64(output_fps)
    ))
    var saved_motion_frames = minimax_h3_save_motion_context_tail_at_pixel_frame(
        video_state,
        audio_state,
        num_latent_frames,
        num_audio_latents,
        continuation_end_pixel_frames,
        runtime_width,
        runtime_height,
        out_dir + String("/motion_context.safetensors"),
        ctx,
    )
    print(
        "  saved native A/V continuation tail (", saved_motion_frames,
        " frames) ->", out_dir + "/motion_context.safetensors",
    )

    var t_denoise1 = perf_counter_ns()
    print("  denoise done (", Float64(t_denoise1 - t_denoise0) / 1.0e9, "s)")
    if step_cache.enabled:
        print(
            "  cache-dit summary: full=", step_cache.full_evaluations,
            " cached=", step_cache.cached_evaluations,
            " residual_bytes=", step_cache.residual_bytes(),
        )

    # ── 7. Audio decode (WIRED, real waveform) ─────────────────────────────
    var t_vae0 = perf_counter_ns()
    var audio_samples = _minimax_h3_decode_audio(
        audio_state, num_audio_latents, config.audio_latents_dim, out_dir, ctx
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
    result_body += String("  \"width\":") + String(runtime_width) + String(",\n")
    result_body += String("  \"height\":") + String(runtime_height) + String(",\n")
    result_body += String("  \"frames\":") + String(output_frames) + String(",\n")
    result_body += String("  \"internal_frames\":") + String(runtime_frames) + String(",\n")
    result_body += String("  \"fps\":") + String(output_fps) + String(",\n")
    result_body += String("  \"internal_fps\":") + String(runtime_fps) + String(",\n")
    result_body += String("  \"sequence_length\":") + String(sequence_length) + String(",\n")
    result_body += String("  \"temporal_rope_scale\":") \
        + String(temporal_rope_scale) + String(",\n")
    result_body += String("  \"motion_context\":") \
        + json_bool(motion_context_enabled) + String(",\n")
    result_body += String("  \"motion_context_frames\":") \
        + String(motion_context_frames if motion_context_enabled else 0) \
        + String(",\n")
    result_body += String("  \"trim_start_frames\":") \
        + String(trim_start_frames) + String(",\n")
    result_body += String("  \"motion_context_artifact\":\"") \
        + json_escape(out_dir + String("/motion_context.safetensors")) \
        + String("\",\n")
    result_body += String("  \"attention_backend\":\"") \
        + attention_backend_name + String("\",\n")
    result_body += String("  \"step_cache\":\"") \
        + step_cache_name + String("\",\n")
    result_body += String("  \"step_cache_full_evaluations\":") \
        + String(step_cache.full_evaluations) + String(",\n")
    result_body += String("  \"step_cache_cached_evaluations\":") \
        + String(step_cache.cached_evaluations) + String(",\n")
    result_body += String("  \"weight_storage\":\"") + (
        (
            String("resident-int8-") + resident_backend_name + String("-")
            + String(resident_blocks_requested)
            + (
                String("+streamed-w8a8-cache-tail")
                if (
                    resident_scheme == MINIMAX_H3_RESIDENT_INT8_W8A8
                )
                else String("+streamed-bf16-tail")
            )
        ) if use_resident
        else String("streamed-bf16")
    ) + String("\",\n")
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

    # A resident DiT store and the GPU video VAE cannot coexist in 24 GiB.
    # A wrapper can now end this process, release its CUDA context, and invoke
    # decode_only in a fresh GPU process.  There is no CPU inference fallback.
    if defer_video_decode:
        print("  video decode deferred; run in a fresh GPU process:")
        print(
            "   ", String(args[0]), "decode", out_dir, steps, seed,
            max_blocks, "decode_only",
            String("--width=") + String(runtime_width),
            String("--height=") + String(runtime_height),
            String("--frames=") + String(runtime_frames),
            String("--output-frames=") + String(output_frames),
            String("--fps=") + String(runtime_fps),
            String("--output-fps=") + String(output_fps),
            String("--trim-start-frames=") + String(trim_start_frames),
            (
                String("--motion-context=") + motion_context_path
                if motion_context_enabled else String("")
            ),
            String("--motion-context-frames=") + String(motion_context_frames),
            String("--quant=") + quant,
            String("--resident-blocks=") + String(resident_blocks_requested),
        )
        return

    var t_vid0 = perf_counter_ns()
    var frames_written = _minimax_h3_decode_video(
        video_state, num_latent_frames, latent_h, latent_w,
        config.latents_dim, String(VIDEO_VAE_DIR), out_dir, ctx,
    )
    var t_vid1 = perf_counter_ns()
    print("  video decode done (", Float64(t_vid1 - t_vid0) / 1.0e9, "s)")
    var artifact = _minimax_h3_mux_av(
        out_dir, output_frames, runtime_width, runtime_height,
        runtime_fps, output_fps, trim_start_frames,
    )
    _minimax_h3_write_decode_result(
        out_dir, artifact, output_frames, runtime_width, runtime_height,
        output_fps,
    )


def _serve_read_line() raises -> String:
    """Blocking newline-delimited read from stdin (fd 0). Returns "" on EOF."""
    var out = String("")
    while True:
        var buf = alloc[UInt8](1)
        var got = external_call["read", Int](Int32(0), buf, 1)
        if got <= 0:
            buf.free()
            return String("")  # EOF/parent closed -> shut down
        var c = Int(buf[0])
        buf.free()
        if c == 10:
            return out
        out += chr(c)


def _serve_redirect_stdout(path: String) raises:
    """dup2 a fresh per-job log over stdout+stderr so the resident worker's
    output lands where the per-job spawn used to point it."""
    var n = path.byte_length()
    var cbuf = alloc[UInt8](n + 1)
    var src = path.as_bytes()
    for i in range(n):
        cbuf[i] = src[i]
    cbuf[n] = 0
    # O_WRONLY|O_CREAT|O_TRUNC = 0x241, mode 0644
    var fd = Int(external_call["open", Int32](cbuf, Int32(0x241), Int32(0o644)))
    cbuf.free()
    if fd < 0:
        raise Error(String("serve: cannot open job log: ") + path)
    _ = external_call["dup2", Int32](Int32(fd), Int32(1))
    _ = external_call["dup2", Int32](Int32(fd), Int32(2))
    _ = external_call["close", Int32](Int32(fd))


def main() raises:
    var raw = argv()
    var serve = False
    var toks = List[String]()
    for i in range(len(raw)):
        var a = String(raw[i])
        if a == "--serve":
            serve = True
        else:
            toks.append(a^)
    if not serve:
        _job_main(toks)
        return

    # ── WARM WORKER (Phase A): one resident process serves many denoise
    # jobs. Each job arrives as one stdin line: a JSON array of the exact
    # argv tokens the per-job spawn used to pass (binary name excluded).
    # stdout/stderr are re-pointed at <out_dir>/runner.log per job, so the
    # Rust side's log-driven progress/status contract is unchanged; job
    # completion is signaled by result.json (success, unchanged) or the
    # "[serve] job FAILED" sentinel below. The process boot, CUDA context,
    # JIT-cache checks, and warm page cache carry across jobs; the deferred
    # video decode stays a fresh process (measured VRAM isolation).
    print("[serve] MiniMax-H3 warm worker ready")
    while True:
        var line = _serve_read_line()
        if line == String(""):
            print("[serve] stdin closed — exiting")
            return
        if line.byte_length() < 3:
            continue
        var job_toks = List[String]()
        job_toks.append(toks[0].copy() if len(toks) > 0 else String("minimax_h3_serenity_runtime"))
        try:
            var arr = _json_loads(line)
            for i in range(arr.length()):
                job_toks.append(arr[i].as_string())
        except e:
            print("[serve] bad job line (", e, ") — ignored")
            continue
        # argv layout: [bin, prompt, out_dir, ...] -> out_dir is token 2.
        if len(job_toks) > 2:
            try:
                _serve_redirect_stdout(job_toks[2] + String("/runner.log"))
            except e:
                print("[serve] log redirect failed (", e, ") — continuing on current log")
        try:
            _job_main(job_toks)
            print("[serve] job complete — worker staying warm")
        except e:
            # A mid-job exception leaves device allocations stranded in this
            # process's CUDA pool (measured: a failed job left 23.1 GiB
            # resident and poisoned every later context on the card). Failed
            # jobs are rare; exit and let the supervisor respawn clean.
            print("[serve] job FAILED:", e)
            print("[serve] exiting after failure — supervisor respawns clean")
            return

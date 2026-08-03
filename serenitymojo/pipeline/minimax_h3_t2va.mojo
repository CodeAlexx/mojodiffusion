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
#   -> video/audio patch-token latents (decode NOT wired this pass — see
#          "VIDEO/AUDIO DECODE" below)
#
# ── THIS PASS'S SCOPE (team-lead instruction) ────────────────────────────
# "Start with structure + preflight + the denoise loop over real streamed
# blocks; report before wiring decode." Both video AND audio VAE decode are
# STUBBED with a loud "not wired" error — see `_minimax_h3_decode_stub`
# below. The denoise loop itself is real: real streamed block loads via
# `minimax_h3_load_block_device`, a real modulation cache, real Euler steps
# through real `ops/tensor_algebra` GPU kernels. Nothing between the initial
# noise and the final patch-token state is a stand-in.
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
# unpack (models/minimax_h3/rearrange.mojo's job, not yet reproduced on the
# device path) is then needed exactly ONCE, at the very end, not every
# step — and this pass stops before that point anyway (decode is stubbed).
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
# ── VIDEO/AUDIO DECODE: STUBBED, NOT WIRED ──────────────────────────────
# Video VAE: team-lead instruction — the existing device port(s) are
# unverified/being rebuilt against the creator's own source (FFN gate/value
# order, fused-qkv split). `models/vae/minimax_h3_video_decoder_device.mojo`
# exists on disk but its gate status is unknown to this file; it is
# deliberately NOT imported here. `_minimax_h3_decode_stub` below is the
# seam: it takes exactly the row-form tensors the denoise loop already
# produces and raises a named "not wired" error. Dropping in the corrected
# decoder later means adding one unpatchify call + one decoder call inside
# that function — this file's structure does not change.
# Audio VAE: the host-float32 oracle (`models/minimax_h3/audio_decoder.
# mojo`) is reported verified against real released weights, but no DEVICE
# port of it exists yet in this repo (searched; only video has a `_device`
# module). Building one is a separate unit, not this pass's — audio decode
# is stubbed the same way as video, not partially wired.
#
# argv: <prompt> <out_dir> [steps=30] [seed=0]
#
# ── CHECKPOINT LAYOUT (matches every H3 device module's own defaults) ───
#   transformer:   .../MiniMax-H3/FL2VA/transformer    (61.73 GiB, 13 shards)
#   text_encoder:  .../MiniMax-H3/FL2VA/text_encoder    (62.13 GiB, 14 shards)
#   processor:     .../MiniMax-H3/FL2VA/processor       (tokenizer.json + config)
# As of this writing (team-lead status): transformer shards landing (6/13);
# text_encoder shards not yet down. Preflight below fails loudly and
# specifically on whichever is still incomplete — that is a VALID outcome
# of running this file today, not a bug in it.
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
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import sys_system
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import reshape
from serenitymojo.serve.product_manifest import json_escape, write_text_file

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
    minimax_h3_encode_conditioning,
)
from serenitymojo.models.dit.minimax_h3_sampling import (
    MiniMaxH3DualSchedule,
    MiniMaxH3SamplingGeometry,
    minimax_h3_build_sampling_geometry,
)


# ── Checkpoint paths (matches every other H3 device module's own defaults) ──
comptime H3_ROOT = "/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA"
comptime TRANSFORMER_DIR = H3_ROOT + "/transformer"
comptime TEXT_ENCODER_DIR = H3_ROOT + "/text_encoder"
comptime PROCESSOR_DIR = H3_ROOT + "/processor"

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
# Video/audio decode seam — STUBBED. See file header "VIDEO/AUDIO DECODE".
# ═════════════════════════════════════════════════════════════════════════════
def _minimax_h3_decode_stub(
    video_state: Tensor,  # [NUM_VIDEO_ROWS, video_patch_dim] F32, patch-token space
    audio_state: Tensor,  # [NUM_AUDIO_ROWS, audio_latents_dim] F32, row space
) raises:
    raise Error(
        String("minimax_h3_t2va: video/audio VAE decode is NOT WIRED. The")
        + " denoise loop completed and produced final patch-token latents"
        " video_state=" + String(video_state.shape())
        + " audio_state=" + String(audio_state.shape())
        + " — video VAE is being rebuilt against the creator's own source"
        " (FFN gate/value order, fused-qkv split) and no device audio VAE"
        " exists yet in this repo. See this file's header, VIDEO/AUDIO"
        " DECODE, for the exact seam to fill in."
    )


# ═════════════════════════════════════════════════════════════════════════════
# main
# ═════════════════════════════════════════════════════════════════════════════
def main() raises:
    var args = argv()
    if len(args) < 3:
        print("usage: minimax_h3_t2va <prompt> <out_dir> [steps=30] [seed=0]")
        print(
            "  compiled geometry:", WIDTH, "x", HEIGHT, ",", FRAMES, "frames,",
            "text_tokens=", TEXT_TOKENS, ", S=", SEQ_LEN,
        )
        return

    var prompt = String(args[1])
    var out_dir = String(args[2])
    var steps = DEFAULT_STEPS
    if len(args) >= 4:
        steps = atol(String(args[3]))
    var seed = UInt64(DEFAULT_SEED)
    if len(args) >= 5:
        seed = UInt64(atol(String(args[4])))

    if steps < 2:
        raise Error("minimax_h3_t2va: steps must be >= 2 (the schedule needs >= 2 sigmas)")

    print("=== MiniMax-H3 t2va ===")
    print("  prompt:", prompt)
    print(
        "  geometry:", WIDTH, "x", HEIGHT, ",", FRAMES, "frames -> latent [",
        NUM_LATENT_FRAMES, ",", LATENT_H, ",", LATENT_W, "], audio_latents=",
        NUM_AUDIO_LATENTS, ", text_tokens=", TEXT_TOKENS, ", S=", SEQ_LEN,
    )
    print("  steps=", steps, " seed=", seed)

    # ── PREFLIGHT (before DeviceContext) ──────────────────────────────────
    var t_preflight0 = perf_counter_ns()
    _preflight_geometry()
    var config = minimax_h3_released_config()
    config.validate()
    print("  preflight: opening transformer shards:", String(TRANSFORMER_DIR))
    var transformer_shards = ShardedSafeTensors.open(String(TRANSFORMER_DIR))
    print(
        "  preflight: ", transformer_shards.num_shards(), "shard(s), ",
        transformer_shards.num_tensors(), "tensors",
    )
    minimax_h3_check_modcache_weights(transformer_shards, config)
    _preflight_block_tensors(transformer_shards, config)
    _preflight_frontend_tensors(transformer_shards, config)
    print("  preflight: transformer OK (adaLN + ", config.num_layers, "blocks + frontend)")

    print("  preflight: opening text_encoder shards:", String(TEXT_ENCODER_DIR))
    var text_encoder_shards = ShardedSafeTensors.open(String(TEXT_ENCODER_DIR))
    print(
        "  preflight: ", text_encoder_shards.num_shards(), "shard(s), ",
        text_encoder_shards.num_tensors(), "tensors",
    )
    var t_preflight1 = perf_counter_ns()
    print("  preflight OK (", Float64(t_preflight1 - t_preflight0) / 1.0e6, "ms)")

    _ = sys_system(String("mkdir -p '") + out_dir + "'")

    var ctx = DeviceContext()

    # ── 1. Conditioning ────────────────────────────────────────────────────
    var t_cond0 = perf_counter_ns()
    var cond = minimax_h3_encode_conditioning(
        String(PROCESSOR_DIR), String(TEXT_ENCODER_DIR), prompt, ctx
    )
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
        "  conditioning: ", TEXT_TOKENS, " tokens (",
        Float64(t_cond1 - t_cond0) / 1.0e9, "s)",
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
    var modcache = minimax_h3_build_modulation_cache(transformer_shards, temb, config, ctx)
    ctx.synchronize()
    var t_mod1 = perf_counter_ns()
    print(
        "  modcache: ", distinct_timesteps, " rows, ",
        Float64(modcache.total_bytes()) / (1024.0 * 1024.0), " MiB (",
        Float64(t_mod1 - t_mod0) / 1.0e9, "s)",
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
        # `embed` is kept alive and its `.hidden` field is REASSIGNED across
        # the 50-block loop in place, rather than moved out into a separate
        # `var` — MiniMaxH3FrontendEmbed has two Tensor fields (hidden, temb)
        # and Mojo does not allow destroying a struct whose field was
        # partially moved out from under it; field reassignment (drop old
        # value, assign new) has no such restriction. `.temb` is never read
        # (the precomputed modcache is the AdaLN source of truth here, not a
        # per-step recompute) and drops with `embed` at the end of the step.
        var embed = minimax_h3_frontend_embed[TEXT_TOKENS, H3_HEADS, H3_HEAD_DIM](
            video_state, audio_state, text_rows, placeholder_ts,
            geometry.video_indices, geometry.audio_indices, geometry.text_indices,
            SEQ_LEN, frontend_w, config, ctx,
        )

        for layer in range(config.num_layers):
            var block_w = minimax_h3_load_block_device(transformer_shards, layer, config, ctx)
            embed.hidden = minimax_h3_block_forward[SEQ_LEN, H3_HEADS, H3_HEAD_DIM](
                embed.hidden, block_w, layer, config, modcache.block_mod[layer][],
                block_adaln_indices, rope[0], rope[1], rotary_dim, ctx,
            )
            # `block_w` drops here: this layer's ~0.77 GiB bf16 is freed
            # before the next layer's load — resident footprint stays at one
            # block, never the full 61.73 GiB stack.

        var frontend_out = minimax_h3_final_layer(
            embed.hidden, modcache.final_mod[], global_row,
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

    var t_denoise1 = perf_counter_ns()
    print("  denoise done (", Float64(t_denoise1 - t_denoise0) / 1.0e9, "s)")

    # ── 7. Decode (STUBBED — see file header) ──────────────────────────────
    var result_body = String("{\n")
    result_body += String("  \"prompt\":\"") + json_escape(prompt) + String("\",\n")
    result_body += String("  \"steps\":") + String(num_steps) + String(",\n")
    result_body += String("  \"seed\":") + String(seed) + String(",\n")
    result_body += String("  \"width\":") + String(WIDTH) + String(",\n")
    result_body += String("  \"height\":") + String(HEIGHT) + String(",\n")
    result_body += String("  \"frames\":") + String(FRAMES) + String(",\n")
    result_body += String("  \"sequence_length\":") + String(SEQ_LEN) + String(",\n")
    result_body += String("  \"timings_ms\":{\n")
    result_body += String("    \"preflight\":") + String(Float64(t_preflight1 - t_preflight0) / 1.0e6) + String(",\n")
    result_body += String("    \"conditioning\":") + String(Float64(t_cond1 - t_cond0) / 1.0e6) + String(",\n")
    result_body += String("    \"modcache\":") + String(Float64(t_mod1 - t_mod0) / 1.0e6) + String(",\n")
    result_body += String("    \"denoise\":") + String(Float64(t_denoise1 - t_denoise0) / 1.0e6) + String("\n")
    result_body += String("  },\n")
    result_body += String("  \"decode\":\"not_wired\"\n")
    result_body += String("}\n")
    write_text_file(out_dir + String("/result.json"), result_body)
    print("  wrote", out_dir + String("/result.json"))

    _minimax_h3_decode_stub(video_state, audio_state)

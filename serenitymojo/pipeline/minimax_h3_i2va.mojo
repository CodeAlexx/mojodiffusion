# serenitymojo/pipeline/minimax_h3_i2va.mojo — MiniMax-H3 KEYFRAME-conditioned
# video+audio generation: I2VA (first frame), FL2VA (first + last), L2VA (last).
# Pure Mojo.
#
# A SEPARATE FILE FROM minimax_h3_t2va.mojo, deliberately. t2va is live — the
# overnight shot batch builds its per-shot binaries from that source on demand —
# and every keyframe change lands in the parts of a request that file has no
# concept of: a condition-row prefix on the video state, a THIRD distinct
# timestep per step, a canvas the keyframe decides, and a decode that drops rows.
# Threading all of that through t2va as flags would put a live batch one typo
# away from a different render. The cost is the duplicated frontend-weight
# loader and audio-decode tables below, which are copied VERBATIM and marked.
#
# ── WHAT A KEYFRAME REQUEST ADDS TO t2va, IN ORDER ──────────────────────────
#  1. CANVAS. `resolve_canvas_size(*keyframes[0].size)` — the FIRST keyframe's
#     aspect ratio decides the render geometry (before_encoder.py:172-173), so
#     the canvas is an OUTPUT of the request, not an input. This binary's
#     geometry is comptime, so it RESOLVES the canvas and then refuses to run if
#     it does not match what was compiled, with a rebuild hint — the same
#     discipline t2va applies to its prompt length.
#  2. PRESENTATION. `"<Picture i>: "` plus a vision block per keyframe, prepended
#     to the prompt, with the vision rows tagged VIDEO not TEXT
#     (encoders.py:152-169). Those rows change the text length, which shifts the
#     whole media rotary clock.
#  3. CONDITION ROWS. Keyframe -> pixel norm -> `_encode_clip` -> posterior
#     SAMPLE at the fixed seed 42 -> fp16 round -> normalize -> patchify -> mix
#     with the request's own noise at t = 0.999. Prepended to the video rows.
#  4. LAYOUT. `keyframe_anchors` puts each condition block at one end of the
#     rotary timeline. Gated in pipeline/parity/minimax_h3_keyframe_layout_probe.
#  5. TIMESTEPS. Condition rows sit at `max(video_t, 0.999)` — a THIRD distinct
#     timestep per step, so the AdaLN modulation cache is `3 * steps` rows, not
#     t2va's `2 * steps`.
#  6. DENOISE. The scheduler writes ONLY the target rows. The conditioning rows
#     are re-imposed by construction, never updated (denoise.py:184-187).
#  7. DECODE. The first `num_condition_video_rows` rows are DROPPED before
#     unpatchify (decoders.py:95-96); they are anchors, not output.
#
# ── SEAM: THE CONDITIONER CANNOT SEE THE PICTURE YET ────────────────────────
# STATUS 2026-08-03. The keyframe's `<Picture i>` vision block needs three
# things. TWO NOW EXIST; the seam is what is left, and it is named precisely so
# nobody re-ports something that is already here:
#
#   BUILT  the presentation — `pipeline/minimax_h3_keyframe_presentation.mojo`,
#          gated bit-exact against the real tokenizer and image processor over
#          20 canvas/keyframe/prompt cases: token ids, the VIDEO-tagged vision
#          rows, the TEXT_TOKENS budget, and the `<|image_pad|>` POSITIONS.
#   BUILT  the vision tower — `models/text_encoder/minimax_h3_qwen3vl_vision.mojo`
#          h3-ref2va's, geometry AND weighted forward in ONE module, gated on
#          the real FL2VA weights. This file CONSUMES it:
#          `minimax_h3_vision_forward` -> `MiniMaxH3VisionOutput`.
#          DO NOT PORT ANOTHER TOWER, and do NOT reach for
#          `lingbot_qwen3vl_vision.mojo` — a DIFFERENT model's tower (depth 18 /
#          hidden 1024 / deepstack [5,11,17] / out 2560). TWO earlier revisions
#          of this header were wrong here: one named lingbot as the only option,
#          and one pointed at a second forward file that no longer exists (a
#          duplicate, removed in favour of the in-module one).
#   BUILT  the image preprocessor — `pipeline/minimax_h3_vision_preprocess.mojo`,
#          gated BIT-EXACT against the real processor's own `pixel_values` over
#          five real canvases (23M values). It has no resampler and refuses a
#          size `smart_resize` would change: every real keyframe canvas is
#          identity there (768 short edge => >= 589,824 px, far above the
#          65,536 min_pixels), so the resampler is unreachable from a real
#          request. That refusal is gated too.
#   WAS-SEAM (a) the Qwen3-VL IMAGE PREPROCESSOR — pixels to the tower's
#              `[num_patches, 1536]` patch rows (smart_resize, the processor's
#              OWN mean=std=0.5 normalization — NOT the video VAE's ImageNet
#              constants — and the patch flattening). `models/minimax_h3/
#              image_grid.mojo` resolves the GEOMETRY and says in its own header
#              that the resampling is a separate unit. It still is.
#          (b) the conditioner's vision splice + DEEPSTACK INJECTION —
#              h3-ref2va owns this in `minimax_h3_qwen3vl_streamed.mojo`: the
#              tower's embeds substituted at the pad positions this file already
#              computes, and the three deepstack tensors added at LANGUAGE
#              decoder layers 0/1/2 at visual positions only. The interface is
#              `MiniMaxH3VisionOutput`.
#
#   -D H3_KF_NO_VISION=1 remains an EXPLICIT degraded mode that drops the vision
#   block entirely. The keyframe still conditions through its VAE condition rows
#   — the strong path, and the one the denoise loop anchors on — but the
#   conditioner never sees it and the §2.1 line's `<Picture 1>` refers to
#   nothing. THIS IS NOT THE RELEASED MODEL'S CONDITIONING, and it says so in
#   result.json.
#
# ── ARGUMENTS ───────────────────────────────────────────────────────────────
#   minimax_h3_i2va i2va  <prompt> <image>              <out_dir> [steps] [seed] [max_blocks]
#   minimax_h3_i2va l2va  <prompt> <last_image>         <out_dir> [steps] [seed] [max_blocks]
#   minimax_h3_i2va fl2va <prompt> <image> <last_image> <out_dir> [steps] [seed] [max_blocks]
#
# The mode is explicit rather than inferred from the argument count: `<prompt>
# <image> [<last_image>] <out_dir>` cannot be parsed unambiguously, and guessing
# wrong silently swaps I2VA for L2VA — two different tasks with the same file
# count. `<prompt>` is the body text; the §2.1 alignment instruction for the mode
# is PREPENDED automatically (pipeline/minimax_h3_ref_prompt.mojo, gated against
# the vendor guide), so a caller passes prose, not a formatted prompt.
#
# LINKER: needs the cuDNN SDPA shim, like t2va:
#   -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa -Xlinker -lm
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
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import sys_system
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.ops.tensor_algebra import reshape, permute, slice, add, mul, concat
from serenitymojo.serve.product_manifest import json_escape, json_bool, write_text_file
from serenitymojo.audio.wav import save_wav

from serenitymojo.models.dit.minimax_h3_dit import (
    MiniMaxH3DiTConfig,
    MINIMAX_H3_ATTN_CUDNN,
    MINIMAX_H3_ATTN_SAGE_INT8,
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
    minimax_h3_check_modcache_weights,
    minimax_h3_build_modulation_cache,
)
from serenitymojo.models.dit.minimax_h3_rope import build_minimax_h3_rope_tables
from serenitymojo.models.dit.minimax_h3_frontend import (
    minimax_h3_frontend_embed,
    minimax_h3_final_layer,
    minimax_h3_timestep_embedding,
)
from serenitymojo.models.dit.minimax_h3_sampling import (
    MiniMaxH3DualSchedule,
    minimax_h3_build_sampling_geometry,
)
from serenitymojo.models.text_encoder.minimax_h3_qwen3vl_streamed import (
    minimax_h3_encode_conditioning_streamed,
)
from serenitymojo.models.text_encoder.minimax_h3_conditioning import (
    MiniMaxH3ConditioningOutput,
    minimax_h3_encode_conditioning,
)
from serenitymojo.models.minimax_h3.packing import (
    minimax_h3_align_num_frames,
    minimax_h3_resolve_canvas_size,
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
from serenitymojo.models.vae.minimax_h3_video_encoder_device import (
    MiniMaxH3VideoEncoderDevice,
    minimax_h3_video_released_encoder_config,
)
from serenitymojo.models.vae.minimax_h3_video_decoder_device import (
    minimax_h3_video_released_decoder_config,
    minimax_h3_video_decoder_device_load,
)
from serenitymojo.pipeline.minimax_h3_keyframe_encode import (
    minimax_h3_keyframe_condition_rows,
    minimax_h3_keyframe_encode_device,
    minimax_h3_target_audio_rows,
    minimax_h3_target_latent_rows,
)
from serenitymojo.pipeline.minimax_h3_keyframe_image import (
    MiniMaxH3RgbImage,
    minimax_h3_exif_transpose,
    minimax_h3_keyframe_anchors,
    minimax_h3_prepare_keyframes,
)
from serenitymojo.pipeline.minimax_h3_media_in import (
    minimax_h3_ffmpeg_read_image,
    minimax_h3_jpeg_exif_orientation,
)
from serenitymojo.models.text_encoder.minimax_h3_qwen3vl_vision import (
    MiniMaxH3VisionGrid,
    MiniMaxH3VisionOutput,
    minimax_h3_vision_forward,
    minimax_h3_vision_load_weights,
)
from serenitymojo.models.minimax_h3_device.vision_tower_device import (
    minimax_h3_vision_device_weights,
    minimax_h3_vision_forward_device,
)
from serenitymojo.pipeline.minimax_h3_vision_preprocess import (
    minimax_h3_vision_patch_rows,
)
from serenitymojo.pipeline.gpu_free_vram_guard import require_free_vram
from serenitymojo.pipeline.minimax_h3_keyframe_presentation import (
    MiniMaxH3KeyframePresentation,
    minimax_h3_keyframe_presentation,
)
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer
from serenitymojo.pipeline.minimax_h3_ref_prompt import (
    MINIMAX_H3_TASK_FL2VA,
    MINIMAX_H3_TASK_I2VA,
    MINIMAX_H3_TASK_L2VA,
    minimax_h3_alignment_instruction,
    minimax_h3_task_from_keyframes,
)
from serenitymojo.pipeline.minimax_h3_torch_cpu_rng import MiniMaxH3TorchCpuGenerator
from serenitymojo.pipeline.minimax_h3_video_vae_spatial_tiling import (
    minimax_h3_video_released_tiling_config,
    minimax_h3_video_tiled_decode,
)
from serenitymojo.pipeline.minimax_h3_video_vae_temporal import (
    MiniMaxH3TemporalDecodeStream,
    minimax_h3_video_released_temporal_config,
)
from serenitymojo.pipeline.minimax_h3_video_vae_pixel_norm import (
    minimax_h3_video_pixel_denormalize,
    minimax_h3_pixel_norm_constants,
)


# ── Checkpoint paths (same layout every H3 module defaults to) ──────────────
comptime H3_ROOT = "/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA"
comptime TRANSFORMER_DIR = H3_ROOT + "/transformer"
comptime TEXT_ENCODER_DIR = H3_ROOT + "/text_encoder"
comptime PROCESSOR_DIR = H3_ROOT + "/processor"
comptime AUDIO_VAE_PATH = H3_ROOT + "/audio_vae/model.safetensors"
comptime VIDEO_VAE_DIR = H3_ROOT + "/video_vae/source"
comptime AUDIO_SAMPLE_RATE = 32000

# ── Geometry, comptime (rebuild to change) ──────────────────────────────────
# Unlike t2va these are CHECKED against what the keyframe implies, not merely
# declared: the canvas belongs to the keyframe.
comptime HEIGHT = get_defined_int["H3_HEIGHT", 480]()
comptime WIDTH = get_defined_int["H3_WIDTH", 832]()
comptime FRAMES = get_defined_int["H3_FRAMES", 22]()
comptime TEXT_TOKENS = get_defined_int["H3_TEXT_TOKENS", 32]()

comptime LATENT_H = HEIGHT // 16
comptime LATENT_W = WIDTH // 16
comptime PATCH_H = 2
comptime PATCH_W = 2
comptime ROWS_PER_FRAME = (LATENT_H // PATCH_H) * (LATENT_W // PATCH_W)
comptime NUM_LATENT_FRAMES = (FRAMES - 5) // 17 * 5 + 2
comptime NUM_AUDIO_LATENTS = Int(round(Float64(FRAMES) / 24.0 * 40.0))
comptime NUM_TARGET_VIDEO_ROWS = NUM_LATENT_FRAMES * ROWS_PER_FRAME
comptime NUM_AUDIO_ROWS = NUM_AUDIO_LATENTS * 2

# How many keyframes this binary is compiled for. The condition-row count enters
# the sequence length, which is comptime everywhere downstream (the attention's
# S), so it cannot be a runtime choice.
comptime KEYFRAMES = get_defined_int["H3_KEYFRAMES", 1]()
comptime NUM_CONDITION_ROWS = KEYFRAMES * ROWS_PER_FRAME
comptime NUM_VIDEO_ROWS = NUM_CONDITION_ROWS + NUM_TARGET_VIDEO_ROWS
comptime SEQ_LEN = TEXT_TOKENS + NUM_CONDITION_ROWS + NUM_AUDIO_ROWS + NUM_TARGET_VIDEO_ROWS

comptime H3_HEADS = 56
comptime H3_HEAD_DIM = 128
comptime LATENT_TILE = 16
comptime VAE_CLIP_LENGTH = 17
comptime VAE_RATIO_T = 4
comptime VAE_TOKEN_DROP = 3
comptime VAE_TOKENS_CHUNK = (VAE_CLIP_LENGTH + VAE_RATIO_T - 1) // VAE_RATIO_T
comptime VAE_TOKEN_OVERLAP = (
    VAE_TOKENS_CHUNK - VAE_TOKEN_DROP % VAE_TOKENS_CHUNK
) % VAE_TOKENS_CHUNK
comptime VAE_TOKENS_PER_CLIP = VAE_TOKENS_CHUNK + VAE_TOKEN_OVERLAP

comptime DEFAULT_STEPS = 30
comptime DEFAULT_SEED = 0

# See this file's header, "SEAM: THE CONDITIONER CANNOT SEE THE PICTURE YET".
comptime NO_VISION = get_defined_int["H3_KF_NO_VISION", 0]()

comptime AUDIO_LATENT_DIM = 2048
comptime AUDIO_DECODER_DIM = 1024


# ═════════════════════════════════════════════════════════════════════════════
# Keyframe ingest — host only, runs BEFORE any DeviceContext.
# ═════════════════════════════════════════════════════════════════════════════
def _read_keyframe(path: String, scratch_dir: String, tag: String) raises -> MiniMaxH3RgbImage:
    """Decode one keyframe and apply its EXIF orientation, in that order — the
    vendor's `ImageOps.exif_transpose(...).convert("RGB")` (before_encoder.py:163).

    The orientation has to be resolved HERE, before the canvas is resolved from
    this image's size: a quarter turn swaps the aspect ratio, and the canvas is
    derived from it."""
    var frames = minimax_h3_ffmpeg_read_image(
        path,
        scratch_dir + "/keyframe_" + tag + ".rgb",
        scratch_dir + "/keyframe_" + tag + ".probe",
    )
    var raw = MiniMaxH3RgbImage(frames.pixels.copy(), frames.height, frames.width)
    var orientation = minimax_h3_jpeg_exif_orientation(path)
    if orientation > 1:
        print(
            "  keyframe", tag, ": EXIF orientation", orientation,
            "applied (", raw.width, "x", raw.height, "->",
            (raw.height if orientation >= 5 else raw.width), "x",
            (raw.width if orientation >= 5 else raw.height), ")",
        )
    return minimax_h3_exif_transpose(raw, orientation)


def _check_canvas(keyframe: MiniMaxH3RgbImage) raises:
    """`resolve_canvas_size(*keyframes[0].size)` vs what this binary compiled.

    PIL's `.size` is `(width, height)`, and `resolve_canvas_size(aspect_width,
    aspect_height)` takes them in that order — swapping them silently produces
    the transposed canvas, which is a legal canvas and therefore invisible."""
    var canvas = minimax_h3_resolve_canvas_size(
        Float64(keyframe.width), Float64(keyframe.height)
    )
    if canvas.height != HEIGHT or canvas.width != WIDTH:
        raise Error(
            String("minimax_h3_i2va: the first keyframe is ")
            + String(keyframe.width) + "x" + String(keyframe.height)
            + ", whose MiniMax-H3 canvas is " + String(canvas.width) + "x"
            + String(canvas.height) + ", but this binary was compiled for "
            + String(WIDTH) + "x" + String(HEIGHT) + ". The canvas belongs to"
            " the keyframe (before_encoder.py:172-173), so rebuild with"
            " -D H3_HEIGHT=" + String(canvas.height) + " -D H3_WIDTH="
            + String(canvas.width) + " (or pre-resize the keyframe)."
        )


def _preflight_geometry() raises:
    if (FRAMES - 5) % 17 != 0:
        raise Error(
            String("minimax_h3_i2va: FRAMES=") + String(FRAMES)
            + " is not of the form 17n+5 (the video VAE's chunk size); rebuild"
            " with an aligned H3_FRAMES — 22, 39, 56, ..., 124, ..."
        )
    if HEIGHT % 32 != 0 or WIDTH % 32 != 0 or HEIGHT <= 0 or WIDTH <= 0:
        raise Error(
            "minimax_h3_i2va: H3_HEIGHT/H3_WIDTH must be positive multiples of 32"
        )
    if KEYFRAMES < 1 or KEYFRAMES > 2:
        raise Error(
            String("minimax_h3_i2va: H3_KEYFRAMES must be 1 (I2VA/L2VA) or 2"
                   " (FL2VA), got ") + String(KEYFRAMES)
        )
    if TEXT_TOKENS <= 0:
        raise Error("minimax_h3_i2va: H3_TEXT_TOKENS must be positive")


def _preflight_block_tensors(
    shards: ShardedSafeTensors, config: MiniMaxH3DiTConfig
) raises:
    """Index-only check of every main-stack block tensor. Verbatim from
    minimax_h3_t2va.mojo — see this file's header on the duplication."""
    for layer in range(config.num_layers):
        var names = minimax_h3_block_tensor_names(layer)
        for i in range(len(names)):
            ref name = names[i]
            if not shards.has_tensor(name):
                raise Error(
                    String("minimax_h3_i2va preflight: missing block tensor ") + name
                )
            var info = shards.tensor_info(name)
            var want = minimax_h3_expected_shape(name, config)
            if len(info.shape) != len(want):
                raise Error(
                    String("minimax_h3_i2va preflight: rank mismatch for ") + name
                )
            for d in range(len(want)):
                if info.shape[d] != want[d]:
                    raise Error(
                        String("minimax_h3_i2va preflight: shape mismatch for ") + name
                    )
            if info.dtype != STDtype.BF16:
                raise Error(
                    String("minimax_h3_i2va preflight: ") + name + " is not BF16"
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


def _load_frontend_weights(
    shards: ShardedSafeTensors, config: MiniMaxH3DiTConfig, ctx: DeviceContext
) raises -> Dict[String, ArcPointer[Tensor]]:
    """Verbatim from minimax_h3_t2va.mojo::_minimax_h3_load_frontend_weights."""
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
# AdaLN row addressing — THREE rows per step, not t2va's two.
# ═════════════════════════════════════════════════════════════════════════════
def _kf_global_timestep_row(
    token_tags: List[Int],
    video_indices: List[Int],
    num_condition_rows: Int,
    step: Int,
    sequence_length: Int,
) raises -> List[Int]:
    """Row `r`'s index into the GLOBAL `3 * num_steps`-row modulation cache.

    `3 * step` video and text (text rows inherit the video timestep),
    `3 * step + 1` audio, `3 * step + 2` the keyframe CONDITION rows.

    The condition rows cannot be found from `token_tags` — they are tagged VIDEO
    like every other video row, which is the whole point of the tag (it selects
    the AdaLN modality table, and a keyframe latent IS video). They are
    identified positionally instead, as the first `num_condition_rows` entries
    of `video_indices` — the same contract `build_row_timesteps` (packing.py:495)
    and the scheduler step (denoise.py:263) both use."""
    var out = List[Int](capacity=sequence_length)
    for i in range(sequence_length):
        var tag = token_tags[i]
        if tag == 2:
            out.append(3 * step + 1)
        else:
            out.append(3 * step)
    for i in range(num_condition_rows):
        out[video_indices[i]] = 3 * step + 2
    return out^


# ═════════════════════════════════════════════════════════════════════════════
# Audio decode — verbatim from minimax_h3_t2va.mojo (see header).
# ═════════════════════════════════════════════════════════════════════════════
def _h3_audio_decoder_rates() -> List[Int]:
    return [5, 5, 2, 2, 2, 2, 2]


def _h3_audio_decoder_kernels() -> List[Int]:
    return [9, 9, 4, 4, 4, 4, 4]


def _h3_audio_resblock_kernels() -> List[Int]:
    return [3, 7, 11]


def _h3_audio_resblock_dilations() -> List[List[Int]]:
    var d: List[Int] = [1, 3, 5]
    return [d.copy(), d.copy(), d.copy()]


def _audio_decoder_config(latent_channels: Int) -> MiniMaxH3AudioDecoderConfig:
    return MiniMaxH3AudioDecoderConfig(
        latent_channels,
        AUDIO_LATENT_DIM,
        AUDIO_DECODER_DIM,
        _h3_audio_decoder_rates(),
        _h3_audio_decoder_kernels(),
        _h3_audio_resblock_kernels(),
        _h3_audio_resblock_dilations(),
    )


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


def _load_audio_vae_weights(path: String) raises -> MiniMaxH3AudioWeights:
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
            raise Error(String("minimax_h3_i2va: audio_vae tensor ") + n + " is not F32")
        var p = tv.data.unsafe_ptr().bitcast[Float32]()
        var v = List[Float32](capacity=tv.numel())
        for j in range(tv.numel()):
            v.append(p[j])
        names.append(String(n))
        values.append(v^)
    return MiniMaxH3AudioWeights(names^, values^)


def _denormalize_audio_channel(
    latent: List[Float32], channels: Int, num_latents: Int,
    mean: List[Float32], std: List[Float32],
) raises -> List[Float32]:
    if len(latent) != channels * num_latents:
        raise Error("minimax_h3_i2va: audio latent channel/length mismatch")
    var out = List[Float32](capacity=len(latent))
    for c in range(channels):
        var m = mean[c]
        var s = std[c]
        for t in range(num_latents):
            out.append(latent[c * num_latents + t] * s + m)
    return out^


def _decode_audio(
    audio_state: Tensor, num_audio_latents: Int, audio_channels: Int,
    out_dir: String, ctx: DeviceContext,
) raises -> Int:
    var rows_host = audio_state.to_host(ctx)
    var latents_2ct = minimax_h3_unpack_audio(rows_host, num_audio_latents, audio_channels)
    var block = audio_channels * num_audio_latents
    var ch0 = List[Float32](capacity=block)
    var ch1 = List[Float32](capacity=block)
    for i in range(block):
        ch0.append(latents_2ct[i])
        ch1.append(latents_2ct[block + i])
    var mean = _h3_audio_latents_mean()
    var std = _h3_audio_latents_std()
    var d0 = _denormalize_audio_channel(ch0, audio_channels, num_audio_latents, mean, std)
    var d1 = _denormalize_audio_channel(ch1, audio_channels, num_audio_latents, mean, std)
    var dec_cfg = _audio_decoder_config(audio_channels)
    var weights = _load_audio_vae_weights(String(AUDIO_VAE_PATH))
    # Device BigVGAN (gate: models/minimax_h3_device/parity/, 11/11 vs host
    # oracle, e2e waveform cos 0.999999999994677 / max_abs 5.7e-6).
    var dev_weights = minimax_h3_audio_device_weights(weights, dec_cfg, ctx)
    var wave_l = minimax_h3_audio_decode_device(dev_weights, dec_cfg, d0, num_audio_latents, ctx)
    var wave_r = minimax_h3_audio_decode_device(dev_weights, dec_cfg, d1, num_audio_latents, ctx)
    if len(wave_l) != len(wave_r):
        raise Error("minimax_h3_i2va: L/R waveform length mismatch")
    var stereo = List[Float32](capacity=2 * len(wave_l))
    for i in range(len(wave_l)):
        stereo.append(wave_l[i])
    for i in range(len(wave_r)):
        stereo.append(wave_r[i])
    var stereo_shape: List[Int] = [2, len(wave_l)]
    var stereo_tensor = Tensor.from_host(stereo, stereo_shape^, STDtype.F32, ctx)
    save_wav(stereo_tensor, out_dir + String("/audio.wav"), AUDIO_SAMPLE_RATE, ctx)
    print("  wrote", out_dir + "/audio.wav", "(", len(wave_l), "samples/channel )")
    return len(wave_l)


# ═════════════════════════════════════════════════════════════════════════════
# Video decode — the condition rows are DROPPED first (decoders.py:95-96).
# ═════════════════════════════════════════════════════════════════════════════
def _write_rgb_frames(
    rgb: Tensor, out_dir: String, first_index: Int, ctx: DeviceContext
) raises -> Int:
    var ps = rgb.shape()
    var frames = ps[1]
    var height = ps[2]
    var width = ps[3]
    for f in range(frames):
        var one = slice(rgb, 1, f, 1, ctx)
        var hwc = reshape(one, [height, width, 3], ctx)
        var chw = permute(hwc, [2, 0, 1], ctx)
        var img = reshape(chw, [1, 3, height, width], ctx)
        var name = String(first_index + f)
        while len(name) < 5:
            name = String("0") + name
        save_png(img, out_dir + "/frame_" + name + ".png", ctx, ValueRange.UNIT)
    return frames


def _video_latents_mean() -> List[Float32]:
    return [
        Float32(0.858090341091156), Float32(-0.9606591463088989),
        Float32(1.0661640167236328), Float32(-0.5090325474739075),
        Float32(-0.2727581858634949), Float32(-1.3675414323806763),
        Float32(-0.2553254961967468), Float32(-0.26907554268836975),
        Float32(-0.5376840829849243), Float32(-0.0464097298681736),
        Float32(0.6657370328903198), Float32(0.19690127670764923),
        Float32(-0.5460608005523682), Float32(-0.4035342037677765),
        Float32(-0.23683024942874908), Float32(0.25928452610969543),
        Float32(-0.30133944749832153), Float32(0.211341992020607),
        Float32(-1.1206848621368408), Float32(0.3581933379173279),
        Float32(-0.04225143790245056), Float32(0.2604829967021942),
        Float32(0.22864092886447906), Float32(0.7056031823158264),
    ]


def _video_latents_std() -> List[Float32]:
    return [
        Float32(1.2223774194717407), Float32(1.2767263650894165),
        Float32(1.6831774711608887), Float32(1.7549455165863037),
        Float32(1.5636216402053833), Float32(2.194143533706665),
        Float32(0.9653137922286987), Float32(1.0569885969161987),
        Float32(0.841948926448822), Float32(0.7729952931404114),
        Float32(1.8955937623977661), Float32(0.946841835975647),
        Float32(0.7996809482574463), Float32(0.44988900423049927),
        Float32(0.7197399735450745), Float32(0.6936293244361877),
        Float32(2.961095094680786), Float32(2.7694199085235596),
        Float32(3.0496184825897217), Float32(2.1088054180145264),
        Float32(3.276226282119751), Float32(3.1627357006073),
        Float32(2.2816812992095947), Float32(2.6127843856811523),
    ]


def _decode_video(
    video_state: Tensor,    # [NUM_VIDEO_ROWS, patch_dim], CONDITION ROWS FIRST
    latent_channels: Int,
    out_dir: String,
    ctx: DeviceContext,
) raises -> Int:
    """Drop the conditioning rows, then the t2va decode tail unchanged.

    The drop is the first thing that happens, before unpatchify — those rows
    hold a keyframe, not a generated frame, and unpatchifying them along with
    the rest would shift every frame by one and produce a video one latent frame
    too long."""
    var target = slice(video_state, 0, NUM_CONDITION_ROWS, NUM_TARGET_VIDEO_ROWS, ctx)

    # Row order: H3 packs each video row CHANNEL-SLOWEST; ops/patchify3d reads
    # within-patch channel-FASTEST. Reorder per row, then unpatchify.
    var rows5 = reshape(
        target, [NUM_TARGET_VIDEO_ROWS, latent_channels, 1, 2, 2], ctx
    )
    var rows5p = permute(rows5, [0, 2, 3, 4, 1], ctx)
    var rows_cf = reshape(
        rows5p, [NUM_TARGET_VIDEO_ROWS, latent_channels * 4], ctx
    )
    var grid_cfhw = unpatchify3d(
        rows_cf, latent_channels, NUM_LATENT_FRAMES, LATENT_H, LATENT_W, 1, 2, 2, ctx
    )
    var perm = [1, 2, 3, 0]
    var grid_fhwc = permute(grid_cfhw, perm^, ctx)
    var latents = reshape(
        grid_fhwc, [1, NUM_LATENT_FRAMES, LATENT_H, LATENT_W, latent_channels], ctx
    )

    var lat_mean_t = Tensor.from_host(_video_latents_mean(), [24], latents.dtype(), ctx)
    var lat_std_t = Tensor.from_host(_video_latents_std(), [24], latents.dtype(), ctx)
    latents = add(mul(latents, lat_std_t, ctx), lat_mean_t, ctx)

    var dcfg = minimax_h3_video_released_decoder_config()
    var decoder = minimax_h3_video_decoder_device_load(String(VIDEO_VAE_DIR), dcfg, ctx)
    var tiling = minimax_h3_video_released_tiling_config()
    var stcfg = minimax_h3_video_released_temporal_config()
    var stream = MiniMaxH3TemporalDecodeStream(latents, stcfg, tiling, ctx)
    var planned = stream.output_frames()
    var norm = minimax_h3_pixel_norm_constants(String("imagenet"))
    var written = 0
    while stream.has_next():
        var part = stream.next_part[
            LATENT_TILE, LATENT_TILE, 32, 64, 5, VAE_TOKENS_PER_CLIP
        ](decoder, ctx)
        if part:
            var rgb_part = minimax_h3_video_pixel_denormalize(part.value(), norm, ctx)
            written += _write_rgb_frames(rgb_part, out_dir, written, ctx)
    stream.finish()
    if written != planned:
        raise Error(
            String("minimax_h3_i2va: streaming decode wrote ") + String(written)
            + " frames but planned " + String(planned)
        )
    print("  wrote", written, "frames to", out_dir)
    return written


# ═════════════════════════════════════════════════════════════════════════════
# main
# ═════════════════════════════════════════════════════════════════════════════
def _usage():
    print("usage:")
    print("  minimax_h3_i2va i2va  <prompt> <image>              <out_dir> [steps] [seed] [max_blocks]")
    print("  minimax_h3_i2va l2va  <prompt> <last_image>         <out_dir> [steps] [seed] [max_blocks]")
    print("  minimax_h3_i2va fl2va <prompt> <image> <last_image> <out_dir> [steps] [seed] [max_blocks]")
    print("")
    print(
        "  compiled geometry:", WIDTH, "x", HEIGHT, ",", FRAMES, "frames,",
        "keyframes=", KEYFRAMES, ", text_tokens=", TEXT_TOKENS, ", S=", SEQ_LEN,
    )
    print("  <prompt> is the BODY text; the §2.1 alignment instruction is prepended for you")
    print("  optional flag: --attention-backend=cudnn|sage-int8 (default cudnn)")


def main() raises:
    var raw_args = argv()
    var args = List[String]()
    var attention_backend = MINIMAX_H3_ATTN_CUDNN
    var attention_backend_name = String("cudnn")
    for i in range(len(raw_args)):
        var arg = String(raw_args[i])
        if arg == String("--attention-backend=sage-int8"):
            attention_backend = MINIMAX_H3_ATTN_SAGE_INT8
            attention_backend_name = String("sage-int8")
            continue
        if arg == String("--attention-backend=cudnn"):
            attention_backend = MINIMAX_H3_ATTN_CUDNN
            attention_backend_name = String("cudnn")
            continue
        if arg.startswith("--attention-backend="):
            raise Error(
                String("unknown attention backend flag: ") + arg
                + String(" (expected cudnn or sage-int8)")
            )
        args.append(arg)
    if len(args) < 5:
        _usage()
        return

    var mode = String(args[1])
    var have_first: Bool
    var have_last: Bool
    if mode == String("i2va"):
        have_first = True
        have_last = False
    elif mode == String("l2va"):
        have_first = False
        have_last = True
    elif mode == String("fl2va"):
        have_first = True
        have_last = True
    else:
        _usage()
        raise Error(String("minimax_h3_i2va: unknown mode '") + mode + "'")

    var num_keyframes = 2 if (have_first and have_last) else 1
    if num_keyframes != KEYFRAMES:
        raise Error(
            String("minimax_h3_i2va: mode '") + mode + "' needs "
            + String(num_keyframes) + " keyframe(s), but this binary was"
            " compiled with H3_KEYFRAMES=" + String(KEYFRAMES)
            + " (the condition rows enter the comptime sequence length);"
            " rebuild with -D H3_KEYFRAMES=" + String(num_keyframes)
        )

    var prompt_body = String(args[2])
    var image_paths = List[String]()
    var argi = 3
    for _ in range(num_keyframes):
        if argi >= len(args):
            _usage()
            raise Error("minimax_h3_i2va: not enough keyframe paths for this mode")
        image_paths.append(String(args[argi]))
        argi += 1
    if argi >= len(args):
        _usage()
        raise Error("minimax_h3_i2va: missing <out_dir>")
    var out_dir = String(args[argi])
    argi += 1
    var steps = DEFAULT_STEPS
    if len(args) > argi:
        steps = atol(String(args[argi]))
    var seed = UInt64(DEFAULT_SEED)
    if len(args) > argi + 1:
        seed = UInt64(atol(String(args[argi + 1])))
    var max_blocks = 50
    if len(args) > argi + 2:
        max_blocks = atol(String(args[argi + 2]))
    if steps < 2:
        raise Error("minimax_h3_i2va: steps must be >= 2")

    var task = minimax_h3_task_from_keyframes(have_first, have_last)
    print("=== MiniMax-H3", mode, "(keyframe-conditioned) ===")
    print(
        "  geometry:", WIDTH, "x", HEIGHT, ",", FRAMES, "frames -> latent [",
        NUM_LATENT_FRAMES, ",", LATENT_H, ",", LATENT_W, "], audio_latents=",
        NUM_AUDIO_LATENTS, ", condition_rows=", NUM_CONDITION_ROWS,
        ", text_tokens=", TEXT_TOKENS, ", S=", SEQ_LEN,
    )
    print("  steps=", steps, " seed=", seed, " max_blocks=", max_blocks)
    print("  attention_backend=", attention_backend_name)

    # ── PREFLIGHT, all before DeviceContext ────────────────────────────────
    var t_pre0 = perf_counter_ns()
    _preflight_geometry()
    var aligned = minimax_h3_align_num_frames(FRAMES)
    if aligned != FRAMES:
        raise Error("minimax_h3_i2va: FRAMES is not aligned (checked twice)")

    _ = sys_system(String("mkdir -p '") + out_dir + "'")

    # Keyframes: decode, EXIF-transpose, resolve+check the canvas, put them on it.
    var raw_keyframes = List[MiniMaxH3RgbImage]()
    for i in range(len(image_paths)):
        var tag = String("first") if (i == 0 and have_first) else String("last")
        raw_keyframes.append(_read_keyframe(image_paths[i], out_dir, tag))
        print(
            "  keyframe", i, ":", image_paths[i], "->",
            raw_keyframes[i].width, "x", raw_keyframes[i].height,
        )
    _check_canvas(raw_keyframes[0])
    var keyframes = minimax_h3_prepare_keyframes(
        raw_keyframes, have_first, have_last, HEIGHT, WIDTH
    )
    var anchors = minimax_h3_keyframe_anchors(have_first, have_last)
    print("  keyframes prepared onto", WIDTH, "x", HEIGHT, ", anchors:", len(anchors))

    # The §2.1 alignment instruction, prepended to the body the caller passed.
    # `final_shot_index` is 1 here: without parsing the caller's prose there is
    # no way to know how many shots it declares, and 1 is the guide's own single
    # shot case. A caller with multiple shots should pass a pre-built prompt.
    var instruction = minimax_h3_alignment_instruction(task, FRAMES, 1)
    var prompt = prompt_body
    if instruction != String(""):
        prompt = instruction + "\n\n" + prompt_body
    print("  alignment instruction:", instruction)

    # The instruction embeds the duration, so a dev geometry puts a duration in
    # the prompt that MiniMax-H3 does not generate. A warning, not an error:
    # short geometries are how the rest of the chain gets exercised cheaply, and
    # the vendor's own 5-15 s check (before_encoder.py:94-104) belongs to a real
    # render, not to a plumbing run.
    var duration = Float64(FRAMES) / 24.0
    if duration < 5.0 or duration > 15.0:
        print(
            "  WARNING: this binary's", FRAMES, "frames are", duration,
            "s — MiniMax-H3 generates 5 to 15 s, and the alignment instruction"
            " above now states a duration outside that range.",
        )

    # ── The PRESENTATION: the vision block's rows are part of the text run ──
    # Built through the gated composition module, which resolves the conditioner
    # grid from the CANVAS (the keyframes are already on it) and returns the
    # token ids, the VIDEO-tagged rows, and the `<|image_pad|>` positions the
    # tower's embeds are spliced into.
    var tokenizer = Qwen3Tokenizer(String(PROCESSOR_DIR) + "/tokenizer.json")
    var presentation = minimax_h3_keyframe_presentation(
        tokenizer, prompt, HEIGHT, WIDTH, KEYFRAMES
    )
    print(
        "  presentation:", presentation.vision_tokens_each,
        "vision tokens per keyframe,", presentation.num_text_tokens(),
        "text rows total,", len(presentation.pad_positions), "image_pad rows",
    )
    if presentation.num_text_tokens() != TEXT_TOKENS:
        raise Error(
            String("minimax_h3_i2va: this request's presentation is ")
            + String(presentation.num_text_tokens()) + " text rows (prompt +"
            " a <Picture i> label + 2 + " + String(presentation.vision_tokens_each)
            + " vision rows per keyframe), but this binary was compiled for"
            " H3_TEXT_TOKENS=" + String(TEXT_TOKENS) + " (S is comptime);"
            " rebuild with -D H3_TEXT_TOKENS="
            + String(presentation.num_text_tokens())
        )

    # ── THE VISION PATH, as a CONSUMED interface ────────────────────────────
    # The tower is h3-ref2va's and is not this file's to reimplement. This file
    # is a CONSUMER of `MiniMaxH3VisionOutput`; once the two seams below close,
    # the call is exactly:
    #
    #     var vw  = minimax_h3_vision_load_weights(String(TEXT_ENCODER_DIR))
    #     var vis = minimax_h3_vision_forward(vw, pixel_patches, vision_grids)
    #     # vis.embeds             [tokens, 5120] -> presentation.pad_positions
    #     # vis.deepstack_block(i) [tokens, 5120] -> language layers 0/1/2
    #
    # Both of the tower's inputs that THIS file owns are already resolved: the
    # grids (from the canvas the presentation counted tokens from) and the
    # splice map. The gap is `pixel_patches`.
    var vision_grids = List[MiniMaxH3VisionGrid]()
    for _ in range(KEYFRAMES):
        vision_grids.append(
            MiniMaxH3VisionGrid(1, presentation.grid_h, presentation.grid_w)
        )
    var expected_vision_tokens = 0
    for gi in range(len(vision_grids)):
        expected_vision_tokens += vision_grids[gi].num_tokens()
    # The tower's token count and the presentation's reserved pad rows are
    # computed by different code from the same canvas; if they ever disagree the
    # embeds cannot be spliced, and that must fail here rather than produce a
    # shape-correct, row-shifted conditioning.
    if expected_vision_tokens != len(presentation.pad_positions):
        raise Error(
            String("minimax_h3_i2va: the vision grids imply ")
            + String(expected_vision_tokens) + " tokens but the presentation"
            " reserved " + String(len(presentation.pad_positions))
            + " image_pad rows"
        )
    print(
        "  vision interface: ", len(vision_grids), "grid(s) of 1 x",
        presentation.grid_h, "x", presentation.grid_w, "->",
        expected_vision_tokens, "embeds to splice",
    )

    # ── GPU GUARD + DeviceContext, MOVED ABOVE THE VISION TOWER 2026-08-04:
    # the device tower (12/12 gated, 5931x over host) needs a DeviceContext.
    # The guard stays the LAST thing before the context exists — the invariant
    # the original placement protected — it just happens earlier now. The
    # transformer preflight below remains host-side and order-independent.
    require_free_vram(
        20000, out_dir + "/.gpu_guard", String("H3"),
        String("minimax_h3_i2va (") + mode + ")",
    )
    var ctx = DeviceContext()

    var vision_out = Optional[MiniMaxH3VisionOutput](None)
    comptime
    if NO_VISION == 0:
        # ── (a) preprocess, then the tower.
        var patch_rows = List[Float32]()
        for ki in range(len(keyframes)):
            var rows = minimax_h3_vision_patch_rows(keyframes[ki])
            for i in range(len(rows)):
                patch_rows.append(rows[i])
        print("  preprocessed", len(keyframes), "keyframe(s) ->",
              len(patch_rows) // 1536, "patch rows")

        # DEVICE tower when the geometry matches its gated comptime
        # instantiation (ONE grid, 48x48 = 2304 patches — the square-768
        # canvas every current i2va run uses; gate: models/minimax_h3_device/
        # parity/, 12/12 vs GPU torch, 0.22 s vs 1290 s host). Any other
        # geometry falls back to the gated HOST tower rather than raising —
        # slow is better than dead, and the device module raises loudly for
        # uninstantiated segment lengths by design.
        var use_device_tower = (
            len(vision_grids) == 1
            and vision_grids[0].t == 1
            and vision_grids[0].h == 48
            and vision_grids[0].w == 48
        )
        var vision: MiniMaxH3VisionOutput
        if use_device_tower:
            var vis_dev = minimax_h3_vision_device_weights(String(TEXT_ENCODER_DIR), ctx)
            vision = minimax_h3_vision_forward_device(vis_dev, patch_rows, vision_grids, ctx)
            print("  vision tower: DEVICE path (2304-patch gated geometry)")
        else:
            var vision_weights = minimax_h3_vision_load_weights(String(TEXT_ENCODER_DIR))
            vision = minimax_h3_vision_forward(
                vision_weights, patch_rows, vision_grids
            )
            print("  vision tower: HOST path (geometry outside the device gate)")
        if vision.num_tokens != len(presentation.pad_positions):
            raise Error(
                String("minimax_h3_i2va: the tower returned ")
                + String(vision.num_tokens) + " embeds but the presentation"
                " reserved " + String(len(presentation.pad_positions)) + " rows"
            )
        print("  vision tower:", vision.num_tokens, "embeds +",
              "3 deepstack blocks, ready to splice")

        # ── (b) CLOSED 2026-08-03: the streamed conditioner accepts
        # MiniMaxH3VisionOutput + visual_positions (deepstack gate 7/7,
        # capstone 6/6). Carry the tower output to the conditioning call,
        # which must run AFTER DeviceContext exists.
        vision_out = Optional[MiniMaxH3VisionOutput](vision^)


    var config = minimax_h3_released_config()
    config.validate()
    if max_blocks > config.num_layers:
        max_blocks = config.num_layers
    var partial_mode = max_blocks < config.num_layers
    var run_config = config
    run_config.num_layers = max_blocks

    print("  preflight: opening transformer shards:", String(TRANSFORMER_DIR))
    var transformer_shards = ShardedSafeTensors.open(String(TRANSFORMER_DIR))
    minimax_h3_check_modcache_weights(transformer_shards, run_config)
    _preflight_block_tensors(transformer_shards, run_config)
    print("  preflight: transformer OK")

    # ── GPU GUARD — the LAST thing before a DeviceContext exists ───────────
    # Placed here, in the binary, and not only in scripts/minimax_h3_i2va_smoke.sh:
    # that wrapper has had this check all along and it did not prevent the
    # 2026-08-03 near-miss, because the binary was invoked directly. A guard
    # outside the binary only protects the invocation that remembered to use it.
    #
    # The budget: this pipeline streams ONE transformer block at a time (~0.77
    # GiB) on top of a resident modcache, the video VAE, and the packed
    # sequence. 20 GiB is the figure the smoke script already used and is the
    # measured working set of a 124-frame run, not a guess at a safety margin.
    # (Guard + DeviceContext now run ABOVE the vision tower — see that block.)

    var t_pre1 = perf_counter_ns()
    print("  preflight OK (", Float64(t_pre1 - t_pre0) / 1.0e6, "ms)")

    # ── 1. Keyframe encode -> condition rows ───────────────────────────────
    var t_kf0 = perf_counter_ns()
    var enc_cfg = minimax_h3_video_released_encoder_config()
    var encoder = MiniMaxH3VideoEncoderDevice.load(String(VIDEO_VAE_DIR), enc_cfg, ctx)
    var tiling = minimax_h3_video_released_tiling_config()
    var moments = List[List[Float32]]()
    for i in range(len(keyframes)):
        moments.append(
            minimax_h3_keyframe_encode_device(encoder, keyframes[i], tiling, ctx)
        )
        print("  keyframe", i, "encoded ->", len(moments[i]), "moment values")

    # The REQUEST's generator. Conditioning noise first, then the target video
    # noise, then the target audio noise — one stream, in that order.
    var gen = MiniMaxH3TorchCpuGenerator(seed)
    var condition_rows = minimax_h3_keyframe_condition_rows(
        moments, gen, LATENT_H, LATENT_W, PATCH_H, PATCH_W
    )
    var video_noise_rows = minimax_h3_target_latent_rows(
        gen, NUM_LATENT_FRAMES, LATENT_H, LATENT_W, PATCH_H, PATCH_W
    )
    var audio_noise_rows = minimax_h3_target_audio_rows(
        gen, NUM_AUDIO_LATENTS, config.audio_latents_dim
    )
    var patch_dim = config.video_patch_dim()
    if len(condition_rows) != NUM_CONDITION_ROWS * patch_dim:
        raise Error(
            String("minimax_h3_i2va: condition rows hold ")
            + String(len(condition_rows)) + " values, expected "
            + String(NUM_CONDITION_ROWS * patch_dim)
        )
    var all_video = List[Float32](capacity=len(condition_rows) + len(video_noise_rows))
    for i in range(len(condition_rows)):
        all_video.append(condition_rows[i])
    for i in range(len(video_noise_rows)):
        all_video.append(video_noise_rows[i])
    var video_shape: List[Int] = [NUM_VIDEO_ROWS, patch_dim]
    var video_state = Tensor.from_host(all_video, video_shape^, STDtype.F32, ctx)
    var audio_shape: List[Int] = [NUM_AUDIO_ROWS, config.audio_latents_dim]
    var audio_state = Tensor.from_host(audio_noise_rows, audio_shape^, STDtype.F32, ctx)
    var t_kf1 = perf_counter_ns()
    print(
        "  keyframe conditioning ready:", NUM_CONDITION_ROWS, "condition rows (",
        Float64(t_kf1 - t_kf0) / 1.0e9, "s)",
    )

    # ── 2. Conditioning (degraded, text-only — see the seam above) ─────────
    var t_cond0 = perf_counter_ns()
    var cond: MiniMaxH3ConditioningOutput
    comptime
    if NO_VISION == 0:
        # REAL vision conditioning: the presentation's exact token ids (pads,
        # labels, alignment line — gated 12/12) through the streamed
        # conditioner with the tower output spliced at the pad rows and the
        # deepstack taps at language layers 0/1/2 (gate 7/7). Same
        # pad-to-dispatch-case treatment as minimax_h3_encode_conditioning:
        # trailing <|endoftext|> pads cannot alter earlier positions (causal),
        # sliced back after.
        var vids = presentation.token_ids.copy()
        var vreal = len(vids)
        var vpad = 8
        while vpad < vreal:
            vpad *= 2
        if vpad > 2048:
            raise Error("minimax_h3_i2va: presentation exceeds the sdpa dispatch table (2048)")
        for _ in range(vpad - vreal):
            vids.append(151643)
        var vemb_p = minimax_h3_encode_conditioning_streamed(
            String(TEXT_ENCODER_DIR), vids, ctx, vision_out^,
            Optional(presentation.pad_positions.copy()),
        )
        var vemb = slice(vemb_p, 1, 0, vreal, ctx)
        cond = MiniMaxH3ConditioningOutput(vemb^, presentation.token_tags.copy())
    else:
        cond = minimax_h3_encode_conditioning(
            String(PROCESSOR_DIR), String(TEXT_ENCODER_DIR), prompt, ctx
        )
    if len(cond.token_tags) != TEXT_TOKENS:
        raise Error(
            String("minimax_h3_i2va: the prompt tokenized to ")
            + String(len(cond.token_tags)) + " tokens, but this binary is"
            " compiled for H3_TEXT_TOKENS=" + String(TEXT_TOKENS)
            + " (S is comptime); rebuild with -D H3_TEXT_TOKENS="
            + String(len(cond.token_tags))
        )
    var text_rows = reshape(cond.embeds, [TEXT_TOKENS, config.text_dim], ctx)
    var t_cond1 = perf_counter_ns()
    print("  conditioning:", TEXT_TOKENS, "tokens (", Float64(t_cond1 - t_cond0) / 1.0e9, "s)")

    # ── 3. Packed layout, WITH anchors ─────────────────────────────────────
    var geometry = minimax_h3_build_sampling_geometry(
        cond.token_tags, NUM_LATENT_FRAMES, LATENT_H, LATENT_W,
        NUM_AUDIO_LATENTS, PATCH_H, PATCH_W, anchors,
    )
    if geometry.sequence_length != SEQ_LEN:
        raise Error(
            String("minimax_h3_i2va: geometry.sequence_length ")
            + String(geometry.sequence_length) + " != compiled S " + String(SEQ_LEN)
        )
    if geometry.num_condition_video_rows != NUM_CONDITION_ROWS:
        raise Error("minimax_h3_i2va: condition row count mismatch")

    # ── 4. RoPE ────────────────────────────────────────────────────────────
    var positions_f32 = List[Float32](capacity=len(geometry.position_ids))
    for i in range(len(geometry.position_ids)):
        positions_f32.append(Float32(geometry.position_ids[i]))
    var positions_shape: List[Int] = [SEQ_LEN * 3]
    var positions_tensor = Tensor.from_host(
        positions_f32, positions_shape^, STDtype.F32, ctx
    )
    var rope = build_minimax_h3_rope_tables(positions_tensor, ctx, config.rope_inv_freq_len)
    var rotary_dim = rope[0].shape()[1]

    # ── 5. Dual schedule + a THREE-row-per-step modulation cache ───────────
    var schedule = MiniMaxH3DualSchedule()
    schedule.set_timesteps(steps)
    var num_steps = schedule.num_inference_steps()

    var t_mod0 = perf_counter_ns()
    var distinct_timesteps = 3 * num_steps
    var temb_timesteps = List[Float32](capacity=distinct_timesteps)
    for i in range(num_steps):
        var vt = schedule.video_timestep(i)
        temb_timesteps.append(vt)
        temb_timesteps.append(schedule.audio_timestep(i))
        # `max(video_t, MINIMAX_H3_KEYFRAME_NOISE_AUG)` — before_denoise.py:417.
        temb_timesteps.append(vt if vt > Float32(0.999) else Float32(0.999))
    var temb_shape: List[Int] = [distinct_timesteps]
    var temb_ts_tensor = Tensor.from_host(temb_timesteps, temb_shape^, STDtype.F32, ctx)

    var frontend_w = _load_frontend_weights(transformer_shards, config, ctx)
    var temb = minimax_h3_timestep_embedding(temb_ts_tensor, frontend_w, config, ctx)
    var modcache = minimax_h3_build_modulation_cache(transformer_shards, temb, run_config, ctx)
    ctx.synchronize()
    var t_mod1 = perf_counter_ns()
    print(
        "  modcache:", distinct_timesteps, "rows (3 per step: video, audio,"
        " condition) (", Float64(t_mod1 - t_mod0) / 1.0e9, "s)",
    )

    # ── 6. Denoise — the condition rows are never written ──────────────────
    var t_den0 = perf_counter_ns()
    for i in range(num_steps):
        var t_step0 = perf_counter_ns()
        var video_ts = schedule.video_timestep(i)
        var audio_ts = schedule.audio_timestep(i)
        var global_row = _kf_global_timestep_row(
            geometry.token_tags, geometry.video_indices, NUM_CONDITION_ROWS, i,
            geometry.sequence_length,
        )
        var block_adaln_indices = minimax_h3_adaln_rows(global_row, geometry.token_tags)

        var placeholder_shape: List[Int] = [1]
        var placeholder_ts = Tensor.from_host(
            [video_ts], placeholder_shape^, STDtype.F32, ctx
        )
        var embed = minimax_h3_frontend_embed[TEXT_TOKENS, H3_HEADS, H3_HEAD_DIM](
            video_state, audio_state, text_rows, placeholder_ts,
            geometry.video_indices, geometry.audio_indices, geometry.text_indices,
            SEQ_LEN, frontend_w, config, ctx,
        )
        var hidden3 = reshape(embed.hidden, [1, SEQ_LEN, config.hidden_size], ctx)
        for layer in range(run_config.num_layers):
            var block_w = minimax_h3_load_block_device(transformer_shards, layer, config, ctx)
            hidden3 = minimax_h3_block_forward[SEQ_LEN, H3_HEADS, H3_HEAD_DIM](
                hidden3, block_w, layer, config, modcache.block_mod[layer][],
                block_adaln_indices, rope[0], rope[1], rotary_dim, ctx,
                attention_backend,
            )
        var hidden2 = reshape(hidden3, [SEQ_LEN, config.hidden_size], ctx)
        var frontend_out = minimax_h3_final_layer(
            hidden2, modcache.final_mod[], global_row,
            geometry.video_indices, geometry.audio_indices,
            frontend_w, config, ctx,
        )

        # ONLY the target rows are stepped. The conditioning rows are re-imposed
        # by construction — "the loop only ever writes the generated rows, so
        # they are never updated again" (encoders.py:222-225, denoise.py:186).
        var cond_part = slice(video_state, 0, 0, NUM_CONDITION_ROWS, ctx)
        var tgt_state = slice(
            video_state, 0, NUM_CONDITION_ROWS, NUM_TARGET_VIDEO_ROWS, ctx
        )
        var tgt_vel = slice(
            frontend_out.video_out, 0, NUM_CONDITION_ROWS, NUM_TARGET_VIDEO_ROWS, ctx
        )
        var stepped = schedule.step_video_device(tgt_vel, video_ts, tgt_state, ctx)
        video_state = concat(0, ctx, cond_part, stepped)
        audio_state = schedule.step_audio_device(
            frontend_out.audio_out, audio_ts, audio_state, ctx
        )
        ctx.synchronize()
        var t_step1 = perf_counter_ns()
        print(
            "  phase=denoise step=", i + 1, " total=", num_steps,
            " video_t=", video_ts, " audio_t=", audio_ts,
            " cond_t=", (video_ts if video_ts > Float32(0.999) else Float32(0.999)),
            " (", Float64(t_step1 - t_step0) / 1.0e9, "s)",
        )

    var lat_names = List[String]()
    lat_names.append(String("video_state_rows"))
    lat_names.append(String("audio_state_rows"))
    var lat_tensors = List[ArcPointer[Tensor]]()
    lat_tensors.append(ArcPointer[Tensor](slice(video_state, 0, 0, NUM_VIDEO_ROWS, ctx)))
    lat_tensors.append(ArcPointer[Tensor](slice(audio_state, 0, 0, NUM_AUDIO_ROWS, ctx)))
    save_safetensors(lat_names, lat_tensors, out_dir + "/latents.safetensors", ctx)
    var t_den1 = perf_counter_ns()
    print("  denoise done (", Float64(t_den1 - t_den0) / 1.0e9, "s)")

    # ── 7. Decode ──────────────────────────────────────────────────────────
    var audio_samples = _decode_audio(
        audio_state, NUM_AUDIO_LATENTS, config.audio_latents_dim, out_dir, ctx
    )
    var frames_written = _decode_video(video_state, config.latents_dim, out_dir, ctx)

    var body = String("{\n")
    body += String("  \"task\":\"") + mode + String("\",\n")
    body += String("  \"vision_conditioning\":\"DEGRADED_NO_VISION_TOWER\",\n")
    body += String(
        "  \"WARNING\":\"The conditioner did NOT see the keyframe. Only the VAE"
        " condition rows anchor this render. NOT the released model's"
        " conditioning.\",\n"
    )
    body += String("  \"partial_mode\":") + json_bool(partial_mode) + String(",\n")
    body += String("  \"max_blocks\":") + String(max_blocks) + String(",\n")
    body += String("  \"prompt\":\"") + json_escape(prompt) + String("\",\n")
    body += String("  \"steps\":") + String(num_steps) + String(",\n")
    body += String("  \"seed\":") + String(seed) + String(",\n")
    body += String("  \"width\":") + String(WIDTH) + String(",\n")
    body += String("  \"height\":") + String(HEIGHT) + String(",\n")
    body += String("  \"frames\":") + String(FRAMES) + String(",\n")
    body += String("  \"keyframes\":") + String(KEYFRAMES) + String(",\n")
    body += String("  \"condition_rows\":") + String(NUM_CONDITION_ROWS) + String(",\n")
    body += String("  \"sequence_length\":") + String(SEQ_LEN) + String(",\n")
    body += String("  \"attention_backend\":\"") \
        + attention_backend_name + String("\",\n")
    body += String("  \"weight_storage\":\"streamed-bf16\",\n")
    body += String("  \"frames_written\":") + String(frames_written) + String(",\n")
    body += String("  \"audio_samples_per_channel\":") + String(audio_samples) + String("\n")
    body += String("}\n")
    write_text_file(out_dir + String("/result.json"), body)
    print("  wrote", out_dir + String("/result.json"))

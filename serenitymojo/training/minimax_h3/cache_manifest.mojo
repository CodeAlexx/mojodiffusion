# MiniMax H3 training-cache manifest contract.
#
# This module validates manifest values already loaded by a future product
# manifest reader.  It deliberately does not discover datasets, build caches,
# open tensors, or initialize DeviceContext.  A reader/artifact-consumer gate
# remains required before a real launch can set `checksum_verified=True`.
#
# Development oracle (never a product dependency):
#   kohya-ss/musubi-tuner
#   commit b8717864713c9e4e7ef3d56eba1fc695a9b626a5
#   minimax_h3_cache_latents.py
#   minimax_h3_cache_text_encoder_outputs.py
#   dataset/cache_io.py::save_{latent,text_encoder_output}_cache_minimax_h3
#   minimax_h3_train_network.py::_runtime_batch_plan

from std.collections import List


comptime MINIMAX_H3_CACHE_MANIFEST_SCHEMA = (
    "serenity.minimax_h3.training_cache.v2"
)
comptime MINIMAX_H3_CACHE_ORACLE_COMMIT = (
    "b8717864713c9e4e7ef3d56eba1fc695a9b626a5"
)
comptime MINIMAX_H3_LATENT_CACHE_FORMAT = "minimax-h3-latent-v2"
comptime MINIMAX_H3_TEXT_CACHE_FORMAT = "minimax-h3-text-v2"
comptime MINIMAX_H3_TEXT_WIDTH = 5120
comptime MINIMAX_H3_TEXT_MAX_ROWS = 32768


def _is_hex_byte(value: UInt8) -> Bool:
    return (
        (value >= 0x30 and value <= 0x39)
        or (value >= 0x61 and value <= 0x66)
    )


def _is_sha256_receipt(value: String) -> Bool:
    if value.byte_length() != 71:
        return False
    var bytes = value.as_bytes()
    var prefix: List[UInt8] = [0x73, 0x68, 0x61, 0x32, 0x35, 0x36, 0x3A]
    for i in range(len(prefix)):
        if bytes[i] != prefix[i]:
            return False
    for i in range(7, 71):
        if not _is_hex_byte(bytes[i]):
            return False
    return True


def minimax_h3_is_sha256_receipt(value: String) -> Bool:
    """Public spelling used by the product config/consumer boundary."""
    return _is_sha256_receipt(value)


def _contains_path_segment(path: String, segment: String) -> Bool:
    return path.find(segment) >= 0


def _is_canonical_absolute_path(path: String) -> Bool:
    if path.byte_length() == 0 or path.as_bytes()[0] != 0x2F:
        return False
    if path.byte_length() > 1 and path.as_bytes()[path.byte_length() - 1] == 0x2F:
        return False
    return (
        not _contains_path_segment(path, String("//"))
        and not _contains_path_segment(path, String("/./"))
        and not _contains_path_segment(path, String("/../"))
        and not path.endswith(String("/."))
        and not path.endswith(String("/.."))
    )


def minimax_h3_is_canonical_absolute_path(path: String) -> Bool:
    """Reject relative, trailing-slash, dot-segment, and doubled paths."""
    return _is_canonical_absolute_path(path)


def _is_supported_task(task: String) -> Bool:
    return (
        task == String("t2va")
        or task == String("fl2va")
        or task == String("ref2va")
    )


def _is_safe_identity(value: String) -> Bool:
    if value.byte_length() == 0:
        return False
    var bytes = value.as_bytes()
    for i in range(value.byte_length()):
        var byte = bytes[i]
        if not (
            (byte >= 0x30 and byte <= 0x39)
            or (byte >= 0x41 and byte <= 0x5A)
            or (byte >= 0x61 and byte <= 0x7A)
            or byte == 0x2D or byte == 0x2E or byte == 0x5F
        ):
            return False
    return True


def _is_supported_text_dtype(dtype: String) -> Bool:
    # Musubi exposes --text_cache_dtype {bf16,float32} at the pinned oracle.
    return dtype == String("bfloat16") or dtype == String("float32")


def _is_supported_text_cache_dtype(dtype: String) -> Bool:
    return dtype == String("bf16") or dtype == String("float32")


@fieldwise_init
struct MiniMaxH3CacheSample(Copyable, Movable):
    # Provenance. IDs are deterministic lowercase SHA-256 receipts rather than
    # ordinal filenames, so cache order cannot silently rename samples.
    var sample_id: String
    var source_fingerprint: String
    var source_image_relative_path: String
    var source_caption_relative_path: String
    var original_width: Int
    var original_height: Int
    # Exact JSON string written by pinned Musubi's
    # `_media_fingerprint_metadata`; content bytes are independently rebound
    # to `source_fingerprint` by the live source intake.
    var media_fingerprints: String
    var presentation_fingerprint: String
    var crop_start_frame: Int
    var latent_artifact_path: String
    var latent_artifact_sha256: String
    var text_artifact_path: String
    var text_artifact_sha256: String

    # Native cached targets (unbatched safetensors payload shapes).
    var source_frame_count: Int
    var target_video_shape: List[Int]  # [24,F,H,W]
    var target_audio_shape: List[Int]  # [32,2,A]
    var latent_dtype: String           # pinned cache writers produce float32
    var audio_present_dtype: String    # scalar float32
    var audio_present: Float32         # exact scalar consumed by loss gating
    var audio_present_binary_verified: Bool

    # Native cached Qwen3-VL conditioning.
    var text_hidden_shape: List[Int]   # [L,5120]
    var text_hidden_dtype: String      # bfloat16 or float32
    var text_token_tags_shape: List[Int]  # [L]
    var text_token_tags_dtype: String     # int64
    var text_token_tags_binary_verified: Bool
    var teacher_text_present: Bool

    # Task-dependent latent roles. No elementwise loss-mask field exists: the
    # pinned native loss consumes full-tensor MSE plus scalar audio presence.
    var has_first_condition: Bool
    var has_last_condition: Bool
    var ref_image_count: Int
    var ref_video_count: Int
    var ref_audio_count: Int  # references whose audio latent is present
    var ref_item_count: Int   # all image/video/audio-only reference items
    var one_frame_target_index: Int       # -1 outside one-frame mode
    var one_frame_control_indices: List[Int]


@fieldwise_init
struct MiniMaxH3CacheManifest(Copyable, Movable):
    var schema: String
    var oracle_commit: String
    var manifest_path: String
    var manifest_sha256: String
    var checksum_verified: Bool
    var dataset_identity: String
    var canonical_dataset_path: String
    var dataset_fingerprint: String
    var cache_root: String
    var cache_namespace: String
    var task: String
    var one_frame: Bool
    var cache_seed: Int
    var text_cache_dtype: String
    var latent_cache_format: String
    var text_cache_format: String
    var video_vae_fingerprint: String
    var audio_vae_fingerprint: String
    var text_encoder_fingerprint: String
    var processor_fingerprint: String
    var sample_ids_digest: String
    var samples: List[MiniMaxH3CacheSample]


def minimax_h3_cache_namespace(
    dataset_identity: String,
    dataset_fingerprint: String,
    task: String,
    one_frame: Bool,
) -> String:
    return (
        String("minimax_h3/") + dataset_identity + String("/")
        + dataset_fingerprint + String("/") + task
        + (String("/one_frame") if one_frame else String("/video"))
    )


def _validate_shape(
    shape: List[Int], expected_rank: Int, label: String,
) raises:
    if len(shape) != expected_rank:
        raise Error(label + String(" has the wrong rank"))
    for dim in shape:
        if dim <= 0:
            raise Error(label + String(" dimensions must be positive"))


def _is_safe_top_level_relative_path(path: String) -> Bool:
    return (
        path.byte_length() > 0
        and path.as_bytes()[0] != 0x2F
        and path.find(String("/")) < 0
        and path.find(String("\\")) < 0
        and path != String(".") and path != String("..")
    )


def validate_minimax_h3_cache_sample(
    sample: MiniMaxH3CacheSample,
    cache_root: String,
    task: String,
    one_frame: Bool,
) raises:
    if not _is_sha256_receipt(sample.sample_id):
        raise Error("MiniMax H3 sample_id must be a lowercase sha256 receipt")
    if not _is_sha256_receipt(sample.source_fingerprint):
        raise Error("MiniMax H3 sample source fingerprint must be sha256-bound")
    if sample.sample_id != sample.source_fingerprint:
        raise Error("MiniMax H3 sample ID must equal its deterministic source-pair fingerprint")
    if (
        not _is_safe_top_level_relative_path(sample.source_image_relative_path)
        or not _is_safe_top_level_relative_path(sample.source_caption_relative_path)
        or sample.source_image_relative_path == sample.source_caption_relative_path
    ):
        raise Error("MiniMax H3 sample source paths must be distinct top-level relative names")
    if sample.original_width <= 0 or sample.original_height <= 0:
        raise Error("MiniMax H3 sample original source dimensions must be positive")
    if sample.media_fingerprints.byte_length() == 0:
        raise Error("MiniMax H3 sample media fingerprints are missing")
    if not _is_sha256_receipt(sample.presentation_fingerprint):
        raise Error("MiniMax H3 sample presentation fingerprint must be sha256-bound")
    if sample.crop_start_frame < 0:
        raise Error("MiniMax H3 sample crop start must be nonnegative")
    if (
        not _is_sha256_receipt(sample.latent_artifact_sha256)
        or not _is_sha256_receipt(sample.text_artifact_sha256)
    ):
        raise Error("MiniMax H3 sample artifacts must be sha256-bound")
    var cache_prefix = cache_root + String("/")
    if (
        not _is_canonical_absolute_path(sample.latent_artifact_path)
        or not sample.latent_artifact_path.startswith(cache_prefix)
        or not _is_canonical_absolute_path(sample.text_artifact_path)
        or not sample.text_artifact_path.startswith(cache_prefix)
    ):
        raise Error("MiniMax H3 sample artifacts must stay under the bound cache root")
    if sample.latent_artifact_path == sample.text_artifact_path:
        raise Error("MiniMax H3 latent and text cache artifacts must be distinct")

    _validate_shape(sample.target_video_shape, 4, String("target video cache"))
    if sample.target_video_shape[0] != 24:
        raise Error("MiniMax H3 target video cache must be [24,F,H,W]")
    _validate_shape(sample.target_audio_shape, 3, String("target audio cache"))
    if sample.target_audio_shape[0] != 32 or sample.target_audio_shape[1] != 2:
        raise Error("MiniMax H3 target audio cache must be [32,2,A]")
    if sample.latent_dtype != String("float32"):
        raise Error("MiniMax H3 pinned native latent caches must be float32")
    if sample.audio_present_dtype != String("float32"):
        raise Error("MiniMax H3 audio_present must be scalar float32")
    if (
        not sample.audio_present_binary_verified
        or (sample.audio_present != 0.0 and sample.audio_present != 1.0)
    ):
        raise Error("MiniMax H3 audio_present must be verified as exactly 0 or 1")

    _validate_shape(sample.text_hidden_shape, 2, String("text hidden cache"))
    if (
        sample.text_hidden_shape[0] > MINIMAX_H3_TEXT_MAX_ROWS
        or sample.text_hidden_shape[1] != MINIMAX_H3_TEXT_WIDTH
    ):
        raise Error("MiniMax H3 text hidden cache must be [L<=32768,5120]")
    if not _is_supported_text_dtype(sample.text_hidden_dtype):
        raise Error("MiniMax H3 text hidden cache must be bfloat16 or float32")
    _validate_shape(sample.text_token_tags_shape, 1, String("text token tags"))
    if sample.text_token_tags_shape[0] != sample.text_hidden_shape[0]:
        raise Error("MiniMax H3 text hidden rows and token-tag rows must match")
    if sample.text_token_tags_dtype != String("int64"):
        raise Error("MiniMax H3 text token tags must be int64")
    if not sample.text_token_tags_binary_verified:
        raise Error("MiniMax H3 text token tags must be verified as only 0 or 1")
    if sample.teacher_text_present:
        raise Error("MiniMax H3 cache has unsupported teacher-matching text rows")
    if one_frame:
        if task == String("ref2va"):
            raise Error("MiniMax H3 one-frame caches do not support ref2va")
        if sample.target_video_shape[1] != 1:
            raise Error("MiniMax H3 one-frame cache requires one video latent frame")
        if sample.source_frame_count != 1 or sample.target_audio_shape[2] != 2:
            raise Error("MiniMax H3 one-frame target requires source F=1 and audio A=2")
        if sample.one_frame_target_index < 0:
            raise Error("MiniMax H3 one-frame cache requires a nonnegative target index")
        if sample.crop_start_frame != 0:
            raise Error("MiniMax H3 one-frame cache requires crop_start_frame=0")
        for index in sample.one_frame_control_indices:
            if index < 0:
                raise Error("MiniMax H3 one-frame control indices must be nonnegative")
    else:
        if (
            sample.source_frame_count < 5
            or (sample.source_frame_count - 5) % 17 != 0
        ):
            raise Error("MiniMax H3 video source frame count must be 17*n+5")
        var expected_video_frames = (
            5 * ((sample.source_frame_count - 5) // 17) + 2
        )
        var expected_audio_frames = (10 * sample.source_frame_count + 3) // 6
        if sample.target_video_shape[1] != expected_video_frames:
            raise Error("MiniMax H3 cached video latent frame count is inconsistent")
        if sample.target_audio_shape[2] != expected_audio_frames:
            raise Error("MiniMax H3 cached audio latent frame count is inconsistent")
        if sample.one_frame_target_index != -1 or len(sample.one_frame_control_indices) != 0:
            raise Error("MiniMax H3 video cache cannot carry one-frame index tensors")

    if task == String("t2va"):
        if (
            sample.has_first_condition or sample.has_last_condition
            or sample.ref_image_count != 0 or sample.ref_video_count != 0
            or sample.ref_audio_count != 0 or sample.ref_item_count != 0
            or len(sample.one_frame_control_indices) != 0
        ):
            raise Error("MiniMax H3 t2va cache cannot carry condition latents")
    elif task == String("fl2va"):
        if (
            sample.ref_image_count != 0 or sample.ref_video_count != 0
            or sample.ref_audio_count != 0 or sample.ref_item_count != 0
        ):
            raise Error("MiniMax H3 fl2va cache cannot carry Ref2VA roles")
        var condition_count = (
            Int(sample.has_first_condition) + Int(sample.has_last_condition)
        )
        if one_frame:
            if not sample.has_first_condition:
                raise Error("MiniMax H3 one-frame fl2va controls must start at first")
            if condition_count < 1 or condition_count > 2:
                raise Error("MiniMax H3 one-frame fl2va requires one or two controls")
            if len(sample.one_frame_control_indices) != condition_count:
                raise Error("MiniMax H3 one-frame fl2va control indices must match controls")
            if condition_count == 2 \
                    and sample.one_frame_control_indices[0] >= sample.one_frame_control_indices[1]:
                raise Error("MiniMax H3 one-frame FL2VA control indices must be ordered")
        elif condition_count != 2 or len(sample.one_frame_control_indices) != 0:
            raise Error("MiniMax H3 video fl2va requires first and last conditions")
    else:
        if sample.has_first_condition or sample.has_last_condition:
            raise Error("MiniMax H3 ref2va cache cannot carry FL2VA roles")
        if (
            sample.ref_image_count < 0 or sample.ref_video_count < 0
            or sample.ref_audio_count < 0 or sample.ref_item_count < 0
        ):
            raise Error("MiniMax H3 reference counts must be nonnegative")
        if sample.ref_image_count > 9 or sample.ref_video_count > 3 or sample.ref_audio_count > 3:
            raise Error("MiniMax H3 ref2va reference count exceeds native bounds")
        if sample.ref_image_count + sample.ref_video_count == 0:
            raise Error("MiniMax H3 ref2va requires at least one visual reference")
        if (
            sample.ref_item_count < sample.ref_image_count + sample.ref_video_count
            or sample.ref_item_count > 12
        ):
            raise Error("MiniMax H3 ref2va total reference count is inconsistent or exceeds 12")


def validate_minimax_h3_cache_manifest(
    manifest: MiniMaxH3CacheManifest,
    expected_dataset_identity: String,
    expected_canonical_dataset_path: String,
    expected_dataset_fingerprint: String,
    expected_video_vae_fingerprint: String,
    expected_audio_vae_fingerprint: String,
    expected_text_encoder_fingerprint: String,
    expected_processor_fingerprint: String,
    expected_task: String,
    expected_one_frame: Bool,
) raises:
    if manifest.schema != String(MINIMAX_H3_CACHE_MANIFEST_SCHEMA):
        raise Error("MiniMax H3 cache manifest schema mismatch")
    if manifest.oracle_commit != String(MINIMAX_H3_CACHE_ORACLE_COMMIT):
        raise Error("MiniMax H3 cache manifest oracle commit mismatch")
    if not _is_canonical_absolute_path(manifest.manifest_path):
        raise Error("MiniMax H3 cache manifest path must be canonical and absolute")
    if not _is_sha256_receipt(manifest.manifest_sha256) or not manifest.checksum_verified:
        raise Error("MiniMax H3 cache manifest checksum is missing or unverified")
    if (
        not _is_safe_identity(expected_dataset_identity)
        or manifest.dataset_identity != expected_dataset_identity
    ):
        raise Error("MiniMax H3 cache manifest dataset identity mismatch")
    if (
        not _is_canonical_absolute_path(expected_canonical_dataset_path)
        or manifest.canonical_dataset_path != expected_canonical_dataset_path
    ):
        raise Error("MiniMax H3 cache manifest canonical dataset path mismatch")
    if (
        not _is_sha256_receipt(expected_dataset_fingerprint)
        or manifest.dataset_fingerprint != expected_dataset_fingerprint
    ):
        raise Error("MiniMax H3 cache manifest dataset fingerprint mismatch")
    if not _is_canonical_absolute_path(manifest.cache_root):
        raise Error("MiniMax H3 cache root must be canonical and absolute")
    if manifest.manifest_path != manifest.cache_root + String("/cache_manifest.json"):
        raise Error("MiniMax H3 cache manifest must live under its bound cache root")
    if not _is_supported_task(expected_task) or manifest.task != expected_task:
        raise Error("MiniMax H3 cache manifest task mismatch")
    if manifest.one_frame != expected_one_frame:
        raise Error("MiniMax H3 cache manifest one-frame mode mismatch")
    if not _is_supported_text_cache_dtype(manifest.text_cache_dtype):
        raise Error("MiniMax H3 manifest text cache dtype must be bf16 or float32")
    if manifest.latent_cache_format != String(MINIMAX_H3_LATENT_CACHE_FORMAT):
        raise Error("MiniMax H3 latent cache format mismatch")
    if manifest.text_cache_format != String(MINIMAX_H3_TEXT_CACHE_FORMAT):
        raise Error("MiniMax H3 text cache format mismatch")
    if (
        not _is_sha256_receipt(expected_video_vae_fingerprint)
        or manifest.video_vae_fingerprint != expected_video_vae_fingerprint
        or not _is_sha256_receipt(expected_audio_vae_fingerprint)
        or manifest.audio_vae_fingerprint != expected_audio_vae_fingerprint
        or not _is_sha256_receipt(expected_text_encoder_fingerprint)
        or manifest.text_encoder_fingerprint != expected_text_encoder_fingerprint
        or not _is_sha256_receipt(expected_processor_fingerprint)
        or manifest.processor_fingerprint != expected_processor_fingerprint
    ):
        raise Error("MiniMax H3 cache model/processor fingerprint mismatch")
    var expected_namespace = minimax_h3_cache_namespace(
        expected_dataset_identity,
        expected_dataset_fingerprint,
        expected_task,
        expected_one_frame,
    )
    if manifest.cache_namespace != expected_namespace:
        raise Error("MiniMax H3 cache namespace is not dataset/task-bound")
    if not _is_sha256_receipt(manifest.sample_ids_digest):
        raise Error("MiniMax H3 sample-ID manifest digest must be sha256-bound")
    if len(manifest.samples) == 0:
        raise Error("MiniMax H3 cache manifest has no samples")
    for i in range(len(manifest.samples)):
        var sample = manifest.samples[i].copy()
        validate_minimax_h3_cache_sample(
            sample, manifest.cache_root, expected_task, expected_one_frame,
        )
        for previous in range(i):
            if manifest.samples[previous].sample_id == sample.sample_id:
                raise Error("MiniMax H3 sample IDs must be unique")
            if (
                manifest.samples[previous].latent_artifact_path
                == sample.latent_artifact_path
                or manifest.samples[previous].text_artifact_path
                == sample.text_artifact_path
            ):
                raise Error("MiniMax H3 cache artifacts cannot be shared across sample IDs")

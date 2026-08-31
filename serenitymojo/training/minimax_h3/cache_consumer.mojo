# Native MiniMax-H3 product cache consumer.
#
# Host preflight is deliberately complete before a DeviceContext is accepted:
# strict JSON schema, dataset/task/fingerprint binding, streaming SHA-256 of the
# manifest and every sample artifact, safetensors key/shape/dtype checks, and
# scalar/tag value checks. Python and Musubi are development oracles only.
#
# Pinned cache writers:
#   kohya-ss/musubi-tuner b8717864713c9e4e7ef3d56eba1fc695a9b626a5
#   dataset/cache_io.py::save_latent_cache_minimax_h3
#   dataset/cache_io.py::save_text_encoder_output_cache_minimax_h3

from json.parser import loads
from json.value import JSONValue
from max.gpu.host import DeviceContext
from std.collections import List
from std.memory import ArcPointer, alloc

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import O_RDONLY, file_size, sys_close, sys_open, sys_pread
from serenitymojo.io.safetensors import SafeTensors, read_f32_scalar_bytes
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.models.dit.minimax_h3_dit import (
    minimax_h3_expected_shape,
    minimax_h3_released_config,
)
from serenitymojo.models.minimax_h3.loader import minimax_h3_is_fp32_key
from serenitymojo.tensor import Tensor
from serenitymojo.training.minimax_h3.cache_manifest import (
    MINIMAX_H3_CACHE_MANIFEST_SCHEMA,
    MINIMAX_H3_CACHE_ORACLE_COMMIT,
    MiniMaxH3CacheManifest,
    MiniMaxH3CacheSample,
    minimax_h3_is_sha256_receipt,
)
from serenitymojo.training.minimax_h3.config import MiniMaxH3ProductConfig
from serenitymojo.training.minimax_h3.contract import (
    validate_minimax_h3_launch_preflight,
)
from serenitymojo.training.minimax_h3.sha256 import (
    minimax_h3_sha256_file,
    minimax_h3_sha256_text,
)
from serenitymojo.training.minimax_h3.source_dataset import (
    MiniMaxH3SourceReceipt,
    intake_minimax_h3_source_dataset,
)


def _read_manifest_text(path: String) raises -> String:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("MiniMax H3 manifest cannot open ") + path)
    var size = file_size(fd)
    if size <= 0 or size > 64 * 1024 * 1024:
        _ = sys_close(fd)
        raise Error("MiniMax H3 manifest size must be in (0,64MiB]")
    var buf = alloc[UInt8](size)
    var done = 0
    while done < size:
        var count = sys_pread(fd, buf + done, size - done, done)
        if count <= 0:
            buf.free()
            _ = sys_close(fd)
            raise Error(String("MiniMax H3 manifest short read: ") + path)
        done += count
    _ = sys_close(fd)
    var text = String(
        StringSlice(unsafe_from_utf8=Span(unsafe_ptr=buf, length=size))
    )
    buf.free()
    return text^


def _exact_keys(obj: JSONValue, expected: List[String], label: String) raises:
    if not obj.is_object() or obj.length() != len(expected):
        raise Error(label + String(" has the wrong object/key count"))
    for key in expected:
        if not obj.contains(key):
            raise Error(label + String(" missing key ") + key)
    for key in obj.keys():
        if key not in expected:
            raise Error(label + String(" unknown key ") + key)


def _req_string(obj: JSONValue, key: String) raises -> String:
    var value = obj[key]
    if not value.is_string():
        raise Error(String("MiniMax H3 manifest string required: ") + key)
    return value.as_string()


def _req_bool(obj: JSONValue, key: String) raises -> Bool:
    var value = obj[key]
    if not value.is_bool():
        raise Error(String("MiniMax H3 manifest bool required: ") + key)
    return value.as_bool()


def _req_int(obj: JSONValue, key: String) raises -> Int:
    var value = obj[key]
    if not value.is_int():
        raise Error(String("MiniMax H3 manifest integer required: ") + key)
    return value.as_int()


def _req_float(obj: JSONValue, key: String) raises -> Float32:
    var value = obj[key]
    if not value.is_number():
        raise Error(String("MiniMax H3 manifest number required: ") + key)
    return Float32(value.as_float())


def _req_ints(obj: JSONValue, key: String) raises -> List[Int]:
    var value = obj[key]
    if not value.is_array():
        raise Error(String("MiniMax H3 manifest integer array required: ") + key)
    var out = List[Int]()
    for index in range(value.length()):
        if not value[index].is_int():
            raise Error(String("MiniMax H3 manifest noninteger in ") + key)
        out.append(value[index].as_int())
    return out^


def _manifest_keys() -> List[String]:
    return [
        String("schema"), String("oracle_commit"),
        String("dataset_identity"), String("canonical_dataset_path"),
        String("dataset_fingerprint"), String("cache_root"),
        String("cache_namespace"), String("task"), String("one_frame"),
        String("cache_seed"), String("text_cache_dtype"),
        String("latent_cache_format"), String("text_cache_format"),
        String("video_vae_fingerprint"), String("audio_vae_fingerprint"),
        String("text_encoder_fingerprint"), String("processor_fingerprint"),
        String("sample_ids_digest"), String("samples"),
    ]


def _sample_keys() -> List[String]:
    return [
        String("sample_id"), String("source_fingerprint"),
        String("source_image_relative_path"),
        String("source_caption_relative_path"),
        String("original_width"), String("original_height"),
        String("media_fingerprints"), String("presentation_fingerprint"),
        String("crop_start_frame"),
        String("latent_artifact_path"), String("latent_artifact_sha256"),
        String("text_artifact_path"), String("text_artifact_sha256"),
        String("source_frame_count"), String("target_video_shape"),
        String("target_audio_shape"), String("latent_dtype"),
        String("audio_present_dtype"), String("audio_present"),
        String("audio_present_binary_verified"), String("text_hidden_shape"),
        String("text_hidden_dtype"), String("text_token_tags_shape"),
        String("text_token_tags_dtype"), String("text_token_tags_binary_verified"),
        String("teacher_text_present"), String("has_first_condition"),
        String("has_last_condition"), String("ref_image_count"),
        String("ref_video_count"), String("ref_audio_count"),
        String("ref_item_count"), String("one_frame_target_index"),
        String("one_frame_control_indices"),
    ]


def _parse_sample(obj: JSONValue) raises -> MiniMaxH3CacheSample:
    _exact_keys(obj, _sample_keys(), String("MiniMax H3 manifest sample"))
    return MiniMaxH3CacheSample(
        _req_string(obj, String("sample_id")),
        _req_string(obj, String("source_fingerprint")),
        _req_string(obj, String("source_image_relative_path")),
        _req_string(obj, String("source_caption_relative_path")),
        _req_int(obj, String("original_width")),
        _req_int(obj, String("original_height")),
        _req_string(obj, String("media_fingerprints")),
        _req_string(obj, String("presentation_fingerprint")),
        _req_int(obj, String("crop_start_frame")),
        _req_string(obj, String("latent_artifact_path")),
        _req_string(obj, String("latent_artifact_sha256")),
        _req_string(obj, String("text_artifact_path")),
        _req_string(obj, String("text_artifact_sha256")),
        _req_int(obj, String("source_frame_count")),
        _req_ints(obj, String("target_video_shape")),
        _req_ints(obj, String("target_audio_shape")),
        _req_string(obj, String("latent_dtype")),
        _req_string(obj, String("audio_present_dtype")),
        _req_float(obj, String("audio_present")),
        _req_bool(obj, String("audio_present_binary_verified")),
        _req_ints(obj, String("text_hidden_shape")),
        _req_string(obj, String("text_hidden_dtype")),
        _req_ints(obj, String("text_token_tags_shape")),
        _req_string(obj, String("text_token_tags_dtype")),
        _req_bool(obj, String("text_token_tags_binary_verified")),
        _req_bool(obj, String("teacher_text_present")),
        _req_bool(obj, String("has_first_condition")),
        _req_bool(obj, String("has_last_condition")),
        _req_int(obj, String("ref_image_count")),
        _req_int(obj, String("ref_video_count")),
        _req_int(obj, String("ref_audio_count")),
        _req_int(obj, String("ref_item_count")),
        _req_int(obj, String("one_frame_target_index")),
        _req_ints(obj, String("one_frame_control_indices")),
    )


def _parse_manifest(
    doc: JSONValue, path: String, digest: String,
) raises -> MiniMaxH3CacheManifest:
    _exact_keys(doc, _manifest_keys(), String("MiniMax H3 manifest"))
    var sample_values = doc[String("samples")]
    if not sample_values.is_array():
        raise Error("MiniMax H3 manifest samples must be an array")
    var samples = List[MiniMaxH3CacheSample]()
    for index in range(sample_values.length()):
        samples.append(_parse_sample(sample_values[index]))
    return MiniMaxH3CacheManifest(
        _req_string(doc, String("schema")),
        _req_string(doc, String("oracle_commit")),
        path,
        digest,
        True,
        _req_string(doc, String("dataset_identity")),
        _req_string(doc, String("canonical_dataset_path")),
        _req_string(doc, String("dataset_fingerprint")),
        _req_string(doc, String("cache_root")),
        _req_string(doc, String("cache_namespace")),
        _req_string(doc, String("task")),
        _req_bool(doc, String("one_frame")),
        _req_int(doc, String("cache_seed")),
        _req_string(doc, String("text_cache_dtype")),
        _req_string(doc, String("latent_cache_format")),
        _req_string(doc, String("text_cache_format")),
        _req_string(doc, String("video_vae_fingerprint")),
        _req_string(doc, String("audio_vae_fingerprint")),
        _req_string(doc, String("text_encoder_fingerprint")),
        _req_string(doc, String("processor_fingerprint")),
        _req_string(doc, String("sample_ids_digest")),
        samples^,
    )


def _sample_ids_digest(samples: List[MiniMaxH3CacheSample]) -> String:
    var text = String("")
    for sample in samples:
        text += sample.sample_id + String("\n")
    return minimax_h3_sha256_text(text)


def _same_shape(got: List[Int], expected: List[Int]) -> Bool:
    if len(got) != len(expected):
        return False
    for index in range(len(got)):
        if got[index] != expected[index]:
            return False
    return True


def _require_tensor(
    st: SafeTensors, key: String, dtype: STDtype, shape: List[Int],
) raises:
    if not st.has_tensor(key):
        raise Error(String("MiniMax H3 cache missing tensor: ") + key)
    var info = st.tensor_info(key)
    if info.dtype != dtype or not _same_shape(info.shape, shape):
        raise Error(String("MiniMax H3 cache dtype/shape mismatch: ") + key)


def _verify_binary_i64(st: SafeTensors, key: String) raises:
    var info = st.tensor_info(key)
    if info.dtype != STDtype.I64 or info.size % 8 != 0:
        raise Error(String("MiniMax H3 binary tags must be int64: ") + key)
    var data = st.tensor_bytes(key)
    for offset in range(0, len(data), 8):
        if data[offset] > UInt8(1):
            raise Error(String("MiniMax H3 binary tag out of range: ") + key)
        for byte in range(1, 8):
            if data[offset + byte] != UInt8(0):
                raise Error(String("MiniMax H3 binary tag out of range: ") + key)


def _verify_exact_nonnegative_i64(
    st: SafeTensors, key: String, expected: List[Int],
) raises:
    var info = st.tensor_info(key)
    if info.dtype != STDtype.I64 or info.size != 8 * len(expected):
        raise Error(String("MiniMax H3 index tensor mismatch: ") + key)
    if key == String("one_frame_target_index_int64"):
        if len(expected) != 1 or len(info.shape) != 0:
            raise Error("MiniMax H3 one-frame target index must be a scalar I64")
    elif (
        len(info.shape) != 1
        or info.shape[0] != len(expected)
    ):
        raise Error("MiniMax H3 one-frame control indices must be I64 [K]")
    var data = st.tensor_bytes(key)
    for index in range(len(expected)):
        if (data[index * 8 + 7] & UInt8(0x80)) != UInt8(0):
            raise Error(String("MiniMax H3 index must be nonnegative: ") + key)
        var actual = 0
        for byte in range(8):
            actual |= Int(data[index * 8 + byte]) << (8 * byte)
        if actual != expected[index]:
            raise Error(String("MiniMax H3 index payload mismatch: ") + key)


def _read_nonnegative_i64_scalar(st: SafeTensors, key: String) raises -> Int:
    var info = st.tensor_info(key)
    if info.dtype != STDtype.I64 or info.size != 8 or len(info.shape) != 0:
        raise Error(String("MiniMax H3 scalar index tensor mismatch: ") + key)
    var data = st.tensor_bytes(key)
    if (data[7] & UInt8(0x80)) != UInt8(0):
        raise Error(String("MiniMax H3 scalar index must be nonnegative: ") + key)
    var actual = 0
    for byte in range(8):
        actual |= Int(data[byte]) << (8 * byte)
    return actual


def _verify_visual_condition(st: SafeTensors, key: String) raises:
    var info = st.tensor_info(key)
    if (
        info.dtype != STDtype.F32 or len(info.shape) != 4
        or info.shape[0] != 24
    ):
        raise Error(String("MiniMax H3 visual condition mismatch: ") + key)


def _verify_audio_condition(st: SafeTensors, key: String) raises:
    var info = st.tensor_info(key)
    if (
        info.dtype != STDtype.F32 or len(info.shape) != 3
        or info.shape[0] != 32 or info.shape[1] != 2
    ):
        raise Error(String("MiniMax H3 audio condition mismatch: ") + key)


def _require_metadata(
    st: SafeTensors, key: String, expected: String, label: String,
) raises:
    if not st.has_metadata(key):
        raise Error(label + String(" metadata missing: ") + key)
    var actual = st.metadata_value(key)
    if actual != expected:
        raise Error(
            label + String(" metadata mismatch: ") + key
            + String(" expected=") + expected + String(" actual=") + actual
        )


def _reject_metadata(st: SafeTensors, key: String, label: String) raises:
    if st.has_metadata(key):
        raise Error(label + String(" metadata is not applicable: ") + key)


def _join_indices(indices: List[Int]) -> String:
    var out = String("")
    for index in range(len(indices)):
        if index > 0:
            out += String(";")
        out += String(indices[index])
    return out^


def _validate_live_media_binding(
    sample: MiniMaxH3CacheSample,
    dataset_root: String,
    task: String,
    one_frame: Bool,
) raises:
    var parsed = loads(sample.media_fingerprints)
    if not parsed.is_object():
        raise Error("MiniMax H3 media_fingerprints metadata must be a JSON object")
    var source_path = dataset_root + String("/") \
        + sample.source_image_relative_path
    if not parsed.contains(source_path) or not parsed[source_path].is_string():
        raise Error("MiniMax H3 media_fingerprints do not bind the live source image")
    var fingerprint = parsed[source_path].as_string()
    if not fingerprint.startswith(String("stat:")):
        raise Error("MiniMax H3 media fingerprint must use pinned Musubi stat identity")
    if one_frame and task == String("t2va") and parsed.length() != 1:
        raise Error("MiniMax H3 one-frame t2va media identity must contain only its target image")


def _verify_latent_artifact(
    sample: MiniMaxH3CacheSample,
    manifest: MiniMaxH3CacheManifest,
) raises:
    if minimax_h3_sha256_file(sample.latent_artifact_path) != sample.latent_artifact_sha256:
        raise Error("MiniMax H3 latent artifact checksum mismatch")
    var st = SafeTensors.open(sample.latent_artifact_path)
    _require_metadata(st, String("architecture"), String("minimax_h3"), String("MiniMax H3 latent"))
    _require_metadata(st, String("width"), String(sample.original_width), String("MiniMax H3 latent"))
    _require_metadata(st, String("height"), String(sample.original_height), String("MiniMax H3 latent"))
    _require_metadata(st, String("format_version"), String("1.0.1"), String("MiniMax H3 latent"))
    _require_metadata(st, String("task"), manifest.task, String("MiniMax H3 latent"))
    _require_metadata(st, String("cache_seed"), String(manifest.cache_seed), String("MiniMax H3 latent"))
    _require_metadata(st, String("crop_start_frame"), String(sample.crop_start_frame), String("MiniMax H3 latent"))
    _require_metadata(st, String("cache_format"), manifest.latent_cache_format, String("MiniMax H3 latent"))
    _require_metadata(
        st, String("video_vae_fingerprint"), manifest.video_vae_fingerprint,
        String("MiniMax H3 latent"),
    )
    _require_metadata(
        st, String("audio_vae_fingerprint"), manifest.audio_vae_fingerprint,
        String("MiniMax H3 latent"),
    )
    _require_metadata(
        st, String("media_fingerprints"), sample.media_fingerprints,
        String("MiniMax H3 latent"),
    )
    if manifest.one_frame:
        _require_metadata(st, String("one_frame"), String("1"), String("MiniMax H3 latent"))
        _require_metadata(
            st, String("one_frame_target_index"),
            String(sample.one_frame_target_index), String("MiniMax H3 latent"),
        )
        if len(sample.one_frame_control_indices) > 0:
            _require_metadata(
                st, String("one_frame_control_indices"),
                _join_indices(sample.one_frame_control_indices),
                String("MiniMax H3 latent"),
            )
        else:
            _reject_metadata(
                st, String("one_frame_control_indices"),
                String("MiniMax H3 latent"),
            )
    else:
        for key in [
            String("one_frame"), String("one_frame_target_index"),
            String("one_frame_control_indices"),
        ]:
            _reject_metadata(st, key, String("MiniMax H3 latent"))
    var video_key = (
        String("latents_") + String(sample.target_video_shape[1]) + String("x")
        + String(sample.target_video_shape[2]) + String("x")
        + String(sample.target_video_shape[3]) + String("_float32")
    )
    var audio_key = (
        String("latents_audio_32x2x") + String(sample.target_audio_shape[2])
        + String("_float32")
    )
    _require_tensor(st, video_key, STDtype.F32, sample.target_video_shape)
    _require_tensor(st, audio_key, STDtype.F32, sample.target_audio_shape)
    _require_tensor(st, String("audio_present_float32"), STDtype.F32, List[Int]())
    var actual_audio_present = read_f32_scalar_bytes(
        st.tensor_bytes(String("audio_present_float32"))
    )
    if actual_audio_present != sample.audio_present \
            or (actual_audio_present != 0.0 and actual_audio_present != 1.0):
        raise Error("MiniMax H3 audio_present payload mismatch")

    var first_count = 0
    var last_count = 0
    var ref_images = 0
    var ref_videos = 0
    var ref_audios = 0
    for key in st.names():
        if key == video_key or key == audio_key or key == String("audio_present_float32"):
            continue
        if key.startswith(String("latents_first_")):
            _verify_visual_condition(st, key)
            first_count += 1
        elif key.startswith(String("latents_last_")):
            _verify_visual_condition(st, key)
            last_count += 1
        elif key.startswith(String("latents_ref_")) and key.find(String("_image_")) >= 0:
            _verify_visual_condition(st, key)
            ref_images += 1
        elif key.startswith(String("latents_ref_")) and key.find(String("_video_")) >= 0:
            _verify_visual_condition(st, key)
            ref_videos += 1
        elif key.startswith(String("latents_ref_")) and key.find(String("_audio_")) >= 0:
            _verify_audio_condition(st, key)
            ref_audios += 1
        elif key == String("one_frame_target_index_int64"):
            _verify_exact_nonnegative_i64(
                st, key, [sample.one_frame_target_index],
            )
        elif key == String("one_frame_control_indices_int64"):
            _verify_exact_nonnegative_i64(
                st, key, sample.one_frame_control_indices,
            )
        else:
            raise Error(String("MiniMax H3 latent cache unsupported tensor: ") + key)
    if first_count != Int(sample.has_first_condition) \
            or last_count != Int(sample.has_last_condition):
        raise Error("MiniMax H3 latent condition-role count mismatch")
    if ref_images != sample.ref_image_count or ref_videos != sample.ref_video_count \
            or ref_audios != sample.ref_audio_count:
        raise Error("MiniMax H3 latent reference-role count mismatch")
    if sample.one_frame_target_index >= 0:
        if not st.has_tensor(String("one_frame_target_index_int64")):
            raise Error("MiniMax H3 one-frame target index tensor missing")
    elif st.has_tensor(String("one_frame_target_index_int64")):
        raise Error("MiniMax H3 video cache carries one-frame target index")
    if len(sample.one_frame_control_indices) > 0:
        if not st.has_tensor(String("one_frame_control_indices_int64")):
            raise Error("MiniMax H3 one-frame control index tensor missing")
    elif st.has_tensor(String("one_frame_control_indices_int64")):
        raise Error("MiniMax H3 cache carries unexpected one-frame controls")


def _verify_text_artifact(
    sample: MiniMaxH3CacheSample,
    manifest: MiniMaxH3CacheManifest,
    live_caption: String,
) raises:
    if minimax_h3_sha256_file(sample.text_artifact_path) != sample.text_artifact_sha256:
        raise Error("MiniMax H3 text artifact checksum mismatch")
    var st = SafeTensors.open(sample.text_artifact_path)
    _require_metadata(st, String("architecture"), String("minimax_h3"), String("MiniMax H3 text"))
    _require_metadata(st, String("caption1"), live_caption, String("MiniMax H3 text"))
    _require_metadata(st, String("format_version"), String("1.0.1"), String("MiniMax H3 text"))
    _require_metadata(st, String("task"), manifest.task, String("MiniMax H3 text"))
    _require_metadata(st, String("crop_start_frame"), String(sample.crop_start_frame), String("MiniMax H3 text"))
    _require_metadata(st, String("cache_format"), manifest.text_cache_format, String("MiniMax H3 text"))
    _require_metadata(
        st, String("text_encoder_fingerprint"),
        manifest.text_encoder_fingerprint, String("MiniMax H3 text"),
    )
    _require_metadata(
        st, String("processor_fingerprint"),
        manifest.processor_fingerprint, String("MiniMax H3 text"),
    )
    _require_metadata(
        st, String("presentation_fingerprint"),
        sample.presentation_fingerprint, String("MiniMax H3 text"),
    )
    _require_metadata(
        st, String("cache_dtype"), manifest.text_cache_dtype,
        String("MiniMax H3 text"),
    )
    var hidden_key = (
        String("varlen_mmh3_hidden_states_") + sample.text_hidden_dtype
    )
    var hidden_dtype = (
        STDtype.BF16 if sample.text_hidden_dtype == String("bfloat16")
        else STDtype.F32
    )
    _require_tensor(st, hidden_key, hidden_dtype, sample.text_hidden_shape)
    _require_tensor(
        st, String("varlen_mmh3_token_tags_int64"), STDtype.I64,
        sample.text_token_tags_shape,
    )
    if st.count() != 2:
        raise Error("MiniMax H3 first product seam rejects teacher/extra text tensors")
    _verify_binary_i64(st, String("varlen_mmh3_token_tags_int64"))


def _released_base_names() -> List[String]:
    var names = List[String]()
    for key in [
        String("video_patch_proj.weight"), String("video_patch_proj.bias"),
        String("audio_patch_proj.weight"), String("audio_patch_proj.bias"),
        String("condition_proj.weight"), String("condition_proj.bias"),
        String("time_embedder.proj_in.weight"), String("time_embedder.proj_in.bias"),
        String("time_embedder.proj_out.weight"), String("time_embedder.proj_out.bias"),
        String("token_refiner.final_norm.weight"), String("final_layer.norm.weight"),
        String("final_layer.adaln_proj.linear.weight"),
        String("final_layer.adaln_proj.linear.bias"),
        String("final_layer.video_out.weight"), String("final_layer.video_out.bias"),
        String("final_layer.audio_out.weight"), String("final_layer.audio_out.bias"),
        String("rope.inv_freq"),
    ]:
        names.append(key)
    for layer in range(50):
        var prefix = String("blocks.") + String(layer)
        for suffix in [
            String(".norm1.weight"), String(".norm2.weight"),
            String(".attn.q_norm.weight"), String(".attn.k_norm.weight"),
            String(".attn.qkv_proj.weight"), String(".attn.out_proj.weight"),
            String(".mlp.fc1.weight"), String(".mlp.fc2.weight"),
            String(".adaln_proj.linear.weight"),
            String(".adaln_proj.linear.bias"),
        ]:
            names.append(prefix + suffix)
    for layer in range(2):
        var prefix = String("token_refiner.blocks.") + String(layer)
        for suffix in [
            String(".norm1.weight"), String(".norm2.weight"),
            String(".attn.q_norm.weight"), String(".attn.k_norm.weight"),
            String(".attn.qkv_proj.weight"), String(".attn.out_proj.weight"),
            String(".mlp.fc1.weight"), String(".mlp.fc2.weight"),
        ]:
            names.append(prefix + suffix)
    return names^


def minimax_h3_released_base_names() -> List[String]:
    """Exact 535-key published raw MiniMax-H3 checkpoint inventory."""
    return _released_base_names()


def minimax_h3_released_base_shape(key: String) raises -> List[Int]:
    """Exact raw shape from the pinned published MiniMax-H3 geometry."""
    var config = minimax_h3_released_config()
    if key == String("video_patch_proj.weight"):
        return [config.hidden_size, config.video_patch_dim()]
    if key == String("video_patch_proj.bias"):
        return [config.hidden_size]
    if key == String("audio_patch_proj.weight"):
        return [config.hidden_size, config.audio_latents_dim]
    if key == String("audio_patch_proj.bias"):
        return [config.hidden_size]
    if key == String("condition_proj.weight"):
        return [config.hidden_size, config.text_dim]
    if key == String("condition_proj.bias"):
        return [config.hidden_size]
    if key == String("time_embedder.proj_in.weight"):
        return [config.hidden_size, config.timestep_input_dim]
    if key == String("time_embedder.proj_in.bias"):
        return [config.hidden_size]
    if key == String("time_embedder.proj_out.weight"):
        return [config.time_embed_dim, config.hidden_size]
    if key == String("time_embedder.proj_out.bias"):
        return [config.time_embed_dim]
    if key == String("token_refiner.final_norm.weight") \
            or key == String("final_layer.norm.weight"):
        return [config.hidden_size]
    if key == String("final_layer.adaln_proj.linear.weight"):
        return [config.final_adaln_out_features, config.time_embed_dim]
    if key == String("final_layer.adaln_proj.linear.bias"):
        return [config.final_adaln_out_features]
    if key == String("final_layer.video_out.weight"):
        return [config.video_patch_dim(), config.hidden_size]
    if key == String("final_layer.video_out.bias"):
        return [config.video_patch_dim()]
    if key == String("final_layer.audio_out.weight"):
        return [config.audio_latents_dim, config.hidden_size]
    if key == String("final_layer.audio_out.bias"):
        return [config.audio_latents_dim]
    if key == String("rope.inv_freq"):
        return [config.rope_inv_freq_len]
    if key.find(String(".adaln_proj.linear.weight")) >= 0:
        return [config.adaln_out_features, config.time_embed_dim]
    if key.find(String(".adaln_proj.linear.bias")) >= 0:
        return [config.adaln_out_features]
    return minimax_h3_expected_shape(key, config)


def validate_minimax_h3_released_base(path: String) raises:
    """Header-only released-base guard; no tensors or DeviceContext are made."""
    var base = ShardedSafeTensors.open(path)
    var expected = _released_base_names()
    if len(expected) != 535 or len(base.name_to_shard) != 535:
        raise Error("MiniMax H3 released base must expose exactly 535 raw tensors")
    for key in expected:
        if key not in base.name_to_shard:
            raise Error(String("MiniMax H3 released base missing raw key: ") + key)
        var required_dtype = (
            STDtype.F32
            if minimax_h3_is_fp32_key(key) or key == String("rope.inv_freq")
            else STDtype.BF16
        )
        var info = base.tensor_info(key)
        if info.dtype != required_dtype:
            raise Error(String("MiniMax H3 released base dtype mismatch: ") + key)
        if not _same_shape(info.shape, minimax_h3_released_base_shape(key)):
            raise Error(String("MiniMax H3 released base shape mismatch: ") + key)


struct MiniMaxH3CachedBatch(Movable):
    var target_video: Tensor
    var target_audio: Tensor
    var hidden_states: Tensor
    var token_tags: Tensor
    var condition_names: List[String]
    var conditions: List[ArcPointer[Tensor]]
    var audio_present: Float32
    # Host layout selector, verified against the cache artifact before load.
    # -1 is reserved for ordinary multi-frame caches.
    var one_frame_target_index: Int

    def __init__(
        out self, var target_video: Tensor, var target_audio: Tensor,
        var hidden_states: Tensor, var token_tags: Tensor,
        var condition_names: List[String],
        var conditions: List[ArcPointer[Tensor]], audio_present: Float32,
        one_frame_target_index: Int,
    ):
        self.target_video = target_video^
        self.target_audio = target_audio^
        self.hidden_states = hidden_states^
        self.token_tags = token_tags^
        self.condition_names = condition_names^
        self.conditions = conditions^
        self.audio_present = audio_present
        self.one_frame_target_index = one_frame_target_index


@fieldwise_init
struct MiniMaxH3CacheConsumer(Copyable, Movable):
    var manifest: MiniMaxH3CacheManifest

    def len(self) -> Int:
        return len(self.manifest.samples)

    def load(self, index: Int, ctx: DeviceContext) raises -> MiniMaxH3CachedBatch:
        if index < 0 or index >= self.len():
            raise Error("MiniMax H3 cache sample index out of range")
        var sample = self.manifest.samples[index].copy()
        var latent = SafeTensors.open(sample.latent_artifact_path)
        var text = SafeTensors.open(sample.text_artifact_path)
        var video_key = (
            String("latents_") + String(sample.target_video_shape[1]) + String("x")
            + String(sample.target_video_shape[2]) + String("x")
            + String(sample.target_video_shape[3]) + String("_float32")
        )
        var audio_key = String("latents_audio_32x2x") \
            + String(sample.target_audio_shape[2]) + String("_float32")
        var hidden_key = String("varlen_mmh3_hidden_states_") + sample.text_hidden_dtype
        var condition_names = List[String]()
        var conditions = List[ArcPointer[Tensor]]()
        for key in latent.names_storage_order():
            if (
                key.startswith(String("latents_first_"))
                or key.startswith(String("latents_last_"))
                or key.startswith(String("latents_ref_"))
            ):
                condition_names.append(key)
                var info = latent.tensor_info(key)
                var bytes = latent.tensor_bytes(key)
                conditions.append(ArcPointer(Tensor.from_view(
                    from_parts(info.dtype, info.shape.copy(), bytes), ctx,
                )))
        var video_info = latent.tensor_info(video_key)
        var video_bytes = latent.tensor_bytes(video_key)
        var audio_info = latent.tensor_info(audio_key)
        var audio_bytes = latent.tensor_bytes(audio_key)
        var hidden_info = text.tensor_info(hidden_key)
        var hidden_bytes = text.tensor_bytes(hidden_key)
        var tags_key = String("varlen_mmh3_token_tags_int64")
        var tags_info = text.tensor_info(tags_key)
        var tags_bytes = text.tensor_bytes(tags_key)
        var one_frame_target_index = -1
        var target_index_key = String("one_frame_target_index_int64")
        if latent.has_tensor(target_index_key):
            one_frame_target_index = _read_nonnegative_i64_scalar(
                latent, target_index_key,
            )
            if one_frame_target_index != sample.one_frame_target_index:
                raise Error(
                    "MiniMax H3 one-frame target index changed after preflight"
                )
        return MiniMaxH3CachedBatch(
            Tensor.from_view(from_parts(
                video_info.dtype, video_info.shape.copy(), video_bytes,
            ), ctx),
            Tensor.from_view(from_parts(
                audio_info.dtype, audio_info.shape.copy(), audio_bytes,
            ), ctx),
            Tensor.from_view(from_parts(
                hidden_info.dtype, hidden_info.shape.copy(), hidden_bytes,
            ), ctx),
            Tensor.from_view_raw(from_parts(
                tags_info.dtype, tags_info.shape.copy(), tags_bytes,
            ), ctx),
            condition_names^,
            conditions^,
            sample.audio_present,
            one_frame_target_index,
        )


def _bind_live_source(
    config: MiniMaxH3ProductConfig,
    manifest: MiniMaxH3CacheManifest,
) raises -> MiniMaxH3SourceReceipt:
    """Re-enumerate the authored source before opening any cache artifact."""
    var source = intake_minimax_h3_source_dataset(
        config.training.dataset_identity,
        config.training.dataset_path,
        config.dataset_trigger,
        config.training.dataset_fingerprint,
    )
    if len(source.samples) != len(manifest.samples):
        raise Error("MiniMax H3 live source/cache sample count mismatch")
    if not manifest.one_frame:
        raise Error(
            "MiniMax H3 current live source intake binds still-image one-frame caches only"
        )
    for index in range(len(source.samples)):
        var live = source.samples[index].copy()
        var cached = manifest.samples[index].copy()
        if (
            cached.source_image_relative_path != live.image_relative_path
            or cached.source_caption_relative_path != live.caption_relative_path
        ):
            raise Error("MiniMax H3 live source/cache relative path or order mismatch")
        if (
            cached.source_fingerprint != live.pair_fingerprint
            or cached.sample_id != live.pair_fingerprint
        ):
            raise Error("MiniMax H3 cache sample is not bound to the live source pair bytes")
        if (
            cached.original_width != live.original_width
            or cached.original_height != live.original_height
        ):
            raise Error("MiniMax H3 cache original dimensions do not match the live source image")
        if cached.source_frame_count != 1 or cached.crop_start_frame != 0:
            raise Error("MiniMax H3 live still-image source requires F=1 and crop_start_frame=0")
        _validate_live_media_binding(
            cached,
            config.training.dataset_path,
            manifest.task,
            manifest.one_frame,
        )
    return source^


def open_minimax_h3_cache(config: MiniMaxH3ProductConfig) raises -> MiniMaxH3CacheConsumer:
    """Complete host-only cache/base preflight. Safe before DeviceContext."""
    var digest = minimax_h3_sha256_file(config.manifest_path)
    if digest != config.manifest_sha256:
        raise Error("MiniMax H3 manifest checksum does not match config")
    var manifest = _parse_manifest(
        loads(_read_manifest_text(config.manifest_path)),
        config.manifest_path,
        digest,
    )
    validate_minimax_h3_launch_preflight(config.training, manifest)
    if manifest.schema != String(MINIMAX_H3_CACHE_MANIFEST_SCHEMA) \
            or manifest.oracle_commit != String(MINIMAX_H3_CACHE_ORACLE_COMMIT):
        raise Error("MiniMax H3 manifest pinned provenance mismatch")
    if _sample_ids_digest(manifest.samples) != manifest.sample_ids_digest:
        raise Error("MiniMax H3 sample-ID digest mismatch")
    if manifest.cache_seed != config.cache_seed:
        raise Error("MiniMax H3 manifest/config cache seed mismatch")
    if manifest.text_cache_dtype != config.text_cache_dtype:
        raise Error("MiniMax H3 manifest/config text cache dtype mismatch")
    var source = _bind_live_source(config, manifest)
    for index in range(len(manifest.samples)):
        var sample = manifest.samples[index].copy()
        var expected_hidden_dtype = (
            String("bfloat16")
            if manifest.text_cache_dtype == String("bf16")
            else String("float32")
        )
        if sample.text_hidden_dtype != expected_hidden_dtype:
            raise Error("MiniMax H3 text tensor dtype disagrees with cache_dtype metadata")
        _verify_latent_artifact(sample, manifest)
        _verify_text_artifact(sample, manifest, source.samples[index].caption)
    validate_minimax_h3_released_base(config.common.checkpoint)
    return MiniMaxH3CacheConsumer(manifest^)

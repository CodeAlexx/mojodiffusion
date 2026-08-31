# Deterministic MiniMax-H3 product preflight/cache-consumer component smoke.
#
# Synthetic tiny safetensors exercise strict config + manifest parsing, SHA-256,
# exact one-frame index payloads, the exact 535-key released-base inventory, and
# device transfer.  This is not dataset-cache generation, model execution, or a
# trainer gate.

from max.gpu.host import DeviceContext
from std.collections import Dict, List
from std.memory import alloc

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import (
    BytePtr,
    O_CREAT,
    O_TRUNC,
    O_WRONLY,
    sys_close,
    sys_mkdirs,
    sys_open,
    sys_pwrite,
)
from serenitymojo.io.safetensors_writer import (
    HostTensorDesc,
    save_safetensors_host_with_metadata,
)
from serenitymojo.serve.product_manifest import write_text_file
from serenitymojo.training.minimax_h3.cache_consumer import (
    minimax_h3_released_base_names,
    open_minimax_h3_cache,
)
from serenitymojo.training.minimax_h3.sha256 import (
    minimax_h3_sha256_file,
    minimax_h3_sha256_text,
)
from serenitymojo.training.minimax_h3.config import (
    read_minimax_h3_product_config,
)
from serenitymojo.training.minimax_h3.source_dataset import (
    intake_minimax_h3_source_dataset,
)


comptime ROOT = "/tmp/serenity_h3_product_preflight_v1"
comptime CACHE = "/tmp/serenity_h3_product_preflight_v1/cache"
comptime DATASET = "/tmp/serenity_h3_product_preflight_v1/eri_with_trigger"
comptime LATENT = "/tmp/serenity_h3_product_preflight_v1/cache/sample.latent.safetensors"
comptime TEXT = "/tmp/serenity_h3_product_preflight_v1/cache/sample.text.safetensors"
comptime BASE = "/tmp/serenity_h3_product_preflight_v1/released_base.safetensors"
comptime BAD_BASE = "/tmp/serenity_h3_product_preflight_v1/released_base_bad_shape.safetensors"
comptime MANIFEST = "/tmp/serenity_h3_product_preflight_v1/cache/cache_manifest.json"
comptime CONFIG = "/tmp/serenity_h3_product_preflight_v1/config.json"
comptime SHA_A = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
comptime SHA_B = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
comptime SHA_C = "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
comptime SHA_D = "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
comptime CAPTION = "  vrtlEri2 \"authored\"\nsecond line  \r\n"
comptime MEDIA_JSON = (
    "{\"/tmp/serenity_h3_product_preflight_v1/eri_with_trigger/sample.png\":"
    "\"stat:24:123456789\"}"
)
comptime MEDIA_JSON_ESCAPED = (
    "{\\\"/tmp/serenity_h3_product_preflight_v1/eri_with_trigger/sample.png\\\":"
    "\\\"stat:24:123456789\\\"}"
)


def _check(ok: Bool, label: String) raises:
    if not ok:
        raise Error(String("MiniMax H3 product preflight smoke failed: ") + label)


def _f32_bytes(count: Int) -> List[UInt8]:
    var out = List[UInt8](length=count * 4, fill=UInt8(0))
    for index in range(count):
        # Deterministic finite, nonzero 1.0f payload.
        out[4 * index + 2] = UInt8(0x80)
        out[4 * index + 3] = UInt8(0x3F)
    return out^


def _bf16_bytes(count: Int) -> List[UInt8]:
    var out = List[UInt8](length=count * 2, fill=UInt8(0))
    for index in range(count):
        out[2 * index] = UInt8(0x80)
        out[2 * index + 1] = UInt8(0x3F)
    return out^


def _i64_bytes(values: List[Int]) -> List[UInt8]:
    var out = List[UInt8](length=len(values) * 8, fill=UInt8(0))
    for index in range(len(values)):
        var value = values[index]
        for byte in range(8):
            out[index * 8 + byte] = UInt8((value >> (8 * byte)) & 0xFF)
    return out^


def _write_bytes(path: String, data: List[UInt8]) raises:
    var fd = sys_open(path, O_WRONLY | O_CREAT | O_TRUNC, 0o644)
    if fd < 0:
        raise Error(String("MiniMax H3 fixture cannot open ") + path)
    var buf = alloc[UInt8](len(data) if len(data) > 0 else 1)
    for index in range(len(data)):
        buf[index] = data[index]
    var wrote = sys_pwrite(
        fd, BytePtr(unsafe_from_address=Int(buf)), len(data), 0,
    )
    buf.free()
    _ = sys_close(fd)
    if wrote != len(data):
        raise Error(String("MiniMax H3 fixture short write: ") + path)


def _write_source(caption: String = String(CAPTION)) raises:
    _ = sys_mkdirs(String(DATASET))
    # Minimal deterministic PNG signature + IHDR dimensions. Source intake is
    # deliberately header-only and never decodes fixture pixels.
    _write_bytes(
        String(DATASET) + String("/sample.png"),
        [
            UInt8(0x89), UInt8(0x50), UInt8(0x4E), UInt8(0x47),
            UInt8(0x0D), UInt8(0x0A), UInt8(0x1A), UInt8(0x0A),
            UInt8(0), UInt8(0), UInt8(0), UInt8(13),
            UInt8(0x49), UInt8(0x48), UInt8(0x44), UInt8(0x52),
            UInt8(0), UInt8(0), UInt8(0), UInt8(8),
            UInt8(0), UInt8(0), UInt8(0), UInt8(8),
        ],
    )
    write_text_file(String(DATASET) + String("/sample.txt"), caption)


def _latent_metadata(mode: String) -> Dict[String, String]:
    var metadata = Dict[String, String]()
    if mode != String("missing"):
        metadata[String("architecture")] = String("minimax_h3")
    metadata[String("width")] = String("8")
    metadata[String("height")] = String("8")
    metadata[String("format_version")] = String("1.0.1")
    metadata[String("task")] = (
        String("fl2va") if mode == String("wrong_task") else String("t2va")
    )
    metadata[String("cache_seed")] = String("123")
    metadata[String("crop_start_frame")] = String("0")
    metadata[String("cache_format")] = (
        String("minimax-h3-latent-v1")
        if mode == String("v1") else String("minimax-h3-latent-v2")
    )
    metadata[String("video_vae_fingerprint")] = (
        String(SHA_D) if mode == String("wrong_vae") else String(SHA_A)
    )
    metadata[String("audio_vae_fingerprint")] = String(SHA_B)
    metadata[String("media_fingerprints")] = String(MEDIA_JSON)
    metadata[String("one_frame")] = String("1")
    metadata[String("one_frame_target_index")] = String("7")
    return metadata^


def _write_latent(
    target_index: Int, wrong_rank: Int = 0,
    metadata_mode: String = String("valid"),
) raises:
    var names = List[String]()
    for key in [
        String("latents_1x1x1_float32"),
        String("latents_audio_32x2x2_float32"),
        String("audio_present_float32"),
        String("one_frame_target_index_int64"),
    ]:
        names.append(key)
    var descs = List[HostTensorDesc]()
    descs.append(HostTensorDesc(STDtype.F32, [24, 1, 1, 1], _f32_bytes(24)))
    descs.append(HostTensorDesc(STDtype.F32, [32, 2, 2], _f32_bytes(128)))
    descs.append(HostTensorDesc(STDtype.F32, List[Int](), _f32_bytes(1)))
    var target_shape = List[Int]()
    if wrong_rank == 1:
        target_shape = [1]
    elif wrong_rank == 2:
        target_shape = [1, 1]
    descs.append(
        HostTensorDesc(STDtype.I64, target_shape^, _i64_bytes([target_index]))
    )
    var metadata = _latent_metadata(metadata_mode)
    save_safetensors_host_with_metadata(
        names^, descs^, metadata, String(LATENT),
    )


def _text_metadata(mode: String) -> Dict[String, String]:
    var metadata = Dict[String, String]()
    metadata[String("architecture")] = String("minimax_h3")
    metadata[String("caption1")] = (
        String("wrong caption") if mode == String("wrong_caption")
        else String(CAPTION)
    )
    metadata[String("format_version")] = String("1.0.1")
    metadata[String("task")] = String("t2va")
    metadata[String("crop_start_frame")] = String("0")
    metadata[String("cache_format")] = (
        String("minimax-h3-text-v1")
        if mode == String("v1") else String("minimax-h3-text-v2")
    )
    metadata[String("text_encoder_fingerprint")] = String(SHA_C)
    metadata[String("processor_fingerprint")] = String(SHA_D)
    metadata[String("presentation_fingerprint")] = String(SHA_A)
    metadata[String("cache_dtype")] = String("bf16")
    return metadata^


def _write_text_cache(metadata_mode: String = String("valid")) raises:
    var tags = _i64_bytes([0, 1])
    var metadata = _text_metadata(metadata_mode)
    save_safetensors_host_with_metadata(
        [
            String("varlen_mmh3_hidden_states_bfloat16"),
            String("varlen_mmh3_token_tags_int64"),
        ],
        [
            HostTensorDesc(STDtype.BF16, [2, 5120], _bf16_bytes(2 * 5120)),
            HostTensorDesc(STDtype.I64, [2], tags^),
        ],
        metadata,
        String(TEXT),
    )


def _manifest_text(latent_digest: String) raises -> String:
    var source = intake_minimax_h3_source_dataset(
        String("eri_with_trigger"), String(DATASET), String("vrtlEri2"),
    )
    var live = source.samples[0].copy()
    var sample_ids_digest = minimax_h3_sha256_text(
        live.pair_fingerprint + String("\n")
    )
    return (
        String("{")
        + String("\"schema\":\"serenity.minimax_h3.training_cache.v2\",")
        + String("\"oracle_commit\":\"b8717864713c9e4e7ef3d56eba1fc695a9b626a5\",")
        + String("\"dataset_identity\":\"eri_with_trigger\",")
        + String("\"canonical_dataset_path\":\"") + String(DATASET) + String("\",")
        + String("\"dataset_fingerprint\":\"") + source.receipt + String("\",")
        + String("\"cache_root\":\"") + String(CACHE) + String("\",")
        + String("\"cache_namespace\":\"minimax_h3/eri_with_trigger/")
        + source.receipt + String("/t2va/one_frame\",")
        + String("\"task\":\"t2va\",\"one_frame\":true,")
        + String("\"cache_seed\":123,\"text_cache_dtype\":\"bf16\",")
        + String("\"latent_cache_format\":\"minimax-h3-latent-v2\",")
        + String("\"text_cache_format\":\"minimax-h3-text-v2\",")
        + String("\"video_vae_fingerprint\":\"") + String(SHA_A) + String("\",")
        + String("\"audio_vae_fingerprint\":\"") + String(SHA_B) + String("\",")
        + String("\"text_encoder_fingerprint\":\"") + String(SHA_C) + String("\",")
        + String("\"processor_fingerprint\":\"") + String(SHA_D) + String("\",")
        + String("\"sample_ids_digest\":\"") + sample_ids_digest + String("\",")
        + String("\"samples\":[{")
        + String("\"sample_id\":\"") + live.pair_fingerprint + String("\",")
        + String("\"source_fingerprint\":\"") + live.pair_fingerprint + String("\",")
        + String("\"source_image_relative_path\":\"sample.png\",")
        + String("\"source_caption_relative_path\":\"sample.txt\",")
        + String("\"original_width\":8,\"original_height\":8,")
        + String("\"media_fingerprints\":\"") + String(MEDIA_JSON_ESCAPED) + String("\",")
        + String("\"presentation_fingerprint\":\"") + String(SHA_A) + String("\",")
        + String("\"crop_start_frame\":0,")
        + String("\"latent_artifact_path\":\"") + String(LATENT) + String("\",")
        + String("\"latent_artifact_sha256\":\"") + latent_digest + String("\",")
        + String("\"text_artifact_path\":\"") + String(TEXT) + String("\",")
        + String("\"text_artifact_sha256\":\"") + minimax_h3_sha256_file(String(TEXT)) + String("\",")
        + String("\"source_frame_count\":1,")
        + String("\"target_video_shape\":[24,1,1,1],")
        + String("\"target_audio_shape\":[32,2,2],")
        + String("\"latent_dtype\":\"float32\",\"audio_present_dtype\":\"float32\",")
        + String("\"audio_present\":1.0,\"audio_present_binary_verified\":true,")
        + String("\"text_hidden_shape\":[2,5120],\"text_hidden_dtype\":\"bfloat16\",")
        + String("\"text_token_tags_shape\":[2],\"text_token_tags_dtype\":\"int64\",")
        + String("\"text_token_tags_binary_verified\":true,\"teacher_text_present\":false,")
        + String("\"has_first_condition\":false,\"has_last_condition\":false,")
        + String("\"ref_image_count\":0,\"ref_video_count\":0,")
        + String("\"ref_audio_count\":0,\"ref_item_count\":0,")
        + String("\"one_frame_target_index\":7,\"one_frame_control_indices\":[]")
        + String("}]}")
    )


def _config_text(
    identity: String, task: String, base_storage: String,
    checkpoint: String, manifest_digest: String, extra_h3: String = "",
) raises -> String:
    var source = intake_minimax_h3_source_dataset(
        String("eri_with_trigger"), String(DATASET), String("vrtlEri2"),
    )
    return (
        String("{")
        + String("\"model_type\":\"minimax_h3\",\"checkpoint\":\"")
        + checkpoint + String("\",\"training_method\":\"LORA\",")
        + String("\"batch_size\":1,\"gradient_accumulation_steps\":1,")
        + String("\"train_dtype\":\"BFLOAT_16\",\"transformer_weight_dtype\":\"BFLOAT_16\",")
        + String("\"lora_rank\":4,\"lora_alpha\":4.0,\"only_cache\":false,")
        + String("\"optimizer\":{\"optimizer\":\"ADAMW\"},\"minimax_h3\":{")
        + String("\"schema\":\"serenity.minimax_h3.training_run.v2\",")
        + String("\"dataset_identity\":\"") + identity + String("\",")
        + String("\"canonical_dataset_path\":\"") + String(DATASET) + String("\",")
        + String("\"dataset_fingerprint\":\"") + source.receipt + String("\",")
        + String("\"dataset_trigger\":\"vrtlEri2\",")
        + String("\"cache_seed\":123,\"text_cache_dtype\":\"bf16\",")
        + String("\"manifest_path\":\"") + String(MANIFEST) + String("\",")
        + String("\"manifest_sha256\":\"") + manifest_digest + String("\",")
        + String("\"video_vae_fingerprint\":\"") + String(SHA_A) + String("\",")
        + String("\"audio_vae_fingerprint\":\"") + String(SHA_B) + String("\",")
        + String("\"text_encoder_fingerprint\":\"") + String(SHA_C) + String("\",")
        + String("\"processor_fingerprint\":\"") + String(SHA_D) + String("\",")
        + String("\"task\":\"") + task + String("\",\"one_frame\":true,")
        + String("\"batch_size\":1,\"gradient_accumulation_steps\":1,")
        + String("\"full_finetune\":false,\"timestep_sampling\":\"uniform\",")
        + String("\"weighting_scheme\":\"none\",\"discrete_flow_shift\":1.0,")
        + String("\"operating_bf16\":true,\"lora_parameter_storage\":\"f32\",")
        + String("\"lora_compute_bf16\":true,\"lora_gradient_storage\":\"f32\",")
        + String("\"optimizer_state_storage\":\"f32\",\"base_storage\":\"")
        + base_storage + String("\",\"base_frozen\":true,")
        + String("\"video_shift\":12.0,\"audio_shift\":3.0,")
        + String("\"audio_loss_weight\":1.0,\"video_only\":false")
        + extra_h3 + String("}}")
    )


def _write_manifest_and_config(
    identity: String = "eri_with_trigger", task: String = "t2va",
    base_storage: String = "bf16", checkpoint: String = String(BASE),
    extra_h3: String = "",
) raises:
    write_text_file(
        String(MANIFEST),
        _manifest_text(minimax_h3_sha256_file(String(LATENT))),
    )
    write_text_file(
        String(CONFIG),
        _config_text(
            identity, task, base_storage, checkpoint,
            minimax_h3_sha256_file(String(MANIFEST)), extra_h3,
        ),
    )


def _expect_config_rejects(label: String) raises:
    var rejected = False
    try:
        _ = read_minimax_h3_product_config(String(CONFIG))
    except:
        rejected = True
    _check(rejected, label)


def _expect_consumer_rejects(label: String) raises:
    var rejected = False
    try:
        var config = read_minimax_h3_product_config(String(CONFIG))
        _ = open_minimax_h3_cache(config)
    except:
        rejected = True
    _check(rejected, label)


def main() raises:
    _check(
        minimax_h3_sha256_text(String(""))
        == String("sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
        "SHA-256 empty known vector",
    )
    _check(
        minimax_h3_sha256_text(String("abc"))
        == String("sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
        "SHA-256 abc known vector",
    )
    _ = sys_mkdirs(String(CACHE))
    _write_source()
    _write_latent(7)
    _write_text_cache()
    _check(
        len(minimax_h3_released_base_names()) == 535,
        "released raw-key inventory count",
    )
    _write_manifest_and_config()

    var valid = read_minimax_h3_product_config(String(CONFIG))
    var host_consumer = open_minimax_h3_cache(valid)
    _check(host_consumer.len() == 1, "valid host consumer")

    _write_latent(7, metadata_mode=String("missing"))
    _write_manifest_and_config()
    _expect_consumer_rejects("missing latent artifact metadata")
    _write_latent(7, metadata_mode=String("v1"))
    _write_manifest_and_config()
    _expect_consumer_rejects("stale latent-v1 artifact metadata")
    _write_latent(7, metadata_mode=String("wrong_task"))
    _write_manifest_and_config()
    _expect_consumer_rejects("wrong latent task metadata")
    _write_latent(7, metadata_mode=String("wrong_vae"))
    _write_manifest_and_config()
    _expect_consumer_rejects("wrong latent VAE metadata")
    _write_latent(7)
    _write_text_cache(String("v1"))
    _write_manifest_and_config()
    _expect_consumer_rejects("stale text-v1 artifact metadata")
    _write_text_cache(String("wrong_caption"))
    _write_manifest_and_config()
    _expect_consumer_rejects("wrong live caption metadata")
    _write_text_cache()
    _write_manifest_and_config()
    _write_source(String("vrtlEri2 mutated after manifest\n"))
    _expect_consumer_rejects("live source pair changed after manifest")
    _write_source()

    _write_manifest_and_config(identity=String("eri2_with_trigger"))
    _expect_config_rejects("substitute dataset identity")
    _write_manifest_and_config(base_storage=String("convrot_int8"))
    _expect_config_rejects("convrot INT8 launch")
    _write_manifest_and_config(task=String("fl2va"))
    _expect_consumer_rejects("config/manifest task mismatch")
    _write_manifest_and_config(checkpoint=String(ROOT) + String("/missing.safetensors"))
    _expect_consumer_rejects("missing released base")
    _write_manifest_and_config(checkpoint=String(BAD_BASE))
    _expect_consumer_rejects("released base wrong geometry")
    _write_manifest_and_config(extra_h3=String(",\"unknown_h3_field\":1"))
    _expect_config_rejects("unknown H3-specific config key")

    _write_latent(8)
    _write_manifest_and_config()
    _expect_consumer_rejects("manifest/index payload mismatch")

    _write_latent(7, wrong_rank=1)
    _write_manifest_and_config()
    _expect_consumer_rejects("rank-1 target index is not a scalar")
    _write_latent(7, wrong_rank=2)
    _write_manifest_and_config()
    _expect_consumer_rejects("rank-2 target index is not a scalar")

    # Restore the valid fixture; only now is a DeviceContext constructed.
    _write_latent(7)
    _write_manifest_and_config()
    var restored = read_minimax_h3_product_config(String(CONFIG))
    var consumer = open_minimax_h3_cache(restored)
    var ctx = DeviceContext()
    var batch = consumer.load(0, ctx)
    _check(batch.target_video.shape() == [24, 1, 1, 1], "device video shape")
    _check(batch.target_audio.shape() == [32, 2, 2], "device audio shape")
    _check(batch.hidden_states.shape() == [2, 5120], "device text shape")
    _check(batch.token_tags.shape() == [2], "device tags shape")
    _check(batch.target_video.dtype() == STDtype.F32, "device video dtype")
    _check(batch.hidden_states.dtype() == STDtype.BF16, "device text dtype")
    _check(batch.token_tags.dtype() == STDtype.I64, "device tags dtype")
    _check(batch.audio_present == Float32(1.0), "audio presence scalar")
    _check(batch.one_frame_target_index == 7, "one-frame target index handoff")
    _check(len(batch.conditions) == 0, "t2va has no conditions")
    print("PASS MiniMax H3 product preflight/cache consumer component smoke")
    print("  exact dataset identity: eri_with_trigger")
    print("  host rejection gates precede DeviceContext")
    print("  device load: storage dtypes/shapes preserved")
    print("  trainer/model backward/update/save-resume: NOT TESTED")

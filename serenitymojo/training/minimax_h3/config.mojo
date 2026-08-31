# MiniMax-H3 product training-config boundary.
#
# The shared TrainConfig reader intentionally tolerates unknown keys for older
# presets. H3 therefore owns one strict nested object: every field below is
# required, type-checked, and rejected if unknown. This is the only requested
# run preset identity: eri_with_trigger. Its path/fingerprints are supplied by
# the trainer-owned config and are never guessed from a similarly named run.

from json.parser import loads
from json.value import JSONValue
from std.collections import List
from std.math import isfinite
from std.memory import alloc

from serenitymojo.io.ffi import (
    O_RDONLY,
    file_size,
    sys_close,
    sys_open,
    sys_pread,
)
from serenitymojo.io.train_config_reader import read_model_config
from serenitymojo.training.train_config import (
    TRAINING_METHOD_LORA,
    TRAIN_DTYPE_BFLOAT_16,
    TRAIN_OPTIMIZER_ADAMW,
    TrainConfig,
)
from serenitymojo.training.minimax_h3.cache_manifest import (
    minimax_h3_is_canonical_absolute_path,
    minimax_h3_is_sha256_receipt,
)
from serenitymojo.training.minimax_h3.contract import (
    MINIMAX_H3_INTAKE_DATASET_IDENTITY,
    MiniMaxH3TrainingContract,
    validate_minimax_h3_policy,
)


comptime MINIMAX_H3_PRODUCT_CONFIG_SCHEMA = (
    "serenity.minimax_h3.training_run.v2"
)
comptime MINIMAX_H3_PRODUCT_DATASET_TRIGGER = "vrtlEri2"


@fieldwise_init
struct MiniMaxH3ProductConfig(Copyable, Movable):
    var common: TrainConfig
    var training: MiniMaxH3TrainingContract
    var dataset_trigger: String
    var cache_seed: Int
    var text_cache_dtype: String
    var manifest_path: String
    var manifest_sha256: String


def _read_text(path: String) raises -> String:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("MiniMax H3 config cannot open ") + path)
    var size = file_size(fd)
    if size <= 0:
        _ = sys_close(fd)
        raise Error(String("MiniMax H3 config is empty: ") + path)
    var buf = alloc[UInt8](size)
    var done = 0
    while done < size:
        var count = sys_pread(fd, buf + done, size - done, done)
        if count <= 0:
            buf.free()
            _ = sys_close(fd)
            raise Error(String("MiniMax H3 config short read: ") + path)
        done += count
    _ = sys_close(fd)
    var text = String(
        StringSlice(unsafe_from_utf8=Span(unsafe_ptr=buf, length=size))
    )
    buf.free()
    return text^


def _required_keys() -> List[String]:
    return [
        String("schema"),
        String("dataset_identity"),
        String("canonical_dataset_path"),
        String("dataset_fingerprint"),
        String("dataset_trigger"),
        String("cache_seed"),
        String("text_cache_dtype"),
        String("manifest_path"),
        String("manifest_sha256"),
        String("video_vae_fingerprint"),
        String("audio_vae_fingerprint"),
        String("text_encoder_fingerprint"),
        String("processor_fingerprint"),
        String("task"),
        String("one_frame"),
        String("batch_size"),
        String("gradient_accumulation_steps"),
        String("full_finetune"),
        String("timestep_sampling"),
        String("weighting_scheme"),
        String("discrete_flow_shift"),
        String("operating_bf16"),
        String("lora_parameter_storage"),
        String("lora_compute_bf16"),
        String("lora_gradient_storage"),
        String("optimizer_state_storage"),
        String("base_storage"),
        String("base_frozen"),
        String("video_shift"),
        String("audio_shift"),
        String("audio_loss_weight"),
        String("video_only"),
    ]


def _validate_exact_keys(obj: JSONValue) raises:
    if not obj.is_object():
        raise Error("MiniMax H3 config field 'minimax_h3' must be an object")
    var expected = _required_keys()
    if obj.length() != len(expected):
        raise Error(
            String("MiniMax H3 config must contain exactly ")
            + String(len(expected)) + String(" H3 fields; got ")
            + String(obj.length())
        )
    for key in expected:
        if not obj.contains(key):
            raise Error(String("MiniMax H3 config missing field: ") + key)
    for key in obj.keys():
        if key not in expected:
            raise Error(String("MiniMax H3 config unknown field: ") + key)


def _string(obj: JSONValue, key: String) raises -> String:
    var value = obj[key]
    if not value.is_string():
        raise Error(String("MiniMax H3 config field must be string: ") + key)
    return value.as_string()


def _bool(obj: JSONValue, key: String) raises -> Bool:
    var value = obj[key]
    if not value.is_bool():
        raise Error(String("MiniMax H3 config field must be bool: ") + key)
    return value.as_bool()


def _int(obj: JSONValue, key: String) raises -> Int:
    var value = obj[key]
    if not value.is_int():
        raise Error(String("MiniMax H3 config field must be integer: ") + key)
    return value.as_int()


def _float(obj: JSONValue, key: String) raises -> Float32:
    var value = obj[key]
    if not value.is_number():
        raise Error(String("MiniMax H3 config field must be numeric: ") + key)
    var out = Float32(value.as_float())
    if not isfinite(out):
        raise Error(String("MiniMax H3 config field must be finite: ") + key)
    return out


def _parse_h3(obj: JSONValue) raises -> MiniMaxH3ProductConfig:
    _validate_exact_keys(obj)
    var contract = MiniMaxH3TrainingContract(
        _string(obj, String("dataset_identity")),
        _string(obj, String("canonical_dataset_path")),
        _string(obj, String("dataset_fingerprint")),
        _string(obj, String("video_vae_fingerprint")),
        _string(obj, String("audio_vae_fingerprint")),
        _string(obj, String("text_encoder_fingerprint")),
        _string(obj, String("processor_fingerprint")),
        _string(obj, String("task")),
        _bool(obj, String("one_frame")),
        _int(obj, String("batch_size")),
        _int(obj, String("gradient_accumulation_steps")),
        _bool(obj, String("full_finetune")),
        _string(obj, String("timestep_sampling")),
        _string(obj, String("weighting_scheme")),
        _float(obj, String("discrete_flow_shift")),
        _bool(obj, String("operating_bf16")),
        _string(obj, String("lora_parameter_storage")),
        _bool(obj, String("lora_compute_bf16")),
        _string(obj, String("lora_gradient_storage")),
        _string(obj, String("optimizer_state_storage")),
        _string(obj, String("base_storage")),
        _bool(obj, String("base_frozen")),
        _float(obj, String("video_shift")),
        _float(obj, String("audio_shift")),
        _float(obj, String("audio_loss_weight")),
        _bool(obj, String("video_only")),
    )
    var common = TrainConfig.default()
    return MiniMaxH3ProductConfig(
        common^,
        contract^,
        _string(obj, String("dataset_trigger")),
        _int(obj, String("cache_seed")),
        _string(obj, String("text_cache_dtype")),
        _string(obj, String("manifest_path")),
        _string(obj, String("manifest_sha256")),
    )


def validate_minimax_h3_product_config(config: MiniMaxH3ProductConfig) raises:
    if config.common.name != String("minimax_h3"):
        raise Error("MiniMax H3 product config model_type must be minimax_h3")
    if config.training.dataset_identity != String(MINIMAX_H3_INTAKE_DATASET_IDENTITY):
        raise Error(
            String("MiniMax H3 requested dataset identity must be exactly ")
            + String(MINIMAX_H3_INTAKE_DATASET_IDENTITY)
        )
    if config.dataset_trigger != String(MINIMAX_H3_PRODUCT_DATASET_TRIGGER):
        raise Error(
            "MiniMax H3 eri_with_trigger preset requires dataset_trigger=vrtlEri2"
        )
    if (
        config.text_cache_dtype != String("bf16")
        and config.text_cache_dtype != String("float32")
    ):
        raise Error("MiniMax H3 text_cache_dtype must be bf16 or float32")
    validate_minimax_h3_policy(config.training)
    if config.training.base_storage != String("bf16"):
        raise Error(
            "MiniMax H3 product training requires the released mixed BF16/F32"
            " base; convrot_int8 has no backward"
        )
    if config.common.checkpoint.byte_length() == 0:
        raise Error("MiniMax H3 BF16 base checkpoint path is missing")
    if not minimax_h3_is_canonical_absolute_path(config.common.checkpoint):
        raise Error("MiniMax H3 BF16 base checkpoint path must be canonical absolute")
    if not minimax_h3_is_canonical_absolute_path(config.training.dataset_path):
        raise Error("MiniMax H3 dataset path must be canonical absolute")
    if not minimax_h3_is_canonical_absolute_path(config.manifest_path):
        raise Error("MiniMax H3 manifest path must be canonical absolute")
    if not minimax_h3_is_sha256_receipt(config.manifest_sha256):
        raise Error("MiniMax H3 manifest SHA-256 receipt is missing")
    for receipt in [
        config.training.dataset_fingerprint,
        config.training.video_vae_fingerprint,
        config.training.audio_vae_fingerprint,
        config.training.text_encoder_fingerprint,
        config.training.processor_fingerprint,
    ]:
        if not minimax_h3_is_sha256_receipt(receipt):
            raise Error("MiniMax H3 config carries an unresolved fingerprint")
    if config.common.training_method != TRAINING_METHOD_LORA:
        raise Error("MiniMax H3 product seam supports LoRA only")
    if config.common.batch_size != config.training.batch_size:
        raise Error("MiniMax H3 common/H3 batch_size mismatch")
    if config.common.grad_accum_steps != config.training.gradient_accumulation_steps:
        raise Error("MiniMax H3 common/H3 gradient accumulation mismatch")
    if config.common.train_dtype != TRAIN_DTYPE_BFLOAT_16:
        raise Error("MiniMax H3 train_dtype must be BFLOAT_16")
    if config.common.transformer_weight_dtype != TRAIN_DTYPE_BFLOAT_16:
        raise Error("MiniMax H3 transformer_weight_dtype must be BFLOAT_16")
    if config.common.optimizer != TRAIN_OPTIMIZER_ADAMW:
        raise Error("MiniMax H3 first product seam supports ADAMW only")
    if config.common.only_cache:
        raise Error("MiniMax H3 native cache builder is not product-wired")
    if config.common.lora_rank <= 0 or config.common.lora_alpha <= Float32(0.0):
        raise Error("MiniMax H3 LoRA rank and alpha must be positive")


def read_minimax_h3_product_config(path: String) raises -> MiniMaxH3ProductConfig:
    """Read shared fields plus the strict, complete `minimax_h3` object."""
    var common = read_model_config(path)
    var root = loads(_read_text(path))
    if not root.is_object() or not root.contains(String("minimax_h3")):
        raise Error("MiniMax H3 config requires a strict 'minimax_h3' object")
    var parsed = _parse_h3(root[String("minimax_h3")])
    parsed.common = common^
    if _string(root[String("minimax_h3")], String("schema")) \
            != String(MINIMAX_H3_PRODUCT_CONFIG_SCHEMA):
        raise Error("MiniMax H3 product config schema mismatch")
    validate_minimax_h3_product_config(parsed)
    return parsed^

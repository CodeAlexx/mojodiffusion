# onetrainer_policy_config_smoke.mojo - focused OneTrainer config-reader slice.
#
# Scope: scalar and nested-object config parsing only. This does not load model
# weights, allocate device tensors, or run a train loop.
#
# Run:
#   timeout 180 prlimit --as=16000000000 pixi run mojo run -I . serenitymojo/io/onetrainer_policy_config_smoke.mojo

from std.memory import alloc

from serenitymojo.io.ffi import (
    sys_open, sys_close, sys_pwrite, BytePtr,
    O_WRONLY, O_CREAT, O_TRUNC,
)
from serenitymojo.io.train_config_reader import read_model_config
from serenitymojo.training.train_config import (
    TrainConfig,
    TRAINING_METHOD_LORA, TRAINING_METHOD_FINE_TUNE,
    GRADIENT_CHECKPOINTING_ON, GRADIENT_CHECKPOINTING_CPU_OFFLOADED,
    TRAIN_DTYPE_FLOAT_8, TRAIN_DTYPE_FLOAT_32, TRAIN_DTYPE_BFLOAT_16,
    TRAIN_OPTIMIZER_ADAMW, TRAIN_OPTIMIZER_ADAFACTOR,
    TRAIN_TIME_UNIT_STEP, TRAIN_TIME_UNIT_NEVER,
    EMA_MODE_GPU,
)


def _write_file(path: String, content: String) raises:
    var fd = sys_open(path, O_CREAT | O_WRONLY | O_TRUNC, Int32(0o644))
    if fd < 0:
        raise Error(String("OneTrainer policy smoke: cannot create ") + path)
    var n = content.byte_length()
    var buf = alloc[UInt8](n if n > 0 else 1)
    var src = content.as_bytes()
    for i in range(n):
        buf[i] = src[i]
    var wrote = sys_pwrite(fd, BytePtr(unsafe_from_address=Int(buf)), n, 0)
    buf.free()
    _ = sys_close(fd)
    if wrote != n:
        raise Error(String("OneTrainer policy smoke: short write to ") + path)


def _eq(name: String, a: Int, b: Int) raises:
    if a != b:
        raise Error(name + " expected " + String(b) + ", got " + String(a))


def _equ64(name: String, a: UInt64, b: UInt64) raises:
    if a != b:
        raise Error(name + " expected " + String(b) + ", got " + String(a))


def _eqb(name: String, a: Bool, b: Bool) raises:
    if a != b:
        raise Error(name + " expected " + String(b) + ", got " + String(a))


def _eqs(name: String, a: String, b: String) raises:
    if a != b:
        raise Error(name + " expected " + b + ", got " + a)


def _close(name: String, a: Float32, b: Float32) raises:
    var d = a - b
    var ad = d if d >= Float32(0.0) else -d
    if ad > Float32(1e-6):
        raise Error(name + " mismatch: got " + String(a) + " expected " + String(b))


def _read_json(path: String, body: String) raises -> TrainConfig:
    var js = String("{") + body + String("}")
    _write_file(path, js)
    return read_model_config(path)


def _gate_qwen_lora_policy() raises:
    print("--- OneTrainer Qwen LoRA-style policy fields ---")
    var body = String("")
    body += '"model_type":"QWEN",'
    body += '"training_method":"LORA",'
    body += '"base_model_name":"/models/qwen-image",'
    body += '"workspace_dir":"/workspace/qwen",'
    body += '"cache_dir":"/workspace-cache/qwen",'
    body += '"output_model_destination":"/output/qwen/lora.safetensors",'
    body += '"output_model_format":"SAFETENSORS",'
    body += '"concept_file_name":"/concepts/qwen.json",'
    body += '"sample_definition_file_name":"/samples/qwen.json",'
    body += '"validation":true,"validate_after":5,"validate_after_unit":"STEP",'
    body += '"continue_last_backup":true,'
    body += '"sample_after":25,"sample_after_unit":"STEP","sample_skip_first":3,'
    body += '"samples_to_tensorboard":false,"non_ema_sampling":false,'
    body += '"backup_after":0,"backup_after_unit":"NEVER","backup_before_save":false,'
    body += '"save_every":50,"save_every_unit":"STEP","save_skip_first":10,'
    body += '"save_filename_prefix":"qwen_",'
    body += '"batch_size":2,"epochs":7,"stop_training_after":100,'
    body += '"stop_training_after_unit":"STEP","resolution":"512","frames":"1",'
    body += '"seed":123,"compile":true,"dataloader_threads":1,'
    body += '"latent_caching":true,"clear_cache_before_training":false,"only_cache":false,'
    body += '"learning_rate":0.0003,"learning_rate_scheduler":"CONSTANT",'
    body += '"learning_rate_warmup_steps":100,"gradient_accumulation_steps":2,'
    body += '"clip_grad_norm":0.75,'
    body += '"train_dtype":"BFLOAT_16","fallback_train_dtype":"BFLOAT_16",'
    body += '"weight_dtype":"BFLOAT_16","output_dtype":"BFLOAT_16",'
    body += '"transformer":{"train":true,"model_name":"/parts/transformer","weight_dtype":"FLOAT_8"},'
    body += '"text_encoder":{"train":false,"model_name":"/parts/text","weight_dtype":"FLOAT_8"},'
    body += '"vae":{"model_name":"/parts/vae","weight_dtype":"FLOAT_32"},'
    body += '"layer_filter":"attn,img_mlp,txt_mlp","layer_filter_preset":"attn-mlp",'
    body += '"layer_filter_regex":false,'
    body += '"quantization":{"layer_filter":"transformer_block","layer_filter_preset":"blocks"},'
    body += '"optimizer":{"optimizer":"ADAMW","beta1":0.91,"beta2":0.98,'
    body += '"eps":0.000001,"weight_decay":0.02,"fused":false,'
    body += '"fused_back_pass":false,"stochastic_rounding":false},'
    body += '"ema":"GPU","ema_decay":0.998,"ema_update_step_interval":4,'
    body += '"masked_training":true,"unmasked_probability":0.2,"unmasked_weight":0.3,'
    body += '"normalize_masked_area_loss":true,"masked_prior_preservation_weight":0.4,'
    body += '"custom_conditioning_image":true,'
    body += '"gradient_checkpointing":"ON"'

    var c = _read_json(String("/tmp/onetrainer_policy_qwen.json"), body)
    _eqs("model_type", c.name, String("QWEN"))
    _eq("training_method", c.training_method, TRAINING_METHOD_LORA)
    _eqs("base_model_name", c.base_model_name, String("/models/qwen-image"))
    _eqs("workspace_dir", c.workspace_dir, String("/workspace/qwen"))
    _eqs("cache_dir", c.cache_dir, String("/workspace-cache/qwen"))
    _eqs("dataset_cache_dir mirror", c.dataset_cache_dir, String("/workspace-cache/qwen"))
    _eqs("output_model_destination", c.output_model_destination, String("/output/qwen/lora.safetensors"))
    _eqs("concept_file_name", c.concept_file_name, String("/concepts/qwen.json"))
    _eqs("sample_definition_file_name", c.sample_definition_file_name, String("/samples/qwen.json"))
    _eqs("validation_prompts_file fallback", c.validation_prompts_file, String("/samples/qwen.json"))
    _eqb("validation", c.validation, True)
    _eq("validate_after_unit", c.validate_after_unit, TRAIN_TIME_UNIT_STEP)
    _eqb("continue_last_backup", c.continue_last_backup, True)
    _eq("sample_every from sample_after", c.sample_every, 25)
    _eq("save_every", c.save_every, 50)
    _eq("save_every_unit", c.save_every_unit, TRAIN_TIME_UNIT_STEP)
    _eq("backup_after_unit", c.backup_after_unit, TRAIN_TIME_UNIT_NEVER)
    _eqb("backup_before_save", c.backup_before_save, False)
    _eq("batch_size", c.batch_size, 2)
    _eq("max_steps from STEP stop_training_after", c.max_steps, 100)
    _equ64("seed", c.seed, UInt64(123))
    _eqb("compile", c.compile_model, True)
    _eq("train_dtype", c.train_dtype, TRAIN_DTYPE_BFLOAT_16)
    _eq("transformer dtype", c.transformer_weight_dtype, TRAIN_DTYPE_FLOAT_8)
    _eq("text_encoder dtype", c.text_encoder_weight_dtype, TRAIN_DTYPE_FLOAT_8)
    _eq("vae dtype", c.vae_weight_dtype, TRAIN_DTYPE_FLOAT_32)
    _eqb("text_encoder train", c.text_encoder_train, False)
    _eqs("transformer model_name", c.transformer_model_name, String("/parts/transformer"))
    _eqs("layer_filter", c.layer_filter, String("attn,img_mlp,txt_mlp"))
    _eqs("quantization layer_filter", c.quantization_layer_filter, String("transformer_block"))
    _eq("optimizer", c.optimizer, TRAIN_OPTIMIZER_ADAMW)
    _close("beta1", c.beta1, Float32(0.91))
    _close("beta2", c.beta2, Float32(0.98))
    _close("eps", c.eps, Float32(0.000001))
    _eqb("optimizer stochastic_rounding", c.optimizer_stochastic_rounding, False)
    _eq("ema mode", c.ema_mode, EMA_MODE_GPU)
    _eqb("ema enabled", c.ema_enabled, True)
    _close("ema decay", c.ema_decay, Float32(0.998))
    _eq("ema update interval", c.ema_update_step_interval, 4)
    _eqb("masked_training", c.masked_training, True)
    _close("unmasked_probability", c.unmasked_probability, Float32(0.2))
    _eqb("normalize_masked_area_loss", c.normalize_masked_area_loss, True)
    _eq("gradient_checkpointing", c.gradient_checkpointing, GRADIENT_CHECKPOINTING_ON)


def _gate_adafactor_policy() raises:
    print("--- OneTrainer ADAFACTOR defaults and extra distributions ---")
    var body = String("")
    body += '"model_type":"CHROMA_1","training_method":"FINE_TUNE",'
    body += '"learning_rate":0.00001,"timestep_distribution":"INVERTED_PARABOLA",'
    body += '"noising_weight":7.7,"gradient_checkpointing":"CPU_OFFLOADED",'
    body += '"layer_offload_fraction":0.4,'
    body += '"optimizer":{"optimizer":"ADAFACTOR"},'
    body += '"optimizer_defaults":{"ADAFACTOR":{"optimizer":"ADAFACTOR",'
    body += '"clip_threshold":1.0,"decay_rate":-0.8,"eps":1e-30,"eps2":0.001,'
    body += '"relative_step":false,"scale_parameter":false,"stochastic_rounding":true,'
    body += '"warmup_init":false,"weight_decay":0.0}},'
    body += '"transformer":{"train":true,"weight_dtype":"BFLOAT_16"},'
    body += '"text_encoder":{"train":false,"weight_dtype":"BFLOAT_16"},'
    body += '"vae":{"weight_dtype":"FLOAT_32"}'

    var c = _read_json(String("/tmp/onetrainer_policy_adafactor.json"), body)
    _eq("training_method", c.training_method, TRAINING_METHOD_FINE_TUNE)
    _eq("optimizer", c.optimizer, TRAIN_OPTIMIZER_ADAFACTOR)
    _close("adafactor clip_threshold", c.optimizer_clip_threshold, Float32(1.0))
    _close("adafactor decay_rate", c.optimizer_decay_rate, Float32(-0.8))
    _close("adafactor eps2", c.optimizer_eps2, Float32(0.001))
    _eqb("adafactor stochastic_rounding", c.optimizer_stochastic_rounding, True)
    _eq("inverted parabola tag", c.timestep_distribution, 5)
    _close("noising_weight", c.timestep_noising_weight, Float32(7.7))
    _eq("gradient_checkpointing", c.gradient_checkpointing, GRADIENT_CHECKPOINTING_CPU_OFFLOADED)
    _eq("transformer dtype", c.transformer_weight_dtype, TRAIN_DTYPE_BFLOAT_16)


def _gate_unknown_dtype_fails() raises:
    print("--- unknown dtype fails loud ---")
    var raised = False
    try:
        var c = _read_json(
            String("/tmp/onetrainer_policy_bad_dtype.json"),
            String('"train_dtype":"FLOAT_12"'),
        )
        _ = c.train_dtype
    except e:
        raised = True
        print("  raised as expected:", String(e))
    if not raised:
        raise Error("OneTrainer policy smoke: unknown dtype did not raise")


def main() raises:
    print("=== OneTrainer policy config reader smoke ===")
    _gate_qwen_lora_policy()
    _gate_adafactor_policy()
    _gate_unknown_dtype_fails()
    print("onetrainer_policy_config_smoke PASS")

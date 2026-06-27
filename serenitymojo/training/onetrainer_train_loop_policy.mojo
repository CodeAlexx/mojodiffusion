# onetrainer_train_loop_policy.mojo
#
# Shared OneTrainer policy helpers for real model train loops. Model files still
# own shape-specific validation and numeric kernels; this module owns the common
# training-method, optimizer, checkpoint/offload, sample, save, and path policy.

from serenitymojo.training.sample_prompt_config import (
    SampleCadence,
    read_sample_cadence_config,
    validate_step_sample_cadence,
    should_sample_completed_step,
    SAMPLE_UNIT_NEVER,
    SAMPLE_UNIT_STEP,
)
from serenitymojo.training.train_config import (
    TrainConfig,
    TRAINING_METHOD_FINE_TUNE,
    GRADIENT_CHECKPOINTING_OFF,
    GRADIENT_CHECKPOINTING_ON,
    GRADIENT_CHECKPOINTING_CPU_OFFLOADED,
)
from serenitymojo.training.lr_schedule import lr_for_step


comptime OT_GRAD_POLICY_ON_ONLY = 0
comptime OT_GRAD_POLICY_ON_OR_CPU_OFFLOADED = 1
comptime OT_GRAD_POLICY_ON_OR_OFF = 2


def ot_full_finetune_unsupported_message(
    trainer_name: String,
) -> String:
    return (
        trainer_name
        + String(" received training_method=FINE_TUNE/full_finetune. ")
        + String("OneTrainer registers full finetune for this model, but the ")
        + String("Mojo loop still lacks model-specific full-weight gradients, ")
        + String("base-weight AdamW/update state, save_full_finetune_model_tensors, ")
        + String("load_full_finetune_model_tensors, ordered full_finetune tensor ")
        + String("manifest, optimizer/master TrainState sidecar binding, and ")
        + String("OneTrainer parity artifacts.")
    )


def validate_ot_lora_only_or_fail_full_finetune(
    cfg: TrainConfig, trainer_name: String,
) raises:
    cfg.validate_training_method_config()
    if cfg.training_method == TRAINING_METHOD_FINE_TUNE:
        raise Error(ot_full_finetune_unsupported_message(trainer_name))
    if not cfg.is_lora_training():
        raise Error(
            trainer_name
            + String(" currently implements training_method=LORA only; parsed training_method tag ")
            + String(cfg.training_method)
        )


def validate_ot_lora_adamw_loop_policy(
    cfg: TrainConfig, trainer_name: String,
) raises:
    cfg.validate_onetrainer_policy_config()
    validate_ot_lora_only_or_fail_full_finetune(cfg, trainer_name)
    if not cfg.optimizer_is_adamw():
        raise Error(
            trainer_name
            + String(" currently implements optimizer=ADAMW only; parsed optimizer tag ")
            + String(cfg.optimizer)
        )


def validate_ot_train_math_policy(
    cfg: TrainConfig, trainer_name: String,
) raises:
    """Validate shared train math before expensive model setup.

    This binds parsed OneTrainer LR/AdamW/masked-loss fields to the product loop
    policy. Unsupported masked/prior modes fail loud for now instead of being
    silently ignored by a model-specific loss path.
    """
    validate_ot_lora_adamw_loop_policy(cfg, trainer_name)
    if cfg.lr <= Float32(0.0):
        raise Error(trainer_name + String(" requires learning_rate > 0"))
    if cfg.max_steps <= 0:
        raise Error(trainer_name + String(" requires max_steps > 0 for LR scheduling"))
    if cfg.lr_warmup_steps < 0:
        raise Error(trainer_name + String(" requires lr_warmup_steps >= 0"))
    if cfg.lr_scheduler < 0 or cfg.lr_scheduler > 5:
        raise Error(trainer_name + String(" has unsupported lr_scheduler tag"))
    if cfg.lr_min_factor < Float32(0.0) or cfg.lr_min_factor > Float32(1.0):
        raise Error(trainer_name + String(" requires lr_min_factor in 0..1"))
    if cfg.lr_cycles <= Float32(0.0):
        raise Error(trainer_name + String(" requires lr_cycles > 0"))
    if cfg.beta1 < Float32(0.0) or cfg.beta1 >= Float32(1.0):
        raise Error(trainer_name + String(" requires AdamW beta1 in [0,1)"))
    if cfg.beta2 < Float32(0.0) or cfg.beta2 >= Float32(1.0):
        raise Error(trainer_name + String(" requires AdamW beta2 in [0,1)"))
    if cfg.eps <= Float32(0.0):
        raise Error(trainer_name + String(" requires AdamW eps > 0"))
    if cfg.weight_decay < Float32(0.0):
        raise Error(trainer_name + String(" requires AdamW weight_decay >= 0"))
    if cfg.masked_training:
        raise Error(
            trainer_name
            + String(" does not implement OneTrainer masked_training loss yet")
        )
    if cfg.normalize_masked_area_loss:
        raise Error(
            trainer_name
            + String(" does not implement normalize_masked_area_loss yet")
        )
    if cfg.masked_prior_preservation_weight != Float32(0.0):
        raise Error(
            trainer_name
            + String(" does not implement masked_prior_preservation_weight yet")
        )
    if cfg.unmasked_weight < Float32(0.0):
        raise Error(trainer_name + String(" requires unmasked_weight >= 0"))


def ot_lr_for_optimizer_step(cfg: TrainConfig, optimizer_step: Int) -> Float32:
    return lr_for_step(
        cfg.lr,
        optimizer_step,
        cfg.lr_warmup_steps,
        cfg.max_steps,
        cfg.lr_scheduler,
        cfg.lr_min_factor,
        cfg.lr_cycles,
        Float32(2.0),
    )


def validate_ot_gradient_checkpointing_policy(
    cfg: TrainConfig, trainer_name: String, policy: Int,
) raises:
    cfg.validate_offload_checkpoint_config()
    if policy == OT_GRAD_POLICY_ON_ONLY:
        if cfg.gradient_checkpointing != GRADIENT_CHECKPOINTING_ON:
            raise Error(
                trainer_name
                + String(" currently requires gradient_checkpointing=ON; ")
                + String("OFF retains too much activation state and CPU_OFFLOADED ")
                + String("needs activation/layer offload runtime plumbing")
            )
        return
    if policy == OT_GRAD_POLICY_ON_OR_CPU_OFFLOADED:
        if (
            cfg.gradient_checkpointing != GRADIENT_CHECKPOINTING_ON
            and cfg.gradient_checkpointing != GRADIENT_CHECKPOINTING_CPU_OFFLOADED
        ):
            raise Error(
                trainer_name
                + String(" requires gradient_checkpointing=ON or CPU_OFFLOADED")
            )
        return
    if policy == OT_GRAD_POLICY_ON_OR_OFF:
        if cfg.gradient_checkpointing == GRADIENT_CHECKPOINTING_CPU_OFFLOADED:
            raise Error(
                trainer_name
                + String(" cannot honor CPU_OFFLOADED activation/layer offload yet; ")
                + String("use ON or OFF for the current resident/recompute path")
            )
        if (
            cfg.gradient_checkpointing != GRADIENT_CHECKPOINTING_ON
            and cfg.gradient_checkpointing != GRADIENT_CHECKPOINTING_OFF
        ):
            raise Error(
                trainer_name
                + String(" requires gradient_checkpointing=ON or OFF")
            )
        return
    raise Error(
        trainer_name
        + String(": unknown OneTrainer gradient checkpoint policy ")
        + String(policy)
    )


def ot_cache_dir_from_train_config(cfg: TrainConfig, default_cache_dir: String) -> String:
    if cfg.dataset_cache_dir != String(""):
        return cfg.dataset_cache_dir.copy()
    return default_cache_dir.copy()


def ot_stage_dir_from_train_config(cfg: TrainConfig, default_cache_dir: String) raises -> String:
    """Precache staging dir, derived deterministically from the cache dir so the
    stager and the prepare step agree without a separate config key:
    `<cache_dir>_stage`. Used by the config-driven Mojo precache tools."""
    var cache_dir = ot_cache_dir_from_train_config(cfg, default_cache_dir)
    if cache_dir == String(""):
        raise Error("ot_stage_dir_from_train_config: config has no cache_dir/dataset_cache_dir")
    return cache_dir + String("_stage")


def ot_dataset_path_from_train_config(cfg: TrainConfig) raises -> String:
    """Raw image+caption source dir for precaching. Fail loud if the config
    omits it — there is no hardcoded dataset fallback."""
    if cfg.dataset_path == String(""):
        raise Error("config has no dataset_path; precache requires a raw image source dir")
    return cfg.dataset_path.copy()


def ot_output_lora_path_from_train_config(
    cfg: TrainConfig, default_lora_dir: String, prefix: String, completed_step: Int,
) -> String:
    if cfg.output_model_destination != String(""):
        return cfg.output_model_destination.copy()
    return (
        default_lora_dir
        + String("/")
        + prefix
        + String("_step")
        + String(completed_step)
        + String(".safetensors")
    )


def ot_output_lora_path_for_stream_from_train_config(
    cfg: TrainConfig,
    default_lora_dir: String,
    prefix: String,
    stream_index: Int,
    completed_step: Int,
) -> String:
    var suffix = String(".safetensors")
    if cfg.output_model_destination != String(""):
        if cfg.output_model_destination.endswith(suffix):
            return (
                String(cfg.output_model_destination.removesuffix(suffix))
                + String("_st")
                + String(stream_index)
                + String("_step")
                + String(completed_step)
                + suffix
            )
        return (
            cfg.output_model_destination
            + String("_st")
            + String(stream_index)
            + String("_step")
            + String(completed_step)
            + suffix
        )
    return (
        default_lora_dir
        + String("/")
        + prefix
        + String("_st")
        + String(stream_index)
        + String("_step")
        + String(completed_step)
        + suffix
    )


def ot_fixed_output_lora_path_from_train_config(
    cfg: TrainConfig, default_lora_path: String,
) -> String:
    if cfg.output_model_destination != String(""):
        return cfg.output_model_destination.copy()
    return default_lora_path.copy()


def ot_state_path_for_lora(lora_path: String) -> String:
    return lora_path + String(".state.safetensors")


def ot_step_lora_path(base_path: String, step: Int) -> String:
    var suffix = String(".safetensors")
    if base_path.endswith(suffix):
        return String(base_path.removesuffix(suffix)) + String("_step") + String(step) + suffix
    return base_path + String("_step") + String(step) + suffix


def ot_final_or_step_lora_path(
    base_path: String,
    default_lora_dir: String,
    default_final_name: String,
    default_step_prefix: String,
    step: Int,
    max_steps: Int,
) -> String:
    var suffix = String(".safetensors")
    if base_path != String(""):
        if step >= max_steps:
            return base_path.copy()
        return ot_step_lora_path(base_path, step)
    if step >= max_steps:
        return default_lora_dir + String("/") + default_final_name
    return (
        default_lora_dir
        + String("/")
        + default_step_prefix
        + String(step)
        + suffix
    )


def ot_sample_cadence_from_train_config(
    cfg_path: String, cfg: TrainConfig,
) raises -> SampleCadence:
    var cadence = read_sample_cadence_config(cfg_path, cfg.sample_every)
    if (
        cfg.sample_definition_file_name != String("")
        and cfg.sample_definition_file_name != String("training_samples/samples.json")
    ):
        cadence.sample_definition_file_name = cfg.sample_definition_file_name.copy()
    elif cadence.sample_definition_file_name == String("") and cfg.validation_prompts_file != String(""):
        cadence.sample_definition_file_name = cfg.validation_prompts_file.copy()
    if cadence.sample_after_unit == SAMPLE_UNIT_STEP and cadence.sample_after <= 0:
        cadence.sample_after_unit = SAMPLE_UNIT_NEVER
        return cadence^
    validate_step_sample_cadence(cadence)
    return cadence^


def ot_sampling_enabled(cadence: SampleCadence) -> Bool:
    return cadence.sample_after_unit != SAMPLE_UNIT_NEVER


def ot_sampling_enabled_checked(cadence: SampleCadence) raises -> Bool:
    validate_step_sample_cadence(cadence)
    return ot_sampling_enabled(cadence)


def ot_should_save_checkpoint(cfg: TrainConfig, completed_step: Int) -> Bool:
    return completed_step > 0 and cfg.save_every > 0 and completed_step % cfg.save_every == 0


def ot_should_save_before_sample(
    cadence: SampleCadence, completed_step: Int, saved_this_step: Bool,
) raises -> Bool:
    return (
        cadence.save_before_sample
        and not saved_this_step
        and should_sample_completed_step(cadence, completed_step)
    )

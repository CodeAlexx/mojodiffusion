# training/onetrainer_resume_contract.mojo
#
# OneTrainer checkpoint/save/resume control contract.
#
# Reference surface:
#   OneTrainer/modules/trainer/GenericTrainer.py
#   OneTrainer/modules/modelSaver/GenericLoRAModelSaver.py
#   OneTrainer/modules/modelSaver/mixin/InternalModelSaverMixin.py
#   OneTrainer/modules/modelLoader/GenericLoRAModelLoader.py
#   OneTrainer/modules/modelLoader/mixin/InternalModelLoaderMixin.py
#
# This is metadata/control only. It does not claim or touch tensor storage dtype.

from serenitymojo.training.onetrainer_lifecycle import (
    OTTrainProgress,
    OTTimedActionState,
    repeating_action_needed_at,
    single_action_elapsed_at,
)
from serenitymojo.training.train_config import (
    TRAINING_METHOD_FINE_TUNE,
    TRAINING_METHOD_LORA,
    TRAIN_OPTIMIZER_ADAMW,
    TrainConfig,
)


comptime OT_RESUME_SURFACE_RAW_LORA = 0
comptime OT_RESUME_SURFACE_INTERNAL_BACKUP = 1
comptime OT_RESUME_SURFACE_FULL_FINETUNE_INTERNAL = 2


@fieldwise_init
struct OTCheckpointActionDecisions(Copyable, Movable):
    var should_sample: Bool
    var should_backup: Bool
    var should_save: Bool
    var sample_runs_before_pending_save: Bool


@fieldwise_init
struct OTResumeManifest(Copyable, Movable):
    var model_type: String
    var training_method: Int
    var optimizer: Int
    var surface: Int
    var global_step: Int
    var epoch: Int
    var epoch_step: Int
    var epoch_sample: Int
    var has_meta_json: Bool
    var has_lora_state: Bool
    var has_optimizer_state: Bool
    var has_param_group_mapping: Bool
    var has_param_group_optimizer_mapping: Bool
    var has_full_model_payload: Bool


def ot_progress_filename(progress: OTTrainProgress) -> String:
    return (
        String(progress.global_step)
        + String("-")
        + String(progress.epoch)
        + String("-")
        + String(progress.epoch_step)
    )


def ot_model_file_extension(output_model_format: String) raises -> String:
    if output_model_format == String("SAFETENSORS"):
        return String(".safetensors")
    if output_model_format == String("LEGACY_SAFETENSORS"):
        return String(".safetensors")
    if output_model_format == String("CKPT"):
        return String(".ckpt")
    if output_model_format == String("DIFFUSERS"):
        return String("")
    if output_model_format == String("INTERNAL"):
        return String("")
    raise Error(
        String("OneTrainer resume contract: unsupported output model format ")
        + output_model_format
    )


def ot_step_save_leaf(
    save_filename_prefix: String,
    timestamp: String,
    progress: OTTrainProgress,
    extension: String,
) -> String:
    return (
        save_filename_prefix
        + timestamp
        + String("-save-")
        + ot_progress_filename(progress)
        + extension
    )


def ot_step_save_path(
    workspace_dir: String,
    save_filename_prefix: String,
    timestamp: String,
    progress: OTTrainProgress,
    output_model_format: String,
) raises -> String:
    return (
        workspace_dir
        + String("/save/")
        + ot_step_save_leaf(
            save_filename_prefix,
            timestamp,
            progress,
            ot_model_file_extension(output_model_format),
        )
    )


def ot_backup_leaf(timestamp: String, progress: OTTrainProgress) -> String:
    return timestamp + String("-backup-") + ot_progress_filename(progress)


def ot_backup_path(
    workspace_dir: String,
    timestamp: String,
    progress: OTTrainProgress,
) -> String:
    return workspace_dir + String("/backup/") + ot_backup_leaf(timestamp, progress)


def ot_training_sample_leaf(
    save_filename_prefix: String,
    timestamp: String,
    progress: OTTrainProgress,
) -> String:
    return (
        save_filename_prefix
        + timestamp
        + String("-training-sample-")
        + ot_progress_filename(progress)
    )


def ot_internal_meta_path(internal_backup_path: String) -> String:
    return internal_backup_path + String("/meta.json")


def ot_internal_lora_state_path(internal_backup_path: String) -> String:
    return internal_backup_path + String("/lora/lora.safetensors")


def ot_internal_optimizer_state_path(internal_backup_path: String) -> String:
    return internal_backup_path + String("/optimizer/optimizer.pt")


def ot_save_runs_before_sample() -> Bool:
    # GenericTrainer enqueues samples first, then executes the sample queue
    # before draining pending backup/save commands when gradients are absent.
    return False


def ot_save_before_sample() -> Bool:
    return ot_save_runs_before_sample()


def ot_lora_resume_requires_optimizer_state() -> Bool:
    # OneTrainer internal resume carries optimizer/optimizer.pt. A raw LoRA file
    # can seed adapters for inference or cold LoRA init, but not a training resume.
    return True


def ot_lora_resume_expects_state_sidecars(manifest: OTResumeManifest) -> Bool:
    return (
        manifest.has_meta_json
        and manifest.has_lora_state
        and manifest.has_optimizer_state
        and manifest.has_param_group_mapping
        and manifest.has_param_group_optimizer_mapping
    )


def ot_full_finetune_resume_expects_state_sidecars(manifest: OTResumeManifest) -> Bool:
    return (
        manifest.has_meta_json
        and manifest.has_optimizer_state
        and manifest.has_param_group_mapping
        and manifest.has_param_group_optimizer_mapping
        and manifest.has_full_model_payload
    )


def ot_sample_interval_due(
    cfg: TrainConfig,
    progress: OTTrainProgress,
    sample_skip_state: OTTimedActionState,
    mut sample_state: OTTimedActionState,
    now_seconds: Float64 = Float64(0.0),
) raises -> Bool:
    cfg.validate_onetrainer_policy_config()
    return (
        single_action_elapsed_at(
            sample_skip_state,
            progress,
            cfg.sample_skip_first,
            cfg.sample_after_unit,
            now_seconds,
        )
        and repeating_action_needed_at(
            sample_state,
            progress,
            cfg.sample_after,
            cfg.sample_after_unit,
            True,
            now_seconds,
        )
    )


def ot_backup_interval_due(
    cfg: TrainConfig,
    progress: OTTrainProgress,
    mut backup_state: OTTimedActionState,
    now_seconds: Float64 = Float64(0.0),
) raises -> Bool:
    cfg.validate_onetrainer_policy_config()
    return repeating_action_needed_at(
        backup_state,
        progress,
        cfg.backup_after,
        cfg.backup_after_unit,
        False,
        now_seconds,
    )


def ot_save_interval_due(
    cfg: TrainConfig,
    progress: OTTrainProgress,
    save_skip_state: OTTimedActionState,
    mut save_state: OTTimedActionState,
    now_seconds: Float64 = Float64(0.0),
) raises -> Bool:
    cfg.validate_onetrainer_policy_config()
    return (
        single_action_elapsed_at(
            save_skip_state,
            progress,
            cfg.save_skip_first,
            cfg.save_every_unit,
            now_seconds,
        )
        and repeating_action_needed_at(
            save_state,
            progress,
            cfg.save_every,
            cfg.save_every_unit,
            False,
            now_seconds,
        )
    )


def ot_checkpoint_action_decisions(
    cfg: TrainConfig,
    progress: OTTrainProgress,
    sample_skip_state: OTTimedActionState,
    mut sample_state: OTTimedActionState,
    mut backup_state: OTTimedActionState,
    save_skip_state: OTTimedActionState,
    mut save_state: OTTimedActionState,
    now_seconds: Float64 = Float64(0.0),
) raises -> OTCheckpointActionDecisions:
    return OTCheckpointActionDecisions(
        ot_sample_interval_due(
            cfg,
            progress,
            sample_skip_state,
            sample_state,
            now_seconds,
        ),
        ot_backup_interval_due(cfg, progress, backup_state, now_seconds),
        ot_save_interval_due(
            cfg,
            progress,
            save_skip_state,
            save_state,
            now_seconds,
        ),
        not ot_save_runs_before_sample(),
    )


def _training_method_name(method: Int) -> String:
    if method == TRAINING_METHOD_LORA:
        return String("LORA")
    if method == TRAINING_METHOD_FINE_TUNE:
        return String("FINE_TUNE")
    return String("UNKNOWN")


def _validate_resume_progress(manifest: OTResumeManifest) raises:
    if not manifest.has_meta_json:
        raise Error("OneTrainer resume contract: internal resume requires meta.json")
    if manifest.global_step < 0:
        raise Error("OneTrainer resume contract: global_step must be present and non-negative")
    if manifest.epoch < 0:
        raise Error("OneTrainer resume contract: epoch must be present and non-negative")
    if manifest.epoch_step < 0:
        raise Error("OneTrainer resume contract: epoch_step must be present and non-negative")
    if manifest.epoch_sample < 0:
        raise Error("OneTrainer resume contract: epoch_sample must be present and non-negative")


def validate_ot_resume_manifest(
    manifest: OTResumeManifest,
    expected_model_type: String,
    expected_training_method: Int,
    expected_optimizer: Int = TRAIN_OPTIMIZER_ADAMW,
    product_loop_supports_full_finetune: Bool = False,
) raises:
    if manifest.model_type != expected_model_type:
        raise Error(
            String("OneTrainer resume contract: incompatible model type; expected ")
            + expected_model_type
            + String(" got ")
            + manifest.model_type
        )
    if manifest.training_method != expected_training_method:
        raise Error(
            String("OneTrainer resume contract: incompatible training method; expected ")
            + _training_method_name(expected_training_method)
            + String(" got ")
            + _training_method_name(manifest.training_method)
        )
    if manifest.optimizer != expected_optimizer:
        raise Error(
            String("OneTrainer resume contract: incompatible optimizer tag; expected ")
            + String(expected_optimizer)
            + String(" got ")
            + String(manifest.optimizer)
        )

    _validate_resume_progress(manifest)

    if manifest.training_method == TRAINING_METHOD_LORA:
        if manifest.surface != OT_RESUME_SURFACE_INTERNAL_BACKUP:
            raise Error(
                "OneTrainer resume contract: LoRA training resume requires an internal backup surface"
            )
        if not manifest.has_lora_state:
            raise Error("OneTrainer resume contract: LoRA resume requires lora/lora.safetensors")
        if not manifest.has_optimizer_state:
            raise Error(
                "OneTrainer resume contract: missing optimizer state for LoRA resume"
            )
        if not manifest.has_param_group_mapping:
            raise Error(
                "OneTrainer resume contract: LoRA optimizer state missing param_group_mapping"
            )
        if not manifest.has_param_group_optimizer_mapping:
            raise Error(
                "OneTrainer resume contract: LoRA optimizer state missing param_group_optimizer_mapping"
            )
        return

    if manifest.training_method == TRAINING_METHOD_FINE_TUNE:
        if not product_loop_supports_full_finetune:
            raise Error(
                "OneTrainer resume contract: full-finetune resume is unsupported until the product-specific loop wires it"
            )
        if manifest.surface != OT_RESUME_SURFACE_FULL_FINETUNE_INTERNAL:
            raise Error(
                "OneTrainer resume contract: full-finetune resume requires an internal full-model surface"
            )
        if not manifest.has_full_model_payload:
            raise Error(
                "OneTrainer resume contract: full-finetune resume missing model payload"
            )
        if not manifest.has_optimizer_state:
            raise Error(
                "OneTrainer resume contract: full-finetune resume missing optimizer state"
            )
        if not manifest.has_param_group_mapping:
            raise Error(
                "OneTrainer resume contract: full-finetune optimizer state missing param_group_mapping"
            )
        if not manifest.has_param_group_optimizer_mapping:
            raise Error(
                "OneTrainer resume contract: full-finetune optimizer state missing param_group_optimizer_mapping"
            )
        return

    raise Error(
        String("OneTrainer resume contract: unsupported training method ")
        + String(manifest.training_method)
    )

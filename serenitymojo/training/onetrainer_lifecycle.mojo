# training/onetrainer_lifecycle.mojo - OneTrainer GenericTrainer lifecycle contract.
#
# This is the product-run control layer, not a model loop. It mirrors the
# OneTrainer pieces that sit around every trainer:
#   train.py -> create_trainer(config) -> GenericTrainer.start/train/end
#   TrainProgress.next_step/next_epoch/filename_string
#   TimedActionMixin cadence checks for validate/sample/backup/save/stop
#
# Model-specific files still own forward/backward/update and sampler execution.

from serenitymojo.sampling.onetrainer_sampler_contract import (
    OT_SAMPLER_ANIMA,
    OT_SAMPLER_CHROMA,
    OT_SAMPLER_ERNIE,
    OT_SAMPLER_FLUX1_DEV,
    OT_SAMPLER_FLUX2_DEV,
    OT_SAMPLER_FLUX2_KLEIN,
    OT_SAMPLER_QWEN,
    OT_SAMPLER_SD3,
    OT_SAMPLER_SDXL,
    OT_SAMPLER_ZIMAGE,
    OneTrainerSamplerPlan,
    default_ot_sampler_plan,
    sampler_contract_summary,
    validate_ot_sampler_plan,
)
from serenitymojo.training.train_config import (
    TRAINING_METHOD_FINE_TUNE,
    TRAINING_METHOD_LORA,
    TRAIN_OPTIMIZER_ADAMW,
    TRAIN_TIME_UNIT_ALWAYS,
    TRAIN_TIME_UNIT_EPOCH,
    TRAIN_TIME_UNIT_HOUR,
    TRAIN_TIME_UNIT_MINUTE,
    TRAIN_TIME_UNIT_NEVER,
    TRAIN_TIME_UNIT_SECOND,
    TRAIN_TIME_UNIT_STEP,
    TrainConfig,
)
from serenitymojo.training.full_finetune_contract import (
    create_full_finetune_product_run_plan,
    full_finetune_product_run_blocker,
    validate_full_finetune_product_run_plan,
)


comptime OT_PRODUCT_RUNNER_UNSUPPORTED = 0
comptime OT_PRODUCT_RUNNER_LORA_REAL = 1
comptime OT_PRODUCT_RUNNER_FULL_FINETUNE_REAL = 2
comptime OT_MODEL_LOOP_INVOCATION_ABSENT = 0
comptime OT_MODEL_LOOP_INVOCATION_DIRECT = 1


@fieldwise_init
struct OTTrainProgress(Copyable, Movable):
    var epoch: Int
    var epoch_step: Int
    var epoch_sample: Int
    var global_step: Int

    @staticmethod
    def zero() -> OTTrainProgress:
        return OTTrainProgress(0, 0, 0, 0)

    def filename_string(self) -> String:
        return (
            String(self.global_step)
            + String("-")
            + String(self.epoch)
            + String("-")
            + String(self.epoch_step)
        )

    def next_step(mut self, batch_size: Int):
        self.epoch_step += 1
        self.epoch_sample += batch_size
        self.global_step += 1

    def next_epoch(mut self):
        self.epoch_step = 0
        self.epoch_sample = 0
        self.epoch += 1


@fieldwise_init
struct OTTimedActionState(Copyable, Movable):
    # OneTrainer stores this as a one-element list and mutates it only for
    # SECOND/MINUTE/HOUR checks.
    var previous_action_seconds: Float64
    var start_seconds: Float64

    @staticmethod
    def zero() -> OTTimedActionState:
        return OTTimedActionState(Float64(0.0), Float64(0.0))


@fieldwise_init
struct OTStepActionDecisions(Copyable, Movable):
    var should_validate: Bool
    var should_sample: Bool
    var should_backup: Bool
    var should_save: Bool
    var should_stop: Bool


@fieldwise_init
struct OTGenericTrainerLifecyclePlan(Copyable, Movable):
    var will_start: Bool
    var will_train: Bool
    var will_end: Bool
    var invocation_kind: Int
    var direct_invocation_supported: Bool
    var runner_name: String
    var invocation_blocker: String

    def has_start_train_end(self) -> Bool:
        return self.will_start and self.will_train and self.will_end


@fieldwise_init
struct OTProductRunPlan(Copyable, Movable):
    var model_type: String
    var runner_name: String
    var runner_kind: Int
    var training_method: Int
    var optimizer: Int
    var workspace_dir: String
    var cache_dir: String
    var output_model_destination: String
    var concept_file_name: String
    var sample_definition_file_name: String
    var validation_prompts_file: String
    var sampler_plan: OneTrainerSamplerPlan
    var sampler_summary: String
    var only_cache: Bool
    var validation_enabled: Bool
    var ema_enabled: Bool

    def is_product_ready(self) -> Bool:
        return self.runner_kind != OT_PRODUCT_RUNNER_UNSUPPORTED


def _time_unit_name(unit: Int) -> String:
    if unit == TRAIN_TIME_UNIT_EPOCH:
        return String("EPOCH")
    if unit == TRAIN_TIME_UNIT_STEP:
        return String("STEP")
    if unit == TRAIN_TIME_UNIT_SECOND:
        return String("SECOND")
    if unit == TRAIN_TIME_UNIT_MINUTE:
        return String("MINUTE")
    if unit == TRAIN_TIME_UNIT_HOUR:
        return String("HOUR")
    if unit == TRAIN_TIME_UNIT_NEVER:
        return String("NEVER")
    if unit == TRAIN_TIME_UNIT_ALWAYS:
        return String("ALWAYS")
    return String("UNKNOWN")


def _time_interval_seconds(interval: Int, unit: Int) raises -> Float64:
    if interval < 0:
        raise Error("OneTrainer lifecycle: negative time interval")
    if unit == TRAIN_TIME_UNIT_SECOND:
        return Float64(interval)
    if unit == TRAIN_TIME_UNIT_MINUTE:
        return Float64(interval * 60)
    if unit == TRAIN_TIME_UNIT_HOUR:
        return Float64(interval * 3600)
    raise Error(
        String("OneTrainer lifecycle: ")
        + _time_unit_name(unit)
        + String(" is not a wall-clock unit")
    )


def _validate_step_or_epoch_interval(interval: Int) raises:
    if interval < 0:
        raise Error("OneTrainer lifecycle: negative cadence interval")


def repeating_action_needed(
    progress: OTTrainProgress,
    interval: Int,
    unit: Int,
    start_at_zero: Bool = False,
) raises -> Bool:
    """OneTrainer TimedActionMixin.repeating_action_needed without wall clock.

    SECOND/MINUTE/HOUR need a monotonic timestamp. Use
    `repeating_action_needed_at` for those so product runs cannot silently treat
    time cadences as step cadences.
    """

    if unit == TRAIN_TIME_UNIT_SECOND or unit == TRAIN_TIME_UNIT_MINUTE or unit == TRAIN_TIME_UNIT_HOUR:
        raise Error(
            String("OneTrainer lifecycle: ")
            + _time_unit_name(unit)
            + String(" cadence needs monotonic seconds")
        )
    var state = OTTimedActionState.zero()
    return repeating_action_needed_at(
        state,
        progress,
        interval,
        unit,
        start_at_zero,
        Float64(0.0),
    )


def repeating_action_needed_at(
    mut action_state: OTTimedActionState,
    progress: OTTrainProgress,
    interval: Int,
    unit: Int,
    start_at_zero: Bool = False,
    now_seconds: Float64 = Float64(0.0),
) raises -> Bool:
    if unit == TRAIN_TIME_UNIT_NEVER:
        return False
    if unit == TRAIN_TIME_UNIT_ALWAYS:
        return True
    if interval == 0:
        return False

    if unit == TRAIN_TIME_UNIT_EPOCH:
        _validate_step_or_epoch_interval(interval)
        if start_at_zero:
            return progress.epoch % interval == 0 and progress.epoch_step == 0
        return (
            progress.epoch % interval == 0
            and progress.epoch_step == 0
            and progress.epoch > 0
        )
    if unit == TRAIN_TIME_UNIT_STEP:
        _validate_step_or_epoch_interval(interval)
        if start_at_zero:
            return progress.global_step % interval == 0
        return (progress.global_step + 1) % interval == 0

    var every = _time_interval_seconds(interval, unit)
    if now_seconds - action_state.previous_action_seconds > every:
        action_state.previous_action_seconds = now_seconds
        return True
    return False


def single_action_elapsed(
    progress: OTTrainProgress,
    delay: Int,
    unit: Int,
) raises -> Bool:
    """OneTrainer TimedActionMixin.single_action_elapsed without wall clock."""

    if unit == TRAIN_TIME_UNIT_SECOND or unit == TRAIN_TIME_UNIT_MINUTE or unit == TRAIN_TIME_UNIT_HOUR:
        raise Error(
            String("OneTrainer lifecycle: ")
            + _time_unit_name(unit)
            + String(" delay needs monotonic seconds")
        )
    var state = OTTimedActionState.zero()
    return single_action_elapsed_at(
        state,
        progress,
        delay,
        unit,
        Float64(0.0),
    )


def single_action_elapsed_at(
    action_state: OTTimedActionState,
    progress: OTTrainProgress,
    delay: Int,
    unit: Int,
    now_seconds: Float64 = Float64(0.0),
) raises -> Bool:
    if delay < 0:
        raise Error("OneTrainer lifecycle: negative single-action delay")
    if unit == TRAIN_TIME_UNIT_NEVER:
        return False
    if unit == TRAIN_TIME_UNIT_ALWAYS:
        return True
    if unit == TRAIN_TIME_UNIT_EPOCH:
        return (progress.epoch + 1) > delay
    if unit == TRAIN_TIME_UNIT_STEP:
        return (progress.global_step + 1) > delay

    var every = _time_interval_seconds(delay, unit)
    return now_seconds - action_state.start_seconds > every


def training_step_actions(
    cfg: TrainConfig,
    progress: OTTrainProgress,
    mut validate_state: OTTimedActionState,
    mut sample_state: OTTimedActionState,
    mut backup_state: OTTimedActionState,
    mut save_state: OTTimedActionState,
    mut stop_state: OTTimedActionState,
    now_seconds: Float64 = Float64(0.0),
) raises -> OTStepActionDecisions:
    cfg.validate_onetrainer_policy_config()
    return OTStepActionDecisions(
        cfg.validation
        and repeating_action_needed_at(
            validate_state,
            progress,
            cfg.validate_after,
            cfg.validate_after_unit,
            True,
            now_seconds,
        ),
        repeating_action_needed_at(
            sample_state,
            progress,
            cfg.sample_after,
            cfg.sample_after_unit,
            True,
            now_seconds,
        ),
        repeating_action_needed_at(
            backup_state,
            progress,
            cfg.backup_after,
            cfg.backup_after_unit,
            False,
            now_seconds,
        ),
        repeating_action_needed_at(
            save_state,
            progress,
            cfg.save_every,
            cfg.save_every_unit,
            False,
            now_seconds,
        ),
        single_action_elapsed_at(
            stop_state,
            progress,
            cfg.stop_training_after,
            cfg.stop_training_after_unit,
            now_seconds,
        ),
    )


def _lora_runner_for_sampler_family(family: Int) -> String:
    if family == OT_SAMPLER_QWEN:
        return String("train_qwenimage_real")
    if family == OT_SAMPLER_ERNIE:
        return String("train_ernie_real")
    if family == OT_SAMPLER_ANIMA:
        return String("train_anima_real")
    if family == OT_SAMPLER_FLUX2_KLEIN:
        return String("train_klein_real")
    if family == OT_SAMPLER_CHROMA:
        return String("train_chroma_real")
    if family == OT_SAMPLER_SD3:
        return String("train_sd35_real")
    if family == OT_SAMPLER_SDXL:
        return String("train_sdxl_real")
    if family == OT_SAMPLER_FLUX1_DEV:
        return String("train_flux_real")
    if family == OT_SAMPLER_ZIMAGE:
        return String("train_zimage_real")
    return String("")


def _is_flux2_dev_alias(model_type: String) -> Bool:
    return model_type == String("flux2_dev") or model_type == String("FLUX_2_DEV")


def _is_flux2_model_type(model_type: String) -> Bool:
    return model_type == String("flux2") or model_type == String("FLUX_2")


def _sampler_model_type_for_product_contract(cfg: TrainConfig) -> String:
    if _is_flux2_dev_alias(cfg.name):
        return String("FLUX_2_DEV")
    if _is_flux2_model_type(cfg.name) and cfg.n_heads == 48:
        return String("FLUX_2_DEV")
    return cfg.name.copy()


def create_onetrainer_product_run_plan(cfg: TrainConfig) raises -> OTProductRunPlan:
    cfg.validate_training_method_config()
    cfg.validate_onetrainer_policy_config()

    var sampler_model_type = _sampler_model_type_for_product_contract(cfg)
    var sampler_plan = default_ot_sampler_plan(sampler_model_type)
    validate_ot_sampler_plan(sampler_plan)
    var runner = _lora_runner_for_sampler_family(sampler_plan.family)
    if runner == String(""):
        if sampler_plan.family == OT_SAMPLER_FLUX2_DEV:
            raise Error(
                String("OneTrainer lifecycle: Flux2 dev has no real Mojo product runner; ")
                + String("OneTrainer marks dev as FLUX_2 with num_attention_heads == 48/Mistral, ")
                + String("so it must not be dispatched to train_klein_real")
            )
        raise Error(
            String("OneTrainer lifecycle: no product runner registered for model_type=")
            + cfg.name
        )

    if cfg.training_method == TRAINING_METHOD_LORA:
        pass
    elif cfg.training_method == TRAINING_METHOD_FINE_TUNE:
        var full_ft_plan = create_full_finetune_product_run_plan(
            cfg.name, cfg.optimizer
        )
        validate_full_finetune_product_run_plan(full_ft_plan)
        raise Error(
            String("OneTrainer lifecycle: full-finetune product loop is not wired; ")
            + full_finetune_product_run_blocker(full_ft_plan)
        )
    else:
        raise Error("OneTrainer lifecycle: invalid training method")

    if cfg.optimizer != TRAIN_OPTIMIZER_ADAMW:
        raise Error(
            String("OneTrainer lifecycle: product LoRA runners currently require ADAMW; parsed optimizer tag ")
            + String(cfg.optimizer)
        )

    return OTProductRunPlan(
        cfg.name.copy(),
        runner^,
        OT_PRODUCT_RUNNER_LORA_REAL,
        cfg.training_method,
        cfg.optimizer,
        cfg.workspace_dir.copy(),
        cfg.cache_dir.copy(),
        cfg.output_model_destination.copy(),
        cfg.concept_file_name.copy(),
        cfg.sample_definition_file_name.copy(),
        cfg.validation_prompts_file.copy(),
        sampler_plan.copy(),
        sampler_contract_summary(sampler_plan),
        cfg.only_cache,
        cfg.validation,
        cfg.ema_enabled,
    )


def validate_onetrainer_product_run_plan(plan: OTProductRunPlan) raises:
    if not plan.is_product_ready():
        raise Error("OneTrainer lifecycle: product runner is not ready")
    if plan.runner_name == String(""):
        raise Error("OneTrainer lifecycle: runner name is empty")
    if plan.training_method != TRAINING_METHOD_LORA:
        raise Error("OneTrainer lifecycle: only LORA product runners are wired")
    if plan.optimizer != TRAIN_OPTIMIZER_ADAMW:
        raise Error("OneTrainer lifecycle: only ADAMW product runners are wired")
    validate_ot_sampler_plan(plan.sampler_plan)


def create_generic_trainer_lifecycle_plan(
    plan: OTProductRunPlan,
) raises -> OTGenericTrainerLifecyclePlan:
    validate_onetrainer_product_run_plan(plan)
    return OTGenericTrainerLifecyclePlan(
        True,
        True,
        True,
        OT_MODEL_LOOP_INVOCATION_ABSENT,
        False,
        plan.runner_name.copy(),
        String("direct Mojo model-loop invocation is not wired in the OneTrainer product-run layer"),
    )


def validate_generic_trainer_lifecycle_plan(
    plan: OTGenericTrainerLifecyclePlan,
) raises:
    if not plan.has_start_train_end():
        raise Error("OneTrainer lifecycle: GenericTrainer start/train/end contract is incomplete")
    if plan.runner_name == String(""):
        raise Error("OneTrainer lifecycle: model-loop runner name is empty")
    if plan.invocation_kind == OT_MODEL_LOOP_INVOCATION_DIRECT:
        if not plan.direct_invocation_supported:
            raise Error("OneTrainer lifecycle: direct invocation kind without support flag")
        return
    if plan.invocation_kind == OT_MODEL_LOOP_INVOCATION_ABSENT:
        if plan.direct_invocation_supported:
            raise Error("OneTrainer lifecycle: absent invocation cannot be marked supported")
        if plan.invocation_blocker == String(""):
            raise Error("OneTrainer lifecycle: absent model-loop invocation needs a fail-loud blocker")
        return
    raise Error("OneTrainer lifecycle: unknown model-loop invocation kind")


def generic_trainer_lifecycle_summary(plan: OTGenericTrainerLifecyclePlan) -> String:
    if plan.direct_invocation_supported:
        return String("lifecycle=start/train/end invoke=direct runner=") + plan.runner_name
    return (
        String("lifecycle=start/train/end invoke=blocked runner=")
        + plan.runner_name
        + String(" blocker=")
        + plan.invocation_blocker
    )


def onetrainer_product_run_summary(plan: OTProductRunPlan) -> String:
    return (
        plan.runner_name
        + String(" model=")
        + plan.model_type
        + String(" sampler=")
        + plan.sampler_summary
        + String(" workspace=")
        + plan.workspace_dir
        + String(" cache=")
        + plan.cache_dir
    )

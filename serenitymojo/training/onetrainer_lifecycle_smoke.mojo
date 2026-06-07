# onetrainer_lifecycle_smoke.mojo - OneTrainer GenericTrainer control smoke.
#
# Run:
#   timeout 180 prlimit --as=16000000000 pixi run mojo run -I . serenitymojo/training/onetrainer_lifecycle_smoke.mojo

from serenitymojo.sampling.onetrainer_sampler_contract import (
    OT_SAMPLER_CHROMA,
    OT_SAMPLER_FLUX1_DEV,
    OT_SAMPLER_FLUX2_KLEIN,
    OT_SAMPLER_QWEN,
    OT_SAMPLER_ZIMAGE,
)
from serenitymojo.training.onetrainer_lifecycle import (
    OT_MODEL_LOOP_INVOCATION_ABSENT,
    OT_PRODUCT_RUNNER_LORA_REAL,
    OTTrainProgress,
    OTTimedActionState,
    create_generic_trainer_lifecycle_plan,
    create_onetrainer_product_run_plan,
    generic_trainer_lifecycle_summary,
    onetrainer_product_run_summary,
    repeating_action_needed,
    repeating_action_needed_at,
    single_action_elapsed,
    single_action_elapsed_at,
    training_step_actions,
    validate_generic_trainer_lifecycle_plan,
    validate_onetrainer_product_run_plan,
)
from serenitymojo.training.train_config import (
    TRAINING_METHOD_FINE_TUNE,
    TRAIN_OPTIMIZER_ADAFACTOR,
    TRAIN_TIME_UNIT_ALWAYS,
    TRAIN_TIME_UNIT_EPOCH,
    TRAIN_TIME_UNIT_MINUTE,
    TRAIN_TIME_UNIT_NEVER,
    TRAIN_TIME_UNIT_SECOND,
    TRAIN_TIME_UNIT_STEP,
    TrainConfig,
)


def _check(cond: Bool, msg: String) raises:
    if not cond:
        raise Error(String("onetrainer_lifecycle_smoke FAILED: ") + msg)


def _gate_train_progress() raises:
    print("--- TrainProgress parity ---")
    var p = OTTrainProgress.zero()
    _check(p.filename_string() == String("0-0-0"), "initial filename string")
    p.next_step(3)
    _check(p.global_step == 1, "global step after next_step")
    _check(p.epoch_step == 1, "epoch_step after next_step")
    _check(p.epoch_sample == 3, "epoch_sample adds batch size")
    _check(p.filename_string() == String("1-0-1"), "filename after step")
    p.next_epoch()
    _check(p.epoch == 1, "epoch after next_epoch")
    _check(p.epoch_step == 0, "epoch_step reset")
    _check(p.epoch_sample == 0, "epoch_sample reset")
    _check(p.filename_string() == String("1-1-0"), "filename after epoch")
    print("  TrainProgress PASS")


def _gate_timed_actions() raises:
    print("--- TimedActionMixin parity ---")
    var p = OTTrainProgress.zero()
    _check(repeating_action_needed(p, 100, TRAIN_TIME_UNIT_STEP, True), "STEP start_at_zero at step 0")
    _check(not repeating_action_needed(p, 100, TRAIN_TIME_UNIT_STEP, False), "STEP no start_at_zero at step 0")
    p.global_step = 99
    _check(not repeating_action_needed(p, 100, TRAIN_TIME_UNIT_STEP, True), "STEP start_at_zero false at 99")
    _check(repeating_action_needed(p, 100, TRAIN_TIME_UNIT_STEP, False), "STEP nonzero true at 99")
    p.global_step = 100
    _check(repeating_action_needed(p, 100, TRAIN_TIME_UNIT_STEP, True), "STEP start_at_zero true at 100")
    _check(not repeating_action_needed(p, 100, TRAIN_TIME_UNIT_STEP, False), "STEP nonzero false at 100")

    p = OTTrainProgress.zero()
    _check(repeating_action_needed(p, 5, TRAIN_TIME_UNIT_EPOCH, True), "EPOCH start_at_zero at epoch 0")
    _check(not repeating_action_needed(p, 5, TRAIN_TIME_UNIT_EPOCH, False), "EPOCH no start_at_zero at epoch 0")
    p.epoch = 5
    p.epoch_step = 0
    _check(repeating_action_needed(p, 5, TRAIN_TIME_UNIT_EPOCH, False), "EPOCH true at boundary")
    p.epoch_step = 1
    _check(not repeating_action_needed(p, 5, TRAIN_TIME_UNIT_EPOCH, False), "EPOCH false mid epoch")

    p = OTTrainProgress.zero()
    _check(not single_action_elapsed(p, 4, TRAIN_TIME_UNIT_STEP), "single STEP false before delay")
    p.global_step = 3
    _check(not single_action_elapsed(p, 4, TRAIN_TIME_UNIT_STEP), "single STEP uses >, not >=")
    p.global_step = 4
    _check(single_action_elapsed(p, 4, TRAIN_TIME_UNIT_STEP), "single STEP true after delay")
    _check(not repeating_action_needed(p, 1, TRAIN_TIME_UNIT_NEVER), "NEVER repeating false")
    _check(repeating_action_needed(p, 1, TRAIN_TIME_UNIT_ALWAYS), "ALWAYS repeating true")
    _check(not single_action_elapsed(p, 1, TRAIN_TIME_UNIT_NEVER), "NEVER single false")
    _check(single_action_elapsed(p, 1, TRAIN_TIME_UNIT_ALWAYS), "ALWAYS single true")

    var minute_without_clock_raised = False
    try:
        _ = repeating_action_needed(
            OTTrainProgress.zero(),
            1,
            TRAIN_TIME_UNIT_MINUTE,
            False,
        )
    except e:
        minute_without_clock_raised = True
        print("  raised as expected [wall-clock cadence without timestamp]:", String(e))
    _check(minute_without_clock_raised, "wall-clock cadence without timestamp must raise")

    var state = OTTimedActionState.zero()
    _check(
        repeating_action_needed_at(state, p, 5, TRAIN_TIME_UNIT_SECOND, False, Float64(6.0)),
        "SECOND first interval elapsed",
    )
    _check(state.previous_action_seconds == Float64(6.0), "SECOND updates previous action")
    _check(
        not repeating_action_needed_at(state, p, 5, TRAIN_TIME_UNIT_SECOND, False, Float64(11.0)),
        "SECOND uses >, not >=",
    )
    _check(
        repeating_action_needed_at(state, p, 5, TRAIN_TIME_UNIT_SECOND, False, Float64(12.0)),
        "SECOND next interval elapsed",
    )
    var single_state = OTTimedActionState(Float64(0.0), Float64(10.0))
    _check(
        not single_action_elapsed_at(single_state, p, 2, TRAIN_TIME_UNIT_SECOND, Float64(12.0)),
        "single SECOND uses >, not >=",
    )
    _check(
        single_action_elapsed_at(single_state, p, 2, TRAIN_TIME_UNIT_SECOND, Float64(12.1)),
        "single SECOND true after delay",
    )
    print("  TimedActionMixin PASS")


def _gate_step_decisions() raises:
    print("--- GenericTrainer step decisions ---")
    var cfg = TrainConfig.default()
    cfg.name = String("qwenimage")
    cfg.validation = True
    cfg.validate_after = 10
    cfg.validate_after_unit = TRAIN_TIME_UNIT_STEP
    cfg.sample_after = 25
    cfg.sample_after_unit = TRAIN_TIME_UNIT_STEP
    cfg.backup_after = 0
    cfg.backup_after_unit = TRAIN_TIME_UNIT_NEVER
    cfg.save_every = 50
    cfg.save_every_unit = TRAIN_TIME_UNIT_STEP
    cfg.stop_training_after = 100
    cfg.stop_training_after_unit = TRAIN_TIME_UNIT_STEP

    var p = OTTrainProgress.zero()
    p.global_step = 49
    var validate_state = OTTimedActionState.zero()
    var sample_state = OTTimedActionState.zero()
    var backup_state = OTTimedActionState.zero()
    var save_state = OTTimedActionState.zero()
    var stop_state = OTTimedActionState.zero()
    var decisions = training_step_actions(
        cfg,
        p,
        validate_state,
        sample_state,
        backup_state,
        save_state,
        stop_state,
    )
    _check(not decisions.should_validate, "validate false at step 49 with start_at_zero")
    _check(not decisions.should_sample, "sample false at step 49")
    _check(not decisions.should_backup, "backup NEVER")
    _check(decisions.should_save, "save true before completed step 50")
    _check(not decisions.should_stop, "stop false before step 100")

    p.global_step = 99
    decisions = training_step_actions(
        cfg,
        p,
        validate_state,
        sample_state,
        backup_state,
        save_state,
        stop_state,
    )
    _check(decisions.should_save, "save true before completed step 100")
    _check(not decisions.should_stop, "stop uses >, not >= at step 99")
    p.global_step = 100
    decisions = training_step_actions(
        cfg,
        p,
        validate_state,
        sample_state,
        backup_state,
        save_state,
        stop_state,
    )
    _check(decisions.should_validate, "validate true at global step 100")
    _check(decisions.should_sample, "sample true at global step 100")
    _check(decisions.should_stop, "stop true after configured step delay")
    print("  GenericTrainer step decisions PASS")


def _expect_product_plan_raises(
    label: String, cfg: TrainConfig, required_message: String
) raises:
    var raised = False
    try:
        var plan = create_onetrainer_product_run_plan(cfg)
        validate_onetrainer_product_run_plan(plan)
    except e:
        raised = True
        var msg = String(e)
        print("  raised as expected [", label, "]:", msg)
        if required_message != String(""):
            _check(msg.find(required_message) >= 0, label + String(" error message"))
    _check(raised, label + String(" must raise"))


def _gate_product_run_plan() raises:
    print("--- product run plan gating ---")
    var cfg = TrainConfig.default()
    cfg.name = String("qwenimage")
    cfg.workspace_dir = String("/workspace/qwen")
    cfg.cache_dir = String("/cache/qwen")
    cfg.output_model_destination = String("/out/qwen.safetensors")
    cfg.validation_prompts_file = String("/samples/qwen.json")
    var plan = create_onetrainer_product_run_plan(cfg)
    validate_onetrainer_product_run_plan(plan)
    var lifecycle = create_generic_trainer_lifecycle_plan(plan)
    validate_generic_trainer_lifecycle_plan(lifecycle)
    _check(plan.runner_kind == OT_PRODUCT_RUNNER_LORA_REAL, "Qwen runner kind")
    _check(plan.runner_name == String("train_qwenimage_real"), "Qwen runner name")
    _check(lifecycle.has_start_train_end(), "Qwen GenericTrainer start/train/end")
    _check(lifecycle.invocation_kind == OT_MODEL_LOOP_INVOCATION_ABSENT, "Qwen direct invocation absent")
    _check(not lifecycle.direct_invocation_supported, "Qwen direct invocation blocked")
    _check(plan.sampler_plan.family == OT_SAMPLER_QWEN, "Qwen sampler family")
    _check(plan.sampler_plan.diffusion_steps == 25, "Qwen sampler steps")
    _check(plan.workspace_dir == String("/workspace/qwen"), "workspace copied")
    print(onetrainer_product_run_summary(plan))
    print(generic_trainer_lifecycle_summary(lifecycle))

    cfg.name = String("klein9b")
    plan = create_onetrainer_product_run_plan(cfg)
    validate_onetrainer_product_run_plan(plan)
    _check(plan.runner_name == String("train_klein_real"), "Klein runner name")
    _check(plan.sampler_plan.family == OT_SAMPLER_FLUX2_KLEIN, "Klein/Flux2 sampler family")

    cfg.name = String("zimage")
    plan = create_onetrainer_product_run_plan(cfg)
    validate_onetrainer_product_run_plan(plan)
    _check(plan.runner_name == String("train_zimage_real"), "Z-Image runner name")
    _check(plan.sampler_plan.family == OT_SAMPLER_ZIMAGE, "Z-Image sampler family")

    cfg.name = String("chroma")
    plan = create_onetrainer_product_run_plan(cfg)
    validate_onetrainer_product_run_plan(plan)
    _check(plan.runner_name == String("train_chroma_real"), "Chroma runner name")
    _check(plan.sampler_plan.family == OT_SAMPLER_CHROMA, "Chroma sampler family")

    cfg.name = String("flux")
    plan = create_onetrainer_product_run_plan(cfg)
    validate_onetrainer_product_run_plan(plan)
    _check(plan.runner_name == String("train_flux_real"), "Flux runner name")
    _check(plan.sampler_plan.family == OT_SAMPLER_FLUX1_DEV, "Flux sampler family")

    cfg.name = String("FLUX_2")
    cfg.n_heads = 32
    plan = create_onetrainer_product_run_plan(cfg)
    validate_onetrainer_product_run_plan(plan)
    _check(plan.runner_name == String("train_klein_real"), "FLUX_2 Klein runner name")
    _check(plan.sampler_plan.family == OT_SAMPLER_FLUX2_KLEIN, "FLUX_2 Klein sampler family")

    var blocked_cfg = TrainConfig.default()
    blocked_cfg.name = String("FLUX_2_DEV")
    _expect_product_plan_raises(
        String("Flux2 dev alias product runner"),
        blocked_cfg,
        String("must not be dispatched to train_klein_real"),
    )

    blocked_cfg = TrainConfig.default()
    blocked_cfg.name = String("FLUX_2")
    blocked_cfg.n_heads = 48
    _expect_product_plan_raises(
        String("Flux2 dev num_attention_heads product runner"),
        blocked_cfg,
        String("num_attention_heads == 48"),
    )

    blocked_cfg = TrainConfig.default()
    blocked_cfg.name = String("qwenimage")
    blocked_cfg.training_method = TRAINING_METHOD_FINE_TUNE
    _expect_product_plan_raises(
        String("full-finetune product loop not wired"),
        blocked_cfg,
        String("full-finetune product loop is not wired"),
    )

    blocked_cfg = TrainConfig.default()
    blocked_cfg.name = String("qwenimage")
    blocked_cfg.optimizer = TRAIN_OPTIMIZER_ADAFACTOR
    _expect_product_plan_raises(
        String("unsupported product optimizer"),
        blocked_cfg,
        String("product LoRA runners currently require ADAMW"),
    )

    blocked_cfg = TrainConfig.default()
    blocked_cfg.name = String("unknown")
    _expect_product_plan_raises(
        String("unsupported model type"),
        blocked_cfg,
        String("unsupported model_type=unknown"),
    )
    print("  product run plan PASS")


def main() raises:
    print("==== OneTrainer lifecycle smoke ====")
    _gate_train_progress()
    _gate_timed_actions()
    _gate_step_decisions()
    _gate_product_run_plan()
    print("onetrainer_lifecycle_smoke PASS")

# onetrainer_train_loop_policy_smoke.mojo
#
# No-CUDA smoke for the shared OneTrainer train-loop policy helpers.

from serenitymojo.training.onetrainer_train_loop_policy import (
    OT_GRAD_POLICY_ON_ONLY,
    OT_GRAD_POLICY_ON_OR_CPU_OFFLOADED,
    OT_GRAD_POLICY_ON_OR_OFF,
    ot_cache_dir_from_train_config,
    ot_fixed_output_lora_path_from_train_config,
    ot_final_or_step_lora_path,
    ot_lr_for_optimizer_step,
    ot_output_lora_path_for_stream_from_train_config,
    ot_output_lora_path_from_train_config,
    ot_sample_cadence_from_train_config,
    ot_sampling_enabled,
    ot_sampling_enabled_checked,
    ot_should_save_before_sample,
    ot_should_save_checkpoint,
    ot_state_path_for_lora,
    ot_step_lora_path,
    validate_ot_gradient_checkpointing_policy,
    validate_ot_lora_adamw_loop_policy,
    validate_ot_train_math_policy,
)
from serenitymojo.training.sample_prompt_config import SAMPLE_UNIT_NEVER, SAMPLE_UNIT_STEP
from serenitymojo.training.train_config import (
    GRADIENT_CHECKPOINTING_CPU_OFFLOADED,
    GRADIENT_CHECKPOINTING_OFF,
    GRADIENT_CHECKPOINTING_ON,
    TRAIN_OPTIMIZER_ADAFACTOR,
    TrainConfig,
)


comptime SAMPLE_CONFIG = "/home/alex/mojodiffusion/serenitymojo/configs/qwenimage.json"


def _check(cond: Bool, msg: String) raises:
    if not cond:
        raise Error(String("onetrainer_train_loop_policy_smoke FAILED: ") + msg)


def _expect_policy_raises(label: String, cfg: TrainConfig, policy: Int) raises:
    var raised = False
    try:
        validate_ot_gradient_checkpointing_policy(cfg, String("Smoke trainer"), policy)
    except e:
        raised = True
    if not raised:
        raise Error(String("expected policy raise: ") + label)


def _expect_lora_policy_raises(label: String, cfg: TrainConfig) raises:
    var raised = False
    try:
        validate_ot_lora_adamw_loop_policy(cfg, String("Smoke trainer"))
    except e:
        raised = True
    if not raised:
        raise Error(String("expected lora policy raise: ") + label)


def _expect_math_policy_raises(label: String, cfg: TrainConfig) raises:
    var raised = False
    try:
        validate_ot_train_math_policy(cfg, String("Smoke trainer"))
    except e:
        raised = True
    if not raised:
        raise Error(String("expected train math policy raise: ") + label)


def main() raises:
    var cfg = TrainConfig.default()
    cfg.dataset_cache_dir = String("/tmp/ot-policy-cache")
    cfg.output_model_destination = String("/tmp/ot-policy/model.safetensors")
    cfg.sample_every = 7
    cfg.save_every = 5

    validate_ot_lora_adamw_loop_policy(cfg, String("Smoke trainer"))
    validate_ot_train_math_policy(cfg, String("Smoke trainer"))
    validate_ot_gradient_checkpointing_policy(cfg, String("Smoke trainer"), OT_GRAD_POLICY_ON_ONLY)
    validate_ot_gradient_checkpointing_policy(cfg, String("Smoke trainer"), OT_GRAD_POLICY_ON_OR_CPU_OFFLOADED)
    validate_ot_gradient_checkpointing_policy(cfg, String("Smoke trainer"), OT_GRAD_POLICY_ON_OR_OFF)

    cfg.gradient_checkpointing = GRADIENT_CHECKPOINTING_CPU_OFFLOADED
    validate_ot_gradient_checkpointing_policy(cfg, String("Smoke trainer"), OT_GRAD_POLICY_ON_OR_CPU_OFFLOADED)
    _expect_policy_raises(String("on-only rejects cpu offload"), cfg, OT_GRAD_POLICY_ON_ONLY)
    _expect_policy_raises(String("on/off rejects cpu offload"), cfg, OT_GRAD_POLICY_ON_OR_OFF)

    cfg.gradient_checkpointing = GRADIENT_CHECKPOINTING_OFF
    validate_ot_gradient_checkpointing_policy(cfg, String("Smoke trainer"), OT_GRAD_POLICY_ON_OR_OFF)
    _expect_policy_raises(String("on-only rejects off"), cfg, OT_GRAD_POLICY_ON_ONLY)

    cfg.gradient_checkpointing = GRADIENT_CHECKPOINTING_ON
    cfg.optimizer = TRAIN_OPTIMIZER_ADAFACTOR
    _expect_lora_policy_raises(String("non-adamw optimizer"), cfg)
    _expect_math_policy_raises(String("non-adamw optimizer"), cfg)
    cfg.optimizer = 0

    cfg.lr_scheduler = 2
    cfg.lr_warmup_steps = 0
    cfg.max_steps = 10
    cfg.lr_min_factor = Float32(0.0)
    var lr_mid = ot_lr_for_optimizer_step(cfg, 5)
    _check(
        lr_mid > Float32(0.0) and lr_mid < cfg.lr,
        String("scheduled lr midpoint"),
    )
    cfg.lr_scheduler = 0
    cfg.max_steps = 3000

    cfg.masked_training = True
    _expect_math_policy_raises(String("masked training unsupported"), cfg)
    cfg.masked_training = False
    cfg.normalize_masked_area_loss = True
    _expect_math_policy_raises(String("normalize masked area unsupported"), cfg)
    cfg.normalize_masked_area_loss = False
    cfg.masked_prior_preservation_weight = Float32(1.0)
    _expect_math_policy_raises(String("masked prior unsupported"), cfg)
    cfg.masked_prior_preservation_weight = Float32(0.0)

    _check(
        ot_cache_dir_from_train_config(cfg, String("/tmp/default-cache"))
        == String("/tmp/ot-policy-cache"),
        String("cache override"),
    )
    _check(
        ot_output_lora_path_from_train_config(
            cfg, String("/tmp/default-out"), String("model_lora"), 12
        ) == String("/tmp/ot-policy/model.safetensors"),
        String("explicit output destination"),
    )
    cfg.output_model_destination = String("")
    _check(
        ot_output_lora_path_from_train_config(
            cfg, String("/tmp/default-out"), String("model_lora"), 12
        ) == String("/tmp/default-out/model_lora_step12.safetensors"),
        String("default step output path"),
    )
    cfg.output_model_destination = String("/tmp/ot-policy/model.safetensors")
    _check(
        ot_output_lora_path_for_stream_from_train_config(
            cfg, String("/tmp/default-out"), String("sdxl_lora"), 1, 12
        ) == String("/tmp/ot-policy/model_st1_step12.safetensors"),
        String("stream output destination"),
    )
    _check(
        ot_fixed_output_lora_path_from_train_config(
            cfg, String("/tmp/default-out/final.safetensors")
        ) == String("/tmp/ot-policy/model.safetensors"),
        String("fixed explicit output path"),
    )
    _check(
        ot_state_path_for_lora(String("/tmp/x.safetensors"))
        == String("/tmp/x.safetensors.state.safetensors"),
        String("state sidecar path"),
    )
    _check(
        ot_step_lora_path(String("/tmp/x.safetensors"), 3)
        == String("/tmp/x_step3.safetensors"),
        String("step path suffix"),
    )
    _check(
        ot_final_or_step_lora_path(
            String(""),
            String("/tmp/final-step"),
            String("final.safetensors"),
            String("lora_step"),
            10,
            10,
        ) == String("/tmp/final-step/final.safetensors"),
        String("final default path"),
    )

    cfg.output_model_destination = String("")
    var cadence = ot_sample_cadence_from_train_config(SAMPLE_CONFIG, cfg)
    _check(cadence.sample_after_unit != SAMPLE_UNIT_NEVER, String("sample cadence active"))
    _check(ot_sampling_enabled(cadence), String("sampling enabled"))
    _check(ot_sampling_enabled_checked(cadence), String("sampling enabled checked"))
    cfg.sample_after = 0
    cadence.sample_after = 0
    cadence.sample_after_unit = SAMPLE_UNIT_STEP
    var disabled = ot_sample_cadence_from_train_config(SAMPLE_CONFIG, cfg)
    disabled.sample_after = 0
    disabled.sample_after_unit = SAMPLE_UNIT_NEVER
    _check(not ot_sampling_enabled(disabled), String("sampling disabled"))
    _check(ot_should_save_checkpoint(cfg, 5), String("save step 5"))
    _check(not ot_should_save_checkpoint(cfg, 4), String("skip step 4"))
    cadence.save_before_sample = True
    cadence.sample_after = 5
    cadence.sample_after_unit = SAMPLE_UNIT_STEP
    _check(ot_should_save_before_sample(cadence, 5, False), String("save before sample"))
    _check(not ot_should_save_before_sample(cadence, 5, True), String("skip saved sample"))
    print("onetrainer_train_loop_policy_smoke PASS")

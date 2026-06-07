# onetrainer_resume_contract_smoke.mojo - OneTrainer checkpoint/resume contract.
#
# Run:
#   timeout 180 prlimit --as=16000000000 pixi run mojo run -I . serenitymojo/training/onetrainer_resume_contract_smoke.mojo

from serenitymojo.training.onetrainer_lifecycle import (
    OTTrainProgress,
    OTTimedActionState,
)
from serenitymojo.training.onetrainer_resume_contract import (
    OT_RESUME_SURFACE_FULL_FINETUNE_INTERNAL,
    OT_RESUME_SURFACE_INTERNAL_BACKUP,
    OT_RESUME_SURFACE_RAW_LORA,
    OTResumeManifest,
    ot_backup_interval_due,
    ot_backup_leaf,
    ot_backup_path,
    ot_checkpoint_action_decisions,
    ot_internal_lora_state_path,
    ot_internal_meta_path,
    ot_internal_optimizer_state_path,
    ot_full_finetune_resume_expects_state_sidecars,
    ot_lora_resume_expects_state_sidecars,
    ot_lora_resume_requires_optimizer_state,
    ot_progress_filename,
    ot_sample_interval_due,
    ot_save_interval_due,
    ot_save_before_sample,
    ot_save_runs_before_sample,
    ot_step_save_leaf,
    ot_step_save_path,
    ot_training_sample_leaf,
    validate_ot_resume_manifest,
)
from serenitymojo.training.train_config import (
    TRAINING_METHOD_FINE_TUNE,
    TRAINING_METHOD_LORA,
    TRAIN_OPTIMIZER_ADAMW,
    TRAIN_OPTIMIZER_ADAFACTOR,
    TRAIN_TIME_UNIT_NEVER,
    TRAIN_TIME_UNIT_STEP,
    TrainConfig,
)


def _check(cond: Bool, msg: String) raises:
    if not cond:
        raise Error(String("onetrainer_resume_contract_smoke FAILED: ") + msg)


def _valid_lora_manifest(model_type: String) -> OTResumeManifest:
    return OTResumeManifest(
        model_type,
        TRAINING_METHOD_LORA,
        TRAIN_OPTIMIZER_ADAMW,
        OT_RESUME_SURFACE_INTERNAL_BACKUP,
        50,
        2,
        7,
        7,
        True,
        True,
        True,
        True,
        True,
        False,
    )


def _gate_names() raises:
    print("--- OneTrainer output naming ---")
    var p = OTTrainProgress(12, 3, 42, 98)
    _check(ot_progress_filename(p) == String("98-12-3"), "progress filename")
    _check(
        ot_step_save_leaf(String("pre-"), String("20260605-120000"), p, String(".safetensors"))
        == String("pre-20260605-120000-save-98-12-3.safetensors"),
        "step save leaf",
    )
    _check(
        ot_step_save_path(
            String("/work/run"),
            String("pre-"),
            String("20260605-120000"),
            p,
            String("SAFETENSORS"),
        )
        == String("/work/run/save/pre-20260605-120000-save-98-12-3.safetensors"),
        "step save path",
    )
    _check(
        ot_backup_leaf(String("20260605-120000"), p)
        == String("20260605-120000-backup-98-12-3"),
        "backup leaf",
    )
    _check(
        ot_backup_path(String("/work/run"), String("20260605-120000"), p)
        == String("/work/run/backup/20260605-120000-backup-98-12-3"),
        "backup path",
    )
    _check(
        ot_training_sample_leaf(String("pre-"), String("20260605-120000"), p)
        == String("pre-20260605-120000-training-sample-98-12-3"),
        "training sample leaf",
    )
    var backup = ot_backup_path(String("/work/run"), String("20260605-120000"), p)
    _check(ot_internal_meta_path(backup) == String("/work/run/backup/20260605-120000-backup-98-12-3/meta.json"), "meta path")
    _check(ot_internal_lora_state_path(backup) == String("/work/run/backup/20260605-120000-backup-98-12-3/lora/lora.safetensors"), "lora path")
    _check(ot_internal_optimizer_state_path(backup) == String("/work/run/backup/20260605-120000-backup-98-12-3/optimizer/optimizer.pt"), "optimizer path")
    _check(not ot_save_runs_before_sample(), "OneTrainer runs queued samples before pending save")
    _check(not ot_save_before_sample(), "save_before_sample contract")
    print("  naming PASS")


def _gate_cadence() raises:
    print("--- OneTrainer checkpoint cadence ---")
    var cfg = TrainConfig.default()
    cfg.name = String("qwenimage")
    cfg.sample_after = 49
    cfg.sample_after_unit = TRAIN_TIME_UNIT_STEP
    cfg.sample_skip_first = 0
    cfg.backup_after = 20
    cfg.backup_after_unit = TRAIN_TIME_UNIT_STEP
    cfg.save_every = 50
    cfg.save_every_unit = TRAIN_TIME_UNIT_STEP
    cfg.save_skip_first = 0

    var p = OTTrainProgress.zero()
    p.global_step = 48
    var sample_skip_state = OTTimedActionState.zero()
    var sample_state = OTTimedActionState.zero()
    var backup_state = OTTimedActionState.zero()
    var save_skip_state = OTTimedActionState.zero()
    var save_state = OTTimedActionState.zero()

    _check(
        not ot_sample_interval_due(cfg, p, sample_skip_state, sample_state),
        "sample start_at_zero cadence false before exact boundary",
    )
    _check(
        not ot_backup_interval_due(cfg, p, backup_state),
        "backup false away from boundary",
    )
    _check(
        not ot_save_interval_due(cfg, p, save_skip_state, save_state),
        "save false away from boundary",
    )

    p.global_step = 49
    var decisions = ot_checkpoint_action_decisions(
        cfg,
        p,
        sample_skip_state,
        sample_state,
        backup_state,
        save_skip_state,
        save_state,
    )
    _check(decisions.should_sample, "sample true at exact start_at_zero boundary")
    _check(not decisions.should_backup, "backup false at 50 boundary")
    _check(decisions.should_save, "save true at 50 boundary")
    _check(decisions.sample_runs_before_pending_save, "sample-before-save execution order")

    cfg.save_every_unit = TRAIN_TIME_UNIT_NEVER
    _check(
        not ot_save_interval_due(cfg, p, save_skip_state, save_state),
        "save NEVER false",
    )
    print("  cadence PASS")


def _gate_resume_validation() raises:
    print("--- OneTrainer resume validation ---")
    var manifest = _valid_lora_manifest(String("qwenimage"))
    validate_ot_resume_manifest(
        manifest,
        String("qwenimage"),
        TRAINING_METHOD_LORA,
        TRAIN_OPTIMIZER_ADAMW,
    )
    _check(ot_lora_resume_requires_optimizer_state(), "LoRA resume requires optimizer sidecar")
    _check(ot_lora_resume_expects_state_sidecars(manifest), "complete LoRA sidecars")

    var klein_manifest = _valid_lora_manifest(String("klein"))
    validate_ot_resume_manifest(
        klein_manifest,
        String("klein"),
        TRAINING_METHOD_LORA,
        TRAIN_OPTIMIZER_ADAMW,
    )
    var flux2_manifest = _valid_lora_manifest(String("FLUX_2"))
    validate_ot_resume_manifest(
        flux2_manifest,
        String("FLUX_2"),
        TRAINING_METHOD_LORA,
        TRAIN_OPTIMIZER_ADAMW,
    )

    var missing_optimizer = _valid_lora_manifest(String("qwenimage"))
    missing_optimizer.has_optimizer_state = False
    var missing_optimizer_raised = False
    try:
        validate_ot_resume_manifest(
            missing_optimizer,
            String("qwenimage"),
            TRAINING_METHOD_LORA,
            TRAIN_OPTIMIZER_ADAMW,
        )
    except e:
        missing_optimizer_raised = True
        print("  raised as expected [missing LoRA optimizer state]:", String(e))
    _check(missing_optimizer_raised, "missing optimizer state must fail")

    var raw_lora = _valid_lora_manifest(String("qwenimage"))
    raw_lora.surface = OT_RESUME_SURFACE_RAW_LORA
    var raw_lora_raised = False
    try:
        validate_ot_resume_manifest(
            raw_lora,
            String("qwenimage"),
            TRAINING_METHOD_LORA,
            TRAIN_OPTIMIZER_ADAMW,
        )
    except e:
        raw_lora_raised = True
        print("  raised as expected [raw LoRA is not resume]:", String(e))
    _check(raw_lora_raised, "raw LoRA surface must fail training resume")

    var wrong_model = _valid_lora_manifest(String("qwenimage"))
    var wrong_model_raised = False
    try:
        validate_ot_resume_manifest(
            wrong_model,
            String("flux"),
            TRAINING_METHOD_LORA,
            TRAIN_OPTIMIZER_ADAMW,
        )
    except e:
        wrong_model_raised = True
        print("  raised as expected [incompatible model type]:", String(e))
    _check(wrong_model_raised, "incompatible model type must fail")

    var wrong_optimizer = _valid_lora_manifest(String("qwenimage"))
    wrong_optimizer.optimizer = TRAIN_OPTIMIZER_ADAFACTOR
    var wrong_optimizer_raised = False
    try:
        validate_ot_resume_manifest(
            wrong_optimizer,
            String("qwenimage"),
            TRAINING_METHOD_LORA,
            TRAIN_OPTIMIZER_ADAMW,
        )
    except e:
        wrong_optimizer_raised = True
        print("  raised as expected [incompatible optimizer]:", String(e))
    _check(wrong_optimizer_raised, "incompatible optimizer must fail")

    var full_ft = OTResumeManifest(
        String("klein"),
        TRAINING_METHOD_FINE_TUNE,
        TRAIN_OPTIMIZER_ADAFACTOR,
        OT_RESUME_SURFACE_FULL_FINETUNE_INTERNAL,
        1,
        0,
        1,
        1,
        True,
        False,
        True,
        True,
        True,
        True,
    )
    var full_ft_blocked = False
    try:
        validate_ot_resume_manifest(
            full_ft,
            String("klein"),
            TRAINING_METHOD_FINE_TUNE,
            TRAIN_OPTIMIZER_ADAFACTOR,
            False,
        )
    except e:
        full_ft_blocked = True
        print("  raised as expected [full-finetune loop not wired]:", String(e))
    _check(full_ft_blocked, "full-finetune resume must fail until loop opts in")
    _check(
        ot_full_finetune_resume_expects_state_sidecars(full_ft),
        "complete full-finetune sidecars",
    )

    var full_ft_missing_mapping = full_ft.copy()
    full_ft_missing_mapping.has_param_group_mapping = False
    var full_ft_missing_mapping_raised = False
    try:
        validate_ot_resume_manifest(
            full_ft_missing_mapping,
            String("klein"),
            TRAINING_METHOD_FINE_TUNE,
            TRAIN_OPTIMIZER_ADAFACTOR,
            True,
        )
    except e:
        full_ft_missing_mapping_raised = True
        print("  raised as expected [full-finetune missing param mapping]:", String(e))
    _check(
        full_ft_missing_mapping_raised,
        "full-finetune resume must require param_group_mapping",
    )

    validate_ot_resume_manifest(
        full_ft,
        String("klein"),
        TRAINING_METHOD_FINE_TUNE,
        TRAIN_OPTIMIZER_ADAFACTOR,
        True,
    )
    print("  resume validation PASS")


def main() raises:
    print("==== OneTrainer resume contract smoke ====")
    _gate_names()
    _gate_cadence()
    _gate_resume_validation()
    print("onetrainer_resume_contract_smoke PASS")

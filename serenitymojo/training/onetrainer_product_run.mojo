# training/onetrainer_product_run.mojo - OneTrainer train.py entrypoint contract.
#
# This is the product-run wrapper contract for:
#   scripts/train.py -> create.create_trainer(config, callbacks, commands)
#                    -> GenericTrainer.start() -> train() -> end()
#
# It intentionally does not call a model loop by shelling out. The model-specific
# train files still own numeric parity, speed, VRAM, save/resume, and sampler
# execution. This layer proves the OneTrainer entrypoint decisions and fails
# before a long run when the product inputs are not available.

from serenitymojo.registry.checkpoints import path_exists
from std.os import listdir
from std.collections import List
from std.memory import alloc
from serenitymojo.io.ffi import (
    sys_close,
    sys_open,
    sys_pwrite,
    sys_system,
    BytePtr,
    O_CREAT,
    O_TRUNC,
    O_WRONLY,
)
from serenitymojo.io.train_config_reader import _read_file_bytes, read_model_config
from serenitymojo.training.onetrainer_lifecycle import (
    OT_MODEL_LOOP_INVOCATION_ABSENT,
    OTGenericTrainerLifecyclePlan,
    OTProductRunPlan,
    create_generic_trainer_lifecycle_plan,
    create_onetrainer_product_run_plan,
    generic_trainer_lifecycle_summary,
    onetrainer_product_run_summary,
    validate_generic_trainer_lifecycle_plan,
    validate_onetrainer_product_run_plan,
)
from serenitymojo.training.onetrainer_cache_preflight import (
    OTCachePreflightPlan,
    create_onetrainer_cache_preflight_plan,
    onetrainer_cache_preflight_summary,
    validate_onetrainer_cache_preflight_plan,
)
from serenitymojo.training.onetrainer_resume_contract import (
    OT_RESUME_SURFACE_FULL_FINETUNE_INTERNAL,
    OT_RESUME_SURFACE_INTERNAL_BACKUP,
    OTResumeManifest,
    ot_internal_lora_state_path,
    ot_internal_meta_path,
    ot_internal_optimizer_state_path,
    validate_ot_resume_manifest,
)
from serenitymojo.training.onetrainer_preset_catalog import (
    OTPresetCatalogEntry,
    default_onetrainer_preset_entry,
    validate_onetrainer_preset_entry,
)
from serenitymojo.training.full_finetune_contract import (
    create_full_finetune_product_run_plan,
    full_finetune_product_run_blocker,
    validate_full_finetune_product_run_plan,
)
from serenitymojo.training.sample_prompt_config import (
    SampleCadence,
    SAMPLE_UNIT_NEVER,
)
from serenitymojo.training.onetrainer_train_loop_policy import (
    ot_sample_cadence_from_train_config,
)
from serenitymojo.training.train_config import (
    TRAINING_METHOD_FINE_TUNE,
    TRAINING_METHOD_LORA,
    TrainConfig,
)


@fieldwise_init
struct OTProductResumePreflight(Copyable, Movable):
    var model_type: String
    var training_method: Int
    var optimizer: Int
    var resume_surface: Int
    var product_loop_supports_full_finetune: Bool
    var continue_last_backup: Bool
    var backup_dir: String
    var last_backup_path: String
    var found_last_backup: Bool
    var meta_path: String
    var lora_state_path: String
    var optimizer_state_path: String
    var meta_present: Bool
    var lora_state_present: Bool
    var optimizer_state_present: Bool
    var has_full_model_payload: Bool
    var will_resume_training_state: Bool
    var fail_loud_policy: String


@fieldwise_init
struct OTTrainEntrypointPlan(Copyable, Movable):
    var config_path: String
    var source_config_path: String
    var named_preset_id: String
    var preset_reference_path: String
    var preset_reference_kind: String
    var preset_recipe_family: String
    var preset_variant_kind: String
    var preset_vram_tier_gb: Int
    var resolved_config_materialized: Bool
    var product_plan: OTProductRunPlan
    var lifecycle_plan: OTGenericTrainerLifecyclePlan
    var cache_preflight: OTCachePreflightPlan
    var resume_preflight: OTProductResumePreflight
    var sample_cadence: SampleCadence
    var concept_file_required: Bool
    var concept_file_present: Bool
    var sample_file_required: Bool
    var sample_file_present: Bool
    var will_save_workspace_config: Bool
    var will_clear_cache_before_training: Bool
    var will_only_cache: Bool
    var will_create_validation_loader: Bool
    var will_create_sampler: Bool
    var will_end_after_train: Bool
    var will_backup_before_final_save: Bool
    var workspace_config_dir: String
    var workspace_backup_dir: String
    var workspace_save_dir: String
    var workspace_samples_dir: String
    var max_steps: Int

    def is_runnable_contract(self) -> Bool:
        return (
            self.product_plan.is_product_ready()
            and self.lifecycle_plan.has_start_train_end()
            and (not self.concept_file_required or self.concept_file_present)
            and (not self.sample_file_required or self.sample_file_present)
        )


def _path_required_and_present(path: String) -> Bool:
    if path == String(""):
        return False
    return path_exists(path)


def _string_gt(a: String, b: String) -> Bool:
    var ab = a.as_bytes()
    var bb = b.as_bytes()
    var n = a.byte_length()
    if b.byte_length() < n:
        n = b.byte_length()
    for i in range(n):
        if ab[i] > bb[i]:
            return True
        if ab[i] < bb[i]:
            return False
    return a.byte_length() > b.byte_length()


def _latest_backup_path(workspace_dir: String) raises -> String:
    var backup_dir = workspace_dir + String("/backup")
    if not path_exists(backup_dir):
        return String("")
    var names = listdir(backup_dir)
    var best = String("")
    for i in range(len(names)):
        var name = names[i]
        if name == String("") or name == String(".") or name == String(".."):
            continue
        var full = backup_dir + String("/") + name
        if path_exists(full):
            if best == String("") or _string_gt(name, best):
                best = name.copy()
    if best == String(""):
        return String("")
    return backup_dir + String("/") + best


def _resume_surface_for_training_method(training_method: Int) raises -> Int:
    if training_method == TRAINING_METHOD_LORA:
        return OT_RESUME_SURFACE_INTERNAL_BACKUP
    if training_method == TRAINING_METHOD_FINE_TUNE:
        return OT_RESUME_SURFACE_FULL_FINETUNE_INTERNAL
    raise Error("OneTrainer product resume preflight: unsupported training method")


def _bool_text(value: Bool) -> String:
    if value:
        return String("true")
    return String("false")


def create_onetrainer_product_resume_preflight(
    cfg: TrainConfig,
) raises -> OTProductResumePreflight:
    var backup_dir = cfg.workspace_dir + String("/backup")
    var resume_surface = _resume_surface_for_training_method(cfg.training_method)
    if not cfg.continue_last_backup:
        return OTProductResumePreflight(
            cfg.name.copy(),
            cfg.training_method,
            cfg.optimizer,
            resume_surface,
            False,
            False,
            backup_dir,
            String(""),
            False,
            String(""),
            String(""),
            String(""),
            False,
            False,
            False,
            False,
            False,
            String("validate_internal_backup_sidecars_before_product_resume"),
        )

    var last = _latest_backup_path(cfg.workspace_dir)
    if last == String(""):
        return OTProductResumePreflight(
            cfg.name.copy(),
            cfg.training_method,
            cfg.optimizer,
            resume_surface,
            False,
            True,
            backup_dir,
            String(""),
            False,
            String(""),
            String(""),
            String(""),
            False,
            False,
            False,
            False,
            False,
            String("validate_internal_backup_sidecars_before_product_resume"),
        )

    var meta = ot_internal_meta_path(last)
    var lora_state = ot_internal_lora_state_path(last)
    var opt_state = ot_internal_optimizer_state_path(last)
    var has_full_payload = False
    if cfg.training_method == TRAINING_METHOD_FINE_TUNE:
        has_full_payload = path_exists(last)
    return OTProductResumePreflight(
        cfg.name.copy(),
        cfg.training_method,
        cfg.optimizer,
        resume_surface,
        False,
        True,
        backup_dir,
        last.copy(),
        True,
        meta.copy(),
        lora_state.copy(),
        opt_state.copy(),
        path_exists(meta),
        path_exists(lora_state),
        path_exists(opt_state),
        has_full_payload,
        True,
        String("validate_internal_backup_sidecars_before_product_resume"),
    )


def validate_onetrainer_product_resume_preflight(
    plan: OTProductResumePreflight,
) raises:
    if plan.fail_loud_policy != String("validate_internal_backup_sidecars_before_product_resume"):
        raise Error("OneTrainer product resume preflight: fail-loud policy drift")
    if not plan.continue_last_backup:
        return
    if not plan.found_last_backup:
        return
    if (
        plan.training_method != TRAINING_METHOD_LORA
        and plan.training_method != TRAINING_METHOD_FINE_TUNE
    ):
        raise Error("OneTrainer product resume preflight: unsupported training method")
    var manifest = OTResumeManifest(
        plan.model_type.copy(),
        plan.training_method,
        plan.optimizer,
        plan.resume_surface,
        0,
        0,
        0,
        0,
        plan.meta_present,
        plan.lora_state_present,
        plan.optimizer_state_present,
        plan.optimizer_state_present,
        plan.optimizer_state_present,
        plan.has_full_model_payload,
    )
    validate_ot_resume_manifest(
        manifest,
        plan.model_type,
        plan.training_method,
        plan.optimizer,
        plan.product_loop_supports_full_finetune,
    )


def onetrainer_product_resume_preflight_summary(plan: OTProductResumePreflight) -> String:
    if not plan.continue_last_backup:
        return String("disabled")
    if not plan.found_last_backup:
        return String("continue_last_backup=true no-backup-found")
    return (
        String("continue_last_backup=true backup=")
        + plan.last_backup_path
        + String(" meta=")
        + _bool_text(plan.meta_present)
        + String(" lora=")
        + _bool_text(plan.lora_state_present)
        + String(" optimizer=")
        + _bool_text(plan.optimizer_state_present)
    )


def create_onetrainer_train_entrypoint_plan(
    config_path: String, cfg: TrainConfig,
) raises -> OTTrainEntrypointPlan:
    if config_path == String("") or not path_exists(config_path):
        raise Error(String("OneTrainer product run: config path is missing: ") + config_path)

    var product_plan = create_onetrainer_product_run_plan(cfg)
    validate_onetrainer_product_run_plan(product_plan)
    var lifecycle_plan = create_generic_trainer_lifecycle_plan(product_plan)
    validate_generic_trainer_lifecycle_plan(lifecycle_plan)
    var cache_preflight = create_onetrainer_cache_preflight_plan(cfg)
    validate_onetrainer_cache_preflight_plan(cache_preflight)
    var resume_preflight = create_onetrainer_product_resume_preflight(cfg)
    validate_onetrainer_product_resume_preflight(resume_preflight)

    var cadence = ot_sample_cadence_from_train_config(config_path, cfg)

    var concept_required = cfg.concept_file_name != String("")
    var concept_present = False
    if concept_required:
        concept_present = path_exists(cfg.concept_file_name)

    var sample_required = cadence.sample_after_unit != SAMPLE_UNIT_NEVER
    var sample_present = True
    if sample_required:
        sample_present = _path_required_and_present(cadence.sample_definition_file_name)

    return OTTrainEntrypointPlan(
        config_path.copy(),
        config_path.copy(),
        String(""),
        String(""),
        String(""),
        String(""),
        String(""),
        0,
        False,
        product_plan.copy(),
        lifecycle_plan.copy(),
        cache_preflight.copy(),
        resume_preflight.copy(),
        cadence.copy(),
        concept_required,
        concept_present,
        sample_required,
        sample_present,
        True,
        cfg.clear_cache_before_training and cfg.latent_caching,
        cfg.only_cache,
        cfg.validation,
        sample_required,
        True,
        cfg.backup_before_save,
        cfg.workspace_dir + String("/config"),
        cfg.workspace_dir + String("/backup"),
        cfg.workspace_dir + String("/save"),
        cfg.workspace_dir + String("/samples"),
        cfg.max_steps,
    )


def validate_onetrainer_train_entrypoint_plan(plan: OTTrainEntrypointPlan) raises:
    validate_onetrainer_product_run_plan(plan.product_plan)
    validate_generic_trainer_lifecycle_plan(plan.lifecycle_plan)
    validate_onetrainer_cache_preflight_plan(plan.cache_preflight)
    validate_onetrainer_product_resume_preflight(plan.resume_preflight)
    if plan.concept_file_required and not plan.concept_file_present:
        raise Error(
            String("OneTrainer product run: concept_file_name does not exist: ")
            + plan.product_plan.concept_file_name
        )
    if plan.sample_file_required and not plan.sample_file_present:
        raise Error(
            String("OneTrainer product run: sampling is enabled but sample file does not exist: ")
            + plan.sample_cadence.sample_definition_file_name
        )


def _is_json_ws(ch: UInt8) -> Bool:
    return ch == 0x20 or ch == 0x0A or ch == 0x0D or ch == 0x09


def _json_prefix_len_without_closing_object(data: List[UInt8]) raises -> Int:
    if len(data) == 0:
        raise Error("OneTrainer preset materializer: empty base config")
    var i = len(data) - 1
    while i >= 0:
        var ch = data[i]
        if _is_json_ws(ch):
            i -= 1
            continue
        if ch != 0x7D:
            raise Error("OneTrainer preset materializer: base config is not a JSON object")
        return i
    raise Error("OneTrainer preset materializer: base config has no JSON object")


def _json_prefix_needs_comma(data: List[UInt8], prefix_len: Int) -> Bool:
    var i = prefix_len - 1
    while i >= 0:
        var ch = data[i]
        if _is_json_ws(ch):
            i -= 1
            continue
        return ch != 0x7B
    return False


def _assert_json_safe_string(value: String, field: String) raises:
    var raw = value.as_bytes()
    for i in range(value.byte_length()):
        var ch = raw[i]
        if ch < 0x20 or ch == 0x22 or ch == 0x5C:
            raise Error(
                String("OneTrainer preset materializer: ")
                + field
                + String(" contains a character that must be JSON-escaped")
            )


def _json_string(value: String, field: String) raises -> String:
    _assert_json_safe_string(value, field)
    return String('"') + value + String('"')


def _training_method_name(method: Int) raises -> String:
    if method == TRAINING_METHOD_LORA:
        return String("LORA")
    if method == TRAINING_METHOD_FINE_TUNE:
        return String("FINE_TUNE")
    raise Error("OneTrainer preset materializer: unsupported training method")


def _optimizer_name(optimizer: Int) raises -> String:
    if optimizer == 0:
        return String("ADAMW")
    if optimizer == 2:
        return String("ADAFACTOR")
    raise Error("OneTrainer preset materializer: unsupported optimizer")


def _write_list_prefix_at(
    fd: Int, data: List[UInt8], count: Int, offset: Int,
) raises -> Int:
    if count <= 0:
        return offset
    var buf = alloc[UInt8](count)
    for i in range(count):
        buf[i] = data[i]
    var wrote = sys_pwrite(
        fd, BytePtr(unsafe_from_address=Int(buf)), count, offset
    )
    buf.free()
    if wrote != count:
        raise Error("OneTrainer preset materializer: short write while copying base config")
    return offset + wrote


def _write_string_at(fd: Int, content: String, offset: Int) raises -> Int:
    var n = content.byte_length()
    if n == 0:
        return offset
    var buf = alloc[UInt8](n)
    var raw = content.as_bytes()
    for i in range(n):
        buf[i] = raw[i]
    var wrote = sys_pwrite(
        fd, BytePtr(unsafe_from_address=Int(buf)), n, offset
    )
    buf.free()
    if wrote != n:
        raise Error("OneTrainer preset materializer: short write while appending overrides")
    return offset + wrote


def _resolved_preset_config_path(entry: OTPresetCatalogEntry) -> String:
    return String("/tmp/mojo-ot-presets/") + entry.preset_id + String(".json")


def _materialized_preset_override_suffix(
    entry: OTPresetCatalogEntry,
    concept_file_name: String,
    sample_definition_file_name: String,
    needs_comma: Bool,
) raises -> String:
    var suffix = String("")
    if needs_comma:
        suffix += String(",")
    suffix += (
        String('"model_type":')
        + _json_string(entry.model_type, String("model_type"))
        + String(",")
    )
    suffix += (
        String('"training_method":')
        + _json_string(_training_method_name(entry.training_method), String("training_method"))
        + String(",")
    )
    suffix += (
        String('"optimizer":{"optimizer":')
        + _json_string(_optimizer_name(entry.optimizer), String("optimizer.optimizer"))
        + String("},")
    )
    suffix += (
        String('"concept_file_name":')
        + _json_string(concept_file_name, String("concept_file_name"))
        + String(",")
    )
    suffix += (
        String('"sample_definition_file_name":')
        + _json_string(sample_definition_file_name, String("sample_definition_file_name"))
        + String(",")
    )
    suffix += (
        String('"validation_prompts_file":')
        + _json_string(sample_definition_file_name, String("validation_prompts_file"))
        + String(",")
    )
    suffix += (
        String('"mojo_onetrainer_preset_id":')
        + _json_string(entry.preset_id, String("mojo_onetrainer_preset_id"))
        + String(",")
    )
    suffix += (
        String('"mojo_onetrainer_reference_path":')
        + _json_string(entry.ot_reference_path, String("mojo_onetrainer_reference_path"))
        + String(",")
    )
    suffix += (
        String('"mojo_onetrainer_variant_kind":')
        + _json_string(entry.variant_kind, String("mojo_onetrainer_variant_kind"))
        + String(",")
    )
    suffix += (
        String('"mojo_onetrainer_vram_tier_gb":')
        + String(entry.vram_tier_gb)
        + String("}")
    )
    return suffix^


def materialize_onetrainer_preset_config(
    entry: OTPresetCatalogEntry,
    concept_file_name: String,
    sample_definition_file_name: String,
) raises -> String:
    var status = sys_system(String("mkdir -p /tmp/mojo-ot-presets"))
    if status != 0:
        raise Error("OneTrainer preset materializer: cannot create /tmp/mojo-ot-presets")

    var data = _read_file_bytes(entry.config_path)
    var prefix_len = _json_prefix_len_without_closing_object(data)
    var suffix = _materialized_preset_override_suffix(
        entry,
        concept_file_name,
        sample_definition_file_name,
        _json_prefix_needs_comma(data, prefix_len),
    )
    var out_path = _resolved_preset_config_path(entry)
    var fd = sys_open(out_path, O_CREAT | O_WRONLY | O_TRUNC, Int32(0o644))
    if fd < 0:
        raise Error(String("OneTrainer preset materializer: cannot create ") + out_path)
    var offset = _write_list_prefix_at(fd, data, prefix_len, 0)
    offset = _write_string_at(fd, suffix, offset)
    _ = sys_close(fd)
    return out_path^


def _apply_preset_entrypoint_overrides(
    mut cfg: TrainConfig,
    entry: OTPresetCatalogEntry,
    concept_file_name: String,
    sample_definition_file_name: String,
) raises:
    cfg.name = entry.model_type.copy()
    cfg.training_method = entry.training_method
    cfg.optimizer = entry.optimizer

    if concept_file_name != String(""):
        cfg.concept_file_name = concept_file_name.copy()
    elif entry.requires_user_concepts:
        raise Error(
            String("OneTrainer preset catalog: ")
            + entry.preset_id
            + String(" requires a concept_file_name override")
        )

    if sample_definition_file_name != String(""):
        cfg.sample_definition_file_name = sample_definition_file_name.copy()
        cfg.validation_prompts_file = sample_definition_file_name.copy()


def create_onetrainer_train_entrypoint_plan_from_preset(
    preset: String,
    concept_file_name: String,
    sample_definition_file_name: String,
) raises -> OTTrainEntrypointPlan:
    var entry = default_onetrainer_preset_entry(preset)
    validate_onetrainer_preset_entry(entry)
    if not entry.product_run_wired:
        if entry.training_method == TRAINING_METHOD_FINE_TUNE:
            var full_ft_plan = create_full_finetune_product_run_plan(
                entry.model_type, entry.optimizer
            )
            validate_full_finetune_product_run_plan(full_ft_plan)
            raise Error(
                String("OneTrainer preset catalog: full-finetune preset is cataloged but not product-wired: ")
                + entry.preset_id
                + String("; ")
                + full_finetune_product_run_blocker(full_ft_plan)
            )
        raise Error(
            String("OneTrainer preset catalog: preset is cataloged but not product-wired: ")
            + entry.preset_id
        )
    var cfg = read_model_config(entry.config_path)
    _apply_preset_entrypoint_overrides(
        cfg,
        entry,
        concept_file_name,
        sample_definition_file_name,
    )
    var resolved_config_path = materialize_onetrainer_preset_config(
        entry,
        cfg.concept_file_name,
        cfg.sample_definition_file_name,
    )
    var resolved_cfg = read_model_config(resolved_config_path)
    var plan = create_onetrainer_train_entrypoint_plan(resolved_config_path, resolved_cfg)
    plan.source_config_path = entry.config_path.copy()
    plan.named_preset_id = entry.preset_id.copy()
    plan.preset_reference_path = entry.ot_reference_path.copy()
    plan.preset_reference_kind = entry.ot_reference_kind.copy()
    plan.preset_recipe_family = entry.recipe_family.copy()
    plan.preset_variant_kind = entry.variant_kind.copy()
    plan.preset_vram_tier_gb = entry.vram_tier_gb
    plan.resolved_config_materialized = True
    return plan^


def onetrainer_train_entrypoint_summary(plan: OTTrainEntrypointPlan) -> String:
    var preset = String("")
    if plan.named_preset_id != String(""):
        preset = (
            String(" preset=")
            + plan.named_preset_id
            + String(" family=")
            + plan.preset_recipe_family
            + String(" kind=")
            + plan.preset_variant_kind
            + String(" vram_gb=")
            + String(plan.preset_vram_tier_gb)
            + String(" source_config=")
            + plan.source_config_path
            + String(" ref=")
            + plan.preset_reference_path
        )
    return (
        String("train.py config=")
        + plan.config_path
        + preset
        + String(" -> ")
        + onetrainer_product_run_summary(plan.product_plan)
        + String(" ")
        + generic_trainer_lifecycle_summary(plan.lifecycle_plan)
        + String(" concept_file=")
        + plan.product_plan.concept_file_name
        + String(" sample_file=")
        + plan.sample_cadence.sample_definition_file_name
        + String(" ")
        + onetrainer_cache_preflight_summary(plan.cache_preflight)
        + String(" resume=")
        + onetrainer_product_resume_preflight_summary(plan.resume_preflight)
        + String(" workspace_config=")
        + plan.workspace_config_dir
    )


def onetrainer_train_runner_command(plan: OTTrainEntrypointPlan) -> String:
    """Dry-run command for the resolved Mojo runner.

    The command is intentionally informational. This product layer validates the
    OneTrainer entrypoint contract; it does not spawn model loops or touch CUDA.
    """
    return (
        String("pixi run mojo run -I . serenitymojo/training/")
        + plan.product_plan.runner_name
        + String(".mojo ")
        + plan.config_path
        + String(" ")
        + String(plan.max_steps)
    )


def validate_onetrainer_model_loop_direct_invocation_ready(
    plan: OTTrainEntrypointPlan,
) raises:
    validate_onetrainer_train_entrypoint_plan(plan)
    if plan.lifecycle_plan.invocation_kind == OT_MODEL_LOOP_INVOCATION_ABSENT:
        raise Error(
            String("OneTrainer product run: direct model-loop invocation is absent; ")
            + plan.lifecycle_plan.invocation_blocker
            + String("; dry-run command only: ")
            + onetrainer_train_runner_command(plan)
        )
    validate_generic_trainer_lifecycle_plan(plan.lifecycle_plan)

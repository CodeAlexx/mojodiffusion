# onetrainer_product_run_smoke.mojo - OneTrainer train.py entrypoint smoke.
#
# Run:
#   timeout 180 prlimit --as=16000000000 pixi run mojo run -I . serenitymojo/training/onetrainer_product_run_smoke.mojo

from std.memory import alloc

from serenitymojo.io.ffi import (
    sys_open, sys_close, sys_pwrite, sys_system, BytePtr,
    O_WRONLY, O_CREAT, O_TRUNC,
)
from serenitymojo.io.train_config_reader import read_model_config
from serenitymojo.training.onetrainer_product_run import (
    create_onetrainer_train_entrypoint_plan_from_preset,
    create_onetrainer_train_entrypoint_plan,
    onetrainer_train_runner_command,
    onetrainer_train_entrypoint_summary,
    validate_onetrainer_model_loop_direct_invocation_ready,
    validate_onetrainer_train_entrypoint_plan,
)
from serenitymojo.training.onetrainer_preset_catalog import (
    default_onetrainer_preset_entry,
    onetrainer_preset_config_path,
    onetrainer_preset_reference_path,
    onetrainer_preset_summary,
    validate_onetrainer_preset_entry,
)
from serenitymojo.training.sample_prompt_config import (
    SAMPLE_UNIT_NEVER,
    SAMPLE_UNIT_STEP,
    should_sample_completed_step,
)
from serenitymojo.training.train_config import (
    TRAINING_METHOD_FINE_TUNE,
    TRAIN_OPTIMIZER_ADAFACTOR,
    TrainConfig,
)


def _write_file(path: String, content: String) raises:
    var fd = sys_open(path, O_CREAT | O_WRONLY | O_TRUNC, Int32(0o644))
    if fd < 0:
        raise Error(String("product run smoke: cannot create ") + path)
    var n = content.byte_length()
    var buf = alloc[UInt8](n if n > 0 else 1)
    var src = content.as_bytes()
    for i in range(n):
        buf[i] = src[i]
    var wrote = sys_pwrite(fd, BytePtr(unsafe_from_address=Int(buf)), n, 0)
    buf.free()
    _ = sys_close(fd)
    if wrote != n:
        raise Error(String("product run smoke: short write to ") + path)


def _check(cond: Bool, msg: String) raises:
    if not cond:
        raise Error(String("onetrainer_product_run_smoke FAILED: ") + msg)


def _write_qwen_config(path: String, concept_path: String, sample_path: String) raises:
    var content = String("{")
    content += '"model_type":"qwenimage",'
    content += '"workspace_dir":"/tmp/ot-product-qwen-workspace",'
    content += '"cache_dir":"/tmp/ot-product-qwen-cache",'
    content += '"output_model_destination":"/tmp/ot-product-qwen/out.safetensors",'
    content += '"concept_file_name":"' + concept_path + String('",')
    content += '"sample_definition_file_name":"' + sample_path + String('",')
    content += '"validation":true,'
    content += '"training_method":"LORA",'
    content += '"max_steps":123,'
    content += '"sample_after":25,'
    content += '"sample_after_unit":"STEP",'
    content += '"sample_skip_first":0,'
    content += '"save_before_sample":true,'
    content += '"optimizer":{"optimizer":"ADAMW"}'
    content += String("}")
    _write_file(path, content)


def _write_min_product_config(path: String, model_type: String, concept_path: String) raises:
    var content = String("{")
    content += '"model_type":"' + model_type + String('",')
    content += '"workspace_dir":"/tmp/ot-product-all-workspace",'
    content += '"cache_dir":"/tmp/ot-product-all-cache",'
    content += '"concept_file_name":"' + concept_path + String('",')
    content += '"training_method":"LORA",'
    content += '"max_steps":17,'
    content += '"sample_after":0,'
    content += '"sample_after_unit":"NEVER",'
    content += '"optimizer":{"optimizer":"ADAMW"}'
    content += String("}")
    _write_file(path, content)


def _write_min_product_config_with_heads(
    path: String, model_type: String, concept_path: String, num_heads: Int
) raises:
    var content = String("{")
    content += '"model_type":"' + model_type + String('",')
    content += '"workspace_dir":"/tmp/ot-product-all-workspace",'
    content += '"cache_dir":"/tmp/ot-product-all-cache",'
    content += '"concept_file_name":"' + concept_path + String('",')
    content += '"training_method":"LORA",'
    content += '"max_steps":17,'
    content += '"num_heads":' + String(num_heads) + String(",")
    content += '"sample_after":0,'
    content += '"sample_after_unit":"NEVER",'
    content += '"optimizer":{"optimizer":"ADAMW"}'
    content += String("}")
    _write_file(path, content)


def _write_resume_config(
    path: String, workspace_dir: String, concept_path: String, continue_backup: Bool,
) raises:
    var content = String("{")
    content += '"model_type":"qwenimage",'
    content += '"workspace_dir":"' + workspace_dir + String('",')
    content += '"cache_dir":"' + workspace_dir + String('/cache",')
    content += '"concept_file_name":"' + concept_path + String('",')
    content += '"training_method":"LORA",'
    content += '"sample_after":0,'
    content += '"sample_after_unit":"NEVER",'
    if continue_backup:
        content += '"continue_last_backup":true,'
    else:
        content += '"continue_last_backup":false,'
    content += '"optimizer":{"optimizer":"ADAMW"}'
    content += String("}")
    _write_file(path, content)


def _check_target_dispatch(
    model_type: String, expected_runner: String, concept_path: String,
) raises:
    var config_path = (
        String("/tmp/ot_product_dispatch_")
        + model_type
        + String(".json")
    )
    _write_min_product_config(config_path, model_type, concept_path)
    var cfg = read_model_config(config_path)
    var plan = create_onetrainer_train_entrypoint_plan(config_path, cfg)
    validate_onetrainer_train_entrypoint_plan(plan)
    _check(plan.product_plan.runner_name == expected_runner, model_type + String(" runner"))
    _check(plan.max_steps == 17, model_type + String(" max steps"))
    _check(not plan.sample_file_required, model_type + String(" sample disabled"))
    _check(plan.cache_preflight.model_type == model_type, model_type + String(" cache preflight model"))
    _check(plan.cache_preflight.text_train_required_fields != String(""), model_type + String(" train cache fields"))
    _check(plan.cache_preflight.text_sample_required_fields != String(""), model_type + String(" sample cache fields"))
    _check(plan.cache_preflight.vae_cache_channels > 0, model_type + String(" VAE cache channels"))
    if model_type == String("klein"):
        _check(plan.cache_preflight.prepared_only(), "Klein current VAE encoder is prepared-only")
    _check(
        onetrainer_train_runner_command(plan)
        == (
            String("pixi run mojo run -I . serenitymojo/training/")
            + expected_runner
            + String(".mojo ")
            + config_path
            + String(" 17")
        ),
        model_type + String(" command"),
    )


def _gate_all_target_dispatch() raises:
    print("--- product entrypoint dispatches every target model ---")
    var concept_path = String("/tmp/ot_product_all_concepts.json")
    _write_file(concept_path, String("[]"))
    _check_target_dispatch(String("qwenimage"), String("train_qwenimage_real"), concept_path)
    _check_target_dispatch(String("ernie_image"), String("train_ernie_real"), concept_path)
    _check_target_dispatch(String("anima"), String("train_anima_real"), concept_path)
    _check_target_dispatch(String("klein"), String("train_klein_real"), concept_path)
    _check_target_dispatch(String("FLUX_2"), String("train_klein_real"), concept_path)
    _check_target_dispatch(String("zimage"), String("train_zimage_real"), concept_path)
    _check_target_dispatch(String("chroma"), String("train_chroma_real"), concept_path)
    _check_target_dispatch(String("flux"), String("train_flux_real"), concept_path)
    _check_target_dispatch(String("STABLE_DIFFUSION_35"), String("train_sd35_real"), concept_path)
    _check_target_dispatch(
        String("STABLE_DIFFUSION_XL_10_BASE"),
        String("train_sdxl_real"),
        concept_path,
    )
    print("  all target dispatch PASS")


def _gate_good_entrypoint() raises:
    print("--- OneTrainer train.py product entrypoint ---")
    var config_path = String("/tmp/ot_product_qwen_config.json")
    var concept_path = String("/tmp/ot_product_concepts.json")
    var sample_path = String("/tmp/ot_product_samples.json")
    _write_file(concept_path, String("[]"))
    _write_file(sample_path, String("[]"))
    _write_qwen_config(config_path, concept_path, sample_path)

    var cfg = read_model_config(config_path)
    var plan = create_onetrainer_train_entrypoint_plan(config_path, cfg)
    validate_onetrainer_train_entrypoint_plan(plan)

    _check(plan.product_plan.runner_name == String("train_qwenimage_real"), "runner dispatch")
    _check(plan.product_plan.model_type == String("qwenimage"), "model type")
    _check(plan.concept_file_required and plan.concept_file_present, "concept file present")
    _check(plan.sample_file_required and plan.sample_file_present, "sample file present")
    _check(plan.will_save_workspace_config, "workspace config save")
    _check(plan.will_clear_cache_before_training, "cache clear policy")
    _check(plan.will_create_validation_loader, "validation loader policy")
    _check(plan.will_create_sampler, "sampler policy")
    _check(plan.will_end_after_train, "end-after-train policy")
    _check(plan.will_backup_before_final_save, "backup-before-final-save policy")
    _check(plan.lifecycle_plan.will_start, "GenericTrainer.start policy")
    _check(plan.lifecycle_plan.will_train, "GenericTrainer.train policy")
    _check(plan.lifecycle_plan.will_end, "GenericTrainer.end policy")
    _check(not plan.lifecycle_plan.direct_invocation_supported, "direct model-loop invocation blocked")
    _check(plan.workspace_config_dir == String("/tmp/ot-product-qwen-workspace/config"), "workspace config dir")
    _check(plan.cache_preflight.raw_vae_cache_ready(), "Qwen raw VAE cache ready")
    _check(plan.cache_preflight.text_sample_requires_mask, "Qwen sample cache mask required")
    _check(plan.max_steps == 123, "max steps")
    _check(
        onetrainer_train_runner_command(plan)
        == String("pixi run mojo run -I . serenitymojo/training/train_qwenimage_real.mojo /tmp/ot_product_qwen_config.json 123"),
        "runner command",
    )
    _check(plan.is_runnable_contract(), "runnable contract")
    print(onetrainer_train_entrypoint_summary(plan))
    print("  good entrypoint PASS")


def _gate_disabled_sampling() raises:
    print("--- product entrypoint with sampling disabled ---")
    var config_path = String("/tmp/ot_product_no_sample_config.json")
    var concept_path = String("/tmp/ot_product_no_sample_concepts.json")
    _write_file(concept_path, String("[]"))
    var content = String("{")
    content += '"model_type":"STABLE_DIFFUSION_35",'
    content += '"workspace_dir":"/tmp/ot-product-sd35-workspace",'
    content += '"cache_dir":"/tmp/ot-product-sd35-cache",'
    content += '"output_model_destination":"/tmp/ot-product-sd35/out.safetensors",'
    content += '"concept_file_name":"' + concept_path + String('",')
    content += '"training_method":"LORA",'
    content += '"sample_after":0,'
    content += '"sample_after_unit":"NEVER",'
    content += '"optimizer":{"optimizer":"ADAMW"}'
    content += String("}")
    _write_file(config_path, content)

    var cfg = read_model_config(config_path)
    var plan = create_onetrainer_train_entrypoint_plan(config_path, cfg)
    validate_onetrainer_train_entrypoint_plan(plan)
    _check(plan.product_plan.runner_name == String("train_sd35_real"), "SD3.5 runner dispatch")
    _check(plan.sample_cadence.sample_after_unit == SAMPLE_UNIT_NEVER, "sample disabled")
    _check(not plan.sample_file_required, "sample file not required")
    _check(plan.is_runnable_contract(), "disabled-sampling contract runnable")
    print("  disabled sampling PASS")


def _check_named_preset_dispatch(
    preset: String,
    expected_runner: String,
    expected_model_type: String,
    concept_path: String,
    sample_path: String,
) raises:
    var entry = default_onetrainer_preset_entry(preset)
    validate_onetrainer_preset_entry(entry)
    _check(entry.model_type == expected_model_type, preset + String(" model type"))
    _check(onetrainer_preset_config_path(preset) == entry.config_path, preset + String(" config path"))
    _check(
        onetrainer_preset_reference_path(preset) == entry.ot_reference_path,
        preset + String(" reference path"),
    )
    var plan = create_onetrainer_train_entrypoint_plan_from_preset(
        preset,
        concept_path,
        sample_path,
    )
    validate_onetrainer_train_entrypoint_plan(plan)
    _check(plan.product_plan.runner_name == expected_runner, preset + String(" runner"))
    _check(plan.product_plan.model_type == expected_model_type, preset + String(" plan model type"))
    _check(plan.named_preset_id == entry.preset_id, preset + String(" preset id"))
    _check(plan.source_config_path == entry.config_path, preset + String(" source config"))
    _check(plan.preset_reference_path == entry.ot_reference_path, preset + String(" reference path"))
    _check(plan.preset_reference_kind == entry.ot_reference_kind, preset + String(" reference kind"))
    _check(plan.preset_recipe_family == entry.recipe_family, preset + String(" recipe family"))
    _check(plan.preset_variant_kind == entry.variant_kind, preset + String(" variant kind"))
    _check(plan.preset_vram_tier_gb == entry.vram_tier_gb, preset + String(" VRAM tier"))
    _check(plan.resolved_config_materialized, preset + String(" resolved config materialized"))
    _check(
        plan.config_path == String("/tmp/mojo-ot-presets/") + entry.preset_id + String(".json"),
        preset + String(" resolved config path"),
    )
    _check(
        onetrainer_train_runner_command(plan)
        == (
            String("pixi run mojo run -I . serenitymojo/training/")
            + expected_runner
            + String(".mojo ")
            + plan.config_path
            + String(" ")
            + String(plan.max_steps)
        ),
        preset + String(" materialized command"),
    )
    _check(plan.concept_file_present, preset + String(" concept override"))
    _check(plan.sample_file_present, preset + String(" sample override"))
    _check(plan.is_runnable_contract(), preset + String(" runnable contract"))
    print("  preset ", onetrainer_preset_summary(entry))


def _expect_named_preset_raises(
    label: String,
    preset: String,
    concept_path: String,
    sample_path: String,
) raises:
    var raised = False
    try:
        var plan = create_onetrainer_train_entrypoint_plan_from_preset(
            preset,
            concept_path,
            sample_path,
        )
        validate_onetrainer_train_entrypoint_plan(plan)
    except e:
        raised = True
        print("  raised as expected [", label, "]:", String(e))
    if not raised:
        raise Error(String("product run smoke: expected named preset raise for ") + label)


def _gate_named_preset_catalog() raises:
    print("--- OneTrainer-style named preset catalog ---")
    var concept_path = String("/tmp/ot_product_named_concepts.json")
    var sample_path = String("/tmp/ot_product_named_samples.json")
    _write_file(concept_path, String("[]"))
    _write_file(sample_path, String("[]"))

    _check_named_preset_dispatch(
        String("qwen_lora_16gb"),
        String("train_qwenimage_real"),
        String("qwenimage"),
        concept_path,
        sample_path,
    )
    _check_named_preset_dispatch(
        String("qwen_lora_24gb"),
        String("train_qwenimage_real"),
        String("qwenimage"),
        concept_path,
        sample_path,
    )
    _check_named_preset_dispatch(
        String("ernie_lora_8gb"),
        String("train_ernie_real"),
        String("ernie_image"),
        concept_path,
        sample_path,
    )
    _check_named_preset_dispatch(
        String("ernie_lora_16gb"),
        String("train_ernie_real"),
        String("ernie_image"),
        concept_path,
        sample_path,
    )
    _check_named_preset_dispatch(
        String("anima_lora"),
        String("train_anima_real"),
        String("anima"),
        concept_path,
        sample_path,
    )
    _check_named_preset_dispatch(
        String("sd3-5"),
        String("train_sd35_real"),
        String("STABLE_DIFFUSION_35"),
        concept_path,
        sample_path,
    )
    _check_named_preset_dispatch(
        String("sdxl_1_0_lora"),
        String("train_sdxl_real"),
        String("sdxl"),
        concept_path,
        sample_path,
    )
    _check_named_preset_dispatch(
        String("flux1_dev"),
        String("train_flux_real"),
        String("flux"),
        concept_path,
        sample_path,
    )
    _check_named_preset_dispatch(
        String("flux2_lora_8gb"),
        String("train_klein_real"),
        String("klein"),
        concept_path,
        sample_path,
    )
    _check_named_preset_dispatch(
        String("flux2_lora_16gb"),
        String("train_klein_real"),
        String("klein"),
        concept_path,
        sample_path,
    )
    _check_named_preset_dispatch(
        String("chroma_lora_8gb"),
        String("train_chroma_real"),
        String("chroma"),
        concept_path,
        sample_path,
    )
    _check_named_preset_dispatch(
        String("chroma_lora_16gb"),
        String("train_chroma_real"),
        String("chroma"),
        concept_path,
        sample_path,
    )
    _check_named_preset_dispatch(
        String("chroma_lora_24gb"),
        String("train_chroma_real"),
        String("chroma"),
        concept_path,
        sample_path,
    )
    _check_named_preset_dispatch(
        String("zimage_lora_8gb"),
        String("train_zimage_real"),
        String("zimage"),
        concept_path,
        sample_path,
    )
    _check_named_preset_dispatch(
        String("zimage_lora_16gb"),
        String("train_zimage_real"),
        String("zimage"),
        concept_path,
        sample_path,
    )

    _expect_named_preset_raises(
        String("plain SD3 catalog alias is blocked"),
        String("sd3"),
        concept_path,
        sample_path,
    )
    _expect_named_preset_raises(
        String("plain SD3 LoRA catalog alias is blocked"),
        String("sd3_lora"),
        concept_path,
        sample_path,
    )
    _expect_named_preset_raises(
        String("plain SD3 model type is blocked"),
        String("STABLE_DIFFUSION_3"),
        concept_path,
        sample_path,
    )
    _expect_named_preset_raises(
        String("full finetune preset is cataloged but not product-wired"),
        String("qwen_finetune_16gb"),
        concept_path,
        sample_path,
    )
    _expect_named_preset_raises(
        String("Klein 4B is not routed through the 9B runner"),
        String("klein4b"),
        concept_path,
        sample_path,
    )
    _expect_named_preset_raises(
        String("Z-Image DeTurbo preset lacks matching local product config"),
        String("zimage_deturbo_lora_16gb"),
        concept_path,
        sample_path,
    )
    _expect_named_preset_raises(
        String("SDXL inpaint preset lacks matching local product config"),
        String("sdxl_inpaint_lora"),
        concept_path,
        sample_path,
    )
    _expect_named_preset_raises(
        String("named preset without user concept file"),
        String("qwen_lora_16gb"),
        String(""),
        sample_path,
    )
    print("  named preset catalog PASS")


def _gate_zimage_product_readiness() raises:
    print("--- Z-Image named preset binds product cache/sample/save policy ---")
    var concept_path = String("/tmp/ot_product_zimage_concepts.json")
    var sample_path = String("/tmp/ot_product_zimage_samples.json")
    _write_file(concept_path, String("[]"))
    _write_file(sample_path, String("[]"))

    var plan = create_onetrainer_train_entrypoint_plan_from_preset(
        String("zimage_lora_16gb"),
        concept_path,
        sample_path,
    )
    validate_onetrainer_train_entrypoint_plan(plan)

    _check(plan.named_preset_id == String("zimage_lora_16gb"), "Z-Image named preset id")
    _check(plan.product_plan.model_type == String("zimage"), "Z-Image product model")
    _check(plan.product_plan.runner_name == String("train_zimage_real"), "Z-Image runner")
    _check(plan.product_plan.is_product_ready(), "Z-Image product runner ready")
    _check(plan.cache_preflight.model_type == String("zimage"), "Z-Image preflight model")
    _check(plan.cache_preflight.text_contract_name == String("zimage"), "Z-Image text contract")
    _check(
        plan.cache_preflight.text_train_required_fields
        == String("tokens,text_encoder_hidden_state,tokens_mask"),
        "Z-Image train cache fields",
    )
    _check(
        plan.cache_preflight.text_sample_required_fields
        == String("text_encoder_hidden_state,tokens_mask"),
        "Z-Image sample cache fields",
    )
    _check(plan.cache_preflight.text_train_requires_mask, "Z-Image train cache mask")
    _check(plan.cache_preflight.text_sample_requires_mask, "Z-Image sample cache mask")
    _check(not plan.cache_preflight.text_requires_runtime_ids, "Z-Image no runtime ids")
    _check(plan.cache_preflight.raw_vae_cache_ready(), "Z-Image raw VAE cache ready")
    _check(plan.cache_preflight.vae_cache_channels == 16, "Z-Image VAE cache channels")
    _check(plan.cache_preflight.vae_prepared_channels == 16, "Z-Image VAE prepared channels")
    _check(
        plan.cache_preflight.vae_cache_to_prepared_patch_size == 1,
        "Z-Image VAE cache/prepared patch",
    )
    _check(
        plan.cache_preflight.dtype_policy
        == String("preserve_checkpoint_or_train_dtype_at_tensor_boundaries"),
        "Z-Image dtype preflight policy",
    )
    _check(plan.sample_file_required and plan.sample_file_present, "Z-Image sample file required")
    _check(plan.sample_cadence.sample_after == 500, "Z-Image sample cadence step")
    _check(plan.sample_cadence.sample_after_unit == SAMPLE_UNIT_STEP, "Z-Image sample cadence unit")
    _check(plan.sample_cadence.save_before_sample, "Z-Image save-before-sample policy")
    _check(should_sample_completed_step(plan.sample_cadence, 500), "Z-Image sample due at 500")
    _check(not should_sample_completed_step(plan.sample_cadence, 499), "Z-Image no early sample")
    _check(plan.will_create_sampler, "Z-Image product sampler creation policy")
    _check(plan.is_runnable_contract(), "Z-Image runnable product contract")
    print("  Z-Image product readiness PASS")


def _gate_resume_preflight() raises:
    print("--- product entrypoint continue_last_backup preflight ---")
    var concept_path = String("/tmp/ot_product_resume_concepts.json")
    _write_file(concept_path, String("[]"))

    var good_workspace = String("/tmp/ot-product-resume-good")
    _ = sys_system(String("rm -rf ") + good_workspace)
    _ = sys_system(
        String("mkdir -p ")
        + good_workspace
        + String("/backup/20260605-120000-backup-5-0-5/lora ")
        + good_workspace
        + String("/backup/20260605-120000-backup-5-0-5/optimizer ")
        + good_workspace
        + String("/backup/20260604-120000-backup-4-0-4/lora ")
        + good_workspace
        + String("/backup/20260604-120000-backup-4-0-4/optimizer")
    )
    _write_file(
        good_workspace + String("/backup/20260605-120000-backup-5-0-5/meta.json"),
        String("{}"),
    )
    _write_file(
        good_workspace + String("/backup/20260605-120000-backup-5-0-5/lora/lora.safetensors"),
        String("lora"),
    )
    _write_file(
        good_workspace + String("/backup/20260605-120000-backup-5-0-5/optimizer/optimizer.pt"),
        String("optimizer"),
    )
    _write_file(
        good_workspace + String("/backup/20260604-120000-backup-4-0-4/meta.json"),
        String("{}"),
    )

    var config_path = String("/tmp/ot_product_resume_good_config.json")
    _write_resume_config(config_path, good_workspace, concept_path, True)
    var cfg = read_model_config(config_path)
    var plan = create_onetrainer_train_entrypoint_plan(config_path, cfg)
    validate_onetrainer_train_entrypoint_plan(plan)
    _check(plan.resume_preflight.continue_last_backup, "resume flag parsed")
    _check(plan.resume_preflight.found_last_backup, "resume backup found")
    _check(
        plan.resume_preflight.last_backup_path
        == good_workspace + String("/backup/20260605-120000-backup-5-0-5"),
        "latest backup selected",
    )
    _check(plan.resume_preflight.meta_present, "resume meta present")
    _check(plan.resume_preflight.lora_state_present, "resume lora present")
    _check(plan.resume_preflight.optimizer_state_present, "resume optimizer present")
    _check(plan.resume_preflight.will_resume_training_state, "resume training state")

    var empty_workspace = String("/tmp/ot-product-resume-empty")
    _ = sys_system(String("rm -rf ") + empty_workspace)
    _ = sys_system(String("mkdir -p ") + empty_workspace)
    config_path = String("/tmp/ot_product_resume_empty_config.json")
    _write_resume_config(config_path, empty_workspace, concept_path, True)
    cfg = read_model_config(config_path)
    plan = create_onetrainer_train_entrypoint_plan(config_path, cfg)
    validate_onetrainer_train_entrypoint_plan(plan)
    _check(plan.resume_preflight.continue_last_backup, "empty resume flag parsed")
    _check(not plan.resume_preflight.found_last_backup, "empty backup is cold start")
    _check(not plan.resume_preflight.will_resume_training_state, "empty backup no resume")

    var missing_workspace = String("/tmp/ot-product-resume-missing-optimizer")
    _ = sys_system(String("rm -rf ") + missing_workspace)
    _ = sys_system(
        String("mkdir -p ")
        + missing_workspace
        + String("/backup/20260605-120000-backup-5-0-5/lora")
    )
    _write_file(
        missing_workspace + String("/backup/20260605-120000-backup-5-0-5/meta.json"),
        String("{}"),
    )
    _write_file(
        missing_workspace + String("/backup/20260605-120000-backup-5-0-5/lora/lora.safetensors"),
        String("lora"),
    )
    config_path = String("/tmp/ot_product_resume_missing_optimizer_config.json")
    _write_resume_config(config_path, missing_workspace, concept_path, True)
    cfg = read_model_config(config_path)
    _expect_plan_raises(String("missing optimizer state for LoRA resume"), config_path, cfg)

    print("  resume preflight PASS")


def _expect_plan_raises(label: String, config_path: String, cfg: TrainConfig) raises:
    var raised = False
    try:
        var plan = create_onetrainer_train_entrypoint_plan(config_path, cfg)
        validate_onetrainer_train_entrypoint_plan(plan)
    except e:
        raised = True
        print("  raised as expected [", label, "]:", String(e))
    if not raised:
        raise Error(String("product run smoke: expected raise for ") + label)


def _gate_fail_loud() raises:
    print("--- product entrypoint fail-loud cases ---")
    var config_path = String("/tmp/ot_product_bad_config.json")
    var concept_path = String("/tmp/ot_product_bad_concepts.json")
    var sample_path = String("/tmp/ot_product_missing_samples.json")
    _write_file(concept_path, String("[]"))
    _write_qwen_config(config_path, concept_path, sample_path)

    var cfg = read_model_config(config_path)
    _expect_plan_raises(String("missing sample file"), config_path, cfg)

    var flux2_dev_alias_path = String("/tmp/ot_product_flux2_dev_alias.json")
    _write_min_product_config(flux2_dev_alias_path, String("FLUX_2_DEV"), concept_path)
    cfg = read_model_config(flux2_dev_alias_path)
    _expect_plan_raises(String("Flux2 dev has no product runner"), flux2_dev_alias_path, cfg)

    var flux2_dev_heads_path = String("/tmp/ot_product_flux2_dev_heads.json")
    _write_min_product_config_with_heads(
        flux2_dev_heads_path, String("FLUX_2"), concept_path, 48
    )
    cfg = read_model_config(flux2_dev_heads_path)
    _expect_plan_raises(
        String("FLUX_2 num_attention_heads==48 has no product runner"),
        flux2_dev_heads_path,
        cfg,
    )

    var plain_sd3_path = String("/tmp/ot_product_plain_sd3.json")
    _write_min_product_config(plain_sd3_path, String("STABLE_DIFFUSION_3"), concept_path)
    cfg = read_model_config(plain_sd3_path)
    _expect_plan_raises(
        String("plain SD3 is not a target"),
        plain_sd3_path,
        cfg,
    )

    cfg = read_model_config(config_path)
    cfg.sample_after_unit = SAMPLE_UNIT_NEVER
    cfg.training_method = TRAINING_METHOD_FINE_TUNE
    _expect_plan_raises(String("full finetune product loop"), config_path, cfg)

    cfg = read_model_config(config_path)
    cfg.sample_after_unit = SAMPLE_UNIT_NEVER
    cfg.optimizer = TRAIN_OPTIMIZER_ADAFACTOR
    _expect_plan_raises(String("unsupported optimizer"), config_path, cfg)

    cfg = read_model_config(config_path)
    cfg.sample_after_unit = SAMPLE_UNIT_NEVER
    cfg.concept_file_name = String("/tmp/ot_product_missing_concepts.json")
    _expect_plan_raises(String("missing concept file"), config_path, cfg)

    cfg = read_model_config(config_path)
    _expect_plan_raises(String("missing config path"), String("/tmp/does-not-exist-ot-product.json"), cfg)

    var direct_path = String("/tmp/ot_product_direct_invocation_config.json")
    _write_min_product_config(direct_path, String("qwenimage"), concept_path)
    cfg = read_model_config(direct_path)
    var direct_plan = create_onetrainer_train_entrypoint_plan(direct_path, cfg)
    validate_onetrainer_train_entrypoint_plan(direct_plan)
    var direct_raised = False
    try:
        validate_onetrainer_model_loop_direct_invocation_ready(direct_plan)
    except e:
        direct_raised = True
        var msg = String(e)
        print("  raised as expected [direct model-loop invocation absent]:", msg)
        _check(
            msg.find(String("direct model-loop invocation is absent")) >= 0,
            "direct invocation absent message",
        )
        _check(
            msg.find(onetrainer_train_runner_command(direct_plan)) >= 0,
            "direct invocation blocker includes dry-run command",
        )
    _check(direct_raised, "direct model-loop invocation must fail loud")

    var only_cache_path = String("/tmp/ot_product_flux_only_cache.json")
    var only_cache_content = String("{")
    only_cache_content += '"model_type":"flux",'
    only_cache_content += '"workspace_dir":"/tmp/ot-product-flux-workspace",'
    only_cache_content += '"cache_dir":"/tmp/ot-product-flux-cache",'
    only_cache_content += '"concept_file_name":"' + concept_path + String('",')
    only_cache_content += '"training_method":"LORA",'
    only_cache_content += '"sample_after":0,'
    only_cache_content += '"sample_after_unit":"NEVER",'
    only_cache_content += '"only_cache":true,'
    only_cache_content += '"optimizer":{"optimizer":"ADAMW"}'
    only_cache_content += String("}")
    _write_file(only_cache_path, only_cache_content)
    cfg = read_model_config(only_cache_path)
    _expect_plan_raises(String("only_cache raw VAE not ready"), only_cache_path, cfg)

    print("  fail-loud cases PASS")


def main() raises:
    print("==== OneTrainer product-run entrypoint smoke ====")
    _gate_good_entrypoint()
    _gate_all_target_dispatch()
    _gate_disabled_sampling()
    _gate_named_preset_catalog()
    _gate_zimage_product_readiness()
    _gate_resume_preflight()
    _gate_fail_loud()
    print("onetrainer_product_run_smoke PASS")

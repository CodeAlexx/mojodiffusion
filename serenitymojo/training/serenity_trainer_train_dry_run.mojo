# serenity_trainer_train_dry_run.mojo -- no-CUDA SerenityTrainer product entrypoint dry run.
#
# This mirrors the front of SerenityTrainer `scripts/train.py`: read one training
# config, validate concept/sample/product runner requirements, and print the
# Mojo runner command. It intentionally does not spawn the runner or create a
# CUDA DeviceContext.
#
# Run:
#   pixi run mojo run -I . serenitymojo/training/serenity_trainer_train_dry_run.mojo <config.json>
#   pixi run mojo run -I . serenitymojo/training/serenity_trainer_train_dry_run.mojo --preset qwen_lora_16gb <concepts.json> <samples.json>

from std.sys import argv

from serenitymojo.io.train_config_reader import read_model_config
from serenitymojo.training.serenity_trainer_product_run import (
    SerenityTrainEntrypointPlan,
    create_serenity_trainer_train_entrypoint_plan,
    create_serenity_trainer_train_entrypoint_plan_from_preset,
    serenity_trainer_train_entrypoint_summary,
    serenity_trainer_train_runner_command,
    validate_serenity_trainer_train_entrypoint_plan,
)


def _print_plan(plan: SerenityTrainEntrypointPlan) raises:
    validate_serenity_trainer_train_entrypoint_plan(plan)

    print("==== SerenityTrainer Mojo dry run ====")
    print(serenity_trainer_train_entrypoint_summary(plan))
    print("runner_command:", serenity_trainer_train_runner_command(plan))
    print("source_config:", plan.source_config_path)
    print("resolved_config_materialized:", plan.resolved_config_materialized)
    print("preset:", plan.named_preset_id)
    print("preset_family:", plan.preset_recipe_family)
    print("preset_kind:", plan.preset_variant_kind)
    print("preset_vram_gb:", plan.preset_vram_tier_gb)
    print("preset_reference:", plan.preset_reference_path)
    print("only_cache:", plan.will_only_cache)
    print("validation:", plan.will_create_validation_loader)
    print("sampler:", plan.will_create_sampler)
    print("text_train_cache_fields:", plan.cache_preflight.text_train_required_fields)
    print("text_sample_cache_fields:", plan.cache_preflight.text_sample_required_fields)
    print("raw_vae_cache_ready:", plan.cache_preflight.raw_vae_cache_ready())
    print("prepared_vae_only:", plan.cache_preflight.prepared_only())
    print("workspace_config:", plan.workspace_config_dir)
    print("workspace_save:", plan.workspace_save_dir)
    print("workspace_backup:", plan.workspace_backup_dir)
    print("workspace_samples:", plan.workspace_samples_dir)
    print("dry_run PASS")


def main() raises:
    var a = argv()
    if len(a) < 2:
        raise Error("usage: serenity_trainer_train_dry_run.mojo <config.json> OR --preset <preset> <concepts.json> [samples.json]")

    if String(a[1]) == String("--preset"):
        if len(a) < 4:
            raise Error("usage: serenity_trainer_train_dry_run.mojo --preset <preset> <concepts.json> [samples.json]")
        var sample_path = String("")
        if len(a) >= 5:
            sample_path = String(a[4])
        var preset_plan = create_serenity_trainer_train_entrypoint_plan_from_preset(
            String(a[2]),
            String(a[3]),
            sample_path,
        )
        _print_plan(preset_plan)
        return

    var config_path = String(a[1])
    var cfg = read_model_config(config_path)
    var plan = create_serenity_trainer_train_entrypoint_plan(config_path, cfg)
    _print_plan(plan)

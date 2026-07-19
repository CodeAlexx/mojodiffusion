# sample_prompt_config_smoke.mojo -- contract gate for reference trainer-style sample prompts.
#
# Build/run:
#   pixi run mojo run -I . serenitymojo/training/sample_prompt_config_smoke.mojo

from std.sys import argv

from serenitymojo.training.sample_prompt_config import (
    SAMPLE_UNIT_STEP,
    cadence_from_prompt_config,
    next_sample_completed_step,
    read_sample_cadence_config,
    read_sample_prompt_config,
    sample_time_unit_name,
    should_sample_completed_step,
)


comptime DEFAULT_WRAPPED = "/home/alex/mojodiffusion/serenitymojo/configs/sample_prompts.example.json"
comptime DEFAULT_SERENITY_SAMPLES = "/home/alex/SerenityTrainer/training_samples/eri2_5prompts.json"
comptime DEFAULT_SERENITY_TRAIN = "/home/alex/SerenityTrainer/configs/eri2_zimage_base_2500.json"


def _check(cond: Bool, msg: String) raises:
    if not cond:
        raise Error(String("sample_prompt_config_smoke FAILED: ") + msg)


def main() raises:
    var args = argv()
    var wrapped_path = String(DEFAULT_WRAPPED)
    if len(args) >= 2:
        wrapped_path = String(args[1])
    var serenity_samples_path = String(DEFAULT_SERENITY_SAMPLES)
    if len(args) >= 3:
        serenity_samples_path = String(args[2])
    var serenity_train_path = String(DEFAULT_SERENITY_TRAIN)
    if len(args) >= 4:
        serenity_train_path = String(args[3])

    print("=== sample prompt config smoke ===")

    var wrapped = read_sample_prompt_config(wrapped_path)
    _check(wrapped.schema == String("serenity.sample_prompts.v1"), "wrapped schema")
    _check(len(wrapped.prompts) == 2, "wrapped prompt count")
    _check(wrapped.prompts[0].label == String("portrait_daylight"), "wrapped id")
    _check(wrapped.prompts[1].seed == UInt64(43), "wrapped per-prompt seed")
    _check(wrapped.prompts[0].width == 1024 and wrapped.prompts[0].height == 1024, "wrapped size")
    var wrapped_cadence = cadence_from_prompt_config(wrapped)
    _check(wrapped_cadence.sample_after == 500, "wrapped cadence every")
    _check(wrapped_cadence.sample_after_unit == SAMPLE_UNIT_STEP, "wrapped cadence unit")
    _check(should_sample_completed_step(wrapped_cadence, 0), "wrapped start sample")
    _check(not should_sample_completed_step(wrapped_cadence, 1), "wrapped no step 1 sample")
    _check(should_sample_completed_step(wrapped_cadence, 500), "wrapped step 500 sample")
    _check(next_sample_completed_step(wrapped_cadence, 0, 2000) == 500, "wrapped next step")
    print("  wrapped prompt config PASS")

    var serenity_samples = read_sample_prompt_config(serenity_samples_path)
    _check(serenity_samples.schema == String("serenity_trainer.samples.v1"), "reference trainer list schema")
    _check(len(serenity_samples.prompts) == 5, "reference trainer prompt count")
    _check(serenity_samples.prompts[0].enabled, "reference trainer enabled")
    _check(serenity_samples.prompts[0].width == 512 and serenity_samples.prompts[0].height == 512, "reference trainer size")
    _check(serenity_samples.prompts[0].steps == 20, "reference trainer diffusion_steps")
    _check(serenity_samples.prompts[0].cfg == Float32(3.5), "reference trainer cfg_scale")
    _check(serenity_samples.prompts[0].noise_scheduler == String("EULER"), "reference trainer scheduler")
    _check(not serenity_samples.precache_required, "reference trainer list does not require caps")
    print("  SerenityTrainer sample list PASS")

    var serenity_cadence = read_sample_cadence_config(serenity_train_path)
    _check(serenity_cadence.sample_definition_file_name == String("/home/alex/SerenityTrainer/training_samples/eri2_5prompts.json"), "reference trainer sample file")
    _check(serenity_cadence.sample_after == 500, "reference trainer sample_after")
    _check(sample_time_unit_name(serenity_cadence.sample_after_unit) == String("STEP"), "reference trainer unit STEP")
    _check(should_sample_completed_step(serenity_cadence, 500), "reference trainer cadence step 500")
    _check(not should_sample_completed_step(serenity_cadence, 501), "reference trainer cadence no step 501")
    _check(next_sample_completed_step(serenity_cadence, 500, 2500) == 1000, "reference trainer cadence next")
    print("  SerenityTrainer cadence PASS")

    print("sample_prompt_config_smoke PASS")

# sample_prompt_config_smoke.mojo -- contract gate for OT-style sample prompts.
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
comptime DEFAULT_OT_SAMPLES = "/home/alex/OneTrainer/training_samples/eri2_5prompts.json"
comptime DEFAULT_OT_TRAIN = "/home/alex/OneTrainer/configs/eri2_zimage_base_2500.json"


def _check(cond: Bool, msg: String) raises:
    if not cond:
        raise Error(String("sample_prompt_config_smoke FAILED: ") + msg)


def main() raises:
    var args = argv()
    var wrapped_path = String(DEFAULT_WRAPPED)
    if len(args) >= 2:
        wrapped_path = String(args[1])
    var ot_samples_path = String(DEFAULT_OT_SAMPLES)
    if len(args) >= 3:
        ot_samples_path = String(args[2])
    var ot_train_path = String(DEFAULT_OT_TRAIN)
    if len(args) >= 4:
        ot_train_path = String(args[3])

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

    var ot_samples = read_sample_prompt_config(ot_samples_path)
    _check(ot_samples.schema == String("onetrainer.samples.v1"), "OT list schema")
    _check(len(ot_samples.prompts) == 5, "OT prompt count")
    _check(ot_samples.prompts[0].enabled, "OT enabled")
    _check(ot_samples.prompts[0].width == 512 and ot_samples.prompts[0].height == 512, "OT size")
    _check(ot_samples.prompts[0].steps == 20, "OT diffusion_steps")
    _check(ot_samples.prompts[0].cfg == Float32(3.5), "OT cfg_scale")
    _check(ot_samples.prompts[0].noise_scheduler == String("EULER"), "OT scheduler")
    _check(not ot_samples.precache_required, "OT list does not require caps")
    print("  OneTrainer sample list PASS")

    var ot_cadence = read_sample_cadence_config(ot_train_path)
    _check(ot_cadence.sample_definition_file_name == String("/home/alex/OneTrainer/training_samples/eri2_5prompts.json"), "OT sample file")
    _check(ot_cadence.sample_after == 500, "OT sample_after")
    _check(sample_time_unit_name(ot_cadence.sample_after_unit) == String("STEP"), "OT unit STEP")
    _check(should_sample_completed_step(ot_cadence, 500), "OT cadence step 500")
    _check(not should_sample_completed_step(ot_cadence, 501), "OT cadence no step 501")
    _check(next_sample_completed_step(ot_cadence, 500, 2500) == 1000, "OT cadence next")
    print("  OneTrainer cadence PASS")

    print("sample_prompt_config_smoke PASS")

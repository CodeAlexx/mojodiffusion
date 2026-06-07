# sampling/product_sampler_harness.mojo - OneTrainer product sampler lifecycle.
#
# This is a fail-loud harness contract, not a denoiser. OneTrainer samples run:
# SampleConfig -> text conditioning -> transformer denoise/progress callbacks ->
# VAE decode -> postprocess/save -> on_sample callback. The model-specific
# samplers own the real math; this file records the product path and measurement
# fields that must be present before speed or image parity can be accepted.

from serenitymojo.sampling.onetrainer_sampler_contract import (
    OneTrainerSamplerPlan,
    ot_sampler_plan_with_sample_overrides,
    validate_ot_sampler_plan,
)
from serenitymojo.training.sample_prompt_config import SamplePrompt


@fieldwise_init
struct ProductSamplerRunContract(Copyable, Movable):
    var plan: OneTrainerSamplerPlan
    var prompt: SamplePrompt
    var destination: String
    var image_format: String
    var progress_callback_name: String
    var output_callback_name: String


@fieldwise_init
struct SamplerProductStageStatus(Copyable, Movable):
    var sample_config_ready: Bool
    var text_conditioning_ready: Bool
    var transformer_denoise_ready: Bool
    var vae_decode_ready: Bool
    var postprocess_save_ready: Bool
    var progress_callbacks_ready: Bool
    var output_callback_ready: Bool
    var timing_ready: Bool
    var vram_ready: Bool


@fieldwise_init
struct SamplerProductMeasurements(Copyable, Movable):
    var ot_baseline_seconds_per_step: Float64
    var ot_peak_vram_mib: Int
    var mojo_total_wall_seconds: Float64
    var mojo_text_stage_seconds: Float64
    var mojo_denoise_stage_seconds: Float64
    var mojo_vae_decode_stage_seconds: Float64
    var mojo_postprocess_save_seconds: Float64
    var mojo_seconds_per_step: Float64
    var mojo_peak_vram_mib: Int
    var progress_updates: Int
    var measured_steps: Int
    var speed_parity_accepted: Bool


def build_product_sampler_run_contract(
    model_type: String,
    prompt: SamplePrompt,
    destination: String,
    image_format: String = String("PNG"),
) raises -> ProductSamplerRunContract:
    var plan = ot_sampler_plan_with_sample_overrides(
        model_type,
        prompt.width,
        prompt.height,
        prompt.steps,
        prompt.cfg,
    )
    validate_ot_sampler_plan(plan)
    return ProductSamplerRunContract(
        plan^,
        prompt.copy(),
        destination.copy(),
        image_format.copy(),
        String("on_update_progress"),
        String("on_sample"),
    )


def sampler_product_scaffold_status() -> SamplerProductStageStatus:
    return SamplerProductStageStatus(
        True,   # SamplePrompt/SampleConfig parsing exists.
        False,  # text conditioning is still owned by the model-specific path.
        False,  # transformer denoise trajectory is not wired here.
        False,  # VAE decode is not wired here.
        False,  # postprocess/save is not wired here.
        False,  # OneTrainer progress callback count is not proven here.
        False,  # on_sample image callback is not proven here.
        False,  # per-stage timing is not measured here.
        False,  # peak VRAM is not measured here.
    )


def sampler_product_ready_status() -> SamplerProductStageStatus:
    return SamplerProductStageStatus(True, True, True, True, True, True, True, True, True)


def empty_sampler_product_measurements() -> SamplerProductMeasurements:
    return SamplerProductMeasurements(
        Float64(0.0),
        0,
        Float64(0.0),
        Float64(0.0),
        Float64(0.0),
        Float64(0.0),
        Float64(0.0),
        Float64(0.0),
        0,
        0,
        0,
        False,
    )


def _missing_append(current: String, name: String) -> String:
    if current == String(""):
        return name.copy()
    return current + String(", ") + name


def product_sampler_missing_summary(status: SamplerProductStageStatus) -> String:
    var missing = String("")
    if not status.sample_config_ready:
        missing = _missing_append(missing, String("sample_config"))
    if not status.text_conditioning_ready:
        missing = _missing_append(missing, String("text_conditioning"))
    if not status.transformer_denoise_ready:
        missing = _missing_append(missing, String("transformer_denoise"))
    if not status.vae_decode_ready:
        missing = _missing_append(missing, String("vae_decode"))
    if not status.postprocess_save_ready:
        missing = _missing_append(missing, String("postprocess_save"))
    if not status.progress_callbacks_ready:
        missing = _missing_append(missing, String("progress_callbacks"))
    if not status.output_callback_ready:
        missing = _missing_append(missing, String("output_callback"))
    if not status.timing_ready:
        missing = _missing_append(missing, String("timing_seconds_per_step"))
    if not status.vram_ready:
        missing = _missing_append(missing, String("peak_vram_mib"))
    if missing == String(""):
        return String("none")
    return missing^


def product_sampler_is_ready(status: SamplerProductStageStatus) -> Bool:
    return product_sampler_missing_summary(status) == String("none")


def validate_product_sampler_run_contract(run: ProductSamplerRunContract) raises:
    validate_ot_sampler_plan(run.plan)
    if not run.prompt.enabled:
        raise Error("product sampler harness: prompt is disabled")
    if run.prompt.prompt == String(""):
        raise Error("product sampler harness: prompt text is empty")
    if run.destination == String(""):
        raise Error("product sampler harness: destination is empty")
    if run.progress_callback_name != String("on_update_progress"):
        raise Error("product sampler harness: progress callback must mirror OneTrainer")
    if run.output_callback_name != String("on_sample"):
        raise Error("product sampler harness: output callback must mirror OneTrainer")


def sampler_speed_ratio(measurements: SamplerProductMeasurements) -> Float64:
    if measurements.ot_baseline_seconds_per_step <= Float64(0.0):
        return Float64(0.0)
    return measurements.mojo_seconds_per_step / measurements.ot_baseline_seconds_per_step


def validate_sampler_measurements(
    measurements: SamplerProductMeasurements,
    expected_steps: Int,
) raises:
    if expected_steps <= 0:
        raise Error("product sampler harness: expected_steps must be positive")
    if measurements.measured_steps != expected_steps:
        raise Error("product sampler harness: measured_steps must equal diffusion steps")
    if measurements.progress_updates != expected_steps:
        raise Error("product sampler harness: progress callback count must equal diffusion steps")
    if measurements.ot_baseline_seconds_per_step <= Float64(0.0):
        raise Error("product sampler harness: missing OneTrainer seconds/step baseline")
    if measurements.ot_peak_vram_mib <= 0:
        raise Error("product sampler harness: missing OneTrainer peak VRAM baseline")
    if measurements.mojo_total_wall_seconds <= Float64(0.0):
        raise Error("product sampler harness: missing Mojo wall seconds")
    if measurements.mojo_text_stage_seconds <= Float64(0.0):
        raise Error("product sampler harness: missing Mojo text-stage seconds")
    if measurements.mojo_denoise_stage_seconds <= Float64(0.0):
        raise Error("product sampler harness: missing Mojo denoise-stage seconds")
    if measurements.mojo_vae_decode_stage_seconds <= Float64(0.0):
        raise Error("product sampler harness: missing Mojo VAE decode seconds")
    if measurements.mojo_postprocess_save_seconds <= Float64(0.0):
        raise Error("product sampler harness: missing Mojo postprocess/save seconds")
    if measurements.mojo_seconds_per_step <= Float64(0.0):
        raise Error("product sampler harness: missing Mojo seconds/step")
    if measurements.mojo_peak_vram_mib <= 0:
        raise Error("product sampler harness: missing Mojo peak VRAM")


def validate_sampler_speed_parity(
    measurements: SamplerProductMeasurements,
    expected_steps: Int,
    max_seconds_ratio: Float64 = Float64(1.25),
) raises:
    validate_sampler_measurements(measurements, expected_steps)
    if not measurements.speed_parity_accepted:
        raise Error("product sampler harness: measurement scaffold only, speed parity not accepted")
    if sampler_speed_ratio(measurements) > max_seconds_ratio:
        raise Error("product sampler harness: Mojo sampler seconds/step is outside accepted ratio")


def validate_product_sampler_ready(
    run: ProductSamplerRunContract,
    status: SamplerProductStageStatus,
    measurements: SamplerProductMeasurements,
) raises:
    validate_product_sampler_run_contract(run)
    var missing = product_sampler_missing_summary(status)
    if missing != String("none"):
        raise Error(String("product sampler harness: missing ") + missing)
    validate_sampler_speed_parity(measurements, run.plan.diffusion_steps)


def product_sampler_contract_summary(run: ProductSamplerRunContract) -> String:
    return (
        run.plan.sampler_name
        + String(" ")
        + String(run.plan.width)
        + String("x")
        + String(run.plan.height)
        + String(" steps=")
        + String(run.plan.diffusion_steps)
        + String(" cfg=")
        + String(run.plan.cfg_scale)
        + String(" progress=")
        + run.progress_callback_name
        + String(" output=")
        + run.output_callback_name
    )

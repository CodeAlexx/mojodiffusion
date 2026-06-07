# Smoke gate for the fail-loud OneTrainer product sampler harness contract.

from serenitymojo.sampling.product_sampler_harness import (
    SamplerProductMeasurements,
    build_product_sampler_run_contract,
    empty_sampler_product_measurements,
    product_sampler_contract_summary,
    product_sampler_is_ready,
    product_sampler_missing_summary,
    sampler_product_ready_status,
    sampler_product_scaffold_status,
    sampler_speed_ratio,
    validate_product_sampler_ready,
    validate_sampler_measurements,
    validate_sampler_speed_parity,
)
from serenitymojo.training.sample_prompt_config import SamplePrompt


def _check(cond: Bool, msg: String) raises:
    if not cond:
        raise Error(String("product sampler harness smoke FAILED: ") + msg)


def _prompt() -> SamplePrompt:
    return SamplePrompt(
        True,
        String("smoke"),
        String("OneTrainer sampler lifecycle smoke"),
        String(""),
        1024,
        1024,
        1,
        Float32(10.0),
        Float32(24.0),
        30,
        Float32(4.0),
        UInt64(42),
        False,
        String(""),
        False,
        String(""),
        String(""),
        String(""),
        String(""),
    )


def _accepted_measurements(steps: Int) -> SamplerProductMeasurements:
    return SamplerProductMeasurements(
        Float64(3.0),
        18000,
        Float64(64.0),
        Float64(2.0),
        Float64(55.0),
        Float64(5.0),
        Float64(1.0),
        Float64(2.0),
        16000,
        steps,
        steps,
        True,
    )


def main() raises:
    print("==== OneTrainer product sampler harness smoke ====")
    var run = build_product_sampler_run_contract(
        String("FLUX_2"), _prompt(), String("/tmp/product_sampler_harness_smoke.png")
    )
    print(product_sampler_contract_summary(run))
    var flux2_dev_run = build_product_sampler_run_contract(
        String("FLUX_2_DEV"), _prompt(), String("/tmp/product_sampler_harness_flux2_dev_smoke.png")
    )
    _check(flux2_dev_run.plan.sampler_name == String("flux2_dev"), "Flux2 dev sampler plan")
    print(product_sampler_contract_summary(flux2_dev_run))

    var scaffold = sampler_product_scaffold_status()
    _check(not product_sampler_is_ready(scaffold), "scaffold must not be product-ready")
    _check(
        product_sampler_missing_summary(scaffold).find(String("vae_decode")) >= 0,
        "missing summary names VAE decode",
    )
    _check(
        product_sampler_missing_summary(scaffold).find(String("peak_vram_mib")) >= 0,
        "missing summary names VRAM",
    )

    var blocked = False
    try:
        validate_product_sampler_ready(run, scaffold, empty_sampler_product_measurements())
    except e:
        blocked = True
        print("  blocked as expected:", String(e))
    _check(blocked, "scaffold readiness must fail loud")

    var measurement_blocked = False
    try:
        validate_sampler_measurements(empty_sampler_product_measurements(), run.plan.diffusion_steps)
    except e:
        measurement_blocked = True
        print("  empty measurement blocked as expected:", String(e))
    _check(measurement_blocked, "empty measurements must fail loud")

    var measurements = _accepted_measurements(run.plan.diffusion_steps)
    validate_sampler_measurements(measurements, run.plan.diffusion_steps)
    _check(sampler_speed_ratio(measurements) < Float64(1.0), "speed ratio")
    validate_sampler_speed_parity(measurements, run.plan.diffusion_steps)
    validate_product_sampler_ready(run, sampler_product_ready_status(), measurements)

    print("product_sampler_harness_smoke PASS")

# Smoke gate for the OneTrainer sampler intent map.

from serenitymojo.sampling.base_sampler import quantize_resolution
from serenitymojo.sampling.onetrainer_sampler_contract import (
    OT_CFG_GUIDANCE_EMBED,
    OT_CFG_GUIDANCE_EMBED_OR_TEXTBOOK,
    OT_CFG_TEXTBOOK,
    OT_SCHEDULER_DIFFUSERS_CREATE,
    OT_SCHEDULER_FLUX2_MU_CUSTOM_SIGMAS,
    OT_SCHEDULER_MODEL_COPY_CUSTOM_SIGMAS,
    OT_SCHEDULER_MODEL_COPY_MU,
    OT_SCHEDULER_MODEL_COPY_SET_TIMESTEPS,
    OT_SAMPLER_ANIMA,
    OT_SAMPLER_CHROMA,
    OT_SAMPLER_ERNIE,
    OT_SAMPLER_FLUX1_DEV,
    OT_SAMPLER_FLUX2_DEV,
    OT_SAMPLER_FLUX2_KLEIN,
    OT_SAMPLER_QWEN,
    OT_SAMPLER_SD3,
    OT_SAMPLER_SDXL,
    OT_SAMPLER_ZIMAGE,
    OT_TIMESTEP_DIV_1000,
    OT_TIMESTEP_RAW,
    default_ot_sampler_plan,
    ot_sampler_family,
    ot_sampler_plan_with_sample_overrides,
    sampler_contract_summary,
    validate_ot_sampler_plan,
)


def _check(cond: Bool, msg: String) raises:
    if not cond:
        raise Error(String("onetrainer sampler contract smoke FAILED: ") + msg)


def _check_plan(
    model_type: String,
    family: Int,
    steps: Int,
    cfg: Float32,
    quant: Int,
    scheduler_mode: Int,
    timestep_mode: Int,
    cfg_mode: Int,
) raises:
    var plan = default_ot_sampler_plan(model_type)
    validate_ot_sampler_plan(plan)
    _check(plan.family == family, model_type + String(" family"))
    _check(plan.width == 1024 and plan.height == 1024, model_type + String(" size"))
    _check(plan.diffusion_steps == steps, model_type + String(" steps"))
    _check(plan.cfg_scale == cfg, model_type + String(" cfg"))
    _check(plan.quantization == quant, model_type + String(" quant"))
    _check(plan.scheduler_mode == scheduler_mode, model_type + String(" scheduler"))
    _check(plan.timestep_mode == timestep_mode, model_type + String(" timestep"))
    _check(plan.cfg_mode == cfg_mode, model_type + String(" cfg mode"))


def _expect_unsupported(model_type: String) raises:
    var raised = False
    try:
        var plan = default_ot_sampler_plan(model_type)
        validate_ot_sampler_plan(plan)
    except e:
        raised = True
        print("  sampler blocked as expected [", model_type, "]:", String(e))
    if not raised:
        raise Error(String("onetrainer sampler contract smoke expected block for ") + model_type)


def main() raises:
    print("==== OneTrainer sampler contract smoke ====")
    _check(quantize_resolution(1024, 64) == 1024, "plain quantize")
    _check(quantize_resolution(1056, 64) == 1024, "bankers tie down to even")
    _check(quantize_resolution(1120, 64) == 1152, "bankers tie up to even")

    _check_plan(
        String("STABLE_DIFFUSION_XL_10_BASE"),
        OT_SAMPLER_SDXL,
        30,
        Float32(7.5),
        64,
        OT_SCHEDULER_DIFFUSERS_CREATE,
        OT_TIMESTEP_RAW,
        OT_CFG_TEXTBOOK,
    )
    _check_plan(
        String("STABLE_DIFFUSION_35"),
        OT_SAMPLER_SD3,
        28,
        Float32(7.0),
        16,
        OT_SCHEDULER_MODEL_COPY_SET_TIMESTEPS,
        OT_TIMESTEP_RAW,
        OT_CFG_TEXTBOOK,
    )
    _expect_unsupported(String("STABLE_DIFFUSION_3"))
    _check_plan(
        String("QWEN"),
        OT_SAMPLER_QWEN,
        25,
        Float32(3.5),
        64,
        OT_SCHEDULER_MODEL_COPY_MU,
        OT_TIMESTEP_RAW,
        OT_CFG_TEXTBOOK,
    )
    _check_plan(
        String("ERNIE"),
        OT_SAMPLER_ERNIE,
        25,
        Float32(4.0),
        64,
        OT_SCHEDULER_MODEL_COPY_CUSTOM_SIGMAS,
        OT_TIMESTEP_RAW,
        OT_CFG_TEXTBOOK,
    )
    _check_plan(
        String("ANIMA"),
        OT_SAMPLER_ANIMA,
        25,
        Float32(4.0),
        64,
        OT_SCHEDULER_MODEL_COPY_CUSTOM_SIGMAS,
        OT_TIMESTEP_DIV_1000,
        OT_CFG_TEXTBOOK,
    )
    _check_plan(
        String("FLUX_DEV_1"),
        OT_SAMPLER_FLUX1_DEV,
        30,
        Float32(3.5),
        64,
        OT_SCHEDULER_MODEL_COPY_MU,
        OT_TIMESTEP_DIV_1000,
        OT_CFG_GUIDANCE_EMBED,
    )
    _check_plan(
        String("FLUX_2_DEV"),
        OT_SAMPLER_FLUX2_DEV,
        30,
        Float32(4.0),
        64,
        OT_SCHEDULER_FLUX2_MU_CUSTOM_SIGMAS,
        OT_TIMESTEP_DIV_1000,
        OT_CFG_GUIDANCE_EMBED_OR_TEXTBOOK,
    )
    _check_plan(
        String("FLUX_2"),
        OT_SAMPLER_FLUX2_KLEIN,
        30,
        Float32(4.0),
        64,
        OT_SCHEDULER_FLUX2_MU_CUSTOM_SIGMAS,
        OT_TIMESTEP_DIV_1000,
        OT_CFG_GUIDANCE_EMBED_OR_TEXTBOOK,
    )
    _check_plan(
        String("CHROMA_1"),
        OT_SAMPLER_CHROMA,
        30,
        Float32(3.5),
        64,
        OT_SCHEDULER_MODEL_COPY_SET_TIMESTEPS,
        OT_TIMESTEP_RAW,
        OT_CFG_TEXTBOOK,
    )
    _check_plan(
        String("Z_IMAGE"),
        OT_SAMPLER_ZIMAGE,
        28,
        Float32(4.0),
        64,
        OT_SCHEDULER_MODEL_COPY_SET_TIMESTEPS,
        OT_TIMESTEP_RAW,
        OT_CFG_TEXTBOOK,
    )

    var over = ot_sampler_plan_with_sample_overrides(
        String("STABLE_DIFFUSION_35"), 1056, 1120, 14, Float32(5.5)
    )
    _check(over.width == 1056, "SD3.5 q=16 width override")
    _check(over.height == 1120, "SD3.5 q=16 height override")
    _check(over.diffusion_steps == 14, "steps override")
    _check(over.cfg_scale == Float32(5.5), "cfg override")
    _check(ot_sampler_family(String("klein9b")) == OT_SAMPLER_FLUX2_KLEIN, "klein alias")
    _check(ot_sampler_family(String("flux2_dev")) == OT_SAMPLER_FLUX2_DEV, "Flux2 dev alias")

    print(sampler_contract_summary(default_ot_sampler_plan(String("FLUX_2_DEV"))))
    print(sampler_contract_summary(default_ot_sampler_plan(String("FLUX_2"))))
    print("OneTrainer sampler contract smoke PASS")

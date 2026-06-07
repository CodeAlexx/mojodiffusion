# sampling/onetrainer_sampler_contract.mojo - OneTrainer sampler intent map.
#
# This is not a denoiser. It records the OneTrainer sampler contract each model
# runner must satisfy: default sample size/steps/CFG, resolution quantization,
# scheduler setup mode, timestep convention, CFG convention, and staged-device
# behavior. The model-specific sampler files own the actual model/VAE calls.

from serenitymojo.sampling.base_sampler import quantize_resolution


comptime OT_SAMPLER_UNKNOWN = -1
comptime OT_SAMPLER_SDXL = 0
comptime OT_SAMPLER_SD3 = 1
comptime OT_SAMPLER_QWEN = 2
comptime OT_SAMPLER_ERNIE = 3
comptime OT_SAMPLER_ANIMA = 4
comptime OT_SAMPLER_FLUX1_DEV = 5
comptime OT_SAMPLER_FLUX2_KLEIN = 6
comptime OT_SAMPLER_CHROMA = 7
comptime OT_SAMPLER_ZIMAGE = 8
comptime OT_SAMPLER_FLUX2_DEV = 9

comptime OT_SCHEDULER_DIFFUSERS_CREATE = 0
comptime OT_SCHEDULER_MODEL_COPY_SET_TIMESTEPS = 1
comptime OT_SCHEDULER_MODEL_COPY_MU = 2
comptime OT_SCHEDULER_MODEL_COPY_CUSTOM_SIGMAS = 3
comptime OT_SCHEDULER_FLUX2_MU_CUSTOM_SIGMAS = 4

comptime OT_TIMESTEP_RAW = 0
comptime OT_TIMESTEP_DIV_1000 = 1
comptime OT_TIMESTEP_SIGMA_1000 = 2

comptime OT_CFG_TEXTBOOK = 0
comptime OT_CFG_GUIDANCE_EMBED = 1
comptime OT_CFG_GUIDANCE_EMBED_OR_TEXTBOOK = 2


@fieldwise_init
struct OneTrainerSamplerPlan(Copyable, Movable):
    var family: Int
    var model_type: String
    var sampler_name: String
    var width: Int
    var height: Int
    var diffusion_steps: Int
    var cfg_scale: Float32
    var quantization: Int
    var scheduler_mode: Int
    var timestep_mode: Int
    var cfg_mode: Int
    var staged_text_transformer_vae: Bool
    var has_default_negative_prompt: Bool


def ot_sampler_family(model_type: String) -> Int:
    if model_type == String("sdxl") or model_type == String("STABLE_DIFFUSION_XL_10_BASE"):
        return OT_SAMPLER_SDXL
    if (
        model_type == String("sd35")
        or model_type == String("sd3.5")
        or model_type == String("sd3-5")
        or model_type == String("STABLE_DIFFUSION_35")
    ):
        return OT_SAMPLER_SD3
    if model_type == String("qwenimage") or model_type == String("qwen") or model_type == String("QWEN"):
        return OT_SAMPLER_QWEN
    if model_type == String("ernie_image") or model_type == String("ernie") or model_type == String("ERNIE"):
        return OT_SAMPLER_ERNIE
    if model_type == String("anima") or model_type == String("ANIMA"):
        return OT_SAMPLER_ANIMA
    if model_type == String("flux") or model_type == String("flux1") or model_type == String("FLUX_DEV_1"):
        return OT_SAMPLER_FLUX1_DEV
    if model_type == String("flux2_dev") or model_type == String("FLUX_2_DEV"):
        return OT_SAMPLER_FLUX2_DEV
    if (
        model_type == String("klein")
        or model_type == String("klein4b")
        or model_type == String("klein9b")
        or model_type == String("flux2")
        or model_type == String("FLUX_2")
    ):
        return OT_SAMPLER_FLUX2_KLEIN
    if model_type == String("chroma") or model_type == String("CHROMA_1"):
        return OT_SAMPLER_CHROMA
    if model_type == String("zimage") or model_type == String("Z_IMAGE"):
        return OT_SAMPLER_ZIMAGE
    return OT_SAMPLER_UNKNOWN


def ot_sampler_family_name(family: Int) -> String:
    if family == OT_SAMPLER_SDXL:
        return String("sdxl")
    if family == OT_SAMPLER_SD3:
        return String("sd35")
    if family == OT_SAMPLER_QWEN:
        return String("qwen")
    if family == OT_SAMPLER_ERNIE:
        return String("ernie")
    if family == OT_SAMPLER_ANIMA:
        return String("anima")
    if family == OT_SAMPLER_FLUX1_DEV:
        return String("flux1_dev")
    if family == OT_SAMPLER_FLUX2_DEV:
        return String("flux2_dev")
    if family == OT_SAMPLER_FLUX2_KLEIN:
        return String("flux2_klein")
    if family == OT_SAMPLER_CHROMA:
        return String("chroma")
    if family == OT_SAMPLER_ZIMAGE:
        return String("zimage")
    return String("unknown")


def default_ot_sampler_plan(model_type: String) raises -> OneTrainerSamplerPlan:
    var family = ot_sampler_family(model_type)
    if family == OT_SAMPLER_UNKNOWN:
        raise Error(String("OneTrainer sampler contract: unsupported model_type=") + model_type)

    var width = 1024
    var height = 1024
    var steps = 30
    var cfg = Float32(4.0)
    var quant = 64
    var scheduler_mode = OT_SCHEDULER_MODEL_COPY_SET_TIMESTEPS
    var timestep_mode = OT_TIMESTEP_RAW
    var cfg_mode = OT_CFG_TEXTBOOK
    var default_neg = False

    if family == OT_SAMPLER_SDXL:
        steps = 30
        cfg = Float32(7.5)
        quant = 64
        scheduler_mode = OT_SCHEDULER_DIFFUSERS_CREATE
    elif family == OT_SAMPLER_SD3:
        steps = 28
        cfg = Float32(7.0)
        quant = 16
        scheduler_mode = OT_SCHEDULER_MODEL_COPY_SET_TIMESTEPS
    elif family == OT_SAMPLER_QWEN:
        steps = 25
        cfg = Float32(3.5)
        quant = 64
        scheduler_mode = OT_SCHEDULER_MODEL_COPY_MU
    elif family == OT_SAMPLER_ERNIE:
        steps = 25
        cfg = Float32(4.0)
        quant = 64
        scheduler_mode = OT_SCHEDULER_MODEL_COPY_CUSTOM_SIGMAS
    elif family == OT_SAMPLER_ANIMA:
        steps = 25
        cfg = Float32(4.0)
        quant = 64
        scheduler_mode = OT_SCHEDULER_MODEL_COPY_CUSTOM_SIGMAS
        timestep_mode = OT_TIMESTEP_DIV_1000
        default_neg = True
    elif family == OT_SAMPLER_FLUX1_DEV:
        steps = 30
        cfg = Float32(3.5)
        quant = 64
        scheduler_mode = OT_SCHEDULER_MODEL_COPY_MU
        timestep_mode = OT_TIMESTEP_DIV_1000
        cfg_mode = OT_CFG_GUIDANCE_EMBED
    elif family == OT_SAMPLER_FLUX2_KLEIN:
        steps = 30
        cfg = Float32(4.0)
        quant = 64
        scheduler_mode = OT_SCHEDULER_FLUX2_MU_CUSTOM_SIGMAS
        timestep_mode = OT_TIMESTEP_DIV_1000
        cfg_mode = OT_CFG_GUIDANCE_EMBED_OR_TEXTBOOK
    elif family == OT_SAMPLER_FLUX2_DEV:
        steps = 30
        cfg = Float32(4.0)
        quant = 64
        scheduler_mode = OT_SCHEDULER_FLUX2_MU_CUSTOM_SIGMAS
        timestep_mode = OT_TIMESTEP_DIV_1000
        cfg_mode = OT_CFG_GUIDANCE_EMBED_OR_TEXTBOOK
    elif family == OT_SAMPLER_CHROMA:
        steps = 30
        cfg = Float32(3.5)
        quant = 64
        scheduler_mode = OT_SCHEDULER_MODEL_COPY_SET_TIMESTEPS
        default_neg = True
    elif family == OT_SAMPLER_ZIMAGE:
        steps = 28
        cfg = Float32(4.0)
        quant = 64
        scheduler_mode = OT_SCHEDULER_MODEL_COPY_SET_TIMESTEPS

    return OneTrainerSamplerPlan(
        family,
        model_type.copy(),
        ot_sampler_family_name(family),
        quantize_resolution(width, quant),
        quantize_resolution(height, quant),
        steps,
        cfg,
        quant,
        scheduler_mode,
        timestep_mode,
        cfg_mode,
        True,
        default_neg,
    )


def ot_sampler_plan_with_sample_overrides(
    model_type: String,
    width: Int,
    height: Int,
    diffusion_steps: Int,
    cfg_scale: Float32,
) raises -> OneTrainerSamplerPlan:
    var plan = default_ot_sampler_plan(model_type)
    if width > 0:
        plan.width = quantize_resolution(width, plan.quantization)
    if height > 0:
        plan.height = quantize_resolution(height, plan.quantization)
    if diffusion_steps > 0:
        plan.diffusion_steps = diffusion_steps
    if cfg_scale > Float32(0.0):
        plan.cfg_scale = cfg_scale
    return plan^


def validate_ot_sampler_plan(plan: OneTrainerSamplerPlan) raises:
    if plan.width <= 0 or plan.height <= 0:
        raise Error("OneTrainer sampler contract: width/height must be positive")
    if plan.diffusion_steps <= 0:
        raise Error("OneTrainer sampler contract: diffusion_steps must be positive")
    if plan.cfg_scale <= Float32(0.0):
        raise Error("OneTrainer sampler contract: cfg_scale must be positive")
    if plan.width % plan.quantization != 0 or plan.height % plan.quantization != 0:
        raise Error("OneTrainer sampler contract: quantized size is not grid-aligned")
    if not plan.staged_text_transformer_vae:
        raise Error("OneTrainer sampler contract: sampler must stage text/transformer/VAE")


def sampler_contract_summary(plan: OneTrainerSamplerPlan) -> String:
    return (
        plan.sampler_name
        + String(" ")
        + String(plan.width)
        + String("x")
        + String(plan.height)
        + String(" steps=")
        + String(plan.diffusion_steps)
        + String(" cfg=")
        + String(plan.cfg_scale)
        + String(" q=")
        + String(plan.quantization)
    )

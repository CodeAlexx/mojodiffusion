# SDXL sampler helper-contract gate.
#
# This is intentionally bounded to cheap deterministic helper math mirrored from
# Serenity StableDiffusionXLSampler.py, SampleConfig.py, and
# create.create_noise_scheduler. It does not run tokenizers, text encoders, UNet
# inference, random noise, VAE decode, postprocess, image saving, or denoise /
# decode / image / end-to-end sampler parity.

from std.collections import List
from std.time import perf_counter_ns

from serenitymojo.io.json_header import _Cursor, _parse_string, _skip_value

from serenity_trainer.modelSampler.StableDiffusionXLSampler import (
    SDXL_NOISE_SCHEDULER_DPMPP,
    SDXL_NOISE_SCHEDULER_EULER_KARRAS,
    SDXL_NOISE_SCHEDULER_UNIPC,
    StableDiffusionXLSampleConfig,
    StableDiffusionXLSampler,
    stable_diffusion_xl_add_time_ids_values,
    stable_diffusion_xl_cfg_batch_size,
    stable_diffusion_xl_cfg_combine_value,
    stable_diffusion_xl_cfg_rescale_for_force_last_timestep,
    stable_diffusion_xl_cfg_rescale_value,
    stable_diffusion_xl_default_sample_noise_scheduler,
    stable_diffusion_xl_denoise_timestep_contract,
    stable_diffusion_xl_erode_kernel_contract,
    stable_diffusion_xl_inpaint_unet_input_channels,
    stable_diffusion_xl_latent_contract_for_image,
    stable_diffusion_xl_noise_scheduler_algorithm_type,
    stable_diffusion_xl_noise_scheduler_class_name,
    stable_diffusion_xl_noise_scheduler_step_accepts_generator,
    stable_diffusion_xl_noise_scheduler_steps_offset,
    stable_diffusion_xl_noise_scheduler_uses_karras_sigmas,
    stable_diffusion_xl_quantize_resolution,
    stable_diffusion_xl_quantized_latent_contract,
    stable_diffusion_xl_sample_plan,
)
from serenity_trainer.util.config.TrainConfigReader import (
    _read_file_bytes,
    _read_scalar,
)
from serenity_trainer.util.enum.ModelType import (
    MODEL_TYPE_STABLE_DIFFUSION_XL_10_BASE,
    MODEL_TYPE_STABLE_DIFFUSION_XL_10_BASE_INPAINTING,
)


comptime SDXL_HELPER_REF_PATH = "trainer/parity/sdxl_sampler_helper_ref.json"


struct SDXLSamplerHelperRef(Movable):
    var scope: String
    var quantize_1025_64: Int
    var quantize_1056_64: Int
    var quantize_1120_64: Int
    var plan_height: Int
    var plan_width: Int
    var plan_latent_h: Int
    var plan_latent_w: Int
    var plan_latent_channels: Int
    var plan_latent_batch: Int
    var plan_cfg_batch: Int
    var base_unet_input_channels: Int
    var base_unet_input_batch: Int
    var inpaint_unet_input_channels: Int
    var inpaint_unet_input_batch: Int
    var latent_mask_channels: Int
    var latent_conditioning_channels: Int
    var conditioning_image_channels: Int
    var conditioning_image_h: Int
    var conditioning_image_w: Int
    var add_time_ids: List[Int]
    var add_time_ids_rows: Int
    var add_time_ids_cols: Int
    var uses_negative_prompt: Bool
    var pooled_text_embedding_used: Bool
    var scheduler_scales_model_input: Bool
    var extra_step_kwargs_may_include_generator: Bool
    var cfg_combine: Float32
    var cfg_rescale_force_last: Float32
    var cfg_rescale_no_force: Float32
    var base_force_timestep_min: Int
    var base_force_timestep_max: Int
    var inpaint_force_timestep_min: Int
    var inpaint_force_timestep_max: Int
    var inpaint_no_force_timestep_min: Int
    var inpaint_no_force_timestep_max: Int
    var sample_inpainting_drops_first_timestep: Bool
    var erode_kernel_radius: Int
    var erode_kernel_size: Int
    var erode_kernel_weight_count: Int
    var erode_kernel_uniform_weight: Float32
    var decode_formula: String
    var postprocess_output_type: String
    var initial_noise_dtype: String
    var decode_input_dtype: String
    var default_scheduler_class: String
    var default_scheduler_steps_offset: Int
    var default_scheduler_step_accepts_generator: Bool
    var dpmpp_steps_offset: Int
    var dpmpp_algorithm_type: String
    var euler_karras_uses_karras_sigmas: Bool
    var unipc_step_accepts_generator: Bool
    var cfg_rescale_noise_pred_sample: Float32
    var cfg_rescale_std_positive: Float32
    var cfg_rescale_std_pred: Float32
    var cfg_rescale: Float32
    var cfg_rescale_value: Float32
    var diffusion_steps: Int
    var default_noise_scheduler_index: Int

    def __init__(out self):
        self.scope = String()
        self.quantize_1025_64 = 0
        self.quantize_1056_64 = 0
        self.quantize_1120_64 = 0
        self.plan_height = 0
        self.plan_width = 0
        self.plan_latent_h = 0
        self.plan_latent_w = 0
        self.plan_latent_channels = 0
        self.plan_latent_batch = 0
        self.plan_cfg_batch = 0
        self.base_unet_input_channels = 0
        self.base_unet_input_batch = 0
        self.inpaint_unet_input_channels = 0
        self.inpaint_unet_input_batch = 0
        self.latent_mask_channels = 0
        self.latent_conditioning_channels = 0
        self.conditioning_image_channels = 0
        self.conditioning_image_h = 0
        self.conditioning_image_w = 0
        self.add_time_ids = List[Int]()
        self.add_time_ids_rows = 0
        self.add_time_ids_cols = 0
        self.uses_negative_prompt = False
        self.pooled_text_embedding_used = False
        self.scheduler_scales_model_input = False
        self.extra_step_kwargs_may_include_generator = False
        self.cfg_combine = Float32(0.0)
        self.cfg_rescale_force_last = Float32(0.0)
        self.cfg_rescale_no_force = Float32(0.0)
        self.base_force_timestep_min = 0
        self.base_force_timestep_max = 0
        self.inpaint_force_timestep_min = 0
        self.inpaint_force_timestep_max = 0
        self.inpaint_no_force_timestep_min = 0
        self.inpaint_no_force_timestep_max = 0
        self.sample_inpainting_drops_first_timestep = False
        self.erode_kernel_radius = 0
        self.erode_kernel_size = 0
        self.erode_kernel_weight_count = 0
        self.erode_kernel_uniform_weight = Float32(0.0)
        self.decode_formula = String()
        self.postprocess_output_type = String()
        self.initial_noise_dtype = String()
        self.decode_input_dtype = String()
        self.default_scheduler_class = String()
        self.default_scheduler_steps_offset = 0
        self.default_scheduler_step_accepts_generator = False
        self.dpmpp_steps_offset = 0
        self.dpmpp_algorithm_type = String()
        self.euler_karras_uses_karras_sigmas = False
        self.unipc_step_accepts_generator = False
        self.cfg_rescale_noise_pred_sample = Float32(0.0)
        self.cfg_rescale_std_positive = Float32(0.0)
        self.cfg_rescale_std_pred = Float32(0.0)
        self.cfg_rescale = Float32(0.0)
        self.cfg_rescale_value = Float32(0.0)
        self.diffusion_steps = 0
        self.default_noise_scheduler_index = 0


def _read_int_array(mut cur: _Cursor) raises -> List[Int]:
    var out = List[Int]()
    cur.expect(0x5B)
    cur.skip_ws()
    if cur.peek() == 0x5D:
        cur.advance()
        return out^
    while True:
        out.append(Int(_read_scalar(cur).num))
        cur.skip_ws()
        var ch = cur.peek()
        if ch == 0x2C:
            cur.advance()
            continue
        if ch == 0x5D:
            cur.advance()
            break
        raise Error(String("SDXL helper ref JSON: expected ',' or ']' at byte ") + String(cur.pos))
    return out^


def _parse_inputs(mut cur: _Cursor, mut expected: SDXLSamplerHelperRef) raises:
    cur.expect(0x7B)
    cur.skip_ws()
    if cur.peek() == 0x7D:
        cur.advance()
        return
    while True:
        var key = _parse_string(cur)
        cur.expect(0x3A)
        if key == "diffusion_steps":
            expected.diffusion_steps = Int(_read_scalar(cur).num)
        elif key == "default_noise_scheduler_index":
            expected.default_noise_scheduler_index = Int(_read_scalar(cur).num)
        else:
            _skip_value(cur)
        cur.skip_ws()
        var ch = cur.peek()
        if ch == 0x2C:
            cur.advance()
            continue
        if ch == 0x7D:
            cur.advance()
            break
        raise Error(String("SDXL helper ref JSON: bad inputs object at byte ") + String(cur.pos))


def _read_string(mut cur: _Cursor) raises -> String:
    var scalar = _read_scalar(cur)
    return scalar.s


def read_sdxl_sampler_helper_ref(path: String) raises -> SDXLSamplerHelperRef:
    var bytes = _read_file_bytes(path)
    var cur = _Cursor(bytes^)
    var expected = SDXLSamplerHelperRef()
    cur.expect(0x7B)
    cur.skip_ws()
    if cur.peek() == 0x7D:
        cur.advance()
        return expected^
    while True:
        var key = _parse_string(cur)
        cur.expect(0x3A)
        if key == "inputs":
            _parse_inputs(cur, expected)
        elif key == "scope":
            expected.scope = _read_string(cur)
        elif key == "quantize_1025_64":
            expected.quantize_1025_64 = Int(_read_scalar(cur).num)
        elif key == "quantize_1056_64":
            expected.quantize_1056_64 = Int(_read_scalar(cur).num)
        elif key == "quantize_1120_64":
            expected.quantize_1120_64 = Int(_read_scalar(cur).num)
        elif key == "plan_height":
            expected.plan_height = Int(_read_scalar(cur).num)
        elif key == "plan_width":
            expected.plan_width = Int(_read_scalar(cur).num)
        elif key == "plan_latent_h":
            expected.plan_latent_h = Int(_read_scalar(cur).num)
        elif key == "plan_latent_w":
            expected.plan_latent_w = Int(_read_scalar(cur).num)
        elif key == "plan_latent_channels":
            expected.plan_latent_channels = Int(_read_scalar(cur).num)
        elif key == "plan_latent_batch":
            expected.plan_latent_batch = Int(_read_scalar(cur).num)
        elif key == "plan_cfg_batch":
            expected.plan_cfg_batch = Int(_read_scalar(cur).num)
        elif key == "base_unet_input_channels":
            expected.base_unet_input_channels = Int(_read_scalar(cur).num)
        elif key == "base_unet_input_batch":
            expected.base_unet_input_batch = Int(_read_scalar(cur).num)
        elif key == "inpaint_unet_input_channels":
            expected.inpaint_unet_input_channels = Int(_read_scalar(cur).num)
        elif key == "inpaint_unet_input_batch":
            expected.inpaint_unet_input_batch = Int(_read_scalar(cur).num)
        elif key == "latent_mask_channels":
            expected.latent_mask_channels = Int(_read_scalar(cur).num)
        elif key == "latent_conditioning_channels":
            expected.latent_conditioning_channels = Int(_read_scalar(cur).num)
        elif key == "conditioning_image_channels":
            expected.conditioning_image_channels = Int(_read_scalar(cur).num)
        elif key == "conditioning_image_h":
            expected.conditioning_image_h = Int(_read_scalar(cur).num)
        elif key == "conditioning_image_w":
            expected.conditioning_image_w = Int(_read_scalar(cur).num)
        elif key == "add_time_ids":
            expected.add_time_ids = _read_int_array(cur)
        elif key == "add_time_ids_rows":
            expected.add_time_ids_rows = Int(_read_scalar(cur).num)
        elif key == "add_time_ids_cols":
            expected.add_time_ids_cols = Int(_read_scalar(cur).num)
        elif key == "uses_negative_prompt":
            expected.uses_negative_prompt = _read_scalar(cur).num != 0.0
        elif key == "pooled_text_embedding_used":
            expected.pooled_text_embedding_used = _read_scalar(cur).num != 0.0
        elif key == "scheduler_scales_model_input":
            expected.scheduler_scales_model_input = _read_scalar(cur).num != 0.0
        elif key == "extra_step_kwargs_may_include_generator":
            expected.extra_step_kwargs_may_include_generator = _read_scalar(cur).num != 0.0
        elif key == "cfg_combine":
            expected.cfg_combine = Float32(_read_scalar(cur).num)
        elif key == "cfg_rescale_force_last":
            expected.cfg_rescale_force_last = Float32(_read_scalar(cur).num)
        elif key == "cfg_rescale_no_force":
            expected.cfg_rescale_no_force = Float32(_read_scalar(cur).num)
        elif key == "base_force_timestep_min":
            expected.base_force_timestep_min = Int(_read_scalar(cur).num)
        elif key == "base_force_timestep_max":
            expected.base_force_timestep_max = Int(_read_scalar(cur).num)
        elif key == "inpaint_force_timestep_min":
            expected.inpaint_force_timestep_min = Int(_read_scalar(cur).num)
        elif key == "inpaint_force_timestep_max":
            expected.inpaint_force_timestep_max = Int(_read_scalar(cur).num)
        elif key == "inpaint_no_force_timestep_min":
            expected.inpaint_no_force_timestep_min = Int(_read_scalar(cur).num)
        elif key == "inpaint_no_force_timestep_max":
            expected.inpaint_no_force_timestep_max = Int(_read_scalar(cur).num)
        elif key == "sample_inpainting_drops_first_timestep":
            expected.sample_inpainting_drops_first_timestep = _read_scalar(cur).num != 0.0
        elif key == "erode_kernel_radius":
            expected.erode_kernel_radius = Int(_read_scalar(cur).num)
        elif key == "erode_kernel_size":
            expected.erode_kernel_size = Int(_read_scalar(cur).num)
        elif key == "erode_kernel_weight_count":
            expected.erode_kernel_weight_count = Int(_read_scalar(cur).num)
        elif key == "erode_kernel_uniform_weight":
            expected.erode_kernel_uniform_weight = Float32(_read_scalar(cur).num)
        elif key == "decode_formula":
            expected.decode_formula = _read_string(cur)
        elif key == "postprocess_output_type":
            expected.postprocess_output_type = _read_string(cur)
        elif key == "initial_noise_dtype":
            expected.initial_noise_dtype = _read_string(cur)
        elif key == "decode_input_dtype":
            expected.decode_input_dtype = _read_string(cur)
        elif key == "default_scheduler_class":
            expected.default_scheduler_class = _read_string(cur)
        elif key == "default_scheduler_steps_offset":
            expected.default_scheduler_steps_offset = Int(_read_scalar(cur).num)
        elif key == "default_scheduler_step_accepts_generator":
            expected.default_scheduler_step_accepts_generator = _read_scalar(cur).num != 0.0
        elif key == "dpmpp_steps_offset":
            expected.dpmpp_steps_offset = Int(_read_scalar(cur).num)
        elif key == "dpmpp_algorithm_type":
            expected.dpmpp_algorithm_type = _read_string(cur)
        elif key == "euler_karras_uses_karras_sigmas":
            expected.euler_karras_uses_karras_sigmas = _read_scalar(cur).num != 0.0
        elif key == "unipc_step_accepts_generator":
            expected.unipc_step_accepts_generator = _read_scalar(cur).num != 0.0
        elif key == "cfg_rescale_noise_pred_sample":
            expected.cfg_rescale_noise_pred_sample = Float32(_read_scalar(cur).num)
        elif key == "cfg_rescale_std_positive":
            expected.cfg_rescale_std_positive = Float32(_read_scalar(cur).num)
        elif key == "cfg_rescale_std_pred":
            expected.cfg_rescale_std_pred = Float32(_read_scalar(cur).num)
        elif key == "cfg_rescale":
            expected.cfg_rescale = Float32(_read_scalar(cur).num)
        elif key == "cfg_rescale_value":
            expected.cfg_rescale_value = Float32(_read_scalar(cur).num)
        else:
            _skip_value(cur)
        cur.skip_ws()
        var ch = cur.peek()
        if ch == 0x2C:
            cur.advance()
            continue
        if ch == 0x7D:
            cur.advance()
            break
        raise Error(String("SDXL helper ref JSON: expected ',' or '}' at byte ") + String(cur.pos))
    return expected^


def _abs(x: Float32) -> Float32:
    if x < Float32(0.0):
        return -x
    return x


def _expect_int(name: String, got: Int, expected: Int) raises:
    if got != expected:
        raise Error(name + String(": got ") + String(got) + String(", expected ") + String(expected))


def _expect_bool(name: String, got: Bool, expected: Bool) raises:
    if got != expected:
        raise Error(name + String(": unexpected bool"))


def _expect_string(name: String, got: String, expected: String) raises:
    if got != expected:
        raise Error(name + String(": got ") + got + String(", expected ") + expected)


def _expect_close(name: String, got: Float32, expected: Float32, tol: Float32) raises:
    var diff = _abs(got - expected)
    if diff > tol:
        raise Error(
            name + String(": got ") + String(got)
            + String(", expected ") + String(expected)
            + String(", |d| ") + String(diff)
        )


def _expect_int_list(name: String, got: List[Int], expected: List[Int]) raises:
    _expect_int(name + String(" length"), len(got), len(expected))
    for i in range(len(expected)):
        _expect_int(name + String("[") + String(i) + String("]"), got[i], expected[i])


def main() raises:
    var t0 = perf_counter_ns()
    var expected = read_sdxl_sampler_helper_ref(String(SDXL_HELPER_REF_PATH))
    _expect_string(
        "artifact scope",
        expected.scope,
        String("helper/contract only; not denoise/decode/image parity; not end-to-end sampler parity"),
    )

    _expect_int("round down", stable_diffusion_xl_quantize_resolution(1025, 64), expected.quantize_1025_64)
    _expect_int("round half to even down", stable_diffusion_xl_quantize_resolution(1056, 64), expected.quantize_1056_64)
    _expect_int("round half to even up", stable_diffusion_xl_quantize_resolution(1120, 64), expected.quantize_1120_64)

    var contract = stable_diffusion_xl_latent_contract_for_image(
        expected.plan_height,
        expected.plan_width,
        expected.plan_latent_channels,
        expected.plan_latent_batch,
    )
    _expect_int("contract latent batch", contract.latent_batch_size, expected.plan_latent_batch)
    _expect_int("contract image h", contract.image_height, expected.plan_height)
    _expect_int("contract image w", contract.image_width, expected.plan_width)
    _expect_int("contract latent channels", contract.latent_channels, expected.plan_latent_channels)
    _expect_int("contract latent h", contract.latent_height, expected.plan_latent_h)
    _expect_int("contract latent w", contract.latent_width, expected.plan_latent_w)
    _expect_int("contract cfg batch", contract.cfg_batch_size, expected.plan_cfg_batch)
    _expect_int("contract base channels", contract.base_unet_input_channels, expected.base_unet_input_channels)
    _expect_int("contract inpaint channels", contract.inpaint_unet_input_channels, expected.inpaint_unet_input_channels)
    _expect_int("contract mask channels", contract.latent_mask_channels, expected.latent_mask_channels)

    var q_contract = stable_diffusion_xl_quantized_latent_contract(
        1025,
        1120,
        expected.plan_latent_channels,
        expected.plan_latent_batch,
    )
    _expect_int("quantized contract image h", q_contract.image_height, expected.plan_height)
    _expect_int("quantized contract image w", q_contract.image_width, expected.plan_width)
    _expect_int("quantized contract latent h", q_contract.latent_height, expected.plan_latent_h)
    _expect_int("quantized contract latent w", q_contract.latent_width, expected.plan_latent_w)
    _expect_int("quantized contract cfg batch", q_contract.cfg_batch_size, expected.plan_cfg_batch)

    _expect_int("cfg batch", stable_diffusion_xl_cfg_batch_size(1), expected.plan_cfg_batch)
    _expect_int("inpaint input channels", stable_diffusion_xl_inpaint_unet_input_channels(4), expected.inpaint_unet_input_channels)
    _expect_close(
        "cfg combine",
        stable_diffusion_xl_cfg_combine_value(Float32(1.0), Float32(3.0), Float32(4.0)),
        expected.cfg_combine,
        Float32(1e-6),
    )
    _expect_close(
        "cfg rescale force",
        stable_diffusion_xl_cfg_rescale_for_force_last_timestep(True),
        expected.cfg_rescale_force_last,
        Float32(1e-6),
    )
    _expect_close(
        "cfg rescale no force",
        stable_diffusion_xl_cfg_rescale_for_force_last_timestep(False),
        expected.cfg_rescale_no_force,
        Float32(1e-6),
    )
    _expect_close(
        "cfg rescale value",
        stable_diffusion_xl_cfg_rescale_value(
            expected.cfg_rescale_noise_pred_sample,
            expected.cfg_rescale_std_positive,
            expected.cfg_rescale_std_pred,
            expected.cfg_rescale,
        ),
        expected.cfg_rescale_value,
        Float32(1e-5),
    )

    var add_time_ids = stable_diffusion_xl_add_time_ids_values(
        expected.plan_height, expected.plan_width
    )
    _expect_int_list("add_time_ids", add_time_ids, expected.add_time_ids)
    _expect_int("add_time_ids rows", expected.plan_cfg_batch, expected.add_time_ids_rows)
    _expect_int("add_time_ids cols", len(expected.add_time_ids), expected.add_time_ids_cols)

    var base_force_steps = stable_diffusion_xl_denoise_timestep_contract(
        expected.diffusion_steps, False, False, True
    )
    _expect_int("base force timestep min", base_force_steps.timestep_count_min, expected.base_force_timestep_min)
    _expect_int("base force timestep max", base_force_steps.timestep_count_max, expected.base_force_timestep_max)

    var inpaint_force_steps = stable_diffusion_xl_denoise_timestep_contract(
        expected.diffusion_steps, True, True, True
    )
    _expect_bool(
        "inpaint drops first timestep",
        inpaint_force_steps.drops_first_timestep_for_inpaint_composition,
        expected.sample_inpainting_drops_first_timestep,
    )
    _expect_int("inpaint force timestep min", inpaint_force_steps.timestep_count_min, expected.inpaint_force_timestep_min)
    _expect_int("inpaint force timestep max", inpaint_force_steps.timestep_count_max, expected.inpaint_force_timestep_max)

    var inpaint_no_force_steps = stable_diffusion_xl_denoise_timestep_contract(
        expected.diffusion_steps, True, True, False
    )
    _expect_int("inpaint no force timestep min", inpaint_no_force_steps.timestep_count_min, expected.inpaint_no_force_timestep_min)
    _expect_int("inpaint no force timestep max", inpaint_no_force_steps.timestep_count_max, expected.inpaint_no_force_timestep_max)

    var erode = stable_diffusion_xl_erode_kernel_contract()
    _expect_int("erode radius", erode.radius, expected.erode_kernel_radius)
    _expect_int("erode size", erode.size, expected.erode_kernel_size)
    _expect_int("erode weight count", erode.weight_count, expected.erode_kernel_weight_count)
    _expect_close("erode uniform weight", erode.uniform_weight, expected.erode_kernel_uniform_weight, Float32(1e-6))

    _expect_int(
        "default scheduler index",
        stable_diffusion_xl_default_sample_noise_scheduler(),
        expected.default_noise_scheduler_index,
    )
    _expect_string(
        "default scheduler class",
        stable_diffusion_xl_noise_scheduler_class_name(expected.default_noise_scheduler_index),
        expected.default_scheduler_class,
    )
    _expect_int(
        "default scheduler steps_offset",
        stable_diffusion_xl_noise_scheduler_steps_offset(expected.default_noise_scheduler_index),
        expected.default_scheduler_steps_offset,
    )
    _expect_bool(
        "default scheduler generator kwarg",
        stable_diffusion_xl_noise_scheduler_step_accepts_generator(expected.default_noise_scheduler_index),
        expected.default_scheduler_step_accepts_generator,
    )
    _expect_int(
        "dpmpp steps_offset",
        stable_diffusion_xl_noise_scheduler_steps_offset(SDXL_NOISE_SCHEDULER_DPMPP),
        expected.dpmpp_steps_offset,
    )
    _expect_string(
        "dpmpp algorithm",
        stable_diffusion_xl_noise_scheduler_algorithm_type(SDXL_NOISE_SCHEDULER_DPMPP),
        expected.dpmpp_algorithm_type,
    )
    _expect_bool(
        "euler karras flag",
        stable_diffusion_xl_noise_scheduler_uses_karras_sigmas(SDXL_NOISE_SCHEDULER_EULER_KARRAS),
        expected.euler_karras_uses_karras_sigmas,
    )
    _expect_bool(
        "unipc generator kwarg",
        stable_diffusion_xl_noise_scheduler_step_accepts_generator(SDXL_NOISE_SCHEDULER_UNIPC),
        expected.unipc_step_accepts_generator,
    )

    var base_config = StableDiffusionXLSampleConfig(
        String("prompt"),
        String("negative"),
        1025,
        1120,
        123,
        False,
        expected.diffusion_steps,
        Float32(4.0),
        expected.default_noise_scheduler_index,
    )
    var base_plan = stable_diffusion_xl_sample_plan(
        MODEL_TYPE_STABLE_DIFFUSION_XL_10_BASE,
        base_config,
        String("/tmp/sdxl-helper.png"),
    )
    _expect_int("base plan height", base_plan.height, expected.plan_height)
    _expect_int("base plan width", base_plan.width, expected.plan_width)
    _expect_int("base plan latent h", base_plan.latent_h, expected.plan_latent_h)
    _expect_int("base plan latent w", base_plan.latent_w, expected.plan_latent_w)
    _expect_int("base plan unet channels", base_plan.unet_input_channels, expected.base_unet_input_channels)
    _expect_int("base plan batch", base_plan.batch_size, expected.base_unet_input_batch)
    _expect_bool("base negative prompt", base_plan.uses_negative_prompt, expected.uses_negative_prompt)
    _expect_bool("base pooled embed", base_plan.pooled_text_embedding_used, expected.pooled_text_embedding_used)
    _expect_bool("base scheduler scaling", base_plan.scheduler_scales_model_input, expected.scheduler_scales_model_input)
    _expect_bool("base generator kwarg maybe", base_plan.extra_step_kwargs_may_include_generator, expected.extra_step_kwargs_may_include_generator)
    _expect_close("base cfg rescale", base_plan.cfg_rescale, expected.cfg_rescale_no_force, Float32(1e-6))

    var inpaint_config = StableDiffusionXLSampleConfig(
        String("prompt"),
        String("negative"),
        1025,
        1120,
        123,
        False,
        expected.diffusion_steps,
        Float32(4.0),
        expected.default_noise_scheduler_index,
        True,
        String("/tmp/base.png"),
        String("/tmp/mask.png"),
        1,
        2,
        True,
    )
    var inpaint_plan = stable_diffusion_xl_sample_plan(
        MODEL_TYPE_STABLE_DIFFUSION_XL_10_BASE_INPAINTING,
        inpaint_config,
        String("/tmp/sdxl-inpaint-helper.png"),
    )
    _expect_int("inpaint plan channels", inpaint_plan.unet_input_channels, expected.inpaint_unet_input_channels)
    _expect_int("inpaint plan batch", inpaint_plan.batch_size, expected.inpaint_unet_input_batch)
    _expect_int("inpaint plan mask channels", inpaint_plan.latent_mask_channels, expected.latent_mask_channels)
    _expect_int("inpaint plan conditioning latent channels", inpaint_plan.latent_conditioning_channels, expected.latent_conditioning_channels)
    _expect_int("inpaint plan conditioning image channels", inpaint_plan.conditioning_image_channels, expected.conditioning_image_channels)
    _expect_int("inpaint plan conditioning image h", inpaint_plan.conditioning_image_h, expected.conditioning_image_h)
    _expect_int("inpaint plan conditioning image w", inpaint_plan.conditioning_image_w, expected.conditioning_image_w)
    _expect_bool("inpaint prepares conditioning", inpaint_plan.prepares_conditioning_image, True)
    _expect_bool("inpaint erodes mask", inpaint_plan.erodes_mask_before_encoding, True)
    _expect_bool("inpaint appends conditioning", inpaint_plan.appends_mask_and_conditioning_latents, True)
    _expect_int("inpaint timestep min", inpaint_plan.timestep_count_min, expected.inpaint_force_timestep_min)
    _expect_int("inpaint timestep max", inpaint_plan.timestep_count_max, expected.inpaint_force_timestep_max)
    _expect_close("inpaint cfg rescale", inpaint_plan.cfg_rescale, expected.cfg_rescale_force_last, Float32(1e-6))
    _expect_int("inpaint te1 skip", inpaint_plan.text_encoder_1_layer_skip, 1)
    _expect_int("inpaint te2 skip", inpaint_plan.text_encoder_2_layer_skip, 2)
    _expect_string("decode formula", inpaint_plan.decode_formula, expected.decode_formula)
    _expect_string("postprocess output type", inpaint_plan.postprocess_output_type, expected.postprocess_output_type)
    _expect_string("initial noise dtype", inpaint_plan.initial_noise_dtype, expected.initial_noise_dtype)
    _expect_string("decode dtype", inpaint_plan.decode_input_dtype, expected.decode_input_dtype)

    var sampler = StableDiffusionXLSampler(MODEL_TYPE_STABLE_DIFFUSION_XL_10_BASE_INPAINTING)
    var sampler_plan = sampler.sample(inpaint_config, String("/tmp/sdxl-inpaint-helper.png"))
    _expect_int("sampler dispatch plan width", sampler_plan.width, expected.plan_width)
    _expect_int("sampler dispatch plan channels", sampler_plan.unet_input_channels, expected.inpaint_unet_input_channels)

    var t1 = perf_counter_ns()
    var runtime_s = Float64(t1 - t0) / Float64(1000000000)

    print("SDXL SAMPLER HELPER CONTRACT GATE OK")
    print("scope = helper/contract only; no denoise, decode, image, or end-to-end sampler parity")
    print(
        "plan =", base_plan.height, "x", base_plan.width,
        " latent =", base_plan.latent_h, "x", base_plan.latent_w,
        " c =", base_plan.latent_channels,
        " cfg_batch =", base_plan.batch_size,
    )
    print(
        "inpaint channels =", inpaint_plan.unet_input_channels,
        " timesteps =", inpaint_plan.timestep_count_min, "-", inpaint_plan.timestep_count_max,
        " cfg_rescale =", inpaint_plan.cfg_rescale,
    )
    print(
        "scheduler =", expected.default_scheduler_class,
        " dpmpp_offset =", expected.dpmpp_steps_offset,
        " euler_karras =", expected.euler_karras_uses_karras_sigmas,
    )
    print("runtime_s =", runtime_s)

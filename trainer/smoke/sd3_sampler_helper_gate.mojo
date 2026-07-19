# SD3 sampler helper contract gate.
#
# This is intentionally bounded to cheap deterministic helper math mirrored from
# Serenity StableDiffusion3Sampler.py and adjacent SD3 setup/model code. It
# does not run tokenizers, text encoders, transformer inference, random noise,
# VAE decode, postprocess, or image saving, and is not an end-to-end image parity
# claim. The scheduler values are helper-contract checks against the generated
# Serenity/diffusers metadata JSON, not SD3 numeric model parity.

from std.collections import List
from std.time import perf_counter_ns

from serenitymojo.io.json_header import _Cursor, _parse_string, _skip_value

from serenity_trainer.modelSampler.StableDiffusion3Sampler import (
    SD3_SAMPLE_CFG_BATCH_SIZE,
    SD3_SAMPLE_DEFAULT_LATENT_CHANNELS,
    SD3_SAMPLE_FILE_TYPE_IMAGE,
    SD3_SAMPLE_NUM_TRAIN_TIMESTEPS,
    SD3_SAMPLE_RESOLUTION_QUANTIZATION,
    SD3_SAMPLE_VAE_SCALE_FACTOR,
    StableDiffusion3SampleConfig,
    StableDiffusion3SamplerSchedulerConfig,
    stable_diffusion3_always_uses_negative_prompt,
    stable_diffusion3_cfg_batch_size,
    stable_diffusion3_cfg_combine_value,
    stable_diffusion3_euler_update_value,
    stable_diffusion3_flow_shift_sigma,
    stable_diffusion3_latent_contract_for_image,
    stable_diffusion3_make_flow_schedule,
    stable_diffusion3_model_type_is_sd35,
    stable_diffusion3_model_type_name,
    stable_diffusion3_quantize_resolution,
    stable_diffusion3_quantized_latent_contract,
    stable_diffusion3_sample_plan,
)
from serenity_trainer.util.enum.ModelType import (
    MODEL_TYPE_STABLE_DIFFUSION_3,
    MODEL_TYPE_STABLE_DIFFUSION_35,
)
from serenity_trainer.util.config.TrainConfigReader import (
    _read_file_bytes,
    _read_scalar,
)


comptime SD3_HELPER_REF_PATH = "trainer/parity/sd3_sampler_helper_ref.json"


struct SD3SamplerHelperRef(Movable):
    var input_height: Int
    var input_width: Int
    var resolution_quantization: Int
    var vae_scale_factor: Int
    var latent_channels: Int
    var latent_batch_size: Int
    var quantize_1025_16: Int
    var quantize_1032_16: Int
    var quantize_1048_16: Int
    var plan_height: Int
    var plan_width: Int
    var plan_latent_h: Int
    var plan_latent_w: Int
    var plan_latent_channels: Int
    var plan_cfg_batch: Int
    var always_uses_negative_prompt: Bool
    var scales_latents_before_transformer: Bool
    var contract_batch: Int
    var contract_latent_h: Int
    var contract_latent_w: Int
    var contract_cfg_batch: Int
    var cfg_batch_scalar: Int
    var cfg_combine: Float32
    var flow_shift_sigma: Float32
    var diffusion_steps: Int
    var num_train_timesteps: Int
    var scheduler_shift: Float32
    var cfg_scale: Float32
    var euler_sample: Float32
    var euler_model_output: Float32
    var schedule_timesteps_len: Int
    var schedule_sigmas_len: Int
    var schedule_timesteps: List[Float32]
    var schedule_sigmas: List[Float32]
    var euler_update: Float32
    var step_accepts_generator: Bool

    def __init__(out self):
        self.input_height = 0
        self.input_width = 0
        self.resolution_quantization = 0
        self.vae_scale_factor = 0
        self.latent_channels = 0
        self.latent_batch_size = 0
        self.quantize_1025_16 = 0
        self.quantize_1032_16 = 0
        self.quantize_1048_16 = 0
        self.plan_height = 0
        self.plan_width = 0
        self.plan_latent_h = 0
        self.plan_latent_w = 0
        self.plan_latent_channels = 0
        self.plan_cfg_batch = 0
        self.always_uses_negative_prompt = False
        self.scales_latents_before_transformer = False
        self.contract_batch = 0
        self.contract_latent_h = 0
        self.contract_latent_w = 0
        self.contract_cfg_batch = 0
        self.cfg_batch_scalar = 0
        self.cfg_combine = Float32(0.0)
        self.flow_shift_sigma = Float32(0.0)
        self.diffusion_steps = 0
        self.num_train_timesteps = 0
        self.scheduler_shift = Float32(0.0)
        self.cfg_scale = Float32(0.0)
        self.euler_sample = Float32(0.0)
        self.euler_model_output = Float32(0.0)
        self.schedule_timesteps_len = 0
        self.schedule_sigmas_len = 0
        self.schedule_timesteps = List[Float32]()
        self.schedule_sigmas = List[Float32]()
        self.euler_update = Float32(0.0)
        self.step_accepts_generator = False


def _read_float32_array(mut cur: _Cursor) raises -> List[Float32]:
    var out = List[Float32]()
    cur.expect(0x5B)
    cur.skip_ws()
    if cur.peek() == 0x5D:
        cur.advance()
        return out^
    while True:
        out.append(Float32(_read_scalar(cur).num))
        cur.skip_ws()
        var ch = cur.peek()
        if ch == 0x2C:
            cur.advance()
            continue
        if ch == 0x5D:
            cur.advance()
            break
        raise Error(String("SD3 helper ref JSON: expected ',' or ']' at byte ") + String(cur.pos))
    return out^


def _parse_inputs(mut cur: _Cursor, mut expected: SD3SamplerHelperRef) raises:
    cur.expect(0x7B)
    cur.skip_ws()
    if cur.peek() == 0x7D:
        cur.advance()
        return
    while True:
        var key = _parse_string(cur)
        cur.expect(0x3A)
        if key == "height":
            expected.input_height = Int(_read_scalar(cur).num)
        elif key == "width":
            expected.input_width = Int(_read_scalar(cur).num)
        elif key == "resolution_quantization":
            expected.resolution_quantization = Int(_read_scalar(cur).num)
        elif key == "vae_scale_factor":
            expected.vae_scale_factor = Int(_read_scalar(cur).num)
        elif key == "latent_channels":
            expected.latent_channels = Int(_read_scalar(cur).num)
        elif key == "latent_batch_size":
            expected.latent_batch_size = Int(_read_scalar(cur).num)
        elif key == "diffusion_steps":
            expected.diffusion_steps = Int(_read_scalar(cur).num)
        elif key == "num_train_timesteps":
            expected.num_train_timesteps = Int(_read_scalar(cur).num)
        elif key == "scheduler_shift":
            expected.scheduler_shift = Float32(_read_scalar(cur).num)
        elif key == "cfg_scale":
            expected.cfg_scale = Float32(_read_scalar(cur).num)
        elif key == "euler_sample":
            expected.euler_sample = Float32(_read_scalar(cur).num)
        elif key == "euler_model_output":
            expected.euler_model_output = Float32(_read_scalar(cur).num)
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
        raise Error(String("SD3 helper ref JSON: bad inputs object at byte ") + String(cur.pos))


def read_sd3_sampler_helper_ref(path: String) raises -> SD3SamplerHelperRef:
    var bytes = _read_file_bytes(path)
    var cur = _Cursor(bytes^)
    var expected = SD3SamplerHelperRef()
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
        elif key == "quantize_1025_16":
            expected.quantize_1025_16 = Int(_read_scalar(cur).num)
        elif key == "quantize_1032_16":
            expected.quantize_1032_16 = Int(_read_scalar(cur).num)
        elif key == "quantize_1048_16":
            expected.quantize_1048_16 = Int(_read_scalar(cur).num)
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
        elif key == "plan_cfg_batch":
            expected.plan_cfg_batch = Int(_read_scalar(cur).num)
        elif key == "always_uses_negative_prompt":
            expected.always_uses_negative_prompt = _read_scalar(cur).num != 0.0
        elif key == "scales_latents_before_transformer":
            expected.scales_latents_before_transformer = _read_scalar(cur).num != 0.0
        elif key == "contract_batch":
            expected.contract_batch = Int(_read_scalar(cur).num)
        elif key == "contract_latent_h":
            expected.contract_latent_h = Int(_read_scalar(cur).num)
        elif key == "contract_latent_w":
            expected.contract_latent_w = Int(_read_scalar(cur).num)
        elif key == "contract_cfg_batch":
            expected.contract_cfg_batch = Int(_read_scalar(cur).num)
        elif key == "cfg_batch_scalar":
            expected.cfg_batch_scalar = Int(_read_scalar(cur).num)
        elif key == "cfg_combine":
            expected.cfg_combine = Float32(_read_scalar(cur).num)
        elif key == "flow_shift_sigma":
            expected.flow_shift_sigma = Float32(_read_scalar(cur).num)
        elif key == "schedule_timesteps_len":
            expected.schedule_timesteps_len = Int(_read_scalar(cur).num)
        elif key == "schedule_sigmas_len":
            expected.schedule_sigmas_len = Int(_read_scalar(cur).num)
        elif key == "schedule_timesteps":
            expected.schedule_timesteps = _read_float32_array(cur)
        elif key == "schedule_sigmas":
            expected.schedule_sigmas = _read_float32_array(cur)
        elif key == "euler_update":
            expected.euler_update = Float32(_read_scalar(cur).num)
        elif key == "step_accepts_generator":
            expected.step_accepts_generator = _read_scalar(cur).num != 0.0
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
        raise Error(String("SD3 helper ref JSON: expected ',' or '}' at byte ") + String(cur.pos))
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


def main() raises:
    var t0 = perf_counter_ns()
    var expected = read_sd3_sampler_helper_ref(String(SD3_HELPER_REF_PATH))

    _expect_int("ref quantization", expected.resolution_quantization, SD3_SAMPLE_RESOLUTION_QUANTIZATION)
    _expect_int("ref vae scale", expected.vae_scale_factor, SD3_SAMPLE_VAE_SCALE_FACTOR)
    _expect_int("ref latent channels", expected.latent_channels, SD3_SAMPLE_DEFAULT_LATENT_CHANNELS)
    _expect_int("ref latent batch", expected.latent_batch_size, 1)
    _expect_int("sampler cfg batch constant", SD3_SAMPLE_CFG_BATCH_SIZE, 2)
    _expect_int("round down", stable_diffusion3_quantize_resolution(1025, 16), expected.quantize_1025_16)
    _expect_int("round half to even down", stable_diffusion3_quantize_resolution(1032, 16), expected.quantize_1032_16)
    _expect_int("round half to even up", stable_diffusion3_quantize_resolution(1048, 16), expected.quantize_1048_16)

    var sample_config = StableDiffusion3SampleConfig(
        String("prompt"),
        String("negative"),
        expected.input_height,
        expected.input_width,
        123,
        False,
        expected.diffusion_steps,
        expected.cfg_scale,
        0,
        1,
        2,
        3,
        True,
    )
    var plan = stable_diffusion3_sample_plan(
        sample_config, String("/tmp/sd3-helper.png")
    )
    _expect_int("plan file type", plan.file_type, SD3_SAMPLE_FILE_TYPE_IMAGE)
    _expect_string("plan destination", plan.destination, String("/tmp/sd3-helper.png"))
    _expect_int("plan height", plan.height, expected.plan_height)
    _expect_int("plan width", plan.width, expected.plan_width)
    _expect_int("plan latent h", plan.latent_h, expected.plan_latent_h)
    _expect_int("plan latent w", plan.latent_w, expected.plan_latent_w)
    _expect_int("plan latent channels", plan.latent_channels, expected.plan_latent_channels)
    _expect_string("plan latent channel source", plan.latent_channels_source, String("transformer.config.in_channels"))
    _expect_int("plan vae scale", plan.vae_scale_factor, expected.vae_scale_factor)
    _expect_int("plan cfg batch", plan.batch_size, expected.plan_cfg_batch)
    _expect_close("plan cfg scale", plan.cfg_scale, expected.cfg_scale, Float32(1e-6))
    _expect_bool("plan negative prompt", plan.always_uses_negative_prompt, expected.always_uses_negative_prompt)
    _expect_int("plan diffusion steps", plan.diffusion_steps, expected.diffusion_steps)
    _expect_string("plan timestep source", plan.timestep_source, String("noise_scheduler.set_timesteps(diffusion_steps).timesteps"))
    _expect_bool("plan scheduler copy", plan.scheduler_copied_from_model, True)
    _expect_bool("plan generator kwarg", plan.extra_step_kwargs_may_include_generator, expected.step_accepts_generator)
    _expect_string("plan initial noise dtype", plan.initial_noise_dtype, String("F32"))
    _expect_string("plan transformer input dtype", plan.transformer_input_dtype, String("model.train_dtype.torch_dtype()"))
    _expect_string("plan prompt dtype", plan.prompt_embedding_input_dtype, String("model.train_dtype.torch_dtype()"))
    _expect_string("plan pooled prompt dtype", plan.pooled_prompt_embedding_input_dtype, String("model.train_dtype.torch_dtype()"))
    _expect_bool("transformer unscaled latents", plan.scales_latents_before_transformer, expected.scales_latents_before_transformer)
    _expect_string("plan decode formula", plan.decode_formula, String("(latent_image / vae.config.scaling_factor) + vae.config.shift_factor"))
    _expect_int("plan te1 layer skip", plan.text_encoder_1_layer_skip, 1)
    _expect_int("plan te2 layer skip", plan.text_encoder_2_layer_skip, 2)
    _expect_int("plan te3 layer skip", plan.text_encoder_3_layer_skip, 3)
    _expect_bool("plan transformer attention mask", plan.transformer_attention_mask, True)

    var contract = stable_diffusion3_latent_contract_for_image(1024, 512, 16, 3)
    _expect_int("contract batch", contract.batch_size, expected.contract_batch)
    _expect_int("contract image h", contract.image_height, 1024)
    _expect_int("contract image w", contract.image_width, 512)
    _expect_int("contract latent channels", contract.latent_channels, expected.latent_channels)
    _expect_int("contract latent h", contract.latent_height, expected.contract_latent_h)
    _expect_int("contract latent w", contract.latent_width, expected.contract_latent_w)
    _expect_int("contract cfg batch", contract.cfg_batch_size, expected.contract_cfg_batch)

    var q_contract = stable_diffusion3_quantized_latent_contract(expected.input_height, expected.input_width, expected.latent_channels, expected.latent_batch_size)
    _expect_int("quantized contract image h", q_contract.image_height, expected.plan_height)
    _expect_int("quantized contract image w", q_contract.image_width, expected.plan_width)
    _expect_int("quantized contract latent h", q_contract.latent_height, expected.plan_latent_h)
    _expect_int("quantized contract latent w", q_contract.latent_width, expected.plan_latent_w)
    _expect_int("quantized contract cfg batch", q_contract.cfg_batch_size, expected.plan_cfg_batch)

    _expect_bool("always negative prompt", stable_diffusion3_always_uses_negative_prompt(), expected.always_uses_negative_prompt)
    _expect_int("cfg batch scalar", stable_diffusion3_cfg_batch_size(2), expected.cfg_batch_scalar)
    _expect_close(
        "cfg combine",
        stable_diffusion3_cfg_combine_value(Float32(1.0), Float32(3.0), Float32(4.0)),
        expected.cfg_combine,
        Float32(1e-6),
    )

    _expect_string(
        "sd3 model type",
        stable_diffusion3_model_type_name(MODEL_TYPE_STABLE_DIFFUSION_3),
        String("STABLE_DIFFUSION_3"),
    )
    _expect_string(
        "sd35 model type",
        stable_diffusion3_model_type_name(MODEL_TYPE_STABLE_DIFFUSION_35),
        String("STABLE_DIFFUSION_35"),
    )
    _expect_bool(
        "sd3.5 predicate false",
        stable_diffusion3_model_type_is_sd35(MODEL_TYPE_STABLE_DIFFUSION_3),
        False,
    )
    _expect_bool(
        "sd3.5 predicate true",
        stable_diffusion3_model_type_is_sd35(MODEL_TYPE_STABLE_DIFFUSION_35),
        True,
    )

    _expect_close(
        "flow shift sigma",
        stable_diffusion3_flow_shift_sigma(Float32(0.5), expected.scheduler_shift),
        expected.flow_shift_sigma,
        Float32(1e-6),
    )
    var scheduler_config = StableDiffusion3SamplerSchedulerConfig(
        expected.num_train_timesteps, expected.scheduler_shift
    )
    _expect_int("scheduler train timesteps", scheduler_config.num_train_timesteps, SD3_SAMPLE_NUM_TRAIN_TIMESTEPS)
    var schedule = stable_diffusion3_make_flow_schedule(expected.diffusion_steps, scheduler_config)
    _expect_int("schedule timesteps", len(schedule.timesteps), expected.schedule_timesteps_len)
    _expect_int("schedule sigmas", len(schedule.sigmas), expected.schedule_sigmas_len)
    _expect_int("reference timesteps", len(expected.schedule_timesteps), expected.schedule_timesteps_len)
    _expect_int("reference sigmas", len(expected.schedule_sigmas), expected.schedule_sigmas_len)
    for i in range(len(expected.schedule_sigmas)):
        var tol = Float32(2e-5)
        if i == 0 or i == len(expected.schedule_sigmas) - 1:
            tol = Float32(1e-6)
        _expect_close(
            String("sigma[") + String(i) + String("]"),
            schedule.sigmas[i],
            expected.schedule_sigmas[i],
            tol,
        )
    for i in range(len(expected.schedule_timesteps)):
        _expect_close(
            String("timestep[") + String(i) + String("]"),
            schedule.timesteps[i],
            expected.schedule_timesteps[i],
            Float32(2e-2),
        )

    var updated = stable_diffusion3_euler_update_value(
        expected.euler_sample,
        expected.euler_model_output,
        schedule.sigmas[0],
        schedule.sigmas[1],
    )
    _expect_close("euler update", updated, expected.euler_update, Float32(2e-5))

    var t1 = perf_counter_ns()
    var runtime_s = Float64(t1 - t0) / Float64(1000000000)

    print("SD3 SAMPLER HELPER CONTRACT OK")
    print(
        "plan =", plan.height, "x", plan.width,
        " latent =", plan.latent_h, "x", plan.latent_w,
        " c =", plan.latent_channels,
        " cfg_batch =", plan.batch_size,
    )
    print(
        "contract batch =", contract.batch_size,
        " latent =", contract.latent_height, "x", contract.latent_width,
        " cfg_batch =", contract.cfg_batch_size,
    )
    print(
        "schedule shift =", expected.scheduler_shift,
        " sigma1 =", schedule.sigmas[1],
        " sigma2 =", schedule.sigmas[2],
        " timestep1 =", schedule.timesteps[1],
        " euler =", updated,
    )
    print("runtime_s =", runtime_s)

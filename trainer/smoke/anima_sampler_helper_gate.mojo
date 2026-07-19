# Anima sampler helper parity gate.
#
# This is intentionally bounded to deterministic helper math mirrored from
# Serenity-anima-ref AnimaSampler.py and AnimaModel.py. It does not run
# tokenizers, text encoders, transformer inference, random noise, VAE decode,
# postprocess, or image saving, and is not an end-to-end image parity claim.

from std.collections import List
from std.time import perf_counter_ns

from serenitymojo.io.json_header import _Cursor, _parse_string, _skip_value

from serenity_trainer.modelLoader.AnimaModelLoader import MODEL_TYPE_ANIMA
from serenity_trainer.modelSampler.AnimaSampler import (
    ANIMA_SAMPLE_NUM_TRAIN_TIMESTEPS,
    AnimaSampleConfig,
    AnimaSampler,
    AnimaSamplerSchedulerConfig,
    anima_batch_size,
    anima_cfg_combine_value,
    anima_euler_update_value,
    anima_latent_contract_for_image,
    anima_make_schedule,
    anima_quantize_resolution,
    anima_quantized_latent_contract,
    anima_sample_plan,
    anima_transformer_timestep_value,
    anima_use_cfg,
)
from serenity_trainer.util.config.TrainConfigReader import (
    _read_file_bytes,
    _read_scalar,
)


comptime ANIMA_HELPER_REF_PATH = "trainer/parity/anima_sampler_helper_ref.json"


struct AnimaSamplerHelperRef(Movable):
    var quantize_1025_64: Int
    var quantize_1056_64: Int
    var quantize_1120_64: Int
    var plan_height: Int
    var plan_width: Int
    var plan_latent_h: Int
    var plan_latent_w: Int
    var plan_latent_channels: Int
    var plan_latent_frames: Int
    var plan_latent_batch: Int
    var plan_cfg_batch: Int
    var padding_mask_batch: Int
    var padding_mask_channels: Int
    var padding_mask_h: Int
    var padding_mask_w: Int
    var use_cfg_at_1: Bool
    var use_cfg_above_1: Bool
    var uses_negative_prompt_when_cfg: Bool
    var uses_negative_prompt_without_cfg: Bool
    var scales_latents_before_transformer: Bool
    var unscales_latents_before_vae_decode: Bool
    var decoded_frame_index: Int
    var cfg_combine: Float32
    var diffusion_steps: Int
    var num_train_timesteps: Int
    var scheduler_shift: Float32
    var scheduler_use_dynamic_shifting: Bool
    var schedule_input_sigmas_len: Int
    var schedule_timesteps_len: Int
    var schedule_sigmas_len: Int
    var schedule_input_sigmas: List[Float32]
    var schedule_timesteps: List[Float32]
    var schedule_sigmas: List[Float32]
    var model_timestep_values: List[Float32]
    var euler_update: Float32
    var initial_noise_dtype: String

    def __init__(out self):
        self.quantize_1025_64 = 0
        self.quantize_1056_64 = 0
        self.quantize_1120_64 = 0
        self.plan_height = 0
        self.plan_width = 0
        self.plan_latent_h = 0
        self.plan_latent_w = 0
        self.plan_latent_channels = 0
        self.plan_latent_frames = 0
        self.plan_latent_batch = 0
        self.plan_cfg_batch = 0
        self.padding_mask_batch = 0
        self.padding_mask_channels = 0
        self.padding_mask_h = 0
        self.padding_mask_w = 0
        self.use_cfg_at_1 = False
        self.use_cfg_above_1 = False
        self.uses_negative_prompt_when_cfg = False
        self.uses_negative_prompt_without_cfg = False
        self.scales_latents_before_transformer = False
        self.unscales_latents_before_vae_decode = False
        self.decoded_frame_index = 0
        self.cfg_combine = Float32(0.0)
        self.diffusion_steps = 0
        self.num_train_timesteps = 0
        self.scheduler_shift = Float32(0.0)
        self.scheduler_use_dynamic_shifting = False
        self.schedule_input_sigmas_len = 0
        self.schedule_timesteps_len = 0
        self.schedule_sigmas_len = 0
        self.schedule_input_sigmas = List[Float32]()
        self.schedule_timesteps = List[Float32]()
        self.schedule_sigmas = List[Float32]()
        self.model_timestep_values = List[Float32]()
        self.euler_update = Float32(0.0)
        self.initial_noise_dtype = String()


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
        raise Error(String("Anima helper ref JSON: expected ',' or ']' at byte ") + String(cur.pos))
    return out^


def _parse_inputs(mut cur: _Cursor, mut expected: AnimaSamplerHelperRef) raises:
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
        elif key == "num_train_timesteps":
            expected.num_train_timesteps = Int(_read_scalar(cur).num)
        elif key == "scheduler_shift":
            expected.scheduler_shift = Float32(_read_scalar(cur).num)
        elif key == "scheduler_use_dynamic_shifting":
            expected.scheduler_use_dynamic_shifting = _read_scalar(cur).num != 0.0
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
        raise Error(String("Anima helper ref JSON: bad inputs object at byte ") + String(cur.pos))


def read_anima_sampler_helper_ref(path: String) raises -> AnimaSamplerHelperRef:
    var bytes = _read_file_bytes(path)
    var cur = _Cursor(bytes^)
    var expected = AnimaSamplerHelperRef()
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
        elif key == "plan_latent_frames":
            expected.plan_latent_frames = Int(_read_scalar(cur).num)
        elif key == "plan_latent_batch":
            expected.plan_latent_batch = Int(_read_scalar(cur).num)
        elif key == "plan_cfg_batch":
            expected.plan_cfg_batch = Int(_read_scalar(cur).num)
        elif key == "padding_mask_batch":
            expected.padding_mask_batch = Int(_read_scalar(cur).num)
        elif key == "padding_mask_channels":
            expected.padding_mask_channels = Int(_read_scalar(cur).num)
        elif key == "padding_mask_h":
            expected.padding_mask_h = Int(_read_scalar(cur).num)
        elif key == "padding_mask_w":
            expected.padding_mask_w = Int(_read_scalar(cur).num)
        elif key == "use_cfg_at_1":
            expected.use_cfg_at_1 = _read_scalar(cur).num != 0.0
        elif key == "use_cfg_above_1":
            expected.use_cfg_above_1 = _read_scalar(cur).num != 0.0
        elif key == "uses_negative_prompt_when_cfg":
            expected.uses_negative_prompt_when_cfg = _read_scalar(cur).num != 0.0
        elif key == "uses_negative_prompt_without_cfg":
            expected.uses_negative_prompt_without_cfg = _read_scalar(cur).num != 0.0
        elif key == "scales_latents_before_transformer":
            expected.scales_latents_before_transformer = _read_scalar(cur).num != 0.0
        elif key == "unscales_latents_before_vae_decode":
            expected.unscales_latents_before_vae_decode = _read_scalar(cur).num != 0.0
        elif key == "decoded_frame_index":
            expected.decoded_frame_index = Int(_read_scalar(cur).num)
        elif key == "cfg_combine":
            expected.cfg_combine = Float32(_read_scalar(cur).num)
        elif key == "schedule_input_sigmas_len":
            expected.schedule_input_sigmas_len = Int(_read_scalar(cur).num)
        elif key == "schedule_timesteps_len":
            expected.schedule_timesteps_len = Int(_read_scalar(cur).num)
        elif key == "schedule_sigmas_len":
            expected.schedule_sigmas_len = Int(_read_scalar(cur).num)
        elif key == "schedule_input_sigmas":
            expected.schedule_input_sigmas = _read_float32_array(cur)
        elif key == "schedule_timesteps":
            expected.schedule_timesteps = _read_float32_array(cur)
        elif key == "schedule_sigmas":
            expected.schedule_sigmas = _read_float32_array(cur)
        elif key == "model_timestep_values":
            expected.model_timestep_values = _read_float32_array(cur)
        elif key == "euler_update":
            expected.euler_update = Float32(_read_scalar(cur).num)
        elif key == "initial_noise_dtype":
            var scalar = _read_scalar(cur)
            expected.initial_noise_dtype = scalar.s
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
        raise Error(String("Anima helper ref JSON: expected ',' or '}' at byte ") + String(cur.pos))
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
    var expected = read_anima_sampler_helper_ref(String(ANIMA_HELPER_REF_PATH))

    _expect_int("round down", anima_quantize_resolution(1025, 64), expected.quantize_1025_64)
    _expect_int("round half to even down", anima_quantize_resolution(1056, 64), expected.quantize_1056_64)
    _expect_int("round half to even up", anima_quantize_resolution(1120, 64), expected.quantize_1120_64)

    var sample_config = AnimaSampleConfig(
        String("prompt"),
        String("negative"),
        1025,
        1120,
        123,
        False,
        expected.diffusion_steps,
        Float32(4.0),
        0,
    )
    var plan = anima_sample_plan(sample_config, String("/tmp/anima-helper.png"))
    _expect_int("plan height", plan.height, expected.plan_height)
    _expect_int("plan width", plan.width, expected.plan_width)
    _expect_int("plan latent h", plan.latent_h, expected.plan_latent_h)
    _expect_int("plan latent w", plan.latent_w, expected.plan_latent_w)
    _expect_int("plan latent channels", plan.latent_channels, expected.plan_latent_channels)
    _expect_int("plan latent frames", plan.latent_frames, expected.plan_latent_frames)
    _expect_int("plan cfg batch", plan.batch_size, expected.plan_cfg_batch)
    _expect_int("plan padding mask h", plan.padding_mask_h, expected.padding_mask_h)
    _expect_int("plan padding mask w", plan.padding_mask_w, expected.padding_mask_w)
    _expect_int("plan padding mask batch", plan.padding_mask_batch, expected.padding_mask_batch)
    _expect_int("plan padding mask channels", plan.padding_mask_channels, expected.padding_mask_channels)
    _expect_bool("plan negative prompt", plan.uses_negative_prompt, expected.uses_negative_prompt_when_cfg)
    _expect_bool("transformer unscaled latents", plan.scales_latents_before_transformer, expected.scales_latents_before_transformer)
    _expect_bool("vae unscale before decode", plan.unscales_latents_before_vae_decode, expected.unscales_latents_before_vae_decode)
    _expect_int("decoded frame index", plan.decoded_frame_index, expected.decoded_frame_index)
    _expect_string("initial noise dtype", plan.initial_noise_dtype, expected.initial_noise_dtype)

    var no_cfg_config = AnimaSampleConfig(
        String("prompt"),
        String("negative"),
        1025,
        1120,
        123,
        False,
        expected.diffusion_steps,
        Float32(1.0),
        0,
    )
    var no_cfg_plan = anima_sample_plan(no_cfg_config, String("/tmp/anima-helper.png"))
    _expect_bool("no-cfg negative prompt", no_cfg_plan.uses_negative_prompt, expected.uses_negative_prompt_without_cfg)
    _expect_int("no-cfg batch", no_cfg_plan.batch_size, 1)

    var contract = anima_latent_contract_for_image(expected.plan_height, expected.plan_width, Float32(4.0))
    _expect_int("contract latent batch", contract.latent_batch_size, expected.plan_latent_batch)
    _expect_int("contract model input batch", contract.model_input_batch_size, expected.plan_cfg_batch)
    _expect_int("contract text batch", contract.text_batch_size, expected.plan_cfg_batch)
    _expect_int("contract latent h", contract.latent_height, expected.plan_latent_h)
    _expect_int("contract latent w", contract.latent_width, expected.plan_latent_w)
    _expect_int("contract latent frames", contract.latent_frames, expected.plan_latent_frames)
    _expect_int("contract padding mask h", contract.padding_mask_height, expected.padding_mask_h)
    _expect_int("contract padding mask w", contract.padding_mask_width, expected.padding_mask_w)

    var q_contract = anima_quantized_latent_contract(1025, 1120, Float32(4.0))
    _expect_int("quantized contract image h", q_contract.image_height, expected.plan_height)
    _expect_int("quantized contract image w", q_contract.image_width, expected.plan_width)

    _expect_bool("cfg path off at 1", anima_use_cfg(Float32(1.0)), expected.use_cfg_at_1)
    _expect_bool("cfg path on above 1", anima_use_cfg(Float32(1.0001)), expected.use_cfg_above_1)
    _expect_int("cfg batch", anima_batch_size(Float32(4.0)), expected.plan_cfg_batch)
    _expect_close(
        "cfg combine",
        anima_cfg_combine_value(Float32(3.0), Float32(1.0), Float32(4.0)),
        expected.cfg_combine,
        Float32(1e-6),
    )

    var scheduler_config = AnimaSamplerSchedulerConfig(
        expected.num_train_timesteps,
        expected.scheduler_shift,
        expected.scheduler_use_dynamic_shifting,
    )
    _expect_int("scheduler train timesteps", scheduler_config.num_train_timesteps, ANIMA_SAMPLE_NUM_TRAIN_TIMESTEPS)
    var schedule = anima_make_schedule(expected.diffusion_steps, scheduler_config)
    _expect_int("schedule timesteps", len(schedule.timesteps), expected.schedule_timesteps_len)
    _expect_int("schedule sigmas", len(schedule.sigmas), expected.schedule_sigmas_len)
    _expect_int("reference input sigmas", len(expected.schedule_input_sigmas), expected.schedule_input_sigmas_len)
    _expect_int("reference timesteps", len(expected.schedule_timesteps), expected.schedule_timesteps_len)
    _expect_int("reference sigmas", len(expected.schedule_sigmas), expected.schedule_sigmas_len)
    _expect_int("reference model timesteps", len(expected.model_timestep_values), expected.schedule_timesteps_len)

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
        _expect_close(
            String("model timestep[") + String(i) + String("]"),
            anima_transformer_timestep_value(schedule, i, scheduler_config),
            expected.model_timestep_values[i],
            Float32(2e-5),
        )

    var updated = anima_euler_update_value(
        Float32(0.25),
        Float32(0.5),
        schedule.sigmas[0],
        schedule.sigmas[1],
    )
    _expect_close("euler update", updated, expected.euler_update, Float32(2e-5))

    var sampler = AnimaSampler(MODEL_TYPE_ANIMA)
    var sampler_plan = sampler.sample(sample_config, String("/tmp/anima-helper.png"))
    _expect_int("sampler dispatch plan width", sampler_plan.width, expected.plan_width)

    var t1 = perf_counter_ns()
    var runtime_s = Float64(t1 - t0) / Float64(1000000000)

    print("ANIMA SAMPLER HELPER GATE OK")
    print(
        "plan =", plan.height, "x", plan.width,
        " latent =", plan.latent_h, "x", plan.latent_w,
        " c =", plan.latent_channels,
        " frames =", plan.latent_frames,
        " cfg_batch =", plan.batch_size,
    )
    print(
        "schedule shift =", expected.scheduler_shift,
        " sigma1 =", schedule.sigmas[1],
        " timestep1 =", schedule.timesteps[1],
        " model_t1 =", anima_transformer_timestep_value(schedule, 1, scheduler_config),
        " euler =", updated,
    )
    print("runtime_s =", runtime_s)

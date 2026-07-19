# Chroma sampler helper parity gate.
#
# This is intentionally bounded to deterministic helper math mirrored from
# Serenity ChromaSampler.py, ChromaModel.py, and BaseChromaSetup.py. It does
# not run tokenizers, text encoders, transformer inference, random noise,
# scheduler tensor stepping, VAE decode, postprocess, or image saving, and is
# not an end-to-end image parity claim.

from std.collections import List
from std.time import perf_counter_ns

from serenitymojo.io.json_header import _Cursor, _parse_string, _skip_value

from serenity_trainer.modelSampler.ChromaSampler import (
    CHROMA_SAMPLE_NUM_TRAIN_TIMESTEPS,
    ChromaSampleConfig,
    ChromaSampler,
    ChromaSamplerSchedulerConfig,
    chroma_add_noise_discrete_value,
    chroma_attention_mask_contract,
    chroma_cfg_batch_size,
    chroma_cfg_combine_value,
    chroma_decode_input_value,
    chroma_deterministic_timestep_index,
    chroma_euler_update_value,
    chroma_flow_matching_one_minus_sigma,
    chroma_flow_matching_sigma_for_timestep,
    chroma_flow_shift_sigma,
    chroma_flow_target_value,
    chroma_has_cfg_rescale,
    chroma_image_id_last_row_value,
    chroma_image_id_row_from_tile,
    chroma_image_id_tile_x_from_row,
    chroma_image_id_tile_y_from_row,
    chroma_latent_contract_for_image,
    chroma_make_flow_schedule,
    chroma_pack_latent_index,
    chroma_predicted_scaled_latent_value,
    chroma_quantize_resolution,
    chroma_quantized_latent_contract,
    chroma_sample_plan,
    chroma_scale_latent_value,
    chroma_shift_timestep_value,
    chroma_text_mask_contract,
    chroma_training_attention_mask_is_passed,
    chroma_transformer_timestep_value,
    chroma_unpack_latent_index,
)
from serenity_trainer.util.config.TrainConfigReader import (
    _read_file_bytes,
    _read_scalar,
)
from serenity_trainer.util.enum.ModelType import MODEL_TYPE_CHROMA_1


comptime CHROMA_HELPER_REF_PATH = "trainer/parity/chroma_sampler_helper_ref.json"


struct ChromaSamplerHelperRef(Movable):
    var runtime_reference_complete: Bool
    var blocker_count: Int
    var blockers_text: String
    var diffusion_steps: Int
    var num_train_timesteps: Int
    var scheduler_shift: Float32
    var scheduler_use_dynamic_shifting: Bool
    var scheduler_invert_sigmas: Bool
    var scheduler_stochastic_sampling: Bool
    var train_latent_sample: Float32
    var train_noise_sample: Float32
    var train_predicted_flow_sample: Float32
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
    var packed_seq_len: Int
    var packed_channels: Int
    var image_ids_rows: Int
    var image_ids_cols: Int
    var image_ids_last_1: Int
    var image_ids_last_2: Int
    var image_id_sample_row: Int
    var image_id_sample_1: Int
    var image_id_sample_2: Int
    var pack_sample_channel: Int
    var pack_sample_latent_y: Int
    var pack_sample_latent_x: Int
    var pack_sample_sequence_index: Int
    var pack_sample_packed_channel: Int
    var positive_input_tokens: Int
    var negative_input_tokens: Int
    var positive_bool_tokens: Int
    var negative_bool_tokens: Int
    var text_max_seq_length: Int
    var text_pads_to_16_because_lengths_differ: Bool
    var text_ids_rows: Int
    var text_ids_cols: Int
    var sampler_tokenized_prompt_unmasks_one_token: Bool
    var cached_tokens_mask_keeps_exact_mask: Bool
    var attention_mask_rows: Int
    var attention_mask_cols: Int
    var image_attention_mask_all_true: Bool
    var sampler_always_passes_attention_mask: Bool
    var training_passes_attention_mask_when_text_not_all_true: Bool
    var training_omits_attention_mask_when_text_all_true: Bool
    var uses_negative_prompt: Bool
    var has_cfg_rescale: Bool
    var cfg_combine: Float32
    var flow_shift_sigma_0_5: Float32
    var schedule_timesteps_len: Int
    var schedule_sigmas_len: Int
    var schedule_timesteps: List[Float32]
    var schedule_sigmas: List[Float32]
    var model_timestep_1: Float32
    var euler_value: Float32
    var vae_scaling_factor: Float32
    var vae_shift_factor: Float32
    var decode_latent_sample: Float32
    var decode_input_sample: Float32
    var train_scaled_latent_sample: Float32
    var deterministic_timestep_index: Int
    var shifted_timestep_sample: Float32
    var flow_sigma_sample: Float32
    var flow_one_minus_sigma_sample: Float32
    var train_noisy_sample: Float32
    var train_target_sample: Float32
    var train_predicted_scaled_latent_sample: Float32
    var initial_noise_dtype: String
    var postprocess_output_type: String
    var output_file_type: String

    def __init__(out self):
        self.runtime_reference_complete = False
        self.blocker_count = 0
        self.blockers_text = String()
        self.diffusion_steps = 0
        self.num_train_timesteps = 0
        self.scheduler_shift = Float32(0.0)
        self.scheduler_use_dynamic_shifting = False
        self.scheduler_invert_sigmas = False
        self.scheduler_stochastic_sampling = False
        self.train_latent_sample = Float32(0.0)
        self.train_noise_sample = Float32(0.0)
        self.train_predicted_flow_sample = Float32(0.0)
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
        self.packed_seq_len = 0
        self.packed_channels = 0
        self.image_ids_rows = 0
        self.image_ids_cols = 0
        self.image_ids_last_1 = 0
        self.image_ids_last_2 = 0
        self.image_id_sample_row = 0
        self.image_id_sample_1 = 0
        self.image_id_sample_2 = 0
        self.pack_sample_channel = 0
        self.pack_sample_latent_y = 0
        self.pack_sample_latent_x = 0
        self.pack_sample_sequence_index = 0
        self.pack_sample_packed_channel = 0
        self.positive_input_tokens = 0
        self.negative_input_tokens = 0
        self.positive_bool_tokens = 0
        self.negative_bool_tokens = 0
        self.text_max_seq_length = 0
        self.text_pads_to_16_because_lengths_differ = False
        self.text_ids_rows = 0
        self.text_ids_cols = 0
        self.sampler_tokenized_prompt_unmasks_one_token = False
        self.cached_tokens_mask_keeps_exact_mask = False
        self.attention_mask_rows = 0
        self.attention_mask_cols = 0
        self.image_attention_mask_all_true = False
        self.sampler_always_passes_attention_mask = False
        self.training_passes_attention_mask_when_text_not_all_true = False
        self.training_omits_attention_mask_when_text_all_true = False
        self.uses_negative_prompt = False
        self.has_cfg_rescale = False
        self.cfg_combine = Float32(0.0)
        self.flow_shift_sigma_0_5 = Float32(0.0)
        self.schedule_timesteps_len = 0
        self.schedule_sigmas_len = 0
        self.schedule_timesteps = List[Float32]()
        self.schedule_sigmas = List[Float32]()
        self.model_timestep_1 = Float32(0.0)
        self.euler_value = Float32(0.0)
        self.vae_scaling_factor = Float32(0.0)
        self.vae_shift_factor = Float32(0.0)
        self.decode_latent_sample = Float32(0.0)
        self.decode_input_sample = Float32(0.0)
        self.train_scaled_latent_sample = Float32(0.0)
        self.deterministic_timestep_index = 0
        self.shifted_timestep_sample = Float32(0.0)
        self.flow_sigma_sample = Float32(0.0)
        self.flow_one_minus_sigma_sample = Float32(0.0)
        self.train_noisy_sample = Float32(0.0)
        self.train_target_sample = Float32(0.0)
        self.train_predicted_scaled_latent_sample = Float32(0.0)
        self.initial_noise_dtype = String()
        self.postprocess_output_type = String()
        self.output_file_type = String()


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
        raise Error(String("Chroma helper ref JSON: expected ',' or ']' at byte ") + String(cur.pos))
    return out^


def _parse_inputs(mut cur: _Cursor, mut expected: ChromaSamplerHelperRef) raises:
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
        elif key == "scheduler_invert_sigmas":
            expected.scheduler_invert_sigmas = _read_scalar(cur).num != 0.0
        elif key == "scheduler_stochastic_sampling":
            expected.scheduler_stochastic_sampling = _read_scalar(cur).num != 0.0
        elif key == "train_latent_sample":
            expected.train_latent_sample = Float32(_read_scalar(cur).num)
        elif key == "train_noise_sample":
            expected.train_noise_sample = Float32(_read_scalar(cur).num)
        elif key == "train_predicted_flow_sample":
            expected.train_predicted_flow_sample = Float32(_read_scalar(cur).num)
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
        raise Error(String("Chroma helper ref JSON: bad inputs object at byte ") + String(cur.pos))


def read_chroma_sampler_helper_ref(path: String) raises -> ChromaSamplerHelperRef:
    var bytes = _read_file_bytes(path)
    var cur = _Cursor(bytes^)
    var expected = ChromaSamplerHelperRef()
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
        elif key == "runtime_reference_complete":
            expected.runtime_reference_complete = _read_scalar(cur).num != 0.0
        elif key == "blocker_count":
            expected.blocker_count = Int(_read_scalar(cur).num)
        elif key == "blockers_text":
            expected.blockers_text = _read_scalar(cur).s
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
        elif key == "packed_seq_len":
            expected.packed_seq_len = Int(_read_scalar(cur).num)
        elif key == "packed_channels":
            expected.packed_channels = Int(_read_scalar(cur).num)
        elif key == "image_ids_rows":
            expected.image_ids_rows = Int(_read_scalar(cur).num)
        elif key == "image_ids_cols":
            expected.image_ids_cols = Int(_read_scalar(cur).num)
        elif key == "image_ids_last_1":
            expected.image_ids_last_1 = Int(_read_scalar(cur).num)
        elif key == "image_ids_last_2":
            expected.image_ids_last_2 = Int(_read_scalar(cur).num)
        elif key == "image_id_sample_row":
            expected.image_id_sample_row = Int(_read_scalar(cur).num)
        elif key == "image_id_sample_1":
            expected.image_id_sample_1 = Int(_read_scalar(cur).num)
        elif key == "image_id_sample_2":
            expected.image_id_sample_2 = Int(_read_scalar(cur).num)
        elif key == "pack_sample_channel":
            expected.pack_sample_channel = Int(_read_scalar(cur).num)
        elif key == "pack_sample_latent_y":
            expected.pack_sample_latent_y = Int(_read_scalar(cur).num)
        elif key == "pack_sample_latent_x":
            expected.pack_sample_latent_x = Int(_read_scalar(cur).num)
        elif key == "pack_sample_sequence_index":
            expected.pack_sample_sequence_index = Int(_read_scalar(cur).num)
        elif key == "pack_sample_packed_channel":
            expected.pack_sample_packed_channel = Int(_read_scalar(cur).num)
        elif key == "positive_input_tokens":
            expected.positive_input_tokens = Int(_read_scalar(cur).num)
        elif key == "negative_input_tokens":
            expected.negative_input_tokens = Int(_read_scalar(cur).num)
        elif key == "positive_bool_tokens":
            expected.positive_bool_tokens = Int(_read_scalar(cur).num)
        elif key == "negative_bool_tokens":
            expected.negative_bool_tokens = Int(_read_scalar(cur).num)
        elif key == "text_max_seq_length":
            expected.text_max_seq_length = Int(_read_scalar(cur).num)
        elif key == "text_pads_to_16_because_lengths_differ":
            expected.text_pads_to_16_because_lengths_differ = _read_scalar(cur).num != 0.0
        elif key == "text_ids_rows":
            expected.text_ids_rows = Int(_read_scalar(cur).num)
        elif key == "text_ids_cols":
            expected.text_ids_cols = Int(_read_scalar(cur).num)
        elif key == "sampler_tokenized_prompt_unmasks_one_token":
            expected.sampler_tokenized_prompt_unmasks_one_token = _read_scalar(cur).num != 0.0
        elif key == "cached_tokens_mask_keeps_exact_mask":
            expected.cached_tokens_mask_keeps_exact_mask = _read_scalar(cur).num != 0.0
        elif key == "attention_mask_rows":
            expected.attention_mask_rows = Int(_read_scalar(cur).num)
        elif key == "attention_mask_cols":
            expected.attention_mask_cols = Int(_read_scalar(cur).num)
        elif key == "image_attention_mask_all_true":
            expected.image_attention_mask_all_true = _read_scalar(cur).num != 0.0
        elif key == "sampler_always_passes_attention_mask":
            expected.sampler_always_passes_attention_mask = _read_scalar(cur).num != 0.0
        elif key == "training_passes_attention_mask_when_text_not_all_true":
            expected.training_passes_attention_mask_when_text_not_all_true = _read_scalar(cur).num != 0.0
        elif key == "training_omits_attention_mask_when_text_all_true":
            expected.training_omits_attention_mask_when_text_all_true = _read_scalar(cur).num != 0.0
        elif key == "uses_negative_prompt":
            expected.uses_negative_prompt = _read_scalar(cur).num != 0.0
        elif key == "has_cfg_rescale":
            expected.has_cfg_rescale = _read_scalar(cur).num != 0.0
        elif key == "cfg_combine":
            expected.cfg_combine = Float32(_read_scalar(cur).num)
        elif key == "scheduler_num_train_timesteps":
            expected.num_train_timesteps = Int(_read_scalar(cur).num)
        elif key == "scheduler_shift":
            expected.scheduler_shift = Float32(_read_scalar(cur).num)
        elif key == "scheduler_use_dynamic_shifting":
            expected.scheduler_use_dynamic_shifting = _read_scalar(cur).num != 0.0
        elif key == "scheduler_invert_sigmas":
            expected.scheduler_invert_sigmas = _read_scalar(cur).num != 0.0
        elif key == "scheduler_stochastic_sampling":
            expected.scheduler_stochastic_sampling = _read_scalar(cur).num != 0.0
        elif key == "flow_shift_sigma_0_5":
            expected.flow_shift_sigma_0_5 = Float32(_read_scalar(cur).num)
        elif key == "schedule_timesteps_len":
            expected.schedule_timesteps_len = Int(_read_scalar(cur).num)
        elif key == "schedule_sigmas_len":
            expected.schedule_sigmas_len = Int(_read_scalar(cur).num)
        elif key == "schedule_timesteps":
            expected.schedule_timesteps = _read_float32_array(cur)
        elif key == "schedule_sigmas":
            expected.schedule_sigmas = _read_float32_array(cur)
        elif key == "model_timestep_1":
            expected.model_timestep_1 = Float32(_read_scalar(cur).num)
        elif key == "euler_value":
            expected.euler_value = Float32(_read_scalar(cur).num)
        elif key == "vae_scaling_factor":
            expected.vae_scaling_factor = Float32(_read_scalar(cur).num)
        elif key == "vae_shift_factor":
            expected.vae_shift_factor = Float32(_read_scalar(cur).num)
        elif key == "decode_latent_sample":
            expected.decode_latent_sample = Float32(_read_scalar(cur).num)
        elif key == "decode_input_sample":
            expected.decode_input_sample = Float32(_read_scalar(cur).num)
        elif key == "train_scaled_latent_sample":
            expected.train_scaled_latent_sample = Float32(_read_scalar(cur).num)
        elif key == "deterministic_timestep_index":
            expected.deterministic_timestep_index = Int(_read_scalar(cur).num)
        elif key == "shifted_timestep_sample":
            expected.shifted_timestep_sample = Float32(_read_scalar(cur).num)
        elif key == "flow_sigma_sample":
            expected.flow_sigma_sample = Float32(_read_scalar(cur).num)
        elif key == "flow_one_minus_sigma_sample":
            expected.flow_one_minus_sigma_sample = Float32(_read_scalar(cur).num)
        elif key == "train_noisy_sample":
            expected.train_noisy_sample = Float32(_read_scalar(cur).num)
        elif key == "train_target_sample":
            expected.train_target_sample = Float32(_read_scalar(cur).num)
        elif key == "train_predicted_scaled_latent_sample":
            expected.train_predicted_scaled_latent_sample = Float32(_read_scalar(cur).num)
        elif key == "initial_noise_dtype":
            expected.initial_noise_dtype = _read_scalar(cur).s
        elif key == "postprocess_output_type":
            expected.postprocess_output_type = _read_scalar(cur).s
        elif key == "output_file_type":
            expected.output_file_type = _read_scalar(cur).s
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
        raise Error(String("Chroma helper ref JSON: expected ',' or '}' at byte ") + String(cur.pos))
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
    var expected = read_chroma_sampler_helper_ref(String(CHROMA_HELPER_REF_PATH))
    if expected.blocker_count > 0:
        print("CHROMA SAMPLER HELPER REFERENCE BLOCKERS:", expected.blockers_text)

    _expect_int("round down", chroma_quantize_resolution(1025, 64), expected.quantize_1025_64)
    _expect_int("round half to even down", chroma_quantize_resolution(1056, 64), expected.quantize_1056_64)
    _expect_int("round half to even up", chroma_quantize_resolution(1120, 64), expected.quantize_1120_64)

    var sample_config = ChromaSampleConfig(
        String("prompt"),
        String("negative"),
        1025,
        1120,
        123,
        False,
        expected.diffusion_steps,
        Float32(3.5),
        0,
    )
    var plan = chroma_sample_plan(sample_config, String("/tmp/chroma-helper.png"))
    _expect_int("plan height", plan.height, expected.plan_height)
    _expect_int("plan width", plan.width, expected.plan_width)
    _expect_int("plan latent h", plan.latent_h, expected.plan_latent_h)
    _expect_int("plan latent w", plan.latent_w, expected.plan_latent_w)
    _expect_int("plan latent channels", plan.latent_channels, expected.plan_latent_channels)
    _expect_int("plan packed seq", plan.packed_seq_len, expected.packed_seq_len)
    _expect_int("plan packed channels", plan.packed_channels, expected.packed_channels)
    _expect_int("plan cfg batch", plan.batch_size, expected.plan_cfg_batch)
    _expect_bool("plan negative prompt", plan.always_uses_negative_prompt, expected.uses_negative_prompt)
    _expect_string("initial noise dtype", plan.initial_noise_dtype, String("F32"))
    _expect_string("postprocess output type", plan.postprocess_output_type, expected.postprocess_output_type)
    _expect_string("output file type", plan.output_file_type, expected.output_file_type)

    var contract = chroma_latent_contract_for_image(
        expected.plan_height,
        expected.plan_width,
        expected.plan_latent_batch,
    )
    _expect_int("contract latent batch", contract.latent_batch_size, expected.plan_latent_batch)
    _expect_int("contract model input batch", contract.model_input_batch_size, expected.plan_cfg_batch)
    _expect_int("contract latent h", contract.latent_height, expected.plan_latent_h)
    _expect_int("contract latent w", contract.latent_width, expected.plan_latent_w)
    _expect_int("contract packed seq", contract.packed_seq_len, expected.packed_seq_len)
    _expect_int("contract packed channels", contract.packed_channels, expected.packed_channels)
    _expect_int("contract image ids rows", contract.image_ids_rows, expected.image_ids_rows)
    _expect_int("contract image ids cols", contract.image_ids_cols, expected.image_ids_cols)

    var q_contract = chroma_quantized_latent_contract(1025, 1120, expected.plan_latent_batch)
    _expect_int("quantized contract image h", q_contract.image_height, expected.plan_height)
    _expect_int("quantized contract image w", q_contract.image_width, expected.plan_width)

    _expect_int("cfg batch scalar", chroma_cfg_batch_size(1), expected.plan_cfg_batch)
    _expect_close(
        "cfg combine",
        chroma_cfg_combine_value(Float32(3.0), Float32(1.0), Float32(3.5)),
        expected.cfg_combine,
        Float32(1e-6),
    )
    _expect_bool("cfg rescale absent", chroma_has_cfg_rescale(), expected.has_cfg_rescale)

    var text_contract = chroma_text_mask_contract(
        expected.positive_input_tokens,
        expected.negative_input_tokens,
    )
    _expect_int("positive bool tokens", text_contract.positive_bool_tokens, expected.positive_bool_tokens)
    _expect_int("negative bool tokens", text_contract.negative_bool_tokens, expected.negative_bool_tokens)
    _expect_int("text max seq", text_contract.max_seq_length, expected.text_max_seq_length)
    _expect_bool("text pads to 16", text_contract.pads_to_16_because_lengths_differ, expected.text_pads_to_16_because_lengths_differ)
    _expect_int("text ids rows", text_contract.text_ids_rows, expected.text_ids_rows)
    _expect_int("text ids cols", text_contract.text_ids_cols, expected.text_ids_cols)
    _expect_bool("sampler tokenized unmask", expected.sampler_tokenized_prompt_unmasks_one_token, True)
    _expect_bool("cached tokens exact mask", expected.cached_tokens_mask_keeps_exact_mask, True)

    var attention = chroma_attention_mask_contract(expected.text_max_seq_length, expected.packed_seq_len)
    _expect_int("attention rows", attention.attention_mask_rows, expected.attention_mask_rows)
    _expect_int("attention cols", attention.attention_mask_cols, expected.attention_mask_cols)
    _expect_bool("image attention all true", attention.image_attention_mask_all_true, expected.image_attention_mask_all_true)
    _expect_bool("sampler passes attention", attention.sampler_always_passes_attention_mask, expected.sampler_always_passes_attention_mask)
    _expect_bool(
        "training passes mask when text not all true",
        chroma_training_attention_mask_is_passed(False),
        expected.training_passes_attention_mask_when_text_not_all_true,
    )
    _expect_bool(
        "training omits mask when text all true",
        not chroma_training_attention_mask_is_passed(True),
        expected.training_omits_attention_mask_when_text_all_true,
    )

    _expect_int("image id last y", chroma_image_id_last_row_value(expected.plan_latent_h), expected.image_ids_last_1)
    _expect_int("image id last x", chroma_image_id_last_row_value(expected.plan_latent_w), expected.image_ids_last_2)
    var image_row = chroma_image_id_row_from_tile(
        expected.image_id_sample_1,
        expected.image_id_sample_2,
        expected.plan_latent_w,
    )
    _expect_int("image id sample row", image_row, expected.image_id_sample_row)
    _expect_int("image id sample y", chroma_image_id_tile_y_from_row(image_row, expected.plan_latent_w), expected.image_id_sample_1)
    _expect_int("image id sample x", chroma_image_id_tile_x_from_row(image_row, expected.plan_latent_w), expected.image_id_sample_2)

    var packed_index = chroma_pack_latent_index(
        expected.pack_sample_channel,
        expected.pack_sample_latent_y,
        expected.pack_sample_latent_x,
        expected.plan_latent_h,
        expected.plan_latent_w,
    )
    _expect_int("pack sequence index", packed_index.sequence_index, expected.pack_sample_sequence_index)
    _expect_int("pack channel index", packed_index.packed_channel, expected.pack_sample_packed_channel)
    var unpacked_index = chroma_unpack_latent_index(
        packed_index.sequence_index,
        packed_index.packed_channel,
        expected.plan_latent_h,
        expected.plan_latent_w,
    )
    _expect_int("unpack channel", unpacked_index.channel, expected.pack_sample_channel)
    _expect_int("unpack latent y", unpacked_index.latent_y, expected.pack_sample_latent_y)
    _expect_int("unpack latent x", unpacked_index.latent_x, expected.pack_sample_latent_x)

    _expect_int("scheduler train timesteps", expected.num_train_timesteps, CHROMA_SAMPLE_NUM_TRAIN_TIMESTEPS)
    var scheduler_config = ChromaSamplerSchedulerConfig(
        expected.num_train_timesteps,
        expected.scheduler_shift,
        expected.scheduler_use_dynamic_shifting,
        expected.scheduler_invert_sigmas,
        expected.scheduler_stochastic_sampling,
    )
    _expect_close(
        "flow shift sigma",
        chroma_flow_shift_sigma(Float32(0.5), expected.scheduler_shift),
        expected.flow_shift_sigma_0_5,
        Float32(1e-6),
    )
    var schedule = chroma_make_flow_schedule(expected.diffusion_steps, scheduler_config)
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

    _expect_close(
        "model timestep[1]",
        chroma_transformer_timestep_value(schedule.timesteps[1]),
        expected.model_timestep_1,
        Float32(2e-5),
    )
    var updated = chroma_euler_update_value(
        Float32(0.25),
        Float32(0.5),
        schedule.sigmas[0],
        schedule.sigmas[1],
    )
    _expect_close("euler update", updated, expected.euler_value, Float32(2e-5))

    _expect_close(
        "decode input",
        chroma_decode_input_value(
            expected.decode_latent_sample,
            expected.vae_scaling_factor,
            expected.vae_shift_factor,
        ),
        expected.decode_input_sample,
        Float32(2e-5),
    )
    var scaled = chroma_scale_latent_value(
        expected.train_latent_sample,
        expected.vae_shift_factor,
        expected.vae_scaling_factor,
    )
    _expect_close("train scaled latent", scaled, expected.train_scaled_latent_sample, Float32(2e-5))
    _expect_int(
        "deterministic timestep",
        chroma_deterministic_timestep_index(expected.num_train_timesteps),
        expected.deterministic_timestep_index,
    )
    _expect_close(
        "shifted timestep",
        chroma_shift_timestep_value(Float32(500.0), expected.num_train_timesteps, expected.scheduler_shift),
        expected.shifted_timestep_sample,
        Float32(2e-5),
    )
    var flow_sigma = chroma_flow_matching_sigma_for_timestep(
        expected.deterministic_timestep_index,
        expected.num_train_timesteps,
    )
    _expect_close("flow sigma", flow_sigma, expected.flow_sigma_sample, Float32(1e-6))
    _expect_close(
        "flow one minus sigma",
        chroma_flow_matching_one_minus_sigma(expected.deterministic_timestep_index, expected.num_train_timesteps),
        expected.flow_one_minus_sigma_sample,
        Float32(1e-6),
    )
    var noisy = chroma_add_noise_discrete_value(
        scaled,
        expected.train_noise_sample,
        expected.deterministic_timestep_index,
        expected.num_train_timesteps,
    )
    _expect_close("train noisy sample", noisy, expected.train_noisy_sample, Float32(2e-5))
    _expect_close(
        "train target sample",
        chroma_flow_target_value(expected.train_noise_sample, scaled),
        expected.train_target_sample,
        Float32(2e-5),
    )
    _expect_close(
        "predicted scaled latent sample",
        chroma_predicted_scaled_latent_value(
            noisy,
            expected.train_predicted_flow_sample,
            flow_sigma,
        ),
        expected.train_predicted_scaled_latent_sample,
        Float32(2e-5),
    )

    var sampler = ChromaSampler(MODEL_TYPE_CHROMA_1)
    var sampler_plan = sampler.sample(sample_config, String("/tmp/chroma-helper.png"))
    _expect_int("sampler dispatch plan width", sampler_plan.width, expected.plan_width)

    var t1 = perf_counter_ns()
    var runtime_s = Float64(t1 - t0) / Float64(1000000000)

    print("CHROMA SAMPLER HELPER GATE OK")
    print(
        "plan =", plan.height, "x", plan.width,
        " latent =", plan.latent_h, "x", plan.latent_w,
        " packed =", plan.packed_seq_len, "x", plan.packed_channels,
        " cfg_batch =", plan.batch_size,
    )
    print(
        "schedule shift =", expected.scheduler_shift,
        " sigma1 =", schedule.sigmas[1],
        " timestep1 =", schedule.timesteps[1],
        " model_t1 =", chroma_transformer_timestep_value(schedule.timesteps[1]),
        " euler =", updated,
    )
    print("runtime_s =", runtime_s)

# BaseQwenSetup.mojo — build-only Qwen setup surface.
#
# Source of truth: /home/alex/Serenity/modules/modelSetup/BaseQwenSetup.py
# Related Serenity model helpers:
#   /home/alex/Serenity/modules/model/QwenModel.py
#
# This file intentionally carries method-surface metadata and scalar scheduler
# helpers only. QwenModel.mojo is still a TODO stub, so predict() runtime math is
# not implemented here and no numeric parity is claimed.

from std.math import exp

from serenity_trainer.modelSetup.mixin.ModelSetupNoiseMixin import (
    _get_timestep_discrete_host as _ts_host,
)


comptime QWEN_NUM_TRAIN_TIMESTEPS = 1000
comptime QWEN_PROMPT_MAX_LENGTH = 512
comptime QWEN_PROMPT_TEMPLATE_CROP_START = 34
comptime QWEN_PROMPT_TOKENIZER_EFFECTIVE_MAX_LENGTH = QWEN_PROMPT_MAX_LENGTH + QWEN_PROMPT_TEMPLATE_CROP_START
comptime QWEN_LATENT_PACK_FACTOR = 2
comptime QWEN_IMAGE_BASIS_BATCH = 1
comptime QWEN_IMAGE_BASIS_FRAME = 1
comptime QWEN_TEXT_ENCODER_HIDDEN_STATE_OUTPUT_INDEX = -1

comptime QWEN_LAYER_PRESET_ATTN_MLP = "attn-mlp"
comptime QWEN_LAYER_PRESET_ATTN_ONLY = "attn-only"
comptime QWEN_LAYER_PRESET_BLOCKS = "blocks"
comptime QWEN_LAYER_PRESET_FULL = "full"

comptime QWEN_LOSS_TYPE_TARGET = "target"
comptime QWEN_PREDICT_KEY_LOSS_TYPE = "loss_type"
comptime QWEN_PREDICT_KEY_TIMESTEP = "timestep"
comptime QWEN_PREDICT_KEY_PREDICTED = "predicted"
comptime QWEN_PREDICT_KEY_TARGET = "target"


def qwen_layer_preset_filters(preset: String) raises -> List[String]:
    """Serenity BaseQwenSetup.LAYER_PRESETS."""
    var filters = List[String]()
    if preset == QWEN_LAYER_PRESET_ATTN_MLP:
        filters.append("attn")
        filters.append("img_mlp")
        filters.append("txt_mlp")
    elif preset == QWEN_LAYER_PRESET_ATTN_ONLY:
        filters.append("attn")
    elif preset == QWEN_LAYER_PRESET_BLOCKS:
        filters.append("transformer_block")
    elif preset == QWEN_LAYER_PRESET_FULL:
        pass
    else:
        raise Error(String("unknown Qwen layer preset: ") + preset)
    return filters^


def qwen_predict_required_batch_fields() -> List[String]:
    """Fields read by Serenity BaseQwenSetup.predict."""
    var fields = List[String]()
    fields.append("latent_image")
    return fields^


def qwen_predict_conditioning_batch_fields() -> List[String]:
    """Optional/conditional prompt fields read by model.encode_text."""
    var fields = List[String]()
    fields.append("tokens")
    fields.append("tokens_mask")
    fields.append("text_encoder_hidden_state")
    return fields^


def qwen_predict_output_fields() -> List[String]:
    """Fields written in Serenity model_output_data."""
    var fields = List[String]()
    fields.append(QWEN_PREDICT_KEY_LOSS_TYPE)
    fields.append(QWEN_PREDICT_KEY_TIMESTEP)
    fields.append(QWEN_PREDICT_KEY_PREDICTED)
    fields.append(QWEN_PREDICT_KEY_TARGET)
    return fields^


def qwen_flow_target_expression() -> String:
    return "latent_noise - scaled_latent_image"


def qwen_noisy_latent_expression() -> String:
    return "latent_noise * sigma + scaled_latent_image * (1 - sigma)"


def qwen_scale_latents_expression() -> String:
    return "(latents - vae.config.latents_mean) * (1 / vae.config.latents_std)"


def qwen_transformer_timestep_expression() -> String:
    return "timestep / 1000"


def qwen_default_prompt_template() -> String:
    return (
        "<|im_start|>system\nDescribe the image by detailing the color, shape, "
        + "size, texture, quantity, text, spatial relationships of the objects "
        + "and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n"
        + "<|im_start|>assistant\n"
    )


def qwen_transformer_text_mask_all_true_becomes_none() -> Bool:
    return True


def qwen_pack_latent_token_count(latent_height: Int, latent_width: Int) -> Int:
    return (latent_height // QWEN_LATENT_PACK_FACTOR) * (
        latent_width // QWEN_LATENT_PACK_FACTOR
    )


def qwen_pack_latent_channel_count(latent_channels: Int) -> Int:
    return latent_channels * QWEN_LATENT_PACK_FACTOR * QWEN_LATENT_PACK_FACTOR


def qwen_unpacked_latent_channel_count(packed_channels: Int) -> Int:
    return packed_channels // (QWEN_LATENT_PACK_FACTOR * QWEN_LATENT_PACK_FACTOR)


def qwen_img_shape_frame_count() -> Int:
    return QWEN_IMAGE_BASIS_FRAME


def qwen_img_shape_height(latent_height: Int) -> Int:
    return latent_height // QWEN_LATENT_PACK_FACTOR


def qwen_img_shape_width(latent_width: Int) -> Int:
    return latent_width // QWEN_LATENT_PACK_FACTOR


def qwen_sigma_from_timestep(
    t: Int, num_timesteps: Int = QWEN_NUM_TRAIN_TIMESTEPS
) -> Float32:
    # ModelSetupFlowMatchingMixin._add_noise_discrete:
    # sigma[t] = arange(1, N+1)[t] / N.
    return Float32(t + 1) / Float32(num_timesteps)


def qwen_model_t_from_timestep(
    t: Int, num_timesteps: Int = QWEN_NUM_TRAIN_TIMESTEPS
) -> Float32:
    # BaseQwenSetup.predict passes timestep / 1000 to the transformer.
    return Float32(t) / Float32(num_timesteps)


def qwen_calculate_timestep_shift(
    latent_height: Int,
    latent_width: Int,
    base_image_seq_len: Float32,
    max_image_seq_len: Float32,
    base_shift: Float32,
    max_shift: Float32,
) -> Float32:
    # QwenModel.calculate_timestep_shift uses patch_size=2 and the scheduler
    # config values. The product makes width/height argument order irrelevant.
    var patch_size = QWEN_LATENT_PACK_FACTOR
    var image_seq_len = Float32(
        (latent_width // patch_size) * (latent_height // patch_size)
    )
    var m = (max_shift - base_shift) / (max_image_seq_len - base_image_seq_len)
    var b = base_shift - m * base_image_seq_len
    var mu = image_seq_len * m + b
    return exp(mu)


def qwen_get_timestep_discrete(
    num_train_timesteps: Int,
    deterministic: Bool,
    seed: UInt64,
    timestep_distribution: Int,
    min_noising_strength: Float32,
    max_noising_strength: Float32,
    noising_weight: Float32,
    noising_bias: Float32,
    shift: Float32,
) raises -> Int:
    if deterministic:
        return Int(Float64(num_train_timesteps) * Float64(0.5)) - 1

    var host = _ts_host(
        num_train_timesteps,
        QWEN_IMAGE_BASIS_BATCH,
        timestep_distribution,
        Float64(min_noising_strength),
        Float64(max_noising_strength),
        Float64(noising_weight),
        Float64(noising_bias),
        Float64(shift),
        List[Float64](),
        seed,
    )
    return host.values[0]


struct QwenPredictContract(Movable):
    var required_batch_fields: List[String]
    var conditioning_batch_fields: List[String]
    var output_fields: List[String]
    var loss_type: String
    var target_expression: String
    var noisy_latent_expression: String
    var scale_latents_expression: String
    var transformer_timestep_expression: String

    def __init__(out self):
        self.required_batch_fields = qwen_predict_required_batch_fields()
        self.conditioning_batch_fields = qwen_predict_conditioning_batch_fields()
        self.output_fields = qwen_predict_output_fields()
        self.loss_type = QWEN_LOSS_TYPE_TARGET
        self.target_expression = qwen_flow_target_expression()
        self.noisy_latent_expression = qwen_noisy_latent_expression()
        self.scale_latents_expression = qwen_scale_latents_expression()
        self.transformer_timestep_expression = qwen_transformer_timestep_expression()


def qwen_predict_contract() -> QwenPredictContract:
    return QwenPredictContract()


struct BaseQwenSetup(Copyable, Movable, ImplicitlyCopyable):
    var debug_mode: Bool

    def __init__(out self, debug_mode: Bool = False):
        self.debug_mode = debug_mode

    def layer_preset_filters(self, preset: String) raises -> List[String]:
        return qwen_layer_preset_filters(preset)

    def predict_contract(self) -> QwenPredictContract:
        return qwen_predict_contract()

    def calculate_loss_consumes_sigmas(self) -> Bool:
        # calculate_loss delegates to _flow_matching_losses(..., sigmas=scheduler.sigmas).
        return True

    def calculate_loss_reduction(self) -> String:
        return "mean"

    def prepare_text_caching_requires_text_encoder_eval(self) -> Bool:
        # BaseQwenSetup.prepare_text_caching moves to temp, then eval() before cache.
        return True

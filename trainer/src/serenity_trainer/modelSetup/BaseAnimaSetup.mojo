# BaseAnimaSetup.mojo - build-only Anima setup contract.
#
# Source of truth:
#   /home/alex/Serenity-anima-ref/modules/modelSetup/BaseAnimaSetup.py
#   /home/alex/Serenity-anima-ref/modules/model/AnimaModel.py
#
# This file records the Serenity Anima setup/predict surface needed by later
# runtime and parity work. It intentionally does not execute the unfinished
# Anima runtime model. Tensor dtype casts in the Python reference are represented
# as contracts; scalar schedule helpers stay on the host.

from std.math import exp

from serenity_trainer.modelSetup.mixin.ModelSetupNoiseMixin import (
    _get_timestep_discrete_host as _ts_host,
)
from serenity_trainer.util.enum.TrainingMethod import TM_LORA


comptime ANIMA_MODEL_TYPE_NAME = "ANIMA"
comptime ANIMA_REFERENCE_MODEL_TYPE_INDEX = 24
comptime ANIMA_NUM_TRAIN_TIMESTEPS = 1000
comptime ANIMA_PROMPT_MAX_LENGTH = 512
comptime ANIMA_LATENT_PATCH_FACTOR = 2
comptime ANIMA_PIXEL_TO_LATENT_FACTOR = 8
comptime ANIMA_IMAGE_BASIS_BATCH = 1
comptime ANIMA_TEXT_CONDITIONER_SEQUENCE_LENGTH = 512
comptime ANIMA_TEXT_CONDITIONER_HIDDEN_SIZE = 1024
comptime ANIMA_LATENT_FRAME_DIM = 1

comptime ANIMA_LAYER_PRESET_ATTN_MLP = "attn-mlp"
comptime ANIMA_LAYER_PRESET_ATTN_ONLY = "attn-only"
comptime ANIMA_LAYER_PRESET_BLOCKS = "blocks"
comptime ANIMA_LAYER_PRESET_FULL = "full"

comptime ANIMA_LOSS_TYPE_TARGET = "target"
comptime ANIMA_PREDICT_KEY_LOSS_TYPE = "loss_type"
comptime ANIMA_PREDICT_KEY_TIMESTEP = "timestep"
comptime ANIMA_PREDICT_KEY_PREDICTED = "predicted"
comptime ANIMA_PREDICT_KEY_TARGET = "target"

comptime ANIMA_PART_TRANSFORMER = "transformer"
comptime ANIMA_PART_TEXT_ENCODER = "text_encoder"
comptime ANIMA_PART_TEXT_CONDITIONER = "text_conditioner"
comptime ANIMA_PART_VAE = "vae"
comptime ANIMA_PART_LORA = "lora"


def anima_layer_preset_filters(preset: String) raises -> List[String]:
    """Serenity BaseAnimaSetup.LAYER_PRESETS."""
    var filters = List[String]()
    if preset == ANIMA_LAYER_PRESET_ATTN_MLP:
        filters.append("attn1")
        filters.append("attn2")
        filters.append("ff")
    elif preset == ANIMA_LAYER_PRESET_ATTN_ONLY:
        filters.append("attn1")
        filters.append("attn2")
    elif preset == ANIMA_LAYER_PRESET_BLOCKS:
        filters.append("transformer_block")
    elif preset == ANIMA_LAYER_PRESET_FULL:
        pass
    else:
        raise Error(String("unknown Anima layer preset: ") + preset)
    return filters^


def anima_predict_required_batch_fields() -> List[String]:
    """Fields read directly by Serenity BaseAnimaSetup.predict."""
    var fields = List[String]()
    fields.append("latent_image")
    return fields^


def anima_predict_conditioning_batch_fields() -> List[String]:
    """Optional/conditional fields passed through model.encode_text."""
    var fields = List[String]()
    fields.append("tokens")
    fields.append("tokens_mask")
    fields.append("text_encoder_hidden_state")
    return fields^


def anima_predict_dataloader_text_fields() -> List[String]:
    """Text fields emitted by AnimaBaseDataLoader; predict consumes cached hidden state."""
    var fields = List[String]()
    fields.append("tokens")
    fields.append("tokens_mask")
    fields.append("t5_tokens")
    fields.append("t5_tokens_mask")
    fields.append("text_encoder_hidden_state")
    return fields^


def anima_predict_output_fields() -> List[String]:
    """Fields written in Serenity model_output_data."""
    var fields = List[String]()
    fields.append(ANIMA_PREDICT_KEY_LOSS_TYPE)
    fields.append(ANIMA_PREDICT_KEY_TIMESTEP)
    fields.append(ANIMA_PREDICT_KEY_PREDICTED)
    fields.append(ANIMA_PREDICT_KEY_TARGET)
    return fields^


def anima_setup_optimization_checkpoint_parts() -> List[String]:
    var parts = List[String]()
    parts.append(ANIMA_PART_TRANSFORMER)
    parts.append(ANIMA_PART_TEXT_ENCODER)
    return parts^


def anima_setup_optimization_checkpoint_helpers() -> List[String]:
    var helpers = List[String]()
    helpers.append("enable_checkpointing_for_qwen_transformer")
    helpers.append("enable_checkpointing_for_qwen3_encoder_layers")
    return helpers^


def anima_setup_optimization_quantized_parts() -> List[String]:
    var parts = List[String]()
    parts.append(ANIMA_PART_TEXT_ENCODER)
    parts.append(ANIMA_PART_VAE)
    parts.append(ANIMA_PART_TRANSFORMER)
    return parts^


def anima_autocast_weight_dtype_parts(training_method: Int) -> List[String]:
    var parts = List[String]()
    parts.append(ANIMA_PART_TRANSFORMER)
    parts.append(ANIMA_PART_TEXT_ENCODER)
    parts.append(ANIMA_PART_VAE)
    if training_method == TM_LORA:
        parts.append(ANIMA_PART_LORA)
    return parts^


def anima_text_encoder_autocast_weight_dtype_parts(training_method: Int) -> List[String]:
    var parts = List[String]()
    parts.append(ANIMA_PART_TEXT_ENCODER)
    if training_method == TM_LORA:
        parts.append(ANIMA_PART_LORA)
    return parts^


def anima_flow_target_expression() -> String:
    return "latent_noise - scaled_latent_image"


def anima_noisy_latent_expression() -> String:
    return "latent_noise * sigma + scaled_latent_image * (1 - sigma)"


def anima_scale_latents_expression() -> String:
    return "(latents - vae.config.latents_mean) * (1 / vae.config.latents_std)"


def anima_unscale_latents_expression() -> String:
    return "latents / (1 / vae.config.latents_std) + vae.config.latents_mean"


def anima_transformer_timestep_expression() -> String:
    return "timestep / 1000"


def anima_transformer_hidden_states_expression() -> String:
    return "scaled_noisy_latent_image.to(dtype=model.train_dtype.torch_dtype())"


def anima_transformer_encoder_hidden_states_expression() -> String:
    return "text_encoder_output.to(dtype=model.train_dtype.torch_dtype())"


def anima_padding_mask_expression() -> String:
    return "zeros(1, 1, latent_h * 8, latent_w * 8).to(dtype=model.train_dtype.torch_dtype())"


def anima_text_encoder_output_expression() -> String:
    return "AnimaTextConditioner(Qwen3 hidden states, T5 token ids) -> (B, 512, 1024)"


def anima_latents_are_5d_video_shape() -> Bool:
    return True


def anima_packs_latents_for_transformer() -> Bool:
    return False


def anima_padding_mask_height(latent_height: Int) -> Int:
    return latent_height * ANIMA_PIXEL_TO_LATENT_FACTOR


def anima_padding_mask_width(latent_width: Int) -> Int:
    return latent_width * ANIMA_PIXEL_TO_LATENT_FACTOR


def anima_text_conditioner_sequence_length() -> Int:
    return ANIMA_TEXT_CONDITIONER_SEQUENCE_LENGTH


def anima_text_conditioner_hidden_size() -> Int:
    return ANIMA_TEXT_CONDITIONER_HIDDEN_SIZE


def anima_sigma_from_timestep(
    t: Int, num_timesteps: Int = ANIMA_NUM_TRAIN_TIMESTEPS
) -> Float32:
    # ModelSetupFlowMatchingMixin._add_noise_discrete:
    # sigma[t] = arange(1, N+1)[t] / N.
    return Float32(t + 1) / Float32(num_timesteps)


def anima_model_t_from_timestep(
    t: Int, num_timesteps: Int = ANIMA_NUM_TRAIN_TIMESTEPS
) -> Float32:
    # BaseAnimaSetup.predict passes timestep / 1000 to CosmosTransformer3DModel.
    return Float32(t) / Float32(num_timesteps)


def anima_calculate_timestep_shift(
    latent_width: Int,
    latent_height: Int,
    base_image_seq_len: Float32,
    max_image_seq_len: Float32,
    base_shift: Float32,
    max_shift: Float32,
) -> Float32:
    # AnimaModel.calculate_timestep_shift reads FlowMatch scheduler config and
    # uses patch_size=2. The product makes width/height order irrelevant.
    var patch_size = ANIMA_LATENT_PATCH_FACTOR
    var image_seq_len = Float32(
        (latent_width // patch_size) * (latent_height // patch_size)
    )
    var m = (max_shift - base_shift) / (max_image_seq_len - base_image_seq_len)
    var b = base_shift - m * base_image_seq_len
    var mu = image_seq_len * m + b
    return exp(mu)


def anima_get_timestep_discrete(
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
        ANIMA_IMAGE_BASIS_BATCH,
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


struct AnimaPredictContract(Movable):
    var required_batch_fields: List[String]
    var conditioning_batch_fields: List[String]
    var dataloader_text_fields: List[String]
    var output_fields: List[String]
    var loss_type: String
    var target_expression: String
    var noisy_latent_expression: String
    var scale_latents_expression: String
    var unscale_latents_expression: String
    var transformer_timestep_expression: String
    var transformer_hidden_states_expression: String
    var transformer_encoder_hidden_states_expression: String
    var padding_mask_expression: String
    var text_encoder_output_expression: String
    var latents_are_5d_video_shape: Bool
    var packs_latents_for_transformer: Bool

    def __init__(out self):
        self.required_batch_fields = anima_predict_required_batch_fields()
        self.conditioning_batch_fields = anima_predict_conditioning_batch_fields()
        self.dataloader_text_fields = anima_predict_dataloader_text_fields()
        self.output_fields = anima_predict_output_fields()
        self.loss_type = ANIMA_LOSS_TYPE_TARGET
        self.target_expression = anima_flow_target_expression()
        self.noisy_latent_expression = anima_noisy_latent_expression()
        self.scale_latents_expression = anima_scale_latents_expression()
        self.unscale_latents_expression = anima_unscale_latents_expression()
        self.transformer_timestep_expression = anima_transformer_timestep_expression()
        self.transformer_hidden_states_expression = (
            anima_transformer_hidden_states_expression()
        )
        self.transformer_encoder_hidden_states_expression = (
            anima_transformer_encoder_hidden_states_expression()
        )
        self.padding_mask_expression = anima_padding_mask_expression()
        self.text_encoder_output_expression = anima_text_encoder_output_expression()
        self.latents_are_5d_video_shape = anima_latents_are_5d_video_shape()
        self.packs_latents_for_transformer = anima_packs_latents_for_transformer()


struct AnimaOptimizationContract(Movable):
    var checkpoint_parts: List[String]
    var checkpoint_helpers: List[String]
    var quantized_parts: List[String]
    var autocast_weight_dtype_parts: List[String]
    var text_encoder_autocast_weight_dtype_parts: List[String]
    var disables_fp16_text_encoder_autocast: Bool
    var text_conditioner_dtype_source: String

    def __init__(out self, training_method: Int):
        self.checkpoint_parts = anima_setup_optimization_checkpoint_parts()
        self.checkpoint_helpers = anima_setup_optimization_checkpoint_helpers()
        self.quantized_parts = anima_setup_optimization_quantized_parts()
        self.autocast_weight_dtype_parts = anima_autocast_weight_dtype_parts(
            training_method
        )
        self.text_encoder_autocast_weight_dtype_parts = (
            anima_text_encoder_autocast_weight_dtype_parts(training_method)
        )
        self.disables_fp16_text_encoder_autocast = True
        self.text_conditioner_dtype_source = "AnimaTextConditioner.from_pretrained(..., torch_dtype=torch.bfloat16)"


struct AnimaTrainDevicePlan(Copyable, Movable, ImplicitlyCopyable):
    var text_encoder_on_train_device: Bool
    var text_conditioner_on_train_device: Bool
    var vae_on_train_device: Bool
    var transformer_on_train_device: Bool
    var text_encoder_train_mode: Bool
    var text_conditioner_train_mode: Bool
    var vae_train_mode: Bool
    var transformer_train_mode: Bool

    def __init__(out self, latent_caching: Bool, transformer_train: Bool):
        self.text_encoder_on_train_device = not latent_caching
        self.text_conditioner_on_train_device = not latent_caching
        self.vae_on_train_device = not latent_caching
        self.transformer_on_train_device = True
        self.text_encoder_train_mode = False
        self.text_conditioner_train_mode = False
        self.vae_train_mode = False
        self.transformer_train_mode = transformer_train


struct AnimaTextCachingPlan(Copyable, Movable, ImplicitlyCopyable):
    var move_model_to_temp_device: Bool
    var move_text_encoder_to_train_device: Bool
    var move_text_conditioner_to_train_device: Bool
    var set_eval_mode: Bool
    var run_torch_gc: Bool

    def __init__(out self):
        self.move_model_to_temp_device = True
        self.move_text_encoder_to_train_device = True
        self.move_text_conditioner_to_train_device = True
        self.set_eval_mode = True
        self.run_torch_gc = True


def anima_predict_contract() -> AnimaPredictContract:
    return AnimaPredictContract()


def anima_optimization_contract(training_method: Int) -> AnimaOptimizationContract:
    return AnimaOptimizationContract(training_method)


def anima_train_device_plan(
    latent_caching: Bool, transformer_train: Bool
) -> AnimaTrainDevicePlan:
    return AnimaTrainDevicePlan(latent_caching, transformer_train)


def anima_text_caching_plan() -> AnimaTextCachingPlan:
    return AnimaTextCachingPlan()


struct BaseAnimaSetup(Copyable, Movable, ImplicitlyCopyable):
    var debug_mode: Bool

    def __init__(out self, debug_mode: Bool = False):
        self.debug_mode = debug_mode

    def layer_preset_filters(self, preset: String) raises -> List[String]:
        return anima_layer_preset_filters(preset)

    def predict_contract(self) -> AnimaPredictContract:
        return anima_predict_contract()

    def optimization_contract(self, training_method: Int) -> AnimaOptimizationContract:
        return anima_optimization_contract(training_method)

    def train_device_plan(
        self, latent_caching: Bool, transformer_train: Bool
    ) -> AnimaTrainDevicePlan:
        return anima_train_device_plan(latent_caching, transformer_train)

    def calculate_loss_consumes_sigmas(self) -> Bool:
        # calculate_loss delegates to _flow_matching_losses(..., sigmas=scheduler.sigmas).
        return True

    def calculate_loss_reduction(self) -> String:
        return "mean"

    def prepare_text_caching_plan(self) -> AnimaTextCachingPlan:
        return anima_text_caching_plan()

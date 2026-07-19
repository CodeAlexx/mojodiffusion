# BaseFluxSetup.mojo - build-only FLUX.1 Dev setup contract.
#
# Source of truth:
#   /home/alex/Serenity/modules/modelSetup/BaseFluxSetup.py
#   /home/alex/Serenity/modules/model/FluxModel.py
#
# This surface records the Serenity FLUX.1 setup/predict contract without
# executing the unfinished Mojo FLUX runtime. It carries field names, setup plans,
# scalar timestep/sigma helpers, and dtype caveats. It does not claim numeric
# parity and does not introduce persistent F32 tensor storage.

from std.collections import Optional
from std.math import exp

from serenity_trainer.modelSetup.mixin.ModelSetupNoiseMixin import (
    _get_timestep_discrete_host as _ts_host,
)
from serenity_trainer.util.enum.ModelType import (
    MODEL_TYPE_FLUX_DEV_1,
    MODEL_TYPE_FLUX_FILL_DEV_1,
)
from serenity_trainer.util.enum.TrainingMethod import TM_LORA


comptime FLUX_DEV_MODEL_TYPE_NAME = "FLUX_DEV_1"
comptime FLUX_FILL_MODEL_TYPE_NAME = "FLUX_FILL_DEV_1"
comptime FLUX_NUM_TRAIN_TIMESTEPS = 1000
comptime FLUX_IMAGE_BASIS_BATCH = 1
comptime FLUX_TOKENIZER_1_MAX_TOKENS = 77
comptime FLUX_VAE_SCALE_FACTOR = 8
comptime FLUX_VAE_LATENT_CHANNELS = 16
comptime FLUX_PACK_PATCH_SIZE = 2
comptime FLUX_FILL_MASK_CHANNELS = FLUX_VAE_SCALE_FACTOR * FLUX_VAE_SCALE_FACTOR

# FlowMatchEulerDiscreteScheduler defaults used by FluxModel.calculate_timestep_shift
# when the loaded scheduler config does not override them.
comptime FLUX_BASE_IMAGE_SEQ_LEN = 256
comptime FLUX_MAX_IMAGE_SEQ_LEN = 4096
comptime FLUX_BASE_SHIFT = Float32(0.5)
comptime FLUX_MAX_SHIFT = Float32(1.15)

# FLUX.1 AutoencoderKL defaults from the public FLUX.1 Dev VAE config. Serenity
# reads these at runtime from model.vae.config; callers should pass checkpoint
# values when they differ.
comptime FLUX_DEFAULT_VAE_SHIFT_FACTOR = Float32(0.1159)
comptime FLUX_DEFAULT_VAE_SCALING_FACTOR = Float32(0.3611)

comptime FLUX_LAYER_PRESET_ATTN_MLP = "attn-mlp"
comptime FLUX_LAYER_PRESET_ATTN_ONLY = "attn-only"
comptime FLUX_LAYER_PRESET_BLOCKS = "blocks"
comptime FLUX_LAYER_PRESET_FULL = "full"

comptime FLUX_LOSS_TYPE_TARGET = "target"
comptime FLUX_PREDICT_KEY_LOSS_TYPE = "loss_type"
comptime FLUX_PREDICT_KEY_TIMESTEP = "timestep"
comptime FLUX_PREDICT_KEY_PREDICTED = "predicted"
comptime FLUX_PREDICT_KEY_TARGET = "target"

comptime FLUX_PART_TRANSFORMER = "transformer"
comptime FLUX_PART_TEXT_ENCODER_1 = "text_encoder_1"
comptime FLUX_PART_TEXT_ENCODER_2 = "text_encoder_2"
comptime FLUX_PART_VAE = "vae"
comptime FLUX_PART_LORA = "lora"
comptime FLUX_PART_EMBEDDING = "embedding"

comptime FLUX_DTYPE_PART_TEXT_ENCODER = "text_encoder"
comptime FLUX_DTYPE_PART_TEXT_ENCODER_2 = "text_encoder_2"


def flux_setup_model_types() -> List[Int]:
    var model_types = List[Int]()
    model_types.append(MODEL_TYPE_FLUX_DEV_1)
    model_types.append(MODEL_TYPE_FLUX_FILL_DEV_1)
    return model_types^


def flux_layer_preset_filters(preset: String) raises -> List[String]:
    """Serenity BaseFluxSetup.LAYER_PRESETS."""
    var filters = List[String]()
    if preset == FLUX_LAYER_PRESET_ATTN_MLP:
        filters.append("attn")
        filters.append("ff.net")
    elif preset == FLUX_LAYER_PRESET_ATTN_ONLY:
        filters.append("attn")
    elif preset == FLUX_LAYER_PRESET_BLOCKS:
        filters.append("transformer_block")
    elif preset == FLUX_LAYER_PRESET_FULL:
        pass
    else:
        raise Error(String("unknown FLUX layer preset: ") + preset)
    return filters^


def flux_predict_required_batch_fields() -> List[String]:
    """Fields read unconditionally by Serenity BaseFluxSetup.predict."""
    var fields = List[String]()
    fields.append("latent_image")
    return fields^


def flux_predict_text_batch_fields() -> List[String]:
    """Text/cache fields accepted by FluxModel.encode_text in predict()."""
    var fields = List[String]()
    fields.append("tokens_1")
    fields.append("tokens_2")
    fields.append("tokens_mask_2")
    fields.append("text_encoder_1_pooled_state")
    fields.append("text_encoder_2_hidden_state")
    return fields^


def flux_predict_conditional_latent_fields() -> List[String]:
    """Only used for FLUX Fill when both mask and conditioning inputs are enabled."""
    var fields = List[String]()
    fields.append("latent_conditioning_image")
    fields.append("latent_mask")
    return fields^


def flux_predict_output_fields() -> List[String]:
    """Fields written in Serenity model_output_data."""
    var fields = List[String]()
    fields.append(FLUX_PREDICT_KEY_LOSS_TYPE)
    fields.append(FLUX_PREDICT_KEY_TIMESTEP)
    fields.append(FLUX_PREDICT_KEY_PREDICTED)
    fields.append(FLUX_PREDICT_KEY_TARGET)
    return fields^


def flux_setup_optimization_checkpoint_parts() -> List[String]:
    var parts = List[String]()
    parts.append(FLUX_PART_TRANSFORMER)
    parts.append(FLUX_PART_TEXT_ENCODER_1)
    parts.append(FLUX_PART_TEXT_ENCODER_2)
    return parts^


def flux_setup_optimization_checkpoint_helpers() -> List[String]:
    var helpers = List[String]()
    helpers.append("enable_checkpointing_for_flux_transformer")
    helpers.append("enable_checkpointing_for_clip_encoder_layers:text_encoder_1")
    helpers.append("enable_checkpointing_for_t5_encoder_layers:text_encoder_2")
    return helpers^


def flux_setup_optimization_quantized_parts() -> List[String]:
    var parts = List[String]()
    parts.append(FLUX_PART_TEXT_ENCODER_1)
    parts.append(FLUX_PART_TEXT_ENCODER_2)
    parts.append(FLUX_PART_VAE)
    parts.append(FLUX_PART_TRANSFORMER)
    return parts^


def flux_autocast_weight_dtype_parts(
    training_method: Int, train_any_embedding: Bool
) -> List[String]:
    var parts = List[String]()
    parts.append(FLUX_PART_TRANSFORMER)
    parts.append(FLUX_DTYPE_PART_TEXT_ENCODER)
    parts.append(FLUX_DTYPE_PART_TEXT_ENCODER_2)
    parts.append(FLUX_PART_VAE)
    if training_method == TM_LORA:
        parts.append(FLUX_PART_LORA)
    if train_any_embedding:
        parts.append(FLUX_PART_EMBEDDING)
    return parts^


def flux_text_encoder_2_autocast_weight_dtype_parts(
    training_method: Int, train_any_embedding: Bool
) -> List[String]:
    var parts = List[String]()
    parts.append(FLUX_DTYPE_PART_TEXT_ENCODER_2)
    if training_method == TM_LORA:
        parts.append(FLUX_PART_LORA)
    if train_any_embedding:
        parts.append(FLUX_PART_EMBEDDING)
    return parts^


def flux_flow_target_expression() -> String:
    return "latent_noise - scaled_latent_image"


def flux_noisy_latent_expression() -> String:
    return "latent_noise * sigma + scaled_latent_image * (1 - sigma)"


def flux_scale_latents_expression() -> String:
    return "(latent_image - vae.config['shift_factor']) * vae.config['scaling_factor']"


def flux_scale_conditioning_latents_expression() -> String:
    return "(latent_conditioning_image - vae.config['shift_factor']) * vae.config['scaling_factor']"


def flux_latent_input_expression() -> String:
    return "concat([scaled_noisy_latent_image, scaled_latent_conditioning_image, latent_mask], dim=1) only when mask+conditioning inputs are enabled; otherwise scaled_noisy_latent_image"


def flux_pack_latents_expression() -> String:
    return "view(B,C,H/2,2,W/2,2) -> permute(0,2,4,1,3,5) -> reshape(B,(H/2)*(W/2),C*4)"


def flux_unpack_latents_expression() -> String:
    return "view(B,H/2,W/2,C/4,2,2) -> permute(0,3,1,4,2,5) -> reshape(B,C/4,H,W)"


def flux_prepare_latent_image_ids_expression() -> String:
    return "zeros(H/2,W/2,3); channel 1 = row id; channel 2 = column id; reshape((H/2)*(W/2),3)"


def flux_transformer_timestep_expression() -> String:
    return "timestep / 1000"


def flux_guidance_expression() -> String:
    return "tensor([config.transformer.guidance_scale]).expand(B) if model.transformer.config.guidance_embeds else None"


def flux_transformer_hidden_states_expression() -> String:
    return "pack_latents(latent_input).to(dtype=model.train_dtype.torch_dtype())"


def flux_transformer_encoder_hidden_states_expression() -> String:
    return "text_encoder_2_output.to(dtype=model.train_dtype.torch_dtype())"


def flux_transformer_pooled_projection_expression() -> String:
    return "text_encoder_1_pooled_state.to(dtype=model.train_dtype.torch_dtype())"


def flux_predicted_expression() -> String:
    return "unpack_latents(model.transformer(..., return_dict=True).sample, latent_input.shape[2], latent_input.shape[3])"


def flux_text_encoder_expression() -> String:
    return "FluxModel.encode_text returns (T5 hidden states, CLIP-L pooled projection); cached states are reused only when their encoder/embedding is frozen"


def flux_dtype_boundary_caveats() -> List[String]:
    var caveats = List[String]()
    caveats.append("Build-only surface: no transformer, VAE, tokenizer, or MGDS runtime execution is implemented here")
    caveats.append("Persistent cached latent_image, latent_conditioning_image, text_encoder_1_pooled_state, and text_encoder_2_hidden_state storage must keep the model/checkpoint dtype")
    caveats.append("Sigma, timestep shift, SNR, and schedule sampling helpers are host scalar math; they do not justify F32 tensor storage boundaries")
    caveats.append("VAE shift/scale and add-noise kernels may use F32 internally, but a runtime implementation must return the input/storage dtype")
    caveats.append("Guidance and timestep scalars are cast only at the transformer call boundary")
    caveats.append("RNG values are not a numeric parity claim; Serenity uses a torch.Generator shared by noise and timestep sampling")
    return caveats^


def flux_unsupported_runtime_paths() -> List[String]:
    var paths = List[String]()
    paths.append("predict runtime tensor path is not implemented in this setup surface")
    paths.append("MGDS data pipeline execution is not implemented in this setup surface")
    paths.append("LoRA/fine-tune wrappers are represented as setup plans only")
    paths.append("Serenity execution, parity, speed, and numeric claims are owned by later runtime work")
    return paths^


def flux_sigma_from_timestep(
    t: Int, num_timesteps: Int = FLUX_NUM_TRAIN_TIMESTEPS
) -> Float32:
    # ModelSetupFlowMatchingMixin._add_noise_discrete:
    # sigma[t] = arange(1, N + 1)[t] / N.
    return Float32(t + 1) / Float32(num_timesteps)


def flux_timestep_from_sigma(
    sigma: Float32, num_timesteps: Int = FLUX_NUM_TRAIN_TIMESTEPS
) -> Int:
    var idx = Int(sigma * Float32(num_timesteps) + Float32(0.5)) - 1
    if idx < 0:
        return 0
    if idx >= num_timesteps:
        return num_timesteps - 1
    return idx


def flux_model_t_from_timestep(
    t: Int, num_timesteps: Int = FLUX_NUM_TRAIN_TIMESTEPS
) -> Float32:
    # BaseFluxSetup.predict passes timestep / 1000 to FluxTransformer2DModel.
    return Float32(t) / Float32(num_timesteps)


def flux_snr_from_sigma(sigma: Float32) -> Float32:
    var s = sigma
    if s < Float32(1e-8):
        s = Float32(1e-8)
    var ratio = (Float32(1.0) - s) / s
    return ratio * ratio


def flux_calculate_timestep_shift(
    latent_h: Int,
    latent_w: Int,
    base_image_seq_len: Int = FLUX_BASE_IMAGE_SEQ_LEN,
    max_image_seq_len: Int = FLUX_MAX_IMAGE_SEQ_LEN,
    base_shift: Float32 = FLUX_BASE_SHIFT,
    max_shift: Float32 = FLUX_MAX_SHIFT,
    patch_size: Int = FLUX_PACK_PATCH_SIZE,
) -> Float32:
    var image_seq_len = Float32((latent_w // patch_size) * (latent_h // patch_size))
    var m = (max_shift - base_shift) / (
        Float32(max_image_seq_len) - Float32(base_image_seq_len)
    )
    var b = base_shift - m * Float32(base_image_seq_len)
    var mu = image_seq_len * m + b
    return exp(mu)


def flux_get_timestep_discrete(
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
        FLUX_IMAGE_BASIS_BATCH,
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


def flux_guidance_value(
    guidance_scale: Float32, guidance_embeds: Bool
) -> Optional[Float32]:
    if not guidance_embeds:
        return Optional[Float32](None)
    return Optional[Float32](guidance_scale)


def flux_packed_latent_token_count(latent_h: Int, latent_w: Int) -> Int:
    return (latent_h // FLUX_PACK_PATCH_SIZE) * (latent_w // FLUX_PACK_PATCH_SIZE)


def flux_packed_latent_channels(latent_channels: Int = FLUX_VAE_LATENT_CHANNELS) -> Int:
    return latent_channels * FLUX_PACK_PATCH_SIZE * FLUX_PACK_PATCH_SIZE


def flux_latent_input_channels(
    has_mask_input: Bool,
    has_conditioning_image_input: Bool,
    latent_channels: Int = FLUX_VAE_LATENT_CHANNELS,
) -> Int:
    if has_mask_input and has_conditioning_image_input:
        return latent_channels + latent_channels + FLUX_FILL_MASK_CHANNELS
    return latent_channels


struct FluxPredictContract(Movable):
    var required_batch_fields: List[String]
    var text_batch_fields: List[String]
    var conditional_latent_fields: List[String]
    var output_fields: List[String]
    var loss_type: String
    var target_expression: String
    var noisy_latent_expression: String
    var scale_latents_expression: String
    var scale_conditioning_latents_expression: String
    var latent_input_expression: String
    var pack_latents_expression: String
    var unpack_latents_expression: String
    var prepare_latent_image_ids_expression: String
    var transformer_timestep_expression: String
    var guidance_expression: String
    var transformer_hidden_states_expression: String
    var transformer_encoder_hidden_states_expression: String
    var transformer_pooled_projection_expression: String
    var predicted_expression: String
    var text_encoder_expression: String
    var dtype_boundary_caveats: List[String]

    def __init__(out self):
        self.required_batch_fields = flux_predict_required_batch_fields()
        self.text_batch_fields = flux_predict_text_batch_fields()
        self.conditional_latent_fields = flux_predict_conditional_latent_fields()
        self.output_fields = flux_predict_output_fields()
        self.loss_type = FLUX_LOSS_TYPE_TARGET
        self.target_expression = flux_flow_target_expression()
        self.noisy_latent_expression = flux_noisy_latent_expression()
        self.scale_latents_expression = flux_scale_latents_expression()
        self.scale_conditioning_latents_expression = (
            flux_scale_conditioning_latents_expression()
        )
        self.latent_input_expression = flux_latent_input_expression()
        self.pack_latents_expression = flux_pack_latents_expression()
        self.unpack_latents_expression = flux_unpack_latents_expression()
        self.prepare_latent_image_ids_expression = (
            flux_prepare_latent_image_ids_expression()
        )
        self.transformer_timestep_expression = flux_transformer_timestep_expression()
        self.guidance_expression = flux_guidance_expression()
        self.transformer_hidden_states_expression = (
            flux_transformer_hidden_states_expression()
        )
        self.transformer_encoder_hidden_states_expression = (
            flux_transformer_encoder_hidden_states_expression()
        )
        self.transformer_pooled_projection_expression = (
            flux_transformer_pooled_projection_expression()
        )
        self.predicted_expression = flux_predicted_expression()
        self.text_encoder_expression = flux_text_encoder_expression()
        self.dtype_boundary_caveats = flux_dtype_boundary_caveats()


struct FluxOptimizationContract(Movable):
    var checkpoint_parts: List[String]
    var checkpoint_helpers: List[String]
    var quantized_parts: List[String]
    var autocast_weight_dtype_parts: List[String]
    var text_encoder_2_autocast_weight_dtype_parts: List[String]
    var disables_fp16_text_encoder_2_autocast: Bool

    def __init__(
        out self, training_method: Int, train_any_embedding: Bool = False
    ):
        self.checkpoint_parts = flux_setup_optimization_checkpoint_parts()
        self.checkpoint_helpers = flux_setup_optimization_checkpoint_helpers()
        self.quantized_parts = flux_setup_optimization_quantized_parts()
        self.autocast_weight_dtype_parts = flux_autocast_weight_dtype_parts(
            training_method, train_any_embedding
        )
        self.text_encoder_2_autocast_weight_dtype_parts = (
            flux_text_encoder_2_autocast_weight_dtype_parts(
                training_method, train_any_embedding
            )
        )
        self.disables_fp16_text_encoder_2_autocast = True


struct FluxTrainDevicePlan(Copyable, Movable, ImplicitlyCopyable):
    var text_encoder_1_on_train_device: Bool
    var text_encoder_2_on_train_device: Bool
    var vae_on_train_device: Bool
    var transformer_on_train_device: Bool
    var text_encoder_1_train_mode: Bool
    var text_encoder_2_train_mode: Bool
    var vae_train_mode: Bool
    var transformer_train_mode: Bool

    def __init__(
        out self,
        latent_caching: Bool,
        train_text_encoder_or_embedding: Bool,
        train_text_encoder_2_or_embedding: Bool,
        text_encoder_1_train: Bool,
        text_encoder_2_train: Bool,
        transformer_train: Bool,
    ):
        self.text_encoder_1_on_train_device = (
            train_text_encoder_or_embedding or not latent_caching
        )
        self.text_encoder_2_on_train_device = (
            train_text_encoder_2_or_embedding or not latent_caching
        )
        self.vae_on_train_device = not latent_caching
        self.transformer_on_train_device = True
        self.text_encoder_1_train_mode = text_encoder_1_train
        self.text_encoder_2_train_mode = text_encoder_2_train
        self.vae_train_mode = False
        self.transformer_train_mode = transformer_train


struct FluxTextCachingPlan(Copyable, Movable, ImplicitlyCopyable):
    var move_model_to_temp_device: Bool
    var move_text_encoder_1_to_train_device: Bool
    var move_text_encoder_2_to_train_device: Bool
    var set_eval_mode: Bool
    var run_torch_gc: Bool

    def __init__(
        out self,
        train_text_encoder_or_embedding: Bool,
        train_text_encoder_2_or_embedding: Bool,
    ):
        self.move_model_to_temp_device = True
        self.move_text_encoder_1_to_train_device = not train_text_encoder_or_embedding
        self.move_text_encoder_2_to_train_device = (
            not train_text_encoder_2_or_embedding
        )
        self.set_eval_mode = True
        self.run_torch_gc = True


struct FluxUnsupportedPaths(Movable):
    var paths: List[String]

    def __init__(out self):
        self.paths = flux_unsupported_runtime_paths()


def flux_predict_contract() -> FluxPredictContract:
    return FluxPredictContract()


def flux_optimization_contract(
    training_method: Int, train_any_embedding: Bool = False
) -> FluxOptimizationContract:
    return FluxOptimizationContract(training_method, train_any_embedding)


def flux_train_device_plan(
    latent_caching: Bool,
    train_text_encoder_or_embedding: Bool,
    train_text_encoder_2_or_embedding: Bool,
    text_encoder_1_train: Bool,
    text_encoder_2_train: Bool,
    transformer_train: Bool,
) -> FluxTrainDevicePlan:
    return FluxTrainDevicePlan(
        latent_caching,
        train_text_encoder_or_embedding,
        train_text_encoder_2_or_embedding,
        text_encoder_1_train,
        text_encoder_2_train,
        transformer_train,
    )


def flux_text_caching_plan(
    train_text_encoder_or_embedding: Bool,
    train_text_encoder_2_or_embedding: Bool,
) -> FluxTextCachingPlan:
    return FluxTextCachingPlan(
        train_text_encoder_or_embedding,
        train_text_encoder_2_or_embedding,
    )


struct BaseFluxSetup(Movable):
    var debug_mode: Bool
    var model_types: List[Int]

    def __init__(out self, debug_mode: Bool = False):
        self.debug_mode = debug_mode
        self.model_types = flux_setup_model_types()

    def layer_preset_filters(self, preset: String) raises -> List[String]:
        return flux_layer_preset_filters(preset)

    def predict_contract(self) -> FluxPredictContract:
        return flux_predict_contract()

    def optimization_contract(
        self, training_method: Int, train_any_embedding: Bool = False
    ) -> FluxOptimizationContract:
        return flux_optimization_contract(training_method, train_any_embedding)

    def train_device_plan(
        self,
        latent_caching: Bool,
        train_text_encoder_or_embedding: Bool,
        train_text_encoder_2_or_embedding: Bool,
        text_encoder_1_train: Bool,
        text_encoder_2_train: Bool,
        transformer_train: Bool,
    ) -> FluxTrainDevicePlan:
        return flux_train_device_plan(
            latent_caching,
            train_text_encoder_or_embedding,
            train_text_encoder_2_or_embedding,
            text_encoder_1_train,
            text_encoder_2_train,
            transformer_train,
        )

    def calculate_loss_consumes_sigmas(self) -> Bool:
        # calculate_loss delegates to _flow_matching_losses(..., sigmas=scheduler.sigmas).
        return True

    def calculate_loss_reduction(self) -> String:
        return "mean"

    def prepare_text_caching_plan(
        self,
        train_text_encoder_or_embedding: Bool,
        train_text_encoder_2_or_embedding: Bool,
    ) -> FluxTextCachingPlan:
        return flux_text_caching_plan(
            train_text_encoder_or_embedding,
            train_text_encoder_2_or_embedding,
        )

    def runtime_predict_implemented(self) -> Bool:
        return False

    def unsupported_runtime_paths(self) -> FluxUnsupportedPaths:
        return FluxUnsupportedPaths()

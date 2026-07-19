# BaseStableDiffusionXLSetup.mojo - build-only SDXL setup/predict contract.
#
# Source of truth:
#   /home/alex/Serenity/modules/modelSetup/BaseStableDiffusionXLSetup.py
#
# This file records Serenity's SDXL setup, predict, optimizer, checkpointing,
# quantization, caching, and scheduler surface without executing the unfinished
# SDXL runtime model. Tensor casts in the Python reference are represented as
# contracts. Scalar schedule helpers run on the host; no persistent F32 tensor
# boundary is introduced here.

from std.math import sqrt

from serenity_trainer.modelSetup.mixin.ModelSetupNoiseMixin import (
    _get_timestep_discrete_host as _ts_host,
)
from serenity_trainer.util.enum.ModelType import (
    MODEL_TYPE_STABLE_DIFFUSION_XL_10_BASE,
    MODEL_TYPE_STABLE_DIFFUSION_XL_10_BASE_INPAINTING,
)
from serenity_trainer.util.enum.TrainingMethod import TM_LORA


comptime SDXL_MODEL_TYPE_NAME = "STABLE_DIFFUSION_XL_10_BASE"
comptime SDXL_INPAINT_MODEL_TYPE_NAME = "STABLE_DIFFUSION_XL_10_BASE_INPAINTING"
comptime SDXL_NUM_TRAIN_TIMESTEPS = 1000
comptime SDXL_IMAGE_BASIS_BATCH = 1

comptime SDXL_LAYER_PRESET_ATTN_MLP = "attn-mlp"
comptime SDXL_LAYER_PRESET_ATTN_ONLY = "attn-only"
comptime SDXL_LAYER_PRESET_FULL = "full"

comptime SDXL_LOSS_TYPE_TARGET = "target"
comptime SDXL_PREDICTION_EPSILON = "epsilon"
comptime SDXL_PREDICTION_V = "v_prediction"
comptime SDXL_PREDICT_KEY_LOSS_TYPE = "loss_type"
comptime SDXL_PREDICT_KEY_TIMESTEP = "timestep"
comptime SDXL_PREDICT_KEY_PREDICTED = "predicted"
comptime SDXL_PREDICT_KEY_TARGET = "target"
comptime SDXL_PREDICT_KEY_PREDICTION_TYPE = "prediction_type"

comptime SDXL_PART_UNET = "unet"
comptime SDXL_PART_TEXT_ENCODER_1 = "text_encoder_1"
comptime SDXL_PART_TEXT_ENCODER_2 = "text_encoder_2"
comptime SDXL_PART_VAE = "vae"
comptime SDXL_PART_LORA = "lora"
comptime SDXL_PART_EMBEDDING = "embedding"
comptime SDXL_PART_UNET_LORA = "unet_lora"

comptime SDXL_DTYPE_PART_TEXT_ENCODER = "text_encoder"
comptime SDXL_DTYPE_PART_TEXT_ENCODER_2 = "text_encoder_2"


def sdxl_setup_model_types() -> List[Int]:
    var model_types = List[Int]()
    model_types.append(MODEL_TYPE_STABLE_DIFFUSION_XL_10_BASE)
    model_types.append(MODEL_TYPE_STABLE_DIFFUSION_XL_10_BASE_INPAINTING)
    return model_types^


def sdxl_layer_preset_filters(preset: String) raises -> List[String]:
    """Serenity BaseStableDiffusionXLSetup.LAYER_PRESETS."""
    var filters = List[String]()
    if preset == SDXL_LAYER_PRESET_ATTN_MLP:
        filters.append("attentions")
    elif preset == SDXL_LAYER_PRESET_ATTN_ONLY:
        filters.append("attn")
    elif preset == SDXL_LAYER_PRESET_FULL:
        pass
    else:
        raise Error(String("unknown SDXL layer preset: ") + preset)
    return filters^


def sdxl_predict_required_batch_fields() -> List[String]:
    """Fields read unconditionally by Serenity BaseStableDiffusionXLSetup.predict."""
    var fields = List[String]()
    fields.append("latent_image")
    fields.append("tokens_1")
    fields.append("tokens_2")
    fields.append("original_resolution")
    fields.append("crop_offset")
    fields.append("crop_resolution")
    return fields^


def sdxl_predict_cached_text_batch_fields() -> List[String]:
    """Fields consumed when the matching text encoder is frozen/cached."""
    var fields = List[String]()
    fields.append("text_encoder_1_hidden_state")
    fields.append("text_encoder_2_hidden_state")
    fields.append("text_encoder_2_pooled_state")
    return fields^


def sdxl_predict_token_mask_fields_not_consumed() -> List[String]:
    """Tokenize emits masks, but SDXL predict does not pass them to CLIP encoders."""
    var fields = List[String]()
    fields.append("tokens_mask_1")
    fields.append("tokens_mask_2")
    return fields^


def sdxl_predict_conditional_latent_fields() -> List[String]:
    """Used only when the model type has both mask and conditioning image inputs."""
    var fields = List[String]()
    fields.append("latent_mask")
    fields.append("latent_conditioning_image")
    return fields^


def sdxl_predict_output_fields() -> List[String]:
    """Fields written in Serenity model_output_data."""
    var fields = List[String]()
    fields.append(SDXL_PREDICT_KEY_LOSS_TYPE)
    fields.append(SDXL_PREDICT_KEY_TIMESTEP)
    fields.append(SDXL_PREDICT_KEY_PREDICTED)
    fields.append(SDXL_PREDICT_KEY_TARGET)
    fields.append(SDXL_PREDICT_KEY_PREDICTION_TYPE)
    return fields^


def sdxl_setup_optimization_checkpoint_parts() -> List[String]:
    var parts = List[String]()
    parts.append(SDXL_PART_UNET)
    parts.append(SDXL_PART_TEXT_ENCODER_1)
    parts.append(SDXL_PART_TEXT_ENCODER_2)
    return parts^


def sdxl_setup_optimization_checkpoint_helpers() -> List[String]:
    var helpers = List[String]()
    helpers.append("model.unet.enable_gradient_checkpointing")
    helpers.append("enable_checkpointing_for_basic_transformer_blocks:unet")
    helpers.append("enable_checkpointing_for_clip_encoder_layers:text_encoder_1")
    helpers.append("enable_checkpointing_for_clip_encoder_layers:text_encoder_2")
    return helpers^


def sdxl_setup_optimization_force_circular_padding_parts(
    has_unet_lora: Bool
) -> List[String]:
    var parts = List[String]()
    parts.append(SDXL_PART_VAE)
    parts.append(SDXL_PART_UNET)
    if has_unet_lora:
        parts.append(SDXL_PART_UNET_LORA)
    return parts^


def sdxl_setup_optimization_quantized_parts() -> List[String]:
    var parts = List[String]()
    parts.append(SDXL_PART_TEXT_ENCODER_1)
    parts.append(SDXL_PART_TEXT_ENCODER_2)
    parts.append(SDXL_PART_VAE)
    parts.append(SDXL_PART_UNET)
    return parts^


def sdxl_autocast_weight_dtype_parts(
    training_method: Int, train_any_embedding: Bool
) -> List[String]:
    var parts = List[String]()
    parts.append(SDXL_PART_UNET)
    parts.append(SDXL_DTYPE_PART_TEXT_ENCODER)
    parts.append(SDXL_DTYPE_PART_TEXT_ENCODER_2)
    parts.append(SDXL_PART_VAE)
    if training_method == TM_LORA:
        parts.append(SDXL_PART_LORA)
    if train_any_embedding:
        parts.append(SDXL_PART_EMBEDDING)
    return parts^


def sdxl_vae_autocast_weight_dtype_parts() -> List[String]:
    var parts = List[String]()
    parts.append(SDXL_PART_VAE)
    return parts^


def sdxl_quantization_dtype_source(part: String) -> String:
    if part == SDXL_PART_VAE:
        return "model.vae_train_dtype"
    return "model.train_dtype"


def sdxl_scale_latents_expression() -> String:
    return "latent_image * vae.config['scaling_factor']"


def sdxl_scale_conditioning_latents_expression() -> String:
    return "latent_conditioning_image * vae.config['scaling_factor']"


def sdxl_noisy_latent_expression() -> String:
    return "scaled_latent_image * sqrt_alphas_cumprod[t] + latent_noise * sqrt_one_minus_alphas_cumprod[t]"


def sdxl_latent_input_expression() -> String:
    return "concat([scaled_noisy_latent_image, latent_mask, scaled_latent_conditioning_image], dim=1) only for inpainting; otherwise scaled_noisy_latent_image"


def sdxl_add_time_ids_expression() -> String:
    return "stack([original_height, original_width, crop_top, crop_left, target_height, target_width], dim=1).to(dtype=scaled_noisy_latent_image.dtype)"


def sdxl_unet_sample_expression() -> String:
    return "latent_input.to(dtype=model.train_dtype.torch_dtype())"


def sdxl_unet_timestep_expression() -> String:
    return "raw discrete timestep tensor"


def sdxl_unet_encoder_hidden_states_expression() -> String:
    return "combined text_encoder_output.to(dtype=model.train_dtype.torch_dtype())"


def sdxl_unet_added_cond_kwargs_expression() -> String:
    return "{'text_embeds': pooled_text_encoder_2_output, 'time_ids': add_time_ids}"


def sdxl_epsilon_target_expression() -> String:
    return "latent_noise"


def sdxl_v_prediction_target_expression() -> String:
    return "noise_scheduler.get_velocity(scaled_latent_image, latent_noise, timestep)"


def sdxl_scheduler_prediction_types() -> List[String]:
    var names = List[String]()
    names.append(SDXL_PREDICTION_EPSILON)
    names.append(SDXL_PREDICTION_V)
    return names^


def sdxl_alpha_cumprod_from_betas(betas: List[Float32], timestep: Int) raises -> Float32:
    if timestep < 0 or timestep >= len(betas):
        raise Error("timestep is outside the beta schedule")
    var alpha = Float32(1.0)
    for i in range(timestep + 1):
        alpha = alpha * (Float32(1.0) - betas[i])
    return alpha


def sdxl_sqrt_alpha_cumprod_from_betas(
    betas: List[Float32], timestep: Int
) raises -> Float32:
    return sqrt(sdxl_alpha_cumprod_from_betas(betas, timestep))


def sdxl_noise_sigma_from_betas(
    betas: List[Float32], timestep: Int
) raises -> Float32:
    return sqrt(
        Float32(1.0) - sdxl_alpha_cumprod_from_betas(betas, timestep)
    )


def sdxl_scheduler_sigma_from_betas(
    betas: List[Float32], timestep: Int
) raises -> Float32:
    var alpha = sdxl_alpha_cumprod_from_betas(betas, timestep)
    if alpha <= Float32(0.0):
        raise Error("alpha_cumprod must be positive")
    return sqrt((Float32(1.0) - alpha) / alpha)


def sdxl_snr_from_betas(betas: List[Float32], timestep: Int) raises -> Float32:
    var alpha = sdxl_alpha_cumprod_from_betas(betas, timestep)
    var denom = Float32(1.0) - alpha
    if denom <= Float32(0.0):
        raise Error("1 - alpha_cumprod must be positive")
    return alpha / denom


def sdxl_model_t_from_timestep(t: Int) -> Float32:
    # SDXL UNet receives the raw discrete timestep tensor.
    return Float32(t)


def sdxl_get_timestep_discrete(
    num_train_timesteps: Int,
    deterministic: Bool,
    seed: UInt64,
    timestep_distribution: Int,
    min_noising_strength: Float32,
    max_noising_strength: Float32,
    noising_weight: Float32,
    noising_bias: Float32,
    timestep_shift: Float32,
) raises -> Int:
    if deterministic:
        return Int(Float64(num_train_timesteps) * Float64(0.5)) - 1

    var host = _ts_host(
        num_train_timesteps,
        SDXL_IMAGE_BASIS_BATCH,
        timestep_distribution,
        Float64(min_noising_strength),
        Float64(max_noising_strength),
        Float64(noising_weight),
        Float64(noising_bias),
        Float64(timestep_shift),
        List[Float64](),
        seed,
    )
    return host.values[0]


def sdxl_dtype_boundary_caveats() -> List[String]:
    var caveats = List[String]()
    caveats.append("Storage dtype is preserved at tensor boundaries; this build-only surface does not create runtime tensors.")
    caveats.append("Python _add_noise_discrete computes coefficients in scheduler dtype and returns scaled_noisy_latent_image to the original latent dtype.")
    caveats.append("UNet sample and text hidden states are cast to model.train_dtype at call time; this is an execution input cast, not persistent model storage.")
    caveats.append("add_time_ids are cast to scaled_noisy_latent_image dtype and passed through added_cond_kwargs.")
    caveats.append("VAE autocast disables FP16 fallback via disable_fp16_autocast_context; VAE quantization uses model.vae_train_dtype.")
    caveats.append("Loss code casts predicted/target tensors to F32 for compute only; loss scalars/statistics may be F32.")
    caveats.append("Serenity RNG uses one torch.Generator for timestep and noise draws; Mojo host RNG helpers do not claim same-seed numeric parity.")
    return caveats^


def sdxl_unsupported_runtime_paths() -> List[String]:
    var paths = List[String]()
    paths.append("predict runtime tensor path is not implemented in this setup surface")
    paths.append("MGDS data pipeline execution is not implemented in this setup surface")
    paths.append("Serenity execution, parity, speed, and numeric claims are owned by later runtime work")
    return paths^


struct SDXLPredictContract(Movable):
    var required_batch_fields: List[String]
    var cached_text_batch_fields: List[String]
    var token_mask_fields_not_consumed: List[String]
    var conditional_latent_fields: List[String]
    var output_fields: List[String]
    var loss_type: String
    var prediction_types: List[String]
    var scale_latents_expression: String
    var scale_conditioning_latents_expression: String
    var noisy_latent_expression: String
    var latent_input_expression: String
    var add_time_ids_expression: String
    var unet_sample_expression: String
    var unet_timestep_expression: String
    var unet_encoder_hidden_states_expression: String
    var unet_added_cond_kwargs_expression: String
    var epsilon_target_expression: String
    var v_prediction_target_expression: String

    def __init__(out self):
        self.required_batch_fields = sdxl_predict_required_batch_fields()
        self.cached_text_batch_fields = sdxl_predict_cached_text_batch_fields()
        self.token_mask_fields_not_consumed = (
            sdxl_predict_token_mask_fields_not_consumed()
        )
        self.conditional_latent_fields = sdxl_predict_conditional_latent_fields()
        self.output_fields = sdxl_predict_output_fields()
        self.loss_type = SDXL_LOSS_TYPE_TARGET
        self.prediction_types = sdxl_scheduler_prediction_types()
        self.scale_latents_expression = sdxl_scale_latents_expression()
        self.scale_conditioning_latents_expression = (
            sdxl_scale_conditioning_latents_expression()
        )
        self.noisy_latent_expression = sdxl_noisy_latent_expression()
        self.latent_input_expression = sdxl_latent_input_expression()
        self.add_time_ids_expression = sdxl_add_time_ids_expression()
        self.unet_sample_expression = sdxl_unet_sample_expression()
        self.unet_timestep_expression = sdxl_unet_timestep_expression()
        self.unet_encoder_hidden_states_expression = (
            sdxl_unet_encoder_hidden_states_expression()
        )
        self.unet_added_cond_kwargs_expression = (
            sdxl_unet_added_cond_kwargs_expression()
        )
        self.epsilon_target_expression = sdxl_epsilon_target_expression()
        self.v_prediction_target_expression = sdxl_v_prediction_target_expression()


struct SDXLOptimizationContract(Movable):
    var checkpoint_parts: List[String]
    var checkpoint_helpers: List[String]
    var force_circular_padding_parts: List[String]
    var quantized_parts: List[String]
    var autocast_weight_dtype_parts: List[String]
    var vae_autocast_weight_dtype_parts: List[String]
    var disables_fp16_vae_autocast: Bool
    var dtype_boundary_caveats: List[String]

    def __init__(
        out self,
        training_method: Int,
        train_any_embedding: Bool = False,
        has_unet_lora: Bool = False,
    ):
        self.checkpoint_parts = sdxl_setup_optimization_checkpoint_parts()
        self.checkpoint_helpers = sdxl_setup_optimization_checkpoint_helpers()
        self.force_circular_padding_parts = (
            sdxl_setup_optimization_force_circular_padding_parts(has_unet_lora)
        )
        self.quantized_parts = sdxl_setup_optimization_quantized_parts()
        self.autocast_weight_dtype_parts = sdxl_autocast_weight_dtype_parts(
            training_method, train_any_embedding
        )
        self.vae_autocast_weight_dtype_parts = sdxl_vae_autocast_weight_dtype_parts()
        self.disables_fp16_vae_autocast = True
        self.dtype_boundary_caveats = sdxl_dtype_boundary_caveats()


struct SDXLTrainDevicePlan(Copyable, Movable, ImplicitlyCopyable):
    var text_encoder_1_on_train_device: Bool
    var text_encoder_2_on_train_device: Bool
    var vae_on_train_device: Bool
    var unet_on_train_device: Bool
    var text_encoder_1_train_mode: Bool
    var text_encoder_2_train_mode: Bool
    var vae_train_mode: Bool
    var unet_train_mode: Bool

    def __init__(
        out self,
        latent_caching: Bool,
        text_encoder_1_train_device_signal: Bool,
        text_encoder_2_train_device_signal: Bool,
        text_encoder_1_train: Bool,
        text_encoder_2_train: Bool,
        unet_train: Bool,
        vae_train_mode: Bool,
    ):
        self.text_encoder_1_on_train_device = (
            text_encoder_1_train_device_signal or not latent_caching
        )
        self.text_encoder_2_on_train_device = (
            text_encoder_2_train_device_signal or not latent_caching
        )
        self.vae_on_train_device = not latent_caching
        self.unet_on_train_device = True
        self.text_encoder_1_train_mode = text_encoder_1_train
        self.text_encoder_2_train_mode = text_encoder_2_train
        self.vae_train_mode = vae_train_mode
        self.unet_train_mode = unet_train


struct SDXLTextCachingPlan(Copyable, Movable, ImplicitlyCopyable):
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
        self.move_text_encoder_1_to_train_device = (
            not train_text_encoder_or_embedding
        )
        self.move_text_encoder_2_to_train_device = (
            not train_text_encoder_2_or_embedding
        )
        self.set_eval_mode = True
        self.run_torch_gc = True


struct SDXLUnsupportedPaths(Movable):
    var paths: List[String]

    def __init__(out self):
        self.paths = sdxl_unsupported_runtime_paths()


def sdxl_predict_contract() -> SDXLPredictContract:
    return SDXLPredictContract()


def sdxl_optimization_contract(
    training_method: Int,
    train_any_embedding: Bool = False,
    has_unet_lora: Bool = False,
) -> SDXLOptimizationContract:
    return SDXLOptimizationContract(
        training_method, train_any_embedding, has_unet_lora
    )


def sdxl_train_device_plan(
    latent_caching: Bool,
    text_encoder_1_train_device_signal: Bool,
    text_encoder_2_train_device_signal: Bool,
    text_encoder_1_train: Bool,
    text_encoder_2_train: Bool,
    unet_train: Bool,
    vae_train_mode: Bool,
) -> SDXLTrainDevicePlan:
    return SDXLTrainDevicePlan(
        latent_caching,
        text_encoder_1_train_device_signal,
        text_encoder_2_train_device_signal,
        text_encoder_1_train,
        text_encoder_2_train,
        unet_train,
        vae_train_mode,
    )


def sdxl_text_caching_plan(
    train_text_encoder_or_embedding: Bool,
    train_text_encoder_2_or_embedding: Bool,
) -> SDXLTextCachingPlan:
    return SDXLTextCachingPlan(
        train_text_encoder_or_embedding,
        train_text_encoder_2_or_embedding,
    )


struct BaseStableDiffusionXLSetup(Movable):
    var debug_mode: Bool
    var model_types: List[Int]

    def __init__(out self, debug_mode: Bool = False):
        self.debug_mode = debug_mode
        self.model_types = sdxl_setup_model_types()

    def layer_preset_filters(self, preset: String) raises -> List[String]:
        return sdxl_layer_preset_filters(preset)

    def predict_contract(self) -> SDXLPredictContract:
        return sdxl_predict_contract()

    def optimization_contract(
        self,
        training_method: Int,
        train_any_embedding: Bool = False,
        has_unet_lora: Bool = False,
    ) -> SDXLOptimizationContract:
        return sdxl_optimization_contract(
            training_method, train_any_embedding, has_unet_lora
        )

    def train_device_plan(
        self,
        latent_caching: Bool,
        text_encoder_1_train_device_signal: Bool,
        text_encoder_2_train_device_signal: Bool,
        text_encoder_1_train: Bool,
        text_encoder_2_train: Bool,
        unet_train: Bool,
        vae_train_mode: Bool,
    ) -> SDXLTrainDevicePlan:
        return sdxl_train_device_plan(
            latent_caching,
            text_encoder_1_train_device_signal,
            text_encoder_2_train_device_signal,
            text_encoder_1_train,
            text_encoder_2_train,
            unet_train,
            vae_train_mode,
        )

    def calculate_loss_consumes_betas(self) -> Bool:
        return True

    def calculate_loss_reduction(self) -> String:
        return "mean"

    def prepare_text_caching_plan(
        self,
        train_text_encoder_or_embedding: Bool,
        train_text_encoder_2_or_embedding: Bool,
    ) -> SDXLTextCachingPlan:
        return sdxl_text_caching_plan(
            train_text_encoder_or_embedding,
            train_text_encoder_2_or_embedding,
        )

    def runtime_predict_implemented(self) -> Bool:
        return False

    def unsupported_runtime_paths(self) -> SDXLUnsupportedPaths:
        return SDXLUnsupportedPaths()

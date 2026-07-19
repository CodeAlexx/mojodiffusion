# BaseStableDiffusion3Setup.mojo - build-only SD3/SD3.5 setup contract.
#
# Source of truth:
#   /home/alex/Serenity/modules/modelSetup/BaseStableDiffusion3Setup.py
#   /home/alex/Serenity/modules/model/StableDiffusion3Model.py
#
# This file records the Serenity setup/predict surface needed by later runtime
# and parity work. It intentionally does not execute the unfinished SD3 runtime
# model. Tensor dtype casts in the Python reference are represented as contracts;
# only scalar schedule helpers live here, so no F32 tensor boundary is introduced.

from serenity_trainer.modelSetup.mixin.ModelSetupNoiseMixin import (
    _get_timestep_discrete_host as _ts_host,
)
from serenity_trainer.util.enum.ModelType import (
    MODEL_TYPE_STABLE_DIFFUSION_3,
    MODEL_TYPE_STABLE_DIFFUSION_35,
)
from serenity_trainer.util.enum.TrainingMethod import TM_LORA


comptime SD3_MODEL_TYPE_NAME = "STABLE_DIFFUSION_3"
comptime SD35_MODEL_TYPE_NAME = "STABLE_DIFFUSION_35"
comptime SD3_NUM_TRAIN_TIMESTEPS = 1000
comptime SD3_IMAGE_BASIS_BATCH = 1
comptime SD3_TOKENIZER_FALLBACK_MAX_TOKENS = 77

comptime SD3_LAYER_PRESET_ATTN_ONLY = "attn-only"
comptime SD3_LAYER_PRESET_BLOCKS = "blocks"
comptime SD3_LAYER_PRESET_FULL = "full"

comptime SD3_LOSS_TYPE_TARGET = "target"
comptime SD3_PREDICT_KEY_LOSS_TYPE = "loss_type"
comptime SD3_PREDICT_KEY_TIMESTEP = "timestep"
comptime SD3_PREDICT_KEY_PREDICTED = "predicted"
comptime SD3_PREDICT_KEY_TARGET = "target"

comptime SD3_PART_TRANSFORMER = "transformer"
comptime SD3_PART_TEXT_ENCODER_1 = "text_encoder_1"
comptime SD3_PART_TEXT_ENCODER_2 = "text_encoder_2"
comptime SD3_PART_TEXT_ENCODER_3 = "text_encoder_3"
comptime SD3_PART_VAE = "vae"
comptime SD3_PART_LORA = "lora"
comptime SD3_PART_EMBEDDING = "embedding"

comptime SD3_DTYPE_PART_TEXT_ENCODER = "text_encoder"
comptime SD3_DTYPE_PART_TEXT_ENCODER_2 = "text_encoder_2"
comptime SD3_DTYPE_PART_TEXT_ENCODER_3 = "text_encoder_3"


def sd3_setup_model_types() -> List[Int]:
    var model_types = List[Int]()
    model_types.append(MODEL_TYPE_STABLE_DIFFUSION_3)
    model_types.append(MODEL_TYPE_STABLE_DIFFUSION_35)
    return model_types^


def sd3_layer_preset_filters(preset: String) raises -> List[String]:
    """Serenity BaseStableDiffusion3Setup.LAYER_PRESETS."""
    var filters = List[String]()
    if preset == SD3_LAYER_PRESET_ATTN_ONLY:
        filters.append("attn")
    elif preset == SD3_LAYER_PRESET_BLOCKS:
        filters.append("transformer_block")
    elif preset == SD3_LAYER_PRESET_FULL:
        pass
    else:
        raise Error(String("unknown SD3 layer preset: ") + preset)
    return filters^


def sd3_predict_required_batch_fields() -> List[String]:
    """Fields read unconditionally by Serenity BaseStableDiffusion3Setup.predict."""
    var fields = List[String]()
    fields.append("latent_image")
    return fields^


def sd3_predict_text_batch_fields() -> List[String]:
    """Text/cache fields passed through StableDiffusion3Model.encode_text."""
    var fields = List[String]()
    fields.append("tokens_1")
    fields.append("tokens_2")
    fields.append("tokens_3")
    fields.append("tokens_mask_1")
    fields.append("tokens_mask_2")
    fields.append("tokens_mask_3")
    fields.append("text_encoder_1_hidden_state")
    fields.append("text_encoder_1_pooled_state")
    fields.append("text_encoder_2_hidden_state")
    fields.append("text_encoder_2_pooled_state")
    fields.append("text_encoder_3_hidden_state")
    return fields^


def sd3_predict_conditional_latent_fields() -> List[String]:
    """Only used when both has_mask_input() and has_conditioning_image_input()."""
    var fields = List[String]()
    fields.append("latent_mask")
    fields.append("latent_conditioning_image")
    return fields^


def sd3_predict_output_fields() -> List[String]:
    """Fields written in Serenity model_output_data."""
    var fields = List[String]()
    fields.append(SD3_PREDICT_KEY_LOSS_TYPE)
    fields.append(SD3_PREDICT_KEY_TIMESTEP)
    fields.append(SD3_PREDICT_KEY_PREDICTED)
    fields.append(SD3_PREDICT_KEY_TARGET)
    return fields^


def sd3_setup_optimization_checkpoint_parts() -> List[String]:
    var parts = List[String]()
    parts.append(SD3_PART_TRANSFORMER)
    parts.append(SD3_PART_TEXT_ENCODER_1)
    parts.append(SD3_PART_TEXT_ENCODER_2)
    parts.append(SD3_PART_TEXT_ENCODER_3)
    return parts^


def sd3_setup_optimization_checkpoint_helpers() -> List[String]:
    var helpers = List[String]()
    helpers.append("enable_checkpointing_for_stable_diffusion_3_transformer")
    helpers.append("enable_checkpointing_for_clip_encoder_layers:text_encoder_1")
    helpers.append("enable_checkpointing_for_clip_encoder_layers:text_encoder_2")
    helpers.append("enable_checkpointing_for_t5_encoder_layers:text_encoder_3")
    return helpers^


def sd3_setup_optimization_quantized_parts() -> List[String]:
    var parts = List[String]()
    parts.append(SD3_PART_TEXT_ENCODER_1)
    parts.append(SD3_PART_TEXT_ENCODER_2)
    parts.append(SD3_PART_TEXT_ENCODER_3)
    parts.append(SD3_PART_VAE)
    parts.append(SD3_PART_TRANSFORMER)
    return parts^


def sd3_autocast_weight_dtype_parts(
    training_method: Int, train_any_embedding: Bool
) -> List[String]:
    var parts = List[String]()
    parts.append(SD3_PART_TRANSFORMER)
    parts.append(SD3_DTYPE_PART_TEXT_ENCODER)
    parts.append(SD3_DTYPE_PART_TEXT_ENCODER_2)
    parts.append(SD3_DTYPE_PART_TEXT_ENCODER_3)
    parts.append(SD3_PART_VAE)
    if training_method == TM_LORA:
        parts.append(SD3_PART_LORA)
    if train_any_embedding:
        parts.append(SD3_PART_EMBEDDING)
    return parts^


def sd3_text_encoder_3_autocast_weight_dtype_parts(
    training_method: Int, train_any_embedding: Bool
) -> List[String]:
    var parts = List[String]()
    parts.append(SD3_DTYPE_PART_TEXT_ENCODER_3)
    if training_method == TM_LORA:
        parts.append(SD3_PART_LORA)
    if train_any_embedding:
        parts.append(SD3_PART_EMBEDDING)
    return parts^


def sd3_flow_target_expression() -> String:
    return "latent_noise - scaled_latent_image"


def sd3_noisy_latent_expression() -> String:
    return "latent_noise * sigma + scaled_latent_image * (1 - sigma)"


def sd3_scale_latents_expression() -> String:
    return "(latent_image - vae.config['shift_factor']) * vae.config['scaling_factor']"


def sd3_scale_conditioning_latents_expression() -> String:
    return "(latent_conditioning_image - vae.config['shift_factor']) * vae.config['scaling_factor']"


def sd3_latent_input_expression() -> String:
    return "concat([scaled_noisy_latent_image, latent_mask, scaled_latent_conditioning_image], dim=1) only when mask+conditioning inputs are enabled; otherwise scaled_noisy_latent_image"


def sd3_transformer_timestep_expression() -> String:
    return "timestep"


def sd3_transformer_hidden_states_expression() -> String:
    return "latent_input.to(dtype=model.train_dtype.torch_dtype())"


def sd3_transformer_encoder_hidden_states_expression() -> String:
    return "combined text_encoder_output.to(dtype=model.train_dtype.torch_dtype())"


def sd3_transformer_pooled_projection_expression() -> String:
    return "pooled_text_encoder_output.to(dtype=model.train_dtype.torch_dtype())"


def sd3_predicted_expression() -> String:
    return "model.transformer(..., return_dict=True).sample"


def sd3_text_encoder_combination_expression() -> String:
    return "concat(CLIP-L, CLIP-G) on hidden dim, pad to T5 hidden dim, then concat with T5 on sequence dim; pooled = concat(CLIP-L pooled, CLIP-G pooled)"


def sd3_sigma_from_timestep(
    t: Int, num_timesteps: Int = SD3_NUM_TRAIN_TIMESTEPS
) -> Float32:
    # ModelSetupFlowMatchingMixin._add_noise_discrete:
    # sigma[t] = arange(1, N+1)[t] / N.
    return Float32(t + 1) / Float32(num_timesteps)


def sd3_timestep_from_sigma(
    sigma: Float32, num_timesteps: Int = SD3_NUM_TRAIN_TIMESTEPS
) -> Int:
    var idx = Int(sigma * Float32(num_timesteps) + Float32(0.5)) - 1
    if idx < 0:
        return 0
    if idx >= num_timesteps:
        return num_timesteps - 1
    return idx


def sd3_model_t_from_timestep(t: Int) -> Float32:
    # BaseStableDiffusion3Setup.predict passes the raw discrete timestep tensor.
    return Float32(t)


def sd3_get_timestep_discrete(
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
        SD3_IMAGE_BASIS_BATCH,
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


def sd3_unsupported_runtime_paths() -> List[String]:
    var paths = List[String]()
    paths.append("predict runtime tensor path is not implemented in this setup surface")
    paths.append("MGDS data pipeline execution is not implemented in this setup surface")
    paths.append("Serenity execution, parity, speed, and numeric claims are owned by the main loop")
    return paths^


struct SD3PredictContract(Movable):
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
    var transformer_timestep_expression: String
    var transformer_hidden_states_expression: String
    var transformer_encoder_hidden_states_expression: String
    var transformer_pooled_projection_expression: String
    var predicted_expression: String
    var text_encoder_combination_expression: String

    def __init__(out self):
        self.required_batch_fields = sd3_predict_required_batch_fields()
        self.text_batch_fields = sd3_predict_text_batch_fields()
        self.conditional_latent_fields = sd3_predict_conditional_latent_fields()
        self.output_fields = sd3_predict_output_fields()
        self.loss_type = SD3_LOSS_TYPE_TARGET
        self.target_expression = sd3_flow_target_expression()
        self.noisy_latent_expression = sd3_noisy_latent_expression()
        self.scale_latents_expression = sd3_scale_latents_expression()
        self.scale_conditioning_latents_expression = (
            sd3_scale_conditioning_latents_expression()
        )
        self.latent_input_expression = sd3_latent_input_expression()
        self.transformer_timestep_expression = sd3_transformer_timestep_expression()
        self.transformer_hidden_states_expression = (
            sd3_transformer_hidden_states_expression()
        )
        self.transformer_encoder_hidden_states_expression = (
            sd3_transformer_encoder_hidden_states_expression()
        )
        self.transformer_pooled_projection_expression = (
            sd3_transformer_pooled_projection_expression()
        )
        self.predicted_expression = sd3_predicted_expression()
        self.text_encoder_combination_expression = (
            sd3_text_encoder_combination_expression()
        )


struct SD3OptimizationContract(Movable):
    var checkpoint_parts: List[String]
    var checkpoint_helpers: List[String]
    var quantized_parts: List[String]
    var autocast_weight_dtype_parts: List[String]
    var text_encoder_3_autocast_weight_dtype_parts: List[String]
    var disables_fp16_text_encoder_3_autocast: Bool

    def __init__(
        out self, training_method: Int, train_any_embedding: Bool = False
    ):
        self.checkpoint_parts = sd3_setup_optimization_checkpoint_parts()
        self.checkpoint_helpers = sd3_setup_optimization_checkpoint_helpers()
        self.quantized_parts = sd3_setup_optimization_quantized_parts()
        self.autocast_weight_dtype_parts = sd3_autocast_weight_dtype_parts(
            training_method, train_any_embedding
        )
        self.text_encoder_3_autocast_weight_dtype_parts = (
            sd3_text_encoder_3_autocast_weight_dtype_parts(
                training_method, train_any_embedding
            )
        )
        self.disables_fp16_text_encoder_3_autocast = True


struct SD3TrainDevicePlan(Copyable, Movable, ImplicitlyCopyable):
    var text_encoder_1_on_train_device: Bool
    var text_encoder_2_on_train_device: Bool
    var text_encoder_3_on_train_device: Bool
    var vae_on_train_device: Bool
    var transformer_on_train_device: Bool
    var text_encoder_1_train_mode: Bool
    var text_encoder_2_train_mode: Bool
    var text_encoder_3_train_mode: Bool
    var vae_train_mode: Bool
    var transformer_train_mode: Bool

    def __init__(
        out self,
        latent_caching: Bool,
        train_text_encoder_or_embedding: Bool,
        train_text_encoder_2_or_embedding: Bool,
        train_text_encoder_3_or_embedding: Bool,
        text_encoder_1_train: Bool,
        text_encoder_2_train: Bool,
        text_encoder_3_train: Bool,
        transformer_train: Bool,
    ):
        self.text_encoder_1_on_train_device = (
            train_text_encoder_or_embedding or not latent_caching
        )
        self.text_encoder_2_on_train_device = (
            train_text_encoder_2_or_embedding or not latent_caching
        )
        self.text_encoder_3_on_train_device = (
            train_text_encoder_3_or_embedding or not latent_caching
        )
        self.vae_on_train_device = not latent_caching
        self.transformer_on_train_device = True
        self.text_encoder_1_train_mode = text_encoder_1_train
        self.text_encoder_2_train_mode = text_encoder_2_train
        self.text_encoder_3_train_mode = text_encoder_3_train
        self.vae_train_mode = False
        self.transformer_train_mode = transformer_train


struct SD3TextCachingPlan(Copyable, Movable, ImplicitlyCopyable):
    var move_model_to_temp_device: Bool
    var move_text_encoder_1_to_train_device: Bool
    var move_text_encoder_2_to_train_device: Bool
    var move_text_encoder_3_to_train_device: Bool
    var set_eval_mode: Bool
    var run_torch_gc: Bool

    def __init__(
        out self,
        train_text_encoder_or_embedding: Bool,
        train_text_encoder_2_or_embedding: Bool,
        train_text_encoder_3_or_embedding: Bool,
    ):
        self.move_model_to_temp_device = True
        self.move_text_encoder_1_to_train_device = not train_text_encoder_or_embedding
        self.move_text_encoder_2_to_train_device = (
            not train_text_encoder_2_or_embedding
        )
        self.move_text_encoder_3_to_train_device = (
            not train_text_encoder_3_or_embedding
        )
        self.set_eval_mode = True
        self.run_torch_gc = True


struct SD3UnsupportedPaths(Movable):
    var paths: List[String]

    def __init__(out self):
        self.paths = sd3_unsupported_runtime_paths()


def sd3_predict_contract() -> SD3PredictContract:
    return SD3PredictContract()


def sd3_optimization_contract(
    training_method: Int, train_any_embedding: Bool = False
) -> SD3OptimizationContract:
    return SD3OptimizationContract(training_method, train_any_embedding)


def sd3_train_device_plan(
    latent_caching: Bool,
    train_text_encoder_or_embedding: Bool,
    train_text_encoder_2_or_embedding: Bool,
    train_text_encoder_3_or_embedding: Bool,
    text_encoder_1_train: Bool,
    text_encoder_2_train: Bool,
    text_encoder_3_train: Bool,
    transformer_train: Bool,
) -> SD3TrainDevicePlan:
    return SD3TrainDevicePlan(
        latent_caching,
        train_text_encoder_or_embedding,
        train_text_encoder_2_or_embedding,
        train_text_encoder_3_or_embedding,
        text_encoder_1_train,
        text_encoder_2_train,
        text_encoder_3_train,
        transformer_train,
    )


def sd3_text_caching_plan(
    train_text_encoder_or_embedding: Bool,
    train_text_encoder_2_or_embedding: Bool,
    train_text_encoder_3_or_embedding: Bool,
) -> SD3TextCachingPlan:
    return SD3TextCachingPlan(
        train_text_encoder_or_embedding,
        train_text_encoder_2_or_embedding,
        train_text_encoder_3_or_embedding,
    )


struct BaseStableDiffusion3Setup(Movable):
    var debug_mode: Bool
    var model_types: List[Int]

    def __init__(out self, debug_mode: Bool = False):
        self.debug_mode = debug_mode
        self.model_types = sd3_setup_model_types()

    def layer_preset_filters(self, preset: String) raises -> List[String]:
        return sd3_layer_preset_filters(preset)

    def predict_contract(self) -> SD3PredictContract:
        return sd3_predict_contract()

    def optimization_contract(
        self, training_method: Int, train_any_embedding: Bool = False
    ) -> SD3OptimizationContract:
        return sd3_optimization_contract(training_method, train_any_embedding)

    def train_device_plan(
        self,
        latent_caching: Bool,
        train_text_encoder_or_embedding: Bool,
        train_text_encoder_2_or_embedding: Bool,
        train_text_encoder_3_or_embedding: Bool,
        text_encoder_1_train: Bool,
        text_encoder_2_train: Bool,
        text_encoder_3_train: Bool,
        transformer_train: Bool,
    ) -> SD3TrainDevicePlan:
        return sd3_train_device_plan(
            latent_caching,
            train_text_encoder_or_embedding,
            train_text_encoder_2_or_embedding,
            train_text_encoder_3_or_embedding,
            text_encoder_1_train,
            text_encoder_2_train,
            text_encoder_3_train,
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
        train_text_encoder_3_or_embedding: Bool,
    ) -> SD3TextCachingPlan:
        return sd3_text_caching_plan(
            train_text_encoder_or_embedding,
            train_text_encoder_2_or_embedding,
            train_text_encoder_3_or_embedding,
        )

    def runtime_predict_implemented(self) -> Bool:
        return False

    def unsupported_runtime_paths(self) -> SD3UnsupportedPaths:
        return SD3UnsupportedPaths()

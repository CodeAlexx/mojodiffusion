# BaseErnieSetup.mojo - build-only Ernie setup contract.
#
# Source of truth: /home/alex/Serenity/modules/modelSetup/BaseErnieSetup.py
# Related Serenity model helpers:
#   /home/alex/Serenity/modules/model/ErnieModel.py
#
# This file records the Serenity setup/predict surface needed by later parity
# gates. It intentionally does not execute the unfinished Ernie runtime model.
# Tensor storage dtype policy is preserved by keeping only scalar host helpers
# here; tensor dtype casts in the Python reference are represented as contracts.

from std.math import exp

from serenity_trainer.modelSetup.mixin.ModelSetupNoiseMixin import (
    _get_timestep_discrete_host as _ts_host,
)
from serenity_trainer.util.enum.TrainingMethod import TM_LORA


comptime ERNIE_NUM_TRAIN_TIMESTEPS = 1000
comptime ERNIE_PROMPT_MAX_LENGTH = 512
comptime ERNIE_HIDDEN_STATES_LAYER = -2
comptime ERNIE_LATENT_PATCH_FACTOR = 2
comptime ERNIE_IMAGE_BASIS_BATCH = 1

comptime ERNIE_LAYER_PRESET_ATTN_MLP = "attn-mlp"
comptime ERNIE_LAYER_PRESET_ATTN_ONLY = "attn-only"
comptime ERNIE_LAYER_PRESET_BLOCKS = "blocks"
comptime ERNIE_LAYER_PRESET_FULL = "full"

comptime ERNIE_LOSS_TYPE_TARGET = "target"
comptime ERNIE_PREDICT_KEY_LOSS_TYPE = "loss_type"
comptime ERNIE_PREDICT_KEY_TIMESTEP = "timestep"
comptime ERNIE_PREDICT_KEY_PREDICTED = "predicted"
comptime ERNIE_PREDICT_KEY_TARGET = "target"

comptime ERNIE_PART_TRANSFORMER = "transformer"
comptime ERNIE_PART_TEXT_ENCODER = "text_encoder"
comptime ERNIE_PART_VAE = "vae"
comptime ERNIE_PART_LORA = "lora"


def ernie_layer_preset_filters(preset: String) raises -> List[String]:
    """Serenity BaseErnieSetup.LAYER_PRESETS."""
    var filters = List[String]()
    if preset == ERNIE_LAYER_PRESET_ATTN_MLP:
        filters.append("self_attention")
        filters.append("mlp")
    elif preset == ERNIE_LAYER_PRESET_ATTN_ONLY:
        filters.append("self_attention")
    elif preset == ERNIE_LAYER_PRESET_BLOCKS:
        filters.append("layers")
    elif preset == ERNIE_LAYER_PRESET_FULL:
        pass
    else:
        raise Error(String("unknown Ernie layer preset: ") + preset)
    return filters^


def ernie_predict_required_batch_fields() -> List[String]:
    """Fields read directly by Serenity BaseErnieSetup.predict."""
    var fields = List[String]()
    fields.append("latent_image")
    return fields^


def ernie_predict_conditioning_batch_fields() -> List[String]:
    """Optional/conditional prompt fields passed through model.encode_text."""
    var fields = List[String]()
    fields.append("tokens")
    fields.append("tokens_mask")
    fields.append("text_encoder_hidden_state")
    return fields^


def ernie_predict_output_fields() -> List[String]:
    """Fields written in Serenity model_output_data."""
    var fields = List[String]()
    fields.append(ERNIE_PREDICT_KEY_LOSS_TYPE)
    fields.append(ERNIE_PREDICT_KEY_TIMESTEP)
    fields.append(ERNIE_PREDICT_KEY_PREDICTED)
    fields.append(ERNIE_PREDICT_KEY_TARGET)
    return fields^


def ernie_setup_optimization_checkpoint_parts() -> List[String]:
    var parts = List[String]()
    parts.append(ERNIE_PART_TRANSFORMER)
    return parts^


def ernie_setup_optimization_quantized_parts() -> List[String]:
    var parts = List[String]()
    parts.append(ERNIE_PART_TEXT_ENCODER)
    parts.append(ERNIE_PART_VAE)
    parts.append(ERNIE_PART_TRANSFORMER)
    return parts^


def ernie_autocast_weight_dtype_parts(training_method: Int) -> List[String]:
    var parts = List[String]()
    parts.append(ERNIE_PART_TRANSFORMER)
    parts.append(ERNIE_PART_TEXT_ENCODER)
    parts.append(ERNIE_PART_VAE)
    if training_method == TM_LORA:
        parts.append(ERNIE_PART_LORA)
    return parts^


def ernie_text_encoder_autocast_weight_dtype_parts(training_method: Int) -> List[String]:
    var parts = List[String]()
    parts.append(ERNIE_PART_TEXT_ENCODER)
    if training_method == TM_LORA:
        parts.append(ERNIE_PART_LORA)
    return parts^


def ernie_flow_target_expression() -> String:
    return "latent_noise - scaled_latent_image"


def ernie_noisy_latent_expression() -> String:
    return "latent_noise * sigma + scaled_latent_image * (1 - sigma)"


def ernie_patchify_latents_expression() -> String:
    return "[B,C,H,W] -> [B,C*4,H/2,W/2]"


def ernie_unpatchify_latents_expression() -> String:
    return "[B,C,H,W] -> [B,C/4,H*2,W*2]"


def ernie_scale_latents_expression() -> String:
    return "(latents - vae.bn.running_mean) / sqrt(vae.bn.running_var + vae.config.batch_norm_eps)"


def ernie_transformer_timestep_expression() -> String:
    return "timestep"


def ernie_text_lengths_expression() -> String:
    return "tokens_mask.sum(dim=1).long()"


def ernie_patchified_latent_height(latent_height: Int) -> Int:
    return latent_height // ERNIE_LATENT_PATCH_FACTOR


def ernie_patchified_latent_width(latent_width: Int) -> Int:
    return latent_width // ERNIE_LATENT_PATCH_FACTOR


def ernie_patchified_latent_channels(latent_channels: Int) -> Int:
    return latent_channels * ERNIE_LATENT_PATCH_FACTOR * ERNIE_LATENT_PATCH_FACTOR


def ernie_unpatchified_latent_channels(patchified_channels: Int) -> Int:
    return patchified_channels // (
        ERNIE_LATENT_PATCH_FACTOR * ERNIE_LATENT_PATCH_FACTOR
    )


def ernie_sigma_from_timestep(
    t: Int, num_timesteps: Int = ERNIE_NUM_TRAIN_TIMESTEPS
) -> Float32:
    # ModelSetupFlowMatchingMixin._add_noise_discrete:
    # sigma[t] = arange(1, N+1)[t] / N.
    return Float32(t + 1) / Float32(num_timesteps)


def ernie_model_t_from_timestep(t: Int) -> Float32:
    # BaseErnieSetup.predict passes the raw discrete timestep to the transformer.
    return Float32(t)


def ernie_calculate_timestep_shift(
    latent_height: Int,
    latent_width: Int,
    base_image_seq_len: Float32,
    max_image_seq_len: Float32,
    base_shift: Float32,
    max_shift: Float32,
) -> Float32:
    # ErnieModel.calculate_timestep_shift uses patch_size=2 and the scheduler
    # config values. The product makes width/height argument order irrelevant.
    var patch_size = ERNIE_LATENT_PATCH_FACTOR
    var image_seq_len = Float32(
        (latent_width // patch_size) * (latent_height // patch_size)
    )
    var m = (max_shift - base_shift) / (max_image_seq_len - base_image_seq_len)
    var b = base_shift - m * base_image_seq_len
    var mu = image_seq_len * m + b
    return exp(mu)


def ernie_get_timestep_discrete(
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
        ERNIE_IMAGE_BASIS_BATCH,
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


struct ErniePredictContract(Movable):
    var required_batch_fields: List[String]
    var conditioning_batch_fields: List[String]
    var output_fields: List[String]
    var loss_type: String
    var target_expression: String
    var noisy_latent_expression: String
    var patchify_latents_expression: String
    var unpatchify_latents_expression: String
    var scale_latents_expression: String
    var transformer_timestep_expression: String
    var text_lengths_expression: String
    var text_hidden_state_layer: Int

    def __init__(out self):
        self.required_batch_fields = ernie_predict_required_batch_fields()
        self.conditioning_batch_fields = ernie_predict_conditioning_batch_fields()
        self.output_fields = ernie_predict_output_fields()
        self.loss_type = ERNIE_LOSS_TYPE_TARGET
        self.target_expression = ernie_flow_target_expression()
        self.noisy_latent_expression = ernie_noisy_latent_expression()
        self.patchify_latents_expression = ernie_patchify_latents_expression()
        self.unpatchify_latents_expression = ernie_unpatchify_latents_expression()
        self.scale_latents_expression = ernie_scale_latents_expression()
        self.transformer_timestep_expression = ernie_transformer_timestep_expression()
        self.text_lengths_expression = ernie_text_lengths_expression()
        self.text_hidden_state_layer = ERNIE_HIDDEN_STATES_LAYER


struct ErnieOptimizationContract(Movable):
    var checkpoint_parts: List[String]
    var quantized_parts: List[String]
    var autocast_weight_dtype_parts: List[String]
    var text_encoder_autocast_weight_dtype_parts: List[String]
    var disables_fp16_text_encoder_autocast: Bool

    def __init__(out self, training_method: Int):
        self.checkpoint_parts = ernie_setup_optimization_checkpoint_parts()
        self.quantized_parts = ernie_setup_optimization_quantized_parts()
        self.autocast_weight_dtype_parts = ernie_autocast_weight_dtype_parts(
            training_method
        )
        self.text_encoder_autocast_weight_dtype_parts = (
            ernie_text_encoder_autocast_weight_dtype_parts(training_method)
        )
        self.disables_fp16_text_encoder_autocast = True


struct ErnieTrainDevicePlan(Copyable, Movable, ImplicitlyCopyable):
    var text_encoder_on_train_device: Bool
    var vae_on_train_device: Bool
    var transformer_on_train_device: Bool
    var text_encoder_train_mode: Bool
    var vae_train_mode: Bool
    var transformer_train_mode: Bool

    def __init__(out self, latent_caching: Bool, transformer_train: Bool):
        self.text_encoder_on_train_device = not latent_caching
        self.vae_on_train_device = not latent_caching
        self.transformer_on_train_device = True
        self.text_encoder_train_mode = False
        self.vae_train_mode = False
        self.transformer_train_mode = transformer_train


struct ErnieTextCachingPlan(Copyable, Movable, ImplicitlyCopyable):
    var move_model_to_temp_device: Bool
    var move_text_encoder_to_train_device: Bool
    var set_eval_mode: Bool
    var run_torch_gc: Bool

    def __init__(out self):
        self.move_model_to_temp_device = True
        self.move_text_encoder_to_train_device = True
        self.set_eval_mode = True
        self.run_torch_gc = True


def ernie_predict_contract() -> ErniePredictContract:
    return ErniePredictContract()


def ernie_optimization_contract(training_method: Int) -> ErnieOptimizationContract:
    return ErnieOptimizationContract(training_method)


def ernie_train_device_plan(
    latent_caching: Bool, transformer_train: Bool
) -> ErnieTrainDevicePlan:
    return ErnieTrainDevicePlan(latent_caching, transformer_train)


def ernie_text_caching_plan() -> ErnieTextCachingPlan:
    return ErnieTextCachingPlan()


struct BaseErnieSetup(Copyable, Movable, ImplicitlyCopyable):
    var debug_mode: Bool

    def __init__(out self, debug_mode: Bool = False):
        self.debug_mode = debug_mode

    def layer_preset_filters(self, preset: String) raises -> List[String]:
        return ernie_layer_preset_filters(preset)

    def predict_contract(self) -> ErniePredictContract:
        return ernie_predict_contract()

    def optimization_contract(self, training_method: Int) -> ErnieOptimizationContract:
        return ernie_optimization_contract(training_method)

    def train_device_plan(
        self, latent_caching: Bool, transformer_train: Bool
    ) -> ErnieTrainDevicePlan:
        return ernie_train_device_plan(latent_caching, transformer_train)

    def calculate_loss_consumes_sigmas(self) -> Bool:
        # calculate_loss delegates to _flow_matching_losses(..., sigmas=scheduler.sigmas).
        return True

    def calculate_loss_reduction(self) -> String:
        return "mean"

    def prepare_text_caching_plan(self) -> ErnieTextCachingPlan:
        return ernie_text_caching_plan()

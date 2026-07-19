# BaseIdeogram4Setup.mojo - build-only Ideogram4 training contract.
#
# This records the ai-toolkit training pipeline facts in Serenity Trainer form:
# patchified 32-channel VAE latents -> 128-channel image tokens, CustomFlowMatch
# add_noise, Qwen3-VL 13-tap text features, transformer receives 1 - t, and its
# output is negated so the trainer predicts noise-clean velocity.

from serenity_trainer.modelSampler.Ideogram4Sampler import (
    IDEOGRAM4_IMAGE_OFFSET,
    IDEOGRAM4_LLM_TOKEN_INDICATOR,
    IDEOGRAM4_NUM_LAYERS,
    IDEOGRAM4_OUTPUT_IMAGE_INDICATOR,
    IDEOGRAM4_PACKED_CHANNELS,
    IDEOGRAM4_PATCH_SIZE,
    IDEOGRAM4_PIXEL_TOKEN_STRIDE,
    IDEOGRAM4_TEXT_FEATURE_DIM,
    IDEOGRAM4_VAE_LATENT_CHANNELS,
    ideogram4_add_noise_scalar,
    ideogram4_flow_target_scalar,
    ideogram4_image_tokens,
)


comptime IDEOGRAM4_TRAIN_SCHEDULER = "CustomFlowMatchEulerDiscreteScheduler"
comptime IDEOGRAM4_TRAIN_TIMESTEP_TYPE = "linear"
comptime IDEOGRAM4_TRAIN_NUM_TIMESTEPS = 1000
comptime IDEOGRAM4_TEXT_ENCODER = "Qwen/Qwen3-VL-8B-Instruct"
comptime IDEOGRAM4_TEXT_TAP_COUNT = 13
comptime IDEOGRAM4_MAX_TEXT_LENGTH = 3072
comptime IDEOGRAM4_CAPTION_EXTENSION = "json"
comptime IDEOGRAM4_BUCKET_DIVISIBILITY = 16


def ideogram4_text_activation_layers() -> List[Int]:
    var out = List[Int]()
    out.append(0)
    out.append(3)
    out.append(6)
    out.append(9)
    out.append(12)
    out.append(15)
    out.append(18)
    out.append(21)
    out.append(24)
    out.append(27)
    out.append(30)
    out.append(33)
    out.append(35)
    return out^


def ideogram4_train_add_noise_scalar(clean_latent: Float32, noise: Float32, timestep_0_to_1000: Float32) -> Float32:
    var t01 = timestep_0_to_1000 / Float32(1000.0)
    return ideogram4_add_noise_scalar(clean_latent, noise, t01)


def ideogram4_train_model_t_scalar(timestep_0_to_1000: Float32) -> Float32:
    return Float32(1.0) - timestep_0_to_1000 / Float32(1000.0)


def ideogram4_train_prediction_scalar(model_output_clean_minus_noise: Float32) -> Float32:
    return -model_output_clean_minus_noise


def ideogram4_train_target_scalar(clean_latent: Float32, noise: Float32) -> Float32:
    return ideogram4_flow_target_scalar(clean_latent, noise)


struct Ideogram4PackedShape(Copyable, Movable, ImplicitlyCopyable):
    var width: Int
    var height: Int
    var latent_h: Int
    var latent_w: Int
    var grid_h: Int
    var grid_w: Int
    var image_tokens: Int
    var channels: Int

    def __init__(out self, width: Int, height: Int) raises:
        self.width = width
        self.height = height
        self.latent_h = height // 8
        self.latent_w = width // 8
        self.grid_h = height // IDEOGRAM4_PIXEL_TOKEN_STRIDE
        self.grid_w = width // IDEOGRAM4_PIXEL_TOKEN_STRIDE
        self.image_tokens = ideogram4_image_tokens(width, height)
        self.channels = IDEOGRAM4_PACKED_CHANNELS


def ideogram4_packed_shape(width: Int, height: Int) raises -> Ideogram4PackedShape:
    return Ideogram4PackedShape(width, height)


struct Ideogram4TrainFlowContract(Copyable, Movable, ImplicitlyCopyable):
    var scheduler_name: String
    var timestep_type: String
    var num_train_timesteps: Int
    var add_noise_expression: String
    var target_expression: String
    var model_t_expression: String
    var model_prediction_expression: String
    var text_encoder: String
    var text_activation_layer_count: Int
    var text_feature_dim: Int
    var max_text_length: Int
    var caption_extension: String
    var bucket_divisibility: Int
    var image_indicator: Int
    var llm_indicator: Int
    var image_position_offset: Int
    var native_training_forward_present: Bool
    var native_lora_backward_present: Bool

    def __init__(out self):
        self.scheduler_name = String(IDEOGRAM4_TRAIN_SCHEDULER)
        self.timestep_type = String(IDEOGRAM4_TRAIN_TIMESTEP_TYPE)
        self.num_train_timesteps = IDEOGRAM4_TRAIN_NUM_TIMESTEPS
        self.add_noise_expression = String("(1 - t/1000) * clean + (t/1000) * noise")
        self.target_expression = String("noise - clean")
        self.model_t_expression = String("1 - t/1000")
        self.model_prediction_expression = String("-transformer_output")
        self.text_encoder = String(IDEOGRAM4_TEXT_ENCODER)
        self.text_activation_layer_count = IDEOGRAM4_TEXT_TAP_COUNT
        self.text_feature_dim = IDEOGRAM4_TEXT_FEATURE_DIM
        self.max_text_length = IDEOGRAM4_MAX_TEXT_LENGTH
        self.caption_extension = String(IDEOGRAM4_CAPTION_EXTENSION)
        self.bucket_divisibility = IDEOGRAM4_BUCKET_DIVISIBILITY
        self.image_indicator = IDEOGRAM4_OUTPUT_IMAGE_INDICATOR
        self.llm_indicator = IDEOGRAM4_LLM_TOKEN_INDICATOR
        self.image_position_offset = IDEOGRAM4_IMAGE_OFFSET
        self.native_training_forward_present = True
        # Aligned with Ideogram4ModelLoader (native_lora_backward_slice =
        # "transformer.layers.* + transformer.final_layer.linear"): the block-stack
        # LoRA backward (Ideogram4LoRABlock.ideogram4_stack_lora_backward) and the
        # final-linear LoRA backward (Ideogram4FinalLinearLoRA.train_step) both exist.
        self.native_lora_backward_present = True


def ideogram4_train_flow_contract() -> Ideogram4TrainFlowContract:
    return Ideogram4TrainFlowContract()


def ideogram4_setup_optimization_quantized_parts() -> List[String]:
    var parts = List[String]()
    parts.append(String("transformer"))
    parts.append(String("text_encoder"))
    return parts^


def ideogram4_frozen_part_names() -> List[String]:
    var parts = List[String]()
    parts.append(String("text_encoder"))
    parts.append(String("vae"))
    parts.append(String("base_transformer_weights"))
    return parts^


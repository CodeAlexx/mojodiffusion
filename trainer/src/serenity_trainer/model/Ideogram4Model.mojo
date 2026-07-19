# Ideogram4Model.mojo — trainer-side identity for the Ideogram-4 native stack.
#
# Ideogram-4 is not Flux2/Klein. The local inference implementation lives in
# /home/alex/mojodiffusion/serenitymojo and has its own structure:
#   - conditional fp8 DiT
#   - unconditional fp8 DiT
#   - Qwen3-VL 13-tap text encoder
#   - structured JSON caption prompts
#   - logit-normal Euler schedule
#   - asymmetric CFG
#   - Flux2-family VAE decode

from serenity_trainer.modelSampler.Ideogram4Sampler import (
    IDEOGRAM4_HIDDEN,
    IDEOGRAM4_HEAD_DIM,
    IDEOGRAM4_IMAGE_OFFSET,
    IDEOGRAM4_ADALN_DIM,
    IDEOGRAM4_INTERMEDIATE_SIZE,
    IDEOGRAM4_LLM_TOKEN_INDICATOR,
    IDEOGRAM4_MROPE_SECTION_0,
    IDEOGRAM4_MROPE_SECTION_1,
    IDEOGRAM4_MROPE_SECTION_2,
    IDEOGRAM4_MROPE_THETA,
    IDEOGRAM4_NUM_HEADS,
    IDEOGRAM4_NUM_LAYERS,
    IDEOGRAM4_OUTPUT_IMAGE_INDICATOR,
    IDEOGRAM4_PACKED_CHANNELS,
    IDEOGRAM4_PATCH_SIZE,
    IDEOGRAM4_PIXEL_TOKEN_STRIDE,
    IDEOGRAM4_QWEN3_VL_HIDDEN,
    IDEOGRAM4_SEQUENCE_PADDING_INDICATOR,
    IDEOGRAM4_TEXT_FEATURE_DIM,
    IDEOGRAM4_TEXT_TAP_COUNT,
    IDEOGRAM4_VAE_LATENT_CHANNELS,
    IDEOGRAM4_VAE_SCALE_FACTOR,
)


struct Ideogram4ModelContract(Copyable, Movable):
    var model_type_name: String
    var arch_name: String
    var checkpoint_family: String
    var conditional_transformer_subdir: String
    var unconditional_transformer_subdir: String
    var text_encoder_subdir: String
    var vae_subdir: String
    var uses_structured_json_captions: Bool
    var uses_magic_prompt: Bool
    var uses_dual_transformer_cfg: Bool
    var uses_logitnormal_schedule: Bool
    var uses_flux2_vae_family: Bool
    var num_layers: Int
    var hidden: Int
    var num_heads: Int
    var head_dim: Int
    var intermediate_size: Int
    var adaln_dim: Int
    var qwen3_vl_hidden: Int
    var text_tap_count: Int
    var text_feature_dim: Int
    var packed_channels: Int
    var vae_latent_channels: Int
    var vae_scale_factor: Int
    var patch_size: Int
    var pixel_token_stride: Int
    var image_offset: Int
    var sequence_padding_indicator: Int
    var llm_token_indicator: Int
    var output_image_indicator: Int
    var rope_section_0: Int
    var rope_section_1: Int
    var rope_section_2: Int
    var rope_theta: Float32

    def __init__(out self):
        self.model_type_name = String("IDEOGRAM_4")
        self.arch_name = String("ideogram4")
        self.checkpoint_family = String("ideogram-ai/ideogram-4-fp8")
        self.conditional_transformer_subdir = String("transformer")
        self.unconditional_transformer_subdir = String("unconditional_transformer")
        self.text_encoder_subdir = String("text_encoder")
        self.vae_subdir = String("vae")
        self.uses_structured_json_captions = True
        self.uses_magic_prompt = True
        self.uses_dual_transformer_cfg = True
        self.uses_logitnormal_schedule = True
        self.uses_flux2_vae_family = True
        self.num_layers = IDEOGRAM4_NUM_LAYERS
        self.hidden = IDEOGRAM4_HIDDEN
        self.num_heads = IDEOGRAM4_NUM_HEADS
        self.head_dim = IDEOGRAM4_HEAD_DIM
        self.intermediate_size = IDEOGRAM4_INTERMEDIATE_SIZE
        self.adaln_dim = IDEOGRAM4_ADALN_DIM
        self.qwen3_vl_hidden = IDEOGRAM4_QWEN3_VL_HIDDEN
        self.text_tap_count = IDEOGRAM4_TEXT_TAP_COUNT
        self.text_feature_dim = IDEOGRAM4_TEXT_FEATURE_DIM
        self.packed_channels = IDEOGRAM4_PACKED_CHANNELS
        self.vae_latent_channels = IDEOGRAM4_VAE_LATENT_CHANNELS
        self.vae_scale_factor = IDEOGRAM4_VAE_SCALE_FACTOR
        self.patch_size = IDEOGRAM4_PATCH_SIZE
        self.pixel_token_stride = IDEOGRAM4_PIXEL_TOKEN_STRIDE
        self.image_offset = IDEOGRAM4_IMAGE_OFFSET
        self.sequence_padding_indicator = IDEOGRAM4_SEQUENCE_PADDING_INDICATOR
        self.llm_token_indicator = IDEOGRAM4_LLM_TOKEN_INDICATOR
        self.output_image_indicator = IDEOGRAM4_OUTPUT_IMAGE_INDICATOR
        self.rope_section_0 = IDEOGRAM4_MROPE_SECTION_0
        self.rope_section_1 = IDEOGRAM4_MROPE_SECTION_1
        self.rope_section_2 = IDEOGRAM4_MROPE_SECTION_2
        self.rope_theta = IDEOGRAM4_MROPE_THETA


def ideogram4_model_contract() -> Ideogram4ModelContract:
    return Ideogram4ModelContract()

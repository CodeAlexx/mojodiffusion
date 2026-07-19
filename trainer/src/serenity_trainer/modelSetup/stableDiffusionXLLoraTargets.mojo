# stableDiffusionXLLoraTargets.mojo -- SDXL LoRA target metadata.
#
# Serenity references:
#   modules/modelSetup/StableDiffusionXLLoRASetup.py wraps optional text encoder
#   LoRAs with prefixes "lora_te1" and "lora_te2", and always wraps the UNet
#   with prefix "lora_unet" plus config.layer_filter.split(",").
#   modules/modelSaver/stableDiffusionXL/StableDiffusionXLLoRASaver.py uses
#   convert_sdxl_lora_key_sets(); INTERNAL output saves OMI keys through
#   LoRASaverMixin.__save_safetensors.
#   modules/util/convert/lora/convert_sdxl_lora.py maps OMI "unet" to raw
#   diffusers "lora_unet".
#   modules/module/LoRAModule.py writes:
#     <prefix>.<module>.lora_down.weight
#     <prefix>.<module>.lora_up.weight
#     <prefix>.<module>.alpha
#
# This file is host metadata for SDXL LoRA key/shape gates. The production
# real-file gate uses Serenity's final BF16 output LoRA with converted
# `lora_unet_*` keys. The backup snapshot in workspace/backup is FP32 because
# Serenity stores train-state LoRA weights with config.lora_weight_dtype
# FLOAT_32; do not use that as the product dtype boundary.


comptime SDXL_LORA_TEXT_ENCODER_1_PREFIX = "lora_te1"
comptime SDXL_LORA_TEXT_ENCODER_2_PREFIX = "lora_te2"
comptime SDXL_LORA_DIFFUSERS_UNET_PREFIX = "lora_unet"
comptime SDXL_LORA_OMI_UNET_PREFIX = "unet"

comptime SDXL_REAL_OMI_UNET_TARGET_COUNT = 794
comptime SDXL_REAL_OMI_UNET_KEY_COUNT = 2382
comptime SDXL_REAL_OMI_UNET_RANK = 16


struct StableDiffusionXLLoraTargetSpecs(Movable):
    var prefixes: List[String]
    var in_features: List[Int]
    var out_features: List[Int]
    var kernel_h: List[Int]
    var kernel_w: List[Int]
    var roles: List[String]
    var source_paths: List[String]

    def __init__(
        out self,
        var prefixes: List[String],
        var in_features: List[Int],
        var out_features: List[Int],
        var kernel_h: List[Int],
        var kernel_w: List[Int],
        var roles: List[String],
        var source_paths: List[String],
    ):
        self.prefixes = prefixes^
        self.in_features = in_features^
        self.out_features = out_features^
        self.kernel_h = kernel_h^
        self.kernel_w = kernel_w^
        self.roles = roles^
        self.source_paths = source_paths^

    def len(self) -> Int:
        return len(self.prefixes)


def sdxl_real_omi_unet_lora_file() -> String:
    return String("/home/alex/Serenity/output/sdxl_100step_baseline/lora_last.safetensors")


def sdxl_real_omi_unet_alpha() -> Float32:
    return Float32(16.0)


def sdxl_lora_prefixed_module(wrapper_prefix: String, module_name: String) -> String:
    if wrapper_prefix == String():
        return module_name
    return wrapper_prefix + String(".") + module_name


def sdxl_omi_unet_lora_prefix(module_name: String) -> String:
    return sdxl_lora_prefixed_module(String(SDXL_LORA_OMI_UNET_PREFIX), module_name)


def sdxl_diffusers_unet_lora_prefix(module_name: String) -> String:
    return sdxl_lora_prefixed_module(String(SDXL_LORA_DIFFUSERS_UNET_PREFIX), module_name)


def sdxl_text_encoder_1_lora_prefix(module_name: String) -> String:
    return sdxl_lora_prefixed_module(String(SDXL_LORA_TEXT_ENCODER_1_PREFIX), module_name)


def sdxl_text_encoder_2_lora_prefix(module_name: String) -> String:
    return sdxl_lora_prefixed_module(String(SDXL_LORA_TEXT_ENCODER_2_PREFIX), module_name)


def sdxl_lora_down_key(prefix: String) -> String:
    return prefix + String(".lora_down.weight")


def sdxl_lora_up_key(prefix: String) -> String:
    return prefix + String(".lora_up.weight")


def sdxl_lora_alpha_key(prefix: String) -> String:
    return prefix + String(".alpha")


def _sdxl_append_target(
    mut prefixes: List[String],
    mut in_features: List[Int],
    mut out_features: List[Int],
    mut kernel_h: List[Int],
    mut kernel_w: List[Int],
    mut roles: List[String],
    mut source_paths: List[String],
    prefix: String,
    in_feature_count: Int,
    out_feature_count: Int,
    kernel_height: Int,
    kernel_width: Int,
    role: String,
    source_path: String,
):
    prefixes.append(prefix.copy())
    in_features.append(in_feature_count)
    out_features.append(out_feature_count)
    kernel_h.append(kernel_height)
    kernel_w.append(kernel_width)
    roles.append(role.copy())
    source_paths.append(source_path.copy())


def sdxl_real_omi_unet_representative_target_specs() -> StableDiffusionXLLoraTargetSpecs:
    """Representative real SDXL final-output LoRA target specs.

    The local Serenity output file contains 794 UNet adapters / 2382 tensors.
    This slice validates the converted `lora_unet_*` namespace, rank, dtype,
    alpha, Linear shapes, Conv2d shapes, and representative down/mid/up block
    locations without claiming a complete generated target inventory.
    """
    var prefixes = List[String]()
    var in_features = List[Int]()
    var out_features = List[Int]()
    var kernel_h = List[Int]()
    var kernel_w = List[Int]()
    var roles = List[String]()
    var source_paths = List[String]()
    var source = String("/home/alex/Serenity/modules/modelSetup/StableDiffusionXLLoRASetup.py | /home/alex/Serenity/modules/modelSaver/stableDiffusionXL/StableDiffusionXLLoRASaver.py | /home/alex/Serenity/modules/util/convert/lora/convert_sdxl_lora.py | /home/alex/Serenity/modules/module/LoRAModule.py")

    _sdxl_append_target(prefixes, in_features, out_features, kernel_h, kernel_w, roles, source_paths, String("lora_unet_conv_in"), 4, 320, 3, 3, String("unet.conv_in"), source)
    _sdxl_append_target(prefixes, in_features, out_features, kernel_h, kernel_w, roles, source_paths, String("lora_unet_time_embedding_linear_1"), 320, 1280, 0, 0, String("unet.time_embedding.linear_1"), source)
    _sdxl_append_target(prefixes, in_features, out_features, kernel_h, kernel_w, roles, source_paths, String("lora_unet_time_embedding_linear_2"), 1280, 1280, 0, 0, String("unet.time_embedding.linear_2"), source)
    _sdxl_append_target(prefixes, in_features, out_features, kernel_h, kernel_w, roles, source_paths, String("lora_unet_add_embedding_linear_1"), 2816, 1280, 0, 0, String("unet.add_embedding.linear_1"), source)
    _sdxl_append_target(prefixes, in_features, out_features, kernel_h, kernel_w, roles, source_paths, String("lora_unet_add_embedding_linear_2"), 1280, 1280, 0, 0, String("unet.add_embedding.linear_2"), source)
    _sdxl_append_target(prefixes, in_features, out_features, kernel_h, kernel_w, roles, source_paths, String("lora_unet_down_blocks_0_resnets_0_time_emb_proj"), 1280, 320, 0, 0, String("unet.down.resnet.time_emb_proj"), source)
    _sdxl_append_target(prefixes, in_features, out_features, kernel_h, kernel_w, roles, source_paths, String("lora_unet_down_blocks_1_resnets_0_conv1"), 320, 640, 3, 3, String("unet.down.resnet.conv1"), source)
    _sdxl_append_target(prefixes, in_features, out_features, kernel_h, kernel_w, roles, source_paths, String("lora_unet_down_blocks_1_attentions_0_proj_in"), 640, 640, 0, 0, String("unet.down.attention.proj_in"), source)
    _sdxl_append_target(prefixes, in_features, out_features, kernel_h, kernel_w, roles, source_paths, String("lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_q"), 640, 640, 0, 0, String("unet.down.self_attention.to_q"), source)
    _sdxl_append_target(prefixes, in_features, out_features, kernel_h, kernel_w, roles, source_paths, String("lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn2_to_k"), 2048, 640, 0, 0, String("unet.down.cross_attention.to_k"), source)
    _sdxl_append_target(prefixes, in_features, out_features, kernel_h, kernel_w, roles, source_paths, String("lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_ff_net_0_proj"), 640, 5120, 0, 0, String("unet.down.ff.in"), source)
    _sdxl_append_target(prefixes, in_features, out_features, kernel_h, kernel_w, roles, source_paths, String("lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_ff_net_2"), 2560, 640, 0, 0, String("unet.down.ff.out"), source)
    _sdxl_append_target(prefixes, in_features, out_features, kernel_h, kernel_w, roles, source_paths, String("lora_unet_mid_block_attentions_0_proj_out"), 1280, 1280, 0, 0, String("unet.mid.attention.proj_out"), source)
    _sdxl_append_target(prefixes, in_features, out_features, kernel_h, kernel_w, roles, source_paths, String("lora_unet_mid_block_attentions_0_transformer_blocks_0_attn2_to_v"), 2048, 1280, 0, 0, String("unet.mid.cross_attention.to_v"), source)
    _sdxl_append_target(prefixes, in_features, out_features, kernel_h, kernel_w, roles, source_paths, String("lora_unet_up_blocks_0_upsamplers_0_conv"), 1280, 1280, 3, 3, String("unet.up.upsampler.conv"), source)
    _sdxl_append_target(prefixes, in_features, out_features, kernel_h, kernel_w, roles, source_paths, String("lora_unet_up_blocks_1_attentions_1_transformer_blocks_1_attn2_to_out_0"), 640, 640, 0, 0, String("unet.up.cross_attention.to_out"), source)
    _sdxl_append_target(prefixes, in_features, out_features, kernel_h, kernel_w, roles, source_paths, String("lora_unet_conv_out"), 320, 4, 3, 3, String("unet.conv_out"), source)

    return StableDiffusionXLLoraTargetSpecs(prefixes^, in_features^, out_features^, kernel_h^, kernel_w^, roles^, source_paths^)

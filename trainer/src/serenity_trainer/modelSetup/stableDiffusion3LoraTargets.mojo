# stableDiffusion3LoraTargets.mojo — SD3 LoRA target metadata.
#
# Serenity references:
#   modules/modelSetup/StableDiffusion3LoRASetup.py wraps optional text encoder
#   LoRAs with prefixes "lora_te1", "lora_te2", "lora_te3", and always wraps the
#   transformer with prefix "lora_transformer" plus config.layer_filter.split(",").
#   modules/module/LoRAModule.py writes raw state_dict keys:
#     <wrapper_prefix>.<module_name>.lora_down.weight
#     <wrapper_prefix>.<module_name>.lora_up.weight
#     <wrapper_prefix>.<module_name>.alpha
#   modules/util/convert/lora/convert_sd3_lora.py maps those raw diffusers keys
#   to OMI/legacy namespaces for file formats that request conversion.
#
# This file is host metadata for SD3 LoRA key/shape gates. The expanded linear
# inventory mirrors Serenity's closest available source paths and conversion
# rules, but does not claim full Serenity parity without a real Serenity SD3
# LoRA reference file generated in-session.


comptime SD3_LORA_TEXT_ENCODER_1_PREFIX = "lora_te1"
comptime SD3_LORA_TEXT_ENCODER_2_PREFIX = "lora_te2"
comptime SD3_LORA_TEXT_ENCODER_3_PREFIX = "lora_te3"
comptime SD3_LORA_TRANSFORMER_PREFIX = "lora_transformer"


struct StableDiffusion3LoraTargetSpecs(Movable):
    var prefixes: List[String]
    var in_features: List[Int]
    var out_features: List[Int]
    var roles: List[String]
    var source_paths: List[String]

    def __init__(
        out self,
        var prefixes: List[String],
        var in_features: List[Int],
        var out_features: List[Int],
        var roles: List[String],
        var source_paths: List[String],
    ):
        self.prefixes = prefixes^
        self.in_features = in_features^
        self.out_features = out_features^
        self.roles = roles^
        self.source_paths = source_paths^

    def len(self) -> Int:
        return len(self.prefixes)


def sd3_lora_prefixed_module(wrapper_prefix: String, module_name: String) -> String:
    if wrapper_prefix == String():
        return module_name
    return wrapper_prefix + String(".") + module_name


def sd3_transformer_module(block_idx: Int, suffix: String) -> String:
    return String("transformer_blocks.") + String(block_idx) + String(".") + suffix


def sd3_text_encoder_1_lora_prefix(module_name: String) -> String:
    return sd3_lora_prefixed_module(String(SD3_LORA_TEXT_ENCODER_1_PREFIX), module_name)


def sd3_text_encoder_2_lora_prefix(module_name: String) -> String:
    return sd3_lora_prefixed_module(String(SD3_LORA_TEXT_ENCODER_2_PREFIX), module_name)


def sd3_text_encoder_3_lora_prefix(module_name: String) -> String:
    return sd3_lora_prefixed_module(String(SD3_LORA_TEXT_ENCODER_3_PREFIX), module_name)


def sd3_transformer_lora_prefix(module_name: String) -> String:
    return sd3_lora_prefixed_module(String(SD3_LORA_TRANSFORMER_PREFIX), module_name)


def sd3_lora_down_key(prefix: String) -> String:
    return prefix + String(".lora_down.weight")


def sd3_lora_up_key(prefix: String) -> String:
    return prefix + String(".lora_up.weight")


def sd3_lora_alpha_key(prefix: String) -> String:
    return prefix + String(".alpha")


def sd3_transformer_representative_suffixes() -> List[String]:
    var out = List[String]()
    out.append(String("pos_embed.proj"))
    out.append(String("transformer_blocks.0.attn.to_q"))
    out.append(String("transformer_blocks.0.attn.add_q_proj"))
    out.append(String("transformer_blocks.0.ff.net.0.proj"))
    out.append(String("transformer_blocks.0.ff_context.net.2"))
    return out^


def _sd3_append_target(
    mut prefixes: List[String],
    mut in_features: List[Int],
    mut out_features: List[Int],
    mut roles: List[String],
    mut source_paths: List[String],
    prefix: String,
    in_feature_count: Int,
    out_feature_count: Int,
    role: String,
    source_path: String,
):
    prefixes.append(prefix.copy())
    in_features.append(in_feature_count)
    out_features.append(out_feature_count)
    roles.append(role.copy())
    source_paths.append(source_path.copy())


def sd3_transformer_linear_conversion_suffixes() -> List[String]:
    """Top-level transformer Linear suffixes from convert_sd3_lora.py."""
    var out = List[String]()
    out.append(String("pos_embed.proj"))
    out.append(String("context_embedder"))
    out.append(String("norm_out.linear"))
    out.append(String("proj_out"))
    out.append(String("time_text_embed.timestep_embedder.linear_1"))
    out.append(String("time_text_embed.timestep_embedder.linear_2"))
    out.append(String("time_text_embed.text_embedder.linear_1"))
    out.append(String("time_text_embed.text_embedder.linear_2"))
    return out^


def sd3_transformer_block_linear_conversion_suffixes() -> List[String]:
    """Per-block transformer Linear suffixes from convert_sd3_lora.py."""
    var out = List[String]()
    out.append(String("attn.to_q"))
    out.append(String("attn.to_k"))
    out.append(String("attn.to_v"))
    out.append(String("attn.add_q_proj"))
    out.append(String("attn.add_k_proj"))
    out.append(String("attn.add_v_proj"))
    out.append(String("attn.to_out.0"))
    out.append(String("attn.to_add_out"))
    out.append(String("norm1.linear"))
    out.append(String("norm1_context.linear"))
    out.append(String("ff.net.0.proj"))
    out.append(String("ff.net.2"))
    out.append(String("attn2.to_q"))
    out.append(String("attn2.to_k"))
    out.append(String("attn2.to_v"))
    out.append(String("attn2.to_out.0"))
    out.append(String("ff_context.net.0.proj"))
    out.append(String("ff_context.net.2"))
    return out^


def sd3_clip_linear_conversion_suffixes() -> List[String]:
    """CLIP Linear suffixes from convert_clip.py."""
    var out = List[String]()
    out.append(String("mlp.fc1"))
    out.append(String("mlp.fc2"))
    out.append(String("self_attn.k_proj"))
    out.append(String("self_attn.out_proj"))
    out.append(String("self_attn.q_proj"))
    out.append(String("self_attn.v_proj"))
    return out^


def sd3_t5_linear_conversion_suffixes() -> List[String]:
    """T5 Linear suffixes from convert_t5.py."""
    var out = List[String]()
    out.append(String("layer.0.SelfAttention.k"))
    out.append(String("layer.0.SelfAttention.o"))
    out.append(String("layer.0.SelfAttention.q"))
    out.append(String("layer.0.SelfAttention.v"))
    out.append(String("layer.1.DenseReluDense.wi_0"))
    out.append(String("layer.1.DenseReluDense.wi_1"))
    out.append(String("layer.1.DenseReluDense.wo"))
    return out^


def sd3_bounded_lora_target_specs(
    transformer_dim: Int,
    transformer_mlp_dim: Int,
    clip_dim: Int,
    t5_dim: Int,
    t5_mlp_dim: Int,
) -> StableDiffusion3LoraTargetSpecs:
    """Tiny representative SD3 LoRA target set for key save/load gates.

    Covers all four Serenity SD3 wrapper namespaces and representative
    transformer attention/MLP names. This is not a full SD3 adapter inventory.
    """
    var prefixes = List[String]()
    var in_features = List[Int]()
    var out_features = List[Int]()
    var roles = List[String]()
    var source_paths = List[String]()
    var source = String("/home/alex/Serenity/modules/modelSetup/StableDiffusion3LoRASetup.py | /home/alex/Serenity/modules/module/LoRAModule.py")

    _sd3_append_target(prefixes, in_features, out_features, roles, source_paths, sd3_text_encoder_1_lora_prefix(String("text_model.encoder.layers.0.self_attn.q_proj")), clip_dim, clip_dim, String("clip_l.self_attn.q_proj"), source)
    _sd3_append_target(prefixes, in_features, out_features, roles, source_paths, sd3_text_encoder_2_lora_prefix(String("text_model.encoder.layers.0.mlp.fc1")), clip_dim, 4 * clip_dim, String("clip_g.mlp.fc1"), source)
    _sd3_append_target(prefixes, in_features, out_features, roles, source_paths, sd3_text_encoder_3_lora_prefix(String("encoder.block.0.layer.0.SelfAttention.q")), t5_dim, t5_dim, String("t5.self_attention.q"), source)
    _sd3_append_target(prefixes, in_features, out_features, roles, source_paths, sd3_transformer_lora_prefix(String("pos_embed.proj")), transformer_dim, transformer_dim, String("transformer.pos_embed.proj"), source)
    _sd3_append_target(prefixes, in_features, out_features, roles, source_paths, sd3_transformer_lora_prefix(sd3_transformer_module(0, String("attn.to_q"))), transformer_dim, transformer_dim, String("transformer.block.attn.to_q"), source)
    _sd3_append_target(prefixes, in_features, out_features, roles, source_paths, sd3_transformer_lora_prefix(sd3_transformer_module(0, String("attn.add_q_proj"))), transformer_dim, transformer_dim, String("transformer.block.attn.add_q_proj"), source)
    _sd3_append_target(prefixes, in_features, out_features, roles, source_paths, sd3_transformer_lora_prefix(sd3_transformer_module(0, String("ff.net.0.proj"))), transformer_dim, transformer_mlp_dim, String("transformer.block.ff.in"), source)
    _sd3_append_target(prefixes, in_features, out_features, roles, source_paths, sd3_transformer_lora_prefix(sd3_transformer_module(0, String("ff_context.net.2"))), transformer_mlp_dim, transformer_dim, String("transformer.block.ff_context.out"), source)
    _sd3_append_target(prefixes, in_features, out_features, roles, source_paths, sd3_text_encoder_3_lora_prefix(String("encoder.block.0.layer.1.DenseReluDense.wi_0")), t5_dim, t5_mlp_dim, String("t5.DenseReluDense.wi_0"), source)

    return StableDiffusion3LoraTargetSpecs(prefixes^, in_features^, out_features^, roles^, source_paths^)


def sd3_linear_lora_inventory_target_specs(
    transformer_blocks: Int,
    clip_layers: Int,
    t5_blocks: Int,
    transformer_dim: Int,
    transformer_mlp_dim: Int,
    pooled_dim: Int,
    clip_dim: Int,
    clip_mlp_dim: Int,
    clip_projection_dim: Int,
    t5_dim: Int,
    t5_mlp_dim: Int,
) -> StableDiffusion3LoraTargetSpecs:
    """Expanded deterministic SD3 Linear LoRA inventory/contract.

    Scope is raw Serenity wrapper keys and Linear shapes inferred from:
    StableDiffusion3LoRASetup.py, LoRAModule.py, convert_sd3_lora.py,
    convert_clip.py, and convert_t5.py. This excludes Conv2d LoRA targets and
    is not full Serenity parity without a real SD3 LoRA reference file.
    """
    var prefixes = List[String]()
    var in_features = List[Int]()
    var out_features = List[Int]()
    var roles = List[String]()
    var source_paths = List[String]()

    var setup_source = String("/home/alex/Serenity/modules/modelSetup/StableDiffusion3LoRASetup.py | /home/alex/Serenity/modules/module/LoRAModule.py | ")
    var sd3_source = setup_source + String("/home/alex/Serenity/modules/util/convert/lora/convert_sd3_lora.py")
    var clip_source = setup_source + String("/home/alex/Serenity/modules/util/convert/lora/convert_clip.py")
    var t5_source = setup_source + String("/home/alex/Serenity/modules/util/convert/lora/convert_t5.py")

    _sd3_append_target(prefixes, in_features, out_features, roles, source_paths, sd3_transformer_lora_prefix(String("pos_embed.proj")), transformer_dim, transformer_dim, String("transformer.x_embedder.proj"), sd3_source)
    _sd3_append_target(prefixes, in_features, out_features, roles, source_paths, sd3_transformer_lora_prefix(String("context_embedder")), t5_dim, pooled_dim, String("transformer.context_embedder"), sd3_source)
    _sd3_append_target(prefixes, in_features, out_features, roles, source_paths, sd3_transformer_lora_prefix(String("norm_out.linear")), transformer_dim, 2 * transformer_dim, String("transformer.final_layer.adaLN_modulation.1"), sd3_source)
    _sd3_append_target(prefixes, in_features, out_features, roles, source_paths, sd3_transformer_lora_prefix(String("proj_out")), transformer_dim, transformer_dim, String("transformer.final_layer.linear"), sd3_source)
    _sd3_append_target(prefixes, in_features, out_features, roles, source_paths, sd3_transformer_lora_prefix(String("time_text_embed.timestep_embedder.linear_1")), transformer_dim, transformer_dim, String("transformer.t_embedder.mlp.0"), sd3_source)
    _sd3_append_target(prefixes, in_features, out_features, roles, source_paths, sd3_transformer_lora_prefix(String("time_text_embed.timestep_embedder.linear_2")), transformer_dim, transformer_dim, String("transformer.t_embedder.mlp.2"), sd3_source)
    _sd3_append_target(prefixes, in_features, out_features, roles, source_paths, sd3_transformer_lora_prefix(String("time_text_embed.text_embedder.linear_1")), pooled_dim, transformer_dim, String("transformer.y_embedder.mlp.0"), sd3_source)
    _sd3_append_target(prefixes, in_features, out_features, roles, source_paths, sd3_transformer_lora_prefix(String("time_text_embed.text_embedder.linear_2")), transformer_dim, transformer_dim, String("transformer.y_embedder.mlp.2"), sd3_source)

    var block_suffixes = sd3_transformer_block_linear_conversion_suffixes()
    for bi in range(transformer_blocks):
        for si in range(len(block_suffixes)):
            var suffix = block_suffixes[si]
            var in_count = transformer_dim
            var out_count = transformer_dim
            if suffix == String("ff.net.0.proj") or suffix == String("ff_context.net.0.proj"):
                out_count = transformer_mlp_dim
            elif suffix == String("ff.net.2") or suffix == String("ff_context.net.2"):
                in_count = transformer_mlp_dim
            elif suffix == String("norm1.linear") or suffix == String("norm1_context.linear"):
                out_count = 6 * transformer_dim
            _sd3_append_target(prefixes, in_features, out_features, roles, source_paths, sd3_transformer_lora_prefix(sd3_transformer_module(bi, suffix)), in_count, out_count, String("transformer.block.") + suffix, sd3_source)

    _sd3_append_target(prefixes, in_features, out_features, roles, source_paths, sd3_text_encoder_1_lora_prefix(String("text_projection")), clip_dim, clip_projection_dim, String("clip_l.text_projection"), clip_source)
    _sd3_append_target(prefixes, in_features, out_features, roles, source_paths, sd3_text_encoder_2_lora_prefix(String("text_projection")), clip_dim, clip_projection_dim, String("clip_g.text_projection"), clip_source)

    var clip_suffixes = sd3_clip_linear_conversion_suffixes()
    for li in range(clip_layers):
        for si in range(len(clip_suffixes)):
            var suffix = clip_suffixes[si]
            var in_count = clip_dim
            var out_count = clip_dim
            if suffix == String("mlp.fc1"):
                out_count = clip_mlp_dim
            elif suffix == String("mlp.fc2"):
                in_count = clip_mlp_dim
            var module_name = String("text_model.encoder.layers.") + String(li) + String(".") + suffix
            _sd3_append_target(prefixes, in_features, out_features, roles, source_paths, sd3_text_encoder_1_lora_prefix(module_name), in_count, out_count, String("clip_l.") + suffix, clip_source)
            _sd3_append_target(prefixes, in_features, out_features, roles, source_paths, sd3_text_encoder_2_lora_prefix(module_name), in_count, out_count, String("clip_g.") + suffix, clip_source)

    var t5_suffixes = sd3_t5_linear_conversion_suffixes()
    for bi in range(t5_blocks):
        for si in range(len(t5_suffixes)):
            var suffix = t5_suffixes[si]
            var in_count = t5_dim
            var out_count = t5_dim
            if suffix == String("layer.1.DenseReluDense.wi_0") or suffix == String("layer.1.DenseReluDense.wi_1"):
                out_count = t5_mlp_dim
            elif suffix == String("layer.1.DenseReluDense.wo"):
                in_count = t5_mlp_dim
            var module_name = String("encoder.block.") + String(bi) + String(".") + suffix
            _sd3_append_target(prefixes, in_features, out_features, roles, source_paths, sd3_text_encoder_3_lora_prefix(module_name), in_count, out_count, String("t5.") + suffix, t5_source)

    return StableDiffusion3LoraTargetSpecs(prefixes^, in_features^, out_features^, roles^, source_paths^)

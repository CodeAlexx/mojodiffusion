# convert_flux_lora.mojo - build-only Flux LoRA conversion metadata.
#
# Source of truth:
#   /home/alex/Serenity/modules/util/convert/lora/convert_flux_lora.py
#   /home/alex/Serenity/modules/modelSaver/flux/FluxLoRASaver.py
#   /home/alex/Serenity/modules/modelLoader/flux/FluxLoRALoader.py
#
# This mirrors the Serenity conversion key families and file-level key helpers
# used by Flux LoRA loader/saver contract gates. It does not rewrite tensors,
# concatenate QKV data, cast dtype, or claim train/runtime parity.


comptime FLUX_LORA_BUNDLE_EMB_PREFIX = "bundle_emb"
comptime FLUX_LORA_TRANSFORMER_OMI_PREFIX = "transformer"
comptime FLUX_LORA_TRANSFORMER_DIFFUSERS_PREFIX = "lora_transformer"
comptime FLUX_LORA_CLIP_L_OMI_PREFIX = "clip_l"
comptime FLUX_LORA_CLIP_L_DIFFUSERS_PREFIX = "lora_te1"
comptime FLUX_LORA_T5_OMI_PREFIX = "t5"
comptime FLUX_LORA_T5_DIFFUSERS_PREFIX = "lora_te2"

comptime FLUX_LORA_RANGE_UPPER_BOUND = 100
comptime FLUX_LORA_TRANSFORMER_ROOT_RULE_COUNT = 10
comptime FLUX_LORA_DOUBLE_BLOCK_RULE_COUNT = 14
comptime FLUX_LORA_SINGLE_BLOCK_RULE_COUNT = 6
comptime FLUX_REAL_LORA_EXPECTED_KEYS = 1512
comptime FLUX_REAL_LORA_EXPECTED_ADAPTERS = 504
comptime FLUX_REAL_LORA_EXPECTED_RANK = 16


struct FluxLoraConversionSummary(Copyable, Movable, ImplicitlyCopyable):
    var has_convert_key_sets: Bool
    var root_bundle_embedding_prefix: String
    var transformer_omi_prefix: String
    var transformer_diffusers_prefix: String
    var clip_l_omi_prefix: String
    var clip_l_diffusers_prefix: String
    var t5_omi_prefix: String
    var t5_diffusers_prefix: String
    var range_upper_bound: Int
    var transformer_root_rule_count: Int
    var double_block_rule_count: Int
    var single_block_rule_count: Int
    var has_qkv_split_rules: Bool
    var has_swap_chunks_rules: Bool
    var has_filter_is_last_rules: Bool

    def __init__(out self):
        self.has_convert_key_sets = True
        self.root_bundle_embedding_prefix = String(FLUX_LORA_BUNDLE_EMB_PREFIX)
        self.transformer_omi_prefix = String(FLUX_LORA_TRANSFORMER_OMI_PREFIX)
        self.transformer_diffusers_prefix = String(
            FLUX_LORA_TRANSFORMER_DIFFUSERS_PREFIX
        )
        self.clip_l_omi_prefix = String(FLUX_LORA_CLIP_L_OMI_PREFIX)
        self.clip_l_diffusers_prefix = String(FLUX_LORA_CLIP_L_DIFFUSERS_PREFIX)
        self.t5_omi_prefix = String(FLUX_LORA_T5_OMI_PREFIX)
        self.t5_diffusers_prefix = String(FLUX_LORA_T5_DIFFUSERS_PREFIX)
        self.range_upper_bound = FLUX_LORA_RANGE_UPPER_BOUND
        self.transformer_root_rule_count = FLUX_LORA_TRANSFORMER_ROOT_RULE_COUNT
        self.double_block_rule_count = FLUX_LORA_DOUBLE_BLOCK_RULE_COUNT
        self.single_block_rule_count = FLUX_LORA_SINGLE_BLOCK_RULE_COUNT
        self.has_qkv_split_rules = True
        self.has_swap_chunks_rules = True
        self.has_filter_is_last_rules = False


struct FluxLoraRepresentativeSpecs(Movable):
    var diffusers_prefixes: List[String]
    var omi_prefixes: List[String]
    var legacy_prefixes: List[String]
    var roles: List[String]

    def __init__(
        out self,
        var diffusers_prefixes: List[String],
        var omi_prefixes: List[String],
        var legacy_prefixes: List[String],
        var roles: List[String],
    ):
        self.diffusers_prefixes = diffusers_prefixes^
        self.omi_prefixes = omi_prefixes^
        self.legacy_prefixes = legacy_prefixes^
        self.roles = roles^

    def len(self) -> Int:
        return len(self.diffusers_prefixes)


def flux_lora_conversion_summary() -> FluxLoraConversionSummary:
    return FluxLoraConversionSummary()


def flux_lora_candidate_files() -> List[String]:
    var paths = List[String]()
    paths.append(String("/home/alex/Serenity/output/flux1_100step_baseline/lora_last.safetensors"))
    paths.append(String("/home/alex/Serenity/output/flux1_100step_baseline/lora.safetensors"))
    return paths^


def flux_lora_down_key(prefix: String) -> String:
    return prefix + String(".lora_down.weight")


def flux_lora_up_key(prefix: String) -> String:
    return prefix + String(".lora_up.weight")


def flux_lora_alpha_key(prefix: String) -> String:
    return prefix + String(".alpha")


def flux_legacy_prefix_from_diffusers(prefix: String) -> String:
    """Serenity legacy safetensors output replaces prefix dots with underscores."""
    var out = String("")
    var bytes = prefix.as_bytes()
    for i in range(prefix.byte_length()):
        if Int(bytes[i]) == 46:
            out += String("_")
        else:
            out += chr(Int(bytes[i]))
    return out


def flux_lora_prefixed_module(wrapper_prefix: String, module_name: String) -> String:
    if wrapper_prefix == String():
        return module_name
    return wrapper_prefix + String(".") + module_name


def flux_lora_legacy_prefixed_module(
    wrapper_prefix: String, module_name: String
) -> String:
    return flux_legacy_prefix_from_diffusers(
        flux_lora_prefixed_module(wrapper_prefix, module_name)
    )


def _flux_append_target(
    mut diffusers_prefixes: List[String],
    mut omi_prefixes: List[String],
    mut legacy_prefixes: List[String],
    mut roles: List[String],
    diffusers_prefix: String,
    omi_prefix: String,
    role: String,
):
    diffusers_prefixes.append(diffusers_prefix.copy())
    omi_prefixes.append(omi_prefix.copy())
    legacy_prefixes.append(flux_legacy_prefix_from_diffusers(diffusers_prefix))
    roles.append(role.copy())


def flux_representative_lora_target_specs() -> FluxLoraRepresentativeSpecs:
    """Representative rules from Serenity convert_flux_lora.py.

    The real-file smoke verifies complete adapter counts from the saved file.
    This list checks representative conversion families and legacy key spelling.
    """
    var diffusers_prefixes = List[String]()
    var omi_prefixes = List[String]()
    var legacy_prefixes = List[String]()
    var roles = List[String]()

    _flux_append_target(
        diffusers_prefixes, omi_prefixes, legacy_prefixes, roles,
        flux_lora_prefixed_module(String(FLUX_LORA_TRANSFORMER_DIFFUSERS_PREFIX), String("context_embedder")),
        flux_lora_prefixed_module(String(FLUX_LORA_TRANSFORMER_OMI_PREFIX), String("txt_in")),
        String("transformer.txt_in"),
    )
    _flux_append_target(
        diffusers_prefixes, omi_prefixes, legacy_prefixes, roles,
        flux_lora_prefixed_module(String(FLUX_LORA_TRANSFORMER_DIFFUSERS_PREFIX), String("norm_out.linear")),
        flux_lora_prefixed_module(String(FLUX_LORA_TRANSFORMER_OMI_PREFIX), String("final_layer.adaLN_modulation.1")),
        String("transformer.final_layer.adaLN"),
    )
    _flux_append_target(
        diffusers_prefixes, omi_prefixes, legacy_prefixes, roles,
        flux_lora_prefixed_module(String(FLUX_LORA_TRANSFORMER_DIFFUSERS_PREFIX), String("proj_out")),
        flux_lora_prefixed_module(String(FLUX_LORA_TRANSFORMER_OMI_PREFIX), String("final_layer.linear")),
        String("transformer.final_layer.linear"),
    )
    _flux_append_target(
        diffusers_prefixes, omi_prefixes, legacy_prefixes, roles,
        flux_lora_prefixed_module(String(FLUX_LORA_TRANSFORMER_DIFFUSERS_PREFIX), String("x_embedder")),
        flux_lora_prefixed_module(String(FLUX_LORA_TRANSFORMER_OMI_PREFIX), String("img_in.proj")),
        String("transformer.img_in.proj"),
    )
    _flux_append_target(
        diffusers_prefixes, omi_prefixes, legacy_prefixes, roles,
        flux_lora_prefixed_module(String(FLUX_LORA_TRANSFORMER_DIFFUSERS_PREFIX), String("transformer_blocks.0.attn.to_q")),
        flux_lora_prefixed_module(String(FLUX_LORA_TRANSFORMER_OMI_PREFIX), String("double_blocks.0.img_attn.qkv.0")),
        String("double.img_attn.q"),
    )
    _flux_append_target(
        diffusers_prefixes, omi_prefixes, legacy_prefixes, roles,
        flux_lora_prefixed_module(String(FLUX_LORA_TRANSFORMER_DIFFUSERS_PREFIX), String("transformer_blocks.0.ff.net.0.proj")),
        flux_lora_prefixed_module(String(FLUX_LORA_TRANSFORMER_OMI_PREFIX), String("double_blocks.0.img_mlp.0")),
        String("double.img_mlp.in"),
    )
    _flux_append_target(
        diffusers_prefixes, omi_prefixes, legacy_prefixes, roles,
        flux_lora_prefixed_module(String(FLUX_LORA_TRANSFORMER_DIFFUSERS_PREFIX), String("transformer_blocks.0.ff.net.2")),
        flux_lora_prefixed_module(String(FLUX_LORA_TRANSFORMER_OMI_PREFIX), String("double_blocks.0.img_mlp.2")),
        String("double.img_mlp.out"),
    )
    _flux_append_target(
        diffusers_prefixes, omi_prefixes, legacy_prefixes, roles,
        flux_lora_prefixed_module(String(FLUX_LORA_TRANSFORMER_DIFFUSERS_PREFIX), String("single_transformer_blocks.0.attn.to_q")),
        flux_lora_prefixed_module(String(FLUX_LORA_TRANSFORMER_OMI_PREFIX), String("single_blocks.0.linear1.0")),
        String("single.attn.q"),
    )
    _flux_append_target(
        diffusers_prefixes, omi_prefixes, legacy_prefixes, roles,
        flux_lora_prefixed_module(String(FLUX_LORA_TRANSFORMER_DIFFUSERS_PREFIX), String("single_transformer_blocks.0.proj_out")),
        flux_lora_prefixed_module(String(FLUX_LORA_TRANSFORMER_OMI_PREFIX), String("single_blocks.0.linear2")),
        String("single.proj_out"),
    )
    _flux_append_target(
        diffusers_prefixes, omi_prefixes, legacy_prefixes, roles,
        flux_lora_prefixed_module(String(FLUX_LORA_CLIP_L_DIFFUSERS_PREFIX), String("text_model.encoder.layers.0.self_attn.q_proj")),
        flux_lora_prefixed_module(String(FLUX_LORA_CLIP_L_OMI_PREFIX), String("text_model.encoder.layers.0.self_attn.q_proj")),
        String("clip_l.self_attn.q"),
    )
    _flux_append_target(
        diffusers_prefixes, omi_prefixes, legacy_prefixes, roles,
        flux_lora_prefixed_module(String(FLUX_LORA_T5_DIFFUSERS_PREFIX), String("encoder.block.0.layer.0.SelfAttention.q")),
        flux_lora_prefixed_module(String(FLUX_LORA_T5_OMI_PREFIX), String("encoder.block.0.layer.0.SelfAttention.q")),
        String("t5.self_attention.q"),
    )

    return FluxLoraRepresentativeSpecs(
        diffusers_prefixes^, omi_prefixes^, legacy_prefixes^, roles^
    )

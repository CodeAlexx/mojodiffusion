# convert_chroma_lora.mojo - build-only Chroma LoRA conversion metadata.
#
# Source of truth:
#   /home/alex/Serenity/modules/util/convert/lora/convert_chroma_lora.py
#
# This mirrors the Serenity Chroma LoRA key-set contract only. It records the
# OMI, diffusers, legacy-diffusers, and bounded range metadata used by loader and
# saver contract gates. It does not rewrite tensors, split QKV values, cast dtype,
# or claim numeric parity.


comptime CHROMA_LORA_BUNDLE_EMB_PREFIX = "bundle_emb"
comptime CHROMA_LORA_TRANSFORMER_OMI_PREFIX = "transformer"
comptime CHROMA_LORA_TRANSFORMER_DIFFUSERS_PREFIX = "lora_transformer"
comptime CHROMA_LORA_T5_OMI_PREFIX = "t5"
comptime CHROMA_LORA_T5_DIFFUSERS_PREFIX = "lora_te"

comptime CHROMA_LORA_SOURCE_NAMESPACES = "omi,diffusers,legacy_diffusers"
comptime CHROMA_LORA_LOAD_TARGET_NAMESPACE = "diffusers"
comptime CHROMA_LORA_SAFETENSORS_SAVE_TARGET_NAMESPACE = "legacy_diffusers"
comptime CHROMA_LORA_INTERNAL_SAVE_TARGET_NAMESPACE = "omi"

comptime CHROMA_LORA_RANGE_UPPER_BOUND = 100
comptime CHROMA_LORA_TRANSFORMER_ROOT_RULE_COUNT = 3
comptime CHROMA_LORA_DOUBLE_BLOCK_RULE_COUNT = 12
comptime CHROMA_LORA_SINGLE_BLOCK_RULE_COUNT = 5
comptime CHROMA_LORA_DISTILLED_GUIDANCE_LAYER_RULE_COUNT = 2
comptime CHROMA_LORA_T5_BLOCK_RULE_COUNT = 7
comptime CHROMA_LORA_CONVERSION_KEY_SET_COUNT = (
    1
    + CHROMA_LORA_TRANSFORMER_ROOT_RULE_COUNT
    + CHROMA_LORA_RANGE_UPPER_BOUND * CHROMA_LORA_DOUBLE_BLOCK_RULE_COUNT
    + CHROMA_LORA_RANGE_UPPER_BOUND * CHROMA_LORA_SINGLE_BLOCK_RULE_COUNT
    + CHROMA_LORA_RANGE_UPPER_BOUND * CHROMA_LORA_DISTILLED_GUIDANCE_LAYER_RULE_COUNT
    + CHROMA_LORA_RANGE_UPPER_BOUND * CHROMA_LORA_T5_BLOCK_RULE_COUNT
)


struct ChromaLoraConversionKeySet(Copyable, Movable):
    var omi_prefix: String
    var diffusers_prefix: String
    var legacy_diffusers_prefix: String
    var next_omi_prefix: String
    var next_diffusers_prefix: String
    var next_legacy_diffusers_prefix: String
    var has_next_prefix: Bool
    var swap_chunks: Bool
    var has_filter_is_last: Bool
    var filter_is_last: Bool

    def __init__(
        out self,
        var omi_prefix: String,
        var diffusers_prefix: String,
        var next_omi_prefix: String,
        var next_diffusers_prefix: String,
        has_next_prefix: Bool,
    ):
        self.omi_prefix = omi_prefix^
        self.diffusers_prefix = diffusers_prefix^
        self.legacy_diffusers_prefix = chroma_lora_legacy_prefix_from_diffusers(
            self.diffusers_prefix
        )
        self.next_omi_prefix = next_omi_prefix^
        self.next_diffusers_prefix = next_diffusers_prefix^
        self.next_legacy_diffusers_prefix = chroma_lora_legacy_prefix_from_diffusers(
            self.next_diffusers_prefix
        )
        self.has_next_prefix = has_next_prefix
        self.swap_chunks = False
        self.has_filter_is_last = False
        self.filter_is_last = False


struct ChromaLoraConversionPlan(Copyable, Movable, ImplicitlyCopyable):
    var has_convert_key_sets: Bool
    var source_namespaces: String
    var load_target_namespace: String
    var safetensors_save_target_namespace: String
    var legacy_save_target_namespace: String
    var internal_save_target_namespace: String
    var root_bundle_embedding_prefix: String
    var transformer_omi_prefix: String
    var transformer_diffusers_prefix: String
    var t5_omi_prefix: String
    var t5_diffusers_prefix: String
    var range_upper_bound: Int
    var transformer_root_rule_count: Int
    var double_block_rule_count: Int
    var single_block_rule_count: Int
    var distilled_guidance_layer_rule_count: Int
    var t5_block_rule_count: Int
    var bounded_conversion_key_set_count: Int
    var has_img_qkv_prefix_rules: Bool
    var has_txt_qkv_prefix_rules: Bool
    var has_distilled_guidance_layer_rules: Bool
    var has_t5_map_rules: Bool
    var has_swap_chunks_rules: Bool
    var has_filter_is_last_rules: Bool

    def __init__(out self):
        self.has_convert_key_sets = True
        self.source_namespaces = String(CHROMA_LORA_SOURCE_NAMESPACES)
        self.load_target_namespace = String(CHROMA_LORA_LOAD_TARGET_NAMESPACE)
        self.safetensors_save_target_namespace = String(
            CHROMA_LORA_SAFETENSORS_SAVE_TARGET_NAMESPACE
        )
        self.legacy_save_target_namespace = String(
            CHROMA_LORA_SAFETENSORS_SAVE_TARGET_NAMESPACE
        )
        self.internal_save_target_namespace = String(
            CHROMA_LORA_INTERNAL_SAVE_TARGET_NAMESPACE
        )
        self.root_bundle_embedding_prefix = String(CHROMA_LORA_BUNDLE_EMB_PREFIX)
        self.transformer_omi_prefix = String(CHROMA_LORA_TRANSFORMER_OMI_PREFIX)
        self.transformer_diffusers_prefix = String(
            CHROMA_LORA_TRANSFORMER_DIFFUSERS_PREFIX
        )
        self.t5_omi_prefix = String(CHROMA_LORA_T5_OMI_PREFIX)
        self.t5_diffusers_prefix = String(CHROMA_LORA_T5_DIFFUSERS_PREFIX)
        self.range_upper_bound = CHROMA_LORA_RANGE_UPPER_BOUND
        self.transformer_root_rule_count = CHROMA_LORA_TRANSFORMER_ROOT_RULE_COUNT
        self.double_block_rule_count = CHROMA_LORA_DOUBLE_BLOCK_RULE_COUNT
        self.single_block_rule_count = CHROMA_LORA_SINGLE_BLOCK_RULE_COUNT
        self.distilled_guidance_layer_rule_count = (
            CHROMA_LORA_DISTILLED_GUIDANCE_LAYER_RULE_COUNT
        )
        self.t5_block_rule_count = CHROMA_LORA_T5_BLOCK_RULE_COUNT
        self.bounded_conversion_key_set_count = CHROMA_LORA_CONVERSION_KEY_SET_COUNT
        self.has_img_qkv_prefix_rules = True
        self.has_txt_qkv_prefix_rules = True
        self.has_distilled_guidance_layer_rules = True
        self.has_t5_map_rules = True
        self.has_swap_chunks_rules = False
        self.has_filter_is_last_rules = False


struct ChromaLoraRepresentativeSpecs(Movable):
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


def chroma_lora_conversion_plan() -> ChromaLoraConversionPlan:
    return ChromaLoraConversionPlan()


def chroma_lora_combine(left: String, right: String) -> String:
    if left.byte_length() == 0:
        return right.copy()
    if right.byte_length() == 0:
        return left.copy()
    return left + String(".") + right


def chroma_lora_range_prefix(root: String, child: String, index: Int) -> String:
    return chroma_lora_combine(
        root,
        chroma_lora_combine(child, String(index)),
    )


def chroma_lora_legacy_prefix_from_diffusers(prefix: String) -> String:
    """Serenity legacy LoRA output replaces prefix dots with underscores."""
    var out = String("")
    var bytes = prefix.as_bytes()
    for i in range(prefix.byte_length()):
        if Int(bytes[i]) == 46:
            out += String("_")
        else:
            out += chr(Int(bytes[i]))
    return out


def chroma_lora_down_key(prefix: String) -> String:
    return prefix + String(".lora_down.weight")


def chroma_lora_up_key(prefix: String) -> String:
    return prefix + String(".lora_up.weight")


def chroma_lora_alpha_key(prefix: String) -> String:
    return prefix + String(".alpha")


def chroma_lora_prefixed_module(wrapper_prefix: String, module_name: String) -> String:
    if wrapper_prefix == String():
        return module_name.copy()
    return wrapper_prefix + String(".") + module_name


def chroma_lora_legacy_prefixed_module(
    wrapper_prefix: String, module_name: String
) -> String:
    return chroma_lora_legacy_prefix_from_diffusers(
        chroma_lora_prefixed_module(wrapper_prefix, module_name)
    )


def _chroma_append_key(
    mut keys: List[ChromaLoraConversionKeySet],
    omi_prefix: String,
    diffusers_prefix: String,
):
    keys.append(
        ChromaLoraConversionKeySet(
            omi_prefix.copy(),
            diffusers_prefix.copy(),
            String(),
            String(),
            False,
        )
    )


def _chroma_append_child_key(
    mut keys: List[ChromaLoraConversionKeySet],
    parent_omi_prefix: String,
    parent_diffusers_prefix: String,
    parent_next_omi_prefix: String,
    parent_next_diffusers_prefix: String,
    omi_suffix: String,
    diffusers_suffix: String,
):
    var has_next_prefix = (
        parent_next_omi_prefix.byte_length() > 0
        or parent_next_diffusers_prefix.byte_length() > 0
    )
    keys.append(
        ChromaLoraConversionKeySet(
            chroma_lora_combine(parent_omi_prefix, omi_suffix),
            chroma_lora_combine(parent_diffusers_prefix, diffusers_suffix),
            parent_next_omi_prefix.copy(),
            parent_next_diffusers_prefix.copy(),
            has_next_prefix,
        )
    )


def _chroma_append_double_transformer_block(
    mut keys: List[ChromaLoraConversionKeySet],
    parent_omi_prefix: String,
    parent_diffusers_prefix: String,
    parent_next_omi_prefix: String,
    parent_next_diffusers_prefix: String,
):
    _chroma_append_child_key(keys, parent_omi_prefix, parent_diffusers_prefix, parent_next_omi_prefix, parent_next_diffusers_prefix, String("img_attn.qkv.0"), String("attn.to_q"))
    _chroma_append_child_key(keys, parent_omi_prefix, parent_diffusers_prefix, parent_next_omi_prefix, parent_next_diffusers_prefix, String("img_attn.qkv.1"), String("attn.to_k"))
    _chroma_append_child_key(keys, parent_omi_prefix, parent_diffusers_prefix, parent_next_omi_prefix, parent_next_diffusers_prefix, String("img_attn.qkv.2"), String("attn.to_v"))

    _chroma_append_child_key(keys, parent_omi_prefix, parent_diffusers_prefix, parent_next_omi_prefix, parent_next_diffusers_prefix, String("txt_attn.qkv.0"), String("attn.add_q_proj"))
    _chroma_append_child_key(keys, parent_omi_prefix, parent_diffusers_prefix, parent_next_omi_prefix, parent_next_diffusers_prefix, String("txt_attn.qkv.1"), String("attn.add_k_proj"))
    _chroma_append_child_key(keys, parent_omi_prefix, parent_diffusers_prefix, parent_next_omi_prefix, parent_next_diffusers_prefix, String("txt_attn.qkv.2"), String("attn.add_v_proj"))

    _chroma_append_child_key(keys, parent_omi_prefix, parent_diffusers_prefix, parent_next_omi_prefix, parent_next_diffusers_prefix, String("img_attn.proj"), String("attn.to_out.0"))
    _chroma_append_child_key(keys, parent_omi_prefix, parent_diffusers_prefix, parent_next_omi_prefix, parent_next_diffusers_prefix, String("img_mlp.0"), String("ff.net.0.proj"))
    _chroma_append_child_key(keys, parent_omi_prefix, parent_diffusers_prefix, parent_next_omi_prefix, parent_next_diffusers_prefix, String("img_mlp.2"), String("ff.net.2"))

    _chroma_append_child_key(keys, parent_omi_prefix, parent_diffusers_prefix, parent_next_omi_prefix, parent_next_diffusers_prefix, String("txt_attn.proj"), String("attn.to_add_out"))
    _chroma_append_child_key(keys, parent_omi_prefix, parent_diffusers_prefix, parent_next_omi_prefix, parent_next_diffusers_prefix, String("txt_mlp.0"), String("ff_context.net.0.proj"))
    _chroma_append_child_key(keys, parent_omi_prefix, parent_diffusers_prefix, parent_next_omi_prefix, parent_next_diffusers_prefix, String("txt_mlp.2"), String("ff_context.net.2"))


def _chroma_append_single_transformer_block(
    mut keys: List[ChromaLoraConversionKeySet],
    parent_omi_prefix: String,
    parent_diffusers_prefix: String,
    parent_next_omi_prefix: String,
    parent_next_diffusers_prefix: String,
):
    _chroma_append_child_key(keys, parent_omi_prefix, parent_diffusers_prefix, parent_next_omi_prefix, parent_next_diffusers_prefix, String("linear1.0"), String("attn.to_q"))
    _chroma_append_child_key(keys, parent_omi_prefix, parent_diffusers_prefix, parent_next_omi_prefix, parent_next_diffusers_prefix, String("linear1.1"), String("attn.to_k"))
    _chroma_append_child_key(keys, parent_omi_prefix, parent_diffusers_prefix, parent_next_omi_prefix, parent_next_diffusers_prefix, String("linear1.2"), String("attn.to_v"))
    _chroma_append_child_key(keys, parent_omi_prefix, parent_diffusers_prefix, parent_next_omi_prefix, parent_next_diffusers_prefix, String("linear1.3"), String("proj_mlp"))
    _chroma_append_child_key(keys, parent_omi_prefix, parent_diffusers_prefix, parent_next_omi_prefix, parent_next_diffusers_prefix, String("linear2"), String("proj_out"))


def _chroma_append_distilled_guidance_layer(
    mut keys: List[ChromaLoraConversionKeySet],
    parent_omi_prefix: String,
    parent_diffusers_prefix: String,
    parent_next_omi_prefix: String,
    parent_next_diffusers_prefix: String,
):
    _chroma_append_child_key(keys, parent_omi_prefix, parent_diffusers_prefix, parent_next_omi_prefix, parent_next_diffusers_prefix, String("in_layer"), String("linear_1"))
    _chroma_append_child_key(keys, parent_omi_prefix, parent_diffusers_prefix, parent_next_omi_prefix, parent_next_diffusers_prefix, String("out_layer"), String("linear_2"))


def _chroma_append_t5_block(
    mut keys: List[ChromaLoraConversionKeySet],
    parent_omi_prefix: String,
    parent_diffusers_prefix: String,
    parent_next_omi_prefix: String,
    parent_next_diffusers_prefix: String,
):
    _chroma_append_child_key(keys, parent_omi_prefix, parent_diffusers_prefix, parent_next_omi_prefix, parent_next_diffusers_prefix, String("layer.0.SelfAttention.k"), String("layer.0.SelfAttention.k"))
    _chroma_append_child_key(keys, parent_omi_prefix, parent_diffusers_prefix, parent_next_omi_prefix, parent_next_diffusers_prefix, String("layer.0.SelfAttention.o"), String("layer.0.SelfAttention.o"))
    _chroma_append_child_key(keys, parent_omi_prefix, parent_diffusers_prefix, parent_next_omi_prefix, parent_next_diffusers_prefix, String("layer.0.SelfAttention.q"), String("layer.0.SelfAttention.q"))
    _chroma_append_child_key(keys, parent_omi_prefix, parent_diffusers_prefix, parent_next_omi_prefix, parent_next_diffusers_prefix, String("layer.0.SelfAttention.v"), String("layer.0.SelfAttention.v"))
    _chroma_append_child_key(keys, parent_omi_prefix, parent_diffusers_prefix, parent_next_omi_prefix, parent_next_diffusers_prefix, String("layer.1.DenseReluDense.wi_0"), String("layer.1.DenseReluDense.wi_0"))
    _chroma_append_child_key(keys, parent_omi_prefix, parent_diffusers_prefix, parent_next_omi_prefix, parent_next_diffusers_prefix, String("layer.1.DenseReluDense.wi_1"), String("layer.1.DenseReluDense.wi_1"))
    _chroma_append_child_key(keys, parent_omi_prefix, parent_diffusers_prefix, parent_next_omi_prefix, parent_next_diffusers_prefix, String("layer.1.DenseReluDense.wo"), String("layer.1.DenseReluDense.wo"))


def _chroma_append_transformer_keys(mut keys: List[ChromaLoraConversionKeySet]):
    var transformer_omi = String(CHROMA_LORA_TRANSFORMER_OMI_PREFIX)
    var transformer_diffusers = String(CHROMA_LORA_TRANSFORMER_DIFFUSERS_PREFIX)

    _chroma_append_child_key(keys, transformer_omi, transformer_diffusers, String(), String(), String("txt_in"), String("context_embedder"))
    _chroma_append_child_key(keys, transformer_omi, transformer_diffusers, String(), String(), String("final_layer.linear"), String("proj_out"))
    _chroma_append_child_key(keys, transformer_omi, transformer_diffusers, String(), String(), String("img_in.proj"), String("x_embedder"))

    for i in range(CHROMA_LORA_RANGE_UPPER_BOUND):
        var parent_omi = chroma_lora_range_prefix(transformer_omi, String("double_blocks"), i)
        var parent_diffusers = chroma_lora_range_prefix(transformer_diffusers, String("transformer_blocks"), i)
        var next_omi = chroma_lora_range_prefix(transformer_omi, String("double_blocks"), i + 1)
        var next_diffusers = chroma_lora_range_prefix(transformer_diffusers, String("transformer_blocks"), i + 1)
        _chroma_append_double_transformer_block(keys, parent_omi, parent_diffusers, next_omi, next_diffusers)

    for i in range(CHROMA_LORA_RANGE_UPPER_BOUND):
        var parent_omi = chroma_lora_range_prefix(transformer_omi, String("single_blocks"), i)
        var parent_diffusers = chroma_lora_range_prefix(transformer_diffusers, String("single_transformer_blocks"), i)
        var next_omi = chroma_lora_range_prefix(transformer_omi, String("single_blocks"), i + 1)
        var next_diffusers = chroma_lora_range_prefix(transformer_diffusers, String("single_transformer_blocks"), i + 1)
        _chroma_append_single_transformer_block(keys, parent_omi, parent_diffusers, next_omi, next_diffusers)

    for i in range(CHROMA_LORA_RANGE_UPPER_BOUND):
        var parent_omi = chroma_lora_range_prefix(transformer_omi, String("distilled_guidance_layer.layers"), i)
        var parent_diffusers = chroma_lora_range_prefix(transformer_diffusers, String("distilled_guidance_layer.layers"), i)
        var next_omi = chroma_lora_range_prefix(transformer_omi, String("distilled_guidance_layer.layers"), i + 1)
        var next_diffusers = chroma_lora_range_prefix(transformer_diffusers, String("distilled_guidance_layer.layers"), i + 1)
        _chroma_append_distilled_guidance_layer(keys, parent_omi, parent_diffusers, next_omi, next_diffusers)


def _chroma_append_t5_keys(mut keys: List[ChromaLoraConversionKeySet]):
    var t5_omi = String(CHROMA_LORA_T5_OMI_PREFIX)
    var t5_diffusers = String(CHROMA_LORA_T5_DIFFUSERS_PREFIX)
    for i in range(CHROMA_LORA_RANGE_UPPER_BOUND):
        var parent_omi = chroma_lora_range_prefix(t5_omi, String("encoder.block"), i)
        var parent_diffusers = chroma_lora_range_prefix(t5_diffusers, String("encoder.block"), i)
        var next_omi = chroma_lora_range_prefix(t5_omi, String("encoder.block"), i + 1)
        var next_diffusers = chroma_lora_range_prefix(t5_diffusers, String("encoder.block"), i + 1)
        _chroma_append_t5_block(keys, parent_omi, parent_diffusers, next_omi, next_diffusers)


def convert_chroma_lora_key_sets() -> List[ChromaLoraConversionKeySet]:
    var keys = List[ChromaLoraConversionKeySet]()
    _chroma_append_key(
        keys,
        String(CHROMA_LORA_BUNDLE_EMB_PREFIX),
        String(CHROMA_LORA_BUNDLE_EMB_PREFIX),
    )
    _chroma_append_transformer_keys(keys)
    _chroma_append_t5_keys(keys)
    return keys^


def _chroma_append_target(
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
    legacy_prefixes.append(chroma_lora_legacy_prefix_from_diffusers(diffusers_prefix))
    roles.append(role.copy())


def chroma_representative_lora_target_specs() -> ChromaLoraRepresentativeSpecs:
    """Representative rules from Serenity convert_chroma_lora.py.

    This is a source-contract sample for smoke checks, not a complete tensor
    conversion or numeric parity claim. The complete bounded metadata is exposed
    by convert_chroma_lora_key_sets().
    """
    var diffusers_prefixes = List[String]()
    var omi_prefixes = List[String]()
    var legacy_prefixes = List[String]()
    var roles = List[String]()

    _chroma_append_target(
        diffusers_prefixes, omi_prefixes, legacy_prefixes, roles,
        chroma_lora_prefixed_module(String(CHROMA_LORA_BUNDLE_EMB_PREFIX), String("token.t5")),
        chroma_lora_prefixed_module(String(CHROMA_LORA_BUNDLE_EMB_PREFIX), String("token.t5")),
        String("bundle_emb.t5"),
    )
    _chroma_append_target(
        diffusers_prefixes, omi_prefixes, legacy_prefixes, roles,
        chroma_lora_prefixed_module(String(CHROMA_LORA_TRANSFORMER_DIFFUSERS_PREFIX), String("context_embedder")),
        chroma_lora_prefixed_module(String(CHROMA_LORA_TRANSFORMER_OMI_PREFIX), String("txt_in")),
        String("transformer.txt_in"),
    )
    _chroma_append_target(
        diffusers_prefixes, omi_prefixes, legacy_prefixes, roles,
        chroma_lora_prefixed_module(String(CHROMA_LORA_TRANSFORMER_DIFFUSERS_PREFIX), String("proj_out")),
        chroma_lora_prefixed_module(String(CHROMA_LORA_TRANSFORMER_OMI_PREFIX), String("final_layer.linear")),
        String("transformer.final_layer.linear"),
    )
    _chroma_append_target(
        diffusers_prefixes, omi_prefixes, legacy_prefixes, roles,
        chroma_lora_prefixed_module(String(CHROMA_LORA_TRANSFORMER_DIFFUSERS_PREFIX), String("x_embedder")),
        chroma_lora_prefixed_module(String(CHROMA_LORA_TRANSFORMER_OMI_PREFIX), String("img_in.proj")),
        String("transformer.img_in.proj"),
    )
    _chroma_append_target(
        diffusers_prefixes, omi_prefixes, legacy_prefixes, roles,
        chroma_lora_prefixed_module(String(CHROMA_LORA_TRANSFORMER_DIFFUSERS_PREFIX), String("transformer_blocks.0.attn.to_q")),
        chroma_lora_prefixed_module(String(CHROMA_LORA_TRANSFORMER_OMI_PREFIX), String("double_blocks.0.img_attn.qkv.0")),
        String("double.img_attn.q"),
    )
    _chroma_append_target(
        diffusers_prefixes, omi_prefixes, legacy_prefixes, roles,
        chroma_lora_prefixed_module(String(CHROMA_LORA_TRANSFORMER_DIFFUSERS_PREFIX), String("transformer_blocks.0.attn.add_q_proj")),
        chroma_lora_prefixed_module(String(CHROMA_LORA_TRANSFORMER_OMI_PREFIX), String("double_blocks.0.txt_attn.qkv.0")),
        String("double.txt_attn.q"),
    )
    _chroma_append_target(
        diffusers_prefixes, omi_prefixes, legacy_prefixes, roles,
        chroma_lora_prefixed_module(String(CHROMA_LORA_TRANSFORMER_DIFFUSERS_PREFIX), String("transformer_blocks.0.ff.net.0.proj")),
        chroma_lora_prefixed_module(String(CHROMA_LORA_TRANSFORMER_OMI_PREFIX), String("double_blocks.0.img_mlp.0")),
        String("double.img_mlp.in"),
    )
    _chroma_append_target(
        diffusers_prefixes, omi_prefixes, legacy_prefixes, roles,
        chroma_lora_prefixed_module(String(CHROMA_LORA_TRANSFORMER_DIFFUSERS_PREFIX), String("single_transformer_blocks.0.attn.to_q")),
        chroma_lora_prefixed_module(String(CHROMA_LORA_TRANSFORMER_OMI_PREFIX), String("single_blocks.0.linear1.0")),
        String("single.attn.q"),
    )
    _chroma_append_target(
        diffusers_prefixes, omi_prefixes, legacy_prefixes, roles,
        chroma_lora_prefixed_module(String(CHROMA_LORA_TRANSFORMER_DIFFUSERS_PREFIX), String("single_transformer_blocks.0.proj_mlp")),
        chroma_lora_prefixed_module(String(CHROMA_LORA_TRANSFORMER_OMI_PREFIX), String("single_blocks.0.linear1.3")),
        String("single.proj_mlp"),
    )
    _chroma_append_target(
        diffusers_prefixes, omi_prefixes, legacy_prefixes, roles,
        chroma_lora_prefixed_module(String(CHROMA_LORA_TRANSFORMER_DIFFUSERS_PREFIX), String("distilled_guidance_layer.layers.0.linear_1")),
        chroma_lora_prefixed_module(String(CHROMA_LORA_TRANSFORMER_OMI_PREFIX), String("distilled_guidance_layer.layers.0.in_layer")),
        String("distilled_guidance.in_layer"),
    )
    _chroma_append_target(
        diffusers_prefixes, omi_prefixes, legacy_prefixes, roles,
        chroma_lora_prefixed_module(String(CHROMA_LORA_T5_DIFFUSERS_PREFIX), String("encoder.block.0.layer.0.SelfAttention.q")),
        chroma_lora_prefixed_module(String(CHROMA_LORA_T5_OMI_PREFIX), String("encoder.block.0.layer.0.SelfAttention.q")),
        String("t5.self_attention.q"),
    )
    _chroma_append_target(
        diffusers_prefixes, omi_prefixes, legacy_prefixes, roles,
        chroma_lora_prefixed_module(String(CHROMA_LORA_T5_DIFFUSERS_PREFIX), String("encoder.block.0.layer.1.DenseReluDense.wo")),
        chroma_lora_prefixed_module(String(CHROMA_LORA_T5_OMI_PREFIX), String("encoder.block.0.layer.1.DenseReluDense.wo")),
        String("t5.dense.wo"),
    )

    return ChromaLoraRepresentativeSpecs(
        diffusers_prefixes^, omi_prefixes^, legacy_prefixes^, roles^
    )

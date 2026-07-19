# chromaLoraTargets.mojo -- Chroma LoRA target metadata and key helpers.
#
# Serenity references:
#   modules/modelSetup/ChromaLoRASetup.py wraps optional T5 LoRA with prefix
#   "lora_te" and always wraps the transformer with prefix "lora_transformer"
#   plus config.layer_filter.split(",").
#   modules/modelSaver/chroma/ChromaLoRASaver.py returns
#   convert_chroma_lora_key_sets() and saves LoRAModuleWrapper.state_dict().
#   modules/util/convert/lora/convert_chroma_lora.py maps OMI
#   "transformer.*" / "t5.*" keys to raw Serenity
#   "lora_transformer.*" / "lora_te.*" keys. Legacy safetensors output uses
#   the same OMI prefixes with dots changed to underscores.
#   modules/module/LoRAModule.py writes:
#     <prefix>.<module>.lora_down.weight
#     <prefix>.<module>.lora_up.weight
#     <prefix>.<module>.alpha
#
# This file is host metadata for Chroma LoRA key/shape gates. No local real
# Serenity Chroma LoRA file was found during the 2026-06-05 audit, so the
# first smoke gate is blocker-aware and only claims key-construction coverage.


comptime CHROMA_LORA_TEXT_ENCODER_PREFIX = "lora_te"
comptime CHROMA_LORA_TRANSFORMER_PREFIX = "lora_transformer"
comptime CHROMA_OMI_TRANSFORMER_PREFIX = "transformer"
comptime CHROMA_OMI_T5_PREFIX = "t5"

comptime CHROMA_EXPECTED_DEFAULT_RANK = 16
comptime CHROMA_EXPECTED_DEFAULT_ALPHA = Float32(1.0)


struct ChromaLoraTargetSpecs(Movable):
    var raw_prefixes: List[String]
    var omi_prefixes: List[String]
    var legacy_prefixes: List[String]
    var roles: List[String]
    var source_paths: List[String]

    def __init__(
        out self,
        var raw_prefixes: List[String],
        var omi_prefixes: List[String],
        var legacy_prefixes: List[String],
        var roles: List[String],
        var source_paths: List[String],
    ):
        self.raw_prefixes = raw_prefixes^
        self.omi_prefixes = omi_prefixes^
        self.legacy_prefixes = legacy_prefixes^
        self.roles = roles^
        self.source_paths = source_paths^

    def len(self) -> Int:
        return len(self.raw_prefixes)


def chroma_lora_candidate_files() -> List[String]:
    var paths = List[String]()
    paths.append(String("/home/alex/Serenity/output/chroma_100step_baseline/lora.safetensors"))
    paths.append(String("/home/alex/Serenity/output/chroma_100step_baseline/lora_last.safetensors"))
    paths.append(String("/home/alex/Serenity/output/chroma_lora_100step_baseline/lora.safetensors"))
    paths.append(String("/home/alex/Serenity/output/chroma_lora_100step_baseline/lora_last.safetensors"))
    return paths^


def chroma_lora_prefixed_module(wrapper_prefix: String, module_name: String) -> String:
    if wrapper_prefix == String():
        return module_name
    return wrapper_prefix + String(".") + module_name


def chroma_transformer_lora_prefix(module_name: String) -> String:
    return chroma_lora_prefixed_module(String(CHROMA_LORA_TRANSFORMER_PREFIX), module_name)


def chroma_text_encoder_lora_prefix(module_name: String) -> String:
    return chroma_lora_prefixed_module(String(CHROMA_LORA_TEXT_ENCODER_PREFIX), module_name)


def chroma_omi_transformer_prefix(module_name: String) -> String:
    return chroma_lora_prefixed_module(String(CHROMA_OMI_TRANSFORMER_PREFIX), module_name)


def chroma_omi_t5_prefix(module_name: String) -> String:
    return chroma_lora_prefixed_module(String(CHROMA_OMI_T5_PREFIX), module_name)


def chroma_lora_down_key(prefix: String) -> String:
    return prefix + String(".lora_down.weight")


def chroma_lora_up_key(prefix: String) -> String:
    return prefix + String(".lora_up.weight")


def chroma_lora_alpha_key(prefix: String) -> String:
    return prefix + String(".alpha")


def chroma_legacy_prefix_from_omi(prefix: String) -> String:
    """Serenity legacy LoRA conversion turns OMI prefix dots into underscores.

    Tensor key names are ASCII, so byte-wise reconstruction is exact here.
    """
    var out = String("")
    var bytes = prefix.as_bytes()
    for i in range(prefix.byte_length()):
        if Int(bytes[i]) == 46:
            out += String("_")
        else:
            out += chr(Int(bytes[i]))
    return out


def _chroma_append_target(
    mut raw_prefixes: List[String],
    mut omi_prefixes: List[String],
    mut legacy_prefixes: List[String],
    mut roles: List[String],
    mut source_paths: List[String],
    raw_prefix: String,
    omi_prefix: String,
    role: String,
    source_path: String,
):
    raw_prefixes.append(raw_prefix.copy())
    omi_prefixes.append(omi_prefix.copy())
    legacy_prefixes.append(chroma_legacy_prefix_from_omi(omi_prefix))
    roles.append(role.copy())
    source_paths.append(source_path.copy())


def chroma_representative_lora_target_specs() -> ChromaLoraTargetSpecs:
    """Representative Chroma targets from Serenity conversion code.

    These are conversion/key-construction checks only. Counts and complete
    target inventory require a real Chroma model instance or Serenity-saved
    Chroma LoRA file.
    """
    var raw_prefixes = List[String]()
    var omi_prefixes = List[String]()
    var legacy_prefixes = List[String]()
    var roles = List[String]()
    var source_paths = List[String]()
    var source = String("/home/alex/Serenity/modules/modelSetup/ChromaLoRASetup.py | /home/alex/Serenity/modules/modelSaver/chroma/ChromaLoRASaver.py | /home/alex/Serenity/modules/util/convert/lora/convert_chroma_lora.py | /home/alex/Serenity/modules/module/LoRAModule.py")

    _chroma_append_target(raw_prefixes, omi_prefixes, legacy_prefixes, roles, source_paths,
        chroma_transformer_lora_prefix(String("context_embedder")),
        chroma_omi_transformer_prefix(String("txt_in")),
        String("transformer.txt_in"),
        source,
    )
    _chroma_append_target(raw_prefixes, omi_prefixes, legacy_prefixes, roles, source_paths,
        chroma_transformer_lora_prefix(String("x_embedder")),
        chroma_omi_transformer_prefix(String("img_in.proj")),
        String("transformer.img_in.proj"),
        source,
    )
    _chroma_append_target(raw_prefixes, omi_prefixes, legacy_prefixes, roles, source_paths,
        chroma_transformer_lora_prefix(String("proj_out")),
        chroma_omi_transformer_prefix(String("final_layer.linear")),
        String("transformer.final_layer.linear"),
        source,
    )
    _chroma_append_target(raw_prefixes, omi_prefixes, legacy_prefixes, roles, source_paths,
        chroma_transformer_lora_prefix(String("transformer_blocks.0.attn.to_q")),
        chroma_omi_transformer_prefix(String("double_blocks.0.img_attn.qkv.0")),
        String("double.img_attn.q"),
        source,
    )
    _chroma_append_target(raw_prefixes, omi_prefixes, legacy_prefixes, roles, source_paths,
        chroma_transformer_lora_prefix(String("transformer_blocks.0.attn.to_k")),
        chroma_omi_transformer_prefix(String("double_blocks.0.img_attn.qkv.1")),
        String("double.img_attn.k"),
        source,
    )
    _chroma_append_target(raw_prefixes, omi_prefixes, legacy_prefixes, roles, source_paths,
        chroma_transformer_lora_prefix(String("transformer_blocks.0.attn.to_v")),
        chroma_omi_transformer_prefix(String("double_blocks.0.img_attn.qkv.2")),
        String("double.img_attn.v"),
        source,
    )
    _chroma_append_target(raw_prefixes, omi_prefixes, legacy_prefixes, roles, source_paths,
        chroma_transformer_lora_prefix(String("transformer_blocks.0.attn.add_q_proj")),
        chroma_omi_transformer_prefix(String("double_blocks.0.txt_attn.qkv.0")),
        String("double.txt_attn.q"),
        source,
    )
    _chroma_append_target(raw_prefixes, omi_prefixes, legacy_prefixes, roles, source_paths,
        chroma_transformer_lora_prefix(String("transformer_blocks.0.attn.add_k_proj")),
        chroma_omi_transformer_prefix(String("double_blocks.0.txt_attn.qkv.1")),
        String("double.txt_attn.k"),
        source,
    )
    _chroma_append_target(raw_prefixes, omi_prefixes, legacy_prefixes, roles, source_paths,
        chroma_transformer_lora_prefix(String("transformer_blocks.0.attn.add_v_proj")),
        chroma_omi_transformer_prefix(String("double_blocks.0.txt_attn.qkv.2")),
        String("double.txt_attn.v"),
        source,
    )
    _chroma_append_target(raw_prefixes, omi_prefixes, legacy_prefixes, roles, source_paths,
        chroma_transformer_lora_prefix(String("transformer_blocks.0.attn.to_out.0")),
        chroma_omi_transformer_prefix(String("double_blocks.0.img_attn.proj")),
        String("double.img_attn.proj"),
        source,
    )
    _chroma_append_target(raw_prefixes, omi_prefixes, legacy_prefixes, roles, source_paths,
        chroma_transformer_lora_prefix(String("transformer_blocks.0.ff.net.0.proj")),
        chroma_omi_transformer_prefix(String("double_blocks.0.img_mlp.0")),
        String("double.img_mlp.in"),
        source,
    )
    _chroma_append_target(raw_prefixes, omi_prefixes, legacy_prefixes, roles, source_paths,
        chroma_transformer_lora_prefix(String("transformer_blocks.0.ff.net.2")),
        chroma_omi_transformer_prefix(String("double_blocks.0.img_mlp.2")),
        String("double.img_mlp.out"),
        source,
    )
    _chroma_append_target(raw_prefixes, omi_prefixes, legacy_prefixes, roles, source_paths,
        chroma_transformer_lora_prefix(String("transformer_blocks.0.attn.to_add_out")),
        chroma_omi_transformer_prefix(String("double_blocks.0.txt_attn.proj")),
        String("double.txt_attn.proj"),
        source,
    )
    _chroma_append_target(raw_prefixes, omi_prefixes, legacy_prefixes, roles, source_paths,
        chroma_transformer_lora_prefix(String("transformer_blocks.0.ff_context.net.0.proj")),
        chroma_omi_transformer_prefix(String("double_blocks.0.txt_mlp.0")),
        String("double.txt_mlp.in"),
        source,
    )
    _chroma_append_target(raw_prefixes, omi_prefixes, legacy_prefixes, roles, source_paths,
        chroma_transformer_lora_prefix(String("transformer_blocks.0.ff_context.net.2")),
        chroma_omi_transformer_prefix(String("double_blocks.0.txt_mlp.2")),
        String("double.txt_mlp.out"),
        source,
    )
    _chroma_append_target(raw_prefixes, omi_prefixes, legacy_prefixes, roles, source_paths,
        chroma_transformer_lora_prefix(String("single_transformer_blocks.0.attn.to_q")),
        chroma_omi_transformer_prefix(String("single_blocks.0.linear1.0")),
        String("single.attn.q"),
        source,
    )
    _chroma_append_target(raw_prefixes, omi_prefixes, legacy_prefixes, roles, source_paths,
        chroma_transformer_lora_prefix(String("single_transformer_blocks.0.attn.to_k")),
        chroma_omi_transformer_prefix(String("single_blocks.0.linear1.1")),
        String("single.attn.k"),
        source,
    )
    _chroma_append_target(raw_prefixes, omi_prefixes, legacy_prefixes, roles, source_paths,
        chroma_transformer_lora_prefix(String("single_transformer_blocks.0.attn.to_v")),
        chroma_omi_transformer_prefix(String("single_blocks.0.linear1.2")),
        String("single.attn.v"),
        source,
    )
    _chroma_append_target(raw_prefixes, omi_prefixes, legacy_prefixes, roles, source_paths,
        chroma_transformer_lora_prefix(String("single_transformer_blocks.0.proj_mlp")),
        chroma_omi_transformer_prefix(String("single_blocks.0.linear1.3")),
        String("single.mlp.in"),
        source,
    )
    _chroma_append_target(raw_prefixes, omi_prefixes, legacy_prefixes, roles, source_paths,
        chroma_transformer_lora_prefix(String("single_transformer_blocks.0.proj_out")),
        chroma_omi_transformer_prefix(String("single_blocks.0.linear2")),
        String("single.proj_out"),
        source,
    )
    _chroma_append_target(raw_prefixes, omi_prefixes, legacy_prefixes, roles, source_paths,
        chroma_transformer_lora_prefix(String("distilled_guidance_layer.layers.0.linear_1")),
        chroma_omi_transformer_prefix(String("distilled_guidance_layer.layers.0.in_layer")),
        String("distilled_guidance.in_layer"),
        source,
    )
    _chroma_append_target(raw_prefixes, omi_prefixes, legacy_prefixes, roles, source_paths,
        chroma_transformer_lora_prefix(String("distilled_guidance_layer.layers.0.linear_2")),
        chroma_omi_transformer_prefix(String("distilled_guidance_layer.layers.0.out_layer")),
        String("distilled_guidance.out_layer"),
        source,
    )
    _chroma_append_target(raw_prefixes, omi_prefixes, legacy_prefixes, roles, source_paths,
        chroma_text_encoder_lora_prefix(String("encoder.block.0.layer.0.SelfAttention.q")),
        chroma_omi_t5_prefix(String("encoder.block.0.layer.0.SelfAttention.q")),
        String("t5.self_attention.q"),
        source,
    )

    return ChromaLoraTargetSpecs(raw_prefixes^, omi_prefixes^, legacy_prefixes^, roles^, source_paths^)

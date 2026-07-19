# qwenLoraTargets.mojo — Qwen LoRA target metadata (leaf module).
#
# Serenity reference:
#   modules/modelSetup/QwenLoRASetup.py wraps:
#     text_encoder_lora = LoRAModuleWrapper(model.text_encoder, "text_encoder", config)
#       only when text-encoder training is enabled or preloaded keys start with
#       "text_encoder";
#     transformer_lora = LoRAModuleWrapper(model.transformer, "transformer", config,
#       config.layer_filter.split(",")) always.
#   modules/modelSetup/BaseQwenSetup.py LAYER_PRESETS:
#     "attn-mlp":  ["attn", "img_mlp", "txt_mlp"]
#     "attn-only": ["attn"]
#     "blocks":    ["transformer_block"]
#     "full":      []
#   modules/module/LoRAModule.py wraps every named Linear/Conv2d matching the
#   filters and saves raw keys:
#     <wrapper_prefix>.<module_name>.lora_down.weight
#     <wrapper_prefix>.<module_name>.lora_up.weight
#     <wrapper_prefix>.<module_name>.alpha
#
# QwenLoRASaver/QwenLoRALoader return no conversion key sets, so these raw
# prefixes are the on-disk contract.


comptime QWEN_LORA_TEXT_ENCODER_PREFIX = "text_encoder"
comptime QWEN_LORA_TRANSFORMER_PREFIX = "transformer"


struct QwenLoraTargetSpecs(Movable):
    var prefixes: List[String]
    var in_features: List[Int]
    var out_features: List[Int]

    def __init__(
        out self,
        var prefixes: List[String],
        var in_features: List[Int],
        var out_features: List[Int],
    ):
        self.prefixes = prefixes^
        self.in_features = in_features^
        self.out_features = out_features^

    def len(self) -> Int:
        return len(self.prefixes)


def qwen_lora_prefixed_module(wrapper_prefix: String, module_name: String) -> String:
    if wrapper_prefix == String():
        return module_name
    return wrapper_prefix + String(".") + module_name


def qwen_transformer_module(block_idx: Int, suffix: String) -> String:
    return String("transformer_blocks.") + String(block_idx) + String(".") + suffix


def qwen_transformer_lora_prefix(module_name: String) -> String:
    return qwen_lora_prefixed_module(String(QWEN_LORA_TRANSFORMER_PREFIX), module_name)


def qwen_text_encoder_lora_prefix(module_name: String) -> String:
    return qwen_lora_prefixed_module(String(QWEN_LORA_TEXT_ENCODER_PREFIX), module_name)


def qwen_lora_down_key(prefix: String) -> String:
    return prefix + String(".lora_down.weight")


def qwen_lora_up_key(prefix: String) -> String:
    return prefix + String(".lora_up.weight")


def qwen_lora_alpha_key(prefix: String) -> String:
    return prefix + String(".alpha")


# Representative QwenImageTransformerBlock Linear suffixes in named_modules()
# order for the Serenity "attn-mlp" preset. The full/block presets can include
# additional modulation/top-level Linears; this bounded list is intentionally not
# a full-model adapter inventory.
def qwen_transformer_attn_mlp_suffixes() -> List[String]:
    var out = List[String]()
    out.append(String("attn.to_q"))
    out.append(String("attn.to_k"))
    out.append(String("attn.to_v"))
    out.append(String("attn.add_q_proj"))
    out.append(String("attn.add_k_proj"))
    out.append(String("attn.add_v_proj"))
    out.append(String("attn.to_out.0"))
    out.append(String("attn.to_add_out"))
    out.append(String("img_mlp.net.0.proj"))
    out.append(String("img_mlp.net.2"))
    out.append(String("txt_mlp.net.0.proj"))
    out.append(String("txt_mlp.net.2"))
    return out^


def qwen_bounded_lora_target_specs(
    dim: Int,
    mlp_dim: Int,
    text_dim: Int,
) -> QwenLoraTargetSpecs:
    """Tiny representative target set for save/load key parity gates.

    This covers the optional text-encoder namespace and one transformer block's
    attention/MLP namespace. It deliberately stays bounded and does not claim
    numeric or full-model coverage.
    """
    var prefixes = List[String]()
    var in_features = List[Int]()
    var out_features = List[Int]()

    prefixes.append(qwen_text_encoder_lora_prefix(String("model.layers.0.self_attn.q_proj")))
    in_features.append(text_dim)
    out_features.append(text_dim)

    prefixes.append(qwen_transformer_lora_prefix(qwen_transformer_module(0, String("attn.to_q"))))
    in_features.append(dim)
    out_features.append(dim)

    prefixes.append(qwen_transformer_lora_prefix(qwen_transformer_module(0, String("attn.add_q_proj"))))
    in_features.append(dim)
    out_features.append(dim)

    prefixes.append(qwen_transformer_lora_prefix(qwen_transformer_module(0, String("img_mlp.net.0.proj"))))
    in_features.append(dim)
    out_features.append(mlp_dim)

    prefixes.append(qwen_transformer_lora_prefix(qwen_transformer_module(0, String("img_mlp.net.2"))))
    in_features.append(mlp_dim)
    out_features.append(dim)

    return QwenLoraTargetSpecs(prefixes^, in_features^, out_features^)


def qwen_transformer_attn_mlp_target_specs(
    num_blocks: Int,
    dim: Int,
    mlp_dim: Int,
) -> QwenLoraTargetSpecs:
    """Full Qwen transformer "attn-mlp" LoRA inventory.

    Mirrors Serenity's BaseQwenSetup LAYER_PRESETS["attn-mlp"] and
    QwenLoRASetup transformer wrapper: every transformer block gets the 8
    attention Linear modules plus img/txt MLP in/out projections. This is the
    real Qwen Image baseline file shape: 60 blocks * 12 targets * 3 keys.
    """
    var suffixes = qwen_transformer_attn_mlp_suffixes()
    var prefixes = List[String]()
    var in_features = List[Int]()
    var out_features = List[Int]()

    for bi in range(num_blocks):
        for si in range(len(suffixes)):
            var suffix = suffixes[si]
            prefixes.append(qwen_transformer_lora_prefix(qwen_transformer_module(bi, suffix)))

            if suffix == String("img_mlp.net.0.proj") or suffix == String("txt_mlp.net.0.proj"):
                in_features.append(dim)
                out_features.append(mlp_dim)
            elif suffix == String("img_mlp.net.2") or suffix == String("txt_mlp.net.2"):
                in_features.append(mlp_dim)
                out_features.append(dim)
            else:
                in_features.append(dim)
                out_features.append(dim)

    return QwenLoraTargetSpecs(prefixes^, in_features^, out_features^)

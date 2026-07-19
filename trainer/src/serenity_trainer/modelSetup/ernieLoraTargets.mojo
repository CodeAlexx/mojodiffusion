# ernieLoraTargets.mojo - Ernie LoRA target metadata.
#
# Serenity reference:
#   modules/modelSetup/ErnieLoRASetup.py wraps only model.transformer with
#   prefix "transformer" and config.layer_filter.split(",").
#   modules/modelSaver/ernie/ErnieLoRASaver.py and the loader path use raw
#   LoRAModule state_dict keys without conversion.
#
# This metadata is a deterministic key/shape contract for PEFT file gates. It is
# not a transformer forward/backward parity claim.


comptime ERNIE_LORA_TRANSFORMER_PREFIX = "transformer"


struct ErnieLoraTargetSpecs(Movable):
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


def ernie_lora_prefixed_module(wrapper_prefix: String, module_name: String) -> String:
    if wrapper_prefix == String():
        return module_name
    return wrapper_prefix + String(".") + module_name


def ernie_transformer_layer_module(layer_idx: Int, suffix: String) -> String:
    return String("layers.") + String(layer_idx) + String(".") + suffix


def ernie_transformer_lora_prefix(module_name: String) -> String:
    return ernie_lora_prefixed_module(String(ERNIE_LORA_TRANSFORMER_PREFIX), module_name)


def ernie_lora_down_key(prefix: String) -> String:
    return prefix + String(".lora_down.weight")


def ernie_lora_up_key(prefix: String) -> String:
    return prefix + String(".lora_up.weight")


def ernie_lora_alpha_key(prefix: String) -> String:
    return prefix + String(".alpha")


def ernie_transformer_attn_mlp_suffixes() -> List[String]:
    var out = List[String]()
    out.append(String("self_attention.to_q"))
    out.append(String("self_attention.to_k"))
    out.append(String("self_attention.to_v"))
    out.append(String("self_attention.to_out.0"))
    out.append(String("mlp.gate_proj"))
    out.append(String("mlp.up_proj"))
    out.append(String("mlp.linear_fc2"))
    return out^


def ernie_transformer_attn_mlp_target_specs(
    num_layers: Int,
    dim: Int,
    mlp_dim: Int,
) -> ErnieLoraTargetSpecs:
    """Full Ernie transformer LoRA inventory for the 100-step baseline file.

    The real Serenity Ernie LoRA baseline has:
      36 layers * 7 targets = 252 adapters, 756 raw keys.
    """
    var suffixes = ernie_transformer_attn_mlp_suffixes()
    var prefixes = List[String]()
    var in_features = List[Int]()
    var out_features = List[Int]()

    for li in range(num_layers):
        for si in range(len(suffixes)):
            var suffix = suffixes[si]
            prefixes.append(ernie_transformer_lora_prefix(ernie_transformer_layer_module(li, suffix)))

            if suffix == String("mlp.gate_proj") or suffix == String("mlp.up_proj"):
                in_features.append(dim)
                out_features.append(mlp_dim)
            elif suffix == String("mlp.linear_fc2"):
                in_features.append(mlp_dim)
                out_features.append(dim)
            else:
                in_features.append(dim)
                out_features.append(dim)

    return ErnieLoraTargetSpecs(prefixes^, in_features^, out_features^)

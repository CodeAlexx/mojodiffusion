# animaLoraTargets.mojo - Anima LoRA target metadata.
#
# Serenity reference:
#   /home/alex/Serenity-anima-ref/modules/modelSetup/AnimaLoRASetup.py
#     wraps only model.transformer with prefix "transformer" and
#     config.layer_filter.split(",").
#   /home/alex/Serenity-anima-ref/modules/modelSetup/BaseAnimaSetup.py
#     LAYER_PRESETS["attn-mlp"] = ["attn1", "attn2", "ff"].
#   /home/alex/Serenity-anima-ref/modules/model/AnimaModel.py maps the
#     CosmosTransformerBlock Linear module names used below.
#   /home/alex/Serenity-anima-ref/modules/modelSaver/anima/AnimaLoRASaver.py
#     returns no conversion key sets, so raw LoRAModule state_dict keys are the
#     on-disk contract.
#
# This metadata is key/shape/dtype inventory only; it is not an Anima numeric
# forward/backward parity claim.


comptime ANIMA_LORA_TRANSFORMER_PREFIX = "transformer"


struct AnimaLoraTargetSpecs(Movable):
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


def anima_lora_prefixed_module(wrapper_prefix: String, module_name: String) -> String:
    if wrapper_prefix == String():
        return module_name
    return wrapper_prefix + String(".") + module_name


def anima_transformer_block_module(block_idx: Int, suffix: String) -> String:
    return String("transformer_blocks.") + String(block_idx) + String(".") + suffix


def anima_transformer_lora_prefix(module_name: String) -> String:
    return anima_lora_prefixed_module(String(ANIMA_LORA_TRANSFORMER_PREFIX), module_name)


def anima_lora_down_key(prefix: String) -> String:
    return prefix + String(".lora_down.weight")


def anima_lora_up_key(prefix: String) -> String:
    return prefix + String(".lora_up.weight")


def anima_lora_alpha_key(prefix: String) -> String:
    return prefix + String(".alpha")


def anima_transformer_attn_mlp_suffixes() -> List[String]:
    var out = List[String]()
    out.append(String("attn1.to_q"))
    out.append(String("attn1.to_k"))
    out.append(String("attn1.to_v"))
    out.append(String("attn1.to_out.0"))
    out.append(String("attn2.to_q"))
    out.append(String("attn2.to_k"))
    out.append(String("attn2.to_v"))
    out.append(String("attn2.to_out.0"))
    out.append(String("ff.net.0.proj"))
    out.append(String("ff.net.2"))
    return out^


def anima_transformer_attn_mlp_target_specs(
    num_blocks: Int,
    dim: Int,
    cross_dim: Int,
    mlp_dim: Int,
) -> AnimaLoraTargetSpecs:
    """Full Anima transformer LoRA inventory for the 100-step baseline file.

    The real Serenity Anima LoRA baseline has:
      28 blocks * 10 targets = 280 adapters, 840 raw keys.
    """
    var suffixes = anima_transformer_attn_mlp_suffixes()
    var prefixes = List[String]()
    var in_features = List[Int]()
    var out_features = List[Int]()

    for bi in range(num_blocks):
        for si in range(len(suffixes)):
            var suffix = suffixes[si]
            prefixes.append(
                anima_transformer_lora_prefix(
                    anima_transformer_block_module(bi, suffix)
                )
            )

            if suffix == String("attn2.to_k") or suffix == String("attn2.to_v"):
                in_features.append(cross_dim)
                out_features.append(dim)
            elif suffix == String("ff.net.0.proj"):
                in_features.append(dim)
                out_features.append(mlp_dim)
            elif suffix == String("ff.net.2"):
                in_features.append(mlp_dim)
                out_features.append(dim)
            else:
                in_features.append(dim)
                out_features.append(dim)

    return AnimaLoraTargetSpecs(prefixes^, in_features^, out_features^)

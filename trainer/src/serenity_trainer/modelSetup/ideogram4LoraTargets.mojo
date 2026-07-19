# ideogram4LoraTargets.mojo - Ideogram4 LoRA target metadata.
#
# ai-toolkit wraps target module class Ideogram4Transformer2DModel. The saved
# LoRA keys keep the normal transformer prefix, then ai-toolkit converts
# "transformer." <-> "diffusion_model." at save/load boundaries.
#
# Native weight names match:
#   /home/alex/mojodiffusion/serenitymojo/models/dit/ideogram4_resident.mojo

from serenity_trainer.modelSampler.Ideogram4Sampler import IDEOGRAM4_NUM_LAYERS


comptime IDEOGRAM4_LORA_PREFIX_SERENITY = "transformer"
comptime IDEOGRAM4_LORA_PREFIX_DIFFUSION_MODEL = "diffusion_model"
comptime IDEOGRAM4_LAYER_TARGET_SLOTS = 6
comptime IDEOGRAM4_GLOBAL_TARGET_SLOTS = 7


def ideogram4_layer_suffixes() -> List[String]:
    return ideogram4_block_layer_suffixes()


def ideogram4_block_layer_suffixes() -> List[String]:
    var out = List[String]()
    out.append(String("attention.qkv"))
    out.append(String("attention.o"))
    out.append(String("feed_forward.w1"))
    out.append(String("feed_forward.w2"))
    out.append(String("feed_forward.w3"))
    out.append(String("adaln_modulation"))
    return out^


def ideogram4_global_suffixes() -> List[String]:
    var out = List[String]()
    out.append(String("input_proj"))
    out.append(String("llm_cond_proj"))
    out.append(String("t_embedding.mlp_in"))
    out.append(String("t_embedding.mlp_out"))
    out.append(String("adaln_proj"))
    out.append(String("final_layer.adaln_modulation"))
    out.append(String("final_layer.linear"))
    return out^


def ideogram4_layer_module(layer_idx: Int, suffix: String) -> String:
    return String("layers.") + String(layer_idx) + String(".") + suffix


def ideogram4_lora_save_prefix(module_name: String) -> String:
    return String(IDEOGRAM4_LORA_PREFIX_SERENITY) + String(".") + module_name


def ideogram4_lora_diffusion_model_prefix(module_name: String) -> String:
    return String(IDEOGRAM4_LORA_PREFIX_DIFFUSION_MODEL) + String(".") + module_name


def ideogram4_block_lora_save_prefixes(num_layers: Int = IDEOGRAM4_NUM_LAYERS) -> List[String]:
    var out = List[String]()
    var suffixes = ideogram4_block_layer_suffixes()
    for layer in range(num_layers):
        for i in range(len(suffixes)):
            out.append(ideogram4_lora_save_prefix(ideogram4_layer_module(layer, suffixes[i])))
    return out^


def ideogram4_full_lora_save_prefixes(num_layers: Int = IDEOGRAM4_NUM_LAYERS) -> List[String]:
    var out = List[String]()
    var globals = ideogram4_global_suffixes()
    for i in range(len(globals)):
        out.append(ideogram4_lora_save_prefix(globals[i]))
    var blocks = ideogram4_block_lora_save_prefixes(num_layers)
    for i in range(len(blocks)):
        out.append(blocks[i].copy())
    return out^


def ideogram4_lora_count(num_layers: Int = IDEOGRAM4_NUM_LAYERS, include_globals: Bool = True) -> Int:
    var count = num_layers * IDEOGRAM4_LAYER_TARGET_SLOTS
    if include_globals:
        count = count + IDEOGRAM4_GLOBAL_TARGET_SLOTS
    return count


def ideogram4_convert_lora_key_before_save(key: String) -> String:
    var old_prefix = String(IDEOGRAM4_LORA_PREFIX_SERENITY) + String(".")
    var new_prefix = String(IDEOGRAM4_LORA_PREFIX_DIFFUSION_MODEL) + String(".")
    if key.byte_length() < old_prefix.byte_length():
        return key.copy()
    if String(key[byte=0:old_prefix.byte_length()]) == old_prefix:
        return new_prefix + String(key[byte=old_prefix.byte_length():])
    return key.copy()


def ideogram4_convert_lora_key_before_load(key: String) -> String:
    var old_prefix = String(IDEOGRAM4_LORA_PREFIX_DIFFUSION_MODEL) + String(".")
    var new_prefix = String(IDEOGRAM4_LORA_PREFIX_SERENITY) + String(".")
    if key.byte_length() < old_prefix.byte_length():
        return key.copy()
    if String(key[byte=0:old_prefix.byte_length()]) == old_prefix:
        return new_prefix + String(key[byte=old_prefix.byte_length():])
    return key.copy()

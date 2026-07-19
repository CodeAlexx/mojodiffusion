# Chroma LoRA conversion contract gate.
#
# Build-only/source-contract coverage against Serenity
# util/convert/lora/convert_chroma_lora.py and Chroma LoRA loader/saver
# conversion references. This is not tensor conversion or numeric parity.

from serenity_trainer.modelLoader.chroma.ChromaLoRALoader import (
    CHROMA_LORA_ROUTE_SAFETENSORS,
    ChromaLoRALoader,
)
from serenity_trainer.util.convert.lora.convert_chroma_lora import (
    CHROMA_LORA_CONVERSION_KEY_SET_COUNT,
    chroma_lora_conversion_plan,
    chroma_lora_down_key,
    chroma_representative_lora_target_specs,
)


def _expect_int(name: String, got: Int, expected: Int) raises:
    if got != expected:
        raise Error(name + String(" got=") + String(got) + String(" expected=") + String(expected))


def _expect_bool(name: String, got: Bool, expected: Bool) raises:
    if got != expected:
        raise Error(name + String(" got=") + String(got) + String(" expected=") + String(expected))


def _expect_string(name: String, got: String, expected: String) raises:
    if got != expected:
        raise Error(name + String(" got=") + got + String(" expected=") + expected)


def main() raises:
    var plan = chroma_lora_conversion_plan()
    var representative = chroma_representative_lora_target_specs()
    _expect_bool("has key sets", plan.has_convert_key_sets, True)
    _expect_string("source namespaces", plan.source_namespaces, String("omi,diffusers,legacy_diffusers"))
    _expect_string("load target", plan.load_target_namespace, String("diffusers"))
    _expect_string("safetensors target", plan.safetensors_save_target_namespace, String("legacy_diffusers"))
    _expect_string("internal target", plan.internal_save_target_namespace, String("omi"))
    _expect_string("transformer diffusers", plan.transformer_diffusers_prefix, String("lora_transformer"))
    _expect_string("t5 diffusers", plan.t5_diffusers_prefix, String("lora_te"))
    _expect_int("bounded count", plan.bounded_conversion_key_set_count, CHROMA_LORA_CONVERSION_KEY_SET_COUNT)
    _expect_int("representative count", representative.len(), 12)
    _expect_string("rep0 role", representative.roles[0], String("bundle_emb.t5"))
    _expect_string("rep1 down", chroma_lora_down_key(representative.diffusers_prefixes[1]), String("lora_transformer.context_embedder.lora_down.weight"))

    var loader = ChromaLoRALoader()
    var load_plan = loader.load(String("/models/chroma-lora.safetensors"))
    _expect_int("load route", load_plan.route, CHROMA_LORA_ROUTE_SAFETENSORS)
    _expect_bool("load key sets", load_plan.has_convert_key_sets, True)
    _expect_string("load namespace", load_plan.converted_target_namespace, String("diffusers"))
    _expect_bool("load preserves dtype", load_plan.preserves_tensor_storage_dtype, True)

    print("CHROMA LORA CONVERSION CONTRACT OK")

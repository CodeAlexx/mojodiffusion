# Flux2/Klein saver surface contract gate.
#
# Source of truth:
#   /home/alex/Serenity/modules/modelSaver/Flux2FineTuneModelSaver.py
#   /home/alex/Serenity/modules/modelSaver/Flux2LoRAModelSaver.py
#   /home/alex/Serenity/modules/modelSaver/flux2/Flux2ModelSaver.py
#   /home/alex/Serenity/modules/modelSaver/flux2/Flux2LoRASaver.py
#
# This is build-only/source-contract coverage. It does not load Flux2 weights,
# run CUDA, instantiate Serenity, write safetensors, or make a numeric parity
# claim.

from serenity_trainer.modelSaver.Flux2FineTuneModelSaver import (
    Flux2FineTuneModelSaver,
)
from serenity_trainer.modelSaver.Flux2LoRAModelSaver import Flux2LoRAModelSaver
from serenity_trainer.modelSaver.flux2.Flux2LoRASaver import (
    FLUX2_FMT_DIFFUSERS,
    FLUX2_FMT_INTERNAL,
    FLUX2_FMT_LEGACY_SAFETENSORS,
    FLUX2_FMT_SAFETENSORS,
    Flux2LoRASaver,
    flux2_lora_saver_has_convert_key_sets,
    flux2_lora_state_dict_source_names,
)
from serenity_trainer.modelSaver.flux2.Flux2ModelSaver import (
    flux2_model_safetensors_state_sources,
)
from serenity_trainer.modelSetup.flux2LoraTargets import (
    flux2_lora_count,
    flux2_lora_save_prefixes,
)
from serenity_trainer.util.enum.ModelType import MODEL_TYPE_FLUX_2


def _expect_int(name: String, got: Int, expected: Int) raises:
    if got != expected:
        raise Error(name + String(": got ") + String(got) + String(", expected ") + String(expected))


def _expect_bool(name: String, got: Bool, expected: Bool) raises:
    if got != expected:
        raise Error(name + String(": got ") + String(got) + String(", expected ") + String(expected))


def _expect_string(name: String, got: String, expected: String) raises:
    if got != expected:
        raise Error(name + String(": got ") + got + String(", expected ") + expected)


def main() raises:
    _expect_int("ModelFormat.DIFFUSERS", FLUX2_FMT_DIFFUSERS, 0)
    _expect_int("ModelFormat.SAFETENSORS", FLUX2_FMT_SAFETENSORS, 2)
    _expect_int("ModelFormat.LEGACY_SAFETENSORS", FLUX2_FMT_LEGACY_SAFETENSORS, 3)
    _expect_int("ModelFormat.INTERNAL", FLUX2_FMT_INTERNAL, 5)

    var ft_saver = Flux2FineTuneModelSaver()
    var ft_contract = ft_saver.contract(MODEL_TYPE_FLUX_2)
    _expect_string("ft factory", ft_contract.factory_name, String("make_fine_tune_model_saver"))
    _expect_string("ft model class", ft_contract.model_class_name, String("Flux2Model"))
    _expect_string("ft leaf saver", ft_contract.model_saver_class_name, String("Flux2ModelSaver"))
    _expect_string("ft embedding saver", ft_contract.embedding_saver_class_name, String("None"))
    _expect_bool("ft has embedding saver", ft_contract.has_embedding_saver, False)
    _expect_bool("ft internal data", ft_contract.internal_save_data_after_leaf_save, True)
    _expect_bool("ft runtime save", ft_contract.runtime_save_implemented, False)

    var diffusers_plan = ft_saver.save_plan(
        MODEL_TYPE_FLUX_2,
        FLUX2_FMT_DIFFUSERS,
        String("/tmp/flux2-diffusers"),
        String("BF16"),
    )
    _expect_string("diffusers route", diffusers_plan.route_name, String("diffusers_pipeline"))
    _expect_bool("diffusers saves pipeline", diffusers_plan.saves_diffusers_pipeline, True)
    _expect_bool("diffusers cpu first", diffusers_plan.save_pipeline_to_cpu_first, True)
    _expect_bool("diffusers deep copy", diffusers_plan.deep_copy_pipeline_when_dtype_override, True)
    _expect_bool("diffusers tokenizer patch", diffusers_plan.tokenizer_deepcopy_patched_for_dtype_override, True)
    _expect_bool("diffusers converter", diffusers_plan.uses_diffusers_checkpoint_to_original_converter, False)

    var safetensors_plan = ft_saver.save_plan(
        MODEL_TYPE_FLUX_2,
        FLUX2_FMT_SAFETENSORS,
        String("/tmp/flux2-transformer.safetensors"),
        String("BF16"),
    )
    _expect_string("safetensors route", safetensors_plan.route_name, String("original_transformer_safetensors_checkpoint"))
    _expect_string("safetensors converter", safetensors_plan.converter_name, String("diffusers_checkpoint_to_original"))
    _expect_string("safetensors converter source", safetensors_plan.converter_source, String("modules/model/Flux2Model.py"))
    _expect_bool("safetensors transformer only", safetensors_plan.safetensors_includes_transformer_state_only, True)
    _expect_bool("safetensors vae omitted", safetensors_plan.safetensors_includes_vae_state_dict, False)
    _expect_bool("safetensors text omitted", safetensors_plan.safetensors_includes_text_encoder_state_dict, False)
    _expect_bool("safetensors header", safetensors_plan.creates_safetensors_header, True)
    _expect_bool("safetensors contiguous", safetensors_plan.makes_tensors_contiguous, True)
    _expect_bool("safetensors dtype override", safetensors_plan.dtype_override_applies_to_safetensors, True)

    var internal_plan = ft_saver.save_plan(
        MODEL_TYPE_FLUX_2,
        FLUX2_FMT_INTERNAL,
        String("/tmp/flux2-internal"),
        String("BF16"),
    )
    _expect_string("internal route", internal_plan.route_name, String("internal_diffusers"))
    _expect_string("internal dtype override ignored", internal_plan.dtype_override, String())
    _expect_bool("internal diffusers", internal_plan.saves_internal_diffusers, True)
    _expect_bool("internal no dtype", internal_plan.internal_uses_diffusers_without_dtype_override, True)

    var model_sources = flux2_model_safetensors_state_sources()
    _expect_int("model source count", len(model_sources), 5)
    _expect_string("model source transformer", model_sources[0], String("model.transformer.state_dict()"))

    var lora_model_saver = Flux2LoRAModelSaver()
    var lora_contract = lora_model_saver.contract(MODEL_TYPE_FLUX_2)
    _expect_string("lora factory", lora_contract.factory_name, String("make_lora_model_saver"))
    _expect_string("lora model class", lora_contract.model_class_name, String("Flux2Model"))
    _expect_string("lora leaf saver", lora_contract.lora_saver_class_name, String("Flux2LoRASaver"))
    _expect_string("lora embedding saver", lora_contract.embedding_saver_class_name, String("None"))
    _expect_bool("lora has embedding saver", lora_contract.has_embedding_saver, False)
    _expect_bool("lora embedding invoked", lora_contract.embedding_saver_invoked, False)
    _expect_bool("lora internal data", lora_contract.internal_save_data_after_leaf_save, True)

    var lora_plan = lora_model_saver.save_plan(
        MODEL_TYPE_FLUX_2,
        FLUX2_FMT_SAFETENSORS,
        String("/tmp/flux2-lora.safetensors"),
    )
    _expect_string("lora route", lora_plan.route_name, String("legacy_safetensors"))
    _expect_string("lora namespace", lora_plan.target_key_namespace, String("raw_diffusers_loramodule_keys"))
    _expect_bool("lora writes", lora_plan.writes_safetensors, True)
    _expect_bool("lora convert keys", lora_plan.has_convert_key_sets, False)
    _expect_bool("lora legacy default", lora_plan.safetensors_uses_legacy_route_without_omi, True)
    _expect_bool("lora dtype before route", lora_plan.dtype_override_applies_before_key_route, True)

    var lora_internal = lora_model_saver.save_plan(
        MODEL_TYPE_FLUX_2,
        FLUX2_FMT_INTERNAL,
        String("/tmp/flux2-internal"),
    )
    _expect_string("lora internal route", lora_internal.route_name, String("internal_lora"))
    _expect_string("lora internal dest", lora_internal.internal_destination, String("/tmp/flux2-internal/lora/lora.safetensors"))
    _expect_bool("lora internal dtype none", lora_internal.internal_passes_dtype_none, True)
    _expect_bool("lora internal data", lora_internal.wrapper_saves_internal_data_after_leaf_save, True)

    var leaf = Flux2LoRASaver()
    var state_contract = leaf.state_dict_contract()
    _expect_bool("leaf convert method", leaf._get_convert_key_sets(), False)
    _expect_bool("leaf convert free", flux2_lora_saver_has_convert_key_sets(), False)
    _expect_bool("state transformer lora", state_contract.includes_transformer_lora, True)
    _expect_bool("state preloaded", state_contract.includes_preloaded_lora_state_dict, True)
    _expect_bool("state text lora omitted", state_contract.includes_text_encoder_lora, False)
    _expect_string("state prefix", state_contract.wrapper_prefix, String("transformer"))
    _expect_string("state down", state_contract.lora_down_suffix, String(".lora_down.weight"))
    _expect_string("state up", state_contract.lora_up_suffix, String(".lora_up.weight"))
    _expect_string("state alpha", state_contract.alpha_suffix, String(".alpha"))

    var source_names = flux2_lora_state_dict_source_names()
    _expect_int("lora source count", len(source_names), 2)
    _expect_string("lora source transformer", source_names[0], String("model.transformer_lora.state_dict() when model.transformer_lora is not None"))
    _expect_string("lora source preloaded", source_names[1], String("model.lora_state_dict when model.lora_state_dict is not None"))

    var prefixes = flux2_lora_save_prefixes(1, 1)
    _expect_int("prefix module count", flux2_lora_count(1, 1), 14)
    _expect_int("prefix count", len(prefixes), 14)
    _expect_string("first double prefix", prefixes[0], String("transformer.transformer_blocks.0.attn.to_q"))
    _expect_string("last double prefix", prefixes[11], String("transformer.transformer_blocks.0.ff_context.linear_out"))
    _expect_string("first single prefix", prefixes[12], String("transformer.single_transformer_blocks.0.attn.to_qkv_mlp_proj"))
    _expect_string("last single prefix", prefixes[13], String("transformer.single_transformer_blocks.0.attn.to_out"))

    print("FLUX2 SURFACE SAVER CONTRACT OK")

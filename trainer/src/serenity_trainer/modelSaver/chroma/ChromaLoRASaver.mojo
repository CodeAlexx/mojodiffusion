# 1:1 surface port of Serenity
#   modules/modelSaver/chroma/ChromaLoRASaver.py
#
# Serenity Chroma LoRA saver:
#   _get_convert_key_sets -> convert_chroma_lora_key_sets()
#   _get_state_dict       -> text_encoder_lora + transformer_lora
#                            + preloaded lora_state_dict
#                            + optional bundled T5 embeddings
#   save                  -> LoRASaverMixin._save
#
# This is metadata/save-route contract only. It does not write files, cast
# tensors, or claim numeric parity.

from serenity_trainer.util.convert.lora.convert_chroma_lora import (
    ChromaLoraConversionKeySet,
    ChromaLoraConversionPlan,
    chroma_lora_conversion_plan,
    convert_chroma_lora_key_sets,
)


comptime CHROMA_FMT_DIFFUSERS = 0
comptime CHROMA_FMT_CKPT = 1
comptime CHROMA_FMT_SAFETENSORS = 2
comptime CHROMA_FMT_LEGACY_SAFETENSORS = 3
comptime CHROMA_FMT_COMFY_LORA = 4
comptime CHROMA_FMT_INTERNAL = 5


struct ChromaLoraStateDictContract(Copyable, Movable, ImplicitlyCopyable):
    var includes_text_encoder_lora: Bool
    var includes_transformer_lora: Bool
    var includes_preloaded_lora_state_dict: Bool
    var can_bundle_additional_embeddings: Bool
    var bundle_requires_train_config_flag: String
    var bundles_only_when_additional_embeddings_present: Bool
    var bundles_vector_only_when_present: Bool
    var bundles_output_vector_only_when_present: Bool
    var placeholder_source: String
    var text_encoder_embedding_vector_source: String
    var text_encoder_embedding_output_vector_source: String
    var text_encoder_embedding_vector_key_template: String
    var text_encoder_embedding_output_vector_key_template: String
    var bundle_embedding_key_count: Int
    var preserves_tensor_storage_dtype: Bool

    def __init__(out self):
        self.includes_text_encoder_lora = True
        self.includes_transformer_lora = True
        self.includes_preloaded_lora_state_dict = True
        self.can_bundle_additional_embeddings = True
        self.bundle_requires_train_config_flag = String("bundle_additional_embeddings")
        self.bundles_only_when_additional_embeddings_present = True
        self.bundles_vector_only_when_present = True
        self.bundles_output_vector_only_when_present = True
        self.placeholder_source = String("embedding.text_encoder_embedding.placeholder")
        self.text_encoder_embedding_vector_source = String(
            "embedding.text_encoder_embedding.vector"
        )
        self.text_encoder_embedding_output_vector_source = String(
            "embedding.text_encoder_embedding.output_vector"
        )
        self.text_encoder_embedding_vector_key_template = String(
            "bundle_emb.{placeholder}.t5"
        )
        self.text_encoder_embedding_output_vector_key_template = String(
            "bundle_emb.{placeholder}.t5_out"
        )
        self.bundle_embedding_key_count = 2
        self.preserves_tensor_storage_dtype = True


struct ChromaLoraSavePlan(Movable):
    var output_model_format: Int
    var output_model_destination: String
    var route_name: String
    var target_key_namespace: String
    var writes_safetensors: Bool
    var internal_destination: String
    var saver_mixin_name: String
    var save_delegates_to_mixin: Bool
    var dtype_override_applies_before_conversion: Bool
    var safetensors_default_uses_legacy_diffusers: Bool
    var internal_uses_omi_safetensors: Bool
    var ckpt_route_supported: Bool
    var comfy_lora_route_supported: Bool
    var has_convert_key_sets: Bool
    var conversion_plan: ChromaLoraConversionPlan
    var state_dict_contract: ChromaLoraStateDictContract
    var bundled_embedding_keys: String
    var preserves_storage_dtype_without_override: Bool

    def __init__(
        out self,
        output_model_format: Int,
        var output_model_destination: String,
        var route_name: String,
        var target_key_namespace: String,
        var internal_destination: String,
        writes_safetensors: Bool,
    ):
        self.output_model_format = output_model_format
        self.output_model_destination = output_model_destination^
        self.route_name = route_name^
        self.target_key_namespace = target_key_namespace^
        self.writes_safetensors = writes_safetensors
        self.internal_destination = internal_destination^
        self.saver_mixin_name = String("LoRASaverMixin")
        self.save_delegates_to_mixin = True
        self.dtype_override_applies_before_conversion = True
        self.safetensors_default_uses_legacy_diffusers = True
        self.internal_uses_omi_safetensors = True
        self.ckpt_route_supported = False
        self.comfy_lora_route_supported = False
        self.has_convert_key_sets = True
        self.conversion_plan = chroma_lora_conversion_plan()
        self.state_dict_contract = ChromaLoraStateDictContract()
        self.bundled_embedding_keys = String(
            "bundle_emb.{placeholder}.t5,bundle_emb.{placeholder}.t5_out"
        )
        self.preserves_storage_dtype_without_override = True


def chroma_lora_saver_has_convert_key_sets() -> Bool:
    return True


def chroma_lora_state_dict_contract() -> ChromaLoraStateDictContract:
    return ChromaLoraStateDictContract()


def chroma_lora_state_dict_source_names() -> List[String]:
    var names = List[String]()
    names.append(String("model.text_encoder_lora.state_dict()"))
    names.append(String("model.transformer_lora.state_dict()"))
    names.append(String("model.lora_state_dict"))
    names.append(String("model.additional_embeddings when model.train_config.bundle_additional_embeddings"))
    return names^


def chroma_lora_bundle_embedding_keys(placeholder: String) -> List[String]:
    var keys = List[String]()
    keys.append(String("bundle_emb.") + placeholder + String(".t5"))
    keys.append(String("bundle_emb.") + placeholder + String(".t5_out"))
    return keys^


def chroma_lora_save_plan(
    output_model_format: Int,
    output_model_destination: String,
) raises -> ChromaLoraSavePlan:
    if output_model_format == CHROMA_FMT_SAFETENSORS:
        return ChromaLoraSavePlan(
            output_model_format,
            output_model_destination.copy(),
            String("legacy_safetensors"),
            String("legacy_diffusers"),
            String(),
            True,
        )
    if output_model_format == CHROMA_FMT_LEGACY_SAFETENSORS:
        return ChromaLoraSavePlan(
            output_model_format,
            output_model_destination.copy(),
            String("legacy_safetensors"),
            String("legacy_diffusers"),
            String(),
            True,
        )
    if output_model_format == CHROMA_FMT_INTERNAL:
        return ChromaLoraSavePlan(
            output_model_format,
            output_model_destination.copy(),
            String("internal_lora"),
            String("omi"),
            output_model_destination + String("/lora/lora.safetensors"),
            True,
        )
    if output_model_format == CHROMA_FMT_DIFFUSERS:
        raise Error("ChromaLoRASaver: DIFFUSERS LoRA output is not implemented in Serenity")
    raise Error("ChromaLoRASaver: unsupported ModelFormat")


struct ChromaLoRASaver(Movable):
    def __init__(out self):
        pass

    def _get_convert_key_sets(self) -> List[ChromaLoraConversionKeySet]:
        return convert_chroma_lora_key_sets()

    def conversion_plan(self) -> ChromaLoraConversionPlan:
        return chroma_lora_conversion_plan()

    def state_dict_contract(self) -> ChromaLoraStateDictContract:
        return chroma_lora_state_dict_contract()

    def save_plan(
        self,
        output_model_format: Int,
        output_model_destination: String,
    ) raises -> ChromaLoraSavePlan:
        return chroma_lora_save_plan(output_model_format, output_model_destination)

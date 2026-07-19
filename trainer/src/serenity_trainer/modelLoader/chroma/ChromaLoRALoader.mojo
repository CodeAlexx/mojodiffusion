# 1:1 surface port of Serenity
#   modules/modelLoader/chroma/ChromaLoRALoader.py
#
# Serenity Chroma LoRA loader:
#   _get_convert_key_sets -> convert_chroma_lora_key_sets()
#   load                  -> LoRALoaderMixin._load(model, model_names)
#
# This is metadata/load-route contract only. It does not open safetensors,
# allocate model tensors, or apply LoRA modules.

from serenity_trainer.util.convert.lora.convert_chroma_lora import (
    ChromaLoraConversionKeySet,
    ChromaLoraConversionPlan,
    chroma_lora_conversion_plan,
    convert_chroma_lora_key_sets,
)


comptime CHROMA_LORA_ROUTE_AUTO = 0
comptime CHROMA_LORA_ROUTE_NONE = 1
comptime CHROMA_LORA_ROUTE_INTERNAL = 2
comptime CHROMA_LORA_ROUTE_CKPT = 3
comptime CHROMA_LORA_ROUTE_SAFETENSORS = 4


struct ChromaLoRALoadPlan(Movable):
    var route: Int
    var lora_model: String
    var model_type_name: String
    var model_names_type_name: String
    var loader_mixin_name: String
    var load_delegates_to_mixin: Bool
    var returns_without_load_when_lora_empty: Bool
    var tries_internal_first: Bool
    var internal_probe_file: String
    var internal_safetensors_path: String
    var tries_ckpt_second_when_extension_matches: Bool
    var ckpt_extension: String
    var tries_safetensors_last: Bool
    var has_convert_key_sets: Bool
    var converted_target_namespace: String
    var conversion_plan: ChromaLoraConversionPlan
    var stores_loaded_state_dict_on_model: Bool
    var preserves_tensor_storage_dtype: Bool

    def __init__(out self, var lora_model: String):
        self.lora_model = lora_model^
        self.route = CHROMA_LORA_ROUTE_SAFETENSORS
        if self.lora_model.byte_length() == 0:
            self.route = CHROMA_LORA_ROUTE_NONE
        self.model_type_name = String("ChromaModel")
        self.model_names_type_name = String("ModelNames")
        self.loader_mixin_name = String("LoRALoaderMixin")
        self.load_delegates_to_mixin = True
        self.returns_without_load_when_lora_empty = True
        self.tries_internal_first = True
        self.internal_probe_file = String("meta.json")
        self.internal_safetensors_path = String("lora/lora.safetensors")
        self.tries_ckpt_second_when_extension_matches = True
        self.ckpt_extension = String(".ckpt")
        self.tries_safetensors_last = True
        self.has_convert_key_sets = True
        self.converted_target_namespace = String("diffusers")
        self.conversion_plan = chroma_lora_conversion_plan()
        self.stores_loaded_state_dict_on_model = True
        self.preserves_tensor_storage_dtype = True


def chroma_lora_loader_has_convert_key_sets() -> Bool:
    return True


def chroma_lora_load_plan(lora_model: String) -> ChromaLoRALoadPlan:
    return ChromaLoRALoadPlan(lora_model.copy())


struct ChromaLoRALoader(Movable):
    def __init__(out self):
        pass

    def _get_convert_key_sets(self) -> List[ChromaLoraConversionKeySet]:
        return convert_chroma_lora_key_sets()

    def conversion_plan(self) -> ChromaLoraConversionPlan:
        return chroma_lora_conversion_plan()

    def load(self, lora_model: String) -> ChromaLoRALoadPlan:
        return chroma_lora_load_plan(lora_model)

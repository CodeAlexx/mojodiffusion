# 1:1 surface port of Serenity
#   modules/modelSaver/Flux2LoRAModelSaver.py
#
# Serenity:
#   Flux2LoRAModelSaver = make_lora_model_saver(
#       ModelType.FLUX_2,
#       model_class=Flux2Model,
#       lora_saver_class=Flux2LoRASaver,
#       embedding_saver_class=None,
#   )
#
# Build-only wrapper contract mirror. The leaf Flux2LoRASaver exposes the raw-key
# LoRA route plan; no Mojo runtime save or numeric parity claim is made here.

from serenity_trainer.modelSaver.flux2.Flux2LoRASaver import (
    FLUX2_FMT_CKPT,
    FLUX2_FMT_COMFY_LORA,
    FLUX2_FMT_DIFFUSERS,
    FLUX2_FMT_INTERNAL,
    FLUX2_FMT_LEGACY_SAFETENSORS,
    FLUX2_FMT_SAFETENSORS,
    Flux2LoraSavePlan,
    Flux2LoRASaver,
)
from serenity_trainer.util.enum.ModelType import MODEL_TYPE_FLUX_2, model_type_str


comptime FMT_DIFFUSERS = FLUX2_FMT_DIFFUSERS
comptime FMT_CKPT = FLUX2_FMT_CKPT
comptime FMT_SAFETENSORS = FLUX2_FMT_SAFETENSORS
comptime FMT_LEGACY_SAFETENSORS = FLUX2_FMT_LEGACY_SAFETENSORS
comptime FMT_COMFY_LORA = FLUX2_FMT_COMFY_LORA
comptime FMT_INTERNAL = FLUX2_FMT_INTERNAL

comptime FLUX2_LORA_MODEL_SAVER_MODEL_TYPE = MODEL_TYPE_FLUX_2
comptime FLUX2_LORA_MODEL_SAVER_FACTORY = "make_lora_model_saver"
comptime FLUX2_LORA_MODEL_SAVER_MODEL_CLASS = "Flux2Model"
comptime FLUX2_LORA_MODEL_SAVER_LORA_SAVER_CLASS = "Flux2LoRASaver"
comptime FLUX2_LORA_MODEL_SAVER_EMBEDDING_SAVER_CLASS = "None"


struct Flux2LoRAModelSaverContract(Movable):
    var model_type: Int
    var factory_name: String
    var model_class_name: String
    var lora_saver_class_name: String
    var embedding_saver_class_name: String
    var has_embedding_saver: Bool
    var leaf_lora_saver_invoked: Bool
    var embedding_saver_invoked: Bool
    var internal_save_data_after_leaf_save: Bool
    var runtime_save_implemented: Bool

    def __init__(out self):
        self.model_type = FLUX2_LORA_MODEL_SAVER_MODEL_TYPE
        self.factory_name = String(FLUX2_LORA_MODEL_SAVER_FACTORY)
        self.model_class_name = String(FLUX2_LORA_MODEL_SAVER_MODEL_CLASS)
        self.lora_saver_class_name = String(FLUX2_LORA_MODEL_SAVER_LORA_SAVER_CLASS)
        self.embedding_saver_class_name = String(FLUX2_LORA_MODEL_SAVER_EMBEDDING_SAVER_CLASS)
        self.has_embedding_saver = False
        self.leaf_lora_saver_invoked = True
        self.embedding_saver_invoked = False
        self.internal_save_data_after_leaf_save = True
        self.runtime_save_implemented = False


def flux2_lora_model_saver_contract() -> Flux2LoRAModelSaverContract:
    return Flux2LoRAModelSaverContract()


struct Flux2LoRAModelSaver(Movable):
    def __init__(out self):
        pass

    def validate_model_type(self, model_type: Int) raises:
        if model_type != MODEL_TYPE_FLUX_2:
            raise Error(String("Flux2LoRAModelSaver: unsupported ModelType ") + model_type_str(model_type))

    def contract(self, model_type: Int) raises -> Flux2LoRAModelSaverContract:
        self.validate_model_type(model_type)
        return flux2_lora_model_saver_contract()

    def save_plan(
        self,
        model_type: Int,
        output_model_format: Int,
        output_model_destination: String,
    ) raises -> Flux2LoraSavePlan:
        self.validate_model_type(model_type)
        var saver = Flux2LoRASaver()
        return saver.save_plan(output_model_format, output_model_destination)

    def runtime_save_supported(self) -> Bool:
        return False

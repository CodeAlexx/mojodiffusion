# 1:1 surface port of Serenity-anima-ref modules/modelSaver/AnimaLoRAModelSaver.py
#
# Serenity:
#   AnimaLoRAModelSaver = make_lora_model_saver(
#       ModelType.ANIMA,
#       model_class=AnimaModel,
#       lora_saver_class=AnimaLoRASaver,
#       embedding_saver_class=None,
#   )

from std.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype

from serenity_trainer.modelLoader.AnimaModelLoader import MODEL_TYPE_ANIMA, anima_model_type_str
from serenity_trainer.modelSaver.anima.AnimaLoRASaver import (
    AnimaLoraStateDict,
    AnimaLoRASaver,
    save_anima_lora_state_dict_as_dtype,
)


struct AnimaLoRAModelSaver(Movable):
    def __init__(out self):
        pass

    def save(
        self,
        var state: AnimaLoraStateDict,
        model_type: Int,
        output_model_format: Int,
        output_model_destination: String,
        ctx: DeviceContext,
    ) raises:
        if model_type != MODEL_TYPE_ANIMA:
            raise Error(String("AnimaLoRAModelSaver.save: unsupported ModelType ") + anima_model_type_str(model_type))
        var saver = AnimaLoRASaver()
        saver.save(state^, output_model_format, output_model_destination, ctx)

    def save_as_dtype(
        self,
        state: AnimaLoraStateDict,
        model_type: Int,
        output_model_format: Int,
        output_model_destination: String,
        dtype: STDtype,
        ctx: DeviceContext,
    ) raises:
        if model_type != MODEL_TYPE_ANIMA:
            raise Error(String("AnimaLoRAModelSaver.save_as_dtype: unsupported ModelType ") + anima_model_type_str(model_type))
        save_anima_lora_state_dict_as_dtype(state, output_model_format, output_model_destination, dtype, ctx)

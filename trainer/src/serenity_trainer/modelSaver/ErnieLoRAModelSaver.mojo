# 1:1 surface port of Serenity modules/modelSaver/ErnieLoRAModelSaver.py
#
# Serenity:
#   ErnieLoRAModelSaver = make_lora_model_saver(
#       ModelType.ERNIE,
#       model_class=ErnieModel,
#       lora_saver_class=ErnieLoRASaver,
#       embedding_saver_class=None,
#   )

from std.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype

from serenity_trainer.modelSaver.ernie.ErnieLoRASaver import (
    ErnieLoraStateDict,
    ErnieLoRASaver,
    save_ernie_lora_state_dict_as_dtype,
)
from serenity_trainer.util.enum.ModelType import MODEL_TYPE_ERNIE, model_type_str


struct ErnieLoRAModelSaver(Movable):
    def __init__(out self):
        pass

    def save(
        self,
        var state: ErnieLoraStateDict,
        model_type: Int,
        output_model_format: Int,
        output_model_destination: String,
        ctx: DeviceContext,
    ) raises:
        if model_type != MODEL_TYPE_ERNIE:
            raise Error(String("ErnieLoRAModelSaver.save: unsupported ModelType ") + model_type_str(model_type))
        var saver = ErnieLoRASaver()
        saver.save(state^, output_model_format, output_model_destination, ctx)

    def save_as_dtype(
        self,
        state: ErnieLoraStateDict,
        model_type: Int,
        output_model_format: Int,
        output_model_destination: String,
        dtype: STDtype,
        ctx: DeviceContext,
    ) raises:
        if model_type != MODEL_TYPE_ERNIE:
            raise Error(String("ErnieLoRAModelSaver.save_as_dtype: unsupported ModelType ") + model_type_str(model_type))
        save_ernie_lora_state_dict_as_dtype(state, output_model_format, output_model_destination, dtype, ctx)

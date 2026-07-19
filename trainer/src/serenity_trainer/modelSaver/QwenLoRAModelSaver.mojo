# 1:1 surface port of Serenity modules/modelSaver/QwenLoRAModelSaver.py
#
# Serenity:
#   QwenLoRAModelSaver = make_lora_model_saver(
#       ModelType.QWEN,
#       model_class=QwenModel,
#       lora_saver_class=QwenLoRASaver,
#       embedding_saver_class=None,
#   )

from std.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype

from serenity_trainer.modelSaver.qwen.QwenLoRASaver import (
    QwenLoraStateDict,
    QwenLoRASaver,
    save_qwen_lora_state_dict,
    save_qwen_lora_state_dict_as_dtype,
)
from serenity_trainer.util.enum.ModelType import MODEL_TYPE_QWEN, model_type_str


struct QwenLoRAModelSaver(Movable):
    def __init__(out self):
        pass

    def save(
        self,
        var state: QwenLoraStateDict,
        model_type: Int,
        output_model_format: Int,
        output_model_destination: String,
        ctx: DeviceContext,
    ) raises:
        if model_type != MODEL_TYPE_QWEN:
            raise Error(String("QwenLoRAModelSaver.save: unsupported ModelType ") + model_type_str(model_type))
        var saver = QwenLoRASaver()
        saver.save(state^, output_model_format, output_model_destination, ctx)

    def save_as_dtype(
        self,
        state: QwenLoraStateDict,
        model_type: Int,
        output_model_format: Int,
        output_model_destination: String,
        dtype: STDtype,
        ctx: DeviceContext,
    ) raises:
        if model_type != MODEL_TYPE_QWEN:
            raise Error(String("QwenLoRAModelSaver.save_as_dtype: unsupported ModelType ") + model_type_str(model_type))
        save_qwen_lora_state_dict_as_dtype(state, output_model_format, output_model_destination, dtype, ctx)

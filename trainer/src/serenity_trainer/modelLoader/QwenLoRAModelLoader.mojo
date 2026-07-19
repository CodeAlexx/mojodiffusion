# 1:1 surface port of Serenity modules/modelLoader/QwenLoRAModelLoader.py
#
# Serenity:
#   QwenLoRAModelLoader = make_lora_model_loader(
#       model_spec_map={ModelType.QWEN: "resources/sd_model_spec/qwen-lora.json"},
#       model_class=QwenModel,
#       model_loader_class=QwenModelLoader,
#       embedding_loader_class=None,
#       lora_loader_class=QwenLoRALoader,
#   )

from serenity_trainer.modelLoader.qwen.QwenModelLoader import (
    QwenLoadPlan,
    QwenModelHandle,
    QwenModelLoader,
    QwenModelNames,
    QwenQuantizationConfig,
    QwenWeightDtypes,
)
from serenity_trainer.util.enum.ModelType import MODEL_TYPE_QWEN, model_type_str


def qwen_lora_default_model_spec_name(model_type: Int) raises -> String:
    if model_type == MODEL_TYPE_QWEN:
        return String("resources/sd_model_spec/qwen-lora.json")
    raise Error(String("QwenLoRAModelLoader: unsupported ModelType ") + model_type_str(model_type))


struct QwenLoRAModelLoader(Movable):
    def __init__(out self):
        pass

    def _default_model_spec_name(self, model_type: Int) raises -> String:
        return qwen_lora_default_model_spec_name(model_type)

    def load(
        self,
        model_type: Int,
        model_names: QwenModelNames,
        weight_dtypes: QwenWeightDtypes,
        quantization: QwenQuantizationConfig,
    ) raises -> QwenLoadPlan:
        if model_type != MODEL_TYPE_QWEN:
            raise Error(String("QwenLoRAModelLoader.load: unsupported ModelType ") + model_type_str(model_type))

        var model = QwenModelHandle(model_type)
        model.model_spec = qwen_lora_default_model_spec_name(model_type)

        var base_loader = QwenModelLoader()
        var plan = base_loader.load(model, model_type, model_names, weight_dtypes, quantization)
        plan.model_spec = qwen_lora_default_model_spec_name(model_type)
        return plan^

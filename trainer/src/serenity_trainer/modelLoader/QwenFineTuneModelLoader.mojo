# 1:1 surface port of Serenity modules/modelLoader/QwenFineTuneModelLoader.py
#
# Serenity:
#   QwenFineTuneModelLoader = make_fine_tune_model_loader(
#       model_spec_map={ModelType.QWEN: "resources/sd_model_spec/qwen.json"},
#       model_class=QwenModel,
#       model_loader_class=QwenModelLoader,
#       embedding_loader_class=None,
#   )

from serenity_trainer.modelLoader.qwen.QwenModelLoader import (
    QwenLoadPlan,
    QwenModelHandle,
    QwenModelLoader,
    QwenModelNames,
    QwenQuantizationConfig,
    QwenWeightDtypes,
    qwen_default_model_spec_name,
)
from serenity_trainer.util.enum.ModelType import MODEL_TYPE_QWEN, model_type_str


struct QwenFineTuneModelLoader(Movable):
    def __init__(out self):
        pass

    def _default_model_spec_name(self, model_type: Int) raises -> String:
        return qwen_default_model_spec_name(model_type)

    def load(
        self,
        model_type: Int,
        model_names: QwenModelNames,
        weight_dtypes: QwenWeightDtypes,
        quantization: QwenQuantizationConfig,
    ) raises -> QwenLoadPlan:
        if model_type != MODEL_TYPE_QWEN:
            raise Error(String("QwenFineTuneModelLoader.load: unsupported ModelType ") + model_type_str(model_type))

        var model = QwenModelHandle(model_type)
        var base_loader = QwenModelLoader()
        return base_loader.load(model, model_type, model_names, weight_dtypes, quantization)

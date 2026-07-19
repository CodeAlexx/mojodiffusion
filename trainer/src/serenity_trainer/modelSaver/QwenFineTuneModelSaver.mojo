# 1:1 surface port of Serenity modules/modelSaver/QwenFineTuneModelSaver.py
#
# Serenity:
#   QwenFineTuneModelSaver = make_fine_tune_model_saver(
#       ModelType.QWEN,
#       model_class=QwenModel,
#       model_saver_class=QwenModelSaver,
#       embedding_saver_class=None,
#   )

from serenity_trainer.modelSaver.qwen.QwenModelSaver import QwenModelSavePlan, QwenModelSaver
from serenity_trainer.util.enum.ModelType import MODEL_TYPE_QWEN, model_type_str


struct QwenFineTuneModelSaver(Movable):
    def __init__(out self):
        pass

    def save_plan(
        self,
        model_type: Int,
        output_model_format: Int,
        output_model_destination: String,
        dtype_override: String,
    ) raises -> QwenModelSavePlan:
        if model_type != MODEL_TYPE_QWEN:
            raise Error(String("QwenFineTuneModelSaver.save_plan: unsupported ModelType ") + model_type_str(model_type))
        var saver = QwenModelSaver()
        return saver.save_plan(output_model_format, output_model_destination, dtype_override)

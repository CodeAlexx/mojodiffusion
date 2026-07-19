# 1:1 surface port of Serenity-anima-ref modules/modelSaver/AnimaFineTuneModelSaver.py
#
# Serenity:
#   AnimaFineTuneModelSaver = make_fine_tune_model_saver(
#       ModelType.ANIMA,
#       model_class=AnimaModel,
#       model_saver_class=AnimaModelSaver,
#       embedding_saver_class=None,
#   )

from serenity_trainer.modelLoader.AnimaModelLoader import MODEL_TYPE_ANIMA, anima_model_type_str
from serenity_trainer.modelSaver.anima.AnimaModelSaver import AnimaModelSavePlan, AnimaModelSaver


struct AnimaFineTuneModelSaver(Movable):
    def __init__(out self):
        pass

    def save_plan(
        self,
        model_type: Int,
        output_model_format: Int,
        output_model_destination: String,
        dtype_override: String,
    ) raises -> AnimaModelSavePlan:
        if model_type != MODEL_TYPE_ANIMA:
            raise Error(String("AnimaFineTuneModelSaver.save_plan: unsupported ModelType ") + anima_model_type_str(model_type))
        var saver = AnimaModelSaver()
        return saver.save_plan(output_model_format, output_model_destination, dtype_override)

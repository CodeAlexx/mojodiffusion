# 1:1 surface port of Serenity modules/modelSaver/ErnieFineTuneModelSaver.py
#
# Serenity:
#   ErnieFineTuneModelSaver = make_fine_tune_model_saver(
#       ModelType.ERNIE,
#       model_class=ErnieModel,
#       model_saver_class=ErnieModelSaver,
#       embedding_saver_class=None,
#   )

from serenity_trainer.modelSaver.ernie.ErnieModelSaver import ErnieModelSavePlan, ErnieModelSaver
from serenity_trainer.util.enum.ModelType import MODEL_TYPE_ERNIE, model_type_str


struct ErnieFineTuneModelSaver(Movable):
    def __init__(out self):
        pass

    def save_plan(
        self,
        model_type: Int,
        output_model_format: Int,
        output_model_destination: String,
        dtype_override: String,
    ) raises -> ErnieModelSavePlan:
        if model_type != MODEL_TYPE_ERNIE:
            raise Error(String("ErnieFineTuneModelSaver.save_plan: unsupported ModelType ") + model_type_str(model_type))
        var saver = ErnieModelSaver()
        return saver.save_plan(output_model_format, output_model_destination, dtype_override)

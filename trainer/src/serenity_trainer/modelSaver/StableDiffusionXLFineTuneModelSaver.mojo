# 1:1 surface port of Serenity
#   modules/modelSaver/StableDiffusionXLFineTuneModelSaver.py

from serenity_trainer.modelSaver.stableDiffusionXL.StableDiffusionXLModelSaver import (
    StableDiffusionXLModelSavePlan,
    StableDiffusionXLModelSaver,
)
from serenity_trainer.util.enum.ModelType import (
    model_type_is_stable_diffusion_xl,
    model_type_str,
)


struct StableDiffusionXLFineTuneModelSaver(Movable):
    def __init__(out self):
        pass

    def save_plan(
        self,
        model_type: Int,
        output_model_format: Int,
        output_model_destination: String,
        dtype_override: String,
    ) raises -> StableDiffusionXLModelSavePlan:
        if not model_type_is_stable_diffusion_xl(model_type):
            raise Error(String("StableDiffusionXLFineTuneModelSaver.save_plan: unsupported ModelType ") + model_type_str(model_type))
        var saver = StableDiffusionXLModelSaver()
        return saver.save_plan(output_model_format, output_model_destination, dtype_override)

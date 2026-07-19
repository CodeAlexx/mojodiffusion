# 1:1 surface port of Serenity
#   modules/modelSaver/StableDiffusion3FineTuneModelSaver.py

from serenity_trainer.modelSaver.stableDiffusion3.StableDiffusion3ModelSaver import (
    StableDiffusion3ModelSavePlan,
    StableDiffusion3ModelSaver,
)
from serenity_trainer.util.enum.ModelType import (
    model_type_is_stable_diffusion_3,
    model_type_str,
)


struct StableDiffusion3FineTuneModelSaver(Movable):
    def __init__(out self):
        pass

    def save_plan(
        self,
        model_type: Int,
        output_model_format: Int,
        output_model_destination: String,
        dtype_override: String,
    ) raises -> StableDiffusion3ModelSavePlan:
        if not model_type_is_stable_diffusion_3(model_type):
            raise Error(String("StableDiffusion3FineTuneModelSaver.save_plan: unsupported ModelType ") + model_type_str(model_type))
        var saver = StableDiffusion3ModelSaver()
        return saver.save_plan(output_model_format, output_model_destination, dtype_override)

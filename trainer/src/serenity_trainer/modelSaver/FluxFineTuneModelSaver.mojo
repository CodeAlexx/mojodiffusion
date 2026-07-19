# 1:1 surface port of Serenity
#   modules/modelSaver/FluxFineTuneModelSaver.py

from serenity_trainer.modelSaver.flux.FluxModelSaver import (
    FluxModelSavePlan,
    FluxModelSaver,
)
from serenity_trainer.util.enum.ModelType import (
    model_type_is_flux_1,
    model_type_str,
)


struct FluxFineTuneModelSaver(Movable):
    def __init__(out self):
        pass

    def save_plan(
        self,
        model_type: Int,
        output_model_format: Int,
        output_model_destination: String,
        dtype_override: String,
    ) raises -> FluxModelSavePlan:
        if not model_type_is_flux_1(model_type):
            raise Error(String("FluxFineTuneModelSaver.save_plan: unsupported ModelType ") + model_type_str(model_type))
        var saver = FluxModelSaver()
        return saver.save_plan(output_model_format, output_model_destination, dtype_override)

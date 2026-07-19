# 1:1 surface port of Serenity
#   modules/modelSaver/FluxEmbeddingModelSaver.py

from serenity_trainer.modelSaver.flux.FluxEmbeddingSaver import (
    FluxEmbeddingSavePlan,
    FluxEmbeddingSaver,
)
from serenity_trainer.util.enum.ModelType import (
    model_type_is_flux_1,
    model_type_str,
)


struct FluxEmbeddingModelSaver(Movable):
    def __init__(out self):
        pass

    def save_single_plan(
        self,
        model_type: Int,
        output_model_format: Int,
        output_model_destination: String,
    ) raises -> FluxEmbeddingSavePlan:
        if not model_type_is_flux_1(model_type):
            raise Error(String("FluxEmbeddingModelSaver.save_single_plan: unsupported ModelType ") + model_type_str(model_type))
        var saver = FluxEmbeddingSaver()
        return saver.save_single_plan(output_model_format, output_model_destination)

    def save_multiple_plan(
        self,
        model_type: Int,
        output_model_format: Int,
        output_model_destination: String,
    ) raises -> FluxEmbeddingSavePlan:
        if not model_type_is_flux_1(model_type):
            raise Error(String("FluxEmbeddingModelSaver.save_multiple_plan: unsupported ModelType ") + model_type_str(model_type))
        var saver = FluxEmbeddingSaver()
        return saver.save_multiple_plan(output_model_format, output_model_destination)

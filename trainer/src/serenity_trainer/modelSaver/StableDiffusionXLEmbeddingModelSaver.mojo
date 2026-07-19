# 1:1 surface port of Serenity
#   modules/modelSaver/StableDiffusionXLEmbeddingModelSaver.py

from serenity_trainer.modelSaver.stableDiffusionXL.StableDiffusionXLEmbeddingSaver import (
    StableDiffusionXLEmbeddingSavePlan,
    StableDiffusionXLEmbeddingSaver,
)
from serenity_trainer.util.enum.ModelType import (
    model_type_is_stable_diffusion_xl,
    model_type_str,
)


struct StableDiffusionXLEmbeddingModelSaver(Movable):
    def __init__(out self):
        pass

    def save_single_plan(
        self,
        model_type: Int,
        output_model_format: Int,
        output_model_destination: String,
    ) raises -> StableDiffusionXLEmbeddingSavePlan:
        if not model_type_is_stable_diffusion_xl(model_type):
            raise Error(String("StableDiffusionXLEmbeddingModelSaver.save_single_plan: unsupported ModelType ") + model_type_str(model_type))
        var saver = StableDiffusionXLEmbeddingSaver()
        return saver.save_single_plan(output_model_format, output_model_destination)

    def save_multiple_plan(
        self,
        model_type: Int,
        output_model_format: Int,
        output_model_destination: String,
    ) raises -> StableDiffusionXLEmbeddingSavePlan:
        if not model_type_is_stable_diffusion_xl(model_type):
            raise Error(String("StableDiffusionXLEmbeddingModelSaver.save_multiple_plan: unsupported ModelType ") + model_type_str(model_type))
        var saver = StableDiffusionXLEmbeddingSaver()
        return saver.save_multiple_plan(output_model_format, output_model_destination)

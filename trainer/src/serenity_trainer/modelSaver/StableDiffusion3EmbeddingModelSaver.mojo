# 1:1 surface port of Serenity
#   modules/modelSaver/StableDiffusion3EmbeddingModelSaver.py

from serenity_trainer.modelSaver.stableDiffusion3.StableDiffusion3EmbeddingSaver import (
    StableDiffusion3EmbeddingSavePlan,
    StableDiffusion3EmbeddingSaver,
)
from serenity_trainer.util.enum.ModelType import (
    model_type_is_stable_diffusion_3,
    model_type_str,
)


struct StableDiffusion3EmbeddingModelSaver(Movable):
    def __init__(out self):
        pass

    def save_single_plan(
        self,
        model_type: Int,
        output_model_format: Int,
        output_model_destination: String,
    ) raises -> StableDiffusion3EmbeddingSavePlan:
        if not model_type_is_stable_diffusion_3(model_type):
            raise Error(String("StableDiffusion3EmbeddingModelSaver.save_single_plan: unsupported ModelType ") + model_type_str(model_type))
        var saver = StableDiffusion3EmbeddingSaver()
        return saver.save_single_plan(output_model_format, output_model_destination)

    def save_multiple_plan(
        self,
        model_type: Int,
        output_model_format: Int,
        output_model_destination: String,
    ) raises -> StableDiffusion3EmbeddingSavePlan:
        if not model_type_is_stable_diffusion_3(model_type):
            raise Error(String("StableDiffusion3EmbeddingModelSaver.save_multiple_plan: unsupported ModelType ") + model_type_str(model_type))
        var saver = StableDiffusion3EmbeddingSaver()
        return saver.save_multiple_plan(output_model_format, output_model_destination)

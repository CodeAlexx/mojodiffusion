# 1:1 surface port of Serenity
#   modules/modelSaver/StableDiffusion3LoRAModelSaver.py

from std.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype

from serenity_trainer.modelSaver.stableDiffusion3.StableDiffusion3LoRASaver import (
    StableDiffusion3LoraSavePlan,
    StableDiffusion3LoraStateDict,
    StableDiffusion3LoRASaver,
    save_stable_diffusion3_lora_state_dict_as_dtype,
)
from serenity_trainer.util.enum.ModelType import (
    model_type_is_stable_diffusion_3,
    model_type_str,
)


struct StableDiffusion3LoRAModelSaver(Movable):
    def __init__(out self):
        pass

    def save_plan(
        self,
        model_type: Int,
        output_model_format: Int,
        output_model_destination: String,
    ) raises -> StableDiffusion3LoraSavePlan:
        if not model_type_is_stable_diffusion_3(model_type):
            raise Error(String("StableDiffusion3LoRAModelSaver.save_plan: unsupported ModelType ") + model_type_str(model_type))
        var saver = StableDiffusion3LoRASaver()
        return saver.save_plan(output_model_format, output_model_destination)

    def save(
        self,
        var state: StableDiffusion3LoraStateDict,
        model_type: Int,
        output_model_format: Int,
        output_model_destination: String,
        ctx: DeviceContext,
    ) raises:
        if not model_type_is_stable_diffusion_3(model_type):
            raise Error(String("StableDiffusion3LoRAModelSaver.save: unsupported ModelType ") + model_type_str(model_type))
        var saver = StableDiffusion3LoRASaver()
        saver.save(state^, output_model_format, output_model_destination, ctx)

    def save_as_dtype(
        self,
        state: StableDiffusion3LoraStateDict,
        model_type: Int,
        output_model_format: Int,
        output_model_destination: String,
        dtype: STDtype,
        ctx: DeviceContext,
    ) raises:
        if not model_type_is_stable_diffusion_3(model_type):
            raise Error(String("StableDiffusion3LoRAModelSaver.save_as_dtype: unsupported ModelType ") + model_type_str(model_type))
        save_stable_diffusion3_lora_state_dict_as_dtype(
            state, output_model_format, output_model_destination, dtype, ctx
        )

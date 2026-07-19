# 1:1 surface port of Serenity
#   modules/modelSaver/StableDiffusionXLLoRAModelSaver.py

from std.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype

from serenity_trainer.modelSaver.stableDiffusionXL.StableDiffusionXLLoRASaver import (
    StableDiffusionXLLoraSavePlan,
    StableDiffusionXLLoraStateDict,
    StableDiffusionXLLoRASaver,
    save_stable_diffusion_xl_lora_state_dict_as_dtype,
)
from serenity_trainer.util.enum.ModelType import (
    model_type_is_stable_diffusion_xl,
    model_type_str,
)


struct StableDiffusionXLLoRAModelSaver(Movable):
    def __init__(out self):
        pass

    def save_plan(
        self,
        model_type: Int,
        output_model_format: Int,
        output_model_destination: String,
    ) raises -> StableDiffusionXLLoraSavePlan:
        if not model_type_is_stable_diffusion_xl(model_type):
            raise Error(String("StableDiffusionXLLoRAModelSaver.save_plan: unsupported ModelType ") + model_type_str(model_type))
        var saver = StableDiffusionXLLoRASaver()
        return saver.save_plan(output_model_format, output_model_destination)

    def save(
        self,
        var state: StableDiffusionXLLoraStateDict,
        model_type: Int,
        output_model_format: Int,
        output_model_destination: String,
        ctx: DeviceContext,
    ) raises:
        if not model_type_is_stable_diffusion_xl(model_type):
            raise Error(String("StableDiffusionXLLoRAModelSaver.save: unsupported ModelType ") + model_type_str(model_type))
        var saver = StableDiffusionXLLoRASaver()
        saver.save(state^, output_model_format, output_model_destination, ctx)

    def save_as_dtype(
        self,
        state: StableDiffusionXLLoraStateDict,
        model_type: Int,
        output_model_format: Int,
        output_model_destination: String,
        dtype: STDtype,
        ctx: DeviceContext,
    ) raises:
        if not model_type_is_stable_diffusion_xl(model_type):
            raise Error(String("StableDiffusionXLLoRAModelSaver.save_as_dtype: unsupported ModelType ") + model_type_str(model_type))
        save_stable_diffusion_xl_lora_state_dict_as_dtype(
            state, output_model_format, output_model_destination, dtype, ctx
        )

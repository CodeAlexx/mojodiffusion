# 1:1 surface port of Serenity
#   modules/modelSaver/FluxLoRAModelSaver.py

from std.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype

from serenity_trainer.modelSaver.flux.FluxLoRASaver import (
    FluxLoraSavePlan,
    FluxLoraStateDict,
    FluxLoRASaver,
    save_flux_lora_state_dict_as_dtype,
)
from serenity_trainer.util.enum.ModelType import (
    model_type_is_flux_1,
    model_type_str,
)


struct FluxLoRAModelSaver(Movable):
    def __init__(out self):
        pass

    def save_plan(
        self,
        model_type: Int,
        output_model_format: Int,
        output_model_destination: String,
    ) raises -> FluxLoraSavePlan:
        if not model_type_is_flux_1(model_type):
            raise Error(String("FluxLoRAModelSaver.save_plan: unsupported ModelType ") + model_type_str(model_type))
        var saver = FluxLoRASaver()
        return saver.save_plan(output_model_format, output_model_destination)

    def save(
        self,
        var state: FluxLoraStateDict,
        model_type: Int,
        output_model_format: Int,
        output_model_destination: String,
        ctx: DeviceContext,
    ) raises:
        if not model_type_is_flux_1(model_type):
            raise Error(String("FluxLoRAModelSaver.save: unsupported ModelType ") + model_type_str(model_type))
        var saver = FluxLoRASaver()
        saver.save(state^, output_model_format, output_model_destination, ctx)

    def save_as_dtype(
        self,
        state: FluxLoraStateDict,
        model_type: Int,
        output_model_format: Int,
        output_model_destination: String,
        dtype: STDtype,
        ctx: DeviceContext,
    ) raises:
        if not model_type_is_flux_1(model_type):
            raise Error(String("FluxLoRAModelSaver.save_as_dtype: unsupported ModelType ") + model_type_str(model_type))
        save_flux_lora_state_dict_as_dtype(
            state, output_model_format, output_model_destination, dtype, ctx
        )

# 1:1 surface port of Serenity modules/modelSaver/ernie/ErnieLoRASaver.py
#
# Serenity ErnieLoRASaver:
#   _get_convert_key_sets -> None
#   _get_state_dict       -> transformer_lora + lora_state_dict
#   save                  -> LoRASaverMixin._save
#
# This file provides the raw-key state-dict save path. It preserves tensor
# storage dtype by default; call the explicit dtype override helper to cast.

from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.tensor import Tensor


comptime TArc = ArcPointer[Tensor]

comptime ERNIE_FMT_DIFFUSERS = 0
comptime ERNIE_FMT_CKPT = 1
comptime ERNIE_FMT_SAFETENSORS = 2
comptime ERNIE_FMT_LEGACY_SAFETENSORS = 3
comptime ERNIE_FMT_COMFY_LORA = 4
comptime ERNIE_FMT_INTERNAL = 5


struct ErnieLoraStateDict(Movable):
    var names: List[String]
    var tensors: List[TArc]

    def __init__(out self, var names: List[String], var tensors: List[TArc]):
        self.names = names^
        self.tensors = tensors^


def ernie_lora_saver_has_convert_key_sets() -> Bool:
    return False


def ernie_lora_state_dict_from_raw(
    var names: List[String],
    var tensors: List[TArc],
) raises -> ErnieLoraStateDict:
    if len(names) != len(tensors):
        raise Error("ernie_lora_state_dict_from_raw: names/tensors length mismatch")
    return ErnieLoraStateDict(names^, tensors^)


def save_ernie_lora_state_dict(
    var state: ErnieLoraStateDict,
    output_model_format: Int,
    output_model_destination: String,
    ctx: DeviceContext,
) raises:
    if output_model_format == ERNIE_FMT_SAFETENSORS \
            or output_model_format == ERNIE_FMT_LEGACY_SAFETENSORS:
        save_safetensors(state.names^, state.tensors^, output_model_destination, ctx)
    elif output_model_format == ERNIE_FMT_INTERNAL:
        var path = output_model_destination + String("/lora/lora.safetensors")
        save_safetensors(state.names^, state.tensors^, path, ctx)
    elif output_model_format == ERNIE_FMT_DIFFUSERS:
        raise Error("ErnieLoRASaver: DIFFUSERS LoRA output is not implemented in Serenity")
    else:
        raise Error("ErnieLoRASaver: unsupported ModelFormat")


def save_ernie_lora_state_dict_as_dtype(
    state: ErnieLoraStateDict,
    output_model_format: Int,
    output_model_destination: String,
    dtype: STDtype,
    ctx: DeviceContext,
) raises:
    var names = List[String]()
    var tensors = List[TArc]()
    for i in range(len(state.names)):
        names.append(state.names[i].copy())
        tensors.append(TArc(cast_tensor(state.tensors[i][], dtype, ctx)))
    var cast_state = ErnieLoraStateDict(names^, tensors^)
    save_ernie_lora_state_dict(cast_state^, output_model_format, output_model_destination, ctx)


struct ErnieLoRASaver(Movable):
    def __init__(out self):
        pass

    def _get_convert_key_sets(self) -> Bool:
        return ernie_lora_saver_has_convert_key_sets()

    def save(
        self,
        var state: ErnieLoraStateDict,
        output_model_format: Int,
        output_model_destination: String,
        ctx: DeviceContext,
    ) raises:
        save_ernie_lora_state_dict(state^, output_model_format, output_model_destination, ctx)

    def save_as_dtype(
        self,
        state: ErnieLoraStateDict,
        output_model_format: Int,
        output_model_destination: String,
        dtype: STDtype,
        ctx: DeviceContext,
    ) raises:
        save_ernie_lora_state_dict_as_dtype(state, output_model_format, output_model_destination, dtype, ctx)

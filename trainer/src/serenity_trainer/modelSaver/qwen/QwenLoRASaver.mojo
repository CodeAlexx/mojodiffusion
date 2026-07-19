# 1:1 surface port of Serenity modules/modelSaver/qwen/QwenLoRASaver.py
#
# Serenity QwenLoRASaver:
#   _get_convert_key_sets -> None
#   _get_state_dict       -> text_encoder_lora + transformer_lora + lora_state_dict
#   save                  -> LoRASaverMixin._save
#
# This file provides the raw-key state-dict save path. It preserves tensor
# storage dtype by default; call the explicit dtype override helper to cast.

from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import zeros_device, add_scalar
from serenitymojo.tensor import Tensor

from serenity_trainer.modelSetup.qwenLoraTargets import (
    QwenLoraTargetSpecs,
    qwen_lora_down_key,
    qwen_lora_up_key,
    qwen_lora_alpha_key,
)


comptime TArc = ArcPointer[Tensor]

comptime QWEN_FMT_DIFFUSERS = 0
comptime QWEN_FMT_CKPT = 1
comptime QWEN_FMT_SAFETENSORS = 2
comptime QWEN_FMT_LEGACY_SAFETENSORS = 3
comptime QWEN_FMT_COMFY_LORA = 4
comptime QWEN_FMT_INTERNAL = 5


struct QwenLoraStateDict(Movable):
    var names: List[String]
    var tensors: List[TArc]

    def __init__(out self, var names: List[String], var tensors: List[TArc]):
        self.names = names^
        self.tensors = tensors^


def qwen_lora_saver_has_convert_key_sets() -> Bool:
    return False


def qwen_lora_state_dict_from_raw(
    var names: List[String],
    var tensors: List[TArc],
) raises -> QwenLoraStateDict:
    if len(names) != len(tensors):
        raise Error("qwen_lora_state_dict_from_raw: names/tensors length mismatch")
    return QwenLoraStateDict(names^, tensors^)


def build_qwen_lora_state_dict_from_targets(
    targets: QwenLoraTargetSpecs,
    rank: Int,
    alpha: Float32,
    ctx: DeviceContext,
    dtype: STDtype = STDtype.BF16,
) raises -> QwenLoraStateDict:
    if rank <= 0:
        raise Error("build_qwen_lora_state_dict_from_targets: rank must be positive")
    if targets.len() == 0:
        raise Error("build_qwen_lora_state_dict_from_targets: no targets")

    var names = List[String]()
    var tensors = List[TArc]()
    for i in range(targets.len()):
        var prefix = targets.prefixes[i]
        var in_features = targets.in_features[i]
        var out_features = targets.out_features[i]

        names.append(qwen_lora_down_key(prefix))
        tensors.append(TArc(_zeros2(rank, in_features, dtype, ctx)))

        names.append(qwen_lora_up_key(prefix))
        tensors.append(TArc(_zeros2(out_features, rank, dtype, ctx)))

        names.append(qwen_lora_alpha_key(prefix))
        tensors.append(TArc(_scalar(alpha, dtype, ctx)))

    return QwenLoraStateDict(names^, tensors^)


def save_qwen_lora_state_dict(
    var state: QwenLoraStateDict,
    output_model_format: Int,
    output_model_destination: String,
    ctx: DeviceContext,
) raises:
    if output_model_format == QWEN_FMT_SAFETENSORS \
            or output_model_format == QWEN_FMT_LEGACY_SAFETENSORS:
        save_safetensors(state.names^, state.tensors^, output_model_destination, ctx)
    elif output_model_format == QWEN_FMT_INTERNAL:
        var path = output_model_destination + String("/lora/lora.safetensors")
        save_safetensors(state.names^, state.tensors^, path, ctx)
    elif output_model_format == QWEN_FMT_DIFFUSERS:
        raise Error("QwenLoRASaver: DIFFUSERS LoRA output is not implemented in Serenity")
    else:
        raise Error("QwenLoRASaver: unsupported ModelFormat")


def save_qwen_lora_state_dict_as_dtype(
    state: QwenLoraStateDict,
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
    var cast_state = QwenLoraStateDict(names^, tensors^)
    save_qwen_lora_state_dict(cast_state^, output_model_format, output_model_destination, ctx)


def _zeros2(rows: Int, cols: Int, dtype: STDtype, ctx: DeviceContext) raises -> Tensor:
    var sh = List[Int]()
    sh.append(rows)
    sh.append(cols)
    return zeros_device(sh^, dtype, ctx)


def _scalar(val: Float32, dtype: STDtype, ctx: DeviceContext) raises -> Tensor:
    # Serenity LoRAModule registers alpha as a scalar buffer and module.to(dtype)
    # casts that buffer with the rest of the LoRA module.
    var sh = List[Int]()
    var z = zeros_device(sh^, dtype, ctx)
    return add_scalar(z, val, ctx)


struct QwenLoRASaver(Movable):
    def __init__(out self):
        pass

    def _get_convert_key_sets(self) -> Bool:
        return qwen_lora_saver_has_convert_key_sets()

    def save(
        self,
        var state: QwenLoraStateDict,
        output_model_format: Int,
        output_model_destination: String,
        ctx: DeviceContext,
    ) raises:
        save_qwen_lora_state_dict(state^, output_model_format, output_model_destination, ctx)

    def save_as_dtype(
        self,
        state: QwenLoraStateDict,
        output_model_format: Int,
        output_model_destination: String,
        dtype: STDtype,
        ctx: DeviceContext,
    ) raises:
        save_qwen_lora_state_dict_as_dtype(state, output_model_format, output_model_destination, dtype, ctx)

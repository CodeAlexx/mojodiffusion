# 1:1 surface port of Serenity modules/modelSaver/qwen/QwenModelSaver.py
#
# Build-only fine-tune saver surface. Serenity can save full diffusers Qwen
# pipelines and transformer-only safetensors. The Mojo Qwen runtime is not in
# scope for this worker, so this file exposes save plans plus a raw transformer
# safetensors helper.

from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.tensor import Tensor

from serenity_trainer.modelSaver.qwen.QwenLoRASaver import (
    QWEN_FMT_DIFFUSERS,
    QWEN_FMT_INTERNAL,
    QWEN_FMT_SAFETENSORS,
)


comptime TArc = ArcPointer[Tensor]


struct QwenModelSavePlan(Movable):
    var output_model_format: Int
    var output_model_destination: String
    var route_name: String
    var dtype_override: String

    def __init__(
        out self,
        output_model_format: Int,
        var output_model_destination: String,
        var route_name: String,
        var dtype_override: String,
    ):
        self.output_model_format = output_model_format
        self.output_model_destination = output_model_destination^
        self.route_name = route_name^
        self.dtype_override = dtype_override^


def qwen_model_saver_plan(
    output_model_format: Int,
    output_model_destination: String,
    dtype_override: String,
) raises -> QwenModelSavePlan:
    if output_model_format == QWEN_FMT_DIFFUSERS:
        return QwenModelSavePlan(
            output_model_format,
            output_model_destination.copy(),
            String("diffusers_pipeline"),
            dtype_override.copy(),
        )
    if output_model_format == QWEN_FMT_SAFETENSORS:
        return QwenModelSavePlan(
            output_model_format,
            output_model_destination.copy(),
            String("transformer_safetensors"),
            dtype_override.copy(),
        )
    if output_model_format == QWEN_FMT_INTERNAL:
        return QwenModelSavePlan(
            output_model_format,
            output_model_destination.copy(),
            String("internal_diffusers"),
            String(),
        )
    raise Error("QwenModelSaver: unsupported ModelFormat")


def save_qwen_transformer_safetensors(
    var names: List[String],
    var tensors: List[TArc],
    output_model_destination: String,
    ctx: DeviceContext,
) raises:
    if len(names) != len(tensors):
        raise Error("save_qwen_transformer_safetensors: names/tensors length mismatch")
    save_safetensors(names^, tensors^, output_model_destination, ctx)


def save_qwen_transformer_safetensors_as_dtype(
    names_in: List[String],
    tensors_in: List[TArc],
    output_model_destination: String,
    dtype: STDtype,
    ctx: DeviceContext,
) raises:
    if len(names_in) != len(tensors_in):
        raise Error("save_qwen_transformer_safetensors_as_dtype: names/tensors length mismatch")
    var names = List[String]()
    var tensors = List[TArc]()
    for i in range(len(names_in)):
        names.append(names_in[i].copy())
        tensors.append(TArc(cast_tensor(tensors_in[i][], dtype, ctx)))
    save_safetensors(names^, tensors^, output_model_destination, ctx)


struct QwenModelSaver(Movable):
    def __init__(out self):
        pass

    def save_plan(
        self,
        output_model_format: Int,
        output_model_destination: String,
        dtype_override: String,
    ) raises -> QwenModelSavePlan:
        return qwen_model_saver_plan(output_model_format, output_model_destination, dtype_override)

# 1:1 surface port of Serenity modules/modelSaver/ernie/ErnieModelSaver.py
#
# Build-only fine-tune saver surface. Serenity can save full diffusers Ernie
# pipelines, internal diffusers directories, and transformer-only safetensors.
# The Mojo Ernie runtime is not in this worker's scope, so this file exposes save
# plans plus a raw transformer safetensors helper.

from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.tensor import Tensor

from serenity_trainer.modelSaver.ernie.ErnieLoRASaver import (
    ERNIE_FMT_DIFFUSERS,
    ERNIE_FMT_INTERNAL,
    ERNIE_FMT_SAFETENSORS,
)


comptime TArc = ArcPointer[Tensor]


struct ErnieModelSavePlan(Movable):
    var output_model_format: Int
    var output_model_destination: String
    var route_name: String
    var dtype_override: String
    var saves_pipeline: Bool
    var saves_transformer_only: Bool

    def __init__(
        out self,
        output_model_format: Int,
        var output_model_destination: String,
        var route_name: String,
        var dtype_override: String,
        saves_pipeline: Bool,
        saves_transformer_only: Bool,
    ):
        self.output_model_format = output_model_format
        self.output_model_destination = output_model_destination^
        self.route_name = route_name^
        self.dtype_override = dtype_override^
        self.saves_pipeline = saves_pipeline
        self.saves_transformer_only = saves_transformer_only


def ernie_model_saver_plan(
    output_model_format: Int,
    output_model_destination: String,
    dtype_override: String,
) raises -> ErnieModelSavePlan:
    if output_model_format == ERNIE_FMT_DIFFUSERS:
        return ErnieModelSavePlan(
            output_model_format,
            output_model_destination.copy(),
            String("diffusers_pipeline"),
            dtype_override.copy(),
            True,
            False,
        )
    if output_model_format == ERNIE_FMT_SAFETENSORS:
        return ErnieModelSavePlan(
            output_model_format,
            output_model_destination.copy(),
            String("transformer_safetensors"),
            dtype_override.copy(),
            False,
            True,
        )
    if output_model_format == ERNIE_FMT_INTERNAL:
        return ErnieModelSavePlan(
            output_model_format,
            output_model_destination.copy(),
            String("internal_diffusers"),
            String(),
            True,
            False,
        )
    raise Error("ErnieModelSaver: unsupported ModelFormat")


def save_ernie_transformer_safetensors(
    var names: List[String],
    var tensors: List[TArc],
    output_model_destination: String,
    ctx: DeviceContext,
) raises:
    if len(names) != len(tensors):
        raise Error("save_ernie_transformer_safetensors: names/tensors length mismatch")
    save_safetensors(names^, tensors^, output_model_destination, ctx)


def save_ernie_transformer_safetensors_as_dtype(
    names_in: List[String],
    tensors_in: List[TArc],
    output_model_destination: String,
    dtype: STDtype,
    ctx: DeviceContext,
) raises:
    if len(names_in) != len(tensors_in):
        raise Error("save_ernie_transformer_safetensors_as_dtype: names/tensors length mismatch")
    var names = List[String]()
    var tensors = List[TArc]()
    for i in range(len(names_in)):
        names.append(names_in[i].copy())
        tensors.append(TArc(cast_tensor(tensors_in[i][], dtype, ctx)))
    save_safetensors(names^, tensors^, output_model_destination, ctx)


struct ErnieModelSaver(Movable):
    def __init__(out self):
        pass

    def save_plan(
        self,
        output_model_format: Int,
        output_model_destination: String,
        dtype_override: String,
    ) raises -> ErnieModelSavePlan:
        return ernie_model_saver_plan(output_model_format, output_model_destination, dtype_override)

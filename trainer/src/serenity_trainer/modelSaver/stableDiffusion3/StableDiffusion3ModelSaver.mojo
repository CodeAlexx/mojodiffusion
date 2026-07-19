# 1:1 surface port of Serenity
#   modules/modelSaver/stableDiffusion3/StableDiffusion3ModelSaver.py
#
# Build-only full-model saver support. Serenity can save SD3/SD3.5 as a
# diffusers pipeline, as a converted original safetensors checkpoint, or as the
# internal diffusers directory used for resume. This file exposes those route
# plans and a raw safetensors helper. It does not implement the SD3 diffusers to
# checkpoint key converter.

from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.tensor import Tensor

from serenity_trainer.modelSaver.stableDiffusion3.StableDiffusion3LoRASaver import (
    SD3_FMT_DIFFUSERS,
    SD3_FMT_INTERNAL,
    SD3_FMT_SAFETENSORS,
)


comptime TArc = ArcPointer[Tensor]


struct StableDiffusion3ModelSavePlan(Movable):
    var output_model_format: Int
    var output_model_destination: String
    var route_name: String
    var dtype_override: String
    var saves_diffusers_pipeline: Bool
    var saves_original_safetensors_checkpoint: Bool
    var saves_internal_diffusers: Bool
    var uses_diffusers_to_ckpt_converter: Bool
    var converter_name: String
    var save_pipeline_to_cpu_first: Bool
    var deep_copy_pipeline_when_dtype_override: Bool
    var preserves_storage_dtype_without_override: Bool
    var patches_t5_max_shard_size: Bool
    var t5_max_shard_size: String
    var includes_vae_state_dict: Bool
    var includes_transformer_state_dict: Bool
    var includes_text_encoder_1_state_dict_when_present: Bool
    var includes_text_encoder_2_state_dict_when_present: Bool
    var includes_text_encoder_3_state_dict_when_present: Bool

    def __init__(
        out self,
        output_model_format: Int,
        var output_model_destination: String,
        var route_name: String,
        var dtype_override: String,
        saves_diffusers_pipeline: Bool,
        saves_original_safetensors_checkpoint: Bool,
        saves_internal_diffusers: Bool,
    ):
        self.output_model_format = output_model_format
        self.output_model_destination = output_model_destination^
        self.route_name = route_name^
        self.dtype_override = dtype_override^
        self.saves_diffusers_pipeline = saves_diffusers_pipeline
        self.saves_original_safetensors_checkpoint = saves_original_safetensors_checkpoint
        self.saves_internal_diffusers = saves_internal_diffusers
        self.uses_diffusers_to_ckpt_converter = saves_original_safetensors_checkpoint
        self.converter_name = String("convert_sd3_diffusers_to_ckpt")
        self.save_pipeline_to_cpu_first = saves_diffusers_pipeline or saves_internal_diffusers
        self.deep_copy_pipeline_when_dtype_override = self.dtype_override.byte_length() > 0
        self.preserves_storage_dtype_without_override = self.dtype_override.byte_length() == 0
        self.patches_t5_max_shard_size = saves_diffusers_pipeline or saves_internal_diffusers
        self.t5_max_shard_size = String("2GB")
        self.includes_vae_state_dict = True
        self.includes_transformer_state_dict = True
        self.includes_text_encoder_1_state_dict_when_present = True
        self.includes_text_encoder_2_state_dict_when_present = True
        self.includes_text_encoder_3_state_dict_when_present = True


def stable_diffusion3_model_saver_plan(
    output_model_format: Int,
    output_model_destination: String,
    dtype_override: String,
) raises -> StableDiffusion3ModelSavePlan:
    if output_model_format == SD3_FMT_DIFFUSERS:
        return StableDiffusion3ModelSavePlan(
            output_model_format,
            output_model_destination.copy(),
            String("diffusers_pipeline"),
            dtype_override.copy(),
            True,
            False,
            False,
        )
    if output_model_format == SD3_FMT_SAFETENSORS:
        return StableDiffusion3ModelSavePlan(
            output_model_format,
            output_model_destination.copy(),
            String("original_safetensors_checkpoint"),
            dtype_override.copy(),
            False,
            True,
            False,
        )
    if output_model_format == SD3_FMT_INTERNAL:
        return StableDiffusion3ModelSavePlan(
            output_model_format,
            output_model_destination.copy(),
            String("internal_diffusers"),
            String(),
            True,
            False,
            True,
        )
    raise Error("StableDiffusion3ModelSaver: unsupported ModelFormat")


def save_stable_diffusion3_checkpoint_safetensors(
    var names: List[String],
    var tensors: List[TArc],
    output_model_destination: String,
    ctx: DeviceContext,
) raises:
    if len(names) != len(tensors):
        raise Error("save_stable_diffusion3_checkpoint_safetensors: names/tensors length mismatch")
    save_safetensors(names^, tensors^, output_model_destination, ctx)


def save_stable_diffusion3_checkpoint_safetensors_as_dtype(
    names_in: List[String],
    tensors_in: List[TArc],
    output_model_destination: String,
    dtype: STDtype,
    ctx: DeviceContext,
) raises:
    if len(names_in) != len(tensors_in):
        raise Error("save_stable_diffusion3_checkpoint_safetensors_as_dtype: names/tensors length mismatch")
    var names = List[String]()
    var tensors = List[TArc]()
    for i in range(len(names_in)):
        names.append(names_in[i].copy())
        tensors.append(TArc(cast_tensor(tensors_in[i][], dtype, ctx)))
    save_safetensors(names^, tensors^, output_model_destination, ctx)


struct StableDiffusion3ModelSaver(Movable):
    def __init__(out self):
        pass

    def save_plan(
        self,
        output_model_format: Int,
        output_model_destination: String,
        dtype_override: String,
    ) raises -> StableDiffusion3ModelSavePlan:
        return stable_diffusion3_model_saver_plan(
            output_model_format,
            output_model_destination,
            dtype_override,
        )

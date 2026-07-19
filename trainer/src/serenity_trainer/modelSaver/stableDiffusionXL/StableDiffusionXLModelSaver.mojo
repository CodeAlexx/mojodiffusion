# 1:1 surface port of Serenity
#   modules/modelSaver/stableDiffusionXL/StableDiffusionXLModelSaver.py
#
# Build-only full-model saver support. Serenity can save SDXL as a diffusers
# pipeline, converted original safetensors checkpoint plus YAML config, or the
# internal diffusers directory used for resume. This file exposes those route
# plans and raw safetensors helpers; it does not implement the diffusers to
# checkpoint key converter.

from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.tensor import Tensor

from serenity_trainer.modelSaver.stableDiffusionXL.StableDiffusionXLLoRASaver import (
    SDXL_FMT_DIFFUSERS,
    SDXL_FMT_INTERNAL,
    SDXL_FMT_SAFETENSORS,
)


comptime TArc = ArcPointer[Tensor]


struct StableDiffusionXLModelSavePlan(Movable):
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
    var writes_yaml_config_sidecar: Bool
    var yaml_sidecar_extension: String
    var includes_vae_state_dict: Bool
    var includes_unet_state_dict: Bool
    var includes_text_encoder_1_state_dict: Bool
    var includes_text_encoder_2_state_dict: Bool
    var includes_noise_scheduler_state: Bool
    var checkpoint_key_roots: String
    var adds_v_prediction_marker_when_scheduler_uses_v_prediction: Bool

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
        self.converter_name = String("convert_sdxl_diffusers_to_ckpt")
        self.save_pipeline_to_cpu_first = saves_diffusers_pipeline or saves_internal_diffusers
        self.deep_copy_pipeline_when_dtype_override = self.dtype_override.byte_length() > 0
        self.preserves_storage_dtype_without_override = self.dtype_override.byte_length() == 0
        self.writes_yaml_config_sidecar = saves_original_safetensors_checkpoint
        self.yaml_sidecar_extension = String(".yaml")
        self.includes_vae_state_dict = True
        self.includes_unet_state_dict = True
        self.includes_text_encoder_1_state_dict = True
        self.includes_text_encoder_2_state_dict = True
        self.includes_noise_scheduler_state = True
        self.checkpoint_key_roots = String("first_stage_model,model.diffusion_model,conditioner.embedders.0.transformer,conditioner.embedders.1")
        self.adds_v_prediction_marker_when_scheduler_uses_v_prediction = True


def stable_diffusion_xl_model_saver_plan(
    output_model_format: Int,
    output_model_destination: String,
    dtype_override: String,
) raises -> StableDiffusionXLModelSavePlan:
    if output_model_format == SDXL_FMT_DIFFUSERS:
        return StableDiffusionXLModelSavePlan(
            output_model_format,
            output_model_destination.copy(),
            String("diffusers_pipeline"),
            dtype_override.copy(),
            True,
            False,
            False,
        )
    if output_model_format == SDXL_FMT_SAFETENSORS:
        return StableDiffusionXLModelSavePlan(
            output_model_format,
            output_model_destination.copy(),
            String("original_safetensors_checkpoint"),
            dtype_override.copy(),
            False,
            True,
            False,
        )
    if output_model_format == SDXL_FMT_INTERNAL:
        return StableDiffusionXLModelSavePlan(
            output_model_format,
            output_model_destination.copy(),
            String("internal_diffusers"),
            String(),
            True,
            False,
            True,
        )
    raise Error("StableDiffusionXLModelSaver: unsupported ModelFormat")


def stable_diffusion_xl_checkpoint_key_roots() -> List[String]:
    var roots = List[String]()
    roots.append(String("first_stage_model"))
    roots.append(String("model.diffusion_model"))
    roots.append(String("conditioner.embedders.0.transformer"))
    roots.append(String("conditioner.embedders.1"))
    roots.append(String("v_pred"))
    return roots^


def save_stable_diffusion_xl_checkpoint_safetensors(
    var names: List[String],
    var tensors: List[TArc],
    output_model_destination: String,
    ctx: DeviceContext,
) raises:
    if len(names) != len(tensors):
        raise Error("save_stable_diffusion_xl_checkpoint_safetensors: names/tensors length mismatch")
    save_safetensors(names^, tensors^, output_model_destination, ctx)


def save_stable_diffusion_xl_checkpoint_safetensors_as_dtype(
    names_in: List[String],
    tensors_in: List[TArc],
    output_model_destination: String,
    dtype: STDtype,
    ctx: DeviceContext,
) raises:
    if len(names_in) != len(tensors_in):
        raise Error("save_stable_diffusion_xl_checkpoint_safetensors_as_dtype: names/tensors length mismatch")
    var names = List[String]()
    var tensors = List[TArc]()
    for i in range(len(names_in)):
        names.append(names_in[i].copy())
        tensors.append(TArc(cast_tensor(tensors_in[i][], dtype, ctx)))
    save_safetensors(names^, tensors^, output_model_destination, ctx)


struct StableDiffusionXLModelSaver(Movable):
    def __init__(out self):
        pass

    def save_plan(
        self,
        output_model_format: Int,
        output_model_destination: String,
        dtype_override: String,
    ) raises -> StableDiffusionXLModelSavePlan:
        return stable_diffusion_xl_model_saver_plan(
            output_model_format,
            output_model_destination,
            dtype_override,
        )

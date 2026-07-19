# 1:1 surface port of Serenity
#   modules/modelSaver/flux/FluxModelSaver.py
#
# Build-only full-model saver support. Serenity can save FLUX as a diffusers
# pipeline, as a converted original transformer safetensors checkpoint, or as
# the internal diffusers directory used for resume. This file exposes those
# route plans and raw safetensors helpers; it does not implement the FLUX
# diffusers-to-original checkpoint key converter.

from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.tensor import Tensor

from serenity_trainer.modelSaver.flux.FluxLoRASaver import (
    FLUX_FMT_DIFFUSERS,
    FLUX_FMT_INTERNAL,
    FLUX_FMT_SAFETENSORS,
)


comptime TArc = ArcPointer[Tensor]


struct FluxModelSavePlan(Movable):
    var output_model_format: Int
    var output_model_destination: String
    var route_name: String
    var dtype_override: String
    var saves_diffusers_pipeline: Bool
    var saves_original_transformer_safetensors: Bool
    var saves_internal_diffusers: Bool
    var uses_diffusers_to_ckpt_converter: Bool
    var converter_name: String
    var converter_input_state_dict: String
    var converter_output_key_namespace: String
    var save_pipeline_to_cpu_first: Bool
    var deep_copy_pipeline_when_dtype_override: Bool
    var tokenizer_2_deepcopy_patched_for_dtype_override: Bool
    var patches_t5_max_shard_size: Bool
    var t5_max_shard_size: String
    var preserves_storage_dtype_without_override: Bool
    var includes_transformer_state_dict_for_safetensors: Bool
    var includes_vae_state_dict_for_safetensors: Bool
    var includes_text_encoder_state_dicts_for_safetensors: Bool
    var checkpoint_double_block_qkv_fusion: Bool
    var checkpoint_single_block_qkv_mlp_fusion: Bool
    var checkpoint_swaps_final_layer_adaln_chunks: Bool
    var checkpoint_key_roots: String

    def __init__(
        out self,
        output_model_format: Int,
        var output_model_destination: String,
        var route_name: String,
        var dtype_override: String,
        saves_diffusers_pipeline: Bool,
        saves_original_transformer_safetensors: Bool,
        saves_internal_diffusers: Bool,
    ):
        self.output_model_format = output_model_format
        self.output_model_destination = output_model_destination^
        self.route_name = route_name^
        self.dtype_override = dtype_override^
        self.saves_diffusers_pipeline = saves_diffusers_pipeline
        self.saves_original_transformer_safetensors = saves_original_transformer_safetensors
        self.saves_internal_diffusers = saves_internal_diffusers
        self.uses_diffusers_to_ckpt_converter = saves_original_transformer_safetensors
        self.converter_name = String("convert_flux_diffusers_to_ckpt")
        self.converter_input_state_dict = String("model.transformer.state_dict()")
        self.converter_output_key_namespace = String("original FLUX transformer root")
        self.save_pipeline_to_cpu_first = saves_diffusers_pipeline or saves_internal_diffusers
        self.deep_copy_pipeline_when_dtype_override = self.dtype_override.byte_length() > 0
        self.tokenizer_2_deepcopy_patched_for_dtype_override = self.deep_copy_pipeline_when_dtype_override
        self.patches_t5_max_shard_size = saves_diffusers_pipeline or saves_internal_diffusers
        self.t5_max_shard_size = String("2GB")
        self.preserves_storage_dtype_without_override = self.dtype_override.byte_length() == 0
        self.includes_transformer_state_dict_for_safetensors = saves_original_transformer_safetensors
        self.includes_vae_state_dict_for_safetensors = False
        self.includes_text_encoder_state_dicts_for_safetensors = False
        self.checkpoint_double_block_qkv_fusion = True
        self.checkpoint_single_block_qkv_mlp_fusion = True
        self.checkpoint_swaps_final_layer_adaln_chunks = True
        self.checkpoint_key_roots = String("double_blocks,single_blocks,txt_in,img_in,time_in,vector_in,guidance_in,final_layer")


def flux_model_saver_plan(
    output_model_format: Int,
    output_model_destination: String,
    dtype_override: String,
) raises -> FluxModelSavePlan:
    if output_model_format == FLUX_FMT_DIFFUSERS:
        return FluxModelSavePlan(
            output_model_format,
            output_model_destination.copy(),
            String("diffusers_pipeline"),
            dtype_override.copy(),
            True,
            False,
            False,
        )
    if output_model_format == FLUX_FMT_SAFETENSORS:
        return FluxModelSavePlan(
            output_model_format,
            output_model_destination.copy(),
            String("original_transformer_safetensors_checkpoint"),
            dtype_override.copy(),
            False,
            True,
            False,
        )
    if output_model_format == FLUX_FMT_INTERNAL:
        return FluxModelSavePlan(
            output_model_format,
            output_model_destination.copy(),
            String("internal_diffusers"),
            String(),
            True,
            False,
            True,
        )
    raise Error("FluxModelSaver: unsupported ModelFormat")


def flux_checkpoint_key_roots() -> List[String]:
    var roots = List[String]()
    roots.append(String("double_blocks"))
    roots.append(String("single_blocks"))
    roots.append(String("txt_in"))
    roots.append(String("img_in"))
    roots.append(String("time_in"))
    roots.append(String("vector_in"))
    roots.append(String("guidance_in"))
    roots.append(String("final_layer"))
    return roots^


def save_flux_checkpoint_safetensors(
    var names: List[String],
    var tensors: List[TArc],
    output_model_destination: String,
    ctx: DeviceContext,
) raises:
    if len(names) != len(tensors):
        raise Error("save_flux_checkpoint_safetensors: names/tensors length mismatch")
    save_safetensors(names^, tensors^, output_model_destination, ctx)


def save_flux_checkpoint_safetensors_as_dtype(
    names_in: List[String],
    tensors_in: List[TArc],
    output_model_destination: String,
    dtype: STDtype,
    ctx: DeviceContext,
) raises:
    if len(names_in) != len(tensors_in):
        raise Error("save_flux_checkpoint_safetensors_as_dtype: names/tensors length mismatch")
    var names = List[String]()
    var tensors = List[TArc]()
    for i in range(len(names_in)):
        names.append(names_in[i].copy())
        tensors.append(TArc(cast_tensor(tensors_in[i][], dtype, ctx)))
    save_safetensors(names^, tensors^, output_model_destination, ctx)


struct FluxModelSaver(Movable):
    def __init__(out self):
        pass

    def save_plan(
        self,
        output_model_format: Int,
        output_model_destination: String,
        dtype_override: String,
    ) raises -> FluxModelSavePlan:
        return flux_model_saver_plan(
            output_model_format,
            output_model_destination,
            dtype_override,
        )

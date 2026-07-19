# 1:1 surface port of Serenity
#   modules/modelSaver/stableDiffusionXL/StableDiffusionXLEmbeddingSaver.py
#
# SDXL embeddings carry up to four tensors:
#   clip_l, clip_g, clip_l_out, clip_g_out
# Tensor storage dtype is preserved unless the explicit dtype override helper is
# used.

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


struct StableDiffusionXLEmbeddingStateDict(Movable):
    var names: List[String]
    var tensors: List[TArc]

    def __init__(out self, var names: List[String], var tensors: List[TArc]):
        self.names = names^
        self.tensors = tensors^


struct StableDiffusionXLEmbeddingSavePlan(Movable):
    var output_model_format: Int
    var output_model_destination: String
    var route_name: String
    var is_multiple: Bool
    var destination_suffix_for_multiple: String
    var multiple_destination_template: String
    var internal_destination_template: String
    var key_clip_l: String
    var key_clip_g: String
    var key_clip_l_out: String
    var key_clip_g_out: String
    var diffusers_supported: Bool
    var current_primary_embedding_excluded_from_multiple: Bool
    var preserves_storage_dtype_without_override: Bool

    def __init__(
        out self,
        output_model_format: Int,
        var output_model_destination: String,
        var route_name: String,
        is_multiple: Bool,
    ):
        self.output_model_format = output_model_format
        self.output_model_destination = output_model_destination^
        self.route_name = route_name^
        self.is_multiple = is_multiple
        self.destination_suffix_for_multiple = String("_embeddings")
        self.multiple_destination_template = String("{output_model_destination}_embeddings/{safe_placeholder}.safetensors")
        self.internal_destination_template = String("{output_model_destination}/embeddings/{embedding_uuid}.safetensors")
        self.key_clip_l = String("clip_l")
        self.key_clip_g = String("clip_g")
        self.key_clip_l_out = String("clip_l_out")
        self.key_clip_g_out = String("clip_g_out")
        self.diffusers_supported = False
        self.current_primary_embedding_excluded_from_multiple = True
        self.preserves_storage_dtype_without_override = True


def stable_diffusion_xl_embedding_keys() -> List[String]:
    var keys = List[String]()
    keys.append(String("clip_l"))
    keys.append(String("clip_g"))
    keys.append(String("clip_l_out"))
    keys.append(String("clip_g_out"))
    return keys^


def stable_diffusion_xl_embedding_save_plan(
    output_model_format: Int,
    output_model_destination: String,
    is_multiple: Bool,
) raises -> StableDiffusionXLEmbeddingSavePlan:
    if output_model_format == SDXL_FMT_DIFFUSERS:
        raise Error("StableDiffusionXLEmbeddingSaver: DIFFUSERS embedding output is not implemented in Serenity")
    if output_model_format == SDXL_FMT_SAFETENSORS:
        return StableDiffusionXLEmbeddingSavePlan(
            output_model_format,
            output_model_destination.copy(),
            String("embedding_safetensors"),
            is_multiple,
        )
    if output_model_format == SDXL_FMT_INTERNAL:
        return StableDiffusionXLEmbeddingSavePlan(
            output_model_format,
            output_model_destination.copy(),
            String("internal_embedding"),
            is_multiple,
        )
    raise Error("StableDiffusionXLEmbeddingSaver: unsupported ModelFormat")


def stable_diffusion_xl_embedding_state_dict_from_raw(
    var names: List[String],
    var tensors: List[TArc],
) raises -> StableDiffusionXLEmbeddingStateDict:
    if len(names) != len(tensors):
        raise Error("stable_diffusion_xl_embedding_state_dict_from_raw: names/tensors length mismatch")
    return StableDiffusionXLEmbeddingStateDict(names^, tensors^)


def save_stable_diffusion_xl_embedding_state_dict(
    var state: StableDiffusionXLEmbeddingStateDict,
    output_model_destination: String,
    ctx: DeviceContext,
) raises:
    save_safetensors(state.names^, state.tensors^, output_model_destination, ctx)


def save_stable_diffusion_xl_embedding_state_dict_as_dtype(
    state: StableDiffusionXLEmbeddingStateDict,
    output_model_destination: String,
    dtype: STDtype,
    ctx: DeviceContext,
) raises:
    var names = List[String]()
    var tensors = List[TArc]()
    for i in range(len(state.names)):
        names.append(state.names[i].copy())
        tensors.append(TArc(cast_tensor(state.tensors[i][], dtype, ctx)))
    var cast_state = StableDiffusionXLEmbeddingStateDict(names^, tensors^)
    save_stable_diffusion_xl_embedding_state_dict(cast_state^, output_model_destination, ctx)


struct StableDiffusionXLEmbeddingSaver(Movable):
    def __init__(out self):
        pass

    def save_single_plan(
        self,
        output_model_format: Int,
        output_model_destination: String,
    ) raises -> StableDiffusionXLEmbeddingSavePlan:
        return stable_diffusion_xl_embedding_save_plan(
            output_model_format,
            output_model_destination,
            False,
        )

    def save_multiple_plan(
        self,
        output_model_format: Int,
        output_model_destination: String,
    ) raises -> StableDiffusionXLEmbeddingSavePlan:
        return stable_diffusion_xl_embedding_save_plan(
            output_model_format,
            output_model_destination,
            True,
        )

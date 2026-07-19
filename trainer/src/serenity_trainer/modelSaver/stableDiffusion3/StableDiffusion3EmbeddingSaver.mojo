# 1:1 surface port of Serenity
#   modules/modelSaver/stableDiffusion3/StableDiffusion3EmbeddingSaver.py
#
# SD3 embeddings carry up to six tensors:
#   clip_l, clip_g, t5, clip_l_out, clip_g_out, t5_out
# This file provides plan metadata plus low-level safetensors writes. Tensor
# storage dtype is preserved unless the explicit dtype override helper is used.

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


struct StableDiffusion3EmbeddingStateDict(Movable):
    var names: List[String]
    var tensors: List[TArc]

    def __init__(out self, var names: List[String], var tensors: List[TArc]):
        self.names = names^
        self.tensors = tensors^


struct StableDiffusion3EmbeddingSavePlan(Movable):
    var output_model_format: Int
    var output_model_destination: String
    var route_name: String
    var is_multiple: Bool
    var destination_suffix_for_multiple: String
    var key_clip_l: String
    var key_clip_g: String
    var key_t5: String
    var key_clip_l_out: String
    var key_clip_g_out: String
    var key_t5_out: String
    var diffusers_supported: Bool
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
        self.key_clip_l = String("clip_l")
        self.key_clip_g = String("clip_g")
        self.key_t5 = String("t5")
        self.key_clip_l_out = String("clip_l_out")
        self.key_clip_g_out = String("clip_g_out")
        self.key_t5_out = String("t5_out")
        self.diffusers_supported = False
        self.preserves_storage_dtype_without_override = True


def stable_diffusion3_embedding_keys() -> List[String]:
    var keys = List[String]()
    keys.append(String("clip_l"))
    keys.append(String("clip_g"))
    keys.append(String("t5"))
    keys.append(String("clip_l_out"))
    keys.append(String("clip_g_out"))
    keys.append(String("t5_out"))
    return keys^


def stable_diffusion3_embedding_save_plan(
    output_model_format: Int,
    output_model_destination: String,
    is_multiple: Bool,
) raises -> StableDiffusion3EmbeddingSavePlan:
    if output_model_format == SD3_FMT_DIFFUSERS:
        raise Error("StableDiffusion3EmbeddingSaver: DIFFUSERS embedding output is not implemented in Serenity")
    if output_model_format == SD3_FMT_SAFETENSORS:
        return StableDiffusion3EmbeddingSavePlan(
            output_model_format,
            output_model_destination.copy(),
            String("embedding_safetensors"),
            is_multiple,
        )
    if output_model_format == SD3_FMT_INTERNAL:
        return StableDiffusion3EmbeddingSavePlan(
            output_model_format,
            output_model_destination.copy(),
            String("internal_embedding"),
            is_multiple,
        )
    raise Error("StableDiffusion3EmbeddingSaver: unsupported ModelFormat")


def stable_diffusion3_embedding_state_dict_from_raw(
    var names: List[String],
    var tensors: List[TArc],
) raises -> StableDiffusion3EmbeddingStateDict:
    if len(names) != len(tensors):
        raise Error("stable_diffusion3_embedding_state_dict_from_raw: names/tensors length mismatch")
    return StableDiffusion3EmbeddingStateDict(names^, tensors^)


def save_stable_diffusion3_embedding_state_dict(
    var state: StableDiffusion3EmbeddingStateDict,
    output_model_destination: String,
    ctx: DeviceContext,
) raises:
    save_safetensors(state.names^, state.tensors^, output_model_destination, ctx)


def save_stable_diffusion3_embedding_state_dict_as_dtype(
    state: StableDiffusion3EmbeddingStateDict,
    output_model_destination: String,
    dtype: STDtype,
    ctx: DeviceContext,
) raises:
    var names = List[String]()
    var tensors = List[TArc]()
    for i in range(len(state.names)):
        names.append(state.names[i].copy())
        tensors.append(TArc(cast_tensor(state.tensors[i][], dtype, ctx)))
    var cast_state = StableDiffusion3EmbeddingStateDict(names^, tensors^)
    save_stable_diffusion3_embedding_state_dict(cast_state^, output_model_destination, ctx)


struct StableDiffusion3EmbeddingSaver(Movable):
    def __init__(out self):
        pass

    def save_single_plan(
        self,
        output_model_format: Int,
        output_model_destination: String,
    ) raises -> StableDiffusion3EmbeddingSavePlan:
        return stable_diffusion3_embedding_save_plan(
            output_model_format,
            output_model_destination,
            False,
        )

    def save_multiple_plan(
        self,
        output_model_format: Int,
        output_model_destination: String,
    ) raises -> StableDiffusion3EmbeddingSavePlan:
        return stable_diffusion3_embedding_save_plan(
            output_model_format,
            output_model_destination,
            True,
        )

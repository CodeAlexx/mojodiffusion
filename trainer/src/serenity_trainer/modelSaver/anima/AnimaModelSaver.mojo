# 1:1 surface port of Serenity-anima-ref modules/modelSaver/anima/AnimaModelSaver.py
#
# Build-only fine-tune saver surface. The reference can save a diffusers pipeline,
# an internal diffusers directory, or original-format safetensors. Original
# safetensors convert CosmosTransformer3DModel keys back to `net.*` names and add
# AnimaTextConditioner weights under `net.llm_adapter.*`.

from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.tensor import Tensor

from serenity_trainer.modelSaver.anima.AnimaLoRASaver import (
    ANIMA_FMT_DIFFUSERS,
    ANIMA_FMT_INTERNAL,
    ANIMA_FMT_SAFETENSORS,
)


comptime TArc = ArcPointer[Tensor]
comptime ANIMA_TEXT_CONDITIONER_ORIGINAL_PREFIX = "net.llm_adapter."


struct AnimaKeyRename(Copyable, Movable):
    var diffusers_prefix: String
    var original_prefix: String

    def __init__(out self, var diffusers_prefix: String, var original_prefix: String):
        self.diffusers_prefix = diffusers_prefix^
        self.original_prefix = original_prefix^

    def __init__(out self, *, copy: Self):
        self.diffusers_prefix = copy.diffusers_prefix.copy()
        self.original_prefix = copy.original_prefix.copy()


struct AnimaModelSavePlan(Movable):
    var output_model_format: Int
    var output_model_destination: String
    var route_name: String
    var dtype_override: String
    var saves_pipeline: Bool
    var saves_original_safetensors: Bool
    var text_conditioner_original_prefix: String
    var transformer_key_map_name: String
    var internal_ignores_dtype_override: Bool

    def __init__(
        out self,
        output_model_format: Int,
        var output_model_destination: String,
        var route_name: String,
        var dtype_override: String,
        saves_pipeline: Bool,
        saves_original_safetensors: Bool,
        internal_ignores_dtype_override: Bool,
    ):
        self.output_model_format = output_model_format
        self.output_model_destination = output_model_destination^
        self.route_name = route_name^
        self.dtype_override = dtype_override^
        self.saves_pipeline = saves_pipeline
        self.saves_original_safetensors = saves_original_safetensors
        self.text_conditioner_original_prefix = String(ANIMA_TEXT_CONDITIONER_ORIGINAL_PREFIX)
        self.transformer_key_map_name = String("diffusers_checkpoint_to_original")
        self.internal_ignores_dtype_override = internal_ignores_dtype_override


struct AnimaOriginalCheckpointStateDict(Movable):
    var names: List[String]
    var tensors: List[TArc]

    def __init__(out self, var names: List[String], var tensors: List[TArc]):
        self.names = names^
        self.tensors = tensors^


def anima_model_saver_plan(
    output_model_format: Int,
    output_model_destination: String,
    dtype_override: String,
) raises -> AnimaModelSavePlan:
    if output_model_format == ANIMA_FMT_DIFFUSERS:
        return AnimaModelSavePlan(
            output_model_format,
            output_model_destination.copy(),
            String("diffusers_pipeline"),
            dtype_override.copy(),
            True,
            False,
            False,
        )
    if output_model_format == ANIMA_FMT_SAFETENSORS:
        return AnimaModelSavePlan(
            output_model_format,
            output_model_destination.copy(),
            String("original_transformer_plus_text_conditioner_safetensors"),
            dtype_override.copy(),
            False,
            True,
            False,
        )
    if output_model_format == ANIMA_FMT_INTERNAL:
        return AnimaModelSavePlan(
            output_model_format,
            output_model_destination.copy(),
            String("internal_diffusers"),
            String(),
            True,
            False,
            True,
        )
    raise Error("AnimaModelSaver: unsupported ModelFormat")


def anima_diffusers_to_original_key_renames(num_transformer_blocks: Int) -> List[AnimaKeyRename]:
    var renames = List[AnimaKeyRename]()
    renames.append(AnimaKeyRename(String("patch_embed.proj"), String("net.x_embedder.proj.1")))
    renames.append(AnimaKeyRename(String("time_embed.t_embedder"), String("net.t_embedder.1")))
    renames.append(AnimaKeyRename(String("time_embed.norm"), String("net.t_embedding_norm")))
    renames.append(AnimaKeyRename(String("norm_out.linear_1"), String("net.final_layer.adaln_modulation.1")))
    renames.append(AnimaKeyRename(String("norm_out.linear_2"), String("net.final_layer.adaln_modulation.2")))
    renames.append(AnimaKeyRename(String("proj_out"), String("net.final_layer.linear")))

    for i in range(num_transformer_blocks):
        var d = String("transformer_blocks.") + String(i)
        var o = String("net.blocks.") + String(i)
        renames.append(AnimaKeyRename(d + String(".norm1.linear_1"), o + String(".adaln_modulation_self_attn.1")))
        renames.append(AnimaKeyRename(d + String(".norm1.linear_2"), o + String(".adaln_modulation_self_attn.2")))
        renames.append(AnimaKeyRename(d + String(".attn1.norm_q"), o + String(".self_attn.q_norm")))
        renames.append(AnimaKeyRename(d + String(".attn1.norm_k"), o + String(".self_attn.k_norm")))
        renames.append(AnimaKeyRename(d + String(".attn1.to_q"), o + String(".self_attn.q_proj")))
        renames.append(AnimaKeyRename(d + String(".attn1.to_k"), o + String(".self_attn.k_proj")))
        renames.append(AnimaKeyRename(d + String(".attn1.to_v"), o + String(".self_attn.v_proj")))
        renames.append(AnimaKeyRename(d + String(".attn1.to_out.0"), o + String(".self_attn.output_proj")))
        renames.append(AnimaKeyRename(d + String(".norm2.linear_1"), o + String(".adaln_modulation_cross_attn.1")))
        renames.append(AnimaKeyRename(d + String(".norm2.linear_2"), o + String(".adaln_modulation_cross_attn.2")))
        renames.append(AnimaKeyRename(d + String(".attn2.norm_q"), o + String(".cross_attn.q_norm")))
        renames.append(AnimaKeyRename(d + String(".attn2.norm_k"), o + String(".cross_attn.k_norm")))
        renames.append(AnimaKeyRename(d + String(".attn2.to_q"), o + String(".cross_attn.q_proj")))
        renames.append(AnimaKeyRename(d + String(".attn2.to_k"), o + String(".cross_attn.k_proj")))
        renames.append(AnimaKeyRename(d + String(".attn2.to_v"), o + String(".cross_attn.v_proj")))
        renames.append(AnimaKeyRename(d + String(".attn2.to_out.0"), o + String(".cross_attn.output_proj")))
        renames.append(AnimaKeyRename(d + String(".norm3.linear_1"), o + String(".adaln_modulation_mlp.1")))
        renames.append(AnimaKeyRename(d + String(".norm3.linear_2"), o + String(".adaln_modulation_mlp.2")))
        renames.append(AnimaKeyRename(d + String(".ff.net.0.proj"), o + String(".mlp.layer1")))
        renames.append(AnimaKeyRename(d + String(".ff.net.2"), o + String(".mlp.layer2")))
    return renames^


def anima_text_conditioner_original_key(name: String) -> String:
    return String(ANIMA_TEXT_CONDITIONER_ORIGINAL_PREFIX) + name


def anima_original_checkpoint_state_dict_from_parts(
    var transformer_original_names: List[String],
    var transformer_tensors: List[TArc],
    var text_conditioner_names: List[String],
    var text_conditioner_tensors: List[TArc],
) raises -> AnimaOriginalCheckpointStateDict:
    if len(transformer_original_names) != len(transformer_tensors):
        raise Error("anima_original_checkpoint_state_dict_from_parts: transformer names/tensors length mismatch")
    if len(text_conditioner_names) != len(text_conditioner_tensors):
        raise Error("anima_original_checkpoint_state_dict_from_parts: conditioner names/tensors length mismatch")

    var names = List[String]()
    var tensors = List[TArc]()
    for i in range(len(transformer_original_names)):
        names.append(transformer_original_names[i].copy())
        tensors.append(transformer_tensors[i].copy())
    for i in range(len(text_conditioner_names)):
        names.append(anima_text_conditioner_original_key(text_conditioner_names[i]))
        tensors.append(text_conditioner_tensors[i].copy())
    return AnimaOriginalCheckpointStateDict(names^, tensors^)


def save_anima_original_checkpoint_safetensors(
    var state: AnimaOriginalCheckpointStateDict,
    output_model_destination: String,
    ctx: DeviceContext,
) raises:
    save_safetensors(state.names^, state.tensors^, output_model_destination, ctx)


def save_anima_original_checkpoint_safetensors_as_dtype(
    state: AnimaOriginalCheckpointStateDict,
    output_model_destination: String,
    dtype: STDtype,
    ctx: DeviceContext,
) raises:
    var names = List[String]()
    var tensors = List[TArc]()
    for i in range(len(state.names)):
        names.append(state.names[i].copy())
        tensors.append(TArc(cast_tensor(state.tensors[i][], dtype, ctx)))
    var cast_state = AnimaOriginalCheckpointStateDict(names^, tensors^)
    save_anima_original_checkpoint_safetensors(cast_state^, output_model_destination, ctx)


struct AnimaModelSaver(Movable):
    def __init__(out self):
        pass

    def save_plan(
        self,
        output_model_format: Int,
        output_model_destination: String,
        dtype_override: String,
    ) raises -> AnimaModelSavePlan:
        return anima_model_saver_plan(output_model_format, output_model_destination, dtype_override)

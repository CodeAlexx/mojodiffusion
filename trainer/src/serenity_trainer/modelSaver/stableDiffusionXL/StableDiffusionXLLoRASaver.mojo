# 1:1 surface port of Serenity
#   modules/modelSaver/stableDiffusionXL/StableDiffusionXLLoRASaver.py
#
# Serenity SDXL LoRA saver:
#   _get_convert_key_sets -> convert_sdxl_lora_key_sets()
#   _get_state_dict       -> TE1 + TE2 + UNet LoRA + preloaded state
#                            + optional bundled clip_l/clip_g embeddings
#   save                  -> LoRASaverMixin._save
#
# The low-level write helpers preserve tensor storage dtype by default. Use the
# explicit dtype override helper when a caller intentionally wants conversion.

from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.tensor import Tensor

from serenity_trainer.modelLoader.stableDiffusionXL.StableDiffusionXLModelLoader import (
    StableDiffusionXLLoraConversionPlan,
    stable_diffusion_xl_lora_conversion_plan,
)


comptime TArc = ArcPointer[Tensor]

comptime SDXL_FMT_DIFFUSERS = 0
comptime SDXL_FMT_CKPT = 1
comptime SDXL_FMT_SAFETENSORS = 2
comptime SDXL_FMT_LEGACY_SAFETENSORS = 3
comptime SDXL_FMT_COMFY_LORA = 4
comptime SDXL_FMT_INTERNAL = 5


struct StableDiffusionXLLoraStateDict(Movable):
    var names: List[String]
    var tensors: List[TArc]

    def __init__(out self, var names: List[String], var tensors: List[TArc]):
        self.names = names^
        self.tensors = tensors^


struct StableDiffusionXLLoraSavePlan(Movable):
    var output_model_format: Int
    var output_model_destination: String
    var route_name: String
    var target_key_namespace: String
    var writes_safetensors: Bool
    var internal_destination: String
    var has_convert_key_sets: Bool
    var conversion_plan: StableDiffusionXLLoraConversionPlan
    var includes_text_encoder_1_lora: Bool
    var includes_text_encoder_2_lora: Bool
    var includes_unet_lora: Bool
    var includes_preloaded_lora_state_dict: Bool
    var can_bundle_additional_embeddings: Bool
    var bundled_embedding_keys: String
    var preserves_storage_dtype_without_override: Bool

    def __init__(
        out self,
        output_model_format: Int,
        var output_model_destination: String,
        var route_name: String,
        var target_key_namespace: String,
        var internal_destination: String,
        writes_safetensors: Bool,
    ):
        self.output_model_format = output_model_format
        self.output_model_destination = output_model_destination^
        self.route_name = route_name^
        self.target_key_namespace = target_key_namespace^
        self.writes_safetensors = writes_safetensors
        self.internal_destination = internal_destination^
        self.has_convert_key_sets = True
        self.conversion_plan = stable_diffusion_xl_lora_conversion_plan()
        self.includes_text_encoder_1_lora = True
        self.includes_text_encoder_2_lora = True
        self.includes_unet_lora = True
        self.includes_preloaded_lora_state_dict = True
        self.can_bundle_additional_embeddings = True
        self.bundled_embedding_keys = String("bundle_emb.{placeholder}.clip_l,clip_g,clip_l_out,clip_g_out")
        self.preserves_storage_dtype_without_override = True


def stable_diffusion_xl_lora_saver_has_convert_key_sets() -> Bool:
    return True


def stable_diffusion_xl_lora_save_plan(
    output_model_format: Int,
    output_model_destination: String,
) raises -> StableDiffusionXLLoraSavePlan:
    if output_model_format == SDXL_FMT_SAFETENSORS:
        return StableDiffusionXLLoraSavePlan(
            output_model_format,
            output_model_destination.copy(),
            String("legacy_safetensors"),
            String("legacy_diffusers"),
            String(),
            True,
        )
    if output_model_format == SDXL_FMT_LEGACY_SAFETENSORS:
        return StableDiffusionXLLoraSavePlan(
            output_model_format,
            output_model_destination.copy(),
            String("legacy_safetensors"),
            String("legacy_diffusers"),
            String(),
            True,
        )
    if output_model_format == SDXL_FMT_INTERNAL:
        return StableDiffusionXLLoraSavePlan(
            output_model_format,
            output_model_destination.copy(),
            String("internal_lora"),
            String("omi"),
            output_model_destination + String("/lora/lora.safetensors"),
            True,
        )
    if output_model_format == SDXL_FMT_DIFFUSERS:
        raise Error("StableDiffusionXLLoRASaver: DIFFUSERS LoRA output is not implemented in Serenity")
    raise Error("StableDiffusionXLLoRASaver: unsupported ModelFormat")


def stable_diffusion_xl_lora_bundle_embedding_keys(placeholder: String) -> List[String]:
    var keys = List[String]()
    keys.append(String("bundle_emb.") + placeholder + String(".clip_l"))
    keys.append(String("bundle_emb.") + placeholder + String(".clip_g"))
    keys.append(String("bundle_emb.") + placeholder + String(".clip_l_out"))
    keys.append(String("bundle_emb.") + placeholder + String(".clip_g_out"))
    return keys^


def stable_diffusion_xl_lora_state_dict_from_raw(
    var names: List[String],
    var tensors: List[TArc],
) raises -> StableDiffusionXLLoraStateDict:
    if len(names) != len(tensors):
        raise Error("stable_diffusion_xl_lora_state_dict_from_raw: names/tensors length mismatch")
    return StableDiffusionXLLoraStateDict(names^, tensors^)


def save_stable_diffusion_xl_lora_state_dict(
    var state: StableDiffusionXLLoraStateDict,
    output_model_format: Int,
    output_model_destination: String,
    ctx: DeviceContext,
) raises:
    var plan = stable_diffusion_xl_lora_save_plan(output_model_format, output_model_destination)
    if not plan.writes_safetensors:
        raise Error("save_stable_diffusion_xl_lora_state_dict: unsupported non-safetensors route")
    if plan.internal_destination.byte_length() > 0:
        save_safetensors(state.names^, state.tensors^, plan.internal_destination, ctx)
    else:
        save_safetensors(state.names^, state.tensors^, output_model_destination, ctx)


def save_stable_diffusion_xl_lora_state_dict_as_dtype(
    state: StableDiffusionXLLoraStateDict,
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
    var cast_state = StableDiffusionXLLoraStateDict(names^, tensors^)
    save_stable_diffusion_xl_lora_state_dict(cast_state^, output_model_format, output_model_destination, ctx)


struct StableDiffusionXLLoRASaver(Movable):
    def __init__(out self):
        pass

    def _get_convert_key_sets(self) -> Bool:
        return stable_diffusion_xl_lora_saver_has_convert_key_sets()

    def conversion_plan(self) -> StableDiffusionXLLoraConversionPlan:
        return stable_diffusion_xl_lora_conversion_plan()

    def save_plan(
        self,
        output_model_format: Int,
        output_model_destination: String,
    ) raises -> StableDiffusionXLLoraSavePlan:
        return stable_diffusion_xl_lora_save_plan(output_model_format, output_model_destination)

    def save(
        self,
        var state: StableDiffusionXLLoraStateDict,
        output_model_format: Int,
        output_model_destination: String,
        ctx: DeviceContext,
    ) raises:
        save_stable_diffusion_xl_lora_state_dict(state^, output_model_format, output_model_destination, ctx)

    def save_as_dtype(
        self,
        state: StableDiffusionXLLoraStateDict,
        output_model_format: Int,
        output_model_destination: String,
        dtype: STDtype,
        ctx: DeviceContext,
    ) raises:
        save_stable_diffusion_xl_lora_state_dict_as_dtype(state, output_model_format, output_model_destination, dtype, ctx)

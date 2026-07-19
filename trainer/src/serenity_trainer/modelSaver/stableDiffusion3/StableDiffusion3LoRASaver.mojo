# 1:1 surface port of Serenity
#   modules/modelSaver/stableDiffusion3/StableDiffusion3LoRASaver.py
#
# Serenity SD3 LoRA saver:
#   _get_convert_key_sets -> convert_sd3_lora_key_sets()
#   _get_state_dict       -> TE1 + TE2 + TE3 + transformer LoRA + preloaded state
#                            + optional bundled embeddings under bundle_emb.*
#   save                  -> LoRASaverMixin._save
#
# This file exposes the key conversion contract and low-level safetensors write
# helpers. The helpers preserve tensor storage dtype by default; use the explicit
# dtype override helper to cast. Actual OMI/legacy key conversion is recorded in
# StableDiffusion3LoraSavePlan. Callers that write through the low-level helper
# must provide names already in the target namespace from that plan.

from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import zeros_device, add_scalar
from serenitymojo.tensor import Tensor

from serenity_trainer.modelLoader.stableDiffusion3.StableDiffusion3ModelLoader import (
    StableDiffusion3LoraConversionPlan,
    stable_diffusion3_lora_conversion_plan,
)
from serenity_trainer.modelSetup.stableDiffusion3LoraTargets import (
    StableDiffusion3LoraTargetSpecs,
    sd3_lora_down_key,
    sd3_lora_up_key,
    sd3_lora_alpha_key,
)


comptime TArc = ArcPointer[Tensor]

comptime SD3_FMT_DIFFUSERS = 0
comptime SD3_FMT_CKPT = 1
comptime SD3_FMT_SAFETENSORS = 2
comptime SD3_FMT_LEGACY_SAFETENSORS = 3
comptime SD3_FMT_COMFY_LORA = 4
comptime SD3_FMT_INTERNAL = 5


struct StableDiffusion3LoraStateDict(Movable):
    var names: List[String]
    var tensors: List[TArc]

    def __init__(out self, var names: List[String], var tensors: List[TArc]):
        self.names = names^
        self.tensors = tensors^


struct StableDiffusion3LoraSavePlan(Movable):
    var output_model_format: Int
    var output_model_destination: String
    var route_name: String
    var target_key_namespace: String
    var writes_safetensors: Bool
    var internal_destination: String
    var has_convert_key_sets: Bool
    var conversion_plan: StableDiffusion3LoraConversionPlan
    var includes_text_encoder_1_lora: Bool
    var includes_text_encoder_2_lora: Bool
    var includes_text_encoder_3_lora: Bool
    var includes_transformer_lora: Bool
    var includes_preloaded_lora_state_dict: Bool
    var can_bundle_additional_embeddings: Bool
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
        self.conversion_plan = stable_diffusion3_lora_conversion_plan()
        self.includes_text_encoder_1_lora = True
        self.includes_text_encoder_2_lora = True
        self.includes_text_encoder_3_lora = True
        self.includes_transformer_lora = True
        self.includes_preloaded_lora_state_dict = True
        self.can_bundle_additional_embeddings = True
        self.preserves_storage_dtype_without_override = True


def stable_diffusion3_lora_saver_has_convert_key_sets() -> Bool:
    return True


def stable_diffusion3_lora_save_plan(
    output_model_format: Int,
    output_model_destination: String,
) raises -> StableDiffusion3LoraSavePlan:
    if output_model_format == SD3_FMT_SAFETENSORS:
        return StableDiffusion3LoraSavePlan(
            output_model_format,
            output_model_destination.copy(),
            String("legacy_safetensors"),
            String("legacy_diffusers"),
            String(),
            True,
        )
    if output_model_format == SD3_FMT_LEGACY_SAFETENSORS:
        return StableDiffusion3LoraSavePlan(
            output_model_format,
            output_model_destination.copy(),
            String("legacy_safetensors"),
            String("legacy_diffusers"),
            String(),
            True,
        )
    if output_model_format == SD3_FMT_INTERNAL:
        return StableDiffusion3LoraSavePlan(
            output_model_format,
            output_model_destination.copy(),
            String("internal_lora"),
            String("omi"),
            output_model_destination + String("/lora/lora.safetensors"),
            True,
        )
    if output_model_format == SD3_FMT_DIFFUSERS:
        raise Error("StableDiffusion3LoRASaver: DIFFUSERS LoRA output is not implemented in Serenity")
    raise Error("StableDiffusion3LoRASaver: unsupported ModelFormat")


def stable_diffusion3_lora_bundle_embedding_keys(placeholder: String) -> List[String]:
    var keys = List[String]()
    keys.append(String("bundle_emb.") + placeholder + String(".clip_l"))
    keys.append(String("bundle_emb.") + placeholder + String(".clip_g"))
    keys.append(String("bundle_emb.") + placeholder + String(".t5"))
    keys.append(String("bundle_emb.") + placeholder + String(".clip_l_out"))
    keys.append(String("bundle_emb.") + placeholder + String(".clip_g_out"))
    keys.append(String("bundle_emb.") + placeholder + String(".t5_out"))
    return keys^


def stable_diffusion3_lora_state_dict_from_raw(
    var names: List[String],
    var tensors: List[TArc],
) raises -> StableDiffusion3LoraStateDict:
    if len(names) != len(tensors):
        raise Error("stable_diffusion3_lora_state_dict_from_raw: names/tensors length mismatch")
    return StableDiffusion3LoraStateDict(names^, tensors^)


def build_stable_diffusion3_lora_state_dict_from_targets(
    targets: StableDiffusion3LoraTargetSpecs,
    rank: Int,
    alpha: Float32,
    ctx: DeviceContext,
    dtype: STDtype = STDtype.BF16,
) raises -> StableDiffusion3LoraStateDict:
    if rank <= 0:
        raise Error("build_stable_diffusion3_lora_state_dict_from_targets: rank must be positive")
    if targets.len() == 0:
        raise Error("build_stable_diffusion3_lora_state_dict_from_targets: no targets")

    var names = List[String]()
    var tensors = List[TArc]()
    for i in range(targets.len()):
        var prefix = targets.prefixes[i]
        var in_features = targets.in_features[i]
        var out_features = targets.out_features[i]

        names.append(sd3_lora_down_key(prefix))
        tensors.append(TArc(_zeros2(rank, in_features, dtype, ctx)))

        names.append(sd3_lora_up_key(prefix))
        tensors.append(TArc(_zeros2(out_features, rank, dtype, ctx)))

        names.append(sd3_lora_alpha_key(prefix))
        tensors.append(TArc(_scalar(alpha, dtype, ctx)))

    return StableDiffusion3LoraStateDict(names^, tensors^)


def save_stable_diffusion3_lora_state_dict(
    var state: StableDiffusion3LoraStateDict,
    output_model_format: Int,
    output_model_destination: String,
    ctx: DeviceContext,
) raises:
    var plan = stable_diffusion3_lora_save_plan(output_model_format, output_model_destination)
    if not plan.writes_safetensors:
        raise Error("save_stable_diffusion3_lora_state_dict: unsupported non-safetensors route")
    if plan.internal_destination.byte_length() > 0:
        save_safetensors(state.names^, state.tensors^, plan.internal_destination, ctx)
    else:
        save_safetensors(state.names^, state.tensors^, output_model_destination, ctx)


def save_stable_diffusion3_lora_state_dict_as_dtype(
    state: StableDiffusion3LoraStateDict,
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
    var cast_state = StableDiffusion3LoraStateDict(names^, tensors^)
    save_stable_diffusion3_lora_state_dict(cast_state^, output_model_format, output_model_destination, ctx)


def _zeros2(rows: Int, cols: Int, dtype: STDtype, ctx: DeviceContext) raises -> Tensor:
    var sh = List[Int]()
    sh.append(rows)
    sh.append(cols)
    return zeros_device(sh^, dtype, ctx)


def _scalar(val: Float32, dtype: STDtype, ctx: DeviceContext) raises -> Tensor:
    # Serenity LoRAModule registers alpha as a scalar buffer; module.to(dtype)
    # casts it with the rest of the LoRA module before state_dict save.
    var sh = List[Int]()
    var z = zeros_device(sh^, dtype, ctx)
    return add_scalar(z, val, ctx)


struct StableDiffusion3LoRASaver(Movable):
    def __init__(out self):
        pass

    def _get_convert_key_sets(self) -> Bool:
        return stable_diffusion3_lora_saver_has_convert_key_sets()

    def conversion_plan(self) -> StableDiffusion3LoraConversionPlan:
        return stable_diffusion3_lora_conversion_plan()

    def save_plan(
        self,
        output_model_format: Int,
        output_model_destination: String,
    ) raises -> StableDiffusion3LoraSavePlan:
        return stable_diffusion3_lora_save_plan(output_model_format, output_model_destination)

    def save(
        self,
        var state: StableDiffusion3LoraStateDict,
        output_model_format: Int,
        output_model_destination: String,
        ctx: DeviceContext,
    ) raises:
        save_stable_diffusion3_lora_state_dict(state^, output_model_format, output_model_destination, ctx)

    def save_as_dtype(
        self,
        state: StableDiffusion3LoraStateDict,
        output_model_format: Int,
        output_model_destination: String,
        dtype: STDtype,
        ctx: DeviceContext,
    ) raises:
        save_stable_diffusion3_lora_state_dict_as_dtype(state, output_model_format, output_model_destination, dtype, ctx)

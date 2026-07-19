# 1:1 surface port of Serenity
#   modules/modelSaver/flux2/Flux2ModelSaver.py
#
# Build-only full-model saver contract. Serenity can save Flux2 as a
# diffusers pipeline, as a converted original transformer safetensors checkpoint,
# or as the internal diffusers directory used for resume. This file records those
# routes and dtype/conversion behavior; it does not write files or claim numeric
# parity.

from serenity_trainer.modelSaver.flux2.Flux2LoRASaver import (
    FLUX2_FMT_DIFFUSERS,
    FLUX2_FMT_INTERNAL,
    FLUX2_FMT_SAFETENSORS,
)


struct Flux2ModelSavePlan(Movable):
    var output_model_format: Int
    var output_model_destination: String
    var route_name: String
    var dtype_override: String
    var saves_diffusers_pipeline: Bool
    var saves_original_transformer_safetensors: Bool
    var saves_internal_diffusers: Bool
    var uses_diffusers_checkpoint_to_original_converter: Bool
    var converter_name: String
    var converter_source: String
    var converter_input_state_dict: String
    var converter_output_key_namespace: String
    var save_pipeline_to_cpu_first: Bool
    var deep_copy_pipeline_when_dtype_override: Bool
    var tokenizer_deepcopy_patched_for_dtype_override: Bool
    var dtype_override_applies_to_diffusers_pipeline: Bool
    var dtype_override_applies_to_safetensors: Bool
    var internal_uses_diffusers_without_dtype_override: Bool
    var creates_destination_directory: Bool
    var creates_safetensors_parent_directory: Bool
    var creates_safetensors_header: Bool
    var makes_tensors_contiguous: Bool
    var safetensors_includes_transformer_state_only: Bool
    var safetensors_includes_vae_state_dict: Bool
    var safetensors_includes_text_encoder_state_dict: Bool
    var preserves_storage_dtype_without_override: Bool
    var supports_diffusers_format: Bool
    var supports_safetensors_format: Bool
    var supports_internal_format: Bool
    var supports_ckpt_format: Bool
    var supports_legacy_safetensors_format: Bool
    var supports_comfy_lora_format: Bool

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
        self.uses_diffusers_checkpoint_to_original_converter = saves_original_transformer_safetensors
        self.converter_name = String("diffusers_checkpoint_to_original")
        self.converter_source = String("modules/model/Flux2Model.py")
        self.converter_input_state_dict = String("model.transformer.state_dict()")
        self.converter_output_key_namespace = String("original Flux2 transformer checkpoint keys")
        self.save_pipeline_to_cpu_first = saves_diffusers_pipeline or saves_internal_diffusers
        self.deep_copy_pipeline_when_dtype_override = (
            saves_diffusers_pipeline and self.dtype_override.byte_length() > 0
        )
        self.tokenizer_deepcopy_patched_for_dtype_override = self.deep_copy_pipeline_when_dtype_override
        self.dtype_override_applies_to_diffusers_pipeline = self.deep_copy_pipeline_when_dtype_override
        self.dtype_override_applies_to_safetensors = (
            saves_original_transformer_safetensors and self.dtype_override.byte_length() > 0
        )
        self.internal_uses_diffusers_without_dtype_override = saves_internal_diffusers
        self.creates_destination_directory = saves_diffusers_pipeline or saves_internal_diffusers
        self.creates_safetensors_parent_directory = saves_original_transformer_safetensors
        self.creates_safetensors_header = saves_original_transformer_safetensors
        self.makes_tensors_contiguous = saves_original_transformer_safetensors
        self.safetensors_includes_transformer_state_only = saves_original_transformer_safetensors
        self.safetensors_includes_vae_state_dict = False
        self.safetensors_includes_text_encoder_state_dict = False
        self.preserves_storage_dtype_without_override = self.dtype_override.byte_length() == 0
        self.supports_diffusers_format = True
        self.supports_safetensors_format = True
        self.supports_internal_format = True
        self.supports_ckpt_format = False
        self.supports_legacy_safetensors_format = False
        self.supports_comfy_lora_format = False


def flux2_model_saver_plan(
    output_model_format: Int,
    output_model_destination: String,
    dtype_override: String,
) raises -> Flux2ModelSavePlan:
    if output_model_format == FLUX2_FMT_DIFFUSERS:
        return Flux2ModelSavePlan(
            output_model_format,
            output_model_destination.copy(),
            String("diffusers_pipeline"),
            dtype_override.copy(),
            True,
            False,
            False,
        )
    if output_model_format == FLUX2_FMT_SAFETENSORS:
        return Flux2ModelSavePlan(
            output_model_format,
            output_model_destination.copy(),
            String("original_transformer_safetensors_checkpoint"),
            dtype_override.copy(),
            False,
            True,
            False,
        )
    if output_model_format == FLUX2_FMT_INTERNAL:
        return Flux2ModelSavePlan(
            output_model_format,
            output_model_destination.copy(),
            String("internal_diffusers"),
            String(),
            False,
            False,
            True,
        )
    raise Error("Flux2ModelSaver: unsupported ModelFormat")


def flux2_model_safetensors_state_sources() -> List[String]:
    var sources = List[String]()
    sources.append(String("model.transformer.state_dict()"))
    sources.append(String("convert(state_dict, diffusers_checkpoint_to_original)"))
    sources.append(String("DtypeModelSaverMixin._convert_state_dict_dtype(state_dict, dtype)"))
    sources.append(String("DtypeModelSaverMixin._convert_state_dict_to_contiguous(state_dict)"))
    sources.append(String("safetensors.torch.save_file(..., _create_safetensors_header(...))"))
    return sources^


struct Flux2ModelSaver(Movable):
    def __init__(out self):
        pass

    def save_plan(
        self,
        output_model_format: Int,
        output_model_destination: String,
        dtype_override: String,
    ) raises -> Flux2ModelSavePlan:
        return flux2_model_saver_plan(
            output_model_format,
            output_model_destination,
            dtype_override,
        )

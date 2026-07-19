# Ideogram4ModelLoader.mojo - trainer-side Ideogram4 loader contract.
#
# Source facts:
#   /home/alex/ai-toolkit/extensions_built_in/diffusion_models/ideogram4/ideogram4.py
#   /home/alex/mojodiffusion/serenitymojo/models/dit/ideogram4_resident.mojo
#   /home/alex/mojodiffusion/serenitymojo/pipeline/ideogram4_generate.mojo
#
# This is a pure-Mojo contract/load-plan layer. It does not import Python, yaml,
# diffusers, transformers, or safetensors runtime objects.

from serenity_trainer.model.Ideogram4Model import Ideogram4ModelContract, ideogram4_model_contract
from serenity_trainer.util.enum.ModelType import (
    MODEL_TYPE_IDEOGRAM_4,
    model_type_is_ideogram_4,
    model_type_str,
)


comptime IDEOGRAM4_FINE_TUNE_MODEL_SPEC = "resources/sd_model_spec/ideogram4.json"
comptime IDEOGRAM4_LORA_MODEL_SPEC = "resources/sd_model_spec/ideogram4-lora.json"
comptime IDEOGRAM4_DEFAULT_LOCAL_ROOT = "/home/alex/.serenity/models/ideogram-4-fp8"
comptime IDEOGRAM4_HF_REPO = "ideogram-ai/ideogram-4-fp8"
comptime IDEOGRAM4_QWEN3_VL_TEXT_ENCODER = "Qwen/Qwen3-VL-8B-Instruct"
comptime IDEOGRAM4_WEIGHT_BASENAME = "diffusion_pytorch_model"
comptime IDEOGRAM4_FP8_SCALE_SUFFIX = ".weight_scale"
comptime IDEOGRAM4_LATENT_NORM_PATH = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/ideogram4_fx_latentnorm.safetensors"


struct Ideogram4ModelNames(Copyable, Movable, ImplicitlyCopyable):
    var base_model: String
    var transformer_model: String
    var unconditional_transformer_model: String
    var text_encoder_model: String
    var vae_model: String
    var latent_norm_model: String
    var lora: String

    def __init__(
        out self,
        var base_model: String,
        var transformer_model: String,
        var unconditional_transformer_model: String,
        var text_encoder_model: String,
        var vae_model: String,
        var latent_norm_model: String,
        var lora: String,
    ):
        self.base_model = base_model^
        self.transformer_model = transformer_model^
        self.unconditional_transformer_model = unconditional_transformer_model^
        self.text_encoder_model = text_encoder_model^
        self.vae_model = vae_model^
        self.latent_norm_model = latent_norm_model^
        self.lora = lora^

    @staticmethod
    def default_values() -> Ideogram4ModelNames:
        return Ideogram4ModelNames(
            String(IDEOGRAM4_DEFAULT_LOCAL_ROOT),
            String(),
            String(),
            String(IDEOGRAM4_QWEN3_VL_TEXT_ENCODER),
            String(),
            String(IDEOGRAM4_LATENT_NORM_PATH),
            String(),
        )

    @staticmethod
    def empty() -> Ideogram4ModelNames:
        return Ideogram4ModelNames(String(), String(), String(), String(), String(), String(), String())


struct Ideogram4RuntimeFlags(Copyable, Movable, ImplicitlyCopyable):
    var quantize: Bool
    var quantize_te: Bool
    var low_vram: Bool
    var qtype_te: String
    var cache_text_embeddings: Bool
    var caption_ext_json: Bool
    var timestep_type: String
    var resident_fp8: Bool

    def __init__(
        out self,
        quantize: Bool,
        quantize_te: Bool,
        low_vram: Bool,
        var qtype_te: String,
        cache_text_embeddings: Bool,
        caption_ext_json: Bool,
        var timestep_type: String,
        resident_fp8: Bool,
    ):
        self.quantize = quantize
        self.quantize_te = quantize_te
        self.low_vram = low_vram
        self.qtype_te = qtype_te^
        self.cache_text_embeddings = cache_text_embeddings
        self.caption_ext_json = caption_ext_json
        self.timestep_type = timestep_type^
        self.resident_fp8 = resident_fp8

    @staticmethod
    def default_values() -> Ideogram4RuntimeFlags:
        return Ideogram4RuntimeFlags(
            True,
            True,
            True,
            String("qfloat8"),
            True,
            True,
            String("linear"),
            True,
        )


struct Ideogram4ModelHandle(Movable):
    var model_type: Int
    var model_spec: String
    var base_loaded: Bool
    var lora_loaded: Bool

    def __init__(out self, model_type: Int):
        self.model_type = model_type
        self.model_spec = String()
        self.base_loaded = False
        self.lora_loaded = False


struct Ideogram4LoadPlan(Movable):
    var model_type: Int
    var model_spec: String
    var base_model: String
    var transformer_model: String
    var unconditional_transformer_model: String
    var text_encoder_model: String
    var vae_model: String
    var latent_norm_model: String
    var lora_model: String
    var transformer_subfolder: String
    var unconditional_transformer_subfolder: String
    var text_encoder_subfolder: String
    var vae_subfolder: String
    var weight_basename: String
    var fp8_scale_suffix: String
    var loader_dequantizes_fp8: Bool
    var low_vram_moves_dequantized_weights_to_cpu: Bool
    var tokenizer_class: String
    var text_encoder_class: String
    var transformer_class: String
    var vae_class: String
    var train_scheduler_class: String
    var train_scheduler_timestep_type: String
    var text_activation_layers: String
    var text_feature_dim: Int
    var model_time_is_one_minus_training_t: Bool
    var model_output_is_negated_for_training_velocity: Bool
    var loss_target_expression: String
    var native_resident_weights_module: String
    var native_generate_module: String
    var native_magic_prompt_module: String
    var native_training_forward_present: Bool
    var native_lora_backward_present: Bool
    var native_lora_backward_slice: String
    var base_loader_invoked: Bool
    var lora_loader_invoked: Bool
    var contract: Ideogram4ModelContract
    var flags: Ideogram4RuntimeFlags

    def __init__(
        out self,
        model_type: Int,
        var model_spec: String,
        model_names: Ideogram4ModelNames,
        flags: Ideogram4RuntimeFlags,
        base_loader_invoked: Bool,
        lora_loader_invoked: Bool,
    ):
        self.model_type = model_type
        self.model_spec = model_spec^
        self.base_model = model_names.base_model.copy()
        self.transformer_model = model_names.transformer_model.copy()
        self.unconditional_transformer_model = model_names.unconditional_transformer_model.copy()
        self.text_encoder_model = model_names.text_encoder_model.copy()
        self.vae_model = model_names.vae_model.copy()
        self.latent_norm_model = model_names.latent_norm_model.copy()
        self.lora_model = model_names.lora.copy()
        self.transformer_subfolder = String("transformer")
        self.unconditional_transformer_subfolder = String("unconditional_transformer")
        self.text_encoder_subfolder = String("text_encoder")
        self.vae_subfolder = String("vae")
        self.weight_basename = String(IDEOGRAM4_WEIGHT_BASENAME)
        self.fp8_scale_suffix = String(IDEOGRAM4_FP8_SCALE_SUFFIX)
        self.loader_dequantizes_fp8 = True
        self.low_vram_moves_dequantized_weights_to_cpu = flags.low_vram
        self.tokenizer_class = String("AutoTokenizer")
        self.text_encoder_class = String("Qwen3-VL-8B-Instruct")
        self.transformer_class = String("Ideogram4Transformer2DModel")
        self.vae_class = String("AutoEncoder")
        self.train_scheduler_class = String("CustomFlowMatchEulerDiscreteScheduler")
        self.train_scheduler_timestep_type = flags.timestep_type.copy()
        self.text_activation_layers = String("0,3,6,9,12,15,18,21,24,27,30,33,35")
        self.text_feature_dim = ideogram4_model_contract().text_feature_dim
        self.model_time_is_one_minus_training_t = True
        self.model_output_is_negated_for_training_velocity = True
        self.loss_target_expression = String("noise - batch.latents")
        self.native_resident_weights_module = String("serenitymojo.models.dit.ideogram4_resident")
        self.native_generate_module = String("serenitymojo.pipeline.ideogram4_generate")
        self.native_magic_prompt_module = String("serenitymojo.pipeline.ideogram4_magic")
        self.native_training_forward_present = True
        self.native_lora_backward_present = True
        self.native_lora_backward_slice = String("transformer.layers.* + transformer.final_layer.linear")
        self.base_loader_invoked = base_loader_invoked
        self.lora_loader_invoked = lora_loader_invoked
        self.contract = ideogram4_model_contract()
        self.flags = flags


def _validate_ideogram4_model_type(caller: String, model_type: Int) raises:
    if not model_type_is_ideogram_4(model_type):
        raise Error(caller + String(": unsupported ModelType ") + model_type_str(model_type))


def ideogram4_default_model_spec_name(model_type: Int) raises -> String:
    _validate_ideogram4_model_type(String("Ideogram4FineTuneModelLoader"), model_type)
    return String(IDEOGRAM4_FINE_TUNE_MODEL_SPEC)


def ideogram4_lora_default_model_spec_name(model_type: Int) raises -> String:
    _validate_ideogram4_model_type(String("Ideogram4LoRAModelLoader"), model_type)
    return String(IDEOGRAM4_LORA_MODEL_SPEC)


def ideogram4_default_model_names() -> Ideogram4ModelNames:
    return Ideogram4ModelNames.default_values()


def ideogram4_default_runtime_flags() -> Ideogram4RuntimeFlags:
    return Ideogram4RuntimeFlags.default_values()


struct Ideogram4ModelLoader(Movable):
    def __init__(out self):
        pass

    def _default_model_spec_name(self, model_type: Int) raises -> String:
        return ideogram4_default_model_spec_name(model_type)

    def load(
        self,
        mut model: Ideogram4ModelHandle,
        model_type: Int,
        model_names: Ideogram4ModelNames,
        flags: Ideogram4RuntimeFlags,
        model_spec: String = String(IDEOGRAM4_FINE_TUNE_MODEL_SPEC),
    ) raises -> Ideogram4LoadPlan:
        _validate_ideogram4_model_type(String("Ideogram4ModelLoader.load"), model_type)
        model.model_type = model_type
        model.model_spec = model_spec.copy()
        model.base_loaded = True
        return Ideogram4LoadPlan(
            model_type,
            model_spec.copy(),
            model_names,
            flags,
            True,
            False,
        )


struct Ideogram4FineTuneModelLoader(Movable):
    def __init__(out self):
        pass

    def _default_model_spec_name(self, model_type: Int) raises -> String:
        return ideogram4_default_model_spec_name(model_type)

    def load(
        self,
        model_type: Int,
        model_names: Ideogram4ModelNames,
        flags: Ideogram4RuntimeFlags,
    ) raises -> Ideogram4LoadPlan:
        var model = Ideogram4ModelHandle(model_type)
        var base_loader = Ideogram4ModelLoader()
        return base_loader.load(
            model,
            model_type,
            model_names,
            flags,
            ideogram4_default_model_spec_name(model_type),
        )


struct Ideogram4LoRAModelLoader(Movable):
    def __init__(out self):
        pass

    def _default_model_spec_name(self, model_type: Int) raises -> String:
        return ideogram4_lora_default_model_spec_name(model_type)

    def load(
        self,
        model_type: Int,
        model_names: Ideogram4ModelNames,
        flags: Ideogram4RuntimeFlags,
    ) raises -> Ideogram4LoadPlan:
        var model = Ideogram4ModelHandle(model_type)
        var base_loader = Ideogram4ModelLoader()
        var plan = base_loader.load(
            model,
            model_type,
            model_names,
            flags,
            ideogram4_lora_default_model_spec_name(model_type),
        )
        plan.lora_loader_invoked = True
        return plan^

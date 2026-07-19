# Flux2ModelLoader.mojo - Flux2/Klein loader source-contract surface.
#
# Serenity references:
#   modules/modelLoader/Flux2ModelLoader.py
#   modules/model/Flux2Model.py
#
# This file intentionally stays lightweight: runtime tensor/safetensors helpers
# live in modelLoader/Flux2RuntimeLoader.mojo. The contract here mirrors loader
# dispatch facts that must stay aligned with Serenity without instantiating
# diffusers, transformers, CUDA, text encoders, VAE, scheduler, or LoRA tensors.

from serenity_trainer.util.enum.ModelType import (
    MODEL_TYPE_FLUX_2,
    model_type_is_flux_2,
    model_type_str,
)


comptime FLUX2_LOAD_AUTO = 0
comptime FLUX2_LOAD_INTERNAL = 1
comptime FLUX2_LOAD_DIFFUSERS = 2
comptime FLUX2_LOAD_SAFETENSORS = 3
comptime FLUX2_LORA_ROUTE_NONE = 0
comptime FLUX2_LORA_ROUTE_MIXIN = 1
comptime FLUX2_DEV_NUM_ATTENTION_HEADS = 48
comptime FLUX2_FINE_TUNE_MODEL_SPEC = "resources/sd_model_spec/flux_2.0.json"
comptime FLUX2_LORA_MODEL_SPEC = "resources/sd_model_spec/flux_2.0-lora.json"


struct Flux2ModelNames(Copyable, Movable, ImplicitlyCopyable):
    var base_model: String
    var transformer_model: String
    var vae_model: String
    var lora: String

    def __init__(
        out self,
        var base_model: String,
        var transformer_model: String,
        var vae_model: String,
        var lora: String,
    ):
        self.base_model = base_model^
        self.transformer_model = transformer_model^
        self.vae_model = vae_model^
        self.lora = lora^

    @staticmethod
    def empty() -> Flux2ModelNames:
        return Flux2ModelNames(String(), String(), String(), String())


struct Flux2WeightDtypes(Copyable, Movable, ImplicitlyCopyable):
    var train_dtype: String
    var fallback_train_dtype: String
    var transformer: String
    var text_encoder: String
    var vae: String
    var lora: String

    def __init__(
        out self,
        var train_dtype: String,
        var fallback_train_dtype: String,
        var transformer: String,
        var text_encoder: String,
        var vae: String,
        var lora: String,
    ):
        self.train_dtype = train_dtype^
        self.fallback_train_dtype = fallback_train_dtype^
        self.transformer = transformer^
        self.text_encoder = text_encoder^
        self.vae = vae^
        self.lora = lora^

    @staticmethod
    def bf16() -> Flux2WeightDtypes:
        return Flux2WeightDtypes(
            String("BF16"),
            String("BF16"),
            String("BF16"),
            String("BF16"),
            String("BF16"),
            String("BF16"),
        )


struct Flux2QuantizationConfig(Copyable, Movable, ImplicitlyCopyable):
    var layer_filter: String
    var layer_filter_preset: String
    var layer_filter_regex: Bool
    var svd_dtype: String
    var svd_rank: Int
    var cache_dir: String

    def __init__(
        out self,
        var layer_filter: String,
        var layer_filter_preset: String,
        layer_filter_regex: Bool,
        var svd_dtype: String,
        svd_rank: Int,
        var cache_dir: String,
    ):
        self.layer_filter = layer_filter^
        self.layer_filter_preset = layer_filter_preset^
        self.layer_filter_regex = layer_filter_regex
        self.svd_dtype = svd_dtype^
        self.svd_rank = svd_rank
        self.cache_dir = cache_dir^

    @staticmethod
    def default_values() -> Flux2QuantizationConfig:
        return Flux2QuantizationConfig(
            String(), String("full"), False, String("NONE"), 16, String()
        )


def _validate_flux2_model_type(caller: String, model_type: Int) raises:
    if not model_type_is_flux_2(model_type):
        raise Error(caller + String(": unsupported ModelType ") + model_type_str(model_type))


def flux2_default_model_spec_name(model_type: Int) raises -> String:
    _validate_flux2_model_type(String("Flux2FineTuneModelLoader"), model_type)
    return String(FLUX2_FINE_TUNE_MODEL_SPEC)


def flux2_lora_default_model_spec_name(model_type: Int) raises -> String:
    _validate_flux2_model_type(String("Flux2LoRAModelLoader"), model_type)
    return String(FLUX2_LORA_MODEL_SPEC)


def flux2_is_dev_num_attention_heads(num_attention_heads: Int) -> Bool:
    return num_attention_heads == FLUX2_DEV_NUM_ATTENTION_HEADS


def flux2_is_klein_num_attention_heads(num_attention_heads: Int) -> Bool:
    return not flux2_is_dev_num_attention_heads(num_attention_heads)


def flux2_tokenizer_class_for_heads(num_attention_heads: Int) -> String:
    if flux2_is_dev_num_attention_heads(num_attention_heads):
        return String("PixtralProcessor.tokenizer")
    return String("Qwen2Tokenizer")


def flux2_text_encoder_class_for_heads(num_attention_heads: Int) -> String:
    if flux2_is_dev_num_attention_heads(num_attention_heads):
        return String("Mistral3ForConditionalGeneration")
    return String("Qwen3ForCausalLM")


def flux2_pipeline_class_for_heads(num_attention_heads: Int) -> String:
    if flux2_is_dev_num_attention_heads(num_attention_heads):
        return String("Flux2Pipeline")
    return String("Flux2KleinPipeline")


def flux2_internal_probe_file() -> String:
    return String("meta.json")


def flux2_load_tries_internal_first() -> Bool:
    return True


def flux2_load_tries_diffusers_second() -> Bool:
    return True


def flux2_load_tries_safetensors_third() -> Bool:
    return True


def flux2_internal_route_delegates_to_diffusers() -> Bool:
    return True


def flux2_single_file_supported() -> Bool:
    return False


def flux2_safetensors_error() -> String:
    return String("Loading of single file Flux2 models not supported. Use the diffusers model instead. Optionally, transformer-only safetensor files can be loaded by overriding the transformer.")


def flux2_tokenizer_subfolder() -> String:
    return String("tokenizer")


def flux2_text_encoder_subfolder() -> String:
    return String("text_encoder")


def flux2_scheduler_subfolder() -> String:
    return String("scheduler")


def flux2_transformer_subfolder() -> String:
    return String("transformer")


def flux2_vae_subfolder() -> String:
    return String("vae")


def flux2_transformer_class() -> String:
    return String("Flux2Transformer2DModel")


def flux2_scheduler_class() -> String:
    return String("FlowMatchEulerDiscreteScheduler")


def flux2_vae_class() -> String:
    return String("AutoencoderKLFlux2")


def flux2_prepares_transformer_submodule_from_base(names: Flux2ModelNames) -> Bool:
    return names.transformer_model.byte_length() == 0


def flux2_prepares_vae_submodule_from_base(names: Flux2ModelNames) -> Bool:
    return names.vae_model.byte_length() == 0


def flux2_prepares_text_encoder_submodule_from_base() -> Bool:
    return True


def flux2_transformer_override_supported() -> Bool:
    return True


def flux2_transformer_override_from_single_file(names: Flux2ModelNames) -> Bool:
    return names.transformer_model.byte_length() > 0


def flux2_transformer_override_config_source() -> String:
    return String("base_model")


def flux2_transformer_override_default_torch_dtype() -> String:
    return String("BF16")


def flux2_transformer_override_gguf_compute_dtype() -> String:
    return String("BF16")


def flux2_transformer_override_avoids_float32_load() -> Bool:
    return True


def flux2_transformer_quantization_supported() -> Bool:
    return True


def flux2_vae_override_supported() -> Bool:
    return True


def flux2_vae_override_from_model_name(names: Flux2ModelNames) -> Bool:
    return names.vae_model.byte_length() > 0


def flux2_text_encoder_uses_fallback_train_dtype() -> Bool:
    return True


def flux2_preserves_storage_dtype_at_boundaries() -> Bool:
    return True


def flux2_klein_relinks_lm_head_to_embed_tokens() -> Bool:
    return True


struct Flux2LoaderWrapperContract(Movable):
    var factory_name: String
    var model_type: Int
    var model_spec: String
    var model_class: String
    var model_loader_class: String
    var lora_loader_class: String
    var embedding_loader_class: String

    def __init__(
        out self,
        var factory_name: String,
        model_type: Int,
        var model_spec: String,
        var model_class: String,
        var model_loader_class: String,
        var lora_loader_class: String,
        var embedding_loader_class: String,
    ):
        self.factory_name = factory_name^
        self.model_type = model_type
        self.model_spec = model_spec^
        self.model_class = model_class^
        self.model_loader_class = model_loader_class^
        self.lora_loader_class = lora_loader_class^
        self.embedding_loader_class = embedding_loader_class^

    def has_lora_loader(self) -> Bool:
        return self.lora_loader_class.byte_length() > 0

    def has_embedding_loader(self) -> Bool:
        return self.embedding_loader_class.byte_length() > 0


def flux2_lora_loader_contract() -> Flux2LoaderWrapperContract:
    return Flux2LoaderWrapperContract(
        String("make_lora_model_loader"),
        MODEL_TYPE_FLUX_2,
        String(FLUX2_LORA_MODEL_SPEC),
        String("Flux2Model"),
        String("Flux2ModelLoader"),
        String("Flux2LoRALoader"),
        String(),
    )


def flux2_fine_tune_loader_contract() -> Flux2LoaderWrapperContract:
    return Flux2LoaderWrapperContract(
        String("make_fine_tune_model_loader"),
        MODEL_TYPE_FLUX_2,
        String(FLUX2_FINE_TUNE_MODEL_SPEC),
        String("Flux2Model"),
        String("Flux2ModelLoader"),
        String(),
        String(),
    )


struct Flux2ModelLoader(Movable):
    def __init__(out self):
        pass

    def route(self) -> Int:
        return FLUX2_LOAD_AUTO


struct Flux2FineTuneModelLoader(Movable):
    def __init__(out self):
        pass

    def _default_model_spec_name(self, model_type: Int) raises -> String:
        return flux2_default_model_spec_name(model_type)

    def contract(self) -> Flux2LoaderWrapperContract:
        return flux2_fine_tune_loader_contract()


struct Flux2LoraLoadPlan(Movable):
    var route: Int
    var delegates_to_lora_mixin: Bool
    var has_convert_key_sets: Bool
    var convert_key_sets_name: String

    def __init__(out self, has_lora_model: Bool):
        if has_lora_model:
            self.route = FLUX2_LORA_ROUTE_MIXIN
        else:
            self.route = FLUX2_LORA_ROUTE_NONE
        self.delegates_to_lora_mixin = True
        self.has_convert_key_sets = False
        self.convert_key_sets_name = String("None")


def flux2_lora_loader_has_convert_key_sets() -> Bool:
    return False


struct Flux2LoRALoader(Movable):
    def __init__(out self):
        pass

    def _get_convert_key_sets(self) -> Bool:
        return flux2_lora_loader_has_convert_key_sets()

    def load(self, model_names: Flux2ModelNames) -> Flux2LoraLoadPlan:
        return Flux2LoraLoadPlan(model_names.lora.byte_length() > 0)


struct Flux2LoRAModelLoader(Movable):
    def __init__(out self):
        pass

    def _default_model_spec_name(self, model_type: Int) raises -> String:
        return flux2_lora_default_model_spec_name(model_type)

    def contract(self) -> Flux2LoaderWrapperContract:
        return flux2_lora_loader_contract()

    def lora_loader_invoked(self, model_names: Flux2ModelNames) -> Bool:
        return model_names.lora.byte_length() > 0

    def embedding_loader_invoked(self) -> Bool:
        return False

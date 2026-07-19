# 1:1 surface port of Serenity
#   modules/modelLoader/chroma/ChromaModelLoader.py
#
# Build-only contract. Serenity instantiates T5Tokenizer,
# FlowMatchEulerDiscreteScheduler, T5EncoderModel, AutoencoderKL, and
# ChromaTransformer2DModel here. This records route order, component classes,
# override behavior, and dtype boundaries without creating runtime objects.
#
# Dtype: Serenity avoids loading transformer overrides in F32 by using BF16
# when weight_dtypes.transformer.torch_dtype() is None, and GGUF compute dtype
# is BF16. This surface never upcasts persistent tensors.

from serenity_trainer.util.enum.ModelType import (
    MODEL_TYPE_CHROMA_1,
    model_type_is_chroma,
    model_type_str,
)


comptime CHROMA_LOAD_AUTO = 0
comptime CHROMA_LOAD_INTERNAL = 1
comptime CHROMA_LOAD_DIFFUSERS = 2
comptime CHROMA_LOAD_SAFETENSORS = 3


struct ChromaEmbeddingName(Copyable, Movable, ImplicitlyCopyable):
    var uuid: String
    var model_name: String

    def __init__(out self, var uuid: String, var model_name: String):
        self.uuid = uuid^
        self.model_name = model_name^

    @staticmethod
    def empty() -> ChromaEmbeddingName:
        return ChromaEmbeddingName(String(), String())


struct ChromaModelNames(Copyable, Movable, ImplicitlyCopyable):
    var base_model: String
    var transformer_model: String
    var vae_model: String
    var lora: String
    var embedding: ChromaEmbeddingName

    def __init__(
        out self,
        var base_model: String,
        var transformer_model: String,
        var vae_model: String,
        var lora: String,
        var embedding: ChromaEmbeddingName,
    ):
        self.base_model = base_model^
        self.transformer_model = transformer_model^
        self.vae_model = vae_model^
        self.lora = lora^
        self.embedding = embedding^

    @staticmethod
    def empty() -> ChromaModelNames:
        return ChromaModelNames(
            String(), String(), String(), String(), ChromaEmbeddingName.empty()
        )


struct ChromaWeightDtypes(Copyable, Movable, ImplicitlyCopyable):
    var train_dtype: String
    var fallback_train_dtype: String
    var transformer: String
    var text_encoder: String
    var vae: String
    var lora: String
    var embedding: String

    def __init__(
        out self,
        var train_dtype: String,
        var fallback_train_dtype: String,
        var transformer: String,
        var text_encoder: String,
        var vae: String,
        var lora: String,
        var embedding: String,
    ):
        self.train_dtype = train_dtype^
        self.fallback_train_dtype = fallback_train_dtype^
        self.transformer = transformer^
        self.text_encoder = text_encoder^
        self.vae = vae^
        self.lora = lora^
        self.embedding = embedding^

    @staticmethod
    def bf16() -> ChromaWeightDtypes:
        return ChromaWeightDtypes(
            String("BF16"),
            String("BF16"),
            String("BF16"),
            String("BF16"),
            String("BF16"),
            String("BF16"),
            String("BF16"),
        )


struct ChromaQuantizationConfig(Copyable, Movable, ImplicitlyCopyable):
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
    def default_values() -> ChromaQuantizationConfig:
        return ChromaQuantizationConfig(
            String(), String("full"), False, String("NONE"), 16, String()
        )


struct ChromaModelHandle(Movable):
    var model_type: Int
    var model_spec: String
    var base_loaded: Bool
    var lora_loaded: Bool
    var embedding_loaded: Bool

    def __init__(out self, model_type: Int):
        self.model_type = model_type
        self.model_spec = String()
        self.base_loaded = False
        self.lora_loaded = False
        self.embedding_loaded = False


struct ChromaLoadPlan(Movable):
    var route: Int
    var model_type: Int
    var model_spec: String
    var generic_factory: String
    var base_model: String
    var transformer_model: String
    var vae_model: String
    var lora_model: String
    var internal_probe_file: String
    var tries_internal_first: Bool
    var tries_diffusers_second: Bool
    var tries_safetensors_third: Bool
    var single_file_supported: Bool
    var safetensors_error: String
    var tokenizer_subfolder: String
    var scheduler_subfolder: String
    var text_encoder_subfolder: String
    var vae_subfolder: String
    var transformer_subfolder: String
    var tokenizer_class: String
    var scheduler_class: String
    var text_encoder_class: String
    var vae_class: String
    var transformer_class: String
    var prepares_transformer_submodule_from_base: Bool
    var prepares_vae_submodule_from_base: Bool
    var prepares_text_encoder_submodule_from_base: Bool
    var transformer_override_supported: Bool
    var transformer_override_from_single_file: Bool
    var transformer_override_default_torch_dtype: String
    var transformer_override_gguf_compute_dtype: String
    var transformer_override_avoids_float32_load: Bool
    var vae_override_supported: Bool
    var text_encoder_storage_dtype: String
    var text_encoder_uses_fallback_train_dtype: Bool
    var transformer_storage_dtype: String
    var vae_storage_dtype: String
    var train_dtype: String
    var fallback_train_dtype: String
    var transformer_quantization_supported: Bool
    var base_loader_invoked: Bool
    var lora_loader_invoked: Bool
    var embedding_loader_invoked: Bool
    var preserves_storage_dtype_at_boundaries: Bool

    def __init__(
        out self,
        route: Int,
        model_type: Int,
        var model_spec: String,
        var generic_factory: String,
        var names: ChromaModelNames,
        var dtypes: ChromaWeightDtypes,
        base_loader_invoked: Bool,
        lora_loader_invoked: Bool,
        embedding_loader_invoked: Bool,
    ):
        self.route = route
        self.model_type = model_type
        self.model_spec = model_spec^
        self.generic_factory = generic_factory^
        self.base_model = names.base_model.copy()
        self.transformer_model = names.transformer_model.copy()
        self.vae_model = names.vae_model.copy()
        self.lora_model = names.lora.copy()
        self.internal_probe_file = String("meta.json")
        self.tries_internal_first = True
        self.tries_diffusers_second = True
        self.tries_safetensors_third = True
        self.single_file_supported = False
        self.safetensors_error = String("Loading of single file Chroma models not supported. Use the diffusers model instead. Optionally, transformer-only safetensor files can be loaded by overriding the transformer.")
        self.tokenizer_subfolder = String("tokenizer")
        self.scheduler_subfolder = String("scheduler")
        self.text_encoder_subfolder = String("text_encoder")
        self.vae_subfolder = String("vae")
        self.transformer_subfolder = String("transformer")
        self.tokenizer_class = String("T5Tokenizer")
        self.scheduler_class = String("FlowMatchEulerDiscreteScheduler")
        self.text_encoder_class = String("T5EncoderModel")
        self.vae_class = String("AutoencoderKL")
        self.transformer_class = String("ChromaTransformer2DModel")
        self.prepares_transformer_submodule_from_base = self.transformer_model == String()
        self.prepares_vae_submodule_from_base = self.vae_model == String()
        self.prepares_text_encoder_submodule_from_base = True
        self.transformer_override_supported = True
        self.transformer_override_from_single_file = self.transformer_model != String()
        self.transformer_override_default_torch_dtype = String("BF16")
        self.transformer_override_gguf_compute_dtype = String("BF16")
        self.transformer_override_avoids_float32_load = True
        self.vae_override_supported = True
        self.text_encoder_storage_dtype = dtypes.text_encoder.copy()
        self.text_encoder_uses_fallback_train_dtype = True
        self.transformer_storage_dtype = dtypes.transformer.copy()
        self.vae_storage_dtype = dtypes.vae.copy()
        self.train_dtype = dtypes.train_dtype.copy()
        self.fallback_train_dtype = dtypes.fallback_train_dtype.copy()
        self.transformer_quantization_supported = True
        self.base_loader_invoked = base_loader_invoked
        self.lora_loader_invoked = lora_loader_invoked
        self.embedding_loader_invoked = embedding_loader_invoked
        self.preserves_storage_dtype_at_boundaries = True


def chroma_base_load_plan(
    model_type: Int,
    names: ChromaModelNames,
    dtypes: ChromaWeightDtypes,
    quantization: ChromaQuantizationConfig,
    model_spec: String,
    generic_factory: String,
    lora_loader_invoked: Bool,
    embedding_loader_invoked: Bool,
) raises -> ChromaLoadPlan:
    if not model_type_is_chroma(model_type):
        raise Error(String("ChromaModelLoader.load: unsupported ModelType ") + model_type_str(model_type))
    _ = quantization
    return ChromaLoadPlan(
        CHROMA_LOAD_AUTO,
        model_type,
        model_spec,
        generic_factory,
        names,
        dtypes,
        True,
        lora_loader_invoked,
        embedding_loader_invoked,
    )


struct ChromaModelLoader(Movable):
    def __init__(out self):
        pass

    def load(
        self,
        model_type: Int,
        names: ChromaModelNames,
        dtypes: ChromaWeightDtypes,
        quantization: ChromaQuantizationConfig,
    ) raises -> ChromaLoadPlan:
        return chroma_base_load_plan(
            model_type,
            names,
            dtypes,
            quantization,
            String("resources/sd_model_spec/chroma.json"),
            String("ChromaModelLoader"),
            False,
            False,
        )

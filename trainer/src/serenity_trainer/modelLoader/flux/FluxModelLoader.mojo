# 1:1 surface port of Serenity
#   modules/modelLoader/flux/FluxModelLoader.py
#
# Build-only FLUX.1 loader support. Serenity constructs HF/diffusers runtime
# objects here; this Mojo file records the load routes, component classes,
# subfolders, LoRA conversion surface, embedding keys, and dtype boundaries.
# It does not instantiate FluxPipeline, FluxTransformer2DModel, CLIP, T5, or VAE
# modules.
#
# Dtype contract: component storage dtypes are recorded at tensor boundaries.
# Serenity uses fallback_train_dtype for the T5 text encoder load/compute path
# and BF16 as the transformer single-file default when no explicit dtype exists.
# This surface does not upcast persistent tensors.

from serenity_trainer.util.enum.ModelType import (
    MODEL_TYPE_FLUX_DEV_1,
    MODEL_TYPE_FLUX_FILL_DEV_1,
    model_type_has_conditioning_image_input,
    model_type_is_flux_1,
    model_type_str,
)


comptime FLUX_LOAD_AUTO = 0
comptime FLUX_LOAD_INTERNAL = 1
comptime FLUX_LOAD_DIFFUSERS = 2
comptime FLUX_LOAD_SAFETENSORS = 3

comptime FLUX_LORA_ROUTE_NONE = 0
comptime FLUX_LORA_ROUTE_INTERNAL = 1
comptime FLUX_LORA_ROUTE_CKPT = 2
comptime FLUX_LORA_ROUTE_SAFETENSORS = 3


struct FluxEmbeddingName(Movable):
    var uuid: String
    var model_name: String

    def __init__(out self, var uuid: String, var model_name: String):
        self.uuid = uuid^
        self.model_name = model_name^

    @staticmethod
    def empty() -> FluxEmbeddingName:
        return FluxEmbeddingName(String(), String())


struct FluxModelNames(Movable):
    var base_model: String
    var transformer_model: String
    var vae_model: String
    var lora: String
    var embedding: FluxEmbeddingName
    var include_text_encoder_1: Bool
    var include_text_encoder_2: Bool

    def __init__(
        out self,
        var base_model: String,
        var transformer_model: String,
        var vae_model: String,
        var lora: String,
        var embedding: FluxEmbeddingName,
        include_text_encoder_1: Bool = True,
        include_text_encoder_2: Bool = True,
    ):
        self.base_model = base_model^
        self.transformer_model = transformer_model^
        self.vae_model = vae_model^
        self.lora = lora^
        self.embedding = embedding^
        self.include_text_encoder_1 = include_text_encoder_1
        self.include_text_encoder_2 = include_text_encoder_2

    @staticmethod
    def empty() -> FluxModelNames:
        return FluxModelNames(
            String(),
            String(),
            String(),
            String(),
            FluxEmbeddingName.empty(),
            True,
            True,
        )


struct FluxWeightDtypes(Movable):
    var train_dtype: String
    var fallback_train_dtype: String
    var transformer: String
    var text_encoder: String
    var text_encoder_2: String
    var vae: String
    var lora: String
    var embedding: String

    def __init__(
        out self,
        var train_dtype: String,
        var fallback_train_dtype: String,
        var transformer: String,
        var text_encoder: String,
        var text_encoder_2: String,
        var vae: String,
        var lora: String,
        var embedding: String,
    ):
        self.train_dtype = train_dtype^
        self.fallback_train_dtype = fallback_train_dtype^
        self.transformer = transformer^
        self.text_encoder = text_encoder^
        self.text_encoder_2 = text_encoder_2^
        self.vae = vae^
        self.lora = lora^
        self.embedding = embedding^

    @staticmethod
    def bf16() -> FluxWeightDtypes:
        return FluxWeightDtypes(
            String("BF16"),
            String("BF16"),
            String("BF16"),
            String("BF16"),
            String("BF16"),
            String("BF16"),
            String("BF16"),
            String("BF16"),
        )


struct FluxQuantizationConfig(Movable):
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
    def default_values() -> FluxQuantizationConfig:
        return FluxQuantizationConfig(
            String(), String("full"), False, String("NONE"), 16, String()
        )


struct FluxModelHandle(Movable):
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


struct FluxLoadPlan(Movable):
    var route: Int
    var model_type: Int
    var model_spec: String
    var base_model: String
    var transformer_model: String
    var vae_model: String
    var lora_model: String
    var internal_probe_file: String
    var internal_route_delegates_to_diffusers: Bool
    var tries_internal_first: Bool
    var tries_diffusers_second: Bool
    var tries_safetensors_last: Bool
    var single_file_supported: Bool
    var single_file_pipeline_class: String
    var single_file_tokenizer_2_repo: String
    var single_file_replaces_tokenizer_2_when_included: Bool
    var tokenizer_1_subfolder: String
    var tokenizer_2_subfolder: String
    var scheduler_subfolder: String
    var text_encoder_1_subfolder: String
    var text_encoder_2_subfolder: String
    var vae_subfolder: String
    var transformer_subfolder: String
    var tokenizer_1_class: String
    var tokenizer_2_class: String
    var scheduler_class: String
    var text_encoder_1_class: String
    var text_encoder_2_class: String
    var vae_class: String
    var transformer_class: String
    var include_text_encoder_1: Bool
    var include_text_encoder_2: Bool
    var prepare_transformer_submodule_from_base: Bool
    var prepare_vae_submodule_from_base: Bool
    var prepare_text_encoder_1_submodule_from_base: Bool
    var prepare_text_encoder_2_submodule_from_base: Bool
    var transformer_override_supported: Bool
    var transformer_override_from_single_file: Bool
    var transformer_override_default_torch_dtype: String
    var transformer_override_gguf_compute_dtype: String
    var transformer_override_avoids_float32_load: Bool
    var vae_override_supported: Bool
    var text_encoder_1_storage_dtype: String
    var text_encoder_2_storage_dtype: String
    var text_encoder_2_uses_fallback_train_dtype: Bool
    var transformer_storage_dtype: String
    var vae_storage_dtype: String
    var train_dtype: String
    var fallback_train_dtype: String
    var transformer_quantization_supported: Bool
    var base_loader_invoked: Bool
    var lora_loader_invoked: Bool
    var embedding_loader_invoked: Bool
    var safetensors_pipeline_may_omit_text_encoder_1: Bool
    var safetensors_pipeline_may_omit_text_encoder_2: Bool
    var preserves_storage_dtype_at_boundaries: Bool

    def __init__(
        out self,
        route: Int,
        model_type: Int,
        var model_spec: String,
        names: FluxModelNames,
        dtypes: FluxWeightDtypes,
        base_loader_invoked: Bool,
        lora_loader_invoked: Bool,
        embedding_loader_invoked: Bool,
    ):
        self.route = route
        self.model_type = model_type
        self.model_spec = model_spec^
        self.base_model = names.base_model.copy()
        self.transformer_model = names.transformer_model.copy()
        self.vae_model = names.vae_model.copy()
        self.lora_model = names.lora.copy()
        self.internal_probe_file = String("meta.json")
        self.internal_route_delegates_to_diffusers = True
        self.tries_internal_first = True
        self.tries_diffusers_second = True
        self.tries_safetensors_last = True
        self.single_file_supported = True
        self.single_file_pipeline_class = String("FluxPipeline.from_single_file")
        self.single_file_tokenizer_2_repo = String("black-forest-labs/FLUX.1-dev")
        self.single_file_replaces_tokenizer_2_when_included = names.include_text_encoder_2
        self.tokenizer_1_subfolder = String("tokenizer")
        self.tokenizer_2_subfolder = String("tokenizer_2")
        self.scheduler_subfolder = String("scheduler")
        self.text_encoder_1_subfolder = String("text_encoder")
        self.text_encoder_2_subfolder = String("text_encoder_2")
        self.vae_subfolder = String("vae")
        self.transformer_subfolder = String("transformer")
        self.tokenizer_1_class = String("CLIPTokenizer")
        self.tokenizer_2_class = String("T5Tokenizer")
        self.scheduler_class = String("FlowMatchEulerDiscreteScheduler")
        self.text_encoder_1_class = String("CLIPTextModel")
        self.text_encoder_2_class = String("T5EncoderModel")
        self.vae_class = String("AutoencoderKL")
        self.transformer_class = String("FluxTransformer2DModel")
        self.include_text_encoder_1 = names.include_text_encoder_1
        self.include_text_encoder_2 = names.include_text_encoder_2
        self.prepare_transformer_submodule_from_base = names.transformer_model.byte_length() == 0
        self.prepare_vae_submodule_from_base = names.vae_model.byte_length() == 0
        self.prepare_text_encoder_1_submodule_from_base = names.include_text_encoder_1
        self.prepare_text_encoder_2_submodule_from_base = names.include_text_encoder_2
        self.transformer_override_supported = True
        self.transformer_override_from_single_file = names.transformer_model.byte_length() > 0
        self.transformer_override_default_torch_dtype = String("BF16")
        self.transformer_override_gguf_compute_dtype = String("BF16")
        self.transformer_override_avoids_float32_load = True
        self.vae_override_supported = True
        self.text_encoder_1_storage_dtype = dtypes.text_encoder.copy()
        self.text_encoder_2_storage_dtype = dtypes.text_encoder_2.copy()
        self.text_encoder_2_uses_fallback_train_dtype = True
        self.transformer_storage_dtype = dtypes.transformer.copy()
        self.vae_storage_dtype = dtypes.vae.copy()
        self.train_dtype = dtypes.train_dtype.copy()
        self.fallback_train_dtype = dtypes.fallback_train_dtype.copy()
        self.transformer_quantization_supported = True
        self.base_loader_invoked = base_loader_invoked
        self.lora_loader_invoked = lora_loader_invoked
        self.embedding_loader_invoked = embedding_loader_invoked
        self.safetensors_pipeline_may_omit_text_encoder_1 = True
        self.safetensors_pipeline_may_omit_text_encoder_2 = True
        self.preserves_storage_dtype_at_boundaries = True


struct FluxLoraConversionPlan(Movable):
    var has_convert_key_sets: Bool
    var source_namespaces: String
    var load_target_namespace: String
    var safetensors_save_target_namespace: String
    var legacy_save_target_namespace: String
    var internal_save_target_namespace: String
    var root_bundle_embedding_prefix: String
    var transformer_omi_prefix: String
    var transformer_diffusers_prefix: String
    var clip_l_omi_prefix: String
    var clip_l_diffusers_prefix: String
    var t5_omi_prefix: String
    var t5_diffusers_prefix: String
    var range_upper_bound: Int
    var transformer_root_rule_count: Int
    var double_block_rule_count: Int
    var single_block_rule_count: Int
    var has_qkv_split_rules: Bool
    var has_swap_chunks_rules: Bool
    var has_filter_is_last_rules: Bool

    def __init__(out self):
        self.has_convert_key_sets = True
        self.source_namespaces = String("omi,diffusers,legacy_diffusers")
        self.load_target_namespace = String("diffusers")
        self.safetensors_save_target_namespace = String("legacy_diffusers")
        self.legacy_save_target_namespace = String("legacy_diffusers")
        self.internal_save_target_namespace = String("omi")
        self.root_bundle_embedding_prefix = String("bundle_emb")
        self.transformer_omi_prefix = String("transformer")
        self.transformer_diffusers_prefix = String("lora_transformer")
        self.clip_l_omi_prefix = String("clip_l")
        self.clip_l_diffusers_prefix = String("lora_te1")
        self.t5_omi_prefix = String("t5")
        self.t5_diffusers_prefix = String("lora_te2")
        self.range_upper_bound = 100
        self.transformer_root_rule_count = 10
        self.double_block_rule_count = 14
        self.single_block_rule_count = 6
        self.has_qkv_split_rules = True
        self.has_swap_chunks_rules = True
        self.has_filter_is_last_rules = False


struct FluxLoraLoadPlan(Movable):
    var route: Int
    var lora_model: String
    var internal_probe_file: String
    var internal_safetensors_path: String
    var loads_ckpt_when_extension_matches: Bool
    var has_convert_key_sets: Bool
    var converted_target_namespace: String
    var preserves_tensor_storage_dtype: Bool

    def __init__(out self, var lora_model: String):
        self.lora_model = lora_model^
        self.route = FLUX_LORA_ROUTE_SAFETENSORS
        if self.lora_model.byte_length() == 0:
            self.route = FLUX_LORA_ROUTE_NONE
        self.internal_probe_file = String("meta.json")
        self.internal_safetensors_path = String("lora/lora.safetensors")
        self.loads_ckpt_when_extension_matches = True
        self.has_convert_key_sets = True
        self.converted_target_namespace = String("diffusers")
        self.preserves_tensor_storage_dtype = True


struct FluxEmbeddingLoadPlan(Movable):
    var directory: String
    var has_embedding_loader: Bool
    var internal_probe_file: String
    var internal_embedding_path_template: String
    var fallback_to_embedding_model_name: Bool
    var key_clip_l: String
    var key_t5: String
    var key_clip_l_out: String
    var key_t5_out: String
    var preserves_tensor_storage_dtype: Bool

    def __init__(out self, var directory: String):
        self.directory = directory^
        self.has_embedding_loader = True
        self.internal_probe_file = String("meta.json")
        self.internal_embedding_path_template = String("embeddings/{embedding_uuid}.safetensors")
        self.fallback_to_embedding_model_name = True
        self.key_clip_l = String("clip_l")
        self.key_t5 = String("t5")
        self.key_clip_l_out = String("clip_l_out")
        self.key_t5_out = String("t5_out")
        self.preserves_tensor_storage_dtype = True


def flux_default_model_spec_name(model_type: Int) raises -> String:
    if model_type == MODEL_TYPE_FLUX_DEV_1:
        return String("resources/sd_model_spec/flux_dev_1.0.json")
    if model_type == MODEL_TYPE_FLUX_FILL_DEV_1:
        return String("resources/sd_model_spec/flux_dev_fill_1.0.json")
    raise Error(String("FluxFineTuneModelLoader: unsupported ModelType ") + model_type_str(model_type))


def flux_lora_default_model_spec_name(model_type: Int) raises -> String:
    if model_type == MODEL_TYPE_FLUX_DEV_1:
        return String("resources/sd_model_spec/flux_dev_1.0-lora.json")
    if model_type == MODEL_TYPE_FLUX_FILL_DEV_1:
        return String("resources/sd_model_spec/flux_dev_fill_1.0-lora.json")
    raise Error(String("FluxLoRAModelLoader: unsupported ModelType ") + model_type_str(model_type))


def flux_embedding_default_model_spec_name(model_type: Int) raises -> String:
    if model_type == MODEL_TYPE_FLUX_DEV_1:
        return String("resources/sd_model_spec/flux_dev_1.0-embedding.json")
    if model_type == MODEL_TYPE_FLUX_FILL_DEV_1:
        return String("resources/sd_model_spec/flux_dev_fill_1.0-embedding.json")
    raise Error(String("FluxEmbeddingModelLoader: unsupported ModelType ") + model_type_str(model_type))


def flux_lora_loader_has_convert_key_sets() -> Bool:
    return True


def flux_lora_conversion_plan() -> FluxLoraConversionPlan:
    return FluxLoraConversionPlan()


def _validate_flux1_model_type(caller: String, model_type: Int) raises:
    if not model_type_is_flux_1(model_type):
        raise Error(caller + String(": unsupported ModelType ") + model_type_str(model_type))


struct FluxModelLoader(Movable):
    def __init__(out self):
        pass

    def _default_model_spec_name(self, model_type: Int) raises -> String:
        return flux_default_model_spec_name(model_type)

    def load(
        self,
        mut model: FluxModelHandle,
        model_type: Int,
        model_names: FluxModelNames,
        weight_dtypes: FluxWeightDtypes,
        quantization: FluxQuantizationConfig,
    ) raises -> FluxLoadPlan:
        _validate_flux1_model_type(String("FluxModelLoader.load"), model_type)
        _ = quantization
        model.model_type = model_type
        model.model_spec = flux_default_model_spec_name(model_type)
        model.base_loaded = True
        return FluxLoadPlan(
            FLUX_LOAD_AUTO,
            model_type,
            model.model_spec.copy(),
            model_names,
            weight_dtypes,
            True,
            False,
            False,
        )

    def load_safetensors(
        self,
        mut model: FluxModelHandle,
        model_type: Int,
        model_names: FluxModelNames,
        weight_dtypes: FluxWeightDtypes,
        quantization: FluxQuantizationConfig,
    ) raises -> FluxLoadPlan:
        _validate_flux1_model_type(String("FluxModelLoader.load_safetensors"), model_type)
        _ = quantization
        model.model_type = model_type
        model.model_spec = flux_default_model_spec_name(model_type)
        model.base_loaded = True
        return FluxLoadPlan(
            FLUX_LOAD_SAFETENSORS,
            model_type,
            model.model_spec.copy(),
            model_names,
            weight_dtypes,
            True,
            False,
            False,
        )


struct FluxLoRALoader(Movable):
    def __init__(out self):
        pass

    def _get_convert_key_sets(self) -> Bool:
        return flux_lora_loader_has_convert_key_sets()

    def conversion_plan(self) -> FluxLoraConversionPlan:
        return flux_lora_conversion_plan()

    def load(
        self,
        mut model: FluxModelHandle,
        model_names: FluxModelNames,
    ) -> FluxLoraLoadPlan:
        var plan = FluxLoraLoadPlan(model_names.lora.copy())
        model.lora_loaded = plan.route != FLUX_LORA_ROUTE_NONE
        return plan^


struct FluxEmbeddingLoader(Movable):
    def __init__(out self):
        pass

    def load(
        self,
        mut model: FluxModelHandle,
        directory: String,
        model_names: FluxModelNames,
    ) -> FluxEmbeddingLoadPlan:
        _ = model_names
        model.embedding_loaded = directory.byte_length() > 0
        return FluxEmbeddingLoadPlan(directory.copy())


struct FluxFineTuneModelLoader(Movable):
    def __init__(out self):
        pass

    def _default_model_spec_name(self, model_type: Int) raises -> String:
        return flux_default_model_spec_name(model_type)

    def load(
        self,
        model_type: Int,
        model_names: FluxModelNames,
        weight_dtypes: FluxWeightDtypes,
        quantization: FluxQuantizationConfig,
    ) raises -> FluxLoadPlan:
        _validate_flux1_model_type(String("FluxFineTuneModelLoader.load"), model_type)
        var model = FluxModelHandle(model_type)
        model.model_spec = flux_default_model_spec_name(model_type)
        var base_loader = FluxModelLoader()
        var plan = base_loader.load(model, model_type, model_names, weight_dtypes, quantization)
        var embedding_loader = FluxEmbeddingLoader()
        _ = embedding_loader.load(model, model_names.base_model.copy(), model_names)
        plan.model_spec = flux_default_model_spec_name(model_type)
        plan.embedding_loader_invoked = True
        return plan^


struct FluxLoRAModelLoader(Movable):
    def __init__(out self):
        pass

    def _default_model_spec_name(self, model_type: Int) raises -> String:
        return flux_lora_default_model_spec_name(model_type)

    def load(
        self,
        model_type: Int,
        model_names: FluxModelNames,
        weight_dtypes: FluxWeightDtypes,
        quantization: FluxQuantizationConfig,
    ) raises -> FluxLoadPlan:
        _validate_flux1_model_type(String("FluxLoRAModelLoader.load"), model_type)
        var model = FluxModelHandle(model_type)
        model.model_spec = flux_lora_default_model_spec_name(model_type)

        var base_loader_invoked = False
        var plan = FluxLoadPlan(
            FLUX_LOAD_AUTO,
            model_type,
            model.model_spec.copy(),
            model_names,
            weight_dtypes,
            False,
            False,
            False,
        )

        if model_names.base_model.byte_length() > 0:
            var base_loader = FluxModelLoader()
            plan = base_loader.load(model, model_type, model_names, weight_dtypes, quantization)
            base_loader_invoked = True

        var lora_loader = FluxLoRALoader()
        _ = lora_loader.load(model, model_names)
        var embedding_loader = FluxEmbeddingLoader()
        _ = embedding_loader.load(model, model_names.lora.copy(), model_names)

        plan.model_spec = flux_lora_default_model_spec_name(model_type)
        plan.base_loader_invoked = base_loader_invoked
        plan.lora_loader_invoked = True
        plan.embedding_loader_invoked = True
        plan.lora_model = model_names.lora.copy()
        return plan^


struct FluxEmbeddingModelLoader(Movable):
    def __init__(out self):
        pass

    def _default_model_spec_name(self, model_type: Int) raises -> String:
        return flux_embedding_default_model_spec_name(model_type)

    def load(
        self,
        model_type: Int,
        model_names: FluxModelNames,
        weight_dtypes: FluxWeightDtypes,
        quantization: FluxQuantizationConfig,
    ) raises -> FluxLoadPlan:
        _validate_flux1_model_type(String("FluxEmbeddingModelLoader.load"), model_type)
        var model = FluxModelHandle(model_type)
        model.model_spec = flux_embedding_default_model_spec_name(model_type)

        var base_loader_invoked = False
        var plan = FluxLoadPlan(
            FLUX_LOAD_AUTO,
            model_type,
            model.model_spec.copy(),
            model_names,
            weight_dtypes,
            False,
            False,
            False,
        )

        if model_names.base_model.byte_length() > 0:
            var base_loader = FluxModelLoader()
            plan = base_loader.load(model, model_type, model_names, weight_dtypes, quantization)
            base_loader_invoked = True

        var embedding_loader = FluxEmbeddingLoader()
        _ = embedding_loader.load(model, model_names.embedding.model_name.copy(), model_names)

        plan.model_spec = flux_embedding_default_model_spec_name(model_type)
        plan.base_loader_invoked = base_loader_invoked
        plan.embedding_loader_invoked = True
        return plan^

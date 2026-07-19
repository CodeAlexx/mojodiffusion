# 1:1 surface port of Serenity
#   modules/modelLoader/stableDiffusionXL/StableDiffusionXLModelLoader.py
#
# Build-only SDXL loader support. This records the component load contract,
# model-spec defaults, generated Generic* loader behavior, LoRA conversion
# presence, and embedding-load keys without instantiating diffusers/HF objects.
#
# Dtype contract: component storage dtypes are recorded at tensor boundaries.
# Train/fallback dtypes are metadata for the reference loader's compute/load
# helpers; this surface does not upcast persistent tensors.

from serenity_trainer.util.enum.ModelType import (
    MODEL_TYPE_STABLE_DIFFUSION_XL_10_BASE,
    MODEL_TYPE_STABLE_DIFFUSION_XL_10_BASE_INPAINTING,
    model_type_has_conditioning_image_input,
    model_type_is_stable_diffusion_xl,
    model_type_str,
)


comptime SDXL_LOAD_AUTO = 0
comptime SDXL_LOAD_INTERNAL = 1
comptime SDXL_LOAD_DIFFUSERS = 2
comptime SDXL_LOAD_SAFETENSORS = 3
comptime SDXL_LOAD_CKPT = 4

comptime SDXL_LORA_ROUTE_NONE = 0
comptime SDXL_LORA_ROUTE_INTERNAL = 1
comptime SDXL_LORA_ROUTE_CKPT = 2
comptime SDXL_LORA_ROUTE_SAFETENSORS = 3


struct StableDiffusionXLEmbeddingName(Movable):
    var uuid: String
    var model_name: String

    def __init__(out self, var uuid: String, var model_name: String):
        self.uuid = uuid^
        self.model_name = model_name^

    @staticmethod
    def empty() -> StableDiffusionXLEmbeddingName:
        return StableDiffusionXLEmbeddingName(String(), String())


struct StableDiffusionXLModelNames(Movable):
    var base_model: String
    var vae_model: String
    var lora: String
    var embedding: StableDiffusionXLEmbeddingName

    def __init__(
        out self,
        var base_model: String,
        var vae_model: String,
        var lora: String,
        var embedding: StableDiffusionXLEmbeddingName,
    ):
        self.base_model = base_model^
        self.vae_model = vae_model^
        self.lora = lora^
        self.embedding = embedding^

    @staticmethod
    def empty() -> StableDiffusionXLModelNames:
        return StableDiffusionXLModelNames(
            String(),
            String(),
            String(),
            StableDiffusionXLEmbeddingName.empty(),
        )


struct StableDiffusionXLWeightDtypes(Movable):
    var train_dtype: String
    var fallback_train_dtype: String
    var unet: String
    var text_encoder: String
    var text_encoder_2: String
    var vae: String
    var lora: String
    var embedding: String

    def __init__(
        out self,
        var train_dtype: String,
        var fallback_train_dtype: String,
        var unet: String,
        var text_encoder: String,
        var text_encoder_2: String,
        var vae: String,
        var lora: String,
        var embedding: String,
    ):
        self.train_dtype = train_dtype^
        self.fallback_train_dtype = fallback_train_dtype^
        self.unet = unet^
        self.text_encoder = text_encoder^
        self.text_encoder_2 = text_encoder_2^
        self.vae = vae^
        self.lora = lora^
        self.embedding = embedding^

    @staticmethod
    def bf16() -> StableDiffusionXLWeightDtypes:
        return StableDiffusionXLWeightDtypes(
            String("BF16"),
            String("BF16"),
            String("BF16"),
            String("BF16"),
            String("BF16"),
            String("BF16"),
            String("BF16"),
            String("BF16"),
        )


struct StableDiffusionXLQuantizationConfig(Movable):
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
    def default_values() -> StableDiffusionXLQuantizationConfig:
        return StableDiffusionXLQuantizationConfig(
            String(), String("full"), False, String("NONE"), 16, String()
        )


struct StableDiffusionXLModelHandle(Movable):
    var model_type: Int
    var model_spec: String
    var sd_config_name: String
    var sd_config_filename: String
    var base_loaded: Bool
    var lora_loaded: Bool
    var embedding_loaded: Bool

    def __init__(out self, model_type: Int):
        self.model_type = model_type
        self.model_spec = String()
        self.sd_config_name = String()
        self.sd_config_filename = String()
        self.base_loaded = False
        self.lora_loaded = False
        self.embedding_loaded = False


struct StableDiffusionXLLoadPlan(Movable):
    var route: Int
    var model_type: Int
    var model_spec: String
    var sd_config_name: String
    var sd_config_filename_source: String
    var base_model: String
    var vae_model: String
    var lora_model: String
    var internal_probe_file: String
    var internal_route_delegates_to_diffusers: Bool
    var tries_internal_first: Bool
    var tries_diffusers_second: Bool
    var tries_safetensors_third: Bool
    var tries_ckpt_last_when_extension_matches: Bool
    var single_file_supported: Bool
    var safetensors_single_file_pipeline_class: String
    var ckpt_single_file_pipeline_class: String
    var inpainting_pipeline_for_safetensors: Bool
    var tokenizer_1_subfolder: String
    var tokenizer_2_subfolder: String
    var scheduler_subfolder: String
    var text_encoder_1_subfolder: String
    var text_encoder_2_subfolder: String
    var vae_subfolder: String
    var unet_subfolder: String
    var tokenizer_1_class: String
    var tokenizer_2_class: String
    var scheduler_class: String
    var scheduler_factory_noise_scheduler: String
    var text_encoder_1_class: String
    var text_encoder_2_class: String
    var vae_class: String
    var unet_class: String
    var text_encoder_1_storage_dtype: String
    var text_encoder_2_storage_dtype: String
    var unet_storage_dtype: String
    var vae_storage_dtype: String
    var train_dtype: String
    var fallback_train_dtype: String
    var vae_override_supported: Bool
    var unet_quantization_supported: Bool
    var base_loader_invoked: Bool
    var lora_loader_invoked: Bool
    var embedding_loader_invoked: Bool
    var preserves_storage_dtype_at_boundaries: Bool

    def __init__(
        out self,
        route: Int,
        model_type: Int,
        var model_spec: String,
        var sd_config_name: String,
        names: StableDiffusionXLModelNames,
        dtypes: StableDiffusionXLWeightDtypes,
        base_loader_invoked: Bool,
        lora_loader_invoked: Bool,
        embedding_loader_invoked: Bool,
    ):
        self.route = route
        self.model_type = model_type
        self.model_spec = model_spec^
        self.sd_config_name = sd_config_name^
        self.sd_config_filename_source = String("_get_sd_config_name(model_type, model_names.base_model)")
        self.base_model = names.base_model.copy()
        self.vae_model = names.vae_model.copy()
        self.lora_model = names.lora.copy()
        self.internal_probe_file = String("meta.json")
        self.internal_route_delegates_to_diffusers = True
        self.tries_internal_first = True
        self.tries_diffusers_second = True
        self.tries_safetensors_third = True
        self.tries_ckpt_last_when_extension_matches = True
        self.single_file_supported = True
        self.inpainting_pipeline_for_safetensors = model_type_has_conditioning_image_input(model_type)
        if self.inpainting_pipeline_for_safetensors:
            self.safetensors_single_file_pipeline_class = String("StableDiffusionXLInpaintPipeline.from_single_file")
        else:
            self.safetensors_single_file_pipeline_class = String("StableDiffusionXLPipeline.from_single_file")
        self.ckpt_single_file_pipeline_class = String("StableDiffusionXLPipeline.from_single_file")
        self.tokenizer_1_subfolder = String("tokenizer")
        self.tokenizer_2_subfolder = String("tokenizer_2")
        self.scheduler_subfolder = String("scheduler")
        self.text_encoder_1_subfolder = String("text_encoder")
        self.text_encoder_2_subfolder = String("text_encoder_2")
        self.vae_subfolder = String("vae")
        self.unet_subfolder = String("unet")
        self.tokenizer_1_class = String("CLIPTokenizer")
        self.tokenizer_2_class = String("CLIPTokenizer")
        self.scheduler_class = String("DDIMScheduler")
        self.scheduler_factory_noise_scheduler = String("NoiseScheduler.DDIM")
        self.text_encoder_1_class = String("CLIPTextModel")
        self.text_encoder_2_class = String("CLIPTextModelWithProjection")
        self.vae_class = String("AutoencoderKL")
        self.unet_class = String("UNet2DConditionModel")
        self.text_encoder_1_storage_dtype = dtypes.text_encoder.copy()
        self.text_encoder_2_storage_dtype = dtypes.text_encoder_2.copy()
        self.unet_storage_dtype = dtypes.unet.copy()
        self.vae_storage_dtype = dtypes.vae.copy()
        self.train_dtype = dtypes.train_dtype.copy()
        self.fallback_train_dtype = dtypes.fallback_train_dtype.copy()
        self.vae_override_supported = True
        self.unet_quantization_supported = True
        self.base_loader_invoked = base_loader_invoked
        self.lora_loader_invoked = lora_loader_invoked
        self.embedding_loader_invoked = embedding_loader_invoked
        self.preserves_storage_dtype_at_boundaries = True


struct StableDiffusionXLLoraConversionPlan(Movable):
    var has_convert_key_sets: Bool
    var source_namespaces: String
    var load_target_namespace: String
    var safetensors_save_target_namespace: String
    var legacy_save_target_namespace: String
    var internal_save_target_namespace: String
    var root_bundle_embedding_prefix: String
    var unet_omi_prefix: String
    var unet_diffusers_prefix: String
    var clip_l_omi_prefix: String
    var clip_l_diffusers_prefix: String
    var clip_g_omi_prefix: String
    var clip_g_diffusers_prefix: String
    var range_upper_bound: Int
    var has_unet_resnet_rules: Bool
    var has_unet_attention_rules: Bool
    var has_clip_projection_rules: Bool
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
        self.unet_omi_prefix = String("unet")
        self.unet_diffusers_prefix = String("lora_unet")
        self.clip_l_omi_prefix = String("clip_l")
        self.clip_l_diffusers_prefix = String("lora_te1")
        self.clip_g_omi_prefix = String("clip_g")
        self.clip_g_diffusers_prefix = String("lora_te2")
        self.range_upper_bound = 100
        self.has_unet_resnet_rules = True
        self.has_unet_attention_rules = True
        self.has_clip_projection_rules = True
        self.has_swap_chunks_rules = False
        self.has_filter_is_last_rules = False


struct StableDiffusionXLLoraLoadPlan(Movable):
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
        self.route = SDXL_LORA_ROUTE_SAFETENSORS
        if self.lora_model.byte_length() == 0:
            self.route = SDXL_LORA_ROUTE_NONE
        self.internal_probe_file = String("meta.json")
        self.internal_safetensors_path = String("lora/lora.safetensors")
        self.loads_ckpt_when_extension_matches = True
        self.has_convert_key_sets = True
        self.converted_target_namespace = String("diffusers")
        self.preserves_tensor_storage_dtype = True


struct StableDiffusionXLEmbeddingLoadPlan(Movable):
    var directory: String
    var has_embedding_loader: Bool
    var internal_probe_file: String
    var internal_embedding_path_template: String
    var fallback_to_embedding_model_name: Bool
    var key_clip_l: String
    var key_clip_g: String
    var key_clip_l_out: String
    var key_clip_g_out: String
    var preserves_tensor_storage_dtype: Bool

    def __init__(out self, var directory: String):
        self.directory = directory^
        self.has_embedding_loader = True
        self.internal_probe_file = String("meta.json")
        self.internal_embedding_path_template = String("embeddings/{embedding_uuid}.safetensors")
        self.fallback_to_embedding_model_name = True
        self.key_clip_l = String("clip_l")
        self.key_clip_g = String("clip_g")
        self.key_clip_l_out = String("clip_l_out")
        self.key_clip_g_out = String("clip_g_out")
        self.preserves_tensor_storage_dtype = True


def stable_diffusion_xl_default_sd_config_name(model_type: Int) raises -> String:
    if model_type == MODEL_TYPE_STABLE_DIFFUSION_XL_10_BASE:
        return String("resources/model_config/stable_diffusion_xl/sd_xl_base.yaml")
    if model_type == MODEL_TYPE_STABLE_DIFFUSION_XL_10_BASE_INPAINTING:
        return String("resources/model_config/stable_diffusion_xl/sd_xl_base-inpainting.yaml")
    raise Error(String("StableDiffusionXLModelLoader: unsupported ModelType ") + model_type_str(model_type))


def stable_diffusion_xl_default_model_spec_name(model_type: Int) raises -> String:
    if model_type == MODEL_TYPE_STABLE_DIFFUSION_XL_10_BASE:
        return String("resources/sd_model_spec/sd_xl_base_1.0.json")
    if model_type == MODEL_TYPE_STABLE_DIFFUSION_XL_10_BASE_INPAINTING:
        return String("resources/sd_model_spec/sd_xl_base_1.0_inpainting.json")
    raise Error(String("StableDiffusionXLFineTuneModelLoader: unsupported ModelType ") + model_type_str(model_type))


def stable_diffusion_xl_lora_default_model_spec_name(model_type: Int) raises -> String:
    if model_type == MODEL_TYPE_STABLE_DIFFUSION_XL_10_BASE:
        return String("resources/sd_model_spec/sd_xl_base_1.0-lora.json")
    if model_type == MODEL_TYPE_STABLE_DIFFUSION_XL_10_BASE_INPAINTING:
        return String("resources/sd_model_spec/sd_xl_base_1.0_inpainting-lora.json")
    raise Error(String("StableDiffusionXLLoRAModelLoader: unsupported ModelType ") + model_type_str(model_type))


def stable_diffusion_xl_embedding_default_model_spec_name(model_type: Int) raises -> String:
    if model_type == MODEL_TYPE_STABLE_DIFFUSION_XL_10_BASE:
        return String("resources/sd_model_spec/sd_xl_base_1.0-embedding.json")
    if model_type == MODEL_TYPE_STABLE_DIFFUSION_XL_10_BASE_INPAINTING:
        return String("resources/sd_model_spec/sd_xl_base_1.0_inpainting-embedding.json")
    raise Error(String("StableDiffusionXLEmbeddingModelLoader: unsupported ModelType ") + model_type_str(model_type))


def stable_diffusion_xl_lora_loader_has_convert_key_sets() -> Bool:
    return True


def stable_diffusion_xl_lora_conversion_plan() -> StableDiffusionXLLoraConversionPlan:
    return StableDiffusionXLLoraConversionPlan()


def _validate_sdxl_model_type(caller: String, model_type: Int) raises:
    if not model_type_is_stable_diffusion_xl(model_type):
        raise Error(caller + String(": unsupported ModelType ") + model_type_str(model_type))


struct StableDiffusionXLModelLoader(Movable):
    def __init__(out self):
        pass

    def _default_sd_config_name(self, model_type: Int) raises -> String:
        return stable_diffusion_xl_default_sd_config_name(model_type)

    def load(
        self,
        mut model: StableDiffusionXLModelHandle,
        model_type: Int,
        model_names: StableDiffusionXLModelNames,
        weight_dtypes: StableDiffusionXLWeightDtypes,
        quantization: StableDiffusionXLQuantizationConfig,
    ) raises -> StableDiffusionXLLoadPlan:
        _validate_sdxl_model_type(String("StableDiffusionXLModelLoader.load"), model_type)
        _ = quantization
        model.model_type = model_type
        model.model_spec = stable_diffusion_xl_default_model_spec_name(model_type)
        model.sd_config_name = stable_diffusion_xl_default_sd_config_name(model_type)
        model.sd_config_filename = String("_get_sd_config_name(model_type, model_names.base_model)")
        model.base_loaded = True
        return StableDiffusionXLLoadPlan(
            SDXL_LOAD_AUTO,
            model_type,
            model.model_spec.copy(),
            model.sd_config_name.copy(),
            model_names,
            weight_dtypes,
            True,
            False,
            False,
        )

    def load_safetensors(
        self,
        mut model: StableDiffusionXLModelHandle,
        model_type: Int,
        model_names: StableDiffusionXLModelNames,
        weight_dtypes: StableDiffusionXLWeightDtypes,
        quantization: StableDiffusionXLQuantizationConfig,
    ) raises -> StableDiffusionXLLoadPlan:
        _validate_sdxl_model_type(String("StableDiffusionXLModelLoader.load_safetensors"), model_type)
        _ = quantization
        model.model_type = model_type
        model.model_spec = stable_diffusion_xl_default_model_spec_name(model_type)
        model.sd_config_name = stable_diffusion_xl_default_sd_config_name(model_type)
        model.sd_config_filename = String("_get_sd_config_name(model_type, model_names.base_model)")
        model.base_loaded = True
        return StableDiffusionXLLoadPlan(
            SDXL_LOAD_SAFETENSORS,
            model_type,
            model.model_spec.copy(),
            model.sd_config_name.copy(),
            model_names,
            weight_dtypes,
            True,
            False,
            False,
        )

    def load_ckpt(
        self,
        mut model: StableDiffusionXLModelHandle,
        model_type: Int,
        model_names: StableDiffusionXLModelNames,
        weight_dtypes: StableDiffusionXLWeightDtypes,
        quantization: StableDiffusionXLQuantizationConfig,
    ) raises -> StableDiffusionXLLoadPlan:
        _validate_sdxl_model_type(String("StableDiffusionXLModelLoader.load_ckpt"), model_type)
        _ = quantization
        model.model_type = model_type
        model.model_spec = stable_diffusion_xl_default_model_spec_name(model_type)
        model.sd_config_name = stable_diffusion_xl_default_sd_config_name(model_type)
        model.sd_config_filename = String("_get_sd_config_name(model_type, model_names.base_model)")
        model.base_loaded = True
        return StableDiffusionXLLoadPlan(
            SDXL_LOAD_CKPT,
            model_type,
            model.model_spec.copy(),
            model.sd_config_name.copy(),
            model_names,
            weight_dtypes,
            True,
            False,
            False,
        )


struct StableDiffusionXLLoRALoader(Movable):
    def __init__(out self):
        pass

    def _get_convert_key_sets(self) -> Bool:
        return stable_diffusion_xl_lora_loader_has_convert_key_sets()

    def conversion_plan(self) -> StableDiffusionXLLoraConversionPlan:
        return stable_diffusion_xl_lora_conversion_plan()

    def load(
        self,
        mut model: StableDiffusionXLModelHandle,
        model_names: StableDiffusionXLModelNames,
    ) -> StableDiffusionXLLoraLoadPlan:
        var plan = StableDiffusionXLLoraLoadPlan(model_names.lora.copy())
        model.lora_loaded = plan.route != SDXL_LORA_ROUTE_NONE
        return plan^


struct StableDiffusionXLEmbeddingLoader(Movable):
    def __init__(out self):
        pass

    def load(
        self,
        mut model: StableDiffusionXLModelHandle,
        directory: String,
        model_names: StableDiffusionXLModelNames,
    ) -> StableDiffusionXLEmbeddingLoadPlan:
        _ = model_names
        model.embedding_loaded = directory.byte_length() > 0
        return StableDiffusionXLEmbeddingLoadPlan(directory.copy())


struct StableDiffusionXLFineTuneModelLoader(Movable):
    def __init__(out self):
        pass

    def _default_model_spec_name(self, model_type: Int) raises -> String:
        return stable_diffusion_xl_default_model_spec_name(model_type)

    def load(
        self,
        model_type: Int,
        model_names: StableDiffusionXLModelNames,
        weight_dtypes: StableDiffusionXLWeightDtypes,
        quantization: StableDiffusionXLQuantizationConfig,
    ) raises -> StableDiffusionXLLoadPlan:
        _validate_sdxl_model_type(String("StableDiffusionXLFineTuneModelLoader.load"), model_type)
        var model = StableDiffusionXLModelHandle(model_type)
        model.model_spec = stable_diffusion_xl_default_model_spec_name(model_type)
        var base_loader = StableDiffusionXLModelLoader()
        var plan = base_loader.load(model, model_type, model_names, weight_dtypes, quantization)
        var embedding_loader = StableDiffusionXLEmbeddingLoader()
        _ = embedding_loader.load(model, model_names.base_model.copy(), model_names)
        plan.model_spec = stable_diffusion_xl_default_model_spec_name(model_type)
        plan.embedding_loader_invoked = True
        return plan^


struct StableDiffusionXLLoRAModelLoader(Movable):
    def __init__(out self):
        pass

    def _default_model_spec_name(self, model_type: Int) raises -> String:
        return stable_diffusion_xl_lora_default_model_spec_name(model_type)

    def load(
        self,
        model_type: Int,
        model_names: StableDiffusionXLModelNames,
        weight_dtypes: StableDiffusionXLWeightDtypes,
        quantization: StableDiffusionXLQuantizationConfig,
    ) raises -> StableDiffusionXLLoadPlan:
        _validate_sdxl_model_type(String("StableDiffusionXLLoRAModelLoader.load"), model_type)
        var model = StableDiffusionXLModelHandle(model_type)
        model.model_spec = stable_diffusion_xl_lora_default_model_spec_name(model_type)

        var base_loader_invoked = False
        var plan = StableDiffusionXLLoadPlan(
            SDXL_LOAD_AUTO,
            model_type,
            model.model_spec.copy(),
            stable_diffusion_xl_default_sd_config_name(model_type),
            model_names,
            weight_dtypes,
            False,
            False,
            False,
        )

        if model_names.base_model.byte_length() > 0:
            var base_loader = StableDiffusionXLModelLoader()
            plan = base_loader.load(model, model_type, model_names, weight_dtypes, quantization)
            base_loader_invoked = True

        var lora_loader = StableDiffusionXLLoRALoader()
        _ = lora_loader.load(model, model_names)
        var embedding_loader = StableDiffusionXLEmbeddingLoader()
        _ = embedding_loader.load(model, model_names.lora.copy(), model_names)

        plan.model_spec = stable_diffusion_xl_lora_default_model_spec_name(model_type)
        plan.base_loader_invoked = base_loader_invoked
        plan.lora_loader_invoked = True
        plan.embedding_loader_invoked = True
        plan.lora_model = model_names.lora.copy()
        return plan^


struct StableDiffusionXLEmbeddingModelLoader(Movable):
    def __init__(out self):
        pass

    def _default_model_spec_name(self, model_type: Int) raises -> String:
        return stable_diffusion_xl_embedding_default_model_spec_name(model_type)

    def load(
        self,
        model_type: Int,
        model_names: StableDiffusionXLModelNames,
        weight_dtypes: StableDiffusionXLWeightDtypes,
        quantization: StableDiffusionXLQuantizationConfig,
    ) raises -> StableDiffusionXLLoadPlan:
        _validate_sdxl_model_type(String("StableDiffusionXLEmbeddingModelLoader.load"), model_type)
        var model = StableDiffusionXLModelHandle(model_type)
        model.model_spec = stable_diffusion_xl_embedding_default_model_spec_name(model_type)

        var base_loader_invoked = False
        var plan = StableDiffusionXLLoadPlan(
            SDXL_LOAD_AUTO,
            model_type,
            model.model_spec.copy(),
            stable_diffusion_xl_default_sd_config_name(model_type),
            model_names,
            weight_dtypes,
            False,
            False,
            False,
        )

        if model_names.base_model.byte_length() > 0:
            var base_loader = StableDiffusionXLModelLoader()
            plan = base_loader.load(model, model_type, model_names, weight_dtypes, quantization)
            base_loader_invoked = True

        var embedding_loader = StableDiffusionXLEmbeddingLoader()
        _ = embedding_loader.load(model, model_names.embedding.model_name.copy(), model_names)

        plan.model_spec = stable_diffusion_xl_embedding_default_model_spec_name(model_type)
        plan.base_loader_invoked = base_loader_invoked
        plan.embedding_loader_invoked = True
        return plan^

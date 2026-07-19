# 1:1 surface port of Serenity modules/modelLoader/qwen/QwenModelLoader.py
#
# Build-only Qwen support. Serenity's Python loader owns the actual HF /
# diffusers objects:
#   tokenizer      = Qwen2Tokenizer.from_pretrained(.../tokenizer)
#   scheduler      = FlowMatchEulerDiscreteScheduler.from_pretrained(.../scheduler)
#   text_encoder   = Qwen2_5_VLForConditionalGeneration
#   vae            = AutoencoderKLQwenImage
#   transformer    = QwenImageTransformer2DModel
#
# The Mojo Qwen model core is not available in this worker's scope, so this file
# exposes the loader method surface and the exact load plan without pretending to
# construct those runtime modules. Persistent tensors are never upcast here.

from serenity_trainer.util.enum.ModelType import MODEL_TYPE_QWEN, model_type_str


comptime QWEN_LOAD_INTERNAL_OR_DIFFUSERS = 0
comptime QWEN_LOAD_SAFETENSORS_UNSUPPORTED = 1


struct QwenModelNames(Movable):
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
    def empty() -> QwenModelNames:
        return QwenModelNames(String(), String(), String(), String())


struct QwenWeightDtypes(Movable):
    var train_dtype: String
    var fallback_train_dtype: String
    var transformer: String
    var text_encoder: String
    var vae: String

    def __init__(
        out self,
        var train_dtype: String,
        var fallback_train_dtype: String,
        var transformer: String,
        var text_encoder: String,
        var vae: String,
    ):
        self.train_dtype = train_dtype^
        self.fallback_train_dtype = fallback_train_dtype^
        self.transformer = transformer^
        self.text_encoder = text_encoder^
        self.vae = vae^

    @staticmethod
    def bf16() -> QwenWeightDtypes:
        return QwenWeightDtypes(
            String("BF16"), String("BF16"), String("BF16"), String("BF16"), String("BF16")
        )


struct QwenQuantizationConfig(Movable):
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
    def default_values() -> QwenQuantizationConfig:
        return QwenQuantizationConfig(
            String(), String("full"), False, String("NONE"), 16, String()
        )


struct QwenModelHandle(Movable):
    var model_type: Int
    var model_spec: String
    var base_loaded: Bool
    var lora_loaded: Bool

    def __init__(out self, model_type: Int):
        self.model_type = model_type
        self.model_spec = String()
        self.base_loaded = False
        self.lora_loaded = False


struct QwenLoadPlan(Movable):
    var route: Int
    var model_type: Int
    var model_spec: String
    var base_model: String
    var transformer_model: String
    var vae_model: String
    var tokenizer_subfolder: String
    var scheduler_subfolder: String
    var text_encoder_subfolder: String
    var transformer_subfolder: String
    var vae_subfolder: String
    var single_file_base_supported: Bool
    var override_transformer_supported: Bool
    var override_vae_supported: Bool

    def __init__(
        out self,
        route: Int,
        model_type: Int,
        var model_spec: String,
        var base_model: String,
        var transformer_model: String,
        var vae_model: String,
    ):
        self.route = route
        self.model_type = model_type
        self.model_spec = model_spec^
        self.base_model = base_model^
        self.transformer_model = transformer_model^
        self.vae_model = vae_model^
        self.tokenizer_subfolder = String("tokenizer")
        self.scheduler_subfolder = String("scheduler")
        self.text_encoder_subfolder = String("text_encoder")
        self.transformer_subfolder = String("transformer")
        self.vae_subfolder = String("vae")
        self.single_file_base_supported = False
        self.override_transformer_supported = True
        self.override_vae_supported = True


def qwen_default_model_spec_name(model_type: Int) raises -> String:
    if model_type == MODEL_TYPE_QWEN:
        return String("resources/sd_model_spec/qwen.json")
    raise Error(String("QwenModelLoader: unsupported ModelType ") + model_type_str(model_type))


struct QwenModelLoader(Movable):
    def __init__(out self):
        pass

    def _default_model_spec_name(self, model_type: Int) raises -> String:
        return qwen_default_model_spec_name(model_type)

    def load(
        self,
        mut model: QwenModelHandle,
        model_type: Int,
        model_names: QwenModelNames,
        weight_dtypes: QwenWeightDtypes,
        quantization: QwenQuantizationConfig,
    ) raises -> QwenLoadPlan:
        if model_type != MODEL_TYPE_QWEN:
            raise Error(String("QwenModelLoader.load: unsupported ModelType ") + model_type_str(model_type))

        _ = weight_dtypes
        _ = quantization

        model.model_type = model_type
        model.model_spec = qwen_default_model_spec_name(model_type)
        model.base_loaded = True

        return QwenLoadPlan(
            QWEN_LOAD_INTERNAL_OR_DIFFUSERS,
            model_type,
            model.model_spec.copy(),
            model_names.base_model.copy(),
            model_names.transformer_model.copy(),
            model_names.vae_model.copy(),
        )

    def load_safetensors(
        self,
        model: QwenModelHandle,
        model_type: Int,
        model_names: QwenModelNames,
        weight_dtypes: QwenWeightDtypes,
        quantization: QwenQuantizationConfig,
    ) raises -> QwenLoadPlan:
        _ = model
        _ = model_names
        _ = weight_dtypes
        _ = quantization
        if model_type != MODEL_TYPE_QWEN:
            raise Error(String("QwenModelLoader.load_safetensors: unsupported ModelType ") + model_type_str(model_type))
        raise Error(
            "Loading of single file Qwen models is not supported. Use the diffusers model, or override the transformer."
        )

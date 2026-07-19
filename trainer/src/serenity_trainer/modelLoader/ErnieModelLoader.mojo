# 1:1 surface port of Serenity modules/modelLoader/ErnieModelLoader.py
#
# Serenity's Ernie loader constructs HF/diffusers runtime objects:
#   tokenizer    = PreTrainedTokenizerFast.from_pretrained(.../tokenizer)
#   scheduler    = FlowMatchEulerDiscreteScheduler.from_pretrained(.../scheduler)
#   text_encoder = Mistral3Model.from_pretrained(.../text_encoder)
#   vae          = AutoencoderKLFlux2.from_pretrained(.../vae)
#   transformer  = ErnieImageTransformer2DModel, optionally from_single_file
#
# The Ernie model core is outside this worker's scope, so this file exposes the
# loader/factory method surface and the exact load plan without pretending to
# instantiate those runtime modules. Persistent tensors are never upcast here.

from serenity_trainer.util.enum.ModelType import MODEL_TYPE_ERNIE, model_type_str


comptime ERNIE_LOAD_INTERNAL_OR_DIFFUSERS = 0
comptime ERNIE_LOAD_SAFETENSORS_UNSUPPORTED = 1


struct ErnieModelNames(Movable):
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
    def empty() -> ErnieModelNames:
        return ErnieModelNames(String(), String(), String(), String())


struct ErnieWeightDtypes(Movable):
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
    def bf16() -> ErnieWeightDtypes:
        return ErnieWeightDtypes(
            String("BF16"), String("BF16"), String("BF16"), String("BF16"), String("BF16")
        )


struct ErnieQuantizationConfig(Movable):
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
    def default_values() -> ErnieQuantizationConfig:
        return ErnieQuantizationConfig(
            String(), String("full"), False, String("NONE"), 16, String()
        )


struct ErnieModelHandle(Movable):
    var model_type: Int
    var model_spec: String
    var base_loaded: Bool
    var lora_loaded: Bool

    def __init__(out self, model_type: Int):
        self.model_type = model_type
        self.model_spec = String()
        self.base_loaded = False
        self.lora_loaded = False


struct ErnieLoadPlan(Movable):
    var route: Int
    var model_type: Int
    var model_spec: String
    var base_model: String
    var transformer_model: String
    var vae_model: String
    var lora_model: String
    var tokenizer_subfolder: String
    var scheduler_subfolder: String
    var text_encoder_subfolder: String
    var transformer_subfolder: String
    var vae_subfolder: String
    var tokenizer_class: String
    var scheduler_class: String
    var text_encoder_class: String
    var transformer_class: String
    var vae_class: String
    var single_file_base_supported: Bool
    var override_transformer_supported: Bool
    var override_vae_supported: Bool
    var registers_ministral3_config: Bool
    var transformer_override_uses_base_config: Bool
    var prepare_transformer_submodule_from_base: Bool
    var prepare_vae_submodule_from_base: Bool
    var prepare_text_encoder_submodule_from_base: Bool
    var text_encoder_uses_fallback_train_dtype: Bool
    var transformer_override_default_torch_dtype: String
    var transformer_override_gguf_compute_dtype: String
    var base_loader_invoked: Bool
    var lora_loader_invoked: Bool
    var embedding_loader_present: Bool

    def __init__(
        out self,
        route: Int,
        model_type: Int,
        var model_spec: String,
        var base_model: String,
        var transformer_model: String,
        var vae_model: String,
        var lora_model: String,
        base_loader_invoked: Bool,
        lora_loader_invoked: Bool,
    ):
        self.route = route
        self.model_type = model_type
        self.model_spec = model_spec^
        self.base_model = base_model^
        self.transformer_model = transformer_model^
        self.vae_model = vae_model^
        self.lora_model = lora_model^
        self.tokenizer_subfolder = String("tokenizer")
        self.scheduler_subfolder = String("scheduler")
        self.text_encoder_subfolder = String("text_encoder")
        self.transformer_subfolder = String("transformer")
        self.vae_subfolder = String("vae")
        self.tokenizer_class = String("PreTrainedTokenizerFast")
        self.scheduler_class = String("FlowMatchEulerDiscreteScheduler")
        self.text_encoder_class = String("Mistral3Model")
        self.transformer_class = String("ErnieImageTransformer2DModel")
        self.vae_class = String("AutoencoderKLFlux2")
        self.single_file_base_supported = False
        self.override_transformer_supported = True
        self.override_vae_supported = True
        self.registers_ministral3_config = True
        self.transformer_override_uses_base_config = self.transformer_model.byte_length() > 0
        self.prepare_transformer_submodule_from_base = self.transformer_model.byte_length() == 0
        self.prepare_vae_submodule_from_base = self.vae_model.byte_length() == 0
        self.prepare_text_encoder_submodule_from_base = True
        self.text_encoder_uses_fallback_train_dtype = True
        self.transformer_override_default_torch_dtype = String("BF16")
        self.transformer_override_gguf_compute_dtype = String("BF16")
        self.base_loader_invoked = base_loader_invoked
        self.lora_loader_invoked = lora_loader_invoked
        self.embedding_loader_present = False


struct ErnieLoraLoadPlan(Movable):
    var lora_model: String
    var has_convert_key_sets: Bool

    def __init__(out self, var lora_model: String):
        self.lora_model = lora_model^
        self.has_convert_key_sets = False


def ernie_default_model_spec_name(model_type: Int) raises -> String:
    if model_type == MODEL_TYPE_ERNIE:
        return String("resources/sd_model_spec/ernie.json")
    raise Error(String("ErnieModelLoader: unsupported ModelType ") + model_type_str(model_type))


def ernie_lora_default_model_spec_name(model_type: Int) raises -> String:
    if model_type == MODEL_TYPE_ERNIE:
        return String("resources/sd_model_spec/ernie-lora.json")
    raise Error(String("ErnieLoRAModelLoader: unsupported ModelType ") + model_type_str(model_type))


def ernie_lora_loader_has_convert_key_sets() -> Bool:
    return False


struct ErnieModelLoader(Movable):
    def __init__(out self):
        pass

    def _default_model_spec_name(self, model_type: Int) raises -> String:
        return ernie_default_model_spec_name(model_type)

    def load(
        self,
        mut model: ErnieModelHandle,
        model_type: Int,
        model_names: ErnieModelNames,
        weight_dtypes: ErnieWeightDtypes,
        quantization: ErnieQuantizationConfig,
    ) raises -> ErnieLoadPlan:
        if model_type != MODEL_TYPE_ERNIE:
            raise Error(String("ErnieModelLoader.load: unsupported ModelType ") + model_type_str(model_type))

        _ = weight_dtypes
        _ = quantization

        model.model_type = model_type
        model.model_spec = ernie_default_model_spec_name(model_type)
        model.base_loaded = True

        return ErnieLoadPlan(
            ERNIE_LOAD_INTERNAL_OR_DIFFUSERS,
            model_type,
            model.model_spec.copy(),
            model_names.base_model.copy(),
            model_names.transformer_model.copy(),
            model_names.vae_model.copy(),
            model_names.lora.copy(),
            True,
            False,
        )

    def load_safetensors(
        self,
        model: ErnieModelHandle,
        model_type: Int,
        model_names: ErnieModelNames,
        weight_dtypes: ErnieWeightDtypes,
        quantization: ErnieQuantizationConfig,
    ) raises -> ErnieLoadPlan:
        _ = model
        _ = model_names
        _ = weight_dtypes
        _ = quantization
        if model_type != MODEL_TYPE_ERNIE:
            raise Error(String("ErnieModelLoader.load_safetensors: unsupported ModelType ") + model_type_str(model_type))
        raise Error(
            "Loading single-file safetensors for Ernie is not supported. Use the diffusers model instead. Transformer-only safetensor files can be loaded by overriding the transformer."
        )


struct ErnieLoRALoader(Movable):
    def __init__(out self):
        pass

    def _get_convert_key_sets(self) -> Bool:
        return ernie_lora_loader_has_convert_key_sets()

    def load(
        self,
        mut model: ErnieModelHandle,
        model_names: ErnieModelNames,
    ) -> ErnieLoraLoadPlan:
        model.lora_loaded = True
        return ErnieLoraLoadPlan(model_names.lora.copy())


struct ErnieFineTuneModelLoader(Movable):
    def __init__(out self):
        pass

    def _default_model_spec_name(self, model_type: Int) raises -> String:
        return ernie_default_model_spec_name(model_type)

    def load(
        self,
        model_type: Int,
        model_names: ErnieModelNames,
        weight_dtypes: ErnieWeightDtypes,
        quantization: ErnieQuantizationConfig,
    ) raises -> ErnieLoadPlan:
        if model_type != MODEL_TYPE_ERNIE:
            raise Error(String("ErnieFineTuneModelLoader.load: unsupported ModelType ") + model_type_str(model_type))

        var model = ErnieModelHandle(model_type)
        model.model_spec = ernie_default_model_spec_name(model_type)

        var base_loader = ErnieModelLoader()
        return base_loader.load(model, model_type, model_names, weight_dtypes, quantization)


struct ErnieLoRAModelLoader(Movable):
    def __init__(out self):
        pass

    def _default_model_spec_name(self, model_type: Int) raises -> String:
        return ernie_lora_default_model_spec_name(model_type)

    def load(
        self,
        model_type: Int,
        model_names: ErnieModelNames,
        weight_dtypes: ErnieWeightDtypes,
        quantization: ErnieQuantizationConfig,
    ) raises -> ErnieLoadPlan:
        if model_type != MODEL_TYPE_ERNIE:
            raise Error(String("ErnieLoRAModelLoader.load: unsupported ModelType ") + model_type_str(model_type))

        var model = ErnieModelHandle(model_type)
        model.model_spec = ernie_lora_default_model_spec_name(model_type)

        var base_loader_invoked = False
        var plan = ErnieLoadPlan(
            ERNIE_LOAD_INTERNAL_OR_DIFFUSERS,
            model_type,
            model.model_spec.copy(),
            model_names.base_model.copy(),
            model_names.transformer_model.copy(),
            model_names.vae_model.copy(),
            model_names.lora.copy(),
            False,
            False,
        )

        if model_names.base_model.byte_length() > 0:
            var base_loader = ErnieModelLoader()
            plan = base_loader.load(model, model_type, model_names, weight_dtypes, quantization)
            base_loader_invoked = True

        var lora_loader = ErnieLoRALoader()
        _ = lora_loader.load(model, model_names)

        plan.model_spec = ernie_lora_default_model_spec_name(model_type)
        plan.base_loader_invoked = base_loader_invoked
        plan.lora_loader_invoked = True
        plan.lora_model = model_names.lora.copy()
        return plan^

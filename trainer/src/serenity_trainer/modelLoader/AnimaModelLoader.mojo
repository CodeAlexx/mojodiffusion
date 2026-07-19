# 1:1 surface port of Serenity-anima-ref modules/modelLoader/AnimaModelLoader.py
#
# Build-only Anima support. The Python reference constructs HF/diffusers runtime
# modules:
#   tokenizer         = Qwen2Tokenizer.from_pretrained(.../tokenizer)
#   t5_tokenizer      = T5TokenizerFast.from_pretrained(.../t5_tokenizer)
#   scheduler         = FlowMatchEulerDiscreteScheduler.from_pretrained(.../scheduler)
#   text_encoder      = Qwen3Model
#   text_conditioner  = AnimaTextConditioner, always BF16
#   vae               = AutoencoderKLQwenImage
#   transformer       = CosmosTransformer3DModel, optionally from_single_file
#
# This Mojo file exposes the loader/factory method surface and exact component
# plan without pretending to instantiate those runtime modules. Persistent tensor
# state loaded through safetensors uses Tensor.from_view, preserving storage dtype.

from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.tensor import Tensor

from serenity_trainer.util.enum.ModelType import MODEL_TYPE_ANIMA, model_type_str

comptime ANIMA_LOAD_INTERNAL_OR_DIFFUSERS = 0
comptime ANIMA_LOAD_SAFETENSORS_UNSUPPORTED = 1

comptime ANIMA_LORA_NONE = 0
comptime ANIMA_LORA_SAFETENSORS = 1
comptime ANIMA_LORA_INTERNAL = 2

comptime TArc = ArcPointer[Tensor]


def anima_model_type_str(model_type: Int) -> String:
    if model_type == MODEL_TYPE_ANIMA:
        return String("ANIMA")
    return model_type_str(model_type)


struct AnimaModelNames(Movable):
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
    def empty() -> AnimaModelNames:
        return AnimaModelNames(String(), String(), String(), String())


struct AnimaWeightDtypes(Movable):
    var train_dtype: String
    var fallback_train_dtype: String
    var transformer: String
    var text_encoder: String
    var vae: String
    var text_conditioner: String

    def __init__(
        out self,
        var train_dtype: String,
        var fallback_train_dtype: String,
        var transformer: String,
        var text_encoder: String,
        var vae: String,
        var text_conditioner: String,
    ):
        self.train_dtype = train_dtype^
        self.fallback_train_dtype = fallback_train_dtype^
        self.transformer = transformer^
        self.text_encoder = text_encoder^
        self.vae = vae^
        self.text_conditioner = text_conditioner^

    @staticmethod
    def bf16() -> AnimaWeightDtypes:
        return AnimaWeightDtypes(
            String("BF16"),
            String("BF16"),
            String("BF16"),
            String("BF16"),
            String("BF16"),
            String("BF16"),
        )


struct AnimaQuantizationConfig(Movable):
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
    def default_values() -> AnimaQuantizationConfig:
        return AnimaQuantizationConfig(
            String(), String("full"), False, String("NONE"), 16, String()
        )


struct AnimaModelHandle(Movable):
    var model_type: Int
    var model_spec: String
    var base_loaded: Bool
    var lora_loaded: Bool

    def __init__(out self, model_type: Int):
        self.model_type = model_type
        self.model_spec = String()
        self.base_loaded = False
        self.lora_loaded = False


struct AnimaLoadPlan(Movable):
    var route: Int
    var model_type: Int
    var model_spec: String
    var base_model: String
    var transformer_model: String
    var vae_model: String
    var lora_model: String
    var tokenizer_subfolder: String
    var t5_tokenizer_subfolder: String
    var scheduler_subfolder: String
    var text_encoder_subfolder: String
    var text_conditioner_subfolder: String
    var transformer_subfolder: String
    var vae_subfolder: String
    var tokenizer_class: String
    var t5_tokenizer_class: String
    var scheduler_class: String
    var text_encoder_class: String
    var text_conditioner_class: String
    var transformer_class: String
    var vae_class: String
    var internal_probe_file: String
    var internal_load_falls_back_to_diffusers: Bool
    var single_file_base_supported: Bool
    var override_transformer_supported: Bool
    var override_vae_supported: Bool
    var transformer_override_uses_base_config: Bool
    var prepare_transformer_submodule_from_base: Bool
    var prepare_vae_submodule_from_base: Bool
    var prepare_text_encoder_submodule_from_base: Bool
    var text_encoder_uses_fallback_train_dtype: Bool
    var text_conditioner_forced_dtype: String
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
        self.t5_tokenizer_subfolder = String("t5_tokenizer")
        self.scheduler_subfolder = String("scheduler")
        self.text_encoder_subfolder = String("text_encoder")
        self.text_conditioner_subfolder = String("text_conditioner")
        self.transformer_subfolder = String("transformer")
        self.vae_subfolder = String("vae")
        self.tokenizer_class = String("Qwen2Tokenizer")
        self.t5_tokenizer_class = String("T5TokenizerFast")
        self.scheduler_class = String("FlowMatchEulerDiscreteScheduler")
        self.text_encoder_class = String("Qwen3Model")
        self.text_conditioner_class = String("AnimaTextConditioner")
        self.transformer_class = String("CosmosTransformer3DModel")
        self.vae_class = String("AutoencoderKLQwenImage")
        self.internal_probe_file = String("meta.json")
        self.internal_load_falls_back_to_diffusers = True
        self.single_file_base_supported = False
        self.override_transformer_supported = True
        self.override_vae_supported = True
        self.transformer_override_uses_base_config = self.transformer_model.byte_length() > 0
        self.prepare_transformer_submodule_from_base = self.transformer_model.byte_length() == 0
        self.prepare_vae_submodule_from_base = self.vae_model.byte_length() == 0
        self.prepare_text_encoder_submodule_from_base = True
        self.text_encoder_uses_fallback_train_dtype = True
        self.text_conditioner_forced_dtype = String("BF16")
        self.transformer_override_default_torch_dtype = String("BF16")
        self.transformer_override_gguf_compute_dtype = String("BF16")
        self.base_loader_invoked = base_loader_invoked
        self.lora_loader_invoked = lora_loader_invoked
        self.embedding_loader_present = False


struct AnimaLoraStateDict(Movable):
    var names: List[String]
    var tensors: List[TArc]
    var route: Int

    def __init__(out self, var names: List[String], var tensors: List[TArc], route: Int):
        self.names = names^
        self.tensors = tensors^
        self.route = route

    @staticmethod
    def empty() -> AnimaLoraStateDict:
        var names = List[String]()
        var tensors = List[TArc]()
        return AnimaLoraStateDict(names^, tensors^, ANIMA_LORA_NONE)


struct AnimaLoraLoadPlan(Movable):
    var lora_model: String
    var has_convert_key_sets: Bool
    var preserves_peft_keys: Bool

    def __init__(out self, var lora_model: String):
        self.lora_model = lora_model^
        self.has_convert_key_sets = False
        self.preserves_peft_keys = True


def anima_default_model_spec_name(model_type: Int) raises -> String:
    if model_type == MODEL_TYPE_ANIMA:
        return String("resources/sd_model_spec/anima.json")
    raise Error(String("AnimaModelLoader: unsupported ModelType ") + anima_model_type_str(model_type))


def anima_lora_default_model_spec_name(model_type: Int) raises -> String:
    if model_type == MODEL_TYPE_ANIMA:
        return String("resources/sd_model_spec/anima-lora.json")
    raise Error(String("AnimaLoRAModelLoader: unsupported ModelType ") + anima_model_type_str(model_type))


def anima_lora_loader_has_convert_key_sets() -> Bool:
    return False


def load_anima_lora_safetensors(path: String, ctx: DeviceContext) raises -> AnimaLoraStateDict:
    var sharded = ShardedSafeTensors.open(path)
    var names = List[String]()
    var tensors = List[TArc]()
    for ref nm in sharded.names():
        var tv = sharded.tensor_view(nm)
        var t = Tensor.from_view(tv, ctx)
        names.append(nm.copy())
        tensors.append(TArc(t^))
    return AnimaLoraStateDict(names^, tensors^, ANIMA_LORA_SAFETENSORS)


def load_anima_lora_internal(path: String, ctx: DeviceContext) raises -> AnimaLoraStateDict:
    var lora_path = path + String("/lora/lora.safetensors")
    var state = load_anima_lora_safetensors(lora_path, ctx)
    state.route = ANIMA_LORA_INTERNAL
    return state^


struct AnimaModelLoader(Movable):
    def __init__(out self):
        pass

    def _default_model_spec_name(self, model_type: Int) raises -> String:
        return anima_default_model_spec_name(model_type)

    def load(
        self,
        mut model: AnimaModelHandle,
        model_type: Int,
        model_names: AnimaModelNames,
        weight_dtypes: AnimaWeightDtypes,
        quantization: AnimaQuantizationConfig,
    ) raises -> AnimaLoadPlan:
        if model_type != MODEL_TYPE_ANIMA:
            raise Error(String("AnimaModelLoader.load: unsupported ModelType ") + anima_model_type_str(model_type))

        _ = weight_dtypes
        _ = quantization

        model.model_type = model_type
        model.model_spec = anima_default_model_spec_name(model_type)
        model.base_loaded = True

        return AnimaLoadPlan(
            ANIMA_LOAD_INTERNAL_OR_DIFFUSERS,
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
        model: AnimaModelHandle,
        model_type: Int,
        model_names: AnimaModelNames,
        weight_dtypes: AnimaWeightDtypes,
        quantization: AnimaQuantizationConfig,
    ) raises -> AnimaLoadPlan:
        _ = model
        _ = model_names
        _ = weight_dtypes
        _ = quantization
        if model_type != MODEL_TYPE_ANIMA:
            raise Error(String("AnimaModelLoader.load_safetensors: unsupported ModelType ") + anima_model_type_str(model_type))
        raise Error(
            "Loading single-file safetensors for Anima base models is not supported. Use the diffusers model instead. Transformer-only safetensor files can be loaded by overriding the transformer."
        )


struct AnimaLoRALoader(Movable):
    def __init__(out self):
        pass

    def _get_convert_key_sets(self) -> Bool:
        return anima_lora_loader_has_convert_key_sets()

    def load(
        self,
        mut model: AnimaModelHandle,
        model_names: AnimaModelNames,
    ) -> AnimaLoraLoadPlan:
        model.lora_loaded = True
        return AnimaLoraLoadPlan(model_names.lora.copy())

    def load_safetensors(
        self,
        mut model: AnimaModelHandle,
        path: String,
        ctx: DeviceContext,
    ) raises -> AnimaLoraStateDict:
        var state = load_anima_lora_safetensors(path, ctx)
        model.lora_loaded = True
        return state^

    def load_internal(
        self,
        mut model: AnimaModelHandle,
        lora_dir: String,
        ctx: DeviceContext,
    ) raises -> AnimaLoraStateDict:
        var state = load_anima_lora_internal(lora_dir, ctx)
        model.lora_loaded = True
        return state^


struct AnimaFineTuneModelLoader(Movable):
    def __init__(out self):
        pass

    def _default_model_spec_name(self, model_type: Int) raises -> String:
        return anima_default_model_spec_name(model_type)

    def load(
        self,
        model_type: Int,
        model_names: AnimaModelNames,
        weight_dtypes: AnimaWeightDtypes,
        quantization: AnimaQuantizationConfig,
    ) raises -> AnimaLoadPlan:
        if model_type != MODEL_TYPE_ANIMA:
            raise Error(String("AnimaFineTuneModelLoader.load: unsupported ModelType ") + anima_model_type_str(model_type))

        var model = AnimaModelHandle(model_type)
        model.model_spec = anima_default_model_spec_name(model_type)

        var base_loader = AnimaModelLoader()
        return base_loader.load(model, model_type, model_names, weight_dtypes, quantization)


struct AnimaLoRAModelLoader(Movable):
    def __init__(out self):
        pass

    def _default_model_spec_name(self, model_type: Int) raises -> String:
        return anima_lora_default_model_spec_name(model_type)

    def load(
        self,
        model_type: Int,
        model_names: AnimaModelNames,
        weight_dtypes: AnimaWeightDtypes,
        quantization: AnimaQuantizationConfig,
    ) raises -> AnimaLoadPlan:
        if model_type != MODEL_TYPE_ANIMA:
            raise Error(String("AnimaLoRAModelLoader.load: unsupported ModelType ") + anima_model_type_str(model_type))

        var model = AnimaModelHandle(model_type)
        model.model_spec = anima_lora_default_model_spec_name(model_type)

        var base_loader_invoked = False
        var plan = AnimaLoadPlan(
            ANIMA_LOAD_INTERNAL_OR_DIFFUSERS,
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
            var base_loader = AnimaModelLoader()
            plan = base_loader.load(model, model_type, model_names, weight_dtypes, quantization)
            base_loader_invoked = True

        var lora_loader = AnimaLoRALoader()
        _ = lora_loader.load(model, model_names)

        plan.model_spec = anima_lora_default_model_spec_name(model_type)
        plan.base_loader_invoked = base_loader_invoked
        plan.lora_loader_invoked = True
        plan.lora_model = model_names.lora.copy()
        return plan^

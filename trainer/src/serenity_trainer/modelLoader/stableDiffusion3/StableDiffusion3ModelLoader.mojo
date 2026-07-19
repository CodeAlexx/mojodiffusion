# 1:1 surface port of Serenity
#   modules/modelLoader/stableDiffusion3/StableDiffusion3ModelLoader.py
#
# Build-only SD3/SD3.5 loader support. The Mojo SD3 transformer, VAE, and text
# encoders are not implemented in this worker's scope, so this file exposes the
# component load contract and generated Generic* loader surfaces without
# pretending to instantiate diffusers/HF runtime objects.
#
# Dtype contract: this surface records the storage dtype requested for each
# component. It does not upcast persistent tensors. SD3's Serenity loader uses
# fallback_train_dtype only for the T5 encoder compute/load path; the component
# storage dtype remains the requested text_encoder_3 dtype.

from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.tensor import Tensor

from serenity_trainer.util.enum.ModelType import (
    MODEL_TYPE_STABLE_DIFFUSION_3,
    MODEL_TYPE_STABLE_DIFFUSION_35,
    model_type_is_stable_diffusion_3,
    model_type_str,
)
from serenity_trainer.modelSetup.stableDiffusion3LoraTargets import (
    StableDiffusion3LoraTargetSpecs,
    sd3_lora_down_key,
    sd3_lora_up_key,
    sd3_lora_alpha_key,
)


comptime SD3_LOAD_AUTO = 0
comptime SD3_LOAD_INTERNAL = 1
comptime SD3_LOAD_DIFFUSERS = 2
comptime SD3_LOAD_SINGLE_FILE = 3

comptime SD3_LORA_ROUTE_NONE = 0
comptime SD3_LORA_ROUTE_INTERNAL = 1
comptime SD3_LORA_ROUTE_CKPT = 2
comptime SD3_LORA_ROUTE_SAFETENSORS = 3

comptime TArc = ArcPointer[Tensor]


struct StableDiffusion3EmbeddingName(Movable):
    var uuid: String
    var model_name: String

    def __init__(out self, var uuid: String, var model_name: String):
        self.uuid = uuid^
        self.model_name = model_name^

    @staticmethod
    def empty() -> StableDiffusion3EmbeddingName:
        return StableDiffusion3EmbeddingName(String(), String())


struct StableDiffusion3ModelNames(Movable):
    var base_model: String
    var vae_model: String
    var lora: String
    var embedding: StableDiffusion3EmbeddingName
    var include_text_encoder_1: Bool
    var include_text_encoder_2: Bool
    var include_text_encoder_3: Bool

    def __init__(
        out self,
        var base_model: String,
        var vae_model: String,
        var lora: String,
        var embedding: StableDiffusion3EmbeddingName,
        include_text_encoder_1: Bool = True,
        include_text_encoder_2: Bool = True,
        include_text_encoder_3: Bool = True,
    ):
        self.base_model = base_model^
        self.vae_model = vae_model^
        self.lora = lora^
        self.embedding = embedding^
        self.include_text_encoder_1 = include_text_encoder_1
        self.include_text_encoder_2 = include_text_encoder_2
        self.include_text_encoder_3 = include_text_encoder_3

    @staticmethod
    def empty() -> StableDiffusion3ModelNames:
        return StableDiffusion3ModelNames(
            String(),
            String(),
            String(),
            StableDiffusion3EmbeddingName.empty(),
            True,
            True,
            True,
        )


struct StableDiffusion3WeightDtypes(Movable):
    var train_dtype: String
    var fallback_train_dtype: String
    var transformer: String
    var text_encoder_1: String
    var text_encoder_2: String
    var text_encoder_3: String
    var vae: String
    var lora: String
    var embedding: String

    def __init__(
        out self,
        var train_dtype: String,
        var fallback_train_dtype: String,
        var transformer: String,
        var text_encoder_1: String,
        var text_encoder_2: String,
        var text_encoder_3: String,
        var vae: String,
        var lora: String,
        var embedding: String,
    ):
        self.train_dtype = train_dtype^
        self.fallback_train_dtype = fallback_train_dtype^
        self.transformer = transformer^
        self.text_encoder_1 = text_encoder_1^
        self.text_encoder_2 = text_encoder_2^
        self.text_encoder_3 = text_encoder_3^
        self.vae = vae^
        self.lora = lora^
        self.embedding = embedding^

    @staticmethod
    def bf16() -> StableDiffusion3WeightDtypes:
        return StableDiffusion3WeightDtypes(
            String("BF16"),
            String("BF16"),
            String("BF16"),
            String("BF16"),
            String("BF16"),
            String("BF16"),
            String("BF16"),
            String("BF16"),
            String("BF16"),
        )


struct StableDiffusion3QuantizationConfig(Movable):
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
    def default_values() -> StableDiffusion3QuantizationConfig:
        return StableDiffusion3QuantizationConfig(
            String(), String("full"), False, String("NONE"), 16, String()
        )


struct StableDiffusion3ModelHandle(Movable):
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


struct StableDiffusion3LoadPlan(Movable):
    var route: Int
    var model_type: Int
    var model_spec: String
    var base_model: String
    var vae_model: String
    var lora_model: String
    var internal_probe_file: String
    var internal_route_delegates_to_diffusers: Bool
    var tries_internal_first: Bool
    var tries_diffusers_second: Bool
    var tries_single_file_last: Bool
    var single_file_supported: Bool
    var single_file_pipeline_class: String
    var single_file_tokenizer_3_repo: String
    var no_prepare_sub_modules: Bool
    var tokenizer_1_subfolder: String
    var tokenizer_2_subfolder: String
    var tokenizer_3_subfolder: String
    var scheduler_subfolder: String
    var text_encoder_1_subfolder: String
    var text_encoder_2_subfolder: String
    var text_encoder_3_subfolder: String
    var vae_subfolder: String
    var transformer_subfolder: String
    var tokenizer_1_class: String
    var tokenizer_2_class: String
    var tokenizer_3_class: String
    var scheduler_class: String
    var text_encoder_1_class: String
    var text_encoder_2_class: String
    var text_encoder_3_class: String
    var vae_class: String
    var transformer_class: String
    var include_text_encoder_1: Bool
    var include_text_encoder_2: Bool
    var include_text_encoder_3: Bool
    var text_encoder_1_storage_dtype: String
    var text_encoder_2_storage_dtype: String
    var text_encoder_3_storage_dtype: String
    var text_encoder_3_uses_fallback_train_dtype: Bool
    var transformer_storage_dtype: String
    var vae_storage_dtype: String
    var train_dtype: String
    var fallback_train_dtype: String
    var vae_override_supported: Bool
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
        names: StableDiffusion3ModelNames,
        dtypes: StableDiffusion3WeightDtypes,
        base_loader_invoked: Bool,
        lora_loader_invoked: Bool,
        embedding_loader_invoked: Bool,
    ):
        self.route = route
        self.model_type = model_type
        self.model_spec = model_spec^
        self.base_model = names.base_model.copy()
        self.vae_model = names.vae_model.copy()
        self.lora_model = names.lora.copy()
        self.internal_probe_file = String("meta.json")
        self.internal_route_delegates_to_diffusers = True
        self.tries_internal_first = True
        self.tries_diffusers_second = True
        self.tries_single_file_last = True
        self.single_file_supported = True
        self.single_file_pipeline_class = String("StableDiffusion3Pipeline.from_single_file")
        self.single_file_tokenizer_3_repo = String("stabilityai/stable-diffusion-3-medium-diffusers")
        self.no_prepare_sub_modules = True
        self.tokenizer_1_subfolder = String("tokenizer")
        self.tokenizer_2_subfolder = String("tokenizer_2")
        self.tokenizer_3_subfolder = String("tokenizer_3")
        self.scheduler_subfolder = String("scheduler")
        self.text_encoder_1_subfolder = String("text_encoder")
        self.text_encoder_2_subfolder = String("text_encoder_2")
        self.text_encoder_3_subfolder = String("text_encoder_3")
        self.vae_subfolder = String("vae")
        self.transformer_subfolder = String("transformer")
        self.tokenizer_1_class = String("CLIPTokenizer")
        self.tokenizer_2_class = String("CLIPTokenizer")
        self.tokenizer_3_class = String("T5Tokenizer")
        self.scheduler_class = String("FlowMatchEulerDiscreteScheduler")
        self.text_encoder_1_class = String("CLIPTextModelWithProjection")
        self.text_encoder_2_class = String("CLIPTextModelWithProjection")
        self.text_encoder_3_class = String("T5EncoderModel")
        self.vae_class = String("AutoencoderKL")
        self.transformer_class = String("SD3Transformer2DModel")
        self.include_text_encoder_1 = names.include_text_encoder_1
        self.include_text_encoder_2 = names.include_text_encoder_2
        self.include_text_encoder_3 = names.include_text_encoder_3
        self.text_encoder_1_storage_dtype = dtypes.text_encoder_1.copy()
        self.text_encoder_2_storage_dtype = dtypes.text_encoder_2.copy()
        self.text_encoder_3_storage_dtype = dtypes.text_encoder_3.copy()
        self.text_encoder_3_uses_fallback_train_dtype = True
        self.transformer_storage_dtype = dtypes.transformer.copy()
        self.vae_storage_dtype = dtypes.vae.copy()
        self.train_dtype = dtypes.train_dtype.copy()
        self.fallback_train_dtype = dtypes.fallback_train_dtype.copy()
        self.vae_override_supported = True
        self.transformer_quantization_supported = True
        self.base_loader_invoked = base_loader_invoked
        self.lora_loader_invoked = lora_loader_invoked
        self.embedding_loader_invoked = embedding_loader_invoked
        self.preserves_storage_dtype_at_boundaries = True


struct StableDiffusion3LoraConversionPlan(Movable):
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
    var clip_g_omi_prefix: String
    var t5_omi_prefix: String
    var range_upper_bound: Int
    var transformer_block_rule_count: Int
    var clip_layer_rule_count: Int
    var t5_block_rule_count: Int
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
        self.clip_g_omi_prefix = String("clip_g")
        self.t5_omi_prefix = String("t5")
        self.range_upper_bound = 100
        self.transformer_block_rule_count = 18
        self.clip_layer_rule_count = 6
        self.t5_block_rule_count = 7
        self.has_swap_chunks_rules = True
        self.has_filter_is_last_rules = True


struct StableDiffusion3LoraLoadPlan(Movable):
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
        self.route = SD3_LORA_ROUTE_SAFETENSORS
        if self.lora_model.byte_length() == 0:
            self.route = SD3_LORA_ROUTE_NONE
        self.internal_probe_file = String("meta.json")
        self.internal_safetensors_path = String("lora/lora.safetensors")
        self.loads_ckpt_when_extension_matches = True
        self.has_convert_key_sets = True
        self.converted_target_namespace = String("diffusers")
        self.preserves_tensor_storage_dtype = True


struct StableDiffusion3LoraReload(Movable):
    var a: List[TArc]
    var b: List[TArc]
    var alpha: List[Float32]
    var rank: Int

    def __init__(out self, var a: List[TArc], var b: List[TArc], var alpha: List[Float32], rank: Int):
        self.a = a^
        self.b = b^
        self.alpha = alpha^
        self.rank = rank


struct StableDiffusion3EmbeddingLoadPlan(Movable):
    var directory: String
    var has_embedding_loader: Bool
    var key_clip_l: String
    var key_clip_g: String
    var key_t5: String
    var key_clip_l_out: String
    var key_clip_g_out: String
    var key_t5_out: String
    var preserves_tensor_storage_dtype: Bool

    def __init__(out self, var directory: String):
        self.directory = directory^
        self.has_embedding_loader = True
        self.key_clip_l = String("clip_l")
        self.key_clip_g = String("clip_g")
        self.key_t5 = String("t5")
        self.key_clip_l_out = String("clip_l_out")
        self.key_clip_g_out = String("clip_g_out")
        self.key_t5_out = String("t5_out")
        self.preserves_tensor_storage_dtype = True


def stable_diffusion3_default_model_spec_name(model_type: Int) raises -> String:
    if model_type == MODEL_TYPE_STABLE_DIFFUSION_3:
        return String("resources/sd_model_spec/sd_3_2b_1.0.json")
    if model_type == MODEL_TYPE_STABLE_DIFFUSION_35:
        return String("resources/sd_model_spec/sd_3.5_1.0.json")
    raise Error(String("StableDiffusion3ModelLoader: unsupported ModelType ") + model_type_str(model_type))


def stable_diffusion3_lora_default_model_spec_name(model_type: Int) raises -> String:
    if model_type == MODEL_TYPE_STABLE_DIFFUSION_3:
        return String("resources/sd_model_spec/sd_3_2b_1.0-lora.json")
    if model_type == MODEL_TYPE_STABLE_DIFFUSION_35:
        return String("resources/sd_model_spec/sd_3.5_1.0-lora.json")
    raise Error(String("StableDiffusion3LoRAModelLoader: unsupported ModelType ") + model_type_str(model_type))


def stable_diffusion3_embedding_default_model_spec_name(model_type: Int) raises -> String:
    if model_type == MODEL_TYPE_STABLE_DIFFUSION_3:
        return String("resources/sd_model_spec/sd_3_2b_1.0-embedding.json")
    if model_type == MODEL_TYPE_STABLE_DIFFUSION_35:
        return String("resources/sd_model_spec/sd_3.5_1.0-embedding.json")
    raise Error(String("StableDiffusion3EmbeddingModelLoader: unsupported ModelType ") + model_type_str(model_type))


def stable_diffusion3_lora_loader_has_convert_key_sets() -> Bool:
    return True


def stable_diffusion3_lora_conversion_plan() -> StableDiffusion3LoraConversionPlan:
    return StableDiffusion3LoraConversionPlan()


def load_stable_diffusion3_lora_targets(
    path: String,
    targets: StableDiffusion3LoraTargetSpecs,
    ctx: DeviceContext,
    expected_dtype: STDtype = STDtype.BF16,
) raises -> StableDiffusion3LoraReload:
    var sharded = ShardedSafeTensors.open(path)
    var have = Dict[String, Int]()
    for ref nm in sharded.names():
        have[nm] = 1

    var a = List[TArc]()
    var b = List[TArc]()
    var alpha = List[Float32]()
    var rank = -1
    for i in range(targets.len()):
        var prefix = targets.prefixes[i]
        var in_features = targets.in_features[i]
        var out_features = targets.out_features[i]
        var ak = sd3_lora_down_key(prefix)
        var bk = sd3_lora_up_key(prefix)
        if not (ak in have):
            raise Error(String("load_stable_diffusion3_lora_targets: missing ") + ak)
        if not (bk in have):
            raise Error(String("load_stable_diffusion3_lora_targets: missing ") + bk)

        var at = Tensor.from_view(sharded.tensor_view(ak), ctx)
        var bt = Tensor.from_view(sharded.tensor_view(bk), ctx)
        _expect_sd3_lora_tensor(ak, at, expected_dtype)
        _expect_sd3_lora_tensor(bk, bt, expected_dtype)

        var ash = at.shape()
        var bsh = bt.shape()
        _expect_sd3_int(ak + String(".rank"), len(ash), 2)
        _expect_sd3_int(bk + String(".rank"), len(bsh), 2)
        _expect_sd3_int(ak + String(".in"), ash[1], in_features)
        _expect_sd3_int(bk + String(".out"), bsh[0], out_features)
        if rank < 0:
            rank = ash[0]
        _expect_sd3_int(ak + String(".rank_dim"), ash[0], rank)
        _expect_sd3_int(bk + String(".rank_dim"), bsh[1], rank)

        var al = Float32(rank)
        var ah = sd3_lora_alpha_key(prefix)
        if ah in have:
            var alt = Tensor.from_view(sharded.tensor_view(ah), ctx)
            _expect_sd3_lora_tensor(ah, alt, expected_dtype)
            var alsh = alt.shape()
            if len(alsh) != 0:
                if len(alsh) != 1 or alsh[0] != 1:
                    raise Error(ah + String(": alpha must be scalar or [1]"))
            var host = alt.to_host(ctx)
            if len(host) > 0:
                al = host[0]

        a.append(TArc(at^))
        b.append(TArc(bt^))
        alpha.append(al)

    return StableDiffusion3LoraReload(a^, b^, alpha^, rank)


def _expect_sd3_lora_tensor(name: String, tensor: Tensor, expected_dtype: STDtype) raises:
    if tensor.dtype() != expected_dtype:
        raise Error(
            name + String(": dtype got ") + tensor.dtype().name()
            + String(", expected ") + expected_dtype.name()
        )


def _expect_sd3_int(name: String, got: Int, expected: Int) raises:
    if got != expected:
        raise Error(
            name + String(": got ") + String(got)
            + String(", expected ") + String(expected)
        )


def _validate_sd3_model_type(caller: String, model_type: Int) raises:
    if not model_type_is_stable_diffusion_3(model_type):
        raise Error(caller + String(": unsupported ModelType ") + model_type_str(model_type))


struct StableDiffusion3ModelLoader(Movable):
    def __init__(out self):
        pass

    def _default_model_spec_name(self, model_type: Int) raises -> String:
        return stable_diffusion3_default_model_spec_name(model_type)

    def load(
        self,
        mut model: StableDiffusion3ModelHandle,
        model_type: Int,
        model_names: StableDiffusion3ModelNames,
        weight_dtypes: StableDiffusion3WeightDtypes,
        quantization: StableDiffusion3QuantizationConfig,
    ) raises -> StableDiffusion3LoadPlan:
        _validate_sd3_model_type(String("StableDiffusion3ModelLoader.load"), model_type)
        _ = quantization
        model.model_type = model_type
        model.model_spec = stable_diffusion3_default_model_spec_name(model_type)
        model.base_loaded = True
        return StableDiffusion3LoadPlan(
            SD3_LOAD_AUTO,
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
        mut model: StableDiffusion3ModelHandle,
        model_type: Int,
        model_names: StableDiffusion3ModelNames,
        weight_dtypes: StableDiffusion3WeightDtypes,
        quantization: StableDiffusion3QuantizationConfig,
    ) raises -> StableDiffusion3LoadPlan:
        _validate_sd3_model_type(String("StableDiffusion3ModelLoader.load_safetensors"), model_type)
        _ = quantization
        model.model_type = model_type
        model.model_spec = stable_diffusion3_default_model_spec_name(model_type)
        model.base_loaded = True
        return StableDiffusion3LoadPlan(
            SD3_LOAD_SINGLE_FILE,
            model_type,
            model.model_spec.copy(),
            model_names,
            weight_dtypes,
            True,
            False,
            False,
        )


struct StableDiffusion3LoRALoader(Movable):
    def __init__(out self):
        pass

    def _get_convert_key_sets(self) -> Bool:
        return stable_diffusion3_lora_loader_has_convert_key_sets()

    def conversion_plan(self) -> StableDiffusion3LoraConversionPlan:
        return stable_diffusion3_lora_conversion_plan()

    def load(
        self,
        mut model: StableDiffusion3ModelHandle,
        model_names: StableDiffusion3ModelNames,
    ) -> StableDiffusion3LoraLoadPlan:
        var plan = StableDiffusion3LoraLoadPlan(model_names.lora.copy())
        model.lora_loaded = plan.route != SD3_LORA_ROUTE_NONE
        return plan^


struct StableDiffusion3EmbeddingLoader(Movable):
    def __init__(out self):
        pass

    def load(
        self,
        mut model: StableDiffusion3ModelHandle,
        directory: String,
        model_names: StableDiffusion3ModelNames,
    ) -> StableDiffusion3EmbeddingLoadPlan:
        _ = model_names
        model.embedding_loaded = directory.byte_length() > 0
        return StableDiffusion3EmbeddingLoadPlan(directory.copy())


struct StableDiffusion3FineTuneModelLoader(Movable):
    def __init__(out self):
        pass

    def _default_model_spec_name(self, model_type: Int) raises -> String:
        return stable_diffusion3_default_model_spec_name(model_type)

    def load(
        self,
        model_type: Int,
        model_names: StableDiffusion3ModelNames,
        weight_dtypes: StableDiffusion3WeightDtypes,
        quantization: StableDiffusion3QuantizationConfig,
    ) raises -> StableDiffusion3LoadPlan:
        _validate_sd3_model_type(String("StableDiffusion3FineTuneModelLoader.load"), model_type)
        var model = StableDiffusion3ModelHandle(model_type)
        model.model_spec = stable_diffusion3_default_model_spec_name(model_type)
        var base_loader = StableDiffusion3ModelLoader()
        var plan = base_loader.load(model, model_type, model_names, weight_dtypes, quantization)
        var embedding_loader = StableDiffusion3EmbeddingLoader()
        _ = embedding_loader.load(model, model_names.base_model.copy(), model_names)
        plan.embedding_loader_invoked = True
        return plan^


struct StableDiffusion3LoRAModelLoader(Movable):
    def __init__(out self):
        pass

    def _default_model_spec_name(self, model_type: Int) raises -> String:
        return stable_diffusion3_lora_default_model_spec_name(model_type)

    def load(
        self,
        model_type: Int,
        model_names: StableDiffusion3ModelNames,
        weight_dtypes: StableDiffusion3WeightDtypes,
        quantization: StableDiffusion3QuantizationConfig,
    ) raises -> StableDiffusion3LoadPlan:
        _validate_sd3_model_type(String("StableDiffusion3LoRAModelLoader.load"), model_type)
        var model = StableDiffusion3ModelHandle(model_type)
        model.model_spec = stable_diffusion3_lora_default_model_spec_name(model_type)

        var base_loader_invoked = False
        var plan = StableDiffusion3LoadPlan(
            SD3_LOAD_AUTO,
            model_type,
            model.model_spec.copy(),
            model_names,
            weight_dtypes,
            False,
            False,
            False,
        )

        if model_names.base_model.byte_length() > 0:
            var base_loader = StableDiffusion3ModelLoader()
            plan = base_loader.load(model, model_type, model_names, weight_dtypes, quantization)
            base_loader_invoked = True

        var lora_loader = StableDiffusion3LoRALoader()
        _ = lora_loader.load(model, model_names)
        var embedding_loader = StableDiffusion3EmbeddingLoader()
        _ = embedding_loader.load(model, model_names.lora.copy(), model_names)

        plan.model_spec = stable_diffusion3_lora_default_model_spec_name(model_type)
        plan.base_loader_invoked = base_loader_invoked
        plan.lora_loader_invoked = True
        plan.embedding_loader_invoked = True
        plan.lora_model = model_names.lora.copy()
        return plan^


struct StableDiffusion3EmbeddingModelLoader(Movable):
    def __init__(out self):
        pass

    def _default_model_spec_name(self, model_type: Int) raises -> String:
        return stable_diffusion3_embedding_default_model_spec_name(model_type)

    def load(
        self,
        model_type: Int,
        model_names: StableDiffusion3ModelNames,
        weight_dtypes: StableDiffusion3WeightDtypes,
        quantization: StableDiffusion3QuantizationConfig,
    ) raises -> StableDiffusion3LoadPlan:
        _validate_sd3_model_type(String("StableDiffusion3EmbeddingModelLoader.load"), model_type)
        var model = StableDiffusion3ModelHandle(model_type)
        model.model_spec = stable_diffusion3_embedding_default_model_spec_name(model_type)

        var base_loader_invoked = False
        var plan = StableDiffusion3LoadPlan(
            SD3_LOAD_AUTO,
            model_type,
            model.model_spec.copy(),
            model_names,
            weight_dtypes,
            False,
            False,
            False,
        )

        if model_names.base_model.byte_length() > 0:
            var base_loader = StableDiffusion3ModelLoader()
            plan = base_loader.load(model, model_type, model_names, weight_dtypes, quantization)
            base_loader_invoked = True

        var embedding_loader = StableDiffusion3EmbeddingLoader()
        _ = embedding_loader.load(model, model_names.embedding.model_name.copy(), model_names)

        plan.model_spec = stable_diffusion3_embedding_default_model_spec_name(model_type)
        plan.base_loader_invoked = base_loader_invoked
        plan.embedding_loader_invoked = True
        return plan^

# Chroma loader surface contract gate.
#
# Build-only/source-contract coverage against Serenity Chroma loader wrappers
# and leaf loader route metadata. This is not numeric parity.

from serenity_trainer.modelLoader.ChromaEmbeddingModelLoader import (
    ChromaEmbeddingModelLoader,
    chroma_embedding_default_model_spec_name,
)
from serenity_trainer.modelLoader.ChromaFineTuneModelLoader import (
    ChromaFineTuneModelLoader,
    chroma_default_model_spec_name,
)
from serenity_trainer.modelLoader.ChromaLoRAModelLoader import (
    ChromaLoRAModelLoader,
    chroma_lora_default_model_spec_name,
)
from serenity_trainer.modelLoader.chroma.ChromaEmbeddingLoader import (
    ChromaEmbeddingLoader,
)
from serenity_trainer.modelLoader.chroma.ChromaModelLoader import (
    CHROMA_LOAD_AUTO,
    ChromaEmbeddingName,
    ChromaModelHandle,
    ChromaModelLoader,
    ChromaModelNames,
    ChromaQuantizationConfig,
    ChromaWeightDtypes,
)
from serenity_trainer.util.enum.ModelType import MODEL_TYPE_CHROMA_1


def _expect_int(name: String, got: Int, expected: Int) raises:
    if got != expected:
        raise Error(name + String(" got=") + String(got) + String(" expected=") + String(expected))


def _expect_bool(name: String, got: Bool, expected: Bool) raises:
    if got != expected:
        raise Error(name + String(" got=") + String(got) + String(" expected=") + String(expected))


def _expect_string(name: String, got: String, expected: String) raises:
    if got != expected:
        raise Error(name + String(" got=") + got + String(" expected=") + expected)


def main() raises:
    var ft_loader = ChromaFineTuneModelLoader()
    var lora_loader = ChromaLoRAModelLoader()
    var emb_loader = ChromaEmbeddingModelLoader()
    var ft = ft_loader.contract()
    var lora = lora_loader.contract()
    var emb = emb_loader.contract()
    _expect_string("ft factory", ft.factory_name, String("make_fine_tune_model_loader"))
    _expect_string("ft spec", chroma_default_model_spec_name(MODEL_TYPE_CHROMA_1), String("resources/sd_model_spec/chroma.json"))
    _expect_string("lora factory", lora.factory_name, String("make_lora_model_loader"))
    _expect_string("lora spec", chroma_lora_default_model_spec_name(MODEL_TYPE_CHROMA_1), String("resources/sd_model_spec/chroma-lora.json"))
    _expect_string("emb factory", emb.factory_name, String("make_embedding_model_loader"))
    _expect_string("emb spec", chroma_embedding_default_model_spec_name(MODEL_TYPE_CHROMA_1), String("resources/sd_model_spec/chroma-embedding.json"))
    _expect_bool("lora wrapper has lora loader", lora.has_lora_loader(), True)
    _expect_bool("embedding wrapper has lora loader", emb.has_lora_loader(), False)

    var names = ChromaModelNames(
        String("/models/chroma"),
        String("/models/chroma-transformer.safetensors"),
        String("/models/chroma-vae"),
        String("/models/chroma-lora.safetensors"),
        ChromaEmbeddingName(String("emb-1"), String("/models/chroma-embedding.safetensors")),
    )
    var dtypes = ChromaWeightDtypes.bf16()
    var quantization = ChromaQuantizationConfig.default_values()
    var model_loader = ChromaModelLoader()
    var plan = model_loader.load(MODEL_TYPE_CHROMA_1, names, dtypes, quantization)
    _expect_int("load route", plan.route, CHROMA_LOAD_AUTO)
    _expect_bool("internal first", plan.tries_internal_first, True)
    _expect_bool("diffusers second", plan.tries_diffusers_second, True)
    _expect_bool("safetensors third", plan.tries_safetensors_third, True)
    _expect_bool("single file unsupported", plan.single_file_supported, False)
    _expect_string("tokenizer", plan.tokenizer_class, String("T5Tokenizer"))
    _expect_string("scheduler", plan.scheduler_class, String("FlowMatchEulerDiscreteScheduler"))
    _expect_string("transformer", plan.transformer_class, String("ChromaTransformer2DModel"))
    _expect_bool("override", plan.transformer_override_supported, True)
    _expect_bool("override single file", plan.transformer_override_from_single_file, True)
    _expect_string("override dtype", plan.transformer_override_default_torch_dtype, String("BF16"))
    _expect_bool("avoids f32", plan.transformer_override_avoids_float32_load, True)
    _expect_bool("preserves dtype", plan.preserves_storage_dtype_at_boundaries, True)

    var model = ChromaModelHandle(MODEL_TYPE_CHROMA_1)
    var embedding_loader = ChromaEmbeddingLoader()
    var embedding_plan = embedding_loader.load(model, String("/models/chroma"), names)
    _expect_bool("model embedding loaded", model.embedding_loaded, True)
    _expect_string("embedding key t5", embedding_plan.key_t5, String("t5"))
    _expect_string("embedding key t5_out", embedding_plan.key_t5_out, String("t5_out"))

    print("CHROMA SURFACE LOADER CONTRACT OK")

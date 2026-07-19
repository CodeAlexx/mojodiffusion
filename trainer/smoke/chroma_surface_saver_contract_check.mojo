# Chroma saver surface contract gate.
#
# Build-only/source-contract coverage against Serenity Chroma saver wrappers
# and leaf saver route metadata. This is not numeric parity.

from serenity_trainer.modelSaver.ChromaEmbeddingModelSaver import (
    ChromaEmbeddingModelSaver,
)
from serenity_trainer.modelSaver.ChromaFineTuneModelSaver import (
    ChromaFineTuneModelSaver,
)
from serenity_trainer.modelSaver.ChromaLoRAModelSaver import ChromaLoRAModelSaver
from serenity_trainer.modelSaver.chroma.ChromaEmbeddingSaver import (
    ChromaEmbeddingSaver,
    chroma_embedding_keys,
)
from serenity_trainer.modelSaver.chroma.ChromaLoRASaver import (
    CHROMA_FMT_INTERNAL,
    CHROMA_FMT_SAFETENSORS,
    ChromaLoRASaver,
    chroma_lora_bundle_embedding_keys,
)
from serenity_trainer.modelSaver.chroma.ChromaModelSaver import ChromaModelSaver
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
    var model_saver = ChromaModelSaver()
    var model_plan = model_saver.save_plan(CHROMA_FMT_SAFETENSORS, String("/tmp/chroma.safetensors"), String("BF16"))
    _expect_string("model route", model_plan.route_name, String("original_safetensors_checkpoint"))
    _expect_bool("model converter", model_plan.uses_diffusers_to_ckpt_converter, True)
    _expect_bool("model contiguous", model_plan.makes_tensors_contiguous, True)
    _expect_bool("model t5 shard patch", model_plan.patches_t5_max_shard_size_2gb, True)
    _expect_bool("model preserves", model_plan.preserves_storage_dtype_without_override, True)

    var lora_saver = ChromaLoRASaver()
    var lora_plan = lora_saver.save_plan(CHROMA_FMT_INTERNAL, String("/tmp/chroma-internal"))
    var bundle_keys = chroma_lora_bundle_embedding_keys(String("tok"))
    _expect_string("lora route", lora_plan.route_name, String("internal_lora"))
    _expect_string("lora namespace", lora_plan.target_key_namespace, String("omi"))
    _expect_bool("lora text", lora_plan.state_dict_contract.includes_text_encoder_lora, True)
    _expect_bool("lora transformer", lora_plan.state_dict_contract.includes_transformer_lora, True)
    _expect_int("bundle count", len(bundle_keys), 2)
    _expect_string("bundle t5", bundle_keys[0], String("bundle_emb.tok.t5"))
    _expect_string("bundle t5_out", bundle_keys[1], String("bundle_emb.tok.t5_out"))

    var embedding_saver = ChromaEmbeddingSaver()
    var embedding_plan = embedding_saver.save_multiple_plan(CHROMA_FMT_SAFETENSORS, String("/tmp/chroma"))
    var embedding_keys = chroma_embedding_keys()
    _expect_string("embedding route", embedding_plan.route_name, String("embedding_safetensors"))
    _expect_bool("embedding multiple", embedding_plan.is_multiple, True)
    _expect_int("embedding key count", len(embedding_keys), 2)
    _expect_string("embedding key t5", embedding_keys[0], String("t5"))
    _expect_string("embedding key t5_out", embedding_keys[1], String("t5_out"))

    var ft_saver = ChromaFineTuneModelSaver()
    var lora_model_saver = ChromaLoRAModelSaver()
    var emb_model_saver = ChromaEmbeddingModelSaver()
    var ft = ft_saver.contract(MODEL_TYPE_CHROMA_1)
    var lora = lora_model_saver.contract(MODEL_TYPE_CHROMA_1)
    var emb = emb_model_saver.contract(MODEL_TYPE_CHROMA_1)
    _expect_string("ft factory", ft.factory_name, String("make_fine_tune_model_saver"))
    _expect_string("lora factory", lora.factory_name, String("make_lora_model_saver"))
    _expect_string("emb factory", emb.factory_name, String("make_embedding_model_saver"))

    print("CHROMA SURFACE SAVER CONTRACT OK")

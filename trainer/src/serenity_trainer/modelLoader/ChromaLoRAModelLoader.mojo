# 1:1 wrapper-contract mirror of Serenity
#   modules/modelLoader/ChromaLoRAModelLoader.py
#
# Serenity reference:
#   ChromaLoRAModelLoader = make_lora_model_loader(
#       model_spec_map={ModelType.CHROMA_1: "resources/sd_model_spec/chroma-lora.json"},
#       model_class=ChromaModel,
#       model_loader_class=ChromaModelLoader,
#       embedding_loader_class=ChromaEmbeddingLoader,
#       lora_loader_class=ChromaLoRALoader,
#   )
#
# This file intentionally mirrors only the generated wrapper contract. It does
# not instantiate ChromaModelLoader, ChromaEmbeddingLoader, ChromaLoRALoader,
# ChromaModel, or any HF/diffusers runtime object, and it makes no numeric
# parity claim.

from serenity_trainer.modelLoader.ChromaFineTuneModelLoader import (
    CHROMA_WRAPPER_EMBEDDING_LOADER_CLASS,
    CHROMA_WRAPPER_LORA_LOADER_CLASS,
    CHROMA_WRAPPER_MODEL_CLASS,
    CHROMA_WRAPPER_MODEL_LOADER_CLASS,
    ChromaLoaderWrapperContract,
    _validate_chroma_wrapper_model_type,
)
from serenity_trainer.util.enum.ModelType import MODEL_TYPE_CHROMA_1


def chroma_lora_default_model_spec_name(model_type: Int) raises -> String:
    _validate_chroma_wrapper_model_type(String("ChromaLoRAModelLoader"), model_type)
    return String("resources/sd_model_spec/chroma-lora.json")


def chroma_lora_loader_contract() -> ChromaLoaderWrapperContract:
    return ChromaLoaderWrapperContract(
        String("make_lora_model_loader"),
        MODEL_TYPE_CHROMA_1,
        String("resources/sd_model_spec/chroma-lora.json"),
        String(CHROMA_WRAPPER_MODEL_CLASS),
        String(CHROMA_WRAPPER_MODEL_LOADER_CLASS),
        String(CHROMA_WRAPPER_EMBEDDING_LOADER_CLASS),
        String(CHROMA_WRAPPER_LORA_LOADER_CLASS),
    )


struct ChromaLoRAModelLoader(Movable):
    def __init__(out self):
        pass

    def _default_model_spec_name(self, model_type: Int) raises -> String:
        return chroma_lora_default_model_spec_name(model_type)

    def contract(self) -> ChromaLoaderWrapperContract:
        return chroma_lora_loader_contract()

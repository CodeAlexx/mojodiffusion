# 1:1 wrapper-contract mirror of Serenity
#   modules/modelLoader/ChromaEmbeddingModelLoader.py
#
# Serenity reference:
#   ChromaEmbeddingModelLoader = make_embedding_model_loader(
#       model_spec_map={ModelType.CHROMA_1: "resources/sd_model_spec/chroma-embedding.json"},
#       model_class=ChromaModel,
#       model_loader_class=ChromaModelLoader,
#       embedding_loader_class=ChromaEmbeddingLoader,
#   )
#
# This file intentionally mirrors only the generated wrapper contract. It does
# not instantiate ChromaModelLoader, ChromaEmbeddingLoader, ChromaModel, or any
# HF/diffusers runtime object, and it makes no numeric parity claim.

from serenity_trainer.modelLoader.ChromaFineTuneModelLoader import (
    CHROMA_WRAPPER_EMBEDDING_LOADER_CLASS,
    CHROMA_WRAPPER_MODEL_CLASS,
    CHROMA_WRAPPER_MODEL_LOADER_CLASS,
    ChromaLoaderWrapperContract,
    _validate_chroma_wrapper_model_type,
)
from serenity_trainer.util.enum.ModelType import MODEL_TYPE_CHROMA_1


def chroma_embedding_default_model_spec_name(model_type: Int) raises -> String:
    _validate_chroma_wrapper_model_type(String("ChromaEmbeddingModelLoader"), model_type)
    return String("resources/sd_model_spec/chroma-embedding.json")


def chroma_embedding_loader_contract() -> ChromaLoaderWrapperContract:
    return ChromaLoaderWrapperContract(
        String("make_embedding_model_loader"),
        MODEL_TYPE_CHROMA_1,
        String("resources/sd_model_spec/chroma-embedding.json"),
        String(CHROMA_WRAPPER_MODEL_CLASS),
        String(CHROMA_WRAPPER_MODEL_LOADER_CLASS),
        String(CHROMA_WRAPPER_EMBEDDING_LOADER_CLASS),
        String(),
    )


struct ChromaEmbeddingModelLoader(Movable):
    def __init__(out self):
        pass

    def _default_model_spec_name(self, model_type: Int) raises -> String:
        return chroma_embedding_default_model_spec_name(model_type)

    def contract(self) -> ChromaLoaderWrapperContract:
        return chroma_embedding_loader_contract()

# 1:1 wrapper-contract mirror of Serenity
#   modules/modelLoader/ChromaFineTuneModelLoader.py
#
# Serenity reference:
#   ChromaFineTuneModelLoader = make_fine_tune_model_loader(
#       model_spec_map={ModelType.CHROMA_1: "resources/sd_model_spec/chroma.json"},
#       model_class=ChromaModel,
#       model_loader_class=ChromaModelLoader,
#       embedding_loader_class=ChromaEmbeddingLoader,
#   )
#
# This file intentionally mirrors only the generated wrapper contract. It does
# not instantiate ChromaModelLoader, ChromaEmbeddingLoader, ChromaModel, or any
# HF/diffusers runtime object, and it makes no numeric parity claim.

from serenity_trainer.util.enum.ModelType import MODEL_TYPE_CHROMA_1, model_type_str


comptime CHROMA_WRAPPER_MODEL_CLASS = "ChromaModel"
comptime CHROMA_WRAPPER_MODEL_LOADER_CLASS = "ChromaModelLoader"
comptime CHROMA_WRAPPER_EMBEDDING_LOADER_CLASS = "ChromaEmbeddingLoader"
comptime CHROMA_WRAPPER_LORA_LOADER_CLASS = "ChromaLoRALoader"


struct ChromaLoaderWrapperContract(Movable):
    var factory_name: String
    var model_type: Int
    var model_spec: String
    var model_class: String
    var model_loader_class: String
    var embedding_loader_class: String
    var lora_loader_class: String

    def __init__(
        out self,
        var factory_name: String,
        model_type: Int,
        var model_spec: String,
        var model_class: String,
        var model_loader_class: String,
        var embedding_loader_class: String,
        var lora_loader_class: String,
    ):
        self.factory_name = factory_name^
        self.model_type = model_type
        self.model_spec = model_spec^
        self.model_class = model_class^
        self.model_loader_class = model_loader_class^
        self.embedding_loader_class = embedding_loader_class^
        self.lora_loader_class = lora_loader_class^

    def has_embedding_loader(self) -> Bool:
        return self.embedding_loader_class.byte_length() > 0

    def has_lora_loader(self) -> Bool:
        return self.lora_loader_class.byte_length() > 0


def _validate_chroma_wrapper_model_type(caller: String, model_type: Int) raises:
    if model_type != MODEL_TYPE_CHROMA_1:
        raise Error(caller + String(": unsupported ModelType ") + model_type_str(model_type))


def chroma_default_model_spec_name(model_type: Int) raises -> String:
    _validate_chroma_wrapper_model_type(String("ChromaFineTuneModelLoader"), model_type)
    return String("resources/sd_model_spec/chroma.json")


def chroma_fine_tune_loader_contract() -> ChromaLoaderWrapperContract:
    return ChromaLoaderWrapperContract(
        String("make_fine_tune_model_loader"),
        MODEL_TYPE_CHROMA_1,
        String("resources/sd_model_spec/chroma.json"),
        String(CHROMA_WRAPPER_MODEL_CLASS),
        String(CHROMA_WRAPPER_MODEL_LOADER_CLASS),
        String(CHROMA_WRAPPER_EMBEDDING_LOADER_CLASS),
        String(),
    )


struct ChromaFineTuneModelLoader(Movable):
    def __init__(out self):
        pass

    def _default_model_spec_name(self, model_type: Int) raises -> String:
        return chroma_default_model_spec_name(model_type)

    def contract(self) -> ChromaLoaderWrapperContract:
        return chroma_fine_tune_loader_contract()

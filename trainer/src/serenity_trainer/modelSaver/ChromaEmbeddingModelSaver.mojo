# 1:1 surface port of Serenity
#   modules/modelSaver/ChromaEmbeddingModelSaver.py
#
# Serenity:
#   ChromaEmbeddingModelSaver = make_embedding_model_saver(
#       ModelType.CHROMA_1,
#       model_class=ChromaModel,
#       embedding_saver_class=ChromaEmbeddingSaver,
#   )
#
# Build-only wrapper contract mirror. The Chroma leaf saver implementation is
# intentionally not pulled in here while it remains unported.

from serenity_trainer.util.enum.ModelType import MODEL_TYPE_CHROMA_1, model_type_str


comptime CHROMA_EMBEDDING_MODEL_SAVER_MODEL_TYPE = MODEL_TYPE_CHROMA_1
comptime CHROMA_EMBEDDING_MODEL_SAVER_FACTORY = "make_embedding_model_saver"
comptime CHROMA_EMBEDDING_MODEL_SAVER_MODEL_CLASS = "ChromaModel"
comptime CHROMA_EMBEDDING_MODEL_SAVER_EMBEDDING_SAVER_CLASS = "ChromaEmbeddingSaver"


struct ChromaEmbeddingModelSaverContract(Movable):
    var model_type: Int
    var factory_name: String
    var model_class_name: String
    var embedding_saver_class_name: String
    var runtime_save_implemented: Bool

    def __init__(out self):
        self.model_type = CHROMA_EMBEDDING_MODEL_SAVER_MODEL_TYPE
        self.factory_name = String(CHROMA_EMBEDDING_MODEL_SAVER_FACTORY)
        self.model_class_name = String(CHROMA_EMBEDDING_MODEL_SAVER_MODEL_CLASS)
        self.embedding_saver_class_name = String(CHROMA_EMBEDDING_MODEL_SAVER_EMBEDDING_SAVER_CLASS)
        self.runtime_save_implemented = False


def chroma_embedding_model_saver_contract() -> ChromaEmbeddingModelSaverContract:
    return ChromaEmbeddingModelSaverContract()


struct ChromaEmbeddingModelSaver(Movable):
    def __init__(out self):
        pass

    def validate_model_type(self, model_type: Int) raises:
        if model_type != MODEL_TYPE_CHROMA_1:
            raise Error(String("ChromaEmbeddingModelSaver: unsupported ModelType ") + model_type_str(model_type))

    def contract(self, model_type: Int) raises -> ChromaEmbeddingModelSaverContract:
        self.validate_model_type(model_type)
        return chroma_embedding_model_saver_contract()

    def runtime_save_supported(self) -> Bool:
        return False

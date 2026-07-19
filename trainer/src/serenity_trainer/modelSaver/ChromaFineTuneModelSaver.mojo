# 1:1 surface port of Serenity
#   modules/modelSaver/ChromaFineTuneModelSaver.py
#
# Serenity:
#   ChromaFineTuneModelSaver = make_fine_tune_model_saver(
#       ModelType.CHROMA_1,
#       model_class=ChromaModel,
#       model_saver_class=ChromaModelSaver,
#       embedding_saver_class=ChromaEmbeddingSaver,
#   )
#
# Build-only wrapper contract mirror. The Chroma leaf saver implementations are
# intentionally not pulled in here while they remain unported.

from serenity_trainer.util.enum.ModelType import MODEL_TYPE_CHROMA_1, model_type_str


comptime CHROMA_FINE_TUNE_MODEL_SAVER_MODEL_TYPE = MODEL_TYPE_CHROMA_1
comptime CHROMA_FINE_TUNE_MODEL_SAVER_FACTORY = "make_fine_tune_model_saver"
comptime CHROMA_FINE_TUNE_MODEL_SAVER_MODEL_CLASS = "ChromaModel"
comptime CHROMA_FINE_TUNE_MODEL_SAVER_MODEL_SAVER_CLASS = "ChromaModelSaver"
comptime CHROMA_FINE_TUNE_MODEL_SAVER_EMBEDDING_SAVER_CLASS = "ChromaEmbeddingSaver"


struct ChromaFineTuneModelSaverContract(Movable):
    var model_type: Int
    var factory_name: String
    var model_class_name: String
    var model_saver_class_name: String
    var embedding_saver_class_name: String
    var has_embedding_saver: Bool
    var runtime_save_implemented: Bool

    def __init__(out self):
        self.model_type = CHROMA_FINE_TUNE_MODEL_SAVER_MODEL_TYPE
        self.factory_name = String(CHROMA_FINE_TUNE_MODEL_SAVER_FACTORY)
        self.model_class_name = String(CHROMA_FINE_TUNE_MODEL_SAVER_MODEL_CLASS)
        self.model_saver_class_name = String(CHROMA_FINE_TUNE_MODEL_SAVER_MODEL_SAVER_CLASS)
        self.embedding_saver_class_name = String(CHROMA_FINE_TUNE_MODEL_SAVER_EMBEDDING_SAVER_CLASS)
        self.has_embedding_saver = True
        self.runtime_save_implemented = False


def chroma_fine_tune_model_saver_contract() -> ChromaFineTuneModelSaverContract:
    return ChromaFineTuneModelSaverContract()


struct ChromaFineTuneModelSaver(Movable):
    def __init__(out self):
        pass

    def validate_model_type(self, model_type: Int) raises:
        if model_type != MODEL_TYPE_CHROMA_1:
            raise Error(String("ChromaFineTuneModelSaver: unsupported ModelType ") + model_type_str(model_type))

    def contract(self, model_type: Int) raises -> ChromaFineTuneModelSaverContract:
        self.validate_model_type(model_type)
        return chroma_fine_tune_model_saver_contract()

    def runtime_save_supported(self) -> Bool:
        return False

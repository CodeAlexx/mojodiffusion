# 1:1 surface port of Serenity
#   modules/modelSaver/ChromaLoRAModelSaver.py
#
# Serenity:
#   ChromaLoRAModelSaver = make_lora_model_saver(
#       ModelType.CHROMA_1,
#       model_class=ChromaModel,
#       lora_saver_class=ChromaLoRASaver,
#       embedding_saver_class=ChromaEmbeddingSaver,
#   )
#
# Build-only wrapper contract mirror. The Chroma leaf saver implementations are
# intentionally not pulled in here while they remain unported.

from serenity_trainer.util.enum.ModelType import MODEL_TYPE_CHROMA_1, model_type_str


comptime CHROMA_LORA_MODEL_SAVER_MODEL_TYPE = MODEL_TYPE_CHROMA_1
comptime CHROMA_LORA_MODEL_SAVER_FACTORY = "make_lora_model_saver"
comptime CHROMA_LORA_MODEL_SAVER_MODEL_CLASS = "ChromaModel"
comptime CHROMA_LORA_MODEL_SAVER_LORA_SAVER_CLASS = "ChromaLoRASaver"
comptime CHROMA_LORA_MODEL_SAVER_EMBEDDING_SAVER_CLASS = "ChromaEmbeddingSaver"


struct ChromaLoRAModelSaverContract(Movable):
    var model_type: Int
    var factory_name: String
    var model_class_name: String
    var lora_saver_class_name: String
    var embedding_saver_class_name: String
    var has_embedding_saver: Bool
    var runtime_save_implemented: Bool

    def __init__(out self):
        self.model_type = CHROMA_LORA_MODEL_SAVER_MODEL_TYPE
        self.factory_name = String(CHROMA_LORA_MODEL_SAVER_FACTORY)
        self.model_class_name = String(CHROMA_LORA_MODEL_SAVER_MODEL_CLASS)
        self.lora_saver_class_name = String(CHROMA_LORA_MODEL_SAVER_LORA_SAVER_CLASS)
        self.embedding_saver_class_name = String(CHROMA_LORA_MODEL_SAVER_EMBEDDING_SAVER_CLASS)
        self.has_embedding_saver = True
        self.runtime_save_implemented = False


def chroma_lora_model_saver_contract() -> ChromaLoRAModelSaverContract:
    return ChromaLoRAModelSaverContract()


struct ChromaLoRAModelSaver(Movable):
    def __init__(out self):
        pass

    def validate_model_type(self, model_type: Int) raises:
        if model_type != MODEL_TYPE_CHROMA_1:
            raise Error(String("ChromaLoRAModelSaver: unsupported ModelType ") + model_type_str(model_type))

    def contract(self, model_type: Int) raises -> ChromaLoRAModelSaverContract:
        self.validate_model_type(model_type)
        return chroma_lora_model_saver_contract()

    def runtime_save_supported(self) -> Bool:
        return False

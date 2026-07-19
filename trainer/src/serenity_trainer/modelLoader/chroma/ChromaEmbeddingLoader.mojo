# 1:1 surface port of Serenity
#   modules/modelLoader/chroma/ChromaEmbeddingLoader.py
#
# ChromaEmbeddingLoader delegates load() to EmbeddingLoaderMixin._load. This
# records route and keys only; it does not torch.load or safetensors-load
# embedding tensors.

from serenity_trainer.modelLoader.chroma.ChromaModelLoader import (
    ChromaModelHandle,
    ChromaModelNames,
)


struct ChromaEmbeddingLoadPlan(Movable):
    var directory: String
    var model_name: String
    var mixin_method: String
    var loads_all_embedding_names: Bool
    var state_dict_target: String
    var key_t5: String
    var key_t5_out: String
    var preserves_tensor_storage_dtype: Bool

    def __init__(out self, var directory: String, var model_name: String):
        self.directory = directory^
        self.model_name = model_name^
        self.mixin_method = String("EmbeddingLoaderMixin._load")
        self.loads_all_embedding_names = True
        self.state_dict_target = String("model.embedding_state_dicts[uuid]")
        self.key_t5 = String("t5")
        self.key_t5_out = String("t5_out")
        self.preserves_tensor_storage_dtype = True


struct ChromaEmbeddingLoader(Movable):
    def __init__(out self):
        pass

    def load(
        self,
        mut model: ChromaModelHandle,
        directory: String,
        names: ChromaModelNames,
    ) -> ChromaEmbeddingLoadPlan:
        model.embedding_loaded = True
        return ChromaEmbeddingLoadPlan(directory, names.embedding.model_name)

# 1:1 surface port of Serenity
#   modules/modelSaver/chroma/ChromaEmbeddingSaver.py
#
# Chroma embeddings save `t5` and `t5_out` tensors. Storage dtype is preserved
# unless Serenity receives an explicit dtype override.

from serenity_trainer.modelSaver.chroma.ChromaModelSaver import (
    CHROMA_FMT_DIFFUSERS,
    CHROMA_FMT_INTERNAL,
    CHROMA_FMT_SAFETENSORS,
)


struct ChromaEmbeddingSavePlan(Movable):
    var output_model_format: Int
    var output_model_destination: String
    var route_name: String
    var is_multiple: Bool
    var destination_suffix_for_multiple: String
    var multiple_destination_template: String
    var internal_destination_template: String
    var key_t5: String
    var key_t5_out: String
    var diffusers_supported: Bool
    var current_primary_embedding_excluded_from_multiple: Bool
    var preserves_storage_dtype_without_override: Bool

    def __init__(
        out self,
        output_model_format: Int,
        var output_model_destination: String,
        var route_name: String,
        is_multiple: Bool,
    ):
        self.output_model_format = output_model_format
        self.output_model_destination = output_model_destination^
        self.route_name = route_name^
        self.is_multiple = is_multiple
        self.destination_suffix_for_multiple = String("_embeddings")
        self.multiple_destination_template = String("{output_model_destination}_embeddings/{safe_placeholder}.safetensors")
        self.internal_destination_template = String("{output_model_destination}/embeddings/{embedding_uuid}.safetensors")
        self.key_t5 = String("t5")
        self.key_t5_out = String("t5_out")
        self.diffusers_supported = False
        self.current_primary_embedding_excluded_from_multiple = True
        self.preserves_storage_dtype_without_override = True


def chroma_embedding_keys() -> List[String]:
    var keys = List[String]()
    keys.append(String("t5"))
    keys.append(String("t5_out"))
    return keys^


def chroma_embedding_save_plan(
    output_model_format: Int,
    output_model_destination: String,
    is_multiple: Bool,
) raises -> ChromaEmbeddingSavePlan:
    if output_model_format == CHROMA_FMT_DIFFUSERS:
        raise Error("ChromaEmbeddingSaver: DIFFUSERS embedding output is not implemented in Serenity")
    if output_model_format == CHROMA_FMT_SAFETENSORS:
        return ChromaEmbeddingSavePlan(
            output_model_format,
            output_model_destination.copy(),
            String("embedding_safetensors"),
            is_multiple,
        )
    if output_model_format == CHROMA_FMT_INTERNAL:
        return ChromaEmbeddingSavePlan(
            output_model_format,
            output_model_destination.copy(),
            String("internal_embedding"),
            is_multiple,
        )
    raise Error("ChromaEmbeddingSaver: unsupported ModelFormat")


struct ChromaEmbeddingSaver(Movable):
    def __init__(out self):
        pass

    def save_single_plan(
        self,
        output_model_format: Int,
        output_model_destination: String,
    ) raises -> ChromaEmbeddingSavePlan:
        return chroma_embedding_save_plan(
            output_model_format, output_model_destination, False
        )

    def save_multiple_plan(
        self,
        output_model_format: Int,
        output_model_destination: String,
    ) raises -> ChromaEmbeddingSavePlan:
        return chroma_embedding_save_plan(
            output_model_format, output_model_destination, True
        )

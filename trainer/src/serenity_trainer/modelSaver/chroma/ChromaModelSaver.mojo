# 1:1 surface port of Serenity
#   modules/modelSaver/chroma/ChromaModelSaver.py
#
# Build-only Chroma base-model saver contract. It records Serenity routes and
# dtype behavior but does not save diffusers pipelines or safetensors.


comptime CHROMA_FMT_DIFFUSERS = 0
comptime CHROMA_FMT_CKPT = 1
comptime CHROMA_FMT_SAFETENSORS = 2
comptime CHROMA_FMT_LEGACY_SAFETENSORS = 3
comptime CHROMA_FMT_COMFY_LORA = 4
comptime CHROMA_FMT_INTERNAL = 5


struct ChromaModelSavePlan(Movable):
    var output_model_format: Int
    var output_model_destination: String
    var route_name: String
    var dtype_override: String
    var saves_diffusers_pipeline: Bool
    var saves_original_safetensors_checkpoint: Bool
    var saves_internal_as_diffusers: Bool
    var uses_diffusers_to_ckpt_converter: Bool
    var converter_name: String
    var creates_safetensors_header: Bool
    var makes_tensors_contiguous: Bool
    var deep_copy_pipeline_when_dtype_override: Bool
    var moves_pipeline_cpu_before_save: Bool
    var patches_t5_max_shard_size_2gb: Bool
    var safetensors_includes_transformer_state_only: Bool
    var preserves_storage_dtype_without_override: Bool

    def __init__(
        out self,
        output_model_format: Int,
        var output_model_destination: String,
        var route_name: String,
        var dtype_override: String,
    ):
        self.output_model_format = output_model_format
        self.output_model_destination = output_model_destination^
        self.route_name = route_name^
        self.dtype_override = dtype_override^
        self.saves_diffusers_pipeline = output_model_format == CHROMA_FMT_DIFFUSERS or output_model_format == CHROMA_FMT_INTERNAL
        self.saves_original_safetensors_checkpoint = output_model_format == CHROMA_FMT_SAFETENSORS
        self.saves_internal_as_diffusers = output_model_format == CHROMA_FMT_INTERNAL
        self.uses_diffusers_to_ckpt_converter = output_model_format == CHROMA_FMT_SAFETENSORS
        self.converter_name = String("convert_chroma_diffusers_to_ckpt")
        self.creates_safetensors_header = output_model_format == CHROMA_FMT_SAFETENSORS
        self.makes_tensors_contiguous = output_model_format == CHROMA_FMT_SAFETENSORS
        self.deep_copy_pipeline_when_dtype_override = True
        self.moves_pipeline_cpu_before_save = True
        self.patches_t5_max_shard_size_2gb = True
        self.safetensors_includes_transformer_state_only = True
        self.preserves_storage_dtype_without_override = True


def chroma_model_save_plan(
    output_model_format: Int,
    output_model_destination: String,
    dtype_override: String,
) raises -> ChromaModelSavePlan:
    if output_model_format == CHROMA_FMT_DIFFUSERS:
        return ChromaModelSavePlan(
            output_model_format,
            output_model_destination.copy(),
            String("diffusers_pipeline"),
            dtype_override.copy(),
        )
    if output_model_format == CHROMA_FMT_SAFETENSORS:
        return ChromaModelSavePlan(
            output_model_format,
            output_model_destination.copy(),
            String("original_safetensors_checkpoint"),
            dtype_override.copy(),
        )
    if output_model_format == CHROMA_FMT_INTERNAL:
        return ChromaModelSavePlan(
            output_model_format,
            output_model_destination.copy(),
            String("internal_diffusers_pipeline"),
            String(),
        )
    raise Error("ChromaModelSaver: unsupported ModelFormat")


struct ChromaModelSaver(Movable):
    def __init__(out self):
        pass

    def save_plan(
        self,
        output_model_format: Int,
        output_model_destination: String,
        dtype_override: String,
    ) raises -> ChromaModelSavePlan:
        return chroma_model_save_plan(
            output_model_format, output_model_destination, dtype_override
        )

# Ideogram4LoRASetup.mojo - Serenity Trainer setup registration for Ideogram4 LoRA.

from serenity_trainer.modelSetup.BaseIdeogram4Setup import (
    Ideogram4TrainFlowContract,
    ideogram4_frozen_part_names,
    ideogram4_setup_optimization_quantized_parts,
    ideogram4_train_flow_contract,
)
from serenity_trainer.modelSetup.ideogram4LoraTargets import (
    ideogram4_block_lora_save_prefixes,
    ideogram4_full_lora_save_prefixes,
    ideogram4_lora_count,
)
from serenity_trainer.util.enum.ModelType import MODEL_TYPE_IDEOGRAM_4
from serenity_trainer.util.enum.TrainingMethod import TM_LORA


comptime IDEOGRAM4_LORA_MODEL_TYPE = MODEL_TYPE_IDEOGRAM_4
comptime IDEOGRAM4_LORA_TRAINING_METHOD = TM_LORA
comptime IDEOGRAM4_LORA_TRANSFORMER_PART = "transformer"
comptime IDEOGRAM4_LORA_MODEL_CLASS = "Ideogram4Transformer2DModel"
comptime IDEOGRAM4_LORA_EMBEDDINGS_SUPPORTED = False


def ideogram4_lora_parameter_group_names() -> List[String]:
    var names = List[String]()
    names.append(String(IDEOGRAM4_LORA_TRANSFORMER_PART))
    return names^


def ideogram4_lora_layer_prefixes(layer_filter_preset: String) -> List[String]:
    if layer_filter_preset == "blocks":
        return ideogram4_block_lora_save_prefixes()
    return ideogram4_full_lora_save_prefixes()


struct Ideogram4LoRASetupRegistration(Copyable, Movable, ImplicitlyCopyable):
    var model_type: Int
    var training_method: Int
    var target_lora_module_class: String
    var creates_text_encoder_lora: Bool
    var creates_transformer_lora: Bool
    var supports_embedding_training: Bool
    var rank: Int
    var alpha: Float32
    var target_count_full: Int
    var target_count_blocks: Int
    var runtime_forward_present: Bool
    var runtime_lora_backward_present: Bool
    var native_trainable_slice: String

    def __init__(out self, rank: Int = 16, alpha: Float32 = Float32(16.0)):
        self.model_type = IDEOGRAM4_LORA_MODEL_TYPE
        self.training_method = IDEOGRAM4_LORA_TRAINING_METHOD
        self.target_lora_module_class = String(IDEOGRAM4_LORA_MODEL_CLASS)
        self.creates_text_encoder_lora = False
        self.creates_transformer_lora = True
        self.supports_embedding_training = IDEOGRAM4_LORA_EMBEDDINGS_SUPPORTED
        self.rank = rank
        self.alpha = alpha
        self.target_count_full = ideogram4_lora_count()
        self.target_count_blocks = ideogram4_lora_count(include_globals=False)
        self.runtime_forward_present = True
        self.runtime_lora_backward_present = True
        self.native_trainable_slice = String("transformer.layers.* + transformer.final_layer.linear")


def ideogram4_lora_setup_registration(
    rank: Int = 16, alpha: Float32 = Float32(16.0)
) -> Ideogram4LoRASetupRegistration:
    return Ideogram4LoRASetupRegistration(rank, alpha)


struct Ideogram4LoRASetup(Copyable, Movable, ImplicitlyCopyable):
    var registration: Ideogram4LoRASetupRegistration
    var train_flow: Ideogram4TrainFlowContract

    def __init__(
        out self, rank: Int = 16, alpha: Float32 = Float32(16.0)
    ):
        self.registration = ideogram4_lora_setup_registration(rank, alpha)
        self.train_flow = ideogram4_train_flow_contract()

    def create_parameters(self) -> List[String]:
        return ideogram4_lora_parameter_group_names()

    def layer_prefixes(self, layer_filter_preset: String = String("full")) -> List[String]:
        return ideogram4_lora_layer_prefixes(layer_filter_preset)

    def frozen_parts(self) -> List[String]:
        return ideogram4_frozen_part_names()

    def quantized_parts(self) -> List[String]:
        return ideogram4_setup_optimization_quantized_parts()

    def freezes_base_text_encoder(self) -> Bool:
        return True

    def freezes_base_transformer(self) -> Bool:
        return True

    def freezes_vae(self) -> Bool:
        return True

    def uses_cached_text_embeddings(self) -> Bool:
        return True

    def uses_json_captions(self) -> Bool:
        return True

    def native_lora_backward_supported(self) -> Bool:
        return True

    def native_trainable_slice(self) -> String:
        return String("transformer.layers.* + transformer.final_layer.linear")

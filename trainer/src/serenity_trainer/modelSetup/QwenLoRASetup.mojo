# QwenLoRASetup.mojo — build-only registration/setup surface.
#
# Source of truth: /home/alex/Serenity/modules/modelSetup/QwenLoRASetup.py

from serenity_trainer.modelSetup.BaseQwenSetup import (
    qwen_layer_preset_filters,
)
from serenity_trainer.util.enum.ModelType import MODEL_TYPE_QWEN
from serenity_trainer.util.enum.TrainingMethod import TM_LORA


comptime QWEN_LORA_MODEL_TYPE = MODEL_TYPE_QWEN
comptime QWEN_LORA_TRAINING_METHOD = TM_LORA
comptime QWEN_LORA_TEXT_ENCODER_PART = "text_encoder"
comptime QWEN_LORA_TRANSFORMER_PART = "transformer"
comptime QWEN_LORA_EMBEDDINGS_SUPPORTED = False


def qwen_lora_parameter_group_names() -> List[String]:
    var names = List[String]()
    names.append(QWEN_LORA_TEXT_ENCODER_PART)
    names.append(QWEN_LORA_TRANSFORMER_PART)
    return names^


def qwen_lora_layer_filters(layer_filter: String) raises -> List[String]:
    return qwen_layer_preset_filters(layer_filter)


struct QwenLoRASetupRegistration(Copyable, Movable, ImplicitlyCopyable):
    var model_type: Int
    var training_method: Int
    var creates_text_encoder_lora: Bool
    var creates_transformer_lora: Bool
    var supports_embedding_training: Bool

    def __init__(out self):
        self.model_type = QWEN_LORA_MODEL_TYPE
        self.training_method = QWEN_LORA_TRAINING_METHOD
        self.creates_text_encoder_lora = True
        self.creates_transformer_lora = True
        self.supports_embedding_training = QWEN_LORA_EMBEDDINGS_SUPPORTED


def qwen_lora_setup_registration() -> QwenLoRASetupRegistration:
    return QwenLoRASetupRegistration()


struct QwenLoRASetup(Copyable, Movable, ImplicitlyCopyable):
    var registration: QwenLoRASetupRegistration

    def __init__(out self):
        self.registration = qwen_lora_setup_registration()

    def create_parameters(self) -> List[String]:
        return qwen_lora_parameter_group_names()

    def layer_filters(self, layer_filter: String) raises -> List[String]:
        return qwen_lora_layer_filters(layer_filter)

    def freezes_base_text_encoder(self) -> Bool:
        return True

    def freezes_base_transformer(self) -> Bool:
        return True

    def freezes_vae(self) -> Bool:
        return True

    def after_optimizer_step_reapplies_requires_grad(self) -> Bool:
        return True

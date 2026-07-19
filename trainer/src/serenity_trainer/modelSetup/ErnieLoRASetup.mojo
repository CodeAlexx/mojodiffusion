# ErnieLoRASetup.mojo - build-only Ernie LoRA setup surface.
#
# Source of truth: /home/alex/Serenity/modules/modelSetup/ErnieLoRASetup.py

from serenity_trainer.modelSetup.BaseErnieSetup import (
    ernie_layer_preset_filters,
    ernie_train_device_plan,
    ErnieTrainDevicePlan,
)
from serenity_trainer.util.enum.ModelType import MODEL_TYPE_ERNIE
from serenity_trainer.util.enum.TrainingMethod import TM_LORA


comptime ERNIE_LORA_MODEL_TYPE = MODEL_TYPE_ERNIE
comptime ERNIE_LORA_TRAINING_METHOD = TM_LORA
comptime ERNIE_LORA_TRANSFORMER_PART = "transformer"
comptime ERNIE_LORA_EMBEDDINGS_SUPPORTED = False


def ernie_lora_parameter_group_names() -> List[String]:
    var names = List[String]()
    names.append(ERNIE_LORA_TRANSFORMER_PART)
    return names^


def ernie_lora_layer_filters(layer_filter: String) raises -> List[String]:
    return ernie_layer_preset_filters(layer_filter)


struct ErnieLoRASetupRegistration(Copyable, Movable, ImplicitlyCopyable):
    var model_type: Int
    var training_method: Int
    var creates_text_encoder_lora: Bool
    var creates_transformer_lora: Bool
    var supports_embedding_training: Bool

    def __init__(out self):
        self.model_type = ERNIE_LORA_MODEL_TYPE
        self.training_method = ERNIE_LORA_TRAINING_METHOD
        self.creates_text_encoder_lora = False
        self.creates_transformer_lora = True
        self.supports_embedding_training = ERNIE_LORA_EMBEDDINGS_SUPPORTED


def ernie_lora_setup_registration() -> ErnieLoRASetupRegistration:
    return ErnieLoRASetupRegistration()


struct ErnieLoRASetup(Copyable, Movable, ImplicitlyCopyable):
    var registration: ErnieLoRASetupRegistration

    def __init__(out self):
        self.registration = ernie_lora_setup_registration()

    def create_parameters(self) -> List[String]:
        return ernie_lora_parameter_group_names()

    def layer_filters(self, layer_filter: String) raises -> List[String]:
        return ernie_lora_layer_filters(layer_filter)

    def train_device_plan(
        self, latent_caching: Bool, transformer_train: Bool
    ) -> ErnieTrainDevicePlan:
        return ernie_train_device_plan(latent_caching, transformer_train)

    def freezes_base_text_encoder(self) -> Bool:
        return True

    def freezes_base_transformer(self) -> Bool:
        return True

    def freezes_vae(self) -> Bool:
        return True

    def initializes_transformer_lora_from_pending_state_dict(self) -> Bool:
        return True

    def after_optimizer_step_reapplies_requires_grad(self) -> Bool:
        return True

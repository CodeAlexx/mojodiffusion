# AnimaLoRASetup.mojo - build-only Anima LoRA setup surface.
#
# Source of truth: /home/alex/Serenity-anima-ref/modules/modelSetup/AnimaLoRASetup.py

from serenity_trainer.modelSetup.BaseAnimaSetup import (
    ANIMA_MODEL_TYPE_NAME,
    ANIMA_REFERENCE_MODEL_TYPE_INDEX,
    anima_train_device_plan,
    AnimaTrainDevicePlan,
)
from serenity_trainer.util.enum.TrainingMethod import TM_LORA


comptime ANIMA_LORA_MODEL_TYPE_NAME = ANIMA_MODEL_TYPE_NAME
comptime ANIMA_LORA_MODEL_TYPE_REFERENCE_INDEX = ANIMA_REFERENCE_MODEL_TYPE_INDEX
comptime ANIMA_LORA_TRAINING_METHOD = TM_LORA
comptime ANIMA_LORA_TRANSFORMER_PART = "transformer"
comptime ANIMA_LORA_WRAPPER_PREFIX = "transformer"
comptime ANIMA_LORA_EMBEDDINGS_SUPPORTED = False


def anima_lora_parameter_group_names() -> List[String]:
    var names = List[String]()
    names.append(ANIMA_LORA_TRANSFORMER_PART)
    return names^


def anima_lora_wrapper_target_filters_expression() -> String:
    return "config.layer_filter.split(',')"


struct AnimaLoRASetupRegistration(Movable):
    var model_type_name: String
    var model_type_reference_index: Int
    var training_method: Int
    var creates_text_encoder_lora: Bool
    var creates_transformer_lora: Bool
    var supports_embedding_training: Bool

    def __init__(out self):
        self.model_type_name = ANIMA_LORA_MODEL_TYPE_NAME
        self.model_type_reference_index = ANIMA_LORA_MODEL_TYPE_REFERENCE_INDEX
        self.training_method = ANIMA_LORA_TRAINING_METHOD
        self.creates_text_encoder_lora = False
        self.creates_transformer_lora = True
        self.supports_embedding_training = ANIMA_LORA_EMBEDDINGS_SUPPORTED


def anima_lora_setup_registration() -> AnimaLoRASetupRegistration:
    return AnimaLoRASetupRegistration()


struct AnimaLoRASetup(Movable):
    var registration: AnimaLoRASetupRegistration

    def __init__(out self):
        self.registration = anima_lora_setup_registration()

    def create_parameters(self) -> List[String]:
        return anima_lora_parameter_group_names()

    def train_device_plan(
        self, latent_caching: Bool, transformer_train: Bool
    ) -> AnimaTrainDevicePlan:
        return anima_train_device_plan(latent_caching, transformer_train)

    def lora_wrapper_prefix(self) -> String:
        return ANIMA_LORA_WRAPPER_PREFIX

    def lora_wrapper_target_filters_expression(self) -> String:
        return anima_lora_wrapper_target_filters_expression()

    def freezes_base_text_encoder(self) -> Bool:
        return True

    def freezes_base_transformer(self) -> Bool:
        return True

    def freezes_vae(self) -> Bool:
        return True

    def initializes_transformer_lora_from_pending_state_dict(self) -> Bool:
        return True

    def sets_lora_dropout_from_config(self) -> Bool:
        return True

    def moves_lora_to_config_weight_dtype(self) -> Bool:
        return True

    def hooks_lora_to_transformer(self) -> Bool:
        return True

    def after_optimizer_step_reapplies_requires_grad(self) -> Bool:
        return True

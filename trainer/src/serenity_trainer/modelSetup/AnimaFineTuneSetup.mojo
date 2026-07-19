# AnimaFineTuneSetup.mojo - build-only Anima fine-tune setup surface.
#
# Source of truth:
#   /home/alex/Serenity-anima-ref/modules/modelSetup/AnimaFineTuneSetup.py

from serenity_trainer.modelSetup.BaseAnimaSetup import (
    ANIMA_MODEL_TYPE_NAME,
    ANIMA_REFERENCE_MODEL_TYPE_INDEX,
    anima_train_device_plan,
    AnimaTrainDevicePlan,
)
from serenity_trainer.util.enum.TrainingMethod import TM_FINE_TUNE


comptime ANIMA_FINE_TUNE_MODEL_TYPE_NAME = ANIMA_MODEL_TYPE_NAME
comptime ANIMA_FINE_TUNE_MODEL_TYPE_REFERENCE_INDEX = ANIMA_REFERENCE_MODEL_TYPE_INDEX
comptime ANIMA_FINE_TUNE_TRAINING_METHOD = TM_FINE_TUNE
comptime ANIMA_FINE_TUNE_TRANSFORMER_PART = "transformer"
comptime ANIMA_FINE_TUNE_TEXT_ENCODER_TRAINABLE = False
comptime ANIMA_FINE_TUNE_TEXT_CONDITIONER_TRAINABLE = False
comptime ANIMA_FINE_TUNE_VAE_TRAINABLE = False
comptime ANIMA_FINE_TUNE_EMBEDDINGS_SUPPORTED = False


def anima_fine_tune_parameter_group_names() -> List[String]:
    var names = List[String]()
    names.append(ANIMA_FINE_TUNE_TRANSFORMER_PART)
    return names^


struct AnimaFineTuneSetupRegistration(Movable):
    var model_type_name: String
    var model_type_reference_index: Int
    var training_method: Int
    var trains_text_encoder: Bool
    var trains_text_conditioner: Bool
    var trains_transformer: Bool
    var trains_vae: Bool
    var supports_embedding_training: Bool

    def __init__(out self):
        self.model_type_name = ANIMA_FINE_TUNE_MODEL_TYPE_NAME
        self.model_type_reference_index = ANIMA_FINE_TUNE_MODEL_TYPE_REFERENCE_INDEX
        self.training_method = ANIMA_FINE_TUNE_TRAINING_METHOD
        self.trains_text_encoder = ANIMA_FINE_TUNE_TEXT_ENCODER_TRAINABLE
        self.trains_text_conditioner = ANIMA_FINE_TUNE_TEXT_CONDITIONER_TRAINABLE
        self.trains_transformer = True
        self.trains_vae = ANIMA_FINE_TUNE_VAE_TRAINABLE
        self.supports_embedding_training = ANIMA_FINE_TUNE_EMBEDDINGS_SUPPORTED


def anima_fine_tune_setup_registration() -> AnimaFineTuneSetupRegistration:
    return AnimaFineTuneSetupRegistration()


struct AnimaFineTuneSetup(Movable):
    var registration: AnimaFineTuneSetupRegistration

    def __init__(out self):
        self.registration = anima_fine_tune_setup_registration()

    def create_parameters(self) -> List[String]:
        return anima_fine_tune_parameter_group_names()

    def train_device_plan(
        self, latent_caching: Bool, transformer_train: Bool
    ) -> AnimaTrainDevicePlan:
        return anima_train_device_plan(latent_caching, transformer_train)

    def uses_module_filter_for_transformer(self) -> Bool:
        return True

    def freezes_text_encoder(self) -> Bool:
        return True

    def freezes_text_conditioner(self) -> Bool:
        return True

    def freezes_vae(self) -> Bool:
        return True

    def after_optimizer_step_reapplies_requires_grad(self) -> Bool:
        return True

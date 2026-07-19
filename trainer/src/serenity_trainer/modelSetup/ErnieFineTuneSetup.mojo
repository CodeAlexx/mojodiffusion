# ErnieFineTuneSetup.mojo - build-only Ernie fine-tune setup surface.
#
# Source of truth: /home/alex/Serenity/modules/modelSetup/ErnieFineTuneSetup.py

from serenity_trainer.modelSetup.BaseErnieSetup import (
    ernie_train_device_plan,
    ErnieTrainDevicePlan,
)
from serenity_trainer.util.enum.ModelType import MODEL_TYPE_ERNIE
from serenity_trainer.util.enum.TrainingMethod import TM_FINE_TUNE


comptime ERNIE_FINE_TUNE_MODEL_TYPE = MODEL_TYPE_ERNIE
comptime ERNIE_FINE_TUNE_TRAINING_METHOD = TM_FINE_TUNE
comptime ERNIE_FINE_TUNE_TRANSFORMER_PART = "transformer"
comptime ERNIE_FINE_TUNE_TEXT_ENCODER_TRAINABLE = False
comptime ERNIE_FINE_TUNE_VAE_TRAINABLE = False
comptime ERNIE_FINE_TUNE_EMBEDDINGS_SUPPORTED = False


def ernie_fine_tune_parameter_group_names() -> List[String]:
    var names = List[String]()
    names.append(ERNIE_FINE_TUNE_TRANSFORMER_PART)
    return names^


struct ErnieFineTuneSetupRegistration(Copyable, Movable, ImplicitlyCopyable):
    var model_type: Int
    var training_method: Int
    var trains_text_encoder: Bool
    var trains_transformer: Bool
    var trains_vae: Bool
    var supports_embedding_training: Bool

    def __init__(out self):
        self.model_type = ERNIE_FINE_TUNE_MODEL_TYPE
        self.training_method = ERNIE_FINE_TUNE_TRAINING_METHOD
        self.trains_text_encoder = ERNIE_FINE_TUNE_TEXT_ENCODER_TRAINABLE
        self.trains_transformer = True
        self.trains_vae = ERNIE_FINE_TUNE_VAE_TRAINABLE
        self.supports_embedding_training = ERNIE_FINE_TUNE_EMBEDDINGS_SUPPORTED


def ernie_fine_tune_setup_registration() -> ErnieFineTuneSetupRegistration:
    return ErnieFineTuneSetupRegistration()


struct ErnieFineTuneSetup(Copyable, Movable, ImplicitlyCopyable):
    var registration: ErnieFineTuneSetupRegistration

    def __init__(out self):
        self.registration = ernie_fine_tune_setup_registration()

    def create_parameters(self) -> List[String]:
        return ernie_fine_tune_parameter_group_names()

    def train_device_plan(
        self, latent_caching: Bool, transformer_train: Bool
    ) -> ErnieTrainDevicePlan:
        return ernie_train_device_plan(latent_caching, transformer_train)

    def uses_module_filter_for_transformer(self) -> Bool:
        return True

    def freezes_text_encoder(self) -> Bool:
        return True

    def freezes_vae(self) -> Bool:
        return True

    def after_optimizer_step_reapplies_requires_grad(self) -> Bool:
        return True

# QwenFineTuneSetup.mojo — build-only registration/setup surface.
#
# Source of truth: /home/alex/Serenity/modules/modelSetup/QwenFineTuneSetup.py

from serenity_trainer.util.enum.ModelType import MODEL_TYPE_QWEN
from serenity_trainer.util.enum.TrainingMethod import TM_FINE_TUNE


comptime QWEN_FINE_TUNE_MODEL_TYPE = MODEL_TYPE_QWEN
comptime QWEN_FINE_TUNE_TRAINING_METHOD = TM_FINE_TUNE
comptime QWEN_FINE_TUNE_TEXT_ENCODER_PART = "text_encoder"
comptime QWEN_FINE_TUNE_TRANSFORMER_PART = "transformer"
comptime QWEN_FINE_TUNE_VAE_TRAINABLE = False
comptime QWEN_FINE_TUNE_EMBEDDINGS_SUPPORTED = False


def qwen_fine_tune_parameter_group_names() -> List[String]:
    var names = List[String]()
    names.append(QWEN_FINE_TUNE_TEXT_ENCODER_PART)
    names.append(QWEN_FINE_TUNE_TRANSFORMER_PART)
    return names^


struct QwenFineTuneSetupRegistration(Copyable, Movable, ImplicitlyCopyable):
    var model_type: Int
    var training_method: Int
    var trains_text_encoder: Bool
    var trains_transformer: Bool
    var trains_vae: Bool
    var supports_embedding_training: Bool

    def __init__(out self):
        self.model_type = QWEN_FINE_TUNE_MODEL_TYPE
        self.training_method = QWEN_FINE_TUNE_TRAINING_METHOD
        self.trains_text_encoder = True
        self.trains_transformer = True
        self.trains_vae = QWEN_FINE_TUNE_VAE_TRAINABLE
        self.supports_embedding_training = QWEN_FINE_TUNE_EMBEDDINGS_SUPPORTED


def qwen_fine_tune_setup_registration() -> QwenFineTuneSetupRegistration:
    return QwenFineTuneSetupRegistration()


struct QwenFineTuneSetup(Copyable, Movable, ImplicitlyCopyable):
    var registration: QwenFineTuneSetupRegistration

    def __init__(out self):
        self.registration = qwen_fine_tune_setup_registration()

    def create_parameters(self) -> List[String]:
        return qwen_fine_tune_parameter_group_names()

    def uses_module_filter_for_transformer(self) -> Bool:
        return True

    def freezes_vae(self) -> Bool:
        return True

    def after_optimizer_step_reapplies_requires_grad(self) -> Bool:
        return True

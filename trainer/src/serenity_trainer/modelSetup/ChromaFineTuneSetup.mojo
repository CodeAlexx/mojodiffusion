# ChromaFineTuneSetup.mojo - build-only Chroma fine-tune setup surface.
#
# Source of truth:
#   /home/alex/Serenity/modules/modelSetup/ChromaFineTuneSetup.py
#
# This records Serenity's Chroma fine-tune registration, parameter groups,
# setup_model side effects, requires-grad plan, train-device plan, and
# after_optimizer_step behavior. Runtime model execution, gradients, and
# optimizer construction are intentionally out of scope.

from serenity_trainer.modelSetup.BaseChromaSetup import (
    CHROMA_PART_EMBEDDINGS,
    CHROMA_PART_TEXT_ENCODER,
    CHROMA_PART_TRANSFORMER,
    ChromaTrainDevicePlan,
    chroma_train_device_plan,
)
from serenity_trainer.util.enum.ModelType import MODEL_TYPE_CHROMA_1
from serenity_trainer.util.enum.TrainingMethod import TM_FINE_TUNE


comptime CHROMA_FINE_TUNE_MODEL_TYPE = MODEL_TYPE_CHROMA_1
comptime CHROMA_FINE_TUNE_TRAINING_METHOD = TM_FINE_TUNE
comptime CHROMA_FINE_TUNE_TEXT_ENCODER_PART = "text_encoder"
comptime CHROMA_FINE_TUNE_TRANSFORMER_PART = "transformer"
comptime CHROMA_FINE_TUNE_EMBEDDINGS_PART = "embeddings"


def chroma_fine_tune_parameter_group_names(
    train_any_embedding_or_output: Bool = False,
    train_text_encoder_embedding: Bool = False,
    has_text_encoder: Bool = True,
) -> List[String]:
    """Serenity ChromaFineTuneSetup.create_parameters group order."""
    var names = List[String]()
    names.append(CHROMA_FINE_TUNE_TEXT_ENCODER_PART)
    if train_any_embedding_or_output:
        if train_text_encoder_embedding and has_text_encoder:
            names.append(CHROMA_FINE_TUNE_EMBEDDINGS_PART)
    names.append(CHROMA_FINE_TUNE_TRANSFORMER_PART)
    return names^


struct ChromaFineTuneSetupRegistration(Copyable, Movable, ImplicitlyCopyable):
    var model_type: Int
    var training_method: Int
    var trains_text_encoder: Bool
    var trains_transformer: Bool
    var trains_vae: Bool
    var supports_embedding_training: Bool
    var supports_output_embedding_training: Bool

    def __init__(out self):
        self.model_type = CHROMA_FINE_TUNE_MODEL_TYPE
        self.training_method = CHROMA_FINE_TUNE_TRAINING_METHOD
        self.trains_text_encoder = True
        self.trains_transformer = True
        self.trains_vae = False
        self.supports_embedding_training = True
        self.supports_output_embedding_training = True


struct ChromaFineTuneSetupModelPlan(Copyable, Movable, ImplicitlyCopyable):
    var moves_input_embeddings_to_embedding_weight_dtype: Bool
    var removes_added_embeddings_from_tokenizer: Bool
    var setups_embeddings: Bool
    var setups_embedding_wrapper: Bool
    var initializes_model_parameters: Bool
    var uses_module_filter_for_transformer: Bool
    var uses_debug_flag_for_transformer_filter: Bool

    def __init__(out self, train_any_embedding: Bool):
        self.moves_input_embeddings_to_embedding_weight_dtype = train_any_embedding
        self.removes_added_embeddings_from_tokenizer = True
        self.setups_embeddings = True
        self.setups_embedding_wrapper = True
        self.initializes_model_parameters = True
        self.uses_module_filter_for_transformer = True
        self.uses_debug_flag_for_transformer_filter = True


struct ChromaFineTuneRequiresGradPlan(Copyable, Movable, ImplicitlyCopyable):
    var setup_embeddings_requires_grad: Bool
    var applies_text_encoder_config: Bool
    var applies_transformer_config: Bool
    var freezes_vae: Bool
    var transformer_uses_module_filter: Bool

    def __init__(out self):
        self.setup_embeddings_requires_grad = True
        self.applies_text_encoder_config = True
        self.applies_transformer_config = True
        self.freezes_vae = True
        self.transformer_uses_module_filter = True


def chroma_fine_tune_setup_registration() -> ChromaFineTuneSetupRegistration:
    return ChromaFineTuneSetupRegistration()


def chroma_fine_tune_setup_model_plan(
    train_any_embedding: Bool,
) -> ChromaFineTuneSetupModelPlan:
    return ChromaFineTuneSetupModelPlan(train_any_embedding)


struct ChromaFineTuneSetup(Movable):
    var registration: ChromaFineTuneSetupRegistration

    def __init__(out self):
        self.registration = chroma_fine_tune_setup_registration()

    def create_parameters(
        self,
        train_any_embedding_or_output: Bool = False,
        train_text_encoder_embedding: Bool = False,
        has_text_encoder: Bool = True,
    ) -> List[String]:
        return chroma_fine_tune_parameter_group_names(
            train_any_embedding_or_output,
            train_text_encoder_embedding,
            has_text_encoder,
        )

    def setup_model_plan(
        self, train_any_embedding: Bool
    ) -> ChromaFineTuneSetupModelPlan:
        return chroma_fine_tune_setup_model_plan(train_any_embedding)

    def requires_grad_plan(self) -> ChromaFineTuneRequiresGradPlan:
        return ChromaFineTuneRequiresGradPlan()

    def train_device_plan(
        self,
        latent_caching: Bool,
        train_text_encoder_or_embedding: Bool,
        text_encoder_train: Bool,
        transformer_train: Bool,
    ) -> ChromaTrainDevicePlan:
        return chroma_train_device_plan(
            latent_caching,
            train_text_encoder_or_embedding,
            text_encoder_train,
            transformer_train,
        )

    def trainable_model_part_names(self) -> List[String]:
        var names = List[String]()
        names.append(CHROMA_PART_TEXT_ENCODER)
        names.append(CHROMA_PART_TRANSFORMER)
        return names^

    def embedding_parameter_group_name(self) -> String:
        return String(CHROMA_PART_EMBEDDINGS)

    def uses_module_filter_for_transformer(self) -> Bool:
        return True

    def freezes_vae(self) -> Bool:
        return True

    def normalizes_embeddings_after_optimizer_step(self) -> Bool:
        return True

    def normalizes_embedding_wrapper_after_optimizer_step(self) -> Bool:
        return True

    def after_optimizer_step_reapplies_requires_grad(self) -> Bool:
        return True

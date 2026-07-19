# FluxFineTuneSetup.mojo - build-only FLUX.1 Dev fine-tune setup surface.
#
# Source of truth:
#   /home/alex/Serenity/modules/modelSetup/FluxFineTuneSetup.py
#
# This file records Serenity's FLUX fine-tune registration, parameter groups,
# setup_model side effects, requires-grad plan, and train-device plan. Runtime
# model execution and optimizer construction are intentionally out of scope.

from serenity_trainer.modelSetup.BaseFluxSetup import (
    FLUX_PART_TRANSFORMER,
    FLUX_PART_TEXT_ENCODER_1,
    FLUX_PART_TEXT_ENCODER_2,
    FluxTrainDevicePlan,
    flux_setup_model_types,
    flux_train_device_plan,
)
from serenity_trainer.util.enum.TrainingMethod import TM_FINE_TUNE


comptime FLUX_FINE_TUNE_TRAINING_METHOD = TM_FINE_TUNE
comptime FLUX_FINE_TUNE_TEXT_ENCODER_1_PART = "text_encoder_1"
comptime FLUX_FINE_TUNE_TEXT_ENCODER_2_PART = "text_encoder_2"
comptime FLUX_FINE_TUNE_TRANSFORMER_PART = "transformer"
comptime FLUX_FINE_TUNE_EMBEDDINGS_1_PART = "embeddings_1"
comptime FLUX_FINE_TUNE_EMBEDDINGS_2_PART = "embeddings_2"


def flux_fine_tune_registered_model_types() -> List[Int]:
    return flux_setup_model_types()


def flux_fine_tune_parameter_group_names(
    train_any_embedding_or_output: Bool = False,
    train_embedding_1: Bool = False,
    train_embedding_2: Bool = False,
    has_text_encoder_1: Bool = True,
    has_text_encoder_2: Bool = True,
) -> List[String]:
    """Serenity FluxFineTuneSetup.create_parameters group order."""
    var names = List[String]()
    names.append(FLUX_FINE_TUNE_TEXT_ENCODER_1_PART)
    names.append(FLUX_FINE_TUNE_TEXT_ENCODER_2_PART)
    if train_any_embedding_or_output:
        if train_embedding_1 and has_text_encoder_1:
            names.append(FLUX_FINE_TUNE_EMBEDDINGS_1_PART)
        if train_embedding_2 and has_text_encoder_2:
            names.append(FLUX_FINE_TUNE_EMBEDDINGS_2_PART)
    names.append(FLUX_FINE_TUNE_TRANSFORMER_PART)
    return names^


struct FluxFineTuneSetupRegistration(Movable):
    var model_types: List[Int]
    var training_method: Int
    var trains_text_encoder_1: Bool
    var trains_text_encoder_2: Bool
    var trains_transformer: Bool
    var trains_vae: Bool
    var supports_embedding_training: Bool
    var supports_output_embedding_training: Bool

    def __init__(out self):
        self.model_types = flux_fine_tune_registered_model_types()
        self.training_method = FLUX_FINE_TUNE_TRAINING_METHOD
        self.trains_text_encoder_1 = True
        self.trains_text_encoder_2 = True
        self.trains_transformer = True
        self.trains_vae = False
        self.supports_embedding_training = True
        self.supports_output_embedding_training = True


struct FluxFineTuneSetupModelPlan(Copyable, Movable, ImplicitlyCopyable):
    var moves_input_embeddings_to_embedding_weight_dtype: Bool
    var removes_added_embeddings_from_tokenizers: Bool
    var setups_embeddings: Bool
    var setups_embedding_wrappers: Bool
    var initializes_model_parameters: Bool
    var uses_module_filter_for_transformer: Bool
    var uses_debug_flag_for_transformer_filter: Bool

    def __init__(out self, train_any_embedding: Bool):
        self.moves_input_embeddings_to_embedding_weight_dtype = train_any_embedding
        self.removes_added_embeddings_from_tokenizers = True
        self.setups_embeddings = True
        self.setups_embedding_wrappers = True
        self.initializes_model_parameters = True
        self.uses_module_filter_for_transformer = True
        self.uses_debug_flag_for_transformer_filter = True


struct FluxFineTuneRequiresGradPlan(Copyable, Movable, ImplicitlyCopyable):
    var setup_embeddings_requires_grad: Bool
    var applies_text_encoder_1_config: Bool
    var applies_text_encoder_2_config: Bool
    var applies_transformer_config: Bool
    var freezes_vae: Bool
    var transformer_uses_module_filter: Bool

    def __init__(out self):
        self.setup_embeddings_requires_grad = True
        self.applies_text_encoder_1_config = True
        self.applies_text_encoder_2_config = True
        self.applies_transformer_config = True
        self.freezes_vae = True
        self.transformer_uses_module_filter = True


def flux_fine_tune_setup_registration() -> FluxFineTuneSetupRegistration:
    return FluxFineTuneSetupRegistration()


def flux_fine_tune_setup_model_plan(
    train_any_embedding: Bool
) -> FluxFineTuneSetupModelPlan:
    return FluxFineTuneSetupModelPlan(train_any_embedding)


struct FluxFineTuneSetup(Movable):
    var registration: FluxFineTuneSetupRegistration

    def __init__(out self):
        self.registration = flux_fine_tune_setup_registration()

    def create_parameters(
        self,
        train_any_embedding_or_output: Bool = False,
        train_embedding_1: Bool = False,
        train_embedding_2: Bool = False,
        has_text_encoder_1: Bool = True,
        has_text_encoder_2: Bool = True,
    ) -> List[String]:
        return flux_fine_tune_parameter_group_names(
            train_any_embedding_or_output,
            train_embedding_1,
            train_embedding_2,
            has_text_encoder_1,
            has_text_encoder_2,
        )

    def setup_model_plan(
        self, train_any_embedding: Bool
    ) -> FluxFineTuneSetupModelPlan:
        return flux_fine_tune_setup_model_plan(train_any_embedding)

    def requires_grad_plan(self) -> FluxFineTuneRequiresGradPlan:
        return FluxFineTuneRequiresGradPlan()

    def train_device_plan(
        self,
        latent_caching: Bool,
        train_text_encoder_or_embedding: Bool,
        train_text_encoder_2_or_embedding: Bool,
        text_encoder_1_train: Bool,
        text_encoder_2_train: Bool,
        transformer_train: Bool,
    ) -> FluxTrainDevicePlan:
        return flux_train_device_plan(
            latent_caching,
            train_text_encoder_or_embedding,
            train_text_encoder_2_or_embedding,
            text_encoder_1_train,
            text_encoder_2_train,
            transformer_train,
        )

    def trainable_model_part_names(self) -> List[String]:
        var names = List[String]()
        names.append(FLUX_PART_TEXT_ENCODER_1)
        names.append(FLUX_PART_TEXT_ENCODER_2)
        names.append(FLUX_PART_TRANSFORMER)
        return names^

    def uses_module_filter_for_transformer(self) -> Bool:
        return True

    def freezes_vae(self) -> Bool:
        return True

    def normalizes_embeddings_after_optimizer_step(self) -> Bool:
        return True

    def after_optimizer_step_reapplies_requires_grad(self) -> Bool:
        return True

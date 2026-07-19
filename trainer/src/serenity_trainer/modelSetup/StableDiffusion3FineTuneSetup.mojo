# StableDiffusion3FineTuneSetup.mojo - build-only SD3/SD3.5 fine-tune surface.
#
# Source of truth:
#   /home/alex/Serenity/modules/modelSetup/StableDiffusion3FineTuneSetup.py

from serenity_trainer.modelSetup.BaseStableDiffusion3Setup import (
    SD3_PART_TRANSFORMER,
    SD3_PART_TEXT_ENCODER_1,
    SD3_PART_TEXT_ENCODER_2,
    SD3_PART_TEXT_ENCODER_3,
    sd3_setup_model_types,
    sd3_train_device_plan,
    SD3TrainDevicePlan,
)
from serenity_trainer.util.enum.TrainingMethod import TM_FINE_TUNE


comptime SD3_FINE_TUNE_TRAINING_METHOD = TM_FINE_TUNE
comptime SD3_FINE_TUNE_TEXT_ENCODER_1_PART = "text_encoder_1"
comptime SD3_FINE_TUNE_TEXT_ENCODER_2_PART = "text_encoder_2"
comptime SD3_FINE_TUNE_TEXT_ENCODER_3_PART = "text_encoder_3"
comptime SD3_FINE_TUNE_TRANSFORMER_PART = "transformer"
comptime SD3_FINE_TUNE_EMBEDDINGS_1_PART = "embeddings_1"
comptime SD3_FINE_TUNE_EMBEDDINGS_2_PART = "embeddings_2"
comptime SD3_FINE_TUNE_EMBEDDINGS_3_PART = "embeddings_3"


def sd3_fine_tune_registered_model_types() -> List[Int]:
    return sd3_setup_model_types()


def sd3_fine_tune_parameter_group_names(
    train_any_embedding_or_output: Bool = False,
    train_embedding_1: Bool = False,
    train_embedding_2: Bool = False,
    train_embedding_3: Bool = False,
    has_text_encoder_1: Bool = True,
    has_text_encoder_2: Bool = True,
    has_text_encoder_3: Bool = True,
) -> List[String]:
    var names = List[String]()
    names.append(SD3_FINE_TUNE_TEXT_ENCODER_1_PART)
    names.append(SD3_FINE_TUNE_TEXT_ENCODER_2_PART)
    names.append(SD3_FINE_TUNE_TEXT_ENCODER_3_PART)
    if train_any_embedding_or_output:
        if train_embedding_1 and has_text_encoder_1:
            names.append(SD3_FINE_TUNE_EMBEDDINGS_1_PART)
        if train_embedding_2 and has_text_encoder_2:
            names.append(SD3_FINE_TUNE_EMBEDDINGS_2_PART)
        if train_embedding_3 and has_text_encoder_3:
            names.append(SD3_FINE_TUNE_EMBEDDINGS_3_PART)
    names.append(SD3_FINE_TUNE_TRANSFORMER_PART)
    return names^


struct SD3FineTuneSetupRegistration(Movable):
    var model_types: List[Int]
    var training_method: Int
    var trains_text_encoder_1: Bool
    var trains_text_encoder_2: Bool
    var trains_text_encoder_3: Bool
    var trains_transformer: Bool
    var trains_vae: Bool
    var supports_embedding_training: Bool
    var supports_output_embedding_training: Bool

    def __init__(out self):
        self.model_types = sd3_fine_tune_registered_model_types()
        self.training_method = SD3_FINE_TUNE_TRAINING_METHOD
        self.trains_text_encoder_1 = True
        self.trains_text_encoder_2 = True
        self.trains_text_encoder_3 = True
        self.trains_transformer = True
        self.trains_vae = False
        self.supports_embedding_training = True
        self.supports_output_embedding_training = True


struct SD3FineTuneSetupModelPlan(Copyable, Movable, ImplicitlyCopyable):
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


def sd3_fine_tune_setup_registration() -> SD3FineTuneSetupRegistration:
    return SD3FineTuneSetupRegistration()


def sd3_fine_tune_setup_model_plan(
    train_any_embedding: Bool
) -> SD3FineTuneSetupModelPlan:
    return SD3FineTuneSetupModelPlan(train_any_embedding)


struct StableDiffusion3FineTuneSetup(Movable):
    var registration: SD3FineTuneSetupRegistration

    def __init__(out self):
        self.registration = sd3_fine_tune_setup_registration()

    def create_parameters(
        self,
        train_any_embedding_or_output: Bool = False,
        train_embedding_1: Bool = False,
        train_embedding_2: Bool = False,
        train_embedding_3: Bool = False,
        has_text_encoder_1: Bool = True,
        has_text_encoder_2: Bool = True,
        has_text_encoder_3: Bool = True,
    ) -> List[String]:
        return sd3_fine_tune_parameter_group_names(
            train_any_embedding_or_output,
            train_embedding_1,
            train_embedding_2,
            train_embedding_3,
            has_text_encoder_1,
            has_text_encoder_2,
            has_text_encoder_3,
        )

    def setup_model_plan(
        self, train_any_embedding: Bool
    ) -> SD3FineTuneSetupModelPlan:
        return sd3_fine_tune_setup_model_plan(train_any_embedding)

    def train_device_plan(
        self,
        latent_caching: Bool,
        train_text_encoder_or_embedding: Bool,
        train_text_encoder_2_or_embedding: Bool,
        train_text_encoder_3_or_embedding: Bool,
        text_encoder_1_train: Bool,
        text_encoder_2_train: Bool,
        text_encoder_3_train: Bool,
        transformer_train: Bool,
    ) -> SD3TrainDevicePlan:
        return sd3_train_device_plan(
            latent_caching,
            train_text_encoder_or_embedding,
            train_text_encoder_2_or_embedding,
            train_text_encoder_3_or_embedding,
            text_encoder_1_train,
            text_encoder_2_train,
            text_encoder_3_train,
            transformer_train,
        )

    def trainable_model_part_names(self) -> List[String]:
        var names = List[String]()
        names.append(SD3_PART_TEXT_ENCODER_1)
        names.append(SD3_PART_TEXT_ENCODER_2)
        names.append(SD3_PART_TEXT_ENCODER_3)
        names.append(SD3_PART_TRANSFORMER)
        return names^

    def uses_module_filter_for_transformer(self) -> Bool:
        return True

    def freezes_vae(self) -> Bool:
        return True

    def normalizes_embeddings_after_optimizer_step(self) -> Bool:
        return True

    def after_optimizer_step_reapplies_requires_grad(self) -> Bool:
        return True

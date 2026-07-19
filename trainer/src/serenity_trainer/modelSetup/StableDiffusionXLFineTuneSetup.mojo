# StableDiffusionXLFineTuneSetup.mojo - build-only SDXL fine-tune surface.
#
# Source of truth:
#   /home/alex/Serenity/modules/modelSetup/StableDiffusionXLFineTuneSetup.py

from serenity_trainer.modelSetup.BaseStableDiffusionXLSetup import (
    SDXL_PART_TEXT_ENCODER_1,
    SDXL_PART_TEXT_ENCODER_2,
    SDXL_PART_UNET,
    SDXLTrainDevicePlan,
    sdxl_setup_model_types,
    sdxl_train_device_plan,
)
from serenity_trainer.util.enum.TrainingMethod import TM_FINE_TUNE


comptime SDXL_FINE_TUNE_TRAINING_METHOD = TM_FINE_TUNE
comptime SDXL_FINE_TUNE_TEXT_ENCODER_1_PART = "text_encoder_1"
comptime SDXL_FINE_TUNE_TEXT_ENCODER_2_PART = "text_encoder_2"
comptime SDXL_FINE_TUNE_UNET_PART = "unet"
comptime SDXL_FINE_TUNE_EMBEDDINGS_1_PART = "embeddings_1"
comptime SDXL_FINE_TUNE_EMBEDDINGS_2_PART = "embeddings_2"


def sdxl_fine_tune_registered_model_types() -> List[Int]:
    return sdxl_setup_model_types()


def sdxl_fine_tune_parameter_group_names(
    train_any_embedding_or_output: Bool = False,
    train_embedding_1: Bool = False,
    train_embedding_2: Bool = False,
    has_text_encoder_1: Bool = True,
    has_text_encoder_2: Bool = True,
) -> List[String]:
    var names = List[String]()
    names.append(SDXL_FINE_TUNE_TEXT_ENCODER_1_PART)
    names.append(SDXL_FINE_TUNE_TEXT_ENCODER_2_PART)
    if train_any_embedding_or_output:
        if train_embedding_1 and has_text_encoder_1:
            names.append(SDXL_FINE_TUNE_EMBEDDINGS_1_PART)
        if train_embedding_2 and has_text_encoder_2:
            names.append(SDXL_FINE_TUNE_EMBEDDINGS_2_PART)
    names.append(SDXL_FINE_TUNE_UNET_PART)
    return names^


struct SDXLFineTuneSetupRegistration(Movable):
    var model_types: List[Int]
    var training_method: Int
    var trains_text_encoder_1: Bool
    var trains_text_encoder_2: Bool
    var trains_unet: Bool
    var trains_vae: Bool
    var supports_embedding_training: Bool
    var supports_output_embedding_training: Bool

    def __init__(out self):
        self.model_types = sdxl_fine_tune_registered_model_types()
        self.training_method = SDXL_FINE_TUNE_TRAINING_METHOD
        self.trains_text_encoder_1 = True
        self.trains_text_encoder_2 = True
        self.trains_unet = True
        self.trains_vae = False
        self.supports_embedding_training = True
        self.supports_output_embedding_training = True


struct SDXLFineTuneSetupModelPlan(Copyable, Movable, ImplicitlyCopyable):
    var moves_input_embeddings_to_embedding_weight_dtype: Bool
    var rescales_noise_scheduler_to_zero_terminal_snr: Bool
    var forces_v_prediction_after_rescale: Bool
    var forces_v_prediction: Bool
    var forces_epsilon_prediction: Bool
    var removes_added_embeddings_from_tokenizers: Bool
    var setups_embeddings: Bool
    var setups_embedding_wrappers: Bool
    var initializes_model_parameters: Bool
    var uses_module_filter_for_unet: Bool
    var uses_debug_flag_for_unet_filter: Bool

    def __init__(
        out self,
        train_any_embedding: Bool,
        rescale_noise_scheduler_to_zero_terminal_snr: Bool,
        force_v_prediction: Bool,
        force_epsilon_prediction: Bool,
    ):
        self.moves_input_embeddings_to_embedding_weight_dtype = train_any_embedding
        self.rescales_noise_scheduler_to_zero_terminal_snr = (
            rescale_noise_scheduler_to_zero_terminal_snr
        )
        self.forces_v_prediction_after_rescale = (
            rescale_noise_scheduler_to_zero_terminal_snr
        )
        self.forces_v_prediction = (
            (not rescale_noise_scheduler_to_zero_terminal_snr)
            and force_v_prediction
        )
        self.forces_epsilon_prediction = (
            (not rescale_noise_scheduler_to_zero_terminal_snr)
            and (not force_v_prediction)
            and force_epsilon_prediction
        )
        self.removes_added_embeddings_from_tokenizers = True
        self.setups_embeddings = True
        self.setups_embedding_wrappers = True
        self.initializes_model_parameters = True
        self.uses_module_filter_for_unet = True
        self.uses_debug_flag_for_unet_filter = True


def sdxl_fine_tune_setup_registration() -> SDXLFineTuneSetupRegistration:
    return SDXLFineTuneSetupRegistration()


def sdxl_fine_tune_setup_model_plan(
    train_any_embedding: Bool,
    rescale_noise_scheduler_to_zero_terminal_snr: Bool,
    force_v_prediction: Bool,
    force_epsilon_prediction: Bool,
) -> SDXLFineTuneSetupModelPlan:
    return SDXLFineTuneSetupModelPlan(
        train_any_embedding,
        rescale_noise_scheduler_to_zero_terminal_snr,
        force_v_prediction,
        force_epsilon_prediction,
    )


struct StableDiffusionXLFineTuneSetup(Movable):
    var registration: SDXLFineTuneSetupRegistration

    def __init__(out self):
        self.registration = sdxl_fine_tune_setup_registration()

    def create_parameters(
        self,
        train_any_embedding_or_output: Bool = False,
        train_embedding_1: Bool = False,
        train_embedding_2: Bool = False,
        has_text_encoder_1: Bool = True,
        has_text_encoder_2: Bool = True,
    ) -> List[String]:
        return sdxl_fine_tune_parameter_group_names(
            train_any_embedding_or_output,
            train_embedding_1,
            train_embedding_2,
            has_text_encoder_1,
            has_text_encoder_2,
        )

    def setup_model_plan(
        self,
        train_any_embedding: Bool,
        rescale_noise_scheduler_to_zero_terminal_snr: Bool,
        force_v_prediction: Bool,
        force_epsilon_prediction: Bool,
    ) -> SDXLFineTuneSetupModelPlan:
        return sdxl_fine_tune_setup_model_plan(
            train_any_embedding,
            rescale_noise_scheduler_to_zero_terminal_snr,
            force_v_prediction,
            force_epsilon_prediction,
        )

    def train_device_plan(
        self,
        latent_caching: Bool,
        text_encoder_1_train: Bool,
        text_encoder_2_train: Bool,
        train_any_embedding: Bool,
        unet_train: Bool,
    ) -> SDXLTrainDevicePlan:
        return sdxl_train_device_plan(
            latent_caching,
            text_encoder_1_train or train_any_embedding,
            text_encoder_2_train or train_any_embedding,
            text_encoder_1_train,
            text_encoder_2_train,
            unet_train,
            True,
        )

    def trainable_model_part_names(self) -> List[String]:
        var names = List[String]()
        names.append(SDXL_PART_TEXT_ENCODER_1)
        names.append(SDXL_PART_TEXT_ENCODER_2)
        names.append(SDXL_PART_UNET)
        return names^

    def uses_module_filter_for_unet(self) -> Bool:
        return True

    def freezes_vae(self) -> Bool:
        return True

    def vae_is_put_in_train_mode(self) -> Bool:
        return True

    def normalizes_embeddings_after_optimizer_step(self) -> Bool:
        return True

    def after_optimizer_step_reapplies_requires_grad(self) -> Bool:
        return True

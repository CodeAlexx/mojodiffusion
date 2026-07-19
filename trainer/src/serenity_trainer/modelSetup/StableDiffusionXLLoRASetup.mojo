# StableDiffusionXLLoRASetup.mojo - build-only SDXL LoRA setup surface.
#
# Source of truth:
#   /home/alex/Serenity/modules/modelSetup/StableDiffusionXLLoRASetup.py

from serenity_trainer.modelSetup.BaseStableDiffusionXLSetup import (
    SDXL_PART_TEXT_ENCODER_1,
    SDXL_PART_TEXT_ENCODER_2,
    SDXL_PART_UNET,
    SDXLTrainDevicePlan,
    sdxl_layer_preset_filters,
    sdxl_setup_model_types,
    sdxl_train_device_plan,
)
from serenity_trainer.util.enum.TrainingMethod import TM_LORA


comptime SDXL_LORA_TRAINING_METHOD = TM_LORA
comptime SDXL_LORA_TEXT_ENCODER_1_PART = "text_encoder_1_lora"
comptime SDXL_LORA_TEXT_ENCODER_2_PART = "text_encoder_2_lora"
comptime SDXL_LORA_UNET_PART = "unet_lora"
comptime SDXL_LORA_EMBEDDINGS_1_PART = "embeddings_1"
comptime SDXL_LORA_EMBEDDINGS_2_PART = "embeddings_2"
comptime SDXL_LORA_PREFIX_TE1 = "lora_te1"
comptime SDXL_LORA_PREFIX_TE2 = "lora_te2"
comptime SDXL_LORA_PREFIX_UNET = "lora_unet"


def sdxl_lora_registered_model_types() -> List[Int]:
    return sdxl_setup_model_types()


def sdxl_lora_parameter_group_names(
    train_any_embedding_or_output: Bool = False,
    train_embedding_1: Bool = False,
    train_embedding_2: Bool = False,
    has_text_encoder_1: Bool = True,
    has_text_encoder_2: Bool = True,
) -> List[String]:
    var names = List[String]()
    names.append(SDXL_LORA_TEXT_ENCODER_1_PART)
    names.append(SDXL_LORA_TEXT_ENCODER_2_PART)
    if train_any_embedding_or_output:
        if train_embedding_1 and has_text_encoder_1:
            names.append(SDXL_LORA_EMBEDDINGS_1_PART)
        if train_embedding_2 and has_text_encoder_2:
            names.append(SDXL_LORA_EMBEDDINGS_2_PART)
    names.append(SDXL_LORA_UNET_PART)
    return names^


def sdxl_lora_state_dict_prefixes() -> List[String]:
    var prefixes = List[String]()
    prefixes.append(SDXL_LORA_PREFIX_TE1)
    prefixes.append(SDXL_LORA_PREFIX_TE2)
    prefixes.append(SDXL_LORA_PREFIX_UNET)
    return prefixes^


def sdxl_lora_layer_filters(layer_filter: String) raises -> List[String]:
    return sdxl_layer_preset_filters(layer_filter)


def sdxl_lora_unet_target_filters_expression() -> String:
    return "config.layer_filter.split(',')"


struct SDXLLoRASetupRegistration(Movable):
    var model_types: List[Int]
    var training_method: Int
    var creates_text_encoder_1_lora: Bool
    var creates_text_encoder_2_lora: Bool
    var creates_unet_lora: Bool
    var supports_embedding_training: Bool
    var supports_output_embedding_training: Bool

    def __init__(out self):
        self.model_types = sdxl_lora_registered_model_types()
        self.training_method = SDXL_LORA_TRAINING_METHOD
        self.creates_text_encoder_1_lora = True
        self.creates_text_encoder_2_lora = True
        self.creates_unet_lora = True
        self.supports_embedding_training = True
        self.supports_output_embedding_training = True


struct SDXLLoRACreationPlan(Movable):
    var create_text_encoder_1_lora: Bool
    var create_text_encoder_2_lora: Bool
    var create_unet_lora: Bool
    var wrapper_prefixes: List[String]
    var unet_target_filters_expression: String
    var loads_pending_state_dict: Bool
    var sets_text_encoder_1_dropout_from_config: Bool
    var sets_text_encoder_2_dropout_from_config: Bool
    var sets_unet_dropout_from_config: Bool
    var moves_text_encoder_1_lora_to_config_weight_dtype: Bool
    var moves_text_encoder_2_lora_to_config_weight_dtype: Bool
    var moves_unet_lora_to_config_weight_dtype: Bool
    var hooks_text_encoder_1_lora_to_module: Bool
    var hooks_text_encoder_2_lora_to_module: Bool
    var hooks_unet_lora_to_module: Bool
    var rescales_noise_scheduler_to_zero_terminal_snr: Bool
    var forces_v_prediction_after_rescale: Bool
    var removes_added_embeddings_from_tokenizers: Bool
    var setups_embeddings: Bool
    var setups_embedding_wrappers: Bool
    var initializes_model_parameters: Bool

    def __init__(
        out self,
        config_trains_text_encoder_1: Bool,
        config_trains_text_encoder_2: Bool,
        state_dict_has_lora_te1: Bool,
        state_dict_has_lora_te2: Bool,
        has_pending_lora_state_dict: Bool,
        rescale_noise_scheduler_to_zero_terminal_snr: Bool,
    ):
        self.create_text_encoder_1_lora = (
            config_trains_text_encoder_1 or state_dict_has_lora_te1
        )
        self.create_text_encoder_2_lora = (
            config_trains_text_encoder_2 or state_dict_has_lora_te2
        )
        self.create_unet_lora = True
        self.wrapper_prefixes = sdxl_lora_state_dict_prefixes()
        self.unet_target_filters_expression = sdxl_lora_unet_target_filters_expression()
        self.loads_pending_state_dict = has_pending_lora_state_dict
        self.sets_text_encoder_1_dropout_from_config = config_trains_text_encoder_1
        self.sets_text_encoder_2_dropout_from_config = config_trains_text_encoder_2
        self.sets_unet_dropout_from_config = True
        self.moves_text_encoder_1_lora_to_config_weight_dtype = (
            self.create_text_encoder_1_lora
        )
        self.moves_text_encoder_2_lora_to_config_weight_dtype = (
            self.create_text_encoder_2_lora
        )
        self.moves_unet_lora_to_config_weight_dtype = True
        self.hooks_text_encoder_1_lora_to_module = self.create_text_encoder_1_lora
        self.hooks_text_encoder_2_lora_to_module = self.create_text_encoder_2_lora
        self.hooks_unet_lora_to_module = True
        self.rescales_noise_scheduler_to_zero_terminal_snr = (
            rescale_noise_scheduler_to_zero_terminal_snr
        )
        self.forces_v_prediction_after_rescale = (
            rescale_noise_scheduler_to_zero_terminal_snr
        )
        self.removes_added_embeddings_from_tokenizers = True
        self.setups_embeddings = True
        self.setups_embedding_wrappers = True
        self.initializes_model_parameters = True


def sdxl_lora_setup_registration() -> SDXLLoRASetupRegistration:
    return SDXLLoRASetupRegistration()


def sdxl_lora_creation_plan(
    config_trains_text_encoder_1: Bool,
    config_trains_text_encoder_2: Bool,
    state_dict_has_lora_te1: Bool,
    state_dict_has_lora_te2: Bool,
    has_pending_lora_state_dict: Bool,
    rescale_noise_scheduler_to_zero_terminal_snr: Bool,
) -> SDXLLoRACreationPlan:
    return SDXLLoRACreationPlan(
        config_trains_text_encoder_1,
        config_trains_text_encoder_2,
        state_dict_has_lora_te1,
        state_dict_has_lora_te2,
        has_pending_lora_state_dict,
        rescale_noise_scheduler_to_zero_terminal_snr,
    )


struct StableDiffusionXLLoRASetup(Movable):
    var registration: SDXLLoRASetupRegistration

    def __init__(out self):
        self.registration = sdxl_lora_setup_registration()

    def create_parameters(
        self,
        train_any_embedding_or_output: Bool = False,
        train_embedding_1: Bool = False,
        train_embedding_2: Bool = False,
        has_text_encoder_1: Bool = True,
        has_text_encoder_2: Bool = True,
    ) -> List[String]:
        return sdxl_lora_parameter_group_names(
            train_any_embedding_or_output,
            train_embedding_1,
            train_embedding_2,
            has_text_encoder_1,
            has_text_encoder_2,
        )

    def creation_plan(
        self,
        config_trains_text_encoder_1: Bool,
        config_trains_text_encoder_2: Bool,
        state_dict_has_lora_te1: Bool,
        state_dict_has_lora_te2: Bool,
        has_pending_lora_state_dict: Bool,
        rescale_noise_scheduler_to_zero_terminal_snr: Bool,
    ) -> SDXLLoRACreationPlan:
        return sdxl_lora_creation_plan(
            config_trains_text_encoder_1,
            config_trains_text_encoder_2,
            state_dict_has_lora_te1,
            state_dict_has_lora_te2,
            has_pending_lora_state_dict,
            rescale_noise_scheduler_to_zero_terminal_snr,
        )

    def layer_filters(self, layer_filter: String) raises -> List[String]:
        return sdxl_lora_layer_filters(layer_filter)

    def train_device_plan(
        self,
        latent_caching: Bool,
        train_text_encoder_or_embedding: Bool,
        train_text_encoder_2_or_embedding: Bool,
        text_encoder_1_train: Bool,
        text_encoder_2_train: Bool,
        unet_train: Bool,
    ) -> SDXLTrainDevicePlan:
        return sdxl_train_device_plan(
            latent_caching,
            train_text_encoder_or_embedding,
            train_text_encoder_2_or_embedding,
            text_encoder_1_train,
            text_encoder_2_train,
            unet_train,
            False,
        )

    def freezes_base_text_encoder_1(self) -> Bool:
        return True

    def freezes_base_text_encoder_2(self) -> Bool:
        return True

    def freezes_base_unet(self) -> Bool:
        return True

    def freezes_vae(self) -> Bool:
        return True

    def trainable_model_part_names(self) -> List[String]:
        var names = List[String]()
        names.append(SDXL_LORA_TEXT_ENCODER_1_PART)
        names.append(SDXL_LORA_TEXT_ENCODER_2_PART)
        names.append(SDXL_LORA_UNET_PART)
        return names^

    def base_model_part_names(self) -> List[String]:
        var names = List[String]()
        names.append(SDXL_PART_TEXT_ENCODER_1)
        names.append(SDXL_PART_TEXT_ENCODER_2)
        names.append(SDXL_PART_UNET)
        return names^

    def removes_added_embeddings_from_tokenizers(self) -> Bool:
        return True

    def normalizes_embeddings_after_optimizer_step(self) -> Bool:
        return True

    def after_optimizer_step_reapplies_requires_grad(self) -> Bool:
        return True

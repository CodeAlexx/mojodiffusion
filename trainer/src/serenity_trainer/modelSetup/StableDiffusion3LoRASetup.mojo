# StableDiffusion3LoRASetup.mojo - build-only SD3/SD3.5 LoRA setup surface.
#
# Source of truth:
#   /home/alex/Serenity/modules/modelSetup/StableDiffusion3LoRASetup.py

from serenity_trainer.modelSetup.BaseStableDiffusion3Setup import (
    SD3_PART_TRANSFORMER,
    SD3_PART_TEXT_ENCODER_1,
    SD3_PART_TEXT_ENCODER_2,
    SD3_PART_TEXT_ENCODER_3,
    sd3_layer_preset_filters,
    sd3_setup_model_types,
    sd3_train_device_plan,
    SD3TrainDevicePlan,
)
from serenity_trainer.util.enum.TrainingMethod import TM_LORA


comptime SD3_LORA_TRAINING_METHOD = TM_LORA
comptime SD3_LORA_TEXT_ENCODER_1_PART = "text_encoder_1_lora"
comptime SD3_LORA_TEXT_ENCODER_2_PART = "text_encoder_2_lora"
comptime SD3_LORA_TEXT_ENCODER_3_PART = "text_encoder_3_lora"
comptime SD3_LORA_TRANSFORMER_PART = "transformer_lora"
comptime SD3_LORA_EMBEDDINGS_1_PART = "embeddings_1"
comptime SD3_LORA_EMBEDDINGS_2_PART = "embeddings_2"
comptime SD3_LORA_EMBEDDINGS_3_PART = "embeddings_3"
comptime SD3_LORA_PREFIX_TE1 = "lora_te1"
comptime SD3_LORA_PREFIX_TE2 = "lora_te2"
comptime SD3_LORA_PREFIX_TE3 = "lora_te3"
comptime SD3_LORA_PREFIX_TRANSFORMER = "lora_transformer"


def sd3_lora_registered_model_types() -> List[Int]:
    return sd3_setup_model_types()


def sd3_lora_parameter_group_names(
    train_any_embedding_or_output: Bool = False,
    train_embedding_1: Bool = False,
    train_embedding_2: Bool = False,
    train_embedding_3: Bool = False,
    has_text_encoder_1: Bool = True,
    has_text_encoder_2: Bool = True,
    has_text_encoder_3: Bool = True,
) -> List[String]:
    var names = List[String]()
    names.append(SD3_LORA_TEXT_ENCODER_1_PART)
    names.append(SD3_LORA_TEXT_ENCODER_2_PART)
    names.append(SD3_LORA_TEXT_ENCODER_3_PART)
    if train_any_embedding_or_output:
        if train_embedding_1 and has_text_encoder_1:
            names.append(SD3_LORA_EMBEDDINGS_1_PART)
        if train_embedding_2 and has_text_encoder_2:
            names.append(SD3_LORA_EMBEDDINGS_2_PART)
        if train_embedding_3 and has_text_encoder_3:
            names.append(SD3_LORA_EMBEDDINGS_3_PART)
    names.append(SD3_LORA_TRANSFORMER_PART)
    return names^


def sd3_lora_state_dict_prefixes() -> List[String]:
    var prefixes = List[String]()
    prefixes.append(SD3_LORA_PREFIX_TE1)
    prefixes.append(SD3_LORA_PREFIX_TE2)
    prefixes.append(SD3_LORA_PREFIX_TE3)
    prefixes.append(SD3_LORA_PREFIX_TRANSFORMER)
    return prefixes^


def sd3_lora_layer_filters(layer_filter: String) raises -> List[String]:
    return sd3_layer_preset_filters(layer_filter)


def sd3_lora_transformer_target_filters_expression() -> String:
    return "config.layer_filter.split(',')"


struct SD3LoRASetupRegistration(Movable):
    var model_types: List[Int]
    var training_method: Int
    var creates_text_encoder_1_lora: Bool
    var creates_text_encoder_2_lora: Bool
    var creates_text_encoder_3_lora: Bool
    var creates_transformer_lora: Bool
    var supports_embedding_training: Bool
    var supports_output_embedding_training: Bool

    def __init__(out self):
        self.model_types = sd3_lora_registered_model_types()
        self.training_method = SD3_LORA_TRAINING_METHOD
        self.creates_text_encoder_1_lora = True
        self.creates_text_encoder_2_lora = True
        self.creates_text_encoder_3_lora = True
        self.creates_transformer_lora = True
        self.supports_embedding_training = True
        self.supports_output_embedding_training = True


struct SD3LoRACreationPlan(Movable):
    var create_text_encoder_1_lora: Bool
    var create_text_encoder_2_lora: Bool
    var create_text_encoder_3_lora: Bool
    var create_transformer_lora: Bool
    var wrapper_prefixes: List[String]
    var transformer_target_filters_expression: String
    var loads_pending_state_dict: Bool
    var sets_dropout_from_config: Bool
    var moves_lora_to_config_weight_dtype: Bool
    var hooks_lora_to_module: Bool
    var moves_input_embeddings_to_embedding_weight_dtype: Bool
    var removes_added_embeddings_from_tokenizers: Bool
    var setups_embedding_wrappers: Bool

    def __init__(
        out self,
        has_text_encoder_1: Bool,
        has_text_encoder_2: Bool,
        has_text_encoder_3: Bool,
        config_trains_text_encoder_1: Bool,
        config_trains_text_encoder_2: Bool,
        config_trains_text_encoder_3: Bool,
        state_dict_has_lora_te1: Bool,
        state_dict_has_lora_te2: Bool,
        state_dict_has_lora_te3: Bool,
        has_pending_lora_state_dict: Bool,
        train_any_embedding: Bool,
    ):
        self.create_text_encoder_1_lora = has_text_encoder_1 and (
            config_trains_text_encoder_1 or state_dict_has_lora_te1
        )
        self.create_text_encoder_2_lora = has_text_encoder_2 and (
            config_trains_text_encoder_2 or state_dict_has_lora_te2
        )
        self.create_text_encoder_3_lora = has_text_encoder_3 and (
            config_trains_text_encoder_3 or state_dict_has_lora_te3
        )
        self.create_transformer_lora = True
        self.wrapper_prefixes = sd3_lora_state_dict_prefixes()
        self.transformer_target_filters_expression = (
            sd3_lora_transformer_target_filters_expression()
        )
        self.loads_pending_state_dict = has_pending_lora_state_dict
        self.sets_dropout_from_config = True
        self.moves_lora_to_config_weight_dtype = True
        self.hooks_lora_to_module = True
        self.moves_input_embeddings_to_embedding_weight_dtype = train_any_embedding
        self.removes_added_embeddings_from_tokenizers = True
        self.setups_embedding_wrappers = True


def sd3_lora_setup_registration() -> SD3LoRASetupRegistration:
    return SD3LoRASetupRegistration()


def sd3_lora_creation_plan(
    has_text_encoder_1: Bool,
    has_text_encoder_2: Bool,
    has_text_encoder_3: Bool,
    config_trains_text_encoder_1: Bool,
    config_trains_text_encoder_2: Bool,
    config_trains_text_encoder_3: Bool,
    state_dict_has_lora_te1: Bool,
    state_dict_has_lora_te2: Bool,
    state_dict_has_lora_te3: Bool,
    has_pending_lora_state_dict: Bool,
    train_any_embedding: Bool,
) -> SD3LoRACreationPlan:
    return SD3LoRACreationPlan(
        has_text_encoder_1,
        has_text_encoder_2,
        has_text_encoder_3,
        config_trains_text_encoder_1,
        config_trains_text_encoder_2,
        config_trains_text_encoder_3,
        state_dict_has_lora_te1,
        state_dict_has_lora_te2,
        state_dict_has_lora_te3,
        has_pending_lora_state_dict,
        train_any_embedding,
    )


struct StableDiffusion3LoRASetup(Movable):
    var registration: SD3LoRASetupRegistration

    def __init__(out self):
        self.registration = sd3_lora_setup_registration()

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
        return sd3_lora_parameter_group_names(
            train_any_embedding_or_output,
            train_embedding_1,
            train_embedding_2,
            train_embedding_3,
            has_text_encoder_1,
            has_text_encoder_2,
            has_text_encoder_3,
        )

    def creation_plan(
        self,
        has_text_encoder_1: Bool,
        has_text_encoder_2: Bool,
        has_text_encoder_3: Bool,
        config_trains_text_encoder_1: Bool,
        config_trains_text_encoder_2: Bool,
        config_trains_text_encoder_3: Bool,
        state_dict_has_lora_te1: Bool,
        state_dict_has_lora_te2: Bool,
        state_dict_has_lora_te3: Bool,
        has_pending_lora_state_dict: Bool,
        train_any_embedding: Bool,
    ) -> SD3LoRACreationPlan:
        return sd3_lora_creation_plan(
            has_text_encoder_1,
            has_text_encoder_2,
            has_text_encoder_3,
            config_trains_text_encoder_1,
            config_trains_text_encoder_2,
            config_trains_text_encoder_3,
            state_dict_has_lora_te1,
            state_dict_has_lora_te2,
            state_dict_has_lora_te3,
            has_pending_lora_state_dict,
            train_any_embedding,
        )

    def layer_filters(self, layer_filter: String) raises -> List[String]:
        return sd3_lora_layer_filters(layer_filter)

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

    def freezes_base_text_encoder_1(self) -> Bool:
        return True

    def freezes_base_text_encoder_2(self) -> Bool:
        return True

    def freezes_base_text_encoder_3(self) -> Bool:
        return True

    def freezes_base_transformer(self) -> Bool:
        return True

    def freezes_vae(self) -> Bool:
        return True

    def trainable_model_part_names(self) -> List[String]:
        var names = List[String]()
        names.append(SD3_LORA_TEXT_ENCODER_1_PART)
        names.append(SD3_LORA_TEXT_ENCODER_2_PART)
        names.append(SD3_LORA_TEXT_ENCODER_3_PART)
        names.append(SD3_LORA_TRANSFORMER_PART)
        return names^

    def base_model_part_names(self) -> List[String]:
        var names = List[String]()
        names.append(SD3_PART_TEXT_ENCODER_1)
        names.append(SD3_PART_TEXT_ENCODER_2)
        names.append(SD3_PART_TEXT_ENCODER_3)
        names.append(SD3_PART_TRANSFORMER)
        return names^

    def normalizes_embeddings_after_optimizer_step(self) -> Bool:
        return True

    def after_optimizer_step_reapplies_requires_grad(self) -> Bool:
        return True

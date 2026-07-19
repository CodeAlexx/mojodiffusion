# FluxLoRASetup.mojo - build-only FLUX.1 Dev LoRA setup surface.
#
# Source of truth:
#   /home/alex/Serenity/modules/modelSetup/FluxLoRASetup.py
#
# This file mirrors Serenity's setup/registration/parameter-group contract for
# FLUX.1 Dev and FLUX.1 Fill Dev. It records the LoRA wrapper prefixes, creation
# gates, optimizer group names, requires-grad plan, and train-device plan without
# instantiating runtime modules.

from serenity_trainer.modelSetup.BaseFluxSetup import (
    FLUX_PART_TRANSFORMER,
    FLUX_PART_TEXT_ENCODER_1,
    FLUX_PART_TEXT_ENCODER_2,
    FluxTrainDevicePlan,
    flux_layer_preset_filters,
    flux_setup_model_types,
    flux_train_device_plan,
)
from serenity_trainer.util.enum.TrainingMethod import TM_LORA


comptime FLUX_LORA_TRAINING_METHOD = TM_LORA
comptime FLUX_LORA_TEXT_ENCODER_1_PART = "text_encoder_1_lora"
comptime FLUX_LORA_TEXT_ENCODER_2_PART = "text_encoder_2_lora"
comptime FLUX_LORA_TRANSFORMER_PART = "transformer_lora"
comptime FLUX_LORA_EMBEDDINGS_1_PART = "embeddings_1"
comptime FLUX_LORA_EMBEDDINGS_2_PART = "embeddings_2"
comptime FLUX_LORA_PREFIX_TE1 = "lora_te1"
comptime FLUX_LORA_PREFIX_TE2 = "lora_te2"
comptime FLUX_LORA_PREFIX_TRANSFORMER = "lora_transformer"


def flux_lora_registered_model_types() -> List[Int]:
    return flux_setup_model_types()


def flux_lora_parameter_group_names(
    train_any_embedding_or_output: Bool = False,
    train_embedding_1: Bool = False,
    train_embedding_2: Bool = False,
    has_text_encoder_1: Bool = True,
    has_text_encoder_2: Bool = True,
) -> List[String]:
    """Serenity FluxLoRASetup.create_parameters group order."""
    var names = List[String]()
    names.append(FLUX_LORA_TEXT_ENCODER_1_PART)
    names.append(FLUX_LORA_TEXT_ENCODER_2_PART)
    if train_any_embedding_or_output:
        if train_embedding_1 and has_text_encoder_1:
            names.append(FLUX_LORA_EMBEDDINGS_1_PART)
        if train_embedding_2 and has_text_encoder_2:
            names.append(FLUX_LORA_EMBEDDINGS_2_PART)
    names.append(FLUX_LORA_TRANSFORMER_PART)
    return names^


def flux_lora_state_dict_prefixes() -> List[String]:
    var prefixes = List[String]()
    prefixes.append(FLUX_LORA_PREFIX_TE1)
    prefixes.append(FLUX_LORA_PREFIX_TE2)
    prefixes.append(FLUX_LORA_PREFIX_TRANSFORMER)
    return prefixes^


def flux_lora_layer_filters(layer_filter: String) raises -> List[String]:
    return flux_layer_preset_filters(layer_filter)


def flux_lora_transformer_target_filters_expression() -> String:
    return "config.layer_filter.split(',')"


struct FluxLoRASetupRegistration(Movable):
    var model_types: List[Int]
    var training_method: Int
    var creates_text_encoder_1_lora: Bool
    var creates_text_encoder_2_lora: Bool
    var creates_transformer_lora: Bool
    var supports_embedding_training: Bool
    var supports_output_embedding_training: Bool

    def __init__(out self):
        self.model_types = flux_lora_registered_model_types()
        self.training_method = FLUX_LORA_TRAINING_METHOD
        self.creates_text_encoder_1_lora = True
        self.creates_text_encoder_2_lora = True
        self.creates_transformer_lora = True
        self.supports_embedding_training = True
        self.supports_output_embedding_training = True


struct FluxLoRACreationPlan(Movable):
    var create_text_encoder_1_lora: Bool
    var create_text_encoder_2_lora: Bool
    var create_transformer_lora: Bool
    var wrapper_prefixes: List[String]
    var transformer_target_filters_expression: String
    var loads_pending_state_dict: Bool
    var clears_pending_state_dict_after_load: Bool
    var sets_dropout_from_config: Bool
    var moves_lora_to_config_weight_dtype: Bool
    var hooks_lora_to_module: Bool
    var moves_input_embeddings_to_embedding_weight_dtype: Bool
    var removes_added_embeddings_from_tokenizers: Bool
    var setups_embeddings: Bool
    var setups_embedding_wrappers: Bool
    var initializes_model_parameters: Bool

    def __init__(
        out self,
        has_text_encoder_1: Bool,
        has_text_encoder_2: Bool,
        config_trains_text_encoder_1: Bool,
        config_trains_text_encoder_2: Bool,
        state_dict_has_lora_te1: Bool,
        state_dict_has_lora_te2: Bool,
        has_pending_lora_state_dict: Bool,
        train_any_embedding: Bool,
    ):
        self.create_text_encoder_1_lora = has_text_encoder_1 and (
            config_trains_text_encoder_1 or state_dict_has_lora_te1
        )
        self.create_text_encoder_2_lora = has_text_encoder_2 and (
            config_trains_text_encoder_2 or state_dict_has_lora_te2
        )
        self.create_transformer_lora = True
        self.wrapper_prefixes = flux_lora_state_dict_prefixes()
        self.transformer_target_filters_expression = (
            flux_lora_transformer_target_filters_expression()
        )
        self.loads_pending_state_dict = has_pending_lora_state_dict
        self.clears_pending_state_dict_after_load = has_pending_lora_state_dict
        self.sets_dropout_from_config = True
        self.moves_lora_to_config_weight_dtype = True
        self.hooks_lora_to_module = True
        self.moves_input_embeddings_to_embedding_weight_dtype = train_any_embedding
        self.removes_added_embeddings_from_tokenizers = True
        self.setups_embeddings = True
        self.setups_embedding_wrappers = True
        self.initializes_model_parameters = True


struct FluxLoRARequiresGradPlan(Copyable, Movable, ImplicitlyCopyable):
    var setup_embeddings_requires_grad: Bool
    var freezes_text_encoder_1_base: Bool
    var freezes_text_encoder_2_base: Bool
    var freezes_transformer_base: Bool
    var freezes_vae: Bool
    var applies_text_encoder_1_lora_config: Bool
    var applies_text_encoder_2_lora_config: Bool
    var applies_transformer_lora_config: Bool

    def __init__(out self):
        self.setup_embeddings_requires_grad = True
        self.freezes_text_encoder_1_base = True
        self.freezes_text_encoder_2_base = True
        self.freezes_transformer_base = True
        self.freezes_vae = True
        self.applies_text_encoder_1_lora_config = True
        self.applies_text_encoder_2_lora_config = True
        self.applies_transformer_lora_config = True


def flux_lora_setup_registration() -> FluxLoRASetupRegistration:
    return FluxLoRASetupRegistration()


def flux_lora_creation_plan(
    has_text_encoder_1: Bool,
    has_text_encoder_2: Bool,
    config_trains_text_encoder_1: Bool,
    config_trains_text_encoder_2: Bool,
    state_dict_has_lora_te1: Bool,
    state_dict_has_lora_te2: Bool,
    has_pending_lora_state_dict: Bool,
    train_any_embedding: Bool,
) -> FluxLoRACreationPlan:
    return FluxLoRACreationPlan(
        has_text_encoder_1,
        has_text_encoder_2,
        config_trains_text_encoder_1,
        config_trains_text_encoder_2,
        state_dict_has_lora_te1,
        state_dict_has_lora_te2,
        has_pending_lora_state_dict,
        train_any_embedding,
    )


struct FluxLoRASetup(Movable):
    var registration: FluxLoRASetupRegistration

    def __init__(out self):
        self.registration = flux_lora_setup_registration()

    def create_parameters(
        self,
        train_any_embedding_or_output: Bool = False,
        train_embedding_1: Bool = False,
        train_embedding_2: Bool = False,
        has_text_encoder_1: Bool = True,
        has_text_encoder_2: Bool = True,
    ) -> List[String]:
        return flux_lora_parameter_group_names(
            train_any_embedding_or_output,
            train_embedding_1,
            train_embedding_2,
            has_text_encoder_1,
            has_text_encoder_2,
        )

    def creation_plan(
        self,
        has_text_encoder_1: Bool,
        has_text_encoder_2: Bool,
        config_trains_text_encoder_1: Bool,
        config_trains_text_encoder_2: Bool,
        state_dict_has_lora_te1: Bool,
        state_dict_has_lora_te2: Bool,
        has_pending_lora_state_dict: Bool,
        train_any_embedding: Bool,
    ) -> FluxLoRACreationPlan:
        return flux_lora_creation_plan(
            has_text_encoder_1,
            has_text_encoder_2,
            config_trains_text_encoder_1,
            config_trains_text_encoder_2,
            state_dict_has_lora_te1,
            state_dict_has_lora_te2,
            has_pending_lora_state_dict,
            train_any_embedding,
        )

    def layer_filters(self, layer_filter: String) raises -> List[String]:
        return flux_lora_layer_filters(layer_filter)

    def requires_grad_plan(self) -> FluxLoRARequiresGradPlan:
        return FluxLoRARequiresGradPlan()

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

    def freezes_base_text_encoder_1(self) -> Bool:
        return True

    def freezes_base_text_encoder_2(self) -> Bool:
        return True

    def freezes_base_transformer(self) -> Bool:
        return True

    def freezes_vae(self) -> Bool:
        return True

    def trainable_model_part_names(self) -> List[String]:
        var names = List[String]()
        names.append(FLUX_LORA_TEXT_ENCODER_1_PART)
        names.append(FLUX_LORA_TEXT_ENCODER_2_PART)
        names.append(FLUX_LORA_TRANSFORMER_PART)
        return names^

    def base_model_part_names(self) -> List[String]:
        var names = List[String]()
        names.append(FLUX_PART_TEXT_ENCODER_1)
        names.append(FLUX_PART_TEXT_ENCODER_2)
        names.append(FLUX_PART_TRANSFORMER)
        return names^

    def normalizes_embeddings_after_optimizer_step(self) -> Bool:
        return True

    def after_optimizer_step_reapplies_requires_grad(self) -> Bool:
        return True

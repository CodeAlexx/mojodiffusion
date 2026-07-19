# ChromaLoRASetup.mojo - build-only Chroma LoRA setup surface.
#
# Source of truth:
#   /home/alex/Serenity/modules/modelSetup/ChromaLoRASetup.py
#
# This records Serenity's Chroma LoRA registration, parameter groups, wrapper
# creation gates, requires-grad plan, train-device plan, and after_optimizer_step
# behavior. Runtime LoRAModuleWrapper construction, gradients, and optimizer
# updates are intentionally out of scope.

from serenity_trainer.modelSetup.BaseChromaSetup import (
    CHROMA_PART_EMBEDDINGS,
    CHROMA_PART_TEXT_ENCODER,
    CHROMA_PART_TRANSFORMER,
    ChromaTrainDevicePlan,
    chroma_train_device_plan,
)
from serenity_trainer.util.enum.ModelType import MODEL_TYPE_CHROMA_1
from serenity_trainer.util.enum.TrainingMethod import TM_LORA


comptime CHROMA_LORA_MODEL_TYPE = MODEL_TYPE_CHROMA_1
comptime CHROMA_LORA_TRAINING_METHOD = TM_LORA
comptime CHROMA_LORA_TEXT_ENCODER_PART = "text_encoder_lora"
comptime CHROMA_LORA_TRANSFORMER_PART = "transformer_lora"
comptime CHROMA_LORA_EMBEDDINGS_PART = "embeddings"
comptime CHROMA_LORA_PREFIX_TE = "lora_te"
comptime CHROMA_LORA_PREFIX_TRANSFORMER = "lora_transformer"


def chroma_lora_parameter_group_names(
    train_any_embedding_or_output: Bool = False,
    train_text_encoder_embedding: Bool = False,
    has_text_encoder: Bool = True,
) -> List[String]:
    """Serenity ChromaLoRASetup.create_parameters group order."""
    var names = List[String]()
    names.append(CHROMA_LORA_TEXT_ENCODER_PART)
    if train_any_embedding_or_output:
        if train_text_encoder_embedding and has_text_encoder:
            names.append(CHROMA_LORA_EMBEDDINGS_PART)
    names.append(CHROMA_LORA_TRANSFORMER_PART)
    return names^


def chroma_lora_state_dict_prefixes() -> List[String]:
    var prefixes = List[String]()
    prefixes.append(CHROMA_LORA_PREFIX_TE)
    prefixes.append(CHROMA_LORA_PREFIX_TRANSFORMER)
    return prefixes^


def chroma_lora_transformer_target_filters_expression() -> String:
    return "config.layer_filter.split(',')"


struct ChromaLoRASetupRegistration(Copyable, Movable, ImplicitlyCopyable):
    var model_type: Int
    var training_method: Int
    var creates_text_encoder_lora: Bool
    var creates_transformer_lora: Bool
    var supports_embedding_training: Bool
    var supports_output_embedding_training: Bool

    def __init__(out self):
        self.model_type = CHROMA_LORA_MODEL_TYPE
        self.training_method = CHROMA_LORA_TRAINING_METHOD
        self.creates_text_encoder_lora = True
        self.creates_transformer_lora = True
        self.supports_embedding_training = True
        self.supports_output_embedding_training = True


struct ChromaLoRACreationPlan(Movable):
    var create_text_encoder_lora: Bool
    var create_transformer_lora: Bool
    var wrapper_prefixes: List[String]
    var transformer_target_filters_expression: String
    var loads_pending_state_dict: Bool
    var clears_pending_state_dict_after_load: Bool
    var sets_dropout_from_config: Bool
    var moves_lora_to_config_weight_dtype: Bool
    var hooks_lora_to_module: Bool
    var moves_input_embeddings_to_embedding_weight_dtype: Bool
    var removes_added_embeddings_from_tokenizer: Bool
    var setups_embeddings: Bool
    var setups_embedding_wrapper: Bool
    var initializes_model_parameters: Bool

    def __init__(
        out self,
        has_text_encoder: Bool,
        config_trains_text_encoder: Bool,
        state_dict_has_lora_te: Bool,
        has_pending_lora_state_dict: Bool,
        train_any_embedding: Bool,
    ):
        self.create_text_encoder_lora = has_text_encoder and (
            config_trains_text_encoder or state_dict_has_lora_te
        )
        self.create_transformer_lora = True
        self.wrapper_prefixes = chroma_lora_state_dict_prefixes()
        self.transformer_target_filters_expression = (
            chroma_lora_transformer_target_filters_expression()
        )
        self.loads_pending_state_dict = has_pending_lora_state_dict
        self.clears_pending_state_dict_after_load = has_pending_lora_state_dict
        self.sets_dropout_from_config = True
        self.moves_lora_to_config_weight_dtype = True
        self.hooks_lora_to_module = True
        self.moves_input_embeddings_to_embedding_weight_dtype = train_any_embedding
        self.removes_added_embeddings_from_tokenizer = True
        self.setups_embeddings = True
        self.setups_embedding_wrapper = True
        self.initializes_model_parameters = True


struct ChromaLoRARequiresGradPlan(Copyable, Movable, ImplicitlyCopyable):
    var setup_embeddings_requires_grad: Bool
    var freezes_text_encoder_base: Bool
    var freezes_transformer_base: Bool
    var freezes_vae: Bool
    var applies_text_encoder_lora_config: Bool
    var applies_transformer_lora_config: Bool

    def __init__(out self):
        self.setup_embeddings_requires_grad = True
        self.freezes_text_encoder_base = True
        self.freezes_transformer_base = True
        self.freezes_vae = True
        self.applies_text_encoder_lora_config = True
        self.applies_transformer_lora_config = True


def chroma_lora_setup_registration() -> ChromaLoRASetupRegistration:
    return ChromaLoRASetupRegistration()


def chroma_lora_creation_plan(
    has_text_encoder: Bool,
    config_trains_text_encoder: Bool,
    state_dict_has_lora_te: Bool,
    has_pending_lora_state_dict: Bool,
    train_any_embedding: Bool,
) -> ChromaLoRACreationPlan:
    return ChromaLoRACreationPlan(
        has_text_encoder,
        config_trains_text_encoder,
        state_dict_has_lora_te,
        has_pending_lora_state_dict,
        train_any_embedding,
    )


struct ChromaLoRASetup(Movable):
    var registration: ChromaLoRASetupRegistration

    def __init__(out self):
        self.registration = chroma_lora_setup_registration()

    def create_parameters(
        self,
        train_any_embedding_or_output: Bool = False,
        train_text_encoder_embedding: Bool = False,
        has_text_encoder: Bool = True,
    ) -> List[String]:
        return chroma_lora_parameter_group_names(
            train_any_embedding_or_output,
            train_text_encoder_embedding,
            has_text_encoder,
        )

    def creation_plan(
        self,
        has_text_encoder: Bool,
        config_trains_text_encoder: Bool,
        state_dict_has_lora_te: Bool,
        has_pending_lora_state_dict: Bool,
        train_any_embedding: Bool,
    ) -> ChromaLoRACreationPlan:
        return chroma_lora_creation_plan(
            has_text_encoder,
            config_trains_text_encoder,
            state_dict_has_lora_te,
            has_pending_lora_state_dict,
            train_any_embedding,
        )

    def requires_grad_plan(self) -> ChromaLoRARequiresGradPlan:
        return ChromaLoRARequiresGradPlan()

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
        names.append(CHROMA_LORA_TEXT_ENCODER_PART)
        names.append(CHROMA_LORA_TRANSFORMER_PART)
        return names^

    def base_model_part_names(self) -> List[String]:
        var names = List[String]()
        names.append(CHROMA_PART_TEXT_ENCODER)
        names.append(CHROMA_PART_TRANSFORMER)
        return names^

    def embedding_parameter_group_name(self) -> String:
        return String(CHROMA_PART_EMBEDDINGS)

    def freezes_base_text_encoder(self) -> Bool:
        return True

    def freezes_base_transformer(self) -> Bool:
        return True

    def freezes_vae(self) -> Bool:
        return True

    def normalizes_embeddings_after_optimizer_step(self) -> Bool:
        return True

    def normalizes_embedding_wrapper_after_optimizer_step(self) -> Bool:
        return True

    def after_optimizer_step_reapplies_requires_grad(self) -> Bool:
        return True

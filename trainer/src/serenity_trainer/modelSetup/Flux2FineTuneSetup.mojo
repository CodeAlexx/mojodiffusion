# Flux2FineTuneSetup.mojo - build-only Flux2/Klein fine-tune setup surface.
#
# Source of truth:
#   /home/alex/Serenity/modules/modelSetup/Flux2FineTuneSetup.py
#
# Flux2 dev and Klein share Serenity ModelType.FLUX_2; the runtime branch is
# decided by Flux2Model.is_dev() from transformer.config.num_attention_heads.
# This file records Serenity's full-finetune setup contract only. It does not
# implement Flux2Transformer2DModel full-weight training, gradients, optimizer
# updates, saver parity, or sampler parity.

from serenity_trainer.util.enum.ModelType import MODEL_TYPE_FLUX_2
from serenity_trainer.util.enum.TrainingMethod import TM_FINE_TUNE


comptime FLUX2_FINE_TUNE_MODEL_TYPE = MODEL_TYPE_FLUX_2
comptime FLUX2_FINE_TUNE_TRAINING_METHOD = TM_FINE_TUNE
comptime FLUX2_FINE_TUNE_TRANSFORMER_PART = "transformer"
comptime FLUX2_FINE_TUNE_TEXT_ENCODER_PART = "text_encoder"
comptime FLUX2_FINE_TUNE_VAE_PART = "vae"


def flux2_fine_tune_parameter_group_names() -> List[String]:
    """Serenity Flux2FineTuneSetup.create_parameters group order."""
    var names = List[String]()
    names.append(FLUX2_FINE_TUNE_TRANSFORMER_PART)
    return names^


def flux2_fine_tune_dtype_caveats() -> List[String]:
    var caveats = List[String]()
    caveats.append("Serenity BaseFlux2Setup.predict intentionally calls batch['latent_image'].float() before patchify/scale compute")
    caveats.append("That float() boundary is compute-only and does not allow persistent checkpoint/model tensor upcasts to F32")
    caveats.append("Transformer hidden_states and encoder_hidden_states are cast to model.train_dtype at the transformer call")
    caveats.append("Flux2 dev and Klein share ModelType.FLUX_2; do not use Klein tensors as Flux2 dev numeric evidence")
    return caveats^


struct Flux2FineTuneSetupRegistration(Copyable, Movable, ImplicitlyCopyable):
    var model_type: Int
    var training_method: Int
    var trains_text_encoder: Bool
    var trains_transformer: Bool
    var trains_vae: Bool
    var supports_embedding_training: Bool
    var supports_output_embedding_training: Bool

    def __init__(out self):
        self.model_type = FLUX2_FINE_TUNE_MODEL_TYPE
        self.training_method = FLUX2_FINE_TUNE_TRAINING_METHOD
        self.trains_text_encoder = False
        self.trains_transformer = True
        self.trains_vae = False
        self.supports_embedding_training = False
        self.supports_output_embedding_training = False


struct Flux2FineTuneSetupModelPlan(Copyable, Movable, ImplicitlyCopyable):
    var initializes_model_parameters: Bool
    var uses_module_filter_for_transformer: Bool
    var uses_debug_flag_for_transformer_filter: Bool
    var setups_embeddings: Bool
    var setups_embedding_wrappers: Bool

    def __init__(out self):
        self.initializes_model_parameters = True
        self.uses_module_filter_for_transformer = True
        self.uses_debug_flag_for_transformer_filter = True
        self.setups_embeddings = False
        self.setups_embedding_wrappers = False


struct Flux2FineTuneRequiresGradPlan(Copyable, Movable, ImplicitlyCopyable):
    var applies_transformer_config: Bool
    var freezes_vae: Bool
    var freezes_text_encoder: Bool
    var transformer_uses_module_filter: Bool

    def __init__(out self):
        self.applies_transformer_config = True
        self.freezes_vae = True
        self.freezes_text_encoder = True
        self.transformer_uses_module_filter = True


struct Flux2FineTuneTrainDevicePlan(Copyable, Movable, ImplicitlyCopyable):
    var text_encoder_on_train_device: Bool
    var vae_on_train_device: Bool
    var transformer_on_train_device: Bool
    var text_encoder_train_mode: Bool
    var vae_train_mode: Bool
    var transformer_train_mode: Bool

    def __init__(out self, latent_caching: Bool, transformer_train: Bool):
        self.text_encoder_on_train_device = not latent_caching
        self.vae_on_train_device = not latent_caching
        self.transformer_on_train_device = True
        self.text_encoder_train_mode = False
        self.vae_train_mode = False
        self.transformer_train_mode = transformer_train


struct Flux2FineTuneTextCachingPlan(Copyable, Movable, ImplicitlyCopyable):
    var move_model_to_temp_device: Bool
    var move_text_encoder_to_train_device: Bool
    var set_eval_mode: Bool
    var run_torch_gc: Bool

    def __init__(out self):
        # BaseFlux2Setup.prepare_text_caching always moves text_encoder to train.
        self.move_model_to_temp_device = True
        self.move_text_encoder_to_train_device = True
        self.set_eval_mode = True
        self.run_torch_gc = True


def flux2_fine_tune_setup_registration() -> Flux2FineTuneSetupRegistration:
    return Flux2FineTuneSetupRegistration()


def flux2_fine_tune_setup_model_plan() -> Flux2FineTuneSetupModelPlan:
    return Flux2FineTuneSetupModelPlan()


def flux2_fine_tune_train_device_plan(
    latent_caching: Bool, transformer_train: Bool
) -> Flux2FineTuneTrainDevicePlan:
    return Flux2FineTuneTrainDevicePlan(latent_caching, transformer_train)


def flux2_fine_tune_text_caching_plan() -> Flux2FineTuneTextCachingPlan:
    return Flux2FineTuneTextCachingPlan()


struct Flux2FineTuneSetup(Copyable, Movable, ImplicitlyCopyable):
    var registration: Flux2FineTuneSetupRegistration

    def __init__(out self):
        self.registration = flux2_fine_tune_setup_registration()

    def create_parameters(self) -> List[String]:
        return flux2_fine_tune_parameter_group_names()

    def setup_model_plan(self) -> Flux2FineTuneSetupModelPlan:
        return flux2_fine_tune_setup_model_plan()

    def requires_grad_plan(self) -> Flux2FineTuneRequiresGradPlan:
        return Flux2FineTuneRequiresGradPlan()

    def train_device_plan(
        self, latent_caching: Bool, transformer_train: Bool
    ) -> Flux2FineTuneTrainDevicePlan:
        return flux2_fine_tune_train_device_plan(latent_caching, transformer_train)

    def prepare_text_caching_plan(self) -> Flux2FineTuneTextCachingPlan:
        return flux2_fine_tune_text_caching_plan()

    def trainable_model_part_names(self) -> List[String]:
        var names = List[String]()
        names.append(FLUX2_FINE_TUNE_TRANSFORMER_PART)
        return names^

    def frozen_model_part_names(self) -> List[String]:
        var names = List[String]()
        names.append(FLUX2_FINE_TUNE_TEXT_ENCODER_PART)
        names.append(FLUX2_FINE_TUNE_VAE_PART)
        return names^

    def uses_module_filter_for_transformer(self) -> Bool:
        return True

    def freezes_text_encoder(self) -> Bool:
        return True

    def freezes_vae(self) -> Bool:
        return True

    def after_optimizer_step_reapplies_requires_grad(self) -> Bool:
        return True

    def dtype_caveats(self) -> List[String]:
        return flux2_fine_tune_dtype_caveats()

# training/serenity_trainer_cache_preflight.mojo
#
# No-CUDA preflight for the SerenityTrainer cache/readiness contracts used by the
# product train entrypoint. This does not inspect cache tensors. It binds the
# model type to the text-conditioning and VAE encode/cache contracts so a product
# run cannot silently claim cache readiness for an unsupported model path.

from serenitymojo.models.text_encoder.serenity_trainer_conditioning_contract import (
    SERENITY_TEXT_CACHE_USE_SAMPLE,
    SERENITY_TEXT_CACHE_USE_TRAIN,
    default_serenity_text_conditioning_plan,
    serenity_text_conditioning_cache_readiness_contract,
    validate_serenity_text_conditioning_cache_readiness,
    validate_serenity_text_conditioning_plan,
)
from serenitymojo.sampling.vae_encoder_contract import (
    SERENITY_VAE_ENCODER_PARTIAL_PREPARED_ONLY,
    SERENITY_VAE_ENCODER_RAW_CACHE_READY,
    SERENITY_VAE_ENCODER_UNSUPPORTED,
    default_serenity_vae_encoder_contract,
    validate_serenity_vae_encoder_contract,
)
from serenitymojo.training.train_config import TrainConfig


@fieldwise_init
struct SerenityCachePreflightPlan(Copyable, Movable):
    var model_type: String
    var text_contract_name: String
    var text_train_required_fields: String
    var text_sample_required_fields: String
    var text_train_requires_mask: Bool
    var text_sample_requires_mask: Bool
    var text_requires_runtime_ids: Bool
    var vae_contract_name: String
    var vae_raw_cache_status: Int
    var vae_prepared_encode_status: Int
    var vae_cache_channels: Int
    var vae_prepared_channels: Int
    var vae_cache_to_prepared_patch_size: Int
    var only_cache_requested: Bool
    var dtype_policy: String
    var fail_loud_policy: String

    def raw_vae_cache_ready(self) -> Bool:
        return self.vae_raw_cache_status == SERENITY_VAE_ENCODER_RAW_CACHE_READY

    def prepared_only(self) -> Bool:
        return (
            self.vae_raw_cache_status != SERENITY_VAE_ENCODER_RAW_CACHE_READY
            and self.vae_prepared_encode_status == SERENITY_VAE_ENCODER_PARTIAL_PREPARED_ONLY
        )


def create_serenity_trainer_cache_preflight_plan(cfg: TrainConfig) raises -> SerenityCachePreflightPlan:
    var text_plan = default_serenity_text_conditioning_plan(cfg.name)
    validate_serenity_text_conditioning_plan(text_plan)

    var train_readiness = serenity_text_conditioning_cache_readiness_contract(
        text_plan, SERENITY_TEXT_CACHE_USE_TRAIN,
    )
    validate_serenity_text_conditioning_cache_readiness(train_readiness)
    var sample_readiness = serenity_text_conditioning_cache_readiness_contract(
        text_plan, SERENITY_TEXT_CACHE_USE_SAMPLE,
    )
    validate_serenity_text_conditioning_cache_readiness(sample_readiness)

    var vae_plan = default_serenity_vae_encoder_contract(cfg.name)
    validate_serenity_vae_encoder_contract(vae_plan)

    return SerenityCachePreflightPlan(
        cfg.name.copy(),
        text_plan.contract_name.copy(),
        train_readiness.required_cache_fields.copy(),
        sample_readiness.required_cache_fields.copy(),
        train_readiness.requires_mask,
        sample_readiness.requires_mask,
        train_readiness.requires_text_ids or train_readiness.requires_img_ids
            or sample_readiness.requires_text_ids or sample_readiness.requires_img_ids,
        vae_plan.name.copy(),
        vae_plan.raw_cache_encode_status,
        vae_plan.prepared_encode_status,
        vae_plan.cache_latent_channels,
        vae_plan.prepared_latent_channels,
        vae_plan.cache_to_prepared_patch_size,
        cfg.only_cache,
        String("preserve_checkpoint_or_train_dtype_at_tensor_boundaries"),
        String("raise_before_product_cache_or_train_when_required_contract_is_missing"),
    )


def validate_serenity_trainer_cache_preflight_plan(plan: SerenityCachePreflightPlan) raises:
    if plan.model_type == String(""):
        raise Error("SerenityTrainer cache preflight: model_type is missing")
    if plan.text_contract_name == String("") or plan.vae_contract_name == String(""):
        raise Error("SerenityTrainer cache preflight: missing text or VAE contract name")
    if plan.text_train_required_fields == String("") or plan.text_sample_required_fields == String(""):
        raise Error("SerenityTrainer cache preflight: text cache fields are missing")
    if plan.vae_cache_channels <= 0 or plan.vae_prepared_channels <= 0:
        raise Error("SerenityTrainer cache preflight: VAE channel counts are invalid")
    if plan.vae_cache_to_prepared_patch_size <= 0:
        raise Error("SerenityTrainer cache preflight: VAE patch size is invalid")
    if (
        plan.vae_raw_cache_status != SERENITY_VAE_ENCODER_RAW_CACHE_READY
        and plan.vae_raw_cache_status != SERENITY_VAE_ENCODER_UNSUPPORTED
    ):
        raise Error("SerenityTrainer cache preflight: unknown raw VAE cache status")
    if (
        plan.vae_prepared_encode_status != SERENITY_VAE_ENCODER_RAW_CACHE_READY
        and plan.vae_prepared_encode_status != SERENITY_VAE_ENCODER_PARTIAL_PREPARED_ONLY
        and plan.vae_prepared_encode_status != SERENITY_VAE_ENCODER_UNSUPPORTED
    ):
        raise Error("SerenityTrainer cache preflight: unknown prepared VAE status")
    if plan.dtype_policy != String("preserve_checkpoint_or_train_dtype_at_tensor_boundaries"):
        raise Error("SerenityTrainer cache preflight: dtype policy drift")
    if plan.fail_loud_policy != String("raise_before_product_cache_or_train_when_required_contract_is_missing"):
        raise Error("SerenityTrainer cache preflight: fail-loud policy drift")
    if plan.only_cache_requested and not plan.raw_vae_cache_ready():
        raise Error(
            String("SerenityTrainer only_cache requires raw VAE cache encode readiness for ")
            + plan.model_type
            + String("; current Mojo VAE contract=")
            + plan.vae_contract_name
            + String(" raw_status=")
            + String(plan.vae_raw_cache_status)
            + String(" prepared_status=")
            + String(plan.vae_prepared_encode_status)
        )


def serenity_trainer_cache_preflight_summary(plan: SerenityCachePreflightPlan) -> String:
    return (
        String("cache_preflight model=")
        + plan.model_type
        + String(" text=")
        + plan.text_contract_name
        + String(" train_fields=")
        + plan.text_train_required_fields
        + String(" sample_fields=")
        + plan.text_sample_required_fields
        + String(" vae=")
        + plan.vae_contract_name
        + String(" raw_vae_ready=")
        + String(plan.raw_vae_cache_ready())
        + String(" prepared_only=")
        + String(plan.prepared_only())
    )

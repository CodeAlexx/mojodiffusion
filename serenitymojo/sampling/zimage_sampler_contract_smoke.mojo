# Smoke gate for the Z-Image OneTrainer sampler contract.

from serenitymojo.sampling.zimage_sampler_contract import (
    ZImageSamplerContract,
    ZIMAGE_CFG_TEXTBOOK_NEGATIVE_FIRST,
    ZIMAGE_SCHEDULER_MODEL_COPY_SET_TIMESTEPS,
    ZIMAGE_TIMESTEP_ONE_MINUS_SIGMA,
    ZIMAGE_TRAIN_SHIFT_DYNAMIC_SCHEDULER_CONFIG,
    ZIMAGE_TRAIN_SHIFT_FIXED_CONFIG,
    zimage_cfg_batch_size,
    zimage_cfg_uses_negative_prompt,
    zimage_default_sampler_contract,
    zimage_latent_dim,
    zimage_model_timestep_from_scheduler_timestep,
    zimage_model_timestep_from_sigma,
    zimage_noisy_latent_value,
    zimage_prediction_from_transformer_sample,
    zimage_sampler_initial_latent_numel,
    zimage_scale_latent_value,
    zimage_textbook_cfg_scalar,
    zimage_training_flow_target,
    zimage_training_reconstruct_scaled_latent,
    zimage_training_timestep_shift_mode,
    zimage_unscale_latent_value,
    validate_zimage_sampler_contract,
)


def _check(cond: Bool, msg: String) raises:
    if not cond:
        raise Error(String("Z-Image sampler contract smoke FAILED: ") + msg)


def _check_close(actual: Float32, expected: Float32, msg: String) raises:
    var diff = actual - expected
    if diff < Float32(0.0):
        diff = -diff
    if diff > Float32(1.0e-5):
        raise Error(
            String("Z-Image sampler contract smoke FAILED: ")
            + msg
            + String(" actual=")
            + String(actual)
            + String(" expected=")
            + String(expected)
        )


def _expect_validation_block(label: String, contract: ZImageSamplerContract) raises:
    var raised = False
    try:
        validate_zimage_sampler_contract(contract)
    except e:
        raised = True
        print("  blocked ", label, " as expected: ", String(e))
    if not raised:
        raise Error(String("Z-Image sampler contract smoke expected block for ") + label)


def main() raises:
    print("==== Z-Image sampler contract smoke ====")

    var contract = zimage_default_sampler_contract()
    validate_zimage_sampler_contract(contract)
    _check(contract.scheduler_setup_mode == ZIMAGE_SCHEDULER_MODEL_COPY_SET_TIMESTEPS, "scheduler mode")
    _check(contract.timestep_mode == ZIMAGE_TIMESTEP_ONE_MINUS_SIGMA, "timestep mode")
    _check(contract.cfg_mode == ZIMAGE_CFG_TEXTBOOK_NEGATIVE_FIRST, "cfg mode")
    _check(contract.width == 1024 and contract.height == 1024, "default size")
    _check(contract.diffusion_steps == 28, "default steps")
    _check_close(contract.cfg_scale, Float32(4.0), "default cfg")
    _check(contract.vae_scale_factor == 8, "vae scale factor")
    _check(contract.latent_channels == 16, "latent channels")
    _check(not contract.external_pack_latents, "no external pack")
    _check(not contract.external_unpack_latents, "no external unpack")
    _check(contract.transformer_input_rank == 5, "rank-5 transformer input")
    _check(contract.transformer_frame_dim == 1, "single frame dim")

    _check(zimage_latent_dim(1024) == 128, "1024 latent dim")
    _check(zimage_sampler_initial_latent_numel(1024, 1024) == 262144, "latent numel")
    _check_close(
        zimage_model_timestep_from_scheduler_timestep(Float32(250.0)),
        Float32(0.75),
        "scheduler timestep",
    )
    _check_close(zimage_model_timestep_from_sigma(Float32(0.25)), Float32(0.75), "sigma timestep")

    var scaled = zimage_scale_latent_value(Float32(0.75), Float32(0.1159), Float32(0.3611))
    var unscaled = zimage_unscale_latent_value(scaled, Float32(0.1159), Float32(0.3611))
    _check_close(unscaled, Float32(0.75), "vae scale round trip")

    _check(zimage_cfg_batch_size(Float32(1.0)) == 1, "cfg=1 single batch")
    _check(zimage_cfg_batch_size(Float32(4.0)) == 2, "cfg>1 dual batch")
    _check(not zimage_cfg_uses_negative_prompt(Float32(1.0)), "cfg=1 no negative")
    _check(zimage_cfg_uses_negative_prompt(Float32(4.0)), "cfg>1 uses negative")
    _check_close(
        zimage_textbook_cfg_scalar(Float32(10.0), Float32(2.0), Float32(4.0)),
        Float32(34.0),
        "textbook cfg",
    )
    _check_close(zimage_prediction_from_transformer_sample(Float32(3.0)), Float32(-3.0), "prediction sign")

    var noisy = zimage_noisy_latent_value(Float32(3.0), Float32(1.0), Float32(0.25))
    var target = zimage_training_flow_target(Float32(3.0), Float32(1.0))
    _check_close(noisy, Float32(1.5), "noisy latent")
    _check_close(target, Float32(2.0), "flow target")
    _check_close(
        zimage_training_reconstruct_scaled_latent(noisy, target, Float32(0.25)),
        Float32(1.0),
        "reconstruct scaled latent",
    )
    _check(
        zimage_training_timestep_shift_mode(False) == ZIMAGE_TRAIN_SHIFT_FIXED_CONFIG,
        "fixed train shift",
    )
    _check(
        zimage_training_timestep_shift_mode(True)
        == ZIMAGE_TRAIN_SHIFT_DYNAMIC_SCHEDULER_CONFIG,
        "dynamic train shift",
    )

    var bad_scheduler = zimage_default_sampler_contract()
    bad_scheduler.scheduler_class = String("EulerDiscreteScheduler")
    _expect_validation_block(String("bad scheduler"), bad_scheduler)

    var bad_pack = zimage_default_sampler_contract()
    bad_pack.external_pack_latents = True
    _expect_validation_block(String("external pack"), bad_pack)

    var bad_claim = zimage_default_sampler_contract()
    bad_claim.sampler_claims_image_or_speed_parity = True
    _expect_validation_block(String("parity claim"), bad_claim)

    print("Z-Image sampler contract smoke PASS")

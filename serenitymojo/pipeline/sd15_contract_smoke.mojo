# SD 1.5 metadata contract smoke.
#
# Header-only: no DeviceContext, no tensor H2D load, no CLIP/UNet math, and no
# VAE decode.

from serenitymojo.models.dit.sd15_contract import (
    SD15_CLIP_HIDDEN,
    SD15_CLIP_LAYERS,
    SD15_DEFAULT_CFG_X10,
    SD15_DEFAULT_STEPS,
    SD15_IMAGE_TOKENS,
    SD15_LATENT_CHANNELS,
    SD15_LATENT_H,
    SD15_LATENT_W,
    SD15_TEXT_ENCODER_TENSORS,
    SD15_TEXT_TOKENS,
    SD15_TOTAL_SEQUENCE,
    SD15_UNET_CONTEXT_DIM,
    SD15_UNET_MODEL_CHANNELS,
    SD15_UNET_NUM_HEADS,
    SD15_UNET_TENSORS,
    SD15_VAE_TENSORS,
    build_sd15_token_plan,
    sd15_default_cfg_scale,
    sd15_vae_scaling_factor,
    validate_sd15_local_paths,
    validate_sd15_metadata_contract,
)
from serenitymojo.runtime.model_manifest import sd15_default_manifest


def main() raises:
    var manifest = sd15_default_manifest()
    var checked_paths = validate_sd15_local_paths()
    var plan = validate_sd15_metadata_contract(manifest)
    var explicit_plan = build_sd15_token_plan(512, 512, 1)
    if explicit_plan.total_sequence != plan.total_sequence:
        raise Error("SD1.5 explicit token plan mismatch")

    print("[sd15-contract] paths checked/missing:", checked_paths, 0)
    print(
        "[sd15-contract] headers unet/text/vae tensors:",
        SD15_UNET_TENSORS,
        SD15_TEXT_ENCODER_TENSORS,
        SD15_VAE_TENSORS,
    )
    print(
        "[sd15-contract] unet model_channels/context/heads:",
        SD15_UNET_MODEL_CHANNELS,
        SD15_UNET_CONTEXT_DIM,
        SD15_UNET_NUM_HEADS,
    )
    print(
        "[sd15-contract] clip hidden/layers/tokens:",
        SD15_CLIP_HIDDEN,
        SD15_CLIP_LAYERS,
        SD15_TEXT_TOKENS,
    )
    print(
        "[sd15-contract] latent/tokens/sequence:",
        SD15_LATENT_CHANNELS,
        SD15_LATENT_H,
        "x",
        SD15_LATENT_W,
        SD15_IMAGE_TOKENS,
        SD15_TEXT_TOKENS,
        SD15_TOTAL_SEQUENCE,
    )
    print(
        "[sd15-contract] steps/cfg/vae_scale:",
        SD15_DEFAULT_STEPS,
        sd15_default_cfg_scale(),
        sd15_vae_scaling_factor(),
    )
    print("SD1.5 metadata contract PASS")

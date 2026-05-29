# Compile/run smoke for SDXL metadata contracts.
#
# This is intentionally header-only: it validates the standalone SDXL UNet and
# LDM VAE files, and validates cached embeddings when the cache exists locally.

from serenitymojo.models.dit.sdxl_contract import (
    sdxl_cached_embeddings_present,
    sdxl_cached_embedding_generator_command,
    sdxl_default_cached_embeddings_path,
    validate_sdxl_cached_embedding_header,
    validate_sdxl_static_checkpoint_contract,
)
from serenitymojo.registry.checkpoints import (
    default_manifest_by_id,
    validate_manifest_paths,
)


def main() raises:
    var manifest = default_manifest_by_id(String("sdxl"))
    var status = validate_manifest_paths(manifest)
    if status.missing != 0:
        raise Error("SDXL contract smoke has missing registered paths")
    validate_sdxl_static_checkpoint_contract(manifest.denoiser_path, manifest.vae_path)
    print("[sdxl-contract] UNet/VAE headers PASS")

    var emb_path = sdxl_default_cached_embeddings_path()
    if sdxl_cached_embeddings_present(emb_path):
        validate_sdxl_cached_embedding_header(emb_path)
        print("[sdxl-contract] cached embeddings PASS")
    else:
        print("[sdxl-contract] cached embeddings missing; strict pipeline is not runnable")
        print("[sdxl-contract] expected artifact:")
        print("  ", emb_path)
        print("[sdxl-contract] generator handoff:")
        print("  ", sdxl_cached_embedding_generator_command(emb_path))

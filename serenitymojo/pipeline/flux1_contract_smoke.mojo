# FLUX.1-dev pipeline contract smoke.
#
# Metadata-only: no DeviceContext, no Tensor allocations, and no weight H2D
# loads. This keeps the FLUX.1-dev 1024 production-wrapper contract checked
# while the heavyweight DiT/VAE path is still a smoke target.

from serenitymojo.models.dit.flux1_contract import (
    flux1_default_cached_inputs_path,
    validate_flux1_cached_inputs_header,
    validate_flux1_pipeline_contract,
)
from serenitymojo.registry.checkpoints import (
    default_manifest_by_id,
    path_exists,
    validate_manifest_paths,
)


def main() raises:
    var manifest = default_manifest_by_id(String("flux1_dev"))
    var status = validate_manifest_paths(manifest)
    print(
        "[flux1-contract] manifest paths checked/missing:",
        status.checked,
        status.missing,
    )
    if status.missing != 0:
        raise Error("FLUX.1 contract smoke has missing registered paths")
    validate_flux1_pipeline_contract(manifest)
    var inputs_path = flux1_default_cached_inputs_path()
    if path_exists(inputs_path):
        validate_flux1_cached_inputs_header(inputs_path)
        print("[flux1-contract] cached inputs PASS:", inputs_path)
    else:
        print("[flux1-contract] cached inputs missing:", inputs_path)
        print("Generate with: cd /home/alex/EriDiffusion/inference-flame && FLUX1_SAVE_INPUTS=1 cargo run --release --bin flux1_infer")
    print("FLUX.1-dev pipeline contract PASS")

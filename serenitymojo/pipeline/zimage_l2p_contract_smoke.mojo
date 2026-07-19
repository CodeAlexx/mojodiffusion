# Z-Image L2P contract smoke.
#
# Metadata-only: checks static pixel-space geometry and the local merged
# safetensors header. No DeviceContext, no Tensor allocations, and no model
# weights are loaded to GPU.

from serenitymojo.models.dit.zimage_l2p_contract import (
    validate_zimage_l2p_conditioning_header,
    validate_zimage_l2p_default_checkpoint_contract,
    zimage_l2p_default_conditioning_path,
    zimage_l2p_default_checkpoint_path,
    zimage_l2p_infer_command,
)
from serenitymojo.registry.checkpoints import path_exists


def main() raises:
    validate_zimage_l2p_default_checkpoint_contract()
    print(
        "Z-Image L2P contract PASS:",
        zimage_l2p_default_checkpoint_path(),
    )

    var embeddings_path = zimage_l2p_default_conditioning_path()
    if path_exists(embeddings_path):
        var cond = validate_zimage_l2p_conditioning_header(embeddings_path, False)
        print(
            "Z-Image L2P conditioning PASS: cap/uncond tokens",
            cond.cap_tokens,
            cond.uncond_tokens,
        )
    else:
        print("Z-Image L2P conditioning sidecar missing:", embeddings_path)
        print(
            "Expected keys: cap_feats [1, seq, 2560], optional cap_feats_uncond [1, seq, 2560]"
        )
        print(
            "Rust handoff:",
            zimage_l2p_infer_command(embeddings_path, String("output/l2p_output.png")),
        )

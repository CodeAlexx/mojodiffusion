#!/usr/bin/env python3
"""Reference oracle for the LTX2 connector REGISTER-REPLACEMENT semantics.

Runs ltx_core's Embeddings1DConnector (the ground truth) on the regenerated
right-padded context dumps WITH the additive attention mask — i.e. pad rows
are replaced by the checkpoint's learnable registers and the blocks then run
with a zeroed mask — and writes inputs + expected outputs for the Mojo gate
(serenitymojo/pipeline/ltx2_connector_mask_probe.mojo).

Run: /home/alex/SerenityTrainer/venv/bin/python scripts/ltx2_connector_mask_oracle.py
"""
import sys

import torch

sys.path.insert(0, "/home/alex/LTX-2/packages/ltx-core/src")
from ltx_core.model.transformer.rope import LTXRopeType  # noqa: E402
from ltx_core.text_encoders.gemma.embeddings_connector import Embeddings1DConnector  # noqa: E402
from ltx_core.text_encoders.gemma.embeddings_processor import convert_to_additive_mask  # noqa: E402

CKPT = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled.safetensors"
DUMP = (
    "/home/alex/EriDiffusion/inference-flame/output/audio_context_dump/"
    "ltx2_audio_context.safetensors"
)
OUT = "/home/alex/mojodiffusion/output/ltx2_connector/connector_mask_ref.safetensors"

import os

DEV = os.environ.get(
    "ORACLE_DEV", "cuda" if torch.cuda.is_available() else "cpu"
)


def load_connector(prefix: str, heads: int, head_dim: int) -> Embeddings1DConnector:
    from safetensors import safe_open

    conn = Embeddings1DConnector(
        attention_head_dim=head_dim,
        num_attention_heads=heads,
        num_layers=8,
        positional_embedding_theta=10000.0,
        positional_embedding_max_pos=[4096],
        num_learnable_registers=128,
        rope_type=LTXRopeType.SPLIT,
        double_precision_rope=True,
        apply_gated_attention=True,
    )
    sd = {}
    with safe_open(CKPT, framework="pt", device="cpu") as h:
        pfx = f"model.diffusion_model.{prefix}."
        for k in h.keys():
            if k.startswith(pfx):
                sd[k[len(pfx):]] = h.get_tensor(k)
    missing, unexpected = conn.load_state_dict(sd, strict=False)
    print(f"[{prefix}] loaded {len(sd)} tensors  missing={missing} unexpected={unexpected}")
    assert not missing and not unexpected, "state dict mismatch"
    return conn.to(DEV).to(torch.bfloat16).eval()


def main() -> None:
    from safetensors import safe_open
    from safetensors.torch import save_file

    with safe_open(DUMP, framework="pt", device="cpu") as h:
        v = h.get_tensor("video_context")
        a = h.get_tensor("audio_context")
        vl = int(h.get_tensor("video_len")[0])
    print(f"[dump] video {tuple(v.shape)} audio {tuple(a.shape)} valid_len={vl}")

    mask = torch.zeros(1, 1024, dtype=torch.int64)
    mask[0, :vl] = 1  # right-padded [valid, pad]
    additive = convert_to_additive_mask(mask, torch.bfloat16).to(DEV)

    vconn = load_connector("video_embeddings_connector", 32, 128)
    aconn = load_connector("audio_embeddings_connector", 32, 64)
    with torch.no_grad():
        venc, vmask_out = vconn(v.to(DEV).to(torch.bfloat16), additive)
        aenc, _ = aconn(a.to(DEV).to(torch.bfloat16), additive)
    assert torch.all(vmask_out == 0), "reference must zero the mask after registers"
    print(f"[ref] video_out std={venc.float().std():.4f} audio_out std={aenc.float().std():.4f}")

    import os

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    save_file(
        {
            "video_in": v.to(torch.bfloat16).contiguous(),
            "audio_in": a.to(torch.bfloat16).contiguous(),
            "video_out": venc.cpu().to(torch.bfloat16).contiguous(),
            "audio_out": aenc.cpu().to(torch.bfloat16).contiguous(),
            "valid_len": torch.tensor([float(vl)], dtype=torch.float32),
        },
        OUT,
        metadata={"producer": "ltx2_connector_mask_oracle.py"},
    )
    print(f"[write] {OUT}")


if __name__ == "__main__":
    main()

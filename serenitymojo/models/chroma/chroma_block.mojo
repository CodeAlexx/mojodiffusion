# models/chroma/chroma_block.mojo — Chroma DiT block training surface.
#
# Chroma1-HD is a FLUX-architecture DiT (19 double + 38 single blocks, D=3072,
# H=24, Dh=128, Fmlp=12288, GELU MLP, biases on every linear, q/k rms_norm over
# Dh, joint concat txt-FIRST then img, interleaved RoPE, sdpa scale 1/sqrt(Dh)).
# Its per-block forward (inference: models/dit/chroma_dit.mojo
# double_block_smoke_forward:328 / single_block_smoke_forward:436) is the SAME
# computation graph the Flux block training unit already implements + gates
# (models/flux/block.mojo, gated 28/28 + 9/9 vs torch — see
# models/flux/parity/double_block_parity.mojo / single_block_parity.mojo).
#
# WHY THIS FILE RE-EXPORTS THE FLUX BLOCK (not a new hand-chain):
#   The ONLY block-level differences Chroma has from Flux are the WEIGHT KEY
#   LAYOUT, not the math:
#     - Chroma double block: SEPARATE attn.to_q/.to_k/.to_v (img) and
#       attn.add_q_proj/.add_k_proj/.add_v_proj (txt), each [D,D] WITH bias,
#       and SEPARATE ff.net.0.proj (gate, [Fmlp,D]) + ff.net.2 (out, [D,Fmlp]).
#       Flux fuses qkv into ONE [3D,D] weight. ROW-STACKING Chroma's three
#       [D,D] matrices (q;k;v) into a [3D,D] matrix and their three [D] biases
#       into [3D] reproduces Flux's wqkv/bqkv EXACTLY: the fused linear output
#       [N,3D] is sliced identically. The base weight grad d_wqkv [3D,D]
#       row-decomposes back into d_to_q/d_to_k/d_to_v [D,D].
#     - Chroma single block: SEPARATE attn.to_q/.to_k/.to_v + SEPARATE proj_mlp
#       ([Fmlp,D]). Flux fuses them into linear1 [3D+Fmlp,D]. ROW-STACKING
#       (to_q;to_k;to_v;proj_mlp) reproduces Flux's w1 [3D+Fmlp,D] EXACTLY.
#       proj_out [D, D+Fmlp] == Flux w2 (no fuse needed).
#   The fuse is performed by the loader (models/chroma/weights.mojo), so the
#   trained-block compute is byte-identical to the proven Flux block. The LoRA
#   variants ALREADY model the separate to_q/to_k/to_v + proj_mlp as distinct
#   per-slice adapters (models/flux/lora_block.mojo header), which is precisely
#   Chroma's OneTrainer/diffusers target set (chroma.rs:44-64). So the Chroma
#   LoRA targets map 1:1 onto the Flux LoRA slot scheme:
#     double img : to_q->D_SQ to_k->D_SK to_v->D_SV to_out.0->D_PROJ
#                  ff.net.0.proj->D_MLP0 ff.net.2->D_MLP2
#     double txt : add_q_proj->D_SQ add_k_proj->D_SK add_v_proj->D_SV
#                  to_add_out->D_PROJ ff_context.net.0.proj->D_MLP0
#                  ff_context.net.2->D_MLP2
#     single     : to_q->S_SQ to_k->S_SK to_v->S_SV proj_mlp->S_PMLP proj_out->S_L2
#
# CHROMA DELTAS NOT AT BLOCK LEVEL (out of per-block-backward scope, stack phase):
#   - No guidance vector, no CLIP-pooled vector embed (Flux Dev has both).
#   - Modulation comes from the distilled_guidance_layer APPROXIMATOR producing
#     per-row ModVecs (chroma_dit.mojo approximator_forward:261). The block
#     consumes precomputed ModVecs/SingleModVecs exactly like Flux/Klein, so the
#     per-block backward contract (mod vecs as inputs, grads returned per block)
#     is unchanged.
#
# Tenet 1 (no new ops): re-uses every ops/*_backward arm Flux already composes
# (modulate, layer_norm, rms_norm, linear, sdpa, gelu, rope, cat, gate_residual).
# NONE needed building for this FLUX-variant — all verified PROVEN in MOJO_MODULES.

# The Chroma double/single block training unit IS the Flux block training unit.
# Re-export under chroma_* names so the stack/weights/parity read as Chroma code
# while the proven implementation stays single-sourced.
from serenitymojo.models.flux.block import (
    # modulation carriers (per-stream / single)
    ModVecs as ChromaModVecs,
    SingleModVecs as ChromaSingleModVecs,
    # double block: weights / saved / forward-result / grads
    StreamWeights as ChromaStreamWeights,
    DoubleBlockWeights as ChromaDoubleBlockWeights,
    StreamSaved as ChromaStreamSaved,
    DoubleBlockSaved as ChromaDoubleBlockSaved,
    DoubleBlockForward as ChromaDoubleBlockForward,
    StreamGrads as ChromaStreamGrads,
    DoubleBlockGrads as ChromaDoubleBlockGrads,
    double_block_forward as chroma_double_block_forward,
    double_block_backward as chroma_double_block_backward,
    # single block: weights / saved / forward-result / grads
    SingleBlockWeights as ChromaSingleBlockWeights,
    SingleBlockSaved as ChromaSingleBlockSaved,
    SingleBlockForward as ChromaSingleBlockForward,
    SingleBlockGrads as ChromaSingleBlockGrads,
    single_block_forward as chroma_single_block_forward,
    single_block_backward as chroma_single_block_backward,
)

# LoRA-on-projection variants (separate to_q/to_k/to_v + proj_mlp adapters =
# Chroma's exact target set). Re-exported under chroma_* names.
from serenitymojo.models.flux.lora_block import (
    StreamLora as ChromaStreamLora,
    DoubleBlockLora as ChromaDoubleBlockLora,
    SingleBlockLora as ChromaSingleBlockLora,
    StreamLoraGrads as ChromaStreamLoraGrads,
    DoubleBlockLoraGrads as ChromaDoubleBlockLoraGrads,
    SingleBlockLoraGrads as ChromaSingleBlockLoraGrads,
    DoubleBlockLoraBackward as ChromaDoubleBlockLoraBackward,
    SingleBlockLoraBackward as ChromaSingleBlockLoraBackward,
    double_block_lora_forward as chroma_double_block_lora_forward,
    double_block_lora_backward as chroma_double_block_lora_backward,
    single_block_lora_forward as chroma_single_block_lora_forward,
    single_block_lora_backward as chroma_single_block_lora_backward,
    # LoRA slot indices (Chroma target -> Flux slot map, see header)
    DBL_STREAM_SLOTS, D_SQ, D_SK, D_SV, D_PROJ, D_MLP0, D_MLP2,
    SGL_SLOTS, S_SQ, S_SK, S_SV, S_PMLP, S_L2,
)

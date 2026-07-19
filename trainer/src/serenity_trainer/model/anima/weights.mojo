# model/anima/weights.mojo — REAL Anima (Cosmos MiniTrainDIT) safetensors -> the
# block/base weight structs the proven streamed fwd/bwd consume.
#
# Anima stores its block projections already in the layout the proven block
# expects (net.blocks.{i}.self_attn.{q,k,v,o}_proj, cross_attn.{q,k,v,o}_proj,
# mlp.layer1/layer2, q_norm/k_norm, adaln; net.x_embedder.proj.1, t_embedder,
# t_embedding_norm, final_layer.adaln_modulation/linear). There is NO separate->
# fused row-stack to do (the genuine Chroma-specific compute), so this file simply
# RE-EXPORTS the proven serenitymojo loaders (serenitymojo/models/anima/
# weights.mojo load_anima_stack_base:263 / load_anima_block_weights_f32:145 /
# load_anima_all_blocks_*:205) under the serenity_trainer namespace — 1:1, no
# transcription. This is the Ernie precedent (model/ernie/weights.mojo re-exports).
#
# The checkpoint is the real single-file Anima transformer
# (anima.json checkpoint = .../diffusion_models/anima-base-v1.0.safetensors). The
# streamed fast path swaps F32 block weights in/out per block (resident base +
# one block at a time), so the full 28-block DiT fits a 24GB card.

from serenitymojo.models.anima.weights import (
    AnimaBlockWeights,
    AnimaStackBase,
    load_anima_stack_base,
    load_anima_block_weights,
    load_anima_block_weights_f32,
    load_anima_block_weights_bf16_normf32,
    load_anima_all_blocks_f32,
    load_anima_all_blocks_bf16_normf32,
    verify_anima_stack_shapes,
)

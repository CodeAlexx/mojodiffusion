# model/ernie/weights.mojo — REAL ERNIE-Image transformer safetensors -> the
# block/base weight structs the proven resident-device fwd/bwd consume.
#
# ERNIE stores its block projections ALREADY in the layout the proven block
# expects (self_attention.to_q/to_k/to_v/to_out.0, mlp.gate_proj/up_proj/
# linear_fc2; x_embedder.proj, text_proj, time_embedding, adaLN_modulation.1,
# final_norm.linear, final_linear). There is NO separate->fused row-stack to do
# (the genuine Chroma-specific compute), so this file simply RE-EXPORTS the
# proven serenitymojo loaders (serenitymojo/models/ernie/weights.mojo
# load_ernie_stack_base:229 / load_ernie_all_blocks_bf16_normf32:266) under the
# serenity_trainer namespace — 1:1, no transcription.
#
# The checkpoint is the real sharded ERNIE-Image transformer
# (/home/alex/models/ERNIE-Image/transformer, BF16 on disk). The resident fast
# path keeps large block matrices BF16 + norm vectors F32 (fits a 3090).

from serenitymojo.models.ernie.weights import (
    ErnieBlockWeights,
    ErnieStackBase,
    load_ernie_stack_base,
    load_ernie_all_blocks,
    load_ernie_all_blocks_bf16_normf32,
    load_ernie_block_weights,
    load_ernie_block_weights_bf16_normf32,
)

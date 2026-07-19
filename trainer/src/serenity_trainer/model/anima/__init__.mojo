# model/anima — Anima (Cosmos-Predict2 MiniTrainDIT) training surface, BORROWED
# (import-style re-export) from serenitymojo/models/anima. Mirrors the Ernie
# port precedent (serenity_trainer.model.ernie re-exports the proven serenitymojo
# stack via import; serenity_trainer imports serenitymojo directly with -I).
#
# LIKE Ernie (and UNLIKE Chroma, which needed a port-local row-stack loader to
# translate Flux's pre-fused offload keys), Anima's serenitymojo stack is ALREADY
# a complete, self-contained MiniTrainDIT (28-block, self-attn + cross-attn +
# GELU MLP, AdaLN-LoRA) training vertical: the streamed forward/backward and
# per-adapter AdamW (serenitymojo.models.anima.anima_stack_lora) are the EXACT
# functions the real-loop driver (serenitymojo/training/train_anima_real.mojo:660
# forward / :683 backward / :701 adamw) calls. So there is NO port-local
# orchestration to copy — re-exporting the proven stack under
# serenity_trainer.model.anima is the faithful 1:1 mirror, and avoids the
# transcription drift hand-rewriting gated block math would risk.
#
# The Anima backward MATH is independently proven torch-faithful:
# serenitymojo/models/anima/parity/lora_stack_parity.mojo passed 64/64
# (forward + every LoRA A/B grad, cos>=0.999, nonfinite 0) vs torch.autograd.
#
# Package layout (mirrors model/ernie):
#   anima_block.mojo       — re-export block + lora_block kinds/structs + slots
#   weights.mojo           — re-export AnimaBlockWeights/AnimaStackBase + loaders
#   anima_stack_lora.mojo  — re-export streamed forward/backward (the driver
#                            spine) + per-adapter AdamW + LoRA carrier/save/resume
#   config.mojo            — read serenitymojo/configs/anima.json

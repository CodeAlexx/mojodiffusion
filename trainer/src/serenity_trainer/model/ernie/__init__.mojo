# model/ernie — ERNIE-Image (Baidu single-stream DiT) training surface, BORROWED
# (import-style re-export) from serenitymojo/models/ernie. Mirrors the Chroma
# port precedent (serenity_trainer.model.chroma re-exports the proven Flux block
# math via import; serenity_trainer imports serenitymojo directly).
#
# UNLIKE Chroma (which had to row-stack separate Flux projections in a port-local
# loader), ERNIE's serenitymojo stack is ALREADY a complete, self-contained
# single-stream DiT training vertical: the resident-device forward/backward and
# AdamW step (serenitymojo.models.ernie.ernie_stack_lora) are the EXACT functions
# the real-loop driver (serenitymojo.training.train_ernie_real.mojo) calls. So
# there is NO port-local orchestration to copy — re-exporting the proven stack
# under serenity_trainer.model.ernie is the faithful 1:1 mirror, and avoids the
# transcription drift that hand-re-writing gated block math would risk.
#
# Package layout (mirrors model/chroma):
#   ernie_block.mojo       — re-export block + lora_block kinds/structs
#   weights.mojo           — re-export ErnieBlockWeights/ErnieStackBase + loaders
#   ernie_stack_lora.mojo  — re-export forward/backward (resident-device) + AdamW
#                            + LoRA carrier/save/resume
#   config.mojo            — read serenitymojo/configs/ernie_image.json

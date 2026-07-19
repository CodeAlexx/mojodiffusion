# model/chroma — Chroma1-HD DiT stack package, BORROWED (copied + namespaced)
# from serenitymojo/models/chroma. The Chroma-OWNED logic (block re-export shim,
# stack forward/backward offload, separate->fused weight loader, SerenityTrainer
# raw-key save/resume) is copied here and namespaced to serenity_trainer.
#
# Per the Ideogram4 port precedent (serenity_trainer imports
# serenitymojo.models.dit.ideogram4_dit directly), the PROVEN Flux block math
# (serenitymojo.models.flux.{block,lora_block,flux_stack,flux_stack_lora}), the
# offload streaming infra (serenitymojo.offload.*), the LoRA carrier/AdamW
# (serenitymojo.training.*, flux_lora_adamw_step), and the Chroma approximator
# (serenitymojo.models.dit.chroma_dit) are single-sourced via import — copying
# the entire proven Flux stack would only risk transcription drift on math that
# is already gated. Foundation (serenitymojo/{tensor,io,ops}) is imported
# unchanged.

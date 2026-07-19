# 1:1 port of Serenity modules/model/Flux2Model.py
# Source of truth: /home/alex/Serenity/modules/model/Flux2Model.py
#
# KLEIN == Serenity's FLUX_2 family. is_klein() = not is_dev() =
# transformer.config.num_attention_heads != 48 (Flux2Model.py:232-236).
# The Klein DiT transformer forward (the diffusers Flux2Transformer2DModel that
# Flux2Model drives via model.transformer(...)) is ported in:
#   model/KleinModel.mojo            — the Serenity-conventioned wrapper
#                                      (training fwd tape, hand-chained backward
#                                       -> LoRA d_A/d_B, separate no-grad
#                                       inference forward for the sampler).
#   model/klein/{double_block,single_block,klein_stack,klein_stack_lora,
#                lora_block,lora_adapter,weights}.mojo
#                                    — the borrowed Klein block math (8 double +
#                                      24 single stream blocks, 80 LoRA adapters),
#                                      copied from serenitymojo/models/klein into
#                                      this namespace; foundation imported unchanged.
#
# The latent packing/scaling/timestep/sigma helpers (patchify_latents,
# scale_latents, prepare_latent_image_ids, prepare_text_ids, calculate_timestep_shift)
# and encode_text live with the reference trainer-conventioned predict() port in
# modelSetup/BaseFlux2Setup.mojo (TODO sibling). KleinModel consumes the modvecs
# built from timestep/1000 (BaseFlux2Setup.py:144) per model/klein/weights.mojo.

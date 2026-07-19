# LensFineTuneSetup.mojo — 1:1 port of Serenity pr-1510
# modules/modelSetup/LensFineTuneSetup.py. Mirrors modelSetup/ZImageFineTuneSetup.mojo.
#
# ── EXACT Serenity SOURCE (pr-1510 LensFineTuneSetup.py) ────────────────────
# class LensFineTuneSetup(BaseLensSetup):
#   def create_parameters(self, model, config) -> NamedParameterGroupCollection:
#       parameter_group_collection = NamedParameterGroupCollection()
#       self._create_model_part_parameters(
#           parameter_group_collection, "transformer", model.transformer,
#           config.transformer, freeze=ModuleFilter.create(config),
#           debug=config.debug_mode)
#       return parameter_group_collection
#   def __setup_requires_grad(self, model, config):
#       self._setup_model_part_requires_grad(
#           "transformer", model.transformer, config.transformer, model.train_progress)
#       model.vae.requires_grad_(False)
#       if model.text_encoder is not None:
#           model.text_encoder.requires_grad_(False)
#   def setup_model(self, model, config):
#       params = self.create_parameters(model, config)
#       self.__setup_requires_grad(model, config)
#       init_model_parameters(model, params, self.train_device)
#   def setup_train_device(self, model, config):
#       vae_on_train_device = not config.latent_caching
#       text_encoder_on_train_device = not config.latent_caching
#       if text_encoder_on_train_device:
#           model.materialize_text_encoder(self.train_device)
#       else:
#           model.release_text_encoder()
#       model.vae_to(self.train_device if vae_on_train_device else self.temp_device)
#       model.transformer_to(self.train_device)
#       if model.text_encoder is not None: model.text_encoder.eval()
#       model.vae.eval()
#       if config.transformer.train: model.transformer.train()
#       else: model.transformer.eval()
#   def after_optimizer_step(self, model, config, train_progress):
#       self.__setup_requires_grad(model, config)
# factory.register(BaseModelSetup, LensFineTuneSetup, ModelType.LENS,
#                  TrainingMethod.FINE_TUNE)
#
# ── PORT NOTES ────────────────────────────────────────────────────────────────
# FineTune INHERITS BaseLensSetup.predict / calculate_loss UNCHANGED — the
# noised-latent + flow-target construction is IDENTICAL to the LoRA path
# (modelSetup/LensLoRASetup.mojo::LensLoRASpec.predict). The ONLY difference vs
# the LoRA setup is WHICH parameters are trainable:
#   * LoRA:     transformer/vae/text-encoder FROZEN; the LoRA overlay trains
#               (LensLoRASetup.py:52-62).
#   * FineTune: the FULL transformer trains (minus modules matched by
#               ModuleFilter.create(config), i.e. config.layer_filter as a
#               FREEZE filter — note the inverted role vs LoRA); vae + text encoder
#               FROZEN (LensFineTuneSetup.py:42-49,52-62).
#
# In this Mojo vertical the per-model "setup class" is a ModelSpec conformance + a
# hand-chained backward (no nn.Module wrapper lifecycle). The full-fine-tune spec
# is therefore the SAME BaseLensSetup.predict math as LensLoRASpec, with the
# hand-chained backward returning grads for EVERY frozen base weight instead of
# only the LoRA A/B factors. That full-weight backward is NOT implemented in this
# slice (mirrors modelSetup/ZImageFineTuneSetup.mojo, which is likewise a contract
# stub). The shared predict math (patchify → scale_latents → noise → timestep →
# add_noise → pack → DiT → unpack → unpatchify → flow target) is fully available
# and verified in modelSetup/BaseLensSetup.mojo + modelSetup/LensLoRASetup.mojo;
# the full-fine-tune backward + optimizer-over-all-params is the remaining surface.
#
# TODO (full fine-tune backward): extend model/lens/lens_backward.mojo to return
# per-base-weight grads (d_img_in, d_txt_in, d_txt_norm[i], d_temb*, per-block
# d_w*, d_norm_out, d_proj_out) — the same reverse chain as lens_backward_full_lora
# but accumulating into base-weight grad buffers rather than (or in addition to)
# the LoRA A/B factors — then drive AdamW over all of them with the
# ModuleFilter.create(config) freeze mask applied.

# LensTrainStep.mojo — Lens train-loop glue, ported from Serenity
#   GenericTrainer.train() (the epoch/step orchestration) wired to the Lens
#   ModelSpec (LensLoRASpec). Structurally mirrored on
#   trainer/GenericTrainer.mojo::train_zimage_cadence.
#
# Serenity wiring (how a model reaches the GenericTrainer loop):
#   model_loader.load(...)                       → frozen transformer weights (LensWeights)
#   model_setup.setup_model(...)                 → LoRAModuleWrapper (LensLoraSet)
#   model_setup.create_parameters(...)           → trainable param groups (the LoRA A/B)
#   GenericTrainer.train():  for epoch: for batch:
#       data = model_setup.predict(model, batch, ...)   (BaseLensSetup.predict)
#       loss = model_setup.calculate_loss(model, batch, data, ...)  (flow MSE)
#       loss.backward()                                  (autograd)
#       optimizer.step(); optimizer.zero_grad()
#       model_setup.after_optimizer_step(...)            (re-freeze base each step)
#
# In the Mojo port the per-step recipe is trainer/train_step.mojo::train_step
# (predict → tape MSE → backward → grad-scale/accumulate → clip → AdamW → zero_grad),
# and the Lens ModelSpec is modelSetup/LensLoRASetup.LensLoRASpec (SLICE B): its
# predict() runs BaseLensSetup.predict's noise/sigma/flow-target math + the
# LoRA-overlaid LensTransformer forward, returning StepOutput(predicted, target,
# timestep). after_optimizer_step (re-freeze base) is a no-op here: the base
# LensWeights are never tracked on the tape, so they receive no grad by construction
# (the Mojo analogue of __setup_requires_grad's transformer.requires_grad_(False)).
#
# This file provides:
#   • train_lens        — the model-agnostic loop (= GenericTrainer.train) for Lens,
#                         then a final LoRA save (the simplest faithful entry).
#   • train_lens_cadence— the cadence-wired loop (IDENTICAL step body to train_from)
#                         with the periodic LoRA save wired in at Serenity's
#                         deferred (not-has-gradient) write point, using the verified
#                         Lens saver (save_lens_lora) — the Mojo analogue of
#                         model_saver.save(SAFETENSORS).
#
# Dtype: params/moments/grads BF16; LR factor + grad norm host F32. No persistent F32.

from std.gpu.host import DeviceContext
from serenitymojo.io.dtype import STDtype

from serenity_trainer.util.config.TrainConfig import TrainConfig
from serenity_trainer.util.lr_scheduler_util import LrSchedule, resolve_schedule
from serenity_trainer.trainer.train_step import ParamSlot, StepResult, train_step, TArc
from serenity_trainer.trainer.TrainState import TrainProgress
from serenity_trainer.trainer.GenericTrainer import DriverResult, train_from
from serenity_trainer.modelSetup.LensLoRASetup import LensLoRASpec
from serenity_trainer.modelSaver.lens.LensLoRASaver import save_lens_lora


# EMA decay CAP Serenity uses for the board loss (GenericTrainer.py:940-943).
comptime _LENS_EMA_DECAY_CAP = Float32(0.99)


# train_lens — the model-agnostic loop for a LensLoRASpec, then a final LoRA save.
# train_from runs the EXACT predict→loss→backward→AdamW→after_optimizer_step body
# (it is generic over ModelSpec); LensLoRASpec conforms, so no per-step rewrite is
# needed. The trained adapters (spec.loras) are written once at the end via the
# verified Lens saver (model_saver.save(SAFETENSORS) analogue).
def train_lens[HL: Int, WL: Int, S_TXT: Int](
    mut spec: LensLoRASpec[HL, WL, S_TXT],
    mut slots: List[ParamSlot],
    cfg: TrainConfig,
    steps_per_epoch: Int,
    base_lr: Float32,
    save_destination: String,
    ctx: DeviceContext,
) raises -> DriverResult:
    var res = train_from[LensLoRASpec[HL, WL, S_TXT]](
        spec, slots, cfg, steps_per_epoch, base_lr, TrainProgress(), 0, ctx
    )
    save_lens_lora(spec.loras, save_destination, ctx, STDtype.BF16)
    return res^


# train_lens_cadence — the cadence-wired Lens loop. IDENTICAL step body to
# train_from (it calls the SAME train_step recipe), with the periodic LoRA save
# wired in at Serenity's positions:
#   top of each batch iteration (GenericTrainer.py:695-699): decide save (latched)
#   after the optimizer step + zero_grad (not has_gradient, :714-723): perform save
# Save fires only on an optimizer-step boundary (gradients clear), exactly as
# Serenity defers the write to the next `not has_gradient` point. The save path
# writes spec.loras via save_lens_lora (the Mojo model_saver.save(SAFETENSORS)).
#
# `save_every_opt_steps <= 0` disables periodic saves (final save only). The save
# file is "<save_dir>/lens_lora_step<N>.safetensors".
def train_lens_cadence[HL: Int, WL: Int, S_TXT: Int](
    mut spec: LensLoRASpec[HL, WL, S_TXT],
    mut slots: List[ParamSlot],
    cfg: TrainConfig,
    steps_per_epoch: Int,
    base_lr: Float32,
    start: TrainProgress,
    start_opt_step: Int,
    save_dir: String,
    save_every_opt_steps: Int,
    ctx: DeviceContext,
) raises -> DriverResult:
    var accum = cfg.gradient_accumulation_steps
    if accum < 1:
        accum = 1

    var sched: LrSchedule = resolve_schedule(
        cfg.lr_scheduler_kind,
        cfg.warmup_steps,
        cfg.lr_num_cycles,
        cfg.lr_min_factor,
        steps_per_epoch,
        cfg.epochs,
        accum,
    )

    var prog = start.copy()
    var global_step = start_opt_step       # optimizer-step counter (see train_from)
    var micro_total = 0
    var ema_loss = Float32(0.0)
    var ema_inited = False
    var ema_loss_steps = 0
    var accumulated_loss = Float32(0.0)
    var last_grad_norm = Float32(-1.0)

    # Latched cadence decision (Serenity commands.save()); consumed after the
    # update step's zero_grad.
    var pending_save = False

    for _epoch in range(prog.epoch, cfg.epochs):
        var skip = prog.epoch_step if _epoch == prog.epoch else 0
        for _s in range(skip, steps_per_epoch):
            var micro_idx = micro_total % accum

            # cadence DECISION at top of batch iteration (GenericTrainer.py:695-699).
            # Latched; the WRITE is deferred to the next not-has-gradient point.
            if save_every_opt_steps > 0 and global_step > 0 \
                    and (global_step % save_every_opt_steps) == 0:
                pending_save = True

            var lr = base_lr * Float32(sched.factor(global_step))

            var res: StepResult = train_step(
                spec, slots, cfg, global_step, micro_idx, lr, ctx
            )
            micro_total += 1

            accumulated_loss = accumulated_loss + res.loss / Float32(accum)

            if res.did_update:
                last_grad_norm = res.grad_norm

                ema_loss_steps += 1
                if not ema_inited:
                    ema_loss = accumulated_loss
                    ema_inited = True
                var decay = Float32(1.0) - Float32(1.0) / Float32(ema_loss_steps)
                if decay > _LENS_EMA_DECAY_CAP:
                    decay = _LENS_EMA_DECAY_CAP
                ema_loss = ema_loss * decay + accumulated_loss * (Float32(1.0) - decay)

                accumulated_loss = Float32(0.0)
                global_step += 1

                # PERFORM pending save now that gradients are clear
                # (GenericTrainer.py:714-723). train_step has run AdamW + zero_grad
                # on this boundary, so spec.loras is the consistent snapshot.
                if pending_save:
                    var path = save_dir + String("/lens_lora_step") \
                        + String(global_step) + String(".safetensors")
                    save_lens_lora(spec.loras, path, ctx, STDtype.BF16)
                    pending_save = False

            prog.next_step(cfg.batch_size)

        prog.next_epoch()

    # Final LoRA save (mirrors Serenity's end-of-train model_saver.save).
    var final_path = save_dir + String("/lens_lora_final.safetensors")
    save_lens_lora(spec.loras, final_path, ctx, STDtype.BF16)

    return DriverResult(global_step, micro_total, ema_loss, last_grad_norm, prog^)

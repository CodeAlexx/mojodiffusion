# driver.mojo — the train LOOP, ported from Serenity
# GenericTrainer.train() L611-878 (the epoch/step orchestration around the
# per-step body). The per-step body itself lives in train_step.mojo; this file
# is the model-AGNOSTIC outer loop:
#
#   resolve LR schedule (create_lr_scheduler)         L664-677
#   for epoch in [start, epochs):                     L635-637
#     for batch in data_loader:                       L686
#       (sample/save/backup cadence — stubbed)        L691-723
#       lr = base_lr · lr_schedule.factor(opt_step)   (LambdaLR)
#       train_step(...)                               L728-855  (predict→loss→
#                                                                backward→clip→
#                                                                step→zero_grad)
#       lr_scheduler.step()  ── folded into factor(opt_step)   L (after step)
#       on_optimizer_step / global_step++             train_progress.next_step
#
# Data loading / MGDS is OUT OF SCOPE (no Python, no data pipeline): the driver
# consumes a ModelSpec whose predict() synthesizes / loads its own batch per
# step (matches the port boundary — the model owns its data seam). The number of
# steps is therefore driven by (epochs · steps_per_epoch) here, not by iterating
# a Python DataLoader.
#
# Dtype policy: all tensors BF16; LR factor and grad norm are host F32 scalars.

from std.gpu.host import DeviceContext
from serenity_trainer.util.config.TrainConfig import TrainConfig
from serenity_trainer.modelSetup.BaseModelSetup import ModelSpec
from serenity_trainer.util.lr_scheduler_util import LrSchedule, resolve_schedule
from serenity_trainer.trainer.train_step import ParamSlot, StepResult, train_step, TArc
from serenity_trainer.trainer.TrainState import (
    TrainProgress,
    LoadedTrainState,
    load_train_state,
    save_train_state,
    restore_moments_into_slots,
)
from serenity_trainer.trainer.cadence.SaveBackupCadence import SaveBackupCadence
from serenity_trainer.modelSetup.ZImageLoRASetup import ZImageLoRASpec
from serenity_trainer.util.callbacks.TrainCallbacks import (
    append_train_callback_progress_line_values,
)


# ─────────────────────────────────────────────────────────────────────────────
# DriverResult — host summary of a run (for smokes / logging). EMA loss mirrors
# Serenity's ema_loss board scalar; last_grad_norm is the final pre-clip norm.
@fieldwise_init
struct DriverResult(Copyable, Movable):
    var optimizer_steps: Int     # number of completed optimizer.step() calls
    var micro_steps: Int         # number of micro-steps (batches) processed
    var ema_loss: Float32        # exponential-moving-average of the host loss
    var last_grad_norm: Float32  # final update step's pre-clip global grad norm
    var progress: TrainProgress  # final TrainProgress (for save_train_state / resume)


# EMA decay CAP Serenity uses for the board loss. Serenity ramps the decay
# per update step: ema_loss_decay = min(0.99, 1 - 1/ema_loss_steps), applied over
# `accumulated_loss` (the windowed sum of per-micro losses), and ONLY on optimizer
# (update) steps — NOT per micro-step (GenericTrainer.py:940-943). Kept as a host
# scalar (diagnostic, not model state).
comptime _EMA_DECAY_CAP = Float32(0.99)


# train — run the full loop for a ModelSpec.
#
# Args:
#   spec            : the per-model surface (predict)            [model_setup]
#   slots           : trainable params + AdamW state             [self.parameters]
#   cfg             : TrainConfig (lr, betas, accum, clip, …)    [self.config]
#   steps_per_epoch : micro-steps (batches) per epoch            [approx_length]
#   base_lr         : the configured base learning rate          [config.lr]
#
# The optimizer-step index advances only on accumulation boundaries (every
# `gradient_accumulation_steps` micro-steps), matching LambdaLR being stepped
# once per optimizer.step() in Serenity (lr_scheduler.step after optimizer.step).
def train[S: ModelSpec](
    mut spec: S,
    mut slots: List[ParamSlot],
    cfg: TrainConfig,
    steps_per_epoch: Int,
    base_lr: Float32,
    ctx: DeviceContext,
) raises -> DriverResult:
    # Cold start: resume from the origin (epoch 0, global_step 0, opt_step 0).
    # Mirrors a fresh Serenity run whose model.train_progress is the default
    # TrainProgress() and whose per-param AdamW state['step'] is 0.
    return train_from[S](
        spec, slots, cfg, steps_per_epoch, base_lr, TrainProgress(), 0, ctx
    )


# train_from — the resume-aware loop. Identical to `train` but STARTS from a
# given TrainProgress (the restored counters), mirroring GenericTrainer.train's
# use of self.model.train_progress: it reads train_progress (L614), iterates
# epochs = range(train_progress.epoch, self.config.epochs) (L635), and starts the
# step counter at train_progress.epoch_step (L683). The optimizer moments (m/v)
# must already be restored into `slots` by the caller (restore_moments_into_slots)
# BEFORE this is called — exactly as Serenity rebuilds the optimizer from the
# saved state_dict before entering train() (optimizer_util.init_model_parameters).
#
# `start_opt_step` is the restored optimizer-step count (= Serenity's per-param
# AdamW state['step'] = number of completed optimizer.step() calls). It seeds the
# loop's optimizer-step `global_step` — which drives the LR factor, the AdamW
# bias-correction step (beta ** step), and the SR seed. This is DISTINCT from the
# micro/per-batch counter prog.global_step (which advances every batch via
# TrainProgress.next_step). The two coincide ONLY at accum == 1; for accum > 1,
# prog.global_step == start_opt_step * accum, so seeding the optimizer-step
# counter from start_opt_step (NOT prog.global_step) is required for a bit-exact
# resume at any accum. Callers that lack a saved opt_step (a cold start) pass
# start.global_step, which equals the optimizer-step count for a fresh run (both 0).
def train_from[S: ModelSpec](
    mut spec: S,
    mut slots: List[ParamSlot],
    cfg: TrainConfig,
    steps_per_epoch: Int,
    base_lr: Float32,
    start: TrainProgress,
    start_opt_step: Int,
    ctx: DeviceContext,
) raises -> DriverResult:
    var accum = cfg.gradient_accumulation_steps
    if accum < 1:
        accum = 1

    # ── resolve the LR schedule once (create_lr_scheduler, L664) ──────────────
    # Serenity passes global_step=train_progress.global_step into
    # create_lr_scheduler (L676) so the LambdaLR resumes at the right factor; the
    # factor below is queried at the live global_step, which gives the same result.
    var sched: LrSchedule = resolve_schedule(
        cfg.lr_scheduler_kind,
        cfg.warmup_steps,
        cfg.lr_num_cycles,
        cfg.lr_min_factor,
        steps_per_epoch,
        cfg.epochs,
        accum,
    )

    # ── live progress (GenericTrainer.train: train_progress = model.train_progress,
    #    L614). global_step is the optimizer-step counter restored from the ckpt. ─
    var prog = start.copy()
    # Optimizer-step counter — seeded from the RESTORED optimizer-step count
    # (start_opt_step), NOT from prog.global_step (the micro/per-batch counter).
    # These coincide only at accum == 1; at accum > 1 prog.global_step is
    # start_opt_step*accum, so seeding from start_opt_step is required for a
    # bit-exact resume (drives LR factor, AdamW bias correction, SR seed).
    var global_step = start_opt_step
    var micro_total = 0          # micro-steps processed THIS run (window phase)
    var ema_loss = Float32(0.0)
    var ema_inited = False
    var ema_loss_steps = 0       # update-step counter for ramped EMA decay
    var accumulated_loss = Float32(0.0)  # windowed sum of per-micro (loss/accum)
    var last_grad_norm = Float32(-1.0)

    # ── epoch loop: range(train_progress.epoch, config.epochs) (L635) ─────────
    # A resumed run skips the epochs already completed; within the resume epoch
    # it skips the micro-steps already done (start.epoch_step, L683).
    for _epoch in range(prog.epoch, cfg.epochs):
        # ── step loop (L686). Data is synthesized inside predict per the port
        #    boundary, so we iterate a fixed step budget per epoch. On the resume
        #    epoch the first start.epoch_step micro-steps are skipped (tqdm
        #    initial=train_progress.epoch_step, L683). ─────────────────────────
        var skip = prog.epoch_step if _epoch == prog.epoch else 0
        for _s in range(skip, steps_per_epoch):
            # micro_idx tracks the accumulation window. On resume, Serenity
            # backs up only on optimizer-step boundaries (has_gradient False ⇒
            # zero_grad done), so epoch_step is always a multiple of accum and a
            # resumed window restarts cleanly at micro_idx 0.
            var micro_idx = micro_total % accum

            # LR for THIS optimizer step: base · LambdaLR factor(global_step).
            # The factor folds warmup + scheduler (lr_schedule.mojo). LR is held
            # constant across the micro-steps of one accumulation window (the
            # scheduler only advances on optimizer.step).
            var lr = base_lr * Float32(sched.factor(global_step))

            var res: StepResult = train_step[S](
                spec, slots, cfg, global_step, micro_idx, lr, ctx
            )
            micro_total += 1

            # accumulate windowed loss (Serenity: accumulated_loss += loss/accum;
            # res.loss is the UNSCALED host MSE, so divide by accum to match L780).
            accumulated_loss = accumulated_loss + res.loss / Float32(accum)

            if res.did_update:
                last_grad_norm = res.grad_norm

                # board loss EMA — updated ONLY on optimizer steps, over the
                # windowed accumulated_loss, with ramped decay (GenericTrainer
                # L940-943): decay = min(0.99, 1 - 1/ema_loss_steps).
                ema_loss_steps += 1
                if not ema_inited:
                    ema_loss = accumulated_loss          # ema_loss or accumulated_loss
                    ema_inited = True
                var decay = Float32(1.0) - Float32(1.0) / Float32(ema_loss_steps)
                if decay > _EMA_DECAY_CAP:
                    decay = _EMA_DECAY_CAP
                ema_loss = ema_loss * decay + accumulated_loss * (Float32(1.0) - decay)

                accumulated_loss = Float32(0.0)          # reset window (L950)
                global_step += 1     # optimizer-step counter (LR/bias-corr/SR seed)

            # train_progress.next_step(batch_size) — runs EVERY micro-step/batch
            # (GenericTrainer.py:971), advancing epoch_step / epoch_sample /
            # global_step of the PROGRESS record (distinct from the optimizer-step
            # `global_step` above, which the existing loop uses for LR + AdamW).
            prog.next_step(cfg.batch_size)

        # train_progress.next_epoch() at the end of each epoch (L977).
        prog.next_epoch()

    return DriverResult(
        global_step, micro_total, ema_loss, last_grad_norm, prog^
    )


# train_with_progress_file — real trainer-side live progress bridge for the
# native Serenity UI. It runs the same model-agnostic loop as `train`, and emits
# Serenity-shaped callback event lines after `TrainProgress.next_step`, matching
# GenericTrainer.py's callback timing.
def train_with_progress_file[S: ModelSpec](
    mut spec: S,
    mut slots: List[ParamSlot],
    cfg: TrainConfig,
    steps_per_epoch: Int,
    base_lr: Float32,
    progress_file_path: String,
    ctx: DeviceContext,
) raises -> DriverResult:
    return train_from_with_progress_file[S](
        spec,
        slots,
        cfg,
        steps_per_epoch,
        base_lr,
        TrainProgress(),
        0,
        progress_file_path,
        ctx,
    )


def train_from_with_progress_file[S: ModelSpec](
    mut spec: S,
    mut slots: List[ParamSlot],
    cfg: TrainConfig,
    steps_per_epoch: Int,
    base_lr: Float32,
    start: TrainProgress,
    start_opt_step: Int,
    progress_file_path: String,
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
    var global_step = start_opt_step
    var micro_total = 0
    var ema_loss = Float32(0.0)
    var ema_inited = False
    var ema_loss_steps = 0
    var accumulated_loss = Float32(0.0)
    var last_grad_norm = Float32(0.0)
    var display_loss = Float32(0.0)
    var display_smooth_loss = Float32(0.0)

    for _epoch in range(prog.epoch, cfg.epochs):
        var skip = prog.epoch_step if _epoch == prog.epoch else 0
        for _s in range(skip, steps_per_epoch):
            var micro_idx = micro_total % accum
            var lr = base_lr * Float32(sched.factor(global_step))

            var res: StepResult = train_step[S](
                spec, slots, cfg, global_step, micro_idx, lr, ctx
            )
            micro_total += 1
            display_loss = res.loss

            accumulated_loss = accumulated_loss + res.loss / Float32(accum)

            if res.did_update:
                last_grad_norm = res.grad_norm

                ema_loss_steps += 1
                if not ema_inited:
                    ema_loss = accumulated_loss
                    ema_inited = True
                var decay = Float32(1.0) - Float32(1.0) / Float32(ema_loss_steps)
                if decay > _EMA_DECAY_CAP:
                    decay = _EMA_DECAY_CAP
                ema_loss = ema_loss * decay + accumulated_loss * (Float32(1.0) - decay)
                display_loss = accumulated_loss
                display_smooth_loss = ema_loss

                accumulated_loss = Float32(0.0)
                global_step += 1
            elif ema_inited:
                display_smooth_loss = ema_loss
            else:
                display_smooth_loss = display_loss

            prog.next_step(cfg.batch_size)
            append_train_callback_progress_line_values(
                progress_file_path.copy(),
                prog.epoch,
                prog.epoch_step,
                prog.global_step,
                steps_per_epoch,
                cfg.epochs,
                display_loss,
                display_smooth_loss,
                last_grad_norm,
                lr,
                String("Training ..."),
            )

        prog.next_epoch()
        append_train_callback_progress_line_values(
            progress_file_path.copy(),
            prog.epoch,
            prog.epoch_step,
            prog.global_step,
            steps_per_epoch,
            cfg.epochs,
            display_loss,
            display_smooth_loss,
            last_grad_norm,
            base_lr * Float32(sched.factor(global_step)),
            String("Training ..."),
        )

    return DriverResult(
        global_step, micro_total, ema_loss, last_grad_norm, prog^
    )


# ─────────────────────────────────────────────────────────────────────────────
# train_resume — load a checkpoint from `state_dir`, restore the AdamW moments
# into `slots` and the TrainProgress, then continue the loop. This is the Mojo
# analogue of Serenity's resume path:
#   start():  model.train_progress / optimizer_state_dict / ema_state_dict are
#             read by InternalModelLoaderMixin._load_internal_data, then
#             init_model_parameters rebinds them into the live optimizer/EMA.
#   train():  iterates from train_progress.epoch / global_step.
#
# The caller must pass `slots` built by the SAME setup (same count/order) used
# when the checkpoint was saved — the moments rebind by slot index (TrainState
# header). LoRA weights themselves are loaded by the model loader (the LoRA
# saver/loader, e.g. ZImageLoRASaver) BEFORE this, exactly as Serenity loads the
# adapter weights via the model loader and the optimizer state via the internal
# mixin. EMA restore (restore_ema) is wired by the model unit that owns EMA slots.
def train_resume[S: ModelSpec](
    mut spec: S,
    mut slots: List[ParamSlot],
    cfg: TrainConfig,
    steps_per_epoch: Int,
    base_lr: Float32,
    state_dir: String,
    ctx: DeviceContext,
) raises -> DriverResult:
    var st: LoadedTrainState = load_train_state(state_dir, ctx)
    restore_moments_into_slots(st, slots, ctx)
    var start = st.prog
    # Seed the optimizer-step counter from the restored opt_step (= per-param
    # AdamW state['step']), NOT from start.global_step (the micro counter). They
    # differ at accum > 1; this is the second half of the accum>1 resume fix.
    var start_opt_step = st.opt_step
    return train_from[S](
        spec, slots, cfg, steps_per_epoch, base_lr, start, start_opt_step, ctx
    )


# ─────────────────────────────────────────────────────────────────────────────
# train_zimage_cadence — the cadence-wired Z-Image loop. IDENTICAL step body to
# train_from (it calls the SAME train_step recipe — the loop is NOT rewritten),
# with GenericTrainer's save/backup cadence wired in at the Serenity positions:
#
#   top of each batch iteration (GenericTrainer.py:695-699):
#     if __needs_backup(train_progress):  commands.backup()
#     if __needs_save(train_progress):    commands.save()
#   then, once gradients are clear (not has_gradient, after an optimizer step ⇒
#   zero_grad done, GenericTrainer.py:714-723):
#     if backup: __backup(...)
#     if save:   __save(...)
#
# Serenity DEFERS the actual write until an update-step boundary (`has_gradient`
# False) so .grad is None at backup time. We mirror that exactly: we LATCH the
# needs-save / needs-backup decisions when they fire (mirroring commands.save() /
# commands.backup()), and PERFORM the write only after `res.did_update` (the
# accumulation boundary, where train_step has already run AdamW + zero_grad).
#
# Specialized on ZImageLoRASpec so the cadence can read the trained adapters
# (`spec.loras`) for save_zimage_lora / the INTERNAL TrainState backup — the
# verified Z-Image saver. `ema` is the (possibly empty) EMA buffer list for the
# TrainState backup; the port models no EMA for Z-Image, so callers pass [].
#
# Returns the same DriverResult as train_from (final TrainProgress for resume).
def train_zimage_cadence[HL: Int, WL: Int, CAPLEN: Int](
    mut spec: ZImageLoRASpec[HL, WL, CAPLEN],
    mut slots: List[ParamSlot],
    cfg: TrainConfig,
    steps_per_epoch: Int,
    base_lr: Float32,
    start: TrainProgress,
    start_opt_step: Int,
    mut cadence: SaveBackupCadence,
    ema: List[TArc],
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

    var prog = start
    # Optimizer-step counter seeded from the restored opt_step (see train_from) —
    # NOT prog.global_step (the micro counter); required for accum>1 resume parity.
    var global_step = start_opt_step
    var micro_total = 0
    var ema_loss = Float32(0.0)
    var ema_inited = False
    var ema_loss_steps = 0
    var accumulated_loss = Float32(0.0)
    var last_grad_norm = Float32(-1.0)

    # Latched cadence decisions (Serenity's commands.save()/backup() flags). Set
    # at the top of the iteration; consumed after the update step's zero_grad.
    var pending_save = False
    var pending_backup = False

    for _epoch in range(prog.epoch, cfg.epochs):
        var skip = prog.epoch_step if _epoch == prog.epoch else 0
        for _s in range(skip, steps_per_epoch):
            var micro_idx = micro_total % accum

            # ── cadence DECISION at top of batch iteration (GenericTrainer.py:695-699).
            #    __needs_backup → commands.backup(); __needs_save → commands.save().
            #    Latched into pending_* (the command queue) — the WRITE is deferred
            #    to the next `not has_gradient` point, exactly as Serenity does. ──
            if cadence.should_backup(prog):
                pending_backup = True
            if cadence.should_save(prog):
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
                if decay > _EMA_DECAY_CAP:
                    decay = _EMA_DECAY_CAP
                ema_loss = ema_loss * decay + accumulated_loss * (Float32(1.0) - decay)

                accumulated_loss = Float32(0.0)
                global_step += 1

                # ── PERFORM pending save/backup now that gradients are clear
                #    (GenericTrainer.py:714-723: `if not has_gradient:` → after an
                #    optimizer step + zero_grad). train_step has run AdamW + dropped
                #    accum on this boundary, so the persisted state is consistent. ─
                #
                # FAITHFUL PROGRESS VALUE (resume off-by-one fix). Serenity defers
                # the write to the TOP of the NEXT iteration (`if not has_gradient:`,
                # GenericTrainer.py:714), at which point train_progress.next_step()
                # (L971) has ALREADY run for the update iteration → Serenity
                # persists epoch_step = m+1 (the count of COMPLETED micro-steps). The
                # Mojo loop performs the write here, BEFORE prog.next_step() below, so
                # we persist a COPY of prog advanced by one next_step — giving the
                # SAME epoch_step/global_step Serenity writes for this logical save.
                # Without this the backup would persist epoch_step = m, and resume's
                # `skip = prog.epoch_step; range(skip, …)` would RE-RUN micro-step m
                # (re-applying the optimizer step whose moments were already saved) →
                # one extra AdamW step vs Serenity. The DECISION checks above
                # (should_backup/should_save) correctly stay on the pre-advance prog;
                # only the WRITE's prog needs the +1. (Bug 2 fix.)
                if pending_backup or pending_save:
                    var write_prog = prog
                    write_prog.next_step(cfg.batch_size)
                    if pending_backup:
                        # global_step here is the post-increment optimizer-step count
                        # (= completed optimizer.step() calls = Serenity per-param
                        # state['step']). Persist it so resume restores the AdamW bias
                        # correction at any accum (TrainState opt_step). Backup fires
                        # AFTER the AdamW step + zero_grad (not has_gradient), so this
                        # is the consistent snapshot point.
                        _ = cadence.backup(
                            spec.loras, slots, ema, write_prog, global_step, ctx
                        )
                        pending_backup = False
                    if pending_save:
                        _ = cadence.save_lora(spec.loras, write_prog, ctx)
                        pending_save = False

            prog.next_step(cfg.batch_size)

        prog.next_epoch()

    return DriverResult(
        global_step, micro_total, ema_loss, last_grad_norm, prog
    )

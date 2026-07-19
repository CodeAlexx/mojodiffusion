# SampleCadence.mojo — sample-during-training cadence, 1:1 port.
#
# Ports Serenity's "should we sample now?" decision + the prompt iteration that
# feeds the sampler. The SOURCE OF TRUTH is Serenity's own .py — every rule
# below cites the exact line.
#
# Three Serenity files are stitched here:
#   1. modules/trainer/GenericTrainer.py   — __needs_sample / the train-loop callsite
#                                             / __sample_during_training (prompt list)
#   2. modules/util/TimedActionMixin.py     — repeating_action_needed / single_action_elapsed
#                                             (the actual cadence arithmetic)
#   3. modules/util/enum/TimeUnit.py        — the EPOCH/STEP/SECOND/... unit enum
#
# __needs_sample (GenericTrainer.py:524-529) is the AND of two helpers:
#     single_action_elapsed("sample_skip_first", sample_skip_first, sample_after_unit, p)   # delay gate
#         AND
#     repeating_action_needed("sample", sample_after, sample_after_unit, p)  # interval gate
#                                                  ^ start_at_zero defaults True (NOT overridden, L527-528)
#
# REUSE (not reimplementation): the cadence arithmetic and the TimeUnit codes are
# the SAME 1:1 ports the sibling SaveBackupCadence.mojo already uses. This struct
# therefore mirrors SaveBackupCadence exactly — it HOLDS a TimedActionMixin and
# calls its single_action_elapsed / repeating_action_needed, and branches on the
# shared TU_* codes. It does NOT redefine TimedActionMixin, TrainProgress, or the
# TimeUnit codes (earlier the shared modules were thought to be stubs; they are
# complete + verified, so the duplicates are removed). One canonical TrainProgress
# (trainer.TrainState.TrainProgress) drives ALL cadence checks — sample, save,
# backup — matching Serenity's single train_progress object.
#
# This struct does NOT run the sampler itself — sample_zimage (ZImageSampler.mojo)
# is the actual sample; this struct decides WHEN + supplies the prompt list.
#
# Dtype policy: pure host scalars (counters, seconds, interval). No Tensor, no GPU.

from std.collections.string import String

# Canonical, shared 1:1 ports — identical types/codes to SaveBackupCadence.mojo so
# one train_progress object drives every cadence check (sample/save/backup).
from serenity_trainer.trainer.TrainState import TrainProgress
from serenity_trainer.util.TimedActionMixin import TimedActionMixin


# ─────────────────────────────────────────────────────────────────────────────
# SampleCadence — owns the cadence config + the TimedActionMixin clock state, and
# exposes should_sample (decision) + the prompt list the driver feeds the sampler.
#
# Serenity keeps the previous-action map on the trainer (via TimedActionMixin);
# we hold the mixin here so the wall-clock (SECOND/MINUTE/HOUR) cadence works
# across calls, matching GenericTrainer's single TimedActionMixin instance — the
# same pattern SaveBackupCadence uses.
#
# Config fields mirror TrainConfig.py:
#   sample_after        (float) — interval                  -> repeating_action_needed.interval
#   sample_after_unit   (TimeUnit code, TU_*)               -> the unit both helpers branch on
#   sample_skip_first   (float) — delay before any sampling -> single_action_elapsed.delay
# prompts: the resolved sample-prompt list (one String per SampleConfig.prompt),
# mirroring the prompts loaded from config.sample_definition_file_name in
# __sample_during_training (GenericTrainer.py:298-307).
struct SampleCadence(Movable):
    var sample_after: Float64
    var sample_after_unit: Int
    var sample_skip_first: Float64
    var prompts: List[String]
    var timed: TimedActionMixin

    # __init__ constructs the TimedActionMixin (its __init__ captures the run-start
    # monotonic time + inits the previous-action map — TimedActionMixin.py:8-11).
    def __init__(
        out self,
        sample_after: Float64,
        sample_after_unit: Int,
        sample_skip_first: Float64,
        var prompts: List[String],
    ):
        self.sample_after = sample_after
        self.sample_after_unit = sample_after_unit
        self.sample_skip_first = sample_skip_first
        self.prompts = prompts^
        self.timed = TimedActionMixin()

    # ── should_sample — __needs_sample (GenericTrainer.py:524-529) ────────────
    # The exact AND of the two TimedActionMixin helpers:
    #   single_action_elapsed("sample_skip_first", sample_skip_first, unit, p)   # skip-first delay
    #     AND
    #   repeating_action_needed("sample", sample_after, unit, p, start_at_zero=True)  # interval
    #
    # start_at_zero defaults True and __needs_sample does NOT override it
    # (GenericTrainer.py:527-528) — so sampling can fire at step/epoch 0, unlike
    # save/backup (start_at_zero=False).
    #
    # Mutating because the time-unit branches advance the mixin's previous-action
    # clock (TimedActionMixin.py:90/99/108). For EPOCH/STEP units it is
    # side-effect-free.
    #
    # Python `a and b` short-circuits: when single_action_elapsed is False the
    # RHS is not evaluated, so the time-unit interval clock is NOT advanced while
    # still inside the skip-first delay window. Mojo `and` short-circuits the same
    # way, so the call order below is faithful.
    #
    # WIRING (see module footer): call once per micro-step at the top of the
    # train-loop body, exactly where GenericTrainer.py:691 evaluates
    # __needs_sample(train_progress), passing the driver's live TrainProgress.
    def should_sample(mut self, progress: TrainProgress) raises -> Bool:
        var elapsed = self.timed.single_action_elapsed(
            "sample_skip_first",
            self.sample_skip_first,
            self.sample_after_unit,
            progress,
        )
        if not elapsed:
            return False
        return self.timed.repeating_action_needed(
            "sample",
            self.sample_after,
            self.sample_after_unit,
            progress,
            True,                  # start_at_zero=True (default, GenericTrainer.py:527-528)
        )

    # ── prompt iteration — __sample_during_training prompt list ───────────────
    # GenericTrainer.py:294-307 resolves sample_params_list to the SampleConfig
    # list (each carrying .prompt); __sample_loop (L217-219) then iterates the
    # ENABLED entries. Here `prompts` already holds the enabled prompt strings, so
    # iteration is a straight pass over the list. The driver calls sample_zimage
    # once per prompt (mirroring the model_sampler.sample call at L262-269).
    def num_prompts(self) -> Int:
        return len(self.prompts)

    def prompt(self, i: Int) raises -> String:
        return self.prompts[i]


# ─────────────────────────────────────────────────────────────────────────────
# WIRING into GenericTrainer.train (trainer/GenericTrainer.mojo)
# ─────────────────────────────────────────────────────────────────────────────
# Serenity ref: GenericTrainer.py:691-694 — the cadence check sits at the top of
# the per-batch body, BEFORE train_step, evaluated on the MICRO-step counter
# (train_progress.global_step, advanced every batch by next_step at L971). It then
# fires the sample later (L714-715) on optimizer-step boundaries.
#
# STEP-UNIT FIDELITY (important): Serenity's global_step is a MICRO-step counter
# (one increment per batch), and __needs_sample is checked every micro-step. The
# existing GenericTrainer.mojo `global_step` is an OPTIMIZER-step counter, so with
# gradient_accumulation_steps > 1 these differ. For faithful STEP-unit sample
# cadence the driver MUST pass a TrainProgress whose global_step is the MICRO-step
# counter (advanced via TrainProgress.next_step(batch_size) every batch), and call
# should_sample every micro-step — NOT only on did_update. Flagged for the driver
# author.
#
# 1. Construct once, before the epoch loop, from the run's sample config + the
#    prompt list loaded from sample_definition_file_name:
#
#       var cadence = SampleCadence(
#           sample_after      = cfg.sample_after,        # add to TrainConfig (TrainConfig.py)
#           sample_after_unit = cfg.sample_after_unit,   # TU_* code
#           sample_skip_first = cfg.sample_skip_first,
#           prompts           = load_sample_prompts(...) ^,  # List[String]
#       )
#       var prog = TrainProgress()                       # canonical 4-field progress
#
# 2. Inside the MICRO-step loop, advance `prog` with TrainProgress.next_step /
#    next_epoch (the same object that drives save/backup cadence), then check the
#    cadence every micro-step (GenericTrainer.py:691):
#
#       if cadence.should_sample(prog):
#           for pi in range(cadence.num_prompts()):
#               var out = sample_zimage[HL, WL, CAPLEN](
#                   cond=encode(cadence.prompt(pi)), uncond=..., seed=...,
#                   diffusion_steps=..., cfg_scale=..., timestep_shift=...,
#                   num_latent_channels=..., weights=..., loras=..., vae=..., ctx=ctx,
#               )
#               # save out.image to the samples dir (modelSaver path)
#       prog.next_step(batch_size)
#
#    For EPOCH-unit cadence the repeating gate requires epoch_step == 0
#    (TimedActionMixin.py:65-68), so the check naturally fires only at the first
#    step of qualifying epochs — next_epoch() resets epoch_step to 0.
#
# 3. The driver still owns the loop (per the port boundary) — this struct ADDS the
#    cadence + prompt iteration; it does not rewrite train(). The SAME `prog` is
#    handed to SaveBackupCadence.should_save / should_backup, so one TrainProgress
#    drives all three cadence checks exactly as Serenity does.

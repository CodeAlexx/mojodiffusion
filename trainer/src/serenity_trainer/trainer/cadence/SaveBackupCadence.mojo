# SaveBackupCadence.mojo — 1:1 port of Serenity's save + backup cadence.
#
# Source of truth: modules/trainer/GenericTrainer.py + modules/util/TimedActionMixin.py.
# This ports the SAVE (LoRA every `save_every`) and BACKUP (full TrainState every
# `backup_after`) cadence that GenericTrainer.train() drives around the step loop —
# WITHOUT touching the existing train loop body (train_step.mojo) or the outer loop
# structure (GenericTrainer.mojo). It supplies:
#
#   * should_save(progress)   ── GenericTrainer.__needs_save   (GenericTrainer.py:536-541)
#   * should_backup(progress) ── GenericTrainer.__needs_backup (GenericTrainer.py:531-534)
#   * save_lora(...)          ── GenericTrainer.__save         (GenericTrainer.py:478-522)
#   * backup(...)             ── GenericTrainer.__backup       (GenericTrainer.py:431-476)
#
# The decision predicates are the TimedActionMixin port (repeating_action_needed /
# single_action_elapsed), and the path/filename construction mirrors __save / __backup
# (output_model_destination + workspace_dir/save + workspace_dir/backup, with
# get_string_timestamp() + train_progress.filename_string()).
#
# WHAT THIS FILE PORTS FAITHFULLY vs WHAT IS A DOCUMENTED SEAM:
#   - The CADENCE (when to save/backup) is exact (TimedActionMixin + __needs_*).
#   - The SAVE writes the Z-Image LoRA via the verified ZImageLoRASaver
#     (save_zimage_lora) — the Mojo analogue of model_saver.save(SAFETENSORS).
#   - The BACKUP in Serenity writes the FULL TrainState via
#     model_saver.save(ModelFormat.INTERNAL) (GenericTrainer.py:448-454). The
#     Z-Image INTERNAL saver (ZImageModelSaver) is NOT yet ported, so backup here
#     writes the SAME trained adapter state into the backup dir (the only model
#     state that changes during LoRA training) under the INTERNAL backup path/
#     naming. This is the faithful subset until ZImageModelSaver lands; the path,
#     naming, timestamp, and cadence are byte-for-byte Serenity. Flagged.
#   - Schedule-free optimizer eval()/train() toggles, EMA copy_ema_to/copy_temp_to,
#     torch_gc, callbacks, tensorboard, rolling-backup pruning, and __save_backup_config
#     are GenericTrainer plumbing OUTSIDE the cadence-decision + write core; they are
#     not modeled here (no schedule-free opt / EMA / TB in the port). Cited inline.
#
# Reuses ONLY: the verified ZImageLoRASaver, the ported TimedActionMixin/TimeUnit/
# TrainProgress/time_util/path_util, and std.os.makedirs. No new model code.
#
# Dtype: LoRA written BF16 verbatim (save_zimage_lora default), matching Serenity's
# output_dtype path for a bf16 run.

from std.os import makedirs
from std.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype

# TrainProgress is imported from trainer.TrainState (the copy the driver threads
# through train_from / DriverResult). It is the same 1:1 port as
# util/TrainProgress.mojo; using TrainState's keeps the cadence's type identical
# to the live progress value the loop passes in (no cross-type conversion).
from serenity_trainer.trainer.TrainState import TrainProgress
from serenity_trainer.util.TimedActionMixin import TimedActionMixin
from serenity_trainer.util.enum.TimeUnit import TU_NEVER, TU_MINUTE, TU_STEP
from serenity_trainer.util.time_util import get_string_timestamp
from serenity_trainer.util.path_util import canonical_join
from serenity_trainer.model.ZImageModel import ZImageLoraSet
from serenity_trainer.modelSaver.zImage.ZImageLoRASaver import save_zimage_lora
from serenity_trainer.trainer.train_step import ParamSlot, TArc
from serenity_trainer.trainer.TrainState import save_train_state


# ─────────────────────────────────────────────────────────────────────────────
# SaveBackupConfig — the cadence + path subset of Serenity's TrainConfig that
# GenericTrainer.__save / __backup / __needs_* consume. Mirrors the field NAMES
# and DEFAULTS from TrainConfig.py:
#   workspace_dir            (TrainConfig.py:351, default "workspace/run")
#   output_model_destination (TrainConfig.py:376, default "models/model.safetensors")
#   save_filename_prefix     (TrainConfig.py:555, default "")
#   save_every               (TrainConfig.py:552, default 0)        ── int interval
#   save_every_unit          (TrainConfig.py:553, default NEVER)
#   save_skip_first          (TrainConfig.py:554, default 0)
#   backup_after             (TrainConfig.py:547, default 30)       ── float interval
#   backup_after_unit        (TrainConfig.py:548, default MINUTE)
#
# These live in a dedicated cadence-config struct (NOT in the port's TrainConfig,
# which is the "shared pipeline scalars" subset by design) so the workspace/path
# concerns stay cohesive with the cadence that uses them.
@fieldwise_init
struct SaveBackupConfig(Copyable, Movable):
    var workspace_dir: String
    var output_model_destination: String
    var save_filename_prefix: String
    # save cadence (GenericTrainer.__needs_save, GenericTrainer.py:536-541)
    var save_every: Int          # config.save_every       (int)
    var save_every_unit: Int     # config.save_every_unit  (TimeUnit code)
    var save_skip_first: Int     # config.save_skip_first
    # backup cadence (GenericTrainer.__needs_backup, GenericTrainer.py:531-534)
    var backup_after: Float64    # config.backup_after     (float)
    var backup_after_unit: Int   # config.backup_after_unit (TimeUnit code)

    # Defaults mirror TrainConfig.py's defaults (cited above). With these defaults
    # save is OFF (save_every_unit=NEVER) and backup is every 30 minutes — exactly
    # Serenity's out-of-box behavior.
    @staticmethod
    def defaults() -> SaveBackupConfig:
        return SaveBackupConfig(
            workspace_dir=String("workspace/run"),
            output_model_destination=String("models/model.safetensors"),
            save_filename_prefix=String(""),
            save_every=0,
            save_every_unit=TU_NEVER,
            save_skip_first=0,
            backup_after=Float64(30.0),
            backup_after_unit=TU_MINUTE,
        )


# ─────────────────────────────────────────────────────────────────────────────
# SaveBackupCadence — holds the cadence config + the TimedActionMixin state, and
# exposes should_save / should_backup (decision) and save_lora / backup (write).
#
# Serenity keeps the previous-action map on the trainer (via TimedActionMixin);
# we hold the mixin here so the wall-clock (MINUTE/SECOND/HOUR) cadence works
# across calls, matching GenericTrainer's single TimedActionMixin instance.
struct SaveBackupCadence(Movable):
    var cfg: SaveBackupConfig
    var timed: TimedActionMixin

    def __init__(out self, cfg: SaveBackupConfig):
        self.cfg = cfg
        self.timed = TimedActionMixin()

    # ── should_save ── GenericTrainer.__needs_save (GenericTrainer.py:536-541) ──
    #   single_action_elapsed("save_skip_first", save_skip_first, save_every_unit)
    #   AND repeating_action_needed("save", save_every, save_every_unit, start_at_zero=False)
    # i.e. only after the skip-first delay has elapsed, then every save_every units
    # (NOT at step 0 — start_at_zero=False).
    def should_save(mut self, progress: TrainProgress) raises -> Bool:
        # Python `and` SHORT-CIRCUITS (GenericTrainer.py:537-541): the repeating
        # check is only evaluated once the skip-first delay has elapsed. This
        # matters because repeating_action_needed has a side effect for the
        # wall-clock units (it stamps _previous_action). Evaluate elapsed first
        # and bail before touching the "save" clock — matching Serenity exactly.
        var elapsed = self.timed.single_action_elapsed(
            "save_skip_first",
            Float64(self.cfg.save_skip_first),
            self.cfg.save_every_unit,
            progress,
        )
        if not elapsed:
            return False
        return self.timed.repeating_action_needed(
            "save",
            Float64(self.cfg.save_every),
            self.cfg.save_every_unit,
            progress,
            False,                 # start_at_zero=False (GenericTrainer.py:540)
        )

    # ── should_backup ── GenericTrainer.__needs_backup (GenericTrainer.py:531-534)
    #   repeating_action_needed("backup", backup_after, backup_after_unit, start_at_zero=False)
    def should_backup(mut self, progress: TrainProgress) raises -> Bool:
        return self.timed.repeating_action_needed(
            "backup",
            self.cfg.backup_after,
            self.cfg.backup_after_unit,
            progress,
            False,                 # start_at_zero=False (GenericTrainer.py:533)
        )

    # ── save_lora ── GenericTrainer.__save (GenericTrainer.py:478-522) ──────────
    # save_path = workspace_dir/save/<prefix><timestamp>-save-<filename_string><ext>
    # (GenericTrainer.py:483-487). ext = ".safetensors" for the LoRA SAFETENSORS
    # format. Writes via the verified ZImageLoRASaver (the Mojo model_saver.save).
    #
    # OMITTED (cited): EMA copy_ema_to/copy_temp_to (L492-493,519-520 — no EMA in
    # port), schedule-free optimizer eval()/train() (L496-508 — no schedule-free
    # opt), torch_gc (L479,522), callbacks/print (L481,488-489). These are
    # plumbing around the same write; the path + filename + format are exact.
    def save_lora(
        mut self, set: ZImageLoraSet, progress: TrainProgress, ctx: DeviceContext
    ) raises -> String:
        var save_dir = canonical_join(self.cfg.workspace_dir, String("save"))
        makedirs(save_dir, exist_ok=True)

        var fname = (
            self.cfg.save_filename_prefix
            + get_string_timestamp()
            + "-save-"
            + progress.filename_string()
            + ".safetensors"        # ModelFormat.SAFETENSORS.file_extension()
        )
        var save_path = canonical_join(save_dir, fname)
        save_zimage_lora(set, save_path, ctx, STDtype.BF16)
        return save_path

    # ── backup ── GenericTrainer.__backup (GenericTrainer.py:431-476) ───────────
    # backup_name = <timestamp>-backup-<filename_string>  (GenericTrainer.py:436)
    # backup_path = workspace_dir/backup/<backup_name>    (GenericTrainer.py:437)
    # Serenity then model_saver.save(ModelFormat.INTERNAL, backup_path) (L448-454).
    # The INTERNAL save persists the FULL resumable TrainState: the LoRA adapter
    # weights PLUS the optimizer state + TrainProgress (+ EMA) — the
    # InternalModelSaverMixin._save_internal_data contract. The Mojo port already
    # has both halves verified:
    #   * adapter weights      → save_zimage_lora      (ZImageLoRASaver, 630 keys)
    #   * optimizer m/v + prog → save_train_state      (trainer/TrainState.mojo)
    # so backup writes BOTH into backup_path, giving a checkpoint that
    # train_resume(state_dir=backup_path) can restore. Path / naming / timestamp /
    # cadence are byte-for-byte Serenity's __backup.
    #
    # `slots` are the driver's optimizer slots (the AdamW moments to persist) and
    # `ema` is the EMA buffer list (empty ⇒ EMA disabled, matching `if model.ema:`
    # InternalModelSaverMixin.py:29). save_train_state intentionally drops .accum
    # (backup fires only on an update-step boundary after zero_grad — see its
    # header; matches GenericTrainer.py:695-723 enqueuing backup only when
    # `not has_gradient`).
    #
    # OMITTED (cited): __save_backup_config (L456 args/concepts/samples copy —
    # config plumbing), rolling-backup __prune_backups (L467-468), schedule-free
    # opt toggles (L440-442,472-474), torch_gc (L432,476), callbacks (L434,446).
    def backup(
        mut self,
        set: ZImageLoraSet,
        slots: List[ParamSlot],
        ema: List[TArc],
        progress: TrainProgress,
        opt_step: Int,
        ctx: DeviceContext,
    ) raises -> String:
        var backup_name = (
            get_string_timestamp() + "-backup-" + progress.filename_string()
        )
        var backup_dir = canonical_join(self.cfg.workspace_dir, String("backup"))
        var backup_path = canonical_join(backup_dir, backup_name)
        makedirs(backup_path, exist_ok=True)

        # 1) adapter weights (the trained LoRA) — INTERNAL layout names the adapter
        #    file lora.safetensors inside the backup dir.
        var adapter_path = canonical_join(backup_path, String("lora.safetensors"))
        save_zimage_lora(set, adapter_path, ctx, STDtype.BF16)

        # 2) optimizer moments + TrainProgress (+ opt_step + EMA) — the resumable
        #    TrainState (save_train_state writes <backup_path>/train_state.safetensors).
        #    opt_step (= per-param AdamW state['step'] = completed optimizer steps)
        #    is persisted so resume restores the bias-correction step at any accum.
        save_train_state(backup_path, slots, ema, progress, opt_step, ctx)
        return backup_path

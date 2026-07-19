# 1:1 port of Serenity modules/util/TimedActionMixin.py
#
# Drives the repeating / single-shot cadence the trainer uses for sample / save /
# backup / gc / validate / update_step. repeating_action_needed and
# single_action_elapsed are ported branch-for-branch from the .py match/case.
#
# Serenity state:
#   self.__previous_action = {}              # name -> last monotonic time (or -1)
#   self.__start_time = time.monotonic()
# Mojo: a Dict[String, Float64] (the previous-action map) + a start_time scalar.
# time.monotonic() -> std.time.perf_counter() (both return monotonic seconds).
#
# TimeUnit is the integer-coded port (util/enum/TimeUnit.mojo). The EPOCH/STEP
# arms compare against train_progress; the SECOND/MINUTE/HOUR arms compare
# wall-clock deltas; NEVER -> False, ALWAYS -> True.

from std.collections.dict import Dict
from std.time import perf_counter

# Use the driver's TrainProgress (trainer.TrainState) so the cadence predicates
# accept the exact progress value the loop threads in. It is the same 1:1 port as
# util/TrainProgress.mojo.
from serenity_trainer.trainer.TrainState import TrainProgress
from serenity_trainer.util.enum.TimeUnit import (
    TU_EPOCH,
    TU_STEP,
    TU_SECOND,
    TU_MINUTE,
    TU_HOUR,
    TU_NEVER,
    TU_ALWAYS,
)


struct TimedActionMixin(Movable):
    var _previous_action: Dict[String, Float64]   # name -> last monotonic time / -1
    var _start_time: Float64                       # time.monotonic() at construction

    # TimedActionMixin.__init__ (TimedActionMixin.py:8-11).
    def __init__(out self):
        self._previous_action = Dict[String, Float64]()
        self._start_time = perf_counter()          # resist system clock changes

    # repeating_action_needed — TimedActionMixin.repeating_action_needed (:13-76).
    # `interval` is Float64 to cover both the int (save_every/step) and float
    # (backup_after/minutes) callers; the EPOCH/STEP arms int()-truncate it exactly
    # like the .py (int(interval)).
    def repeating_action_needed(
        mut self,
        name: String,
        interval: Float64,
        unit: Int,
        train_progress: TrainProgress,
        start_at_zero: Bool = True,
    ) raises -> Bool:
        # if name not in self.__previous_action: self.__previous_action[name] = -1
        if name not in self._previous_action:
            self._previous_action[name] = Float64(-1.0)

        if unit == TU_EPOCH:
            var iv = Int(interval)
            if iv == 0:
                return False
            if start_at_zero:
                return (
                    train_progress.epoch % iv == 0
                    and train_progress.epoch_step == 0
                )
            else:
                # last step of each epoch is unknown; Serenity fires at step 0 of
                # a matching epoch>0 (TimedActionMixin.py:31-33).
                return (
                    train_progress.epoch % iv == 0
                    and train_progress.epoch_step == 0
                    and train_progress.epoch > 0
                )
        elif unit == TU_STEP:
            var iv = Int(interval)
            if iv == 0:
                return False
            if start_at_zero:
                return train_progress.global_step % iv == 0
            else:
                return (train_progress.global_step + 1) % iv == 0
        elif unit == TU_SECOND:
            if not start_at_zero and self._previous_action[name] < 0:
                self._previous_action[name] = perf_counter()
            var since = perf_counter() - self._previous_action[name]
            if since > interval:
                self._previous_action[name] = perf_counter()
                return True
            else:
                return False
        elif unit == TU_MINUTE:
            if not start_at_zero and self._previous_action[name] < 0:
                self._previous_action[name] = perf_counter()
            var since = perf_counter() - self._previous_action[name]
            if since > (interval * 60.0):
                self._previous_action[name] = perf_counter()
                return True
            else:
                return False
        elif unit == TU_HOUR:
            if not start_at_zero and self._previous_action[name] < 0:
                self._previous_action[name] = perf_counter()
            var since = perf_counter() - self._previous_action[name]
            if since > (interval * 60.0 * 60.0):
                self._previous_action[name] = perf_counter()
                return True
            else:
                return False
        elif unit == TU_NEVER:
            return False
        elif unit == TU_ALWAYS:
            return True
        else:
            return False

    # single_action_elapsed — TimedActionMixin.single_action_elapsed (:78-107).
    def single_action_elapsed(
        mut self,
        name: String,
        delay: Float64,
        unit: Int,
        train_progress: TrainProgress,
    ) raises -> Bool:
        # if name not in self.__previous_action: self.__previous_action[name] = monotonic()
        if name not in self._previous_action:
            self._previous_action[name] = perf_counter()

        if unit == TU_EPOCH:
            return (train_progress.epoch + 1) > Int(delay)
        elif unit == TU_STEP:
            return (train_progress.global_step + 1) > Int(delay)
        elif unit == TU_SECOND:
            return (perf_counter() - self._start_time) > delay
        elif unit == TU_MINUTE:
            return (perf_counter() - self._start_time) > (delay * 60.0)
        elif unit == TU_HOUR:
            return (perf_counter() - self._start_time) > (delay * 60.0 * 60.0)
        elif unit == TU_NEVER:
            return False
        elif unit == TU_ALWAYS:
            return True
        else:
            return False

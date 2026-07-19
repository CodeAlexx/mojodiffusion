# lr_schedule.mojo — pure-Mojo port of Serenity's LR schedulers.
# Source (faithful): modules/util/lr_scheduler_util.py + create.create_lr_scheduler.
#
# Serenity uses LambdaLR: the scheduler returns a FACTOR in [0,1] that scales the
# base lr. We reproduce the factor here as a host-scalar function; the driver
# multiplies base_lr * factor(step). Pure host math (no GPU, no Tensor) — lr is a
# scalar schedule (allowed F64/F32 host scalar, not a stored tensor).
#
# Kinds (LearningRateScheduler enum): CONSTANT, LINEAR, COSINE,
# COSINE_WITH_RESTARTS, COSINE_WITH_HARD_RESTARTS, REX. (ADAFACTOR/CUSTOM are
# out of scope — Adafactor lands in Phase 5.)

from std.math import cos, pi

comptime LR_CONSTANT = 0
comptime LR_LINEAR = 1
comptime LR_COSINE = 2
comptime LR_COSINE_RESTARTS = 3
comptime LR_COSINE_HARD_RESTARTS = 4
comptime LR_REX = 5


# apply_min_factor(value, min_factor) = min_factor + (1-min_factor)*value
def _apply_min_factor(value: Float64, min_factor: Float64) -> Float64:
    return min_factor + (1.0 - min_factor) * value


@fieldwise_init
struct LrSchedule(Copyable, Movable):
    """Resolved LR schedule. `warmup_steps` and `scheduler_steps` are the
    post-resolution step counts (after create_lr_scheduler's accumulation/percent
    handling, done by the driver). `factor(step)` returns the LambdaLR factor."""

    var kind: Int
    var warmup_steps: Int
    var scheduler_steps: Int
    var num_cycles: Float64
    var min_factor: Float64

    # The bare (pre-warmup) schedule factor at a step already past warmup.
    def _base_factor(self, s: Int) -> Float64:
        var ss = Float64(self.scheduler_steps)
        if self.kind == LR_CONSTANT:
            return 1.0
        elif self.kind == LR_LINEAR:
            var lin = (ss - Float64(s)) / ss
            if lin < 0.0:
                lin = 0.0
            return _apply_min_factor(lin, self.min_factor)
        elif self.kind == LR_COSINE:
            var progress = Float64(s) / ss
            var cv = 0.5 * (1.0 + cos(progress * pi))
            if cv < 0.0:
                cv = 0.0
            return _apply_min_factor(cv, self.min_factor)
        elif self.kind == LR_COSINE_RESTARTS:
            var sc = s if s < (self.scheduler_steps - 1) else (self.scheduler_steps - 1)
            var progress = Float64(sc) / ss
            var cv = 0.5 * (1.0 + cos(progress * 2.0 * pi * self.num_cycles))
            if cv < 0.0:
                cv = 0.0
            return _apply_min_factor(cv, self.min_factor)
        elif self.kind == LR_COSINE_HARD_RESTARTS:
            var sc = s if s < (self.scheduler_steps - 1) else (self.scheduler_steps - 1)
            var progress = Float64(sc) / ss
            var pc = (progress * self.num_cycles)
            pc = pc - Float64(Int(pc))  # % 1.0  (pc >= 0)
            var cv = 0.5 * (1.0 + cos(pc * pi))
            if cv < 0.0:
                cv = 0.0
            return _apply_min_factor(cv, self.min_factor)
        else:  # LR_REX
            var val: Float64
            if s < self.scheduler_steps:
                var progress = Float64(s) / ss
                var d = 0.9
                var div = (1.0 - d) + (d * (1.0 - progress))
                val = (1.0 - progress) / div
            else:
                val = 0.0
            return _apply_min_factor(val, self.min_factor)

    # LambdaLR factor at `current_step` (0-based), including the linear warmup
    # ramp Serenity wraps every non-schedule-free schedule with.
    def factor(self, current_step: Int) -> Float64:
        if self.warmup_steps > 0:
            if current_step < self.warmup_steps:
                return Float64(current_step) / Float64(self.warmup_steps)
            return self._base_factor(current_step - self.warmup_steps)
        return self._base_factor(current_step)


# create_lr_scheduler's step resolution (the part that turns config warmup/epochs
# into concrete step counts). Mirrors create.py:1120-1130.
def resolve_schedule(
    kind: Int,
    warmup_steps_cfg: Float64,   # >1 = literal steps; (0,1] = fraction of total
    num_cycles: Float64,
    min_factor: Float64,
    steps_per_epoch: Int,
    num_epochs: Int,
    grad_accum_steps: Int,
) -> LrSchedule:
    var total_steps = Int(steps_per_epoch * num_epochs / grad_accum_steps)
    var warmup: Int
    if warmup_steps_cfg > 1.0:
        warmup = Int(warmup_steps_cfg / Float64(grad_accum_steps))
    elif warmup_steps_cfg > 0.0 and warmup_steps_cfg <= 1.0:
        warmup = Int(warmup_steps_cfg * Float64(total_steps))
    else:
        warmup = 0
    var scheduler_steps = total_steps - warmup
    return LrSchedule(kind, warmup, scheduler_steps, num_cycles, min_factor)

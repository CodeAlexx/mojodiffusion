# 1:1 port of Serenity modules/util/enum/LearningRateScheduler.py
# Source of truth: /home/alex/Serenity/modules/util/enum/LearningRateScheduler.py
#
# comptime-int constants matching the Python LearningRateScheduler members
# exactly (names + order). String value == member name.

comptime LR_SCHED_CONSTANT = 0                   # LearningRateScheduler.CONSTANT
comptime LR_SCHED_LINEAR = 1                     # LearningRateScheduler.LINEAR
comptime LR_SCHED_COSINE = 2                     # LearningRateScheduler.COSINE
comptime LR_SCHED_COSINE_WITH_RESTARTS = 3       # COSINE_WITH_RESTARTS
comptime LR_SCHED_COSINE_WITH_HARD_RESTARTS = 4  # COSINE_WITH_HARD_RESTARTS
comptime LR_SCHED_REX = 5                        # LearningRateScheduler.REX
comptime LR_SCHED_ADAFACTOR = 6                  # LearningRateScheduler.ADAFACTOR
comptime LR_SCHED_CUSTOM = 7                     # LearningRateScheduler.CUSTOM


def lr_scheduler_str(kind: Int) -> String:
    if kind == LR_SCHED_CONSTANT:
        return "CONSTANT"
    elif kind == LR_SCHED_LINEAR:
        return "LINEAR"
    elif kind == LR_SCHED_COSINE:
        return "COSINE"
    elif kind == LR_SCHED_COSINE_WITH_RESTARTS:
        return "COSINE_WITH_RESTARTS"
    elif kind == LR_SCHED_COSINE_WITH_HARD_RESTARTS:
        return "COSINE_WITH_HARD_RESTARTS"
    elif kind == LR_SCHED_REX:
        return "REX"
    elif kind == LR_SCHED_ADAFACTOR:
        return "ADAFACTOR"
    else:
        return "CUSTOM"

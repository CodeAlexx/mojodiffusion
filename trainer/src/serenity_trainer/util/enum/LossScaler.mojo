# 1:1 port of Serenity modules/util/enum/LossScaler.py
# Source of truth: /home/alex/Serenity/modules/util/enum/LossScaler.py
#
# comptime-int constants matching the Python LossScaler members exactly
# (names + order). String value == member name.

comptime LOSS_SCALER_NONE = 0                  # LossScaler.NONE
comptime LOSS_SCALER_BATCH = 1                 # LossScaler.BATCH
comptime LOSS_SCALER_GLOBAL_BATCH = 2          # LossScaler.GLOBAL_BATCH
comptime LOSS_SCALER_GRADIENT_ACCUMULATION = 3 # LossScaler.GRADIENT_ACCUMULATION
comptime LOSS_SCALER_BOTH = 4                  # LossScaler.BOTH
comptime LOSS_SCALER_GLOBAL_BOTH = 5           # LossScaler.GLOBAL_BOTH


# get_scale  (LossScaler.py:17-32). world_size == 1 here (multi-GPU out of
# scope), so GLOBAL_* reduce to their local counterparts.
#   NONE -> 1
#   BATCH -> batch_size
#   GLOBAL_BATCH -> batch_size * world_size
#   GRADIENT_ACCUMULATION -> accumulation_steps
#   BOTH -> accumulation_steps * batch_size
#   GLOBAL_BOTH -> accumulation_steps * batch_size * world_size
def loss_scaler_get_scale(
    kind: Int, batch_size: Int, accumulation_steps: Int
) -> Int:
    var world_size = 1  # multi.world_size() — single-GPU
    if kind == LOSS_SCALER_NONE:
        return 1
    elif kind == LOSS_SCALER_BATCH:
        return batch_size
    elif kind == LOSS_SCALER_GLOBAL_BATCH:
        return batch_size * world_size
    elif kind == LOSS_SCALER_GRADIENT_ACCUMULATION:
        return accumulation_steps
    elif kind == LOSS_SCALER_BOTH:
        return accumulation_steps * batch_size
    else:  # LOSS_SCALER_GLOBAL_BOTH
        return accumulation_steps * batch_size * world_size


def loss_scaler_str(kind: Int) -> String:
    if kind == LOSS_SCALER_NONE:
        return "NONE"
    elif kind == LOSS_SCALER_BATCH:
        return "BATCH"
    elif kind == LOSS_SCALER_GLOBAL_BATCH:
        return "GLOBAL_BATCH"
    elif kind == LOSS_SCALER_GRADIENT_ACCUMULATION:
        return "GRADIENT_ACCUMULATION"
    elif kind == LOSS_SCALER_BOTH:
        return "BOTH"
    else:
        return "GLOBAL_BOTH"

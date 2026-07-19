# 1:1 port of Serenity modules/util/enum/TimeUnit.py
#
# Serenity's TimeUnit is a string Enum (TimeUnit.py):
#   EPOCH, STEP, SECOND, MINUTE, HOUR, NEVER, ALWAYS
# Mojo has no string enums; we assign stable integer codes in declaration order
# (matching the .py source order) so the cadence (TimedActionMixin port) can
# branch on them. The codes are internal — only equality/branch logic depends on
# them, never serialized.
#
# is_time_unit() mirrors TimeUnit.is_time_unit (SECOND/MINUTE/HOUR are wall-clock
# units; EPOCH/STEP are progress units; NEVER/ALWAYS are sentinels).

comptime TU_EPOCH  = 0
comptime TU_STEP   = 1
comptime TU_SECOND = 2
comptime TU_MINUTE = 3
comptime TU_HOUR   = 4
comptime TU_NEVER  = 5
comptime TU_ALWAYS = 6


# is_time_unit — True for the wall-clock units (SECOND/MINUTE/HOUR).
# Mirrors TimeUnit.is_time_unit (TimeUnit.py).
def is_time_unit(unit: Int) -> Bool:
    return unit == TU_SECOND or unit == TU_MINUTE or unit == TU_HOUR

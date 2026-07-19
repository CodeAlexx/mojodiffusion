# 1:1 port of Serenity modules/util/enum/EMAMode.py
# Source of truth: /home/alex/Serenity/modules/util/enum/EMAMode.py
#
# Mojo has no string-valued Enum; the members are comptime-int constants in the
# exact order/name of the Python EMAMode members. The string value equals the
# member name (Python `__str__` returns self.value == name).

comptime EMA_OFF = 0   # EMAMode.OFF
comptime EMA_GPU = 1   # EMAMode.GPU
comptime EMA_CPU = 2   # EMAMode.CPU


def ema_mode_str(kind: Int) -> String:
    # __str__ returns the value string (== member name)
    if kind == EMA_OFF:
        return "OFF"
    elif kind == EMA_GPU:
        return "GPU"
    else:
        return "CPU"

# 1:1 port of Serenity modules/util/enum/Optimizer.py
# Source of truth: /home/alex/Serenity/modules/util/enum/Optimizer.py
#
# comptime-int constants matching the Python Optimizer members exactly
# (names + order). String value == member name.
# `maybe_adjust_lrs` (Optimizer.py:119-127) needs a live torch optimizer object
# and is out of scope for the pure-Mojo loss slice — omitted here.

comptime OPT_ADAGRAD = 0       # Optimizer.ADAGRAD
comptime OPT_ADAGRAD_8BIT = 1  # Optimizer.ADAGRAD_8BIT
comptime OPT_ADAM = 2          # Optimizer.ADAM
comptime OPT_ADAM_8BIT = 3     # Optimizer.ADAM_8BIT
comptime OPT_ADAMW = 4         # Optimizer.ADAMW
comptime OPT_ADAMW_8BIT = 5    # Optimizer.ADAMW_8BIT
comptime OPT_ADAMW_ADV = 6     # Optimizer.ADAMW_ADV
comptime OPT_AdEMAMix = 7      # Optimizer.AdEMAMix
comptime OPT_AdEMAMix_8BIT = 8 # Optimizer.AdEMAMix_8BIT
comptime OPT_ADOPT = 9         # Optimizer.ADOPT
comptime OPT_ADOPT_ADV = 10    # Optimizer.ADOPT_ADV
comptime OPT_LAMB = 11         # Optimizer.LAMB
comptime OPT_LAMB_8BIT = 12    # Optimizer.LAMB_8BIT
comptime OPT_LARS = 13         # Optimizer.LARS
comptime OPT_LARS_8BIT = 14    # Optimizer.LARS_8BIT
comptime OPT_LION = 15         # Optimizer.LION
comptime OPT_LION_8BIT = 16    # Optimizer.LION_8BIT
comptime OPT_LION_ADV = 17     # Optimizer.LION_ADV
comptime OPT_RMSPROP = 18      # Optimizer.RMSPROP
comptime OPT_RMSPROP_8BIT = 19 # Optimizer.RMSPROP_8BIT
comptime OPT_SGD = 20          # Optimizer.SGD
comptime OPT_SGD_8BIT = 21     # Optimizer.SGD_8BIT
comptime OPT_SIGNSGD_ADV = 22  # Optimizer.SIGNSGD_ADV
comptime OPT_SCHEDULE_FREE_ADAMW = 23  # Optimizer.SCHEDULE_FREE_ADAMW
comptime OPT_SCHEDULE_FREE_SGD = 24    # Optimizer.SCHEDULE_FREE_SGD
comptime OPT_DADAPT_ADA_GRAD = 25      # Optimizer.DADAPT_ADA_GRAD
comptime OPT_DADAPT_ADAM = 26          # Optimizer.DADAPT_ADAM
comptime OPT_DADAPT_ADAN = 27          # Optimizer.DADAPT_ADAN
comptime OPT_DADAPT_LION = 28          # Optimizer.DADAPT_LION
comptime OPT_DADAPT_SGD = 29           # Optimizer.DADAPT_SGD
comptime OPT_PRODIGY = 30              # Optimizer.PRODIGY
comptime OPT_PRODIGY_PLUS_SCHEDULE_FREE = 31  # Optimizer.PRODIGY_PLUS_SCHEDULE_FREE
comptime OPT_PRODIGY_ADV = 32          # Optimizer.PRODIGY_ADV
comptime OPT_ADAFACTOR = 33            # Optimizer.ADAFACTOR
comptime OPT_CAME = 34                 # Optimizer.CAME
comptime OPT_CAME_8BIT = 35            # Optimizer.CAME_8BIT
comptime OPT_MUON = 36                 # Optimizer.MUON
comptime OPT_MUON_ADV = 37             # Optimizer.MUON_ADV
comptime OPT_ADAMUON_ADV = 38          # Optimizer.ADAMUON_ADV
comptime OPT_ADABELIEF = 39            # Optimizer.ADABELIEF
comptime OPT_TIGER = 40                # Optimizer.TIGER
comptime OPT_AIDA = 41                 # Optimizer.AIDA
comptime OPT_YOGI = 42                 # Optimizer.YOGI


# is_adaptive  (Optimizer.py:80-91)
def optimizer_is_adaptive(kind: Int) -> Bool:
    return (
        kind == OPT_DADAPT_SGD
        or kind == OPT_DADAPT_ADAM
        or kind == OPT_DADAPT_ADAN
        or kind == OPT_DADAPT_ADA_GRAD
        or kind == OPT_DADAPT_LION
        or kind == OPT_PRODIGY
        or kind == OPT_PRODIGY_PLUS_SCHEDULE_FREE
        or kind == OPT_PRODIGY_ADV
    )


# is_schedule_free  (Optimizer.py:93-99)
def optimizer_is_schedule_free(kind: Int) -> Bool:
    return (
        kind == OPT_SCHEDULE_FREE_ADAMW
        or kind == OPT_SCHEDULE_FREE_SGD
        or kind == OPT_PRODIGY_PLUS_SCHEDULE_FREE
    )


# supports_fused_back_pass  (Optimizer.py:101-116)
def optimizer_supports_fused_back_pass(kind: Int) -> Bool:
    return (
        kind == OPT_ADAFACTOR
        or kind == OPT_CAME
        or kind == OPT_CAME_8BIT
        or kind == OPT_ADAM
        or kind == OPT_ADAMW
        or kind == OPT_ADAMW_ADV
        or kind == OPT_ADOPT_ADV
        or kind == OPT_PRODIGY_PLUS_SCHEDULE_FREE
        or kind == OPT_PRODIGY_ADV
        or kind == OPT_LION_ADV
        or kind == OPT_MUON_ADV
        or kind == OPT_ADAMUON_ADV
        or kind == OPT_SIGNSGD_ADV
    )


def optimizer_str(kind: Int) -> String:
    if kind == OPT_ADAGRAD:
        return "ADAGRAD"
    elif kind == OPT_ADAGRAD_8BIT:
        return "ADAGRAD_8BIT"
    elif kind == OPT_ADAM:
        return "ADAM"
    elif kind == OPT_ADAM_8BIT:
        return "ADAM_8BIT"
    elif kind == OPT_ADAMW:
        return "ADAMW"
    elif kind == OPT_ADAMW_8BIT:
        return "ADAMW_8BIT"
    elif kind == OPT_ADAMW_ADV:
        return "ADAMW_ADV"
    elif kind == OPT_AdEMAMix:
        return "AdEMAMix"
    elif kind == OPT_AdEMAMix_8BIT:
        return "AdEMAMix_8BIT"
    elif kind == OPT_ADOPT:
        return "ADOPT"
    elif kind == OPT_ADOPT_ADV:
        return "ADOPT_ADV"
    elif kind == OPT_LAMB:
        return "LAMB"
    elif kind == OPT_LAMB_8BIT:
        return "LAMB_8BIT"
    elif kind == OPT_LARS:
        return "LARS"
    elif kind == OPT_LARS_8BIT:
        return "LARS_8BIT"
    elif kind == OPT_LION:
        return "LION"
    elif kind == OPT_LION_8BIT:
        return "LION_8BIT"
    elif kind == OPT_LION_ADV:
        return "LION_ADV"
    elif kind == OPT_RMSPROP:
        return "RMSPROP"
    elif kind == OPT_RMSPROP_8BIT:
        return "RMSPROP_8BIT"
    elif kind == OPT_SGD:
        return "SGD"
    elif kind == OPT_SGD_8BIT:
        return "SGD_8BIT"
    elif kind == OPT_SIGNSGD_ADV:
        return "SIGNSGD_ADV"
    elif kind == OPT_SCHEDULE_FREE_ADAMW:
        return "SCHEDULE_FREE_ADAMW"
    elif kind == OPT_SCHEDULE_FREE_SGD:
        return "SCHEDULE_FREE_SGD"
    elif kind == OPT_DADAPT_ADA_GRAD:
        return "DADAPT_ADA_GRAD"
    elif kind == OPT_DADAPT_ADAM:
        return "DADAPT_ADAM"
    elif kind == OPT_DADAPT_ADAN:
        return "DADAPT_ADAN"
    elif kind == OPT_DADAPT_LION:
        return "DADAPT_LION"
    elif kind == OPT_DADAPT_SGD:
        return "DADAPT_SGD"
    elif kind == OPT_PRODIGY:
        return "PRODIGY"
    elif kind == OPT_PRODIGY_PLUS_SCHEDULE_FREE:
        return "PRODIGY_PLUS_SCHEDULE_FREE"
    elif kind == OPT_PRODIGY_ADV:
        return "PRODIGY_ADV"
    elif kind == OPT_ADAFACTOR:
        return "ADAFACTOR"
    elif kind == OPT_CAME:
        return "CAME"
    elif kind == OPT_CAME_8BIT:
        return "CAME_8BIT"
    elif kind == OPT_MUON:
        return "MUON"
    elif kind == OPT_MUON_ADV:
        return "MUON_ADV"
    elif kind == OPT_ADAMUON_ADV:
        return "ADAMUON_ADV"
    elif kind == OPT_ADABELIEF:
        return "ADABELIEF"
    elif kind == OPT_TIGER:
        return "TIGER"
    elif kind == OPT_AIDA:
        return "AIDA"
    else:
        return "YOGI"

# 1:1 port of Serenity modules/util/enum/LossWeight.py
# Source of truth: /home/alex/Serenity/modules/util/enum/LossWeight.py
#
# comptime-int constants matching the Python LossWeight members exactly
# (names + order). String value == member name.
#
# NOTE: ModelSetupDiffusionLossMixin.mojo already defines its own LW_* constants
# for the weighting kernels and uses a DIFFERENT ordering (LW_CONSTANT=0,
# LW_MIN_SNR_GAMMA=1, LW_DEBIASED_ESTIMATION=2, LW_P2=3, LW_SIGMA=4). This file
# is the faithful enum-member order matching the Python source. Callers mapping
# config -> mixin kernel must translate between the two; the names are 1:1.

comptime LOSS_WEIGHT_CONSTANT = 0             # LossWeight.CONSTANT
comptime LOSS_WEIGHT_P2 = 1                   # LossWeight.P2
comptime LOSS_WEIGHT_MIN_SNR_GAMMA = 2        # LossWeight.MIN_SNR_GAMMA
comptime LOSS_WEIGHT_DEBIASED_ESTIMATION = 3  # LossWeight.DEBIASED_ESTIMATION
comptime LOSS_WEIGHT_SIGMA = 4                # LossWeight.SIGMA


# supports_flow_matching  (LossWeight.py:11-13): CONSTANT or SIGMA
def loss_weight_supports_flow_matching(kind: Int) -> Bool:
    return kind == LOSS_WEIGHT_CONSTANT or kind == LOSS_WEIGHT_SIGMA


def loss_weight_str(kind: Int) -> String:
    if kind == LOSS_WEIGHT_CONSTANT:
        return "CONSTANT"
    elif kind == LOSS_WEIGHT_P2:
        return "P2"
    elif kind == LOSS_WEIGHT_MIN_SNR_GAMMA:
        return "MIN_SNR_GAMMA"
    elif kind == LOSS_WEIGHT_DEBIASED_ESTIMATION:
        return "DEBIASED_ESTIMATION"
    else:
        return "SIGMA"

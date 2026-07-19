# 1:1 port of Serenity modules/util/enum/TrainingMethod.py
# Source of truth: /home/alex/Serenity/modules/util/enum/TrainingMethod.py
#
# comptime-int constants matching the Python TrainingMethod members exactly
# (names + order). String value == member name.

comptime TM_FINE_TUNE = 0      # TrainingMethod.FINE_TUNE
comptime TM_LORA = 1           # TrainingMethod.LORA
comptime TM_EMBEDDING = 2      # TrainingMethod.EMBEDDING
comptime TM_FINE_TUNE_VAE = 3  # TrainingMethod.FINE_TUNE_VAE


def training_method_str(kind: Int) -> String:
    if kind == TM_FINE_TUNE:
        return "FINE_TUNE"
    elif kind == TM_LORA:
        return "LORA"
    elif kind == TM_EMBEDDING:
        return "EMBEDDING"
    else:
        return "FINE_TUNE_VAE"

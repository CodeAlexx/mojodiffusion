from serenity_trainer.util.optimizer.adam_extensions import adam_step
from serenity_trainer.util.optimizer.CAME import came_step
from serenity_trainer.util.optimizer.adafactor_extensions import adafactor_step
from serenity_trainer.util.bf16_stochastic_rounding import copy_stochastic_value
def main() raises:
    print("optim slice compiles: adam_step/came_step/adafactor_step/copy_stochastic_value resolved")

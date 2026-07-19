# Verify read_train_config parses the alina preset to Serenity's values.
from serenity_trainer.util.config.TrainConfigReader import read_train_config

def main() raises:
    var cfg = read_train_config(String("/home/alex/Serenity/configs/alina_zimage_serenity_preset_100_baseline.json"))
    print("learning_rate =", cfg.learning_rate, " (reference trainer 0.0003)")
    print("batch_size =", cfg.batch_size, " (reference trainer 2)")
    print("epochs =", cfg.epochs, " (reference trainer 100)")
    print("seed =", cfg.seed, " (reference trainer 42)")
    print("timestep_distribution =", cfg.timestep_distribution, " (reference trainer LOGIT_NORMAL=2)")
    var ok = (cfg.learning_rate == Float32(0.0003)) and (cfg.batch_size == 2) and (cfg.epochs == 100) and (cfg.seed == UInt32(42)) and (cfg.timestep_distribution == 2)
    print("PRESET READER PARITY", "PASS" if ok else "FAIL")

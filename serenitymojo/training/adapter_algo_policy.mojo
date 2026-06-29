# training/adapter_algo_policy.mojo -- shared LyCORIS adapter selection guards.
#
# The parser accepts ai-toolkit-style network_algorithm strings. A trainer must
# still opt into the math path it can actually execute, so unsupported LyCORIS
# variants fail before checkpoint load instead of silently training plain LoRA.

from serenitymojo.training.train_config import (
    TrainConfig,
    TRAIN_ADAPTER_ALGO_LORA,
    TRAIN_ADAPTER_ALGO_FULL,
    TRAIN_ADAPTER_ALGO_LOHA,
    TRAIN_ADAPTER_ALGO_DORA,
    TRAIN_ADAPTER_ALGO_LOKR,
    TRAIN_ADAPTER_ALGO_OFT,
    TRAIN_ADAPTER_ALGO_BOFT,
    TRAIN_ADAPTER_ALGO_LOCON,
)


def adapter_algo_name(algo: Int) -> String:
    if algo == TRAIN_ADAPTER_ALGO_LORA:
        return String("lora")
    elif algo == TRAIN_ADAPTER_ALGO_FULL:
        return String("full")
    elif algo == TRAIN_ADAPTER_ALGO_LOHA:
        return String("loha")
    elif algo == TRAIN_ADAPTER_ALGO_DORA:
        return String("dora")
    elif algo == TRAIN_ADAPTER_ALGO_LOKR:
        return String("lokr")
    elif algo == TRAIN_ADAPTER_ALGO_OFT:
        return String("oft")
    elif algo == TRAIN_ADAPTER_ALGO_BOFT:
        return String("boft")
    elif algo == TRAIN_ADAPTER_ALGO_LOCON:
        return String("locon")
    return String("unknown(") + String(algo) + String(")")


def require_lora_or_locon_linear(cfg: TrainConfig, trainer_name: String) raises:
    """Accept plain LoRA plus LoCon's linear-compatible down/up path.

    LoCon on linear projections is the same low-rank additive delta currently
    used by the model LoRA stacks; conv LoCon is only valid where a trainer wires
    the conv primitive. LoKr/LoHa/DoRA/OFT need model-stack carrier wiring, so
    this helper rejects them before any large model allocation.
    """
    if cfg.adapter_algo == TRAIN_ADAPTER_ALGO_LORA:
        return
    if cfg.adapter_algo == TRAIN_ADAPTER_ALGO_LOCON:
        print(
            String("[") + trainer_name + String("-locon] network_algorithm=locon: ")
            + String("using the linear LoRA-compatible down/up path; conv targets are not wired here")
        )
        return
    raise Error(
        trainer_name + String(" trainer: network_algorithm=")
        + adapter_algo_name(cfg.adapter_algo)
        + String(" is not wired end-to-end for this model. Supported here: lora, locon. ")
        + String("LoKr/LoHa/DoRA/OFT are allowed only where the trainer has a ")
        + String("model-specific carrier path and 24 GB preflight.")
    )


def require_lora_only_ltx2(cfg: TrainConfig, trainer_name: String) raises:
    if cfg.adapter_algo != TRAIN_ADAPTER_ALGO_LORA:
        raise Error(
            trainer_name
            + String(" trainer: LyCORIS adapters are disabled for LTX2; use network_algorithm=lora")
        )

# readiness.mojo -- executable readiness status for the LTX-2 AV trainer.


@fieldwise_init
struct LTX2Readiness(Copyable, Movable):
    var config_cli_contract: Bool
    var cache_record_contract: Bool
    var conditioning_contract: Bool
    var schedule_contract: Bool
    var lora_surface_contract: Bool
    var checkpoint_contract: Bool
    var validation_contract: Bool
    var acceptance_runner: Bool
    var av_backward_ready: Bool
    var av_lora_runtime_ready: Bool

    def foundation_ready(self) -> Bool:
        return (
            self.config_cli_contract
            and self.cache_record_contract
            and self.conditioning_contract
            and self.schedule_contract
            and self.lora_surface_contract
            and self.checkpoint_contract
            and self.validation_contract
            and self.acceptance_runner
        )

    def production_training_ready(self) -> Bool:
        return self.foundation_ready() and self.av_backward_ready and self.av_lora_runtime_ready


def default_readiness() -> LTX2Readiness:
    return LTX2Readiness(
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
    )


def print_readiness(r: LTX2Readiness):
    print("LTX2 AV trainer readiness")
    print("  foundation_ready:", r.foundation_ready())
    print("  production_training_ready:", r.production_training_ready())
    print("  config/cli:", r.config_cli_contract)
    print("  cache records:", r.cache_record_contract)
    print("  conditioning:", r.conditioning_contract)
    print("  schedules:", r.schedule_contract)
    print("  lora surface:", r.lora_surface_contract)
    print("  checkpoint contract:", r.checkpoint_contract)
    print("  validation contract:", r.validation_contract)
    print("  AV backward ready:", r.av_backward_ready)
    print("  train-time AV LoRA runtime ready:", r.av_lora_runtime_ready)

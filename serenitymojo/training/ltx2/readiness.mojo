# readiness.mojo -- executable readiness status for the LTX-2 AV trainer.
#
# SUPERSEDED-IN-PLACE (P6.0, 2026-07-18): July-10 Codex-era contract SKETCH. The
# REAL readiness gate is the trainer's own startup fail-louds (cache pairing,
# preset guards, dtype/geometry asserts) in train_ltx2_av.mojo. KEPT + STILL
# LIVE: train_ltx2_av calls run_acceptance()+print_readiness() at startup and
# acceptance.mojo populates LTX2Readiness — this struct is the readiness-REPORT
# surface (extended, not replaced, at P6.0). Do NOT treat its boolean fields as
# the source of truth for whether training is safe to start.


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
    # av_backward_ready / av_lora_runtime_ready flipped False -> True 2026-07-19
    # (P6.2). Both were accurate while the AV arm was scaffolding; both are now
    # wired and GATED, so leaving them False made this contract lie:
    #   av_backward   — models/ltx2/ltx2_av_stack.mojo ltx2_av_stack_backward,
    #                   composition gated 114 PASS / 0 FAIL vs torch autograd at
    #                   BOTH conductor arms (LTX2_SAVE_ACTS=0 recompute and =2
    #                   retained), digits identical across arms
    #                   (parity/ltx2_av_stack_bwd_parity.mojo)
    #   av_lora_runtime — training/train_ltx2_av.mojo _run_geometry_av: 4-step
    #                   runs RC=0, both modality losses moving, 672-key musubi
    #                   audio save (1344 tensors, 14 modules/block x48), and the
    #                   final-step inactive-slot assert max|A|,|B| == 0
    return LTX2Readiness(
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
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

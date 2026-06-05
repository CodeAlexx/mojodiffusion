# train_ltx2_av.mojo -- thin LTX-2.3 AV trainer foundation entrypoint.

from std.sys import argv
from std.collections import List
from serenitymojo.training.ltx2.acceptance import run_acceptance
from serenitymojo.training.ltx2.config import LTX2TrainerConfig, print_config_summary
from serenitymojo.training.ltx2.checkpointing import resume_checkpoint_contract, save_checkpoint_contract
from serenitymojo.training.ltx2.lora_surface import lora_surface_summary
from serenitymojo.training.ltx2.readiness import print_readiness
from serenitymojo.training.ltx2.validation import default_validation_contract


def _has_arg(args: List[String], name: String) -> Bool:
    for i in range(len(args)):
        if String(args[i]) == name:
            return True
    return False


def main() raises:
    var raw_args = argv()
    var args = List[String]()
    for i in range(len(raw_args)):
        args.append(String(raw_args[i]))
    var cfg = LTX2TrainerConfig.from_args(args)
    var readiness = run_acceptance(True)

    if _has_arg(args, "--acceptance"):
        print_readiness(readiness)
        return

    print_config_summary(cfg)
    print("  surface:", lora_surface_summary(cfg.lora_target_preset))
    print("  resume:", resume_checkpoint_contract(cfg))
    print("  save contract:", save_checkpoint_contract(cfg, cfg.save_every))
    var vc = default_validation_contract(cfg)
    print("  validation enabled:", vc.enabled(), " sample_every:", vc.sample_every)
    print_readiness(readiness)
    if cfg.fail_on_unready and not readiness.production_training_ready():
        raise Error("LTX2 AV trainer foundation is not production-training-ready: full AV backward is still blocked")
    print("DONE LTX2 AV trainer foundation check")

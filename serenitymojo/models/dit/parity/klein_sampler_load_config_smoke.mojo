# klein_sampler_load_config_smoke.mojo — prove the config-driven sampler loader
# sizes itself to the 4B checkpoint (5 double + 20 single) from the config file,
# fixing the "double_blocks.5.img_attn.qkv.weight not found" bug (which the old
# hardcoded klein9b_all_keys / KleinConfig.klein_9b path triggered on 4B).
#
# Run (after the compile lock frees):
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#     pixi run mojo run -I . serenitymojo/models/dit/parity/klein_sampler_load_config_smoke.mojo

from std.gpu.host import DeviceContext
from serenitymojo.models.dit.klein_dit import Klein9BDiT, KleinConfig
from serenitymojo.io.train_config_reader import read_model_config


comptime CONFIG = "/home/alex/mojodiffusion/serenitymojo/configs/klein4b.json"


def main() raises:
    var ctx = DeviceContext()
    var tc = read_model_config(String(CONFIG))
    print("config:", tc.name, " double=", tc.num_double, " single=", tc.num_single,
          " ckpt=", tc.checkpoint)

    var kcfg = KleinConfig.from_train_config(tc)
    if kcfg.num_double != 5 or kcfg.num_single != 20:
        raise Error("KleinConfig.from_train_config block counts wrong: "
            + String(kcfg.num_double) + "+" + String(kcfg.num_single))

    # The old path raised "double_blocks.5 not found" here on 4B. This must load
    # all 5+20 blocks cleanly.
    var model = Klein9BDiT.load_with_config(tc.checkpoint, kcfg, ctx)

    # Spot-check that the highest 4B block keys are present (would be absent if the
    # loader had stopped at the 9B counts or mis-sized).
    var probe = List[String]()
    probe.append(String("double_blocks.4.img_attn.qkv.weight"))
    probe.append(String("single_blocks.19.linear1.weight"))
    probe.append(String("final_layer.linear.weight"))
    for i in range(len(probe)):
        if probe[i] not in model.name_to_idx:
            raise Error("expected key missing after load: " + probe[i])

    # And that 9B-only blocks are NOT requested (no double_blocks.5+).
    if String("double_blocks.5.img_attn.qkv.weight") in model.name_to_idx:
        raise Error("loader pulled a 9B-only block on a 4B checkpoint")

    print("loaded weights:", len(model.weights),
          " stored config double/single:", model.config.num_double, model.config.num_single)
    print("klein_sampler_load_config_smoke PASS (4B sampler load is config-driven)")

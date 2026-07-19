# SenseNova-U1 T2I load probe.
#
# Loads only the T2I-required shared resident weights through SenseNovaU1.load
# and exits before prefix forward / denoise. This is a cheap GPU-memory sanity
# check after trimming unused lm_head + understanding-side vision weights.

from std.gpu.host import DeviceContext

from serenitymojo.models.dit.sensenova_u1 import SenseNovaU1


comptime WEIGHTS_DIR = "/home/alex/.serenity/models/sensenova_u1"
comptime L_TOKENS = 4
comptime TEXT_LEN = 18


def main() raises:
    var ctx = DeviceContext()
    print("[sensenova_u1_load] loading T2I shared weights from", WEIGHTS_DIR)
    var model = SenseNovaU1[L_TOKENS, TEXT_LEN].load(WEIGHTS_DIR, ctx)
    print("[sensenova_u1_load] resident shared tensors=", len(model.shared))
    ctx.synchronize()
    print("[sensenova_u1_load] complete")

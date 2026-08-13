# ltx2_has_gate_probe.mojo — L5 prerequisite step 1: does the LTX-2.3 VIDEO block
# actually carry the per-head attention gate? has_gate = weights._has(
# "<attn>.to_gate_logits.weight") on block 0 of the fp8 checkpoint. If BOTH attn1
# and attn2 are False, the host round-trip (ltx2_av_backward.mojo:357) never fires
# for the video model -> the capture blocker is MOOT and the kernel port is
# unnecessary. Measure before building (autograd-v2 Tenet 4).
#
#   pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
#     serenitymojo/autograd_v2/tests/ltx2_has_gate_probe.mojo -o /tmp/ltx2_has_gate_probe
#   env LD_LIBRARY_PATH=.pixi/envs/default/lib /tmp/ltx2_has_gate_probe

from max.gpu.host import DeviceContext
from serenitymojo.models.dit.ltx2_dit import LTX2Config
from serenitymojo.models.ltx2.ltx2_video_stack import LTX2VideoBlockSource

comptime CKPT = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-dev-fp8.safetensors"


def main() raises:
    print("=== LTX2 VIDEO block has_gate probe (block 0, fp8 ckpt) ===")
    var ctx = DeviceContext()
    var cfg = LTX2Config.ltx2()
    var src = LTX2VideoBlockSource.open(CKPT, cfg, False)
    var w = src.get_block(0, ctx)
    var g1 = w._has("attn1.to_gate_logits.weight")
    var g2 = w._has("attn2.to_gate_logits.weight")
    print("  attn1.to_gate_logits.weight present:", g1)
    print("  attn2.to_gate_logits.weight present:", g2)
    if g1 or g2:
        print("VERDICT: has_gate FIRES -> the host round-trip is a REAL capture blocker;",
              "the device gate-grad kernel is needed.")
    else:
        print("VERDICT: has_gate does NOT fire on the video block -> the round-trip is",
              "MOOT; NO kernel port needed. (Report this LOUDLY.)")

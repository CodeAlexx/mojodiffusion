# qwen3vl_vision_parity.mojo — per-stage parity gate for the pure-Mojo
# Qwen3-VL vision tower vs the torch oracle (qwen3vl_vision_oracle.py).
#
# Loads model.visual.* from the LingBot text_encoder, feeds the oracle's fixed
# pixel_values (test image -> grid (1,56,32) -> seq=1792, Nmerged=448), runs the
# Mojo forward, and compares each stage against the oracle safetensors with
# cos>=0.999 (ParityHarness).
#
# Run: cd /home/alex/mojodiffusion && pixi run mojo run -I . \
#        serenitymojo/models/text_encoder/parity/qwen3vl_vision_parity.mojo

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.models.text_encoder.lingbot_qwen3vl_vision import (
    Qwen3VLVisionModel, VisionOutput,
)

comptime TE_DIR = "/mnt/disk1/models/lingbot-video-moe/text_encoder"
comptime ORACLE = (
    "/home/alex/mojodiffusion/serenitymojo/models/text_encoder/parity/"
    "qwen3vl_vision_oracle_f32.safetensors"
)
# Fixed test image grid (from qwen3vl_vision_oracle.py print): (t,h,w)=(1,56,32).
comptime T = 1
comptime H = 56
comptime W = 32
comptime S = 1792   # T*H*W


def _ref(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> List[Float32]:
    var tv = st.tensor_view(name)
    var t = Tensor.from_view(tv, ctx)
    return t.to_host(ctx)


def _gate(
    harness: ParityHarness, tag: String, actual: Tensor,
    st: ShardedSafeTensors, ref_name: String, ctx: DeviceContext,
) raises -> Bool:
    var reference = _ref(st, ref_name, ctx)
    var r = harness.compare(actual, reference, ctx)
    var ok = r.cos >= 0.999
    var status = "PASS" if ok else "FAIL"
    print("  [", status, "] ", tag, "  cos=", r.cos, "  max_abs=", r.max_abs)
    return ok


def main() raises:
    var ctx = DeviceContext()
    var st = ShardedSafeTensors.open(String(ORACLE))
    var harness = ParityHarness(0.999)

    print("[parity] loading Qwen3-VL vision tower ...")
    var model = Qwen3VLVisionModel.load(String(TE_DIR), ctx)

    # oracle pixel_values [seq,1536] f32; keep F32 activation stream (weights
    # stay BF16 inside linear via the mixed-dtype gemm path).
    var pixel_values = Tensor.from_view(st.tensor_view(String("pixel_values")), ctx)

    print("[parity] forward S=", S, " (t,h,w)=(", T, ",", H, ",", W, ")")
    var out = model.forward[S](pixel_values, T, H, W, ctx)

    print("[parity] per-stage gates (threshold cos>=0.999):")
    var all_ok = True
    all_ok = _gate(harness, "S1 hidden+posembed ", out.s1_hidden, st, String("s1_hidden_posembed"), ctx) and all_ok
    all_ok = _gate(harness, "S2 cos             ", out.cos64, st, String("s2_cos"), ctx) and all_ok
    all_ok = _gate(harness, "S2 sin             ", out.sin64, st, String("s2_sin"), ctx) and all_ok
    all_ok = _gate(harness, "S3 block0 out      ", out.block0, st, String("s3_block0_out"), ctx) and all_ok
    all_ok = _gate(harness, "S4 pooler_output   ", out.pooler, st, String("s4_pooler_output"), ctx) and all_ok
    all_ok = _gate(harness, "S4 deepstack[0]    ", out.ds0, st, String("s4_deepstack_0"), ctx) and all_ok
    all_ok = _gate(harness, "S4 deepstack[1]    ", out.ds1, st, String("s4_deepstack_1"), ctx) and all_ok
    all_ok = _gate(harness, "S4 deepstack[2]    ", out.ds2, st, String("s4_deepstack_2"), ctx) and all_ok

    if all_ok:
        print("[parity] ALL STAGES PASS")
    else:
        print("[parity] SOME STAGES FAILED")



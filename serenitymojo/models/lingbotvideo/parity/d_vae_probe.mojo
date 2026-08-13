# Chunk D gate: LingBotWanVaeDecoder (Wan2.1-style 3D causal VAE), image mode.
#
# Loads decoder.* + post_quant_conv.* (fp32) and the captured z/out from
# oracle_d.safetensors, runs decode, and compares to `out` with ParityHarness
# (gate cos >= 0.999) plus magnitude ratio |mine|/|ref| and max_abs.
#
# Run (JIT):
#   cd /home/alex/mojodiffusion && \
#     pixi run mojo run -I . serenitymojo/models/lingbotvideo/parity/d_vae_probe.mojo

from max.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.parity import ParityHarness
from serenitymojo.models.lingbotvideo.vae_decoder import LingBotWanVaeDecoder

comptime PARITY_DIR = "/home/alex/mojodiffusion/serenitymojo/models/lingbotvideo/parity"
comptime LH = 2
comptime LW = 2


def _sh(s: List[Int]) -> String:
    var out = String("[")
    for i in range(len(s)):
        if i > 0:
            out += ", "
        out += String(s[i])
    out += "]"
    return out


def _l2(v: List[Float32]) -> Float64:
    var acc: Float64 = 0.0
    for i in range(len(v)):
        acc += Float64(v[i]) * Float64(v[i])
    return sqrt(acc)


def main() raises:
    var ctx = DeviceContext()
    var oracle_path = String(PARITY_DIR) + "/oracle_d.safetensors"

    print("[chunk D] loading LingBot Wan VAE decoder weights from", oracle_path)
    var dec = LingBotWanVaeDecoder[LH, LW].load(oracle_path, ctx)

    print("[chunk D] loading oracle z / out")
    var st = ShardedSafeTensors.open(oracle_path)
    var z = Tensor.from_view_as_f32(st.tensor_view("z"), ctx)
    var out_ref = Tensor.from_view_as_f32(st.tensor_view("out"), ctx)
    var ref_host = out_ref.to_host(ctx)
    print("  z shape    =", _sh(z.shape()))
    print("  out shape  =", _sh(out_ref.shape()))

    print("[chunk D] running decode")
    var out = dec.decode(z, ctx)
    print("  decoded shape =", _sh(out.shape()))

    var mine_host = out.to_host(ctx)
    var harness = ParityHarness(0.999)
    var res = harness.compare_host(mine_host, ref_host)

    var mag_mine = _l2(mine_host)
    var mag_ref = _l2(ref_host)
    var ratio = mag_mine / mag_ref if mag_ref > 0.0 else 0.0

    print("[chunk D] decode parity:", res)
    print("[chunk D] cos           =", res.cos)
    print("[chunk D] magnitude     |mine|/|ref| =", ratio, "(|mine|=", mag_mine, "|ref|=", mag_ref, ")")
    print("[chunk D] max_abs_diff  =", res.max_abs)
    if res.passed:
        print("[chunk D] GATE PASS (cos >= 0.999)")
    else:
        print("[chunk D] GATE FAIL (cos < 0.999)")

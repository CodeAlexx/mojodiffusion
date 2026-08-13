# models/lingbotvideo/parity/d_vid_probe.mojo — TEMPORAL Wan VAE decode gate.
#
# Gates LingBotWanVaeDecoder.decode_video (multi-frame causal temporal decode:
# time_conv + DupUp interleave + causal feat-cache) against oracle_dvid.safetensors
# (seeded synthetic weights; z [1,16,2,4,4] -> out [1,3,5,32,32]).
#
# Run (JIT):
#   cd /home/alex/mojodiffusion && \
#     pixi run mojo run -I . serenitymojo/models/lingbotvideo/parity/d_vid_probe.mojo

from std.math import sqrt
from max.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.models.lingbotvideo.vae_decoder import LingBotWanVaeDecoder

comptime PARITY_DIR = "/home/alex/mojodiffusion/serenitymojo/models/lingbotvideo/parity"
comptime LH = 4
comptime LW = 4


def _cos(mine: List[Float32], reference: List[Float32]) -> Tuple[Float64, Float64]:
    var dot: Float64 = 0.0
    var nm: Float64 = 0.0
    var nr: Float64 = 0.0
    var n = len(mine) if len(mine) < len(reference) else len(reference)
    for i in range(n):
        dot += Float64(mine[i]) * Float64(reference[i])
        nm += Float64(mine[i]) * Float64(mine[i])
        nr += Float64(reference[i]) * Float64(reference[i])
    var cos = dot / (sqrt(nm) * sqrt(nr)) if (nm > 0.0 and nr > 0.0) else 0.0
    var mag = sqrt(nm) / sqrt(nr) if nr > 0.0 else 0.0
    return (cos, mag)


def main() raises:
    var ctx = DeviceContext()
    var path = String(PARITY_DIR) + "/oracle_dvid.safetensors"
    print("[DVID] loading oracle + decoder from", path)
    var st = ShardedSafeTensors.open(path)
    var z = Tensor.from_view_as_f32(st.tensor_view("z"), ctx)     # [1,16,2,4,4]
    var out_ref = Tensor.from_view_as_f32(st.tensor_view("out"), ctx).to_host(ctx)

    var dec = LingBotWanVaeDecoder[LH, LW].load(path, ctx)
    print("[DVID] running decode_video on z", z.shape()[0], z.shape()[1],
          z.shape()[2], z.shape()[3], z.shape()[4])
    var pixels = dec.decode_video(z, ctx)   # [1,3,5,32,32]
    var ps = pixels.shape()
    print("[DVID] out shape:", ps[0], ps[1], ps[2], ps[3], ps[4],
          " (expect 1 3 5 32 32)")
    var mine = pixels.to_host(ctx)

    var cm = _cos(mine, out_ref)
    print("[DVID] pixels cos =", cm[0], "  |mine|/|ref| =", cm[1],
          "  n_mine =", len(mine), " n_ref =", len(out_ref))
    if cm[0] >= 0.999 and len(mine) == len(out_ref):
        print("[DVID] ===== TEMPORAL VAE GATE PASS (cos >= 0.999) =====")
    else:
        print("[DVID] ===== TEMPORAL VAE GATE FAIL =====")

# ltx2_noise_spatial_dump.mojo — dump the ltx2 driver's EXACT noise draws for
# the spatial-structure gate (krea2-letterbox class, MJ pending).
#
# WHY: the 07-10 eri2 image run's artifacts were attributed to overtraining;
# Alex retracted that (2026-07-17) — the run used the pre-fix non-torch-
# compatible noise. The krea2-letterbox proof (2026-07-13) showed our noise
# can pass MARGINAL stats (mean/std/tails/adjacent-corr — what
# ltx2_device_noise_stats_probe gates) while carrying SPATIAL structure that
# biases composition. This dump feeds scripts/check_ltx2_noise_spatial.py,
# which compares per-axis autocorrelation + FFT spectrum flatness against
# torch.randn on identical shapes.
#
# Dumps: steps 1..8 of the driver's exact stream — randn(shape, seed +
# step*104729) — for BOTH geometries (video [1,128,4,9,16], image512
# [1,128,1,16,16]), seed 42, F32, to /tmp/ltx2_noise_spatial_dump.safetensors.
#
# Build/run:
#   pixi run mojo build -O2 -I . -Xlinker -lm -Xlinker -lcuda \
#     serenitymojo/models/ltx2/parity/ltx2_noise_spatial_dump.mojo \
#     -o /tmp/ltx2_noise_dump && /tmp/ltx2_noise_dump

from std.collections import List
from std.memory import ArcPointer
from max.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.ops.random import randn


def _sh5(a: Int, b: Int, c: Int, d: Int, e: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a); s.append(b); s.append(c); s.append(d); s.append(e)
    return s^


def main() raises:
    var ctx = DeviceContext()
    comptime SEED = UInt64(42)
    var names = List[String]()
    var tensors = List[ArcPointer[Tensor]]()
    for step in range(1, 9):
        # the driver's exact derivation: cfg.seed + step*104729 (train_ltx2_av)
        var s = SEED + UInt64(step) * UInt64(104729)
        var nv = randn(_sh5(1, 128, 4, 9, 16), s, STDtype.F32, ctx)
        names.append(String("video_step") + String(step))
        tensors.append(ArcPointer(nv^))
        var ni = randn(_sh5(1, 128, 1, 16, 16), s, STDtype.F32, ctx)
        names.append(String("image_step") + String(step))
        tensors.append(ArcPointer(ni^))
    save_safetensors(names, tensors, String("/tmp/ltx2_noise_spatial_dump.safetensors"), ctx)
    print("DUMPED 16 noise tensors (8 steps x 2 geometries) ->",
          "/tmp/ltx2_noise_spatial_dump.safetensors")

# Dump the exact stage-1 initial latents randn draw for structure analysis.
from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.ops.random import randn

comptime OUT = "/home/alex/mojodiffusion/output/ltx2_connector/randn_probe.safetensors"


def main() raises:
    var ctx = DeviceContext()
    var sh = List[Int]()
    sh.append(1)
    sh.append(128)
    sh.append(16)
    sh.append(16)
    sh.append(24)
    var v42 = randn(sh.copy(), UInt64(42), STDtype.BF16, ctx)
    var v43 = randn(sh.copy(), UInt64(1337), STDtype.BF16, ctx)
    var names = List[String]()
    names.append(String("seed42"))
    names.append(String("seed1337"))
    var tensors = List[ArcPointer[Tensor]]()
    tensors.append(ArcPointer(v42^))
    tensors.append(ArcPointer(v43^))
    save_safetensors(names, tensors, String(OUT), ctx)
    print("wrote", OUT)

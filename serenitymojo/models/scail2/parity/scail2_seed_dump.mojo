# Emit SCAIL-2's production same-GPU PyTorch-compatible Philox noise.
# Compare with scripts/scail2_seed_oracle.py using safetensors.

from std.collections import List
from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from std.sys import argv

from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.ops.random_torch import randn_torch
from serenitymojo.tensor import Tensor


def main() raises:
    var args = argv()
    if len(args) != 2:
        raise Error("usage: scail2_seed_dump <output.safetensors>")
    var ctx = DeviceContext()
    var noise = randn_torch([16, 17, 64, 112], UInt64(42), ctx)
    var names = List[String]()
    names.append(String("noise"))
    var tensors = List[ArcPointer[Tensor]]()
    tensors.append(ArcPointer(noise^))
    save_safetensors(names, tensors, String(args[1]), ctx)
    print("GATE SCAIL-2 seed 42 noise emitted [16,17,64,112]")

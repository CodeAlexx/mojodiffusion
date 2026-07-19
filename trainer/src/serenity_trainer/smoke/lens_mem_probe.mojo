# lens_mem_probe.mojo — VRAM diagnosis for the Lens 1024 single forward.
# Loads the REAL transformer weights (BF16 resident) and runs ONE
# lens_forward_full_infer[S_IMG=4096, S_TXT=201] forward on synthetic finite
# inputs. Run with LENS_MEM_DEBUG=1 to print per-block + intra-block (after-qkv /
# after-sdpa / after-mlp) used-VRAM, isolating the single biggest allocation.
# This is a memory probe, NOT a parity gate.

from std.gpu.host import DeviceContext
from std.math import sin as fsin

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype

from serenity_trainer.modelLoader.LensModelLoader import (
    LensWeights, LENS_TRANSFORMER_DIR,
)
from serenity_trainer.model.LensDiT import (
    build_lens_lora_set, LArc, lens_forward_full_infer,
)

comptime H_PX = 1024
comptime W_PX = 1024
comptime S_IMG = (H_PX // 16) * (W_PX // 16)   # 4096
comptime S_TXT = 201
comptime HID = 2880
comptime MIB = 1024.0 * 1024.0


def _synth(n: Int, seed: Float32, scale: Float32, bias: Float32, ctx: DeviceContext) raises -> Tensor:
    var vals = List[Float32]()
    for i in range(n):
        vals.append(scale * fsin(seed + Float32(i) * 0.0007) + bias)
    var sh = List[Int]()
    sh.append(n)
    return Tensor.from_host(vals, sh^, STDtype.BF16, ctx)


def _reshape3(t: Tensor, a: Int, b: Int, c: Int, ctx: DeviceContext) raises -> Tensor:
    var vals = t.to_host(ctx)
    var sh = List[Int]()
    sh.append(a); sh.append(b); sh.append(c)
    return Tensor.from_host(vals, sh^, STDtype.BF16, ctx)


def main() raises:
    var ctx = DeviceContext()
    var mi0 = ctx.get_memory_info()
    var total = Float64(Int(mi0[1]))
    print("=== Lens 1024 memory probe (S_IMG =", S_IMG, " S_TXT =", S_TXT, ") ===")
    print("[mem] start used =", Int((total - Float64(Int(mi0[0]))) / MIB),
          "MiB  total =", Int(total / MIB), "MiB")

    var weights = LensWeights.load(String(LENS_TRANSFORMER_DIR), ctx)
    var mi1 = ctx.get_memory_info()
    print("[mem] weights loaded (", weights.count(), "tensors) used =",
          Int((total - Float64(Int(mi1[0]))) / MIB), "MiB")

    var loras = build_lens_lora_set(8, Float32(8.0), ctx)

    var hidden = _reshape3(_synth(S_IMG * 128, 0.1, 0.05, 0.0, ctx), 1, S_IMG, 128, ctx)
    var t0 = _reshape3(_synth(S_TXT * HID, 0.2, 0.03, 0.0, ctx), 1, S_TXT, HID, ctx)
    var t1 = _reshape3(_synth(S_TXT * HID, 0.3, 0.03, 0.0, ctx), 1, S_TXT, HID, ctx)
    var t2 = _reshape3(_synth(S_TXT * HID, 0.4, 0.03, 0.0, ctx), 1, S_TXT, HID, ctx)
    var t3 = _reshape3(_synth(S_TXT * HID, 0.5, 0.03, 0.0, ctx), 1, S_TXT, HID, ctx)

    var mvals = List[Float32]()
    for i in range(S_TXT):
        mvals.append(Float32(1.0) if i < 104 else Float32(0.0))
    var msh = List[Int]()
    msh.append(1); msh.append(S_TXT)
    var mask = Tensor.from_host(mvals^, msh^, STDtype.F32, ctx)

    var mi2 = ctx.get_memory_info()
    print("[mem] before forward used =",
          Int((total - Float64(Int(mi2[0]))) / MIB), "MiB")

    var flow = lens_forward_full_infer[S_IMG, S_TXT](
        hidden, t0, t1, t2, t3, mask, Float32(0.9), weights, loras, ctx
    )
    ctx.synchronize()
    var mi3 = ctx.get_memory_info()
    var fh = flow.to_host(ctx)
    print("[mem] after forward used =",
          Int((total - Float64(Int(mi3[0]))) / MIB), "MiB")
    print("[probe] flow elems =", len(fh), " flow[0] =", fh[0], " flow[last] =", fh[len(fh) - 1])
    print("=== probe done ===")

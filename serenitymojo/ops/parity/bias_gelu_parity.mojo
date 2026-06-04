# bias_gelu_parity.mojo — GPU bf16 parity for fused bias_gelu vs torch oracle.
#
# Verifies serenitymojo/ops/fused_bias_gelu.mojo `bias_gelu` against a torch
# GPU-bf16 oracle (bias_gelu_oracle.py) computing the SAME math:
#   o = GELU_tanh(x + bias)   (tanh-approx, flame-core bias_gelu).
# Inputs (x, bias) are read from the SAME .bin the oracle dumped so both sides
# are fed byte-identical f32 inputs; both run in bf16 on GPU. Gate: cos >= 0.999.
# Also reports the magnitude ratio (||actual|| / ||ref||).
#
# Run the oracle first, then the probe:
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/ops/parity/bias_gelu_oracle.py
#   pixi run mojo run -I . serenitymojo/ops/parity/bias_gelu_parity.mojo

from std.math import sqrt
from std.gpu.host import DeviceContext
from std.memory import alloc
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.ops.fused_bias_gelu import bias_gelu
from serenitymojo.ops.cast import cast_tensor


comptime DIR = "/home/alex/mojodiffusion/serenitymojo/ops/parity/"
comptime ROWS = 12
comptime H = 64


# Read a raw little-endian f32 .bin file into a List[Float32].
def _read_f32_bin(name: String) raises -> List[Float32]:
    var path = String(DIR) + name
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open ") + path + " (run the oracle first)")
    var nbytes = file_size(fd)
    if nbytes <= 0:
        _ = sys_close(fd)
        raise Error(String("empty bin: ") + path)
    var buf = alloc[UInt8](nbytes)
    var done = 0
    while done < nbytes:
        var got = sys_pread(fd, buf + done, nbytes - done, done)
        if got <= 0:
            break
        done += got
    _ = sys_close(fd)
    var nfloats = done // 4
    var out = List[Float32]()
    var fptr = buf.bitcast[Float32]()
    for i in range(nfloats):
        out.append(fptr[i])
    buf.free()
    return out^


def main() raises:
    var ctx = DeviceContext()
    print("=== fused bias_gelu parity (rows=", ROWS, " H=", H, ", bf16 GPU) ===")

    var x_h = _read_f32_bin("bias_gelu_x.bin")
    var b_h = _read_f32_bin("bias_gelu_bias.bin")
    var refv = _read_f32_bin("bias_gelu_ref.bin")
    if len(x_h) != ROWS * H or len(b_h) != H or len(refv) != ROWS * H:
        raise Error("bias_gelu_parity: input/ref size mismatch with oracle")

    # Build f32 device tensors from the SAME bytes, then cast to bf16 on GPU so
    # the Mojo path runs in bf16 exactly like the torch oracle.
    var x_f32 = Tensor.from_host(x_h.copy(), [ROWS, H], STDtype.F32, ctx)
    var b_f32 = Tensor.from_host(b_h.copy(), [H], STDtype.F32, ctx)
    var x_bf16 = cast_tensor(x_f32, STDtype.BF16, ctx)
    var b_bf16 = cast_tensor(b_f32, STDtype.BF16, ctx)

    var out_bf16 = bias_gelu(x_bf16, b_bf16, ctx)
    var out_f32 = cast_tensor(out_bf16, STDtype.F32, ctx)

    var h = ParityHarness(0.999)
    var r = h.compare(out_f32, refv, ctx)
    print("    bias_gelu(bf16):", r)

    # magnitude ratio ||actual|| / ||ref||
    var got = out_f32.to_host(ctx)
    var na: Float64 = 0.0
    var nb: Float64 = 0.0
    for i in range(len(got)):
        na += Float64(got[i]) * Float64(got[i])
        nb += Float64(refv[i]) * Float64(refv[i])
    var mag_ratio = sqrt(na) / sqrt(nb) if nb > 0.0 else 0.0
    print("    magRatio (||a||/||ref||):", mag_ratio)

    if r.passed:
        print("PASS: bias_gelu bf16 cos>=0.999")
    else:
        raise Error("bias_gelu_parity gate FAILED (cos<0.999)")

# serenitymojo/ops/tests/sage_attention_int8_mma_gate.mojo
#
# Host-exact gate for the local m16n8k32.s8 tensor-core primitive.

from max.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.sage_attention_int8 import sage_int8_mma_tile
from serenitymojo.tensor import Tensor


def _i8_tensor(
    values: List[Int32], shape: List[Int], ctx: DeviceContext
) raises -> Tensor:
    var host = ctx.enqueue_create_host_buffer[DType.uint8](len(values))
    for i in range(len(values)):
        host.unsafe_ptr()[i] = UInt8(Int(values[i]) & 255)
    var dev = ctx.enqueue_create_buffer[DType.uint8](len(values))
    ctx.enqueue_copy(dst_buf=dev, src_buf=host)
    ctx.synchronize()
    return Tensor(dev^, shape.copy(), STDtype.I8)


def main() raises:
    var ah = List[Int32]()
    var bh = List[Int32]()
    for m in range(16):
        for k in range(32):
            ah.append(Int32(((m * 17 + k * 7 + 3) % 23) - 11))
    for k in range(32):
        for n in range(8):
            bh.append(Int32(((k * 13 + n * 5 + 1) % 19) - 9))

    var ctx = DeviceContext()
    var a = _i8_tensor(ah, [16, 32], ctx)
    var b = _i8_tensor(bh, [32, 8], ctx)
    var got = sage_int8_mma_tile(a, b, ctx)
    var ghost = ctx.enqueue_create_host_buffer[DType.uint8](16 * 8 * 4)
    ctx.enqueue_copy(dst_buf=ghost, src_buf=got.buf)
    ctx.synchronize()
    var gp = ghost.unsafe_ptr().bitcast[Int32]()

    var mismatches = 0
    var max_abs = Float32(0.0)
    for m in range(16):
        for n in range(8):
            var want = Int32(0)
            for k in range(32):
                want += Int32(ah[m * 32 + k]) * Int32(bh[k * 8 + n])
            var gv = gp[m * 8 + n]
            var d = gv - want
            var ad = Float32(d if d >= 0 else -d)
            if ad > max_abs:
                max_abs = ad
            if gv != want:
                mismatches += 1
    print("sage int8 MMA m16n8k32: mismatches=", mismatches,
          " max_abs=", max_abs)
    if mismatches != 0:
        raise Error("sage int8 MMA register-map gate failed")
    print("PASS: signed-int8 tensor-core tile is host-exact")

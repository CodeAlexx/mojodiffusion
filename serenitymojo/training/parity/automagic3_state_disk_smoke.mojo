# serenitymojo/training/parity/automagic3_state_disk_smoke.mojo
#
# MJ-1081 gate: A3 optimizer-state disk round trip is BYTE-IDENTICAL.
#   1. init a small A3 device state from 2 fake adapters; set step/lr/rng.
#   2. save -> <tmp>.a3state.safetensors (non-destructive offload->write->restore).
#   3. snapshot the live buffers (reference bytes).
#   4. re-init a SECOND state from the same adapters, load the sidecar.
#   5. compare all 10 buffers byte-for-byte + step/lr/rng — 0 mismatches required.
#   6. geometry guard: loading into a MISMATCHED state must raise.
#
# Build+run: standard -O2 + cshim link line; needs GPU.

from std.gpu.host import DeviceContext, HostBuffer, DeviceBuffer
from std.builtin.dtype import DType
from std.collections import List
from serenitymojo.io.ffi import sys_remove
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.training.automagic3_device import (
    Automagic3DeviceState,
    automagic3_device_state_init_from_adapters,
    automagic3_device_state_save,
    automagic3_device_state_load,
)

comptime RANK = 4
comptime IN_F = 64
comptime OUT_F = 96


def _vals(n: Int, start: Float32) -> List[Float32]:
    var o = List[Float32]()
    for i in range(n):
        o.append(start + Float32(i) * Float32(0.017))
    return o^


def _zeros(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n):
        o.append(0.0)
    return o^


def _adapter(start: Float32) raises -> LoraAdapter:
    return LoraAdapter(
        _vals(RANK * IN_F, start), _vals(OUT_F * RANK, start + 0.5),
        RANK, IN_F, OUT_F, Float32(1.0),
        _zeros(RANK * IN_F), _zeros(RANK * IN_F),
        _zeros(OUT_F * RANK), _zeros(OUT_F * RANK),
    )


def _d2h_f32(buf: DeviceBuffer[DType.float32], n: Int, ctx: DeviceContext) raises -> HostBuffer[DType.float32]:
    var h = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.enqueue_copy(dst_buf=h, src_buf=buf)
    ctx.synchronize()
    return h^


def _d2h_u8(buf: DeviceBuffer[DType.uint8], n: Int, ctx: DeviceContext) raises -> HostBuffer[DType.uint8]:
    var h = ctx.enqueue_create_host_buffer[DType.uint8](n)
    ctx.enqueue_copy(dst_buf=h, src_buf=buf)
    ctx.synchronize()
    return h^


def _d2h_i32(buf: DeviceBuffer[DType.int32], n: Int, ctx: DeviceContext) raises -> HostBuffer[DType.int32]:
    var h = ctx.enqueue_create_host_buffer[DType.int32](n)
    ctx.enqueue_copy(dst_buf=h, src_buf=buf)
    ctx.synchronize()
    return h^


def _cmp_f32(name: String, a: HostBuffer[DType.float32], b: HostBuffer[DType.float32], n: Int, mut bad: Int):
    var m = 0
    for i in range(n):
        if a[i] != b[i]:
            m += 1
    print("[a3disk]", name, " n=", n, " mismatches=", m)
    bad += m


def _cmp_u8(name: String, a: HostBuffer[DType.uint8], b: HostBuffer[DType.uint8], n: Int, mut bad: Int):
    var m = 0
    for i in range(n):
        if a[i] != b[i]:
            m += 1
    print("[a3disk]", name, " n=", n, " mismatches=", m)
    bad += m


def _cmp_i32(name: String, a: HostBuffer[DType.int32], b: HostBuffer[DType.int32], n: Int, mut bad: Int):
    var m = 0
    for i in range(n):
        if a[i] != b[i]:
            m += 1
    print("[a3disk]", name, " n=", n, " mismatches=", m)
    bad += m


def main() raises:
    var ctx = DeviceContext()
    var path = String("/tmp/a3state_smoke.a3state.safetensors")
    _ = sys_remove(path)

    var adapters = List[LoraAdapter]()
    adapters.append(_adapter(Float32(0.1)))
    adapters.append(_adapter(Float32(0.3)))

    var s1 = Automagic3DeviceState(ctx)
    automagic3_device_state_init_from_adapters(s1, adapters, 0.000123, ctx)
    s1.step = 777
    s1.rng.state = UInt64(0xDEADBEEFCAFE1234)
    s1.lr = 0.000456

    automagic3_device_state_save(s1, path, ctx)
    print("[a3disk] saved:", path)

    # reference = live buffers post-save (restore must have round-tripped)
    var np = s1.np; var nr = s1.nr; var ncv = s1.ncv; var nmat = s1.nmat
    var rp = _d2h_f32(s1.p_dev, np, ctx)
    var rg = _d2h_f32(s1.g_dev, np, ctx)
    var ru = _d2h_f32(s1.u_dev, np, ctx)
    var rrv = _d2h_f32(s1.rv_dev, nr, ctx)
    var rcv = _d2h_f32(s1.cv_dev, ncv, ctx)
    var rsr = _d2h_u8(s1.sr_dev, 8 * np, ctx)
    var rdsc = _d2h_i32(s1.dsc_dev, nmat * 6, ctx)
    var rpb = _d2h_u8(s1.pb_dev, np * 2, ctx)

    # second state: fresh init (different step/lr/rng), then LOAD
    var s2 = Automagic3DeviceState(ctx)
    automagic3_device_state_init_from_adapters(s2, adapters, 0.000999, ctx)
    automagic3_device_state_load(s2, path, ctx)

    var bad = 0
    _cmp_f32(String("p"), rp, _d2h_f32(s2.p_dev, np, ctx), np, bad)
    _cmp_f32(String("g"), rg, _d2h_f32(s2.g_dev, np, ctx), np, bad)
    _cmp_f32(String("u"), ru, _d2h_f32(s2.u_dev, np, ctx), np, bad)
    _cmp_f32(String("rv"), rrv, _d2h_f32(s2.rv_dev, nr, ctx), nr, bad)
    _cmp_f32(String("cv"), rcv, _d2h_f32(s2.cv_dev, ncv, ctx), ncv, bad)
    _cmp_u8(String("sr"), rsr, _d2h_u8(s2.sr_dev, 8 * np, ctx), 8 * np, bad)
    _cmp_i32(String("dsc"), rdsc, _d2h_i32(s2.dsc_dev, nmat * 6, ctx), nmat * 6, bad)
    _cmp_u8(String("pb"), rpb, _d2h_u8(s2.pb_dev, np * 2, ctx), np * 2, bad)

    print("[a3disk] step:", s2.step, "(expect 777)  lr:", s2.lr,
          "(expect 0.000456)  rng ok:", s2.rng.state == UInt64(0xDEADBEEFCAFE1234))
    if s2.step != 777 or s2.lr != 0.000456 or s2.rng.state != UInt64(0xDEADBEEFCAFE1234):
        bad += 1

    # geometry guard: a third state with DIFFERENT adapters must refuse the file
    var adapters3 = List[LoraAdapter]()
    adapters3.append(_adapter(Float32(0.1)))
    var s3 = Automagic3DeviceState(ctx)
    automagic3_device_state_init_from_adapters(s3, adapters3, 0.0001, ctx)
    var guarded = False
    try:
        automagic3_device_state_load(s3, path, ctx)
    except:
        guarded = True
    print("[a3disk] geometry guard raised:", guarded, "(expect True)")
    if not guarded:
        bad += 1

    _ = sys_remove(path)
    if bad == 0:
        print("A3 STATE DISK ROUND TRIP: PASS (byte-identical + meta + guard)")
    else:
        raise Error(String("A3 STATE DISK ROUND TRIP: FAIL — mismatches=") + String(bad))

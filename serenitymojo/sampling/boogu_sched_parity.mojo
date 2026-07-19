# boogu_sched_parity.mojo — C8a (v1 scheduler) parity gate vs the real torch scheduler.
# Compares build_boogu_timesteps(8) to the dumped torch _timesteps (9 values).
#   /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/sampling/boogu_sched_oracle.py 8
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/sampling/boogu_sched_parity.mojo
from std.gpu.host import DeviceContext
from std.collections import List
from std.memory import alloc
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.sampling.flow_match import build_boogu_timesteps

comptime DUMP = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/boogu_dumps/boogu_sched_ts_8.bin"
comptime N = 8


def _read_bin_f32(path: String) raises -> List[Float32]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open (run boogu_sched_oracle.py 8 first): ") + path)
    var n = file_size(fd)
    var buf = alloc[UInt8](n)
    var done = 0
    while done < n:
        var got = sys_pread(fd, buf + done, n - done, done)
        if got <= 0:
            break
        done += got
    _ = sys_close(fd)
    var fp = buf.bitcast[Float32]()
    var out = List[Float32]()
    for i in range(n // 4):
        out.append(fp[i])
    buf.free()
    return out^


def main() raises:
    print("=== C8a (Boogu v1 scheduler) parity vs torch ===")
    var ref_ts = _read_bin_f32(DUMP)        # 9 values
    var mine = build_boogu_timesteps(N)     # 9 values
    if len(mine) != len(ref_ts):
        raise Error("len mismatch: mine=" + String(len(mine)) + " ref=" + String(len(ref_ts)))
    var maxabs = Float32(0.0)
    for i in range(len(ref_ts)):
        var d = mine[i] - ref_ts[i]
        if d < 0.0:
            d = -d
        if d > maxabs:
            maxabs = d
        print("  i=", i, " mine=", mine[i], " ref=", ref_ts[i])
    print("  max-abs-diff =", maxabs)
    if maxabs > 1.0e-4:
        raise Error("C8a scheduler parity FAIL: max-abs " + String(maxabs) + " > 1e-4")
    print("VERDICT: C8a PASS — Boogu v1 timesteps match torch (max-abs <= 1e-4)")

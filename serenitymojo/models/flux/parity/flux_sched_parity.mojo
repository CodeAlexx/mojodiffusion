# flux_sched_parity.mojo — GATE B2: FLUX.1 sigma schedule exactness vs BFL.
#
# build_flux1_sigma_schedule(20, 4096) vs BFL get_schedule (dumped to
# flux_sched_ref.bin). Deterministic scalar math -> exact equality bar (max abs
# diff < 1e-6). Verifies the dynamic-shift schedule that drives the denoise loop.
#
# Run:
#   python3 - (writes flux_sched_ref.bin, see flux_sched_parity gate notes)
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/models/flux/parity/flux_sched_parity.mojo

from std.collections import List
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc
from serenitymojo.sampling.flux1_dev import build_flux1_sigma_schedule


comptime REF = "/home/alex/mojodiffusion/serenitymojo/models/flux/parity/flux_sched_ref.bin"
comptime STEPS = 20
comptime N_IMG = 4096


def _read_bin_f32(path: String) raises -> List[Float32]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open: ") + path)
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


def _abs(v: Float32) -> Float32:
    return v if v >= 0.0 else -v


def main() raises:
    print("=== GATE B2: FLUX.1 sigma schedule exactness vs BFL ===")
    var mojo = build_flux1_sigma_schedule(STEPS, N_IMG)
    var oracle = _read_bin_f32(REF)
    if len(mojo) != len(oracle):
        raise Error("len mismatch " + String(len(mojo)) + " vs " + String(len(oracle)))
    var maxd: Float32 = 0.0
    for i in range(len(mojo)):
        var d = _abs(mojo[i] - oracle[i])
        if d > maxd:
            maxd = d
        print("  i", i, "mojo", mojo[i], "bfl", oracle[i], "d", d)
    print("  max abs diff =", maxd)
    if maxd > 1.0e-6:
        raise Error("schedule parity FAIL: max abs diff " + String(maxd) + " > 1e-6")
    print("VERDICT: PASS — FLUX.1 sigma schedule exact vs BFL, max abs diff =", maxd)

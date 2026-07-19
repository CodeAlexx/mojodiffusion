# Bernini-R APG parity against pinned ByteDance creator source.
#
# Oracle first:
#   /home/alex/SwarmUI/dlbackend/ComfyUI/venv/bin/python \
#     scripts/bernini_r_apg_oracle.py
# Then:
#   pixi run mojo run -I . serenitymojo/sampling/parity/bernini_apg_parity.mojo

from std.collections import List
from std.memory import alloc
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.sampling.bernini_apg import BerniniAPGMomentum, bernini_apg_guidance


comptime REF = "/home/alex/mojodiffusion-sync/output/checks/bernini_r/apg_oracle/"
comptime B = 2
comptime C = 3
comptime T = 2
comptime H = 2
comptime W = 3


def _read_f32(name: String) raises -> List[Float32]:
    var path = String(REF) + name + String(".bin")
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open oracle fixture: ") + path)
    var n = file_size(fd)
    if n <= 0 or n % 4 != 0:
        _ = sys_close(fd)
        raise Error(String("invalid oracle fixture: ") + path)
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


def _max_abs(actual: List[Float32], expected: List[Float32]) raises -> Float32:
    if len(actual) != len(expected):
        raise Error("Bernini APG parity length mismatch")
    var maximum = Float32(0.0)
    for i in range(len(actual)):
        var d = actual[i] - expected[i]
        if d < 0.0:
            d = -d
        if d > maximum:
            maximum = d
    return maximum


def main() raises:
    var state = BerniniAPGMomentum(-0.5)
    var out_1 = bernini_apg_guidance(
        _read_f32("cond_1"), _read_f32("uncond_1"), 4.0,
        state, 0.5, 2.5, B, C, T, H, W,
    )
    var out_1_diff = _max_abs(out_1, _read_f32("out_1"))
    var running_1_diff = _max_abs(state.running_average, _read_f32("running_1"))
    var out_2 = bernini_apg_guidance(
        _read_f32("cond_2"), _read_f32("uncond_2"), 4.0,
        state, 0.5, 2.5, B, C, T, H, W,
    )
    var out_2_diff = _max_abs(out_2, _read_f32("out_2"))
    var running_2_diff = _max_abs(state.running_average, _read_f32("running_2"))
    print("Bernini APG parity:")
    print("  out_1 max_abs=", out_1_diff)
    print("  running_1 max_abs=", running_1_diff)
    print("  out_2 max_abs=", out_2_diff)
    print("  running_2 max_abs=", running_2_diff)
    var tolerance = Float32(2.0e-6)
    if out_1_diff > tolerance or running_1_diff > tolerance or out_2_diff > tolerance or running_2_diff > tolerance:
        raise Error("Bernini APG oracle parity FAIL")
    print("GATE PASS Bernini APG matches pinned creator source")

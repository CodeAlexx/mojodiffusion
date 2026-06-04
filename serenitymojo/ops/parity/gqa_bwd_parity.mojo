# Parity gate for ops/gqa_backward.mojo (GQA repeat_kv fwd + bwd) vs torch.
# Run (oracle FIRST, SEPARATE command):
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/ops/parity/gqa_bwd_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/ops/parity/gqa_bwd_parity.mojo

from std.gpu.host import DeviceContext
from std.collections import List
from std.memory import alloc
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.gqa_backward import repeat_kv_f32, repeat_kv_backward


comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/ops/parity/"
comptime S = 5
comptime HKV = 8
comptime NREP = 2
comptime DH = 6
comptime H = HKV * NREP


def _read_bin_f32(path: String) raises -> List[Float32]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open: ") + path)
    var n = file_size(fd)
    if n <= 0:
        _ = sys_close(fd)
        raise Error(String("empty/missing ref (run oracle first): ") + path)
    var buf = alloc[UInt8](n)
    var done = 0
    while done < n:
        var got = sys_pread(fd, buf + done, n - done, done)
        if got <= 0:
            break
        done += got
    _ = sys_close(fd)
    var nf = n // 4
    var fp = buf.bitcast[Float32]()
    var out = List[Float32]()
    for i in range(nf):
        out.append(fp[i])
    buf.free()
    return out^


def _in(name: String) raises -> List[Float32]:
    return _read_bin_f32(REF_DIR + name + ".bin")


def main() raises:
    var ctx = DeviceContext()
    print("==== gqa_bwd_parity (repeat_kv fwd+bwd vs torch) ====")
    print("S=", S, " HKV=", HKV, " NREP=", NREP, " DH=", DH)
    var src = Tensor.from_host(_in("in_src"), [1, S, HKV, DH], STDtype.F32, ctx)
    var d_dst = Tensor.from_host(_in("in_d_dst"), [1, S, H, DH], STDtype.F32, ctx)

    var dst = repeat_kv_f32(src, S, HKV, NREP, DH, ctx)
    var d_src = repeat_kv_backward(d_dst, S, HKV, NREP, DH, ctx)

    var harness = ParityHarness()
    var allok = True
    var rf = harness.compare_host(dst.to_host(ctx), _in("ref_dst"))
    print("  cos(dst)  =", rf.cos, " max_abs =", rf.max_abs, " ",
          "PASS" if rf.passed else "FAIL")
    if not rf.passed:
        allok = False
    var rb = harness.compare_host(d_src.to_host(ctx), _in("ref_d_src"))
    print("  cos(d_src)=", rb.cos, " max_abs =", rb.max_abs, " ",
          "PASS" if rb.passed else "FAIL")
    if not rb.passed:
        allok = False
    if allok:
        print("GQA_BWD_GATE: PASS")
    else:
        print("GQA_BWD_GATE: FAIL")

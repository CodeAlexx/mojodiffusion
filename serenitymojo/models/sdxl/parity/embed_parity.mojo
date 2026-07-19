# embed_parity.mojo — GPU gate for SDXL time+label embedding fwd+bwd
# (models/sdxl/embed.mojo) vs torch autograd (embed_oracle.py).
# GATE: emb + both MLPs' Linear weight/bias grads + d_ts + d_y at cos >= 0.999.
#
# Run the oracle first, then:
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/models/sdxl/parity/embed_oracle.py
#   pixi run mojo build -I . -Xlinker -lm serenitymojo/models/sdxl/parity/embed_parity.mojo -o /tmp/sdxl_emb
#   /tmp/sdxl_emb

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.models.sdxl.embed import (
    embed_forward, embed_backward, EmbWeights, EmbActs, EmbFwd, EmbGrads,
)
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc


comptime REF_PATH = (
    "/home/alex/mojodiffusion/serenitymojo/models/sdxl/parity/embed_ref.txt"
)

comptime B = 2
comptime Sdim = 8
comptime Tdim = 16
comptime Adm = 12


def _fill(n: Int, a: Int, b: Int, c: Float32) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append((Float32((i * a) % b) - c) * 0.05)
    return out^

def _sh1(a: Int) -> List[Int]:
    var s = List[Int](); s.append(a); return s^
def _sh2(a: Int, b: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); return s^


def _read_ref(tag: String) raises -> List[Float32]:
    var fd = sys_open(String(REF_PATH), O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open ref: ") + String(REF_PATH))
    var n = file_size(fd)
    if n <= 0:
        _ = sys_close(fd)
        raise Error("empty ref file")
    var buf = alloc[UInt8](n)
    var done = 0
    while done < n:
        var got = sys_pread(fd, buf + done, n - done, done)
        if got <= 0:
            break
        done += got
    _ = sys_close(fd)
    var prefix = tag + " "
    var pl = prefix.byte_length()
    var out = List[Float32]()
    var i = 0
    while i < done:
        var le = i
        while le < done and Int(buf[le]) != 0x0A:
            le += 1
        var is_match = (le - i) > pl
        if is_match:
            for j in range(pl):
                if Int(buf[i + j]) != ord(prefix[byte=j]):
                    is_match = False
                    break
        if is_match:
            var p = i + pl
            while p < le:
                var c = Int(buf[p])
                if c == 0x20:
                    p += 1
                    continue
                var ne = p
                while ne < le and Int(buf[ne]) != 0x20:
                    ne += 1
                var chars = List[UInt8]()
                for q in range(p, ne):
                    chars.append(buf[q])
                var s = String(from_utf8=chars)
                out.append(Float32(atof(s)))
                p = ne + 1
            buf.free()
            return out^
        i = le + 1
    buf.free()
    raise Error(String("ref tag not found: ") + tag)


def main() raises:
    var ctx = DeviceContext()
    var h = ParityHarness()
    var all_pass = True

    # t = [3.0, 17.0] (MUST match embed_oracle.py)
    var th = List[Float32]()
    th.append(3.0); th.append(17.0)
    var t = Tensor.from_host(th, _sh1(B), STDtype.F32, ctx)
    var y = Tensor.from_host(_fill(B * Adm, 5, 11, 5.0), _sh2(B, Adm), STDtype.F32, ctx)

    var t0_w = Tensor.from_host(_fill(Tdim * Sdim, 6, 17, 8.0), _sh2(Tdim, Sdim), STDtype.F32, ctx)
    var t0_b = Tensor.from_host(_fill(Tdim, 3, 9, 4.0), _sh1(Tdim), STDtype.F32, ctx)
    var t2_w = Tensor.from_host(_fill(Tdim * Tdim, 5, 11, 5.0), _sh2(Tdim, Tdim), STDtype.F32, ctx)
    var t2_b = Tensor.from_host(_fill(Tdim, 4, 10, 5.0), _sh1(Tdim), STDtype.F32, ctx)
    var l0_w = Tensor.from_host(_fill(Tdim * Adm, 7, 13, 6.0), _sh2(Tdim, Adm), STDtype.F32, ctx)
    var l0_b = Tensor.from_host(_fill(Tdim, 2, 7, 3.0), _sh1(Tdim), STDtype.F32, ctx)
    var l2_w = Tensor.from_host(_fill(Tdim * Tdim, 6, 17, 8.0), _sh2(Tdim, Tdim), STDtype.F32, ctx)
    var l2_b = Tensor.from_host(_fill(Tdim, 3, 9, 4.0), _sh1(Tdim), STDtype.F32, ctx)

    var w = EmbWeights(t0_w^, t0_b^, t2_w^, t2_b^, l0_w^, l0_b^, l2_w^, l2_b^)

    var fwd = embed_forward[B, Sdim, Tdim, Adm](t, y, w, ctx)
    var r_emb = h.compare_host(fwd.emb.to_host(ctx), _read_ref(String("emb")))
    print("embed emb   vs torch:", r_emb)
    all_pass = all_pass and r_emb.passed

    var go = Tensor.from_host(_fill(B * Tdim, 2, 7, 3.0), _sh2(B, Tdim), STDtype.F32, ctx)
    var g = embed_backward[B, Sdim, Tdim, Adm](go, fwd.acts, w, ctx)

    var r_t0w = h.compare_host(g.dt0_w.to_host(ctx), _read_ref(String("dt0_w")))
    var r_t0b = h.compare_host(g.dt0_b.to_host(ctx), _read_ref(String("dt0_b")))
    var r_t2w = h.compare_host(g.dt2_w.to_host(ctx), _read_ref(String("dt2_w")))
    var r_t2b = h.compare_host(g.dt2_b.to_host(ctx), _read_ref(String("dt2_b")))
    var r_l0w = h.compare_host(g.dl0_w.to_host(ctx), _read_ref(String("dl0_w")))
    var r_l0b = h.compare_host(g.dl0_b.to_host(ctx), _read_ref(String("dl0_b")))
    var r_l2w = h.compare_host(g.dl2_w.to_host(ctx), _read_ref(String("dl2_w")))
    var r_l2b = h.compare_host(g.dl2_b.to_host(ctx), _read_ref(String("dl2_b")))
    var r_dts = h.compare_host(g.d_ts.to_host(ctx), _read_ref(String("d_ts")))
    var r_dy = h.compare_host(g.d_y.to_host(ctx), _read_ref(String("d_y")))

    print("embed dt0_w vs torch:", r_t0w)
    print("embed dt0_b vs torch:", r_t0b)
    print("embed dt2_w vs torch:", r_t2w)
    print("embed dt2_b vs torch:", r_t2b)
    print("embed dl0_w vs torch:", r_l0w)
    print("embed dl0_b vs torch:", r_l0b)
    print("embed dl2_w vs torch:", r_l2w)
    print("embed dl2_b vs torch:", r_l2b)
    print("embed d_ts  vs torch:", r_dts)
    print("embed d_y   vs torch:", r_dy)

    all_pass = (all_pass and r_t0w.passed and r_t0b.passed and r_t2w.passed
        and r_t2b.passed and r_l0w.passed and r_l0b.passed and r_l2w.passed
        and r_l2b.passed and r_dts.passed and r_dy.passed)

    print("")
    if all_pass:
        print("ALL SDXL TIME+LABEL EMBED FWD+BWD GATES PASSED (cos >= 0.999 vs PyTorch)")
    else:
        print("SDXL EMBED PARITY FAILURE")
        raise Error("embed_parity gate failed")

# shape_bwd_parity.mojo — GPU verification of the Tier-0 shape BACKWARD arms.
#
# Phase T1 gate (FULL_PORT_TRAINING_PLAN §5): grad-parity cos >= 0.999 of every
# shape-op backward vs a PyTorch reference (shape_bwd_oracle.py -> shape_bwd_ref.txt).
#
# Run the oracle first, then:
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/ops/parity/shape_bwd_oracle.py
#   pixi run mojo run -I . serenitymojo/ops/parity/shape_bwd_parity.mojo

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.ops.shape_backward import (
    cat_backward, split_backward, slice_backward, reshape_backward,
    transpose_backward, permute_backward, broadcast_backward, repeat_backward,
    where_backward, clamp_backward, maximum_backward, minimum_backward,
    cast_backward, index_select_backward,
)
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc


comptime REF_PATH = (
    "/home/alex/mojodiffusion/serenitymojo/ops/parity/shape_bwd_ref.txt"
)


# ── Deterministic fills — MUST match shape_bwd_oracle.py ──────────────────────
def _fill(n: Int, mul: Int, mod: Int, sub: Float32, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append((Float32((i * mul) % mod) - sub) * scale)
    return out^


def _fill_default(n: Int) -> List[Float32]:
    return _fill(n, 7, 13, Float32(6.0), Float32(0.1))


def _fill_alt(n: Int) -> List[Float32]:
    return _fill(n, 5, 11, Float32(5.0), Float32(0.1))


def _fill_grad(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append((Float32((i * 2) % 7) - 3.0) * 0.05)
    return out^


def _scaled(xs: List[Float32], s: Float32) -> List[Float32]:
    var out = List[Float32]()
    for i in range(len(xs)):
        out.append(xs[i] * s)
    return out^


def _cond_mask(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append(Float32(1.0) if (i % 2 == 0) else Float32(0.0))
    return out^


def _shape(d0: Int) -> List[Int]:
    var s = List[Int]()
    s.append(d0)
    return s^


def _shape2(d0: Int, d1: Int) -> List[Int]:
    var s = List[Int]()
    s.append(d0); s.append(d1)
    return s^


def _shape3(d0: Int, d1: Int, d2: Int) -> List[Int]:
    var s = List[Int]()
    s.append(d0); s.append(d1); s.append(d2)
    return s^


# ── read one tagged space-separated float line (copied from reduce_bwd_parity) ─
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

    # ── CAT bwd: d_y [5,4] -> pieces [2,4],[3,4] along dim 0 ──────────────────
    var cat_gy = Tensor.from_host(_fill_grad(5 * 4), _shape2(5, 4), STDtype.F32, ctx)
    var cat_grads = cat_backward(cat_gy, 2, 3, 0, ctx)
    var r_cat0 = h.compare_host(cat_grads.d_0.to_host(ctx), _read_ref(String("cat_d0")))
    var r_cat1 = h.compare_host(cat_grads.d_1.to_host(ctx), _read_ref(String("cat_d1")))
    print("cat_d0       vs torch:", r_cat0)
    print("cat_d1       vs torch:", r_cat1)
    all_pass = all_pass and r_cat0.passed and r_cat1.passed

    # ── SPLIT bwd: concat 2 piece-grads [2,4] + [3,4]*2 -> d_x [5,4] ──────────
    var sp0 = Tensor.from_host(_fill_grad(2 * 4), _shape2(2, 4), STDtype.F32, ctx)
    var sp1 = Tensor.from_host(_scaled(_fill_grad(3 * 4), 2.0), _shape2(3, 4), STDtype.F32, ctx)
    var split_dx = split_backward(sp0, sp1, 0, ctx)
    var r_split = h.compare_host(split_dx.to_host(ctx), _read_ref(String("split_dx")))
    print("split_dx     vs torch:", r_split)
    all_pass = all_pass and r_split.passed

    # ── SLICE bwd: d_y [3,4] scatter into zeros [6,4] at dim0 start=1 ─────────
    var sl_gy = Tensor.from_host(_fill_grad(3 * 4), _shape2(3, 4), STDtype.F32, ctx)
    var slice_dx = slice_backward(sl_gy, _shape2(6, 4), 0, 1, ctx)
    var r_slice = h.compare_host(slice_dx.to_host(ctx), _read_ref(String("slice_dx")))
    print("slice_dx     vs torch:", r_slice)
    all_pass = all_pass and r_slice.passed

    # ── RESHAPE bwd: d_y [2,12] -> d_x [4,6] ─────────────────────────────────
    var rs_gy = Tensor.from_host(_fill_grad(24), _shape2(2, 12), STDtype.F32, ctx)
    var reshape_dx = reshape_backward(rs_gy, _shape2(4, 6), ctx)
    var r_reshape = h.compare_host(reshape_dx.to_host(ctx), _read_ref(String("reshape_dx")))
    print("reshape_dx   vs torch:", r_reshape)
    all_pass = all_pass and r_reshape.passed

    # ── TRANSPOSE bwd: d_y [5,3] -> d_x [3,5], swap (0,1) ────────────────────
    var tr_gy = Tensor.from_host(_fill_grad(15), _shape2(5, 3), STDtype.F32, ctx)
    var transpose_dx = transpose_backward(tr_gy, 0, 1, ctx)
    var r_transpose = h.compare_host(transpose_dx.to_host(ctx), _read_ref(String("transpose_dx")))
    print("transpose_dx vs torch:", r_transpose)
    all_pass = all_pass and r_transpose.passed

    # ── PERMUTE bwd: forward perm (2,0,1) on [2,3,4] -> d_y [4,2,3] ──────────
    var pm_gy = Tensor.from_host(_fill_grad(24), _shape3(4, 2, 3), STDtype.F32, ctx)
    var pm_perm = List[Int]()
    pm_perm.append(2); pm_perm.append(0); pm_perm.append(1)
    var permute_dx = permute_backward(pm_gy, pm_perm, ctx)
    var r_permute = h.compare_host(permute_dx.to_host(ctx), _read_ref(String("permute_dx")))
    print("permute_dx   vs torch:", r_permute)
    all_pass = all_pass and r_permute.passed

    # ── BROADCAST bwd: d_y [3,4] -> d_x [1,4] (sum over the broadcast dim) ────
    var bc_gy = Tensor.from_host(_fill_grad(12), _shape2(3, 4), STDtype.F32, ctx)
    var broadcast_dx = broadcast_backward(bc_gy, _shape2(1, 4), ctx)
    var r_broadcast = h.compare_host(broadcast_dx.to_host(ctx), _read_ref(String("broadcast_dx")))
    print("broadcast_dx vs torch:", r_broadcast)
    all_pass = all_pass and r_broadcast.passed

    # ── REPEAT bwd: d_y [4,6] -> d_x [2,3] with reps (2,2) ───────────────────
    var rp_gy = Tensor.from_host(_fill_grad(24), _shape2(4, 6), STDtype.F32, ctx)
    var rp_reps = List[Int]()
    rp_reps.append(2); rp_reps.append(2)
    var repeat_dx = repeat_backward(rp_gy, _shape2(2, 3), rp_reps, ctx)
    var r_repeat = h.compare_host(repeat_dx.to_host(ctx), _read_ref(String("repeat_dx")))
    print("repeat_dx    vs torch:", r_repeat)
    all_pass = all_pass and r_repeat.passed

    # ── WHERE bwd: cond (even=1) [8]; d_a, d_b ───────────────────────────────
    var wh_gy = Tensor.from_host(_fill_grad(8), _shape(8), STDtype.F32, ctx)
    var wh_cond = Tensor.from_host(_cond_mask(8), _shape(8), STDtype.F32, ctx)
    var wh_grads = where_backward(wh_gy, wh_cond, ctx)
    var r_wda = h.compare_host(wh_grads.d_a.to_host(ctx), _read_ref(String("where_da")))
    var r_wdb = h.compare_host(wh_grads.d_b.to_host(ctx), _read_ref(String("where_db")))
    print("where_da     vs torch:", r_wda)
    print("where_db     vs torch:", r_wdb)
    all_pass = all_pass and r_wda.passed and r_wdb.passed

    # ── CLAMP bwd: clamp(x,-0.2,0.2) on signed [16] ──────────────────────────
    var cl_gy = Tensor.from_host(_fill_grad(16), _shape(16), STDtype.F32, ctx)
    var cl_x = Tensor.from_host(_fill_default(16), _shape(16), STDtype.F32, ctx)
    var clamp_dx = clamp_backward(cl_gy, cl_x, Float32(-0.2), Float32(0.2), ctx)
    var r_clamp = h.compare_host(clamp_dx.to_host(ctx), _read_ref(String("clamp_dx")))
    print("clamp_dx     vs torch:", r_clamp)
    all_pass = all_pass and r_clamp.passed

    # ── MAXIMUM / MINIMUM bwd: a=default, b=alt, [16] ────────────────────────
    var mx_gy = Tensor.from_host(_fill_grad(16), _shape(16), STDtype.F32, ctx)
    var mx_a = Tensor.from_host(_fill_default(16), _shape(16), STDtype.F32, ctx)
    var mx_b = Tensor.from_host(_fill_alt(16), _shape(16), STDtype.F32, ctx)
    var max_grads = maximum_backward(mx_gy, mx_a, mx_b, ctx)
    var r_maxa = h.compare_host(max_grads.d_a.to_host(ctx), _read_ref(String("max_da")))
    var r_maxb = h.compare_host(max_grads.d_b.to_host(ctx), _read_ref(String("max_db")))
    print("max_da       vs torch:", r_maxa)
    print("max_db       vs torch:", r_maxb)
    all_pass = all_pass and r_maxa.passed and r_maxb.passed

    var mn_gy = Tensor.from_host(_fill_grad(16), _shape(16), STDtype.F32, ctx)
    var mn_a = Tensor.from_host(_fill_default(16), _shape(16), STDtype.F32, ctx)
    var mn_b = Tensor.from_host(_fill_alt(16), _shape(16), STDtype.F32, ctx)
    var min_grads = minimum_backward(mn_gy, mn_a, mn_b, ctx)
    var r_mina = h.compare_host(min_grads.d_a.to_host(ctx), _read_ref(String("min_da")))
    var r_minb = h.compare_host(min_grads.d_b.to_host(ctx), _read_ref(String("min_db")))
    print("min_da       vs torch:", r_mina)
    print("min_db       vs torch:", r_minb)
    all_pass = all_pass and r_mina.passed and r_minb.passed

    # ── CAST bwd: identity [12] ──────────────────────────────────────────────
    var ca_gy = Tensor.from_host(_fill_grad(12), _shape(12), STDtype.F32, ctx)
    var cast_dx = cast_backward(ca_gy, ctx)
    var r_cast = h.compare_host(cast_dx.to_host(ctx), _read_ref(String("cast_dx")))
    print("cast_dx      vs torch:", r_cast)
    all_pass = all_pass and r_cast.passed

    # ── INDEX_SELECT bwd: table [5,4], idx [0,2,2,4], d_y [4,4] -> d_x [5,4] ──
    var is_gy = Tensor.from_host(_fill_grad(16), _shape2(4, 4), STDtype.F32, ctx)
    var is_idx = List[Int]()
    is_idx.append(0); is_idx.append(2); is_idx.append(2); is_idx.append(4)
    var index_select_dx = index_select_backward(is_gy, is_idx, 0, _shape2(5, 4), ctx)
    var r_is = h.compare_host(index_select_dx.to_host(ctx), _read_ref(String("index_select_dx")))
    print("index_select_dx vs torch:", r_is)
    all_pass = all_pass and r_is.passed

    print("")
    if all_pass:
        print("ALL SHAPE BACKWARD GATES PASSED (cos >= 0.999 vs PyTorch)")
    else:
        print("SHAPE BACKWARD PARITY FAILURE")
        raise Error("shape_bwd_parity gate failed")

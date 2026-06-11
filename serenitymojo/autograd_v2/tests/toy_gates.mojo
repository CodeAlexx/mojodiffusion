# autograd_v2/tests/toy_gates.mojo - Phase P1 toy-graph gates for the
# dependency-counted engine (AUTOGRAD_V2_MOJO_DESIGN.md P1; flame's Phase 2
# toy list, engine.rs tests). Each gate builds a tiny F32 graph through the
# generic Graph.record surface, drives engine.execute, and compares leaf grads
# EXACTLY (values chosen to be exact in F32: 1.0, 2.0, 0.5, ...).
#
# Build: cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#   pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
#     serenitymojo/autograd_v2/tests/toy_gates.mojo -o /tmp/toy_gates
# Run:   LD_LIBRARY_PATH=/home/alex/mojodiffusion/.pixi/envs/default/lib /tmp/toy_gates

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.autograd_v2.node import (
    Edge,
    TArc,
    OPK_ADD,
    OPK_MUL,
    OPK_MATMUL,
    OPK_SUM,
    _raw_add,
    _raw_mul,
)
from serenitymojo.autograd_v2.graph import Graph
from serenitymojo.autograd_v2.engine import execute


def _fmt(vals: List[Float32]) -> String:
    var s = String("[")
    for i in range(len(vals)):
        if i > 0:
            s += ","
        s += String(Float32(vals[i]))
    s += "]"
    return s^


def _gate(name: String, got: List[Float32], want: List[Float32]) -> Bool:
    var ok = len(got) == len(want)
    if ok:
        for i in range(len(want)):
            if got[i] != want[i]:  # exact-equality contract (F32-exact values)
                ok = False
    var verdict = String("PASS") if ok else String("FAIL")
    print("GATE " + name + " " + verdict + " got=" + _fmt(got) + " want=" + _fmt(want))
    return ok


def _scalar_one(ctx: DeviceContext) raises -> TArc:
    # The root grad seed: a [1] F32 tensor holding 1.0 (the engine's caller
    # supplies the seed; flame defaults to ones_like, engine.rs:317-326).
    return TArc(Tensor.from_host([Float32(1.0)], [1], STDtype.F32, ctx))


def _consts(n: Int, v: Float32) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(v)
    return out^


# Record a SUM node over tensor `t` (already id-stamped + registered in g) and
# return the SUM node idx. saved[0] = the forward input (ones_like target).
def _record_sum(mut g: Graph, t: Tensor, ctx: DeviceContext) raises -> Int:
    var yid = g.fresh_tensor_id()
    var edges = List[Edge]()
    edges.append(g.edge_for(t.id))
    var saved = List[TArc]()
    saved.append(TArc(t.clone(ctx)))
    var oids: List[Int] = [yid]
    return g.record(OPK_SUM, edges^, saved^, List[Int](), List[Float32](), oids)


# ── gate 1: y = sum(x), x [2,3] -> dx == ones ────────────────────────────────
def gate_single_leaf_sum(ctx: DeviceContext) raises -> Bool:
    var g = Graph()
    var xv: List[Float32] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    var x = Tensor.from_host(xv, [2, 3], STDtype.F32, ctx)
    x.set_id(g.fresh_tensor_id())
    _ = g.leaf(x.id)
    var root = _record_sum(g, x, ctx)
    var grads = execute(g, root, _scalar_one(ctx), ctx)
    var dx = grads[x.id][].to_host(ctx)
    return _gate("single_leaf_sum", dx, _consts(6, 1.0))


# Shared builder for two_branches / diamond: y = sum(x*c1 + x*c2) with c1, c2
# untracked constants -> dx = c1 + c2 elementwise. Two MUL nodes route dx into
# the SAME leaf slot -> exercises InputBuffer accumulation across two edges
# into one leaf (flame input_buffer.rs out-of-place accumulate path).
def _two_branch_dx(c1: Float32, c2: Float32, ctx: DeviceContext) raises -> List[Float32]:
    var g = Graph()
    var xv: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    var x = Tensor.from_host(xv, [4], STDtype.F32, ctx)
    x.set_id(g.fresh_tensor_id())
    _ = g.leaf(x.id)
    var a = Tensor.from_host(_consts(4, c1), [4], STDtype.F32, ctx)  # id 0: untracked
    var b = Tensor.from_host(_consts(4, c2), [4], STDtype.F32, ctx)

    # u = x*a (MUL; edge 1 is null because a is untracked -> dB dropped, C7)
    var u = _raw_mul(x, a, ctx)
    u.set_id(g.fresh_tensor_id())
    var ue = List[Edge]()
    ue.append(g.edge_for(x.id))
    ue.append(g.edge_for(a.id))  # a.id == 0 -> null edge
    var us = List[TArc]()
    us.append(TArc(x.clone(ctx)))
    us.append(TArc(a.clone(ctx)))
    var uo: List[Int] = [u.id]
    _ = g.record(OPK_MUL, ue^, us^, List[Int](), List[Float32](), uo)

    # v = x*b
    var v = _raw_mul(x, b, ctx)
    v.set_id(g.fresh_tensor_id())
    var ve = List[Edge]()
    ve.append(g.edge_for(x.id))
    ve.append(g.edge_for(b.id))
    var vs = List[TArc]()
    vs.append(TArc(x.clone(ctx)))
    vs.append(TArc(b.clone(ctx)))
    var vo: List[Int] = [v.id]
    _ = g.record(OPK_MUL, ve^, vs^, List[Int](), List[Float32](), vo)

    # s = u + v (ADD saves nothing)
    var s = _raw_add(u, v, ctx)
    s.set_id(g.fresh_tensor_id())
    var se = List[Edge]()
    se.append(g.edge_for(u.id))
    se.append(g.edge_for(v.id))
    var so: List[Int] = [s.id]
    _ = g.record(OPK_ADD, se^, List[TArc](), List[Int](), List[Float32](), so)

    var root = _record_sum(g, s, ctx)
    var grads = execute(g, root, _scalar_one(ctx), ctx)
    return grads[x.id][].to_host(ctx)


# ── gate 2: y = sum(x*a + x*b) -> dx == a+b ──────────────────────────────────
def gate_two_branches(ctx: DeviceContext) raises -> Bool:
    var dx = _two_branch_dx(2.0, 0.5, ctx)
    return _gate("two_branches", dx, _consts(4, 2.5))


# ── gate 3: diamond u=x*c1, v=x*c2, y=sum(u+v) -> dx == c1+c2 ───────────────
def gate_diamond(ctx: DeviceContext) raises -> Bool:
    var dx = _two_branch_dx(3.0, 0.25, ctx)
    return _gate("diamond", dx, _consts(4, 3.25))


# ── gate 4: y = sum(A@B), A [2,3], B [3,2] -> dA, dB vs hand-computed ───────
def gate_matmul_grads(ctx: DeviceContext) raises -> Bool:
    comptime M = 2
    comptime K = 3
    comptime N = 2
    var g = Graph()
    var av: List[Float32] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]   # A [2,3]
    var bv: List[Float32] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]   # B [3,2]
    var A = Tensor.from_host(av, [M, K], STDtype.F32, ctx)
    var B = Tensor.from_host(bv, [K, N], STDtype.F32, ctx)
    A.set_id(g.fresh_tensor_id())
    B.set_id(g.fresh_tensor_id())
    _ = g.leaf(A.id)
    _ = g.leaf(B.id)

    # Forward C = A@B computed host-side ("Python-style" per the gate spec);
    # the engine only reads C's SHAPE (SUM's ones_like target), never values.
    var cv = List[Float32]()
    for m in range(M):
        for n in range(N):
            var acc: Float32 = 0.0
            for k in range(K):
                acc += av[m * K + k] * bv[k * N + n]
            cv.append(acc)
    var C = Tensor.from_host(cv, [M, N], STDtype.F32, ctx)
    C.set_id(g.fresh_tensor_id())
    var ce = List[Edge]()
    ce.append(g.edge_for(A.id))
    ce.append(g.edge_for(B.id))
    var cs = List[TArc]()
    cs.append(TArc(A.clone(ctx)))
    cs.append(TArc(B.clone(ctx)))
    var cmeta: List[Int] = [M, N, K]
    var co: List[Int] = [C.id]
    _ = g.record(OPK_MATMUL, ce^, cs^, cmeta^, List[Float32](), co)

    var root = _record_sum(g, C, ctx)
    var grads = execute(g, root, _scalar_one(ctx), ctx)

    # Hand-computed expectations with dC = ones[M,N]:
    #   dA[m,k] = sum_n B[k,n]   ;   dB[k,n] = sum_m A[m,k]
    var exp_da = List[Float32]()
    for _m in range(M):
        for k in range(K):
            var s: Float32 = 0.0
            for n in range(N):
                s += bv[k * N + n]
            exp_da.append(s)
    var exp_db = List[Float32]()
    for k in range(K):
        for _n in range(N):
            var s: Float32 = 0.0
            for m in range(M):
                s += av[m * K + k]
            exp_db.append(s)

    var da = grads[A.id][].to_host(ctx)
    var db = grads[B.id][].to_host(ctx)
    var ok_a = _gate("matmul_grads_dA", da, exp_da)
    var ok_b = _gate("matmul_grads_dB", db, exp_db)
    return ok_a and ok_b


# ── gate 5: undefined slot — a node with a dropped (-1) edge routes nothing
# and doesn't crash; remaining grads still correct. y = sum(x*c), c untracked
# -> dx == c; dB = g*x is produced by the MUL arm and DROPPED at the explicit
# Edge.null() (flame Edge{function: None}, node.rs:52-76 / engine.rs:532-535).
def gate_undefined_slot(ctx: DeviceContext) raises -> Bool:
    var g = Graph()
    var xv: List[Float32] = [1.0, 2.0]
    var cv: List[Float32] = [0.5, 4.0]
    var x = Tensor.from_host(xv, [2], STDtype.F32, ctx)
    var c = Tensor.from_host(cv, [2], STDtype.F32, ctx)  # untracked
    x.set_id(g.fresh_tensor_id())
    _ = g.leaf(x.id)
    var u = _raw_mul(x, c, ctx)
    u.set_id(g.fresh_tensor_id())
    var ue = List[Edge]()
    ue.append(g.edge_for(x.id))
    ue.append(Edge.null())  # explicit dropped-grad edge
    var us = List[TArc]()
    us.append(TArc(x.clone(ctx)))
    us.append(TArc(c.clone(ctx)))
    var uo: List[Int] = [u.id]
    _ = g.record(OPK_MUL, ue^, us^, List[Int](), List[Float32](), uo)
    var root = _record_sum(g, u, ctx)
    var grads = execute(g, root, _scalar_one(ctx), ctx)
    var dx = grads[x.id][].to_host(ctx)
    return _gate("undefined_slot", dx, cv)


# ── gate 6: fire exactly once — multi-consumer node. u = x*c is consumed
# TWICE by one ADD (s = u+u): dep[u] = 2, u must fire ONCE with the
# accumulated grad (1+1) -> dx = 2*c. The engine's fired==reachable invariant
# raises on any double/missed fire (dep-count exactness).
def gate_fire_exactly_once(ctx: DeviceContext) raises -> Bool:
    var g = Graph()
    var xv: List[Float32] = [1.0, 2.0, 3.0]
    var x = Tensor.from_host(xv, [3], STDtype.F32, ctx)
    var c = Tensor.from_host(_consts(3, 2.0), [3], STDtype.F32, ctx)
    x.set_id(g.fresh_tensor_id())
    _ = g.leaf(x.id)
    var u = _raw_mul(x, c, ctx)
    u.set_id(g.fresh_tensor_id())
    var ue = List[Edge]()
    ue.append(g.edge_for(x.id))
    ue.append(g.edge_for(c.id))
    var us = List[TArc]()
    us.append(TArc(x.clone(ctx)))
    us.append(TArc(c.clone(ctx)))
    var uo: List[Int] = [u.id]
    _ = g.record(OPK_MUL, ue^, us^, List[Int](), List[Float32](), uo)
    # s = u + u: BOTH edges point at the u-node (multi-consumer).
    var s = _raw_add(u, u, ctx)
    s.set_id(g.fresh_tensor_id())
    var se = List[Edge]()
    se.append(g.edge_for(u.id))
    se.append(g.edge_for(u.id))
    var so: List[Int] = [s.id]
    _ = g.record(OPK_ADD, se^, List[TArc](), List[Int](), List[Float32](), so)
    var root = _record_sum(g, s, ctx)
    # execute raises if fired != reachable; returning at all proves the
    # invariant held for this multi-consumer graph.
    var grads = execute(g, root, _scalar_one(ctx), ctx)
    var dx = grads[x.id][].to_host(ctx)
    return _gate("fire_exactly_once", dx, _consts(3, 4.0))


# ── gate 7: nested execute — the engine carries NO state across calls (flame
# Engine is field-less, engine.rs:171-177; here every counter/buffer is a
# local). P1's apply() cannot host user code (no fn-values, C2), so the
# flame-faithful reentrancy reading is: with one execute's results live, run a
# second graph's execute, then RE-execute the first graph - all three results
# must be correct and independent.
def gate_nested_execute(ctx: DeviceContext) raises -> Bool:
    # graph A: y = sum(x), x [2] -> dx = ones
    var ga = Graph()
    var xv: List[Float32] = [1.0, 2.0]
    var x = Tensor.from_host(xv, [2], STDtype.F32, ctx)
    x.set_id(ga.fresh_tensor_id())
    _ = ga.leaf(x.id)
    var root_a = _record_sum(ga, x, ctx)
    var grads_a1 = execute(ga, root_a, _scalar_one(ctx), ctx)

    # graph B (built+run while grads_a1 is live): y = sum(w*c) -> dw = c
    var gb = Graph()
    var wv: List[Float32] = [1.0, 2.0, 3.0]
    var w = Tensor.from_host(wv, [3], STDtype.F32, ctx)
    var c = Tensor.from_host(_consts(3, 3.0), [3], STDtype.F32, ctx)
    w.set_id(gb.fresh_tensor_id())
    _ = gb.leaf(w.id)
    var u = _raw_mul(w, c, ctx)
    u.set_id(gb.fresh_tensor_id())
    var ue = List[Edge]()
    ue.append(gb.edge_for(w.id))
    ue.append(gb.edge_for(c.id))
    var us = List[TArc]()
    us.append(TArc(w.clone(ctx)))
    us.append(TArc(c.clone(ctx)))
    var uo: List[Int] = [u.id]
    _ = gb.record(OPK_MUL, ue^, us^, List[Int](), List[Float32](), uo)
    var root_b = _record_sum(gb, u, ctx)
    var grads_b = execute(gb, root_b, _scalar_one(ctx), ctx)

    # re-execute graph A: all engine state was call-local, the graph retains
    # its saved tensors (retain_graph semantics are implicit in P1).
    var grads_a2 = execute(ga, root_a, _scalar_one(ctx), ctx)

    var ok1 = _gate("nested_execute_a1", grads_a1[x.id][].to_host(ctx), _consts(2, 1.0))
    var ok2 = _gate("nested_execute_b", grads_b[w.id][].to_host(ctx), _consts(3, 3.0))
    var ok3 = _gate("nested_execute_a2", grads_a2[x.id][].to_host(ctx), _consts(2, 1.0))
    return ok1 and ok2 and ok3


def main() raises:
    var ctx = DeviceContext()
    var ok = True
    ok = gate_single_leaf_sum(ctx) and ok
    ok = gate_two_branches(ctx) and ok
    ok = gate_diamond(ctx) and ok
    ok = gate_matmul_grads(ctx) and ok
    ok = gate_undefined_slot(ctx) and ok
    ok = gate_fire_exactly_once(ctx) and ok
    ok = gate_nested_execute(ctx) and ok
    if not ok:
        raise Error("toy_gates: at least one GATE FAILED")
    print("ALL P1 TOY GATES PASS")

# autograd_v2/tests/krea2_op_parity.mojo — per-op BIT-EQUAL gates for the 2 new
# device-routable krea2 kinds (OPK_REPEAT_KV, OPK_SIGMOID), the krea2 siblings of
# dit_op_parity.mojo (AUTOGRAD_V2_MOJO_DESIGN.md P2 per-op discipline).
#
# For each kind, on real-ish krea2 shapes:
#   1. hand-chain: forward op + the existing ops/*_backward directly on a
#      deterministic upstream grad (the SAME call the krea2 oracle makes,
#      models/krea2/krea2_block.mojo: repeat_kv_backward :647-648 /
#      sigmoid_backward :602);
#   2. engine: record the op on IDENTICAL tensors -> execute, seed the same grad;
# compare every gradient BIT-EQUAL (to_host F32 exact compare).
#
# OPK_KREA2_PROJ_LORA is NOT gated here: its LoRA grads are HOST lists captured
# out-of-band (they cannot flow through the engine's TArc-only Dict), so it is
# covered by the BLOCK gate (krea2_block_parity.mojo), which exercises it in the
# real chain. F32 throughout (the krea2 parity-gate convention; the math is
# dtype-agnostic for the bit-equality claim, and F32 sidesteps mixed precision).
#
# Build:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#   pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
#     serenitymojo/autograd_v2/tests/krea2_op_parity.mojo -o /tmp/krea2_op_parity
# Run:
#   LD_LIBRARY_PATH=/home/alex/mojodiffusion/.pixi/envs/default/lib /tmp/krea2_op_parity

from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import mul_scalar
from serenitymojo.ops.gqa_backward import repeat_kv_backward
from serenitymojo.ops.activation_backward import sigmoid_backward
from serenitymojo.autograd_v2.node import TArc
from serenitymojo.autograd_v2.graph import Graph
from serenitymojo.autograd_v2.engine import execute
from serenitymojo.autograd_v2.ops_record import (
    record_repeat_kv,
    record_sigmoid,
)

comptime TArcT = ArcPointer[Tensor]


def _f32(var shape: List[Int], seed: UInt64, ctx: DeviceContext) raises -> Tensor:
    return mul_scalar(randn(shape^, seed, STDtype.F32, ctx), Float32(0.5), ctx)


def _leaf(mut g: Graph, var shape: List[Int], seed: UInt64, ctx: DeviceContext) raises -> TArc:
    var t = _f32(shape^, seed, ctx)
    t.set_id(g.fresh_tensor_id())
    _ = g.leaf(t.id)
    return TArc(t^)


def _root_of(g: Graph, y: TArc) raises -> Int:
    return g.node_of_tensor[y[].id]


def _cmp(name: String, got: Tensor, want: Tensor, ctx: DeviceContext) raises -> Bool:
    var hg = got.to_host(ctx)
    var hw = want.to_host(ctx)
    if len(hg) != len(hw):
        print("GATE " + name + " FAIL numel " + String(len(hg)) + " != " + String(len(hw)))
        return False
    var bad = 0
    var nz = 0
    for i in range(len(hg)):
        if hg[i] != hw[i]:
            bad += 1
        if hw[i] != Float32(0.0):
            nz += 1
    var verdict = String("PASS") if (bad == 0 and nz > 0) else String("FAIL")
    print("GATE " + name + " " + verdict + " n_mismatch=" + String(bad) + "/" + String(len(hg)) + " (nonzero_oracle=" + String(nz) + ")")
    return bad == 0 and nz > 0


# ── GATE repeat_kv: x [1,L,KVHEADS,Dh] -> [1,L,HEADS,Dh]; hand =
# repeat_kv_backward (the oracle's :647-648 grouped sum-reduce). ──────────────
def gate_repeat_kv(ctx: DeviceContext) raises -> Bool:
    comptime L = 512
    comptime KVHEADS = 12
    comptime HEADDIM = 128
    comptime N_REP = 4   # HEADS=48 / KVHEADS=12
    comptime HEADS = KVHEADS * N_REP
    var g = Graph()
    var x = _leaf(g, [1, L, KVHEADS, HEADDIM], UInt64(10), ctx)
    var x_id = x[].id
    var gy = _f32([1, L, HEADS, HEADDIM], UInt64(11), ctx)   # upstream grad on dst

    var hand = repeat_kv_backward(gy, L, KVHEADS, N_REP, HEADDIM, ctx)

    var y = record_repeat_kv(g, x, L, KVHEADS, N_REP, HEADDIM, ctx)
    var grads = execute(g, _root_of(g, y), TArcT(Tensor(gy.buf.copy(), gy.shape(), gy.dtype())), ctx)
    return _cmp("repeat_kv", grads[x_id][], hand, ctx)


# ── GATE sigmoid: sg = sigmoid(x); hand = sigmoid_backward(g, x) (oracle :602).
def gate_sigmoid(ctx: DeviceContext) raises -> Bool:
    comptime L = 512
    comptime F = 6144
    var g = Graph()
    var x = _leaf(g, [1, L, F], UInt64(20), ctx)
    var x_id = x[].id
    var gy = _f32([1, L, F], UInt64(21), ctx)

    var hand = sigmoid_backward(gy, x[], ctx)

    var y = record_sigmoid(g, x, ctx)
    var grads = execute(g, _root_of(g, y), TArcT(Tensor(gy.buf.copy(), gy.shape(), gy.dtype())), ctx)
    return _cmp("sigmoid", grads[x_id][], hand, ctx)


def main() raises:
    var ctx = DeviceContext()
    var ok = True
    ok = gate_repeat_kv(ctx) and ok
    ok = gate_sigmoid(ctx) and ok
    if ok:
        print("KREA2 OP PARITY PASS (repeat_kv + sigmoid engine == hand-chain, bit-equal)")
    else:
        raise Error("KREA2 OP PARITY FAIL")

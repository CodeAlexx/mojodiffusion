# tests/lokr_st_parity.mojo — T2.G SimpleTuner-LoKr training-parity gate.
# NEW file (T2.G owns it; the T2.F-2 family gate trio is untouched).
#
# Oracle: /tmp/lokr_st_oracle.safetensors from
#   python3 serenitymojo/training/tests/lokr_st_parity_ref.py
# (pip lycoris_lora 3.4.0 + SimpleTuner peft_init.py semantics).
#
# Gates:
#   T1 factorization table — lokr_factorization(dim,factor) EXACT vs upstream
#      lycoris.functional.factorization for 19 cases incl. factor=-1,
#      non-divisor factors, odd dims, factor>sqrt(dim), klein-9B dims.
#   T2 leg-selection/shape table — new_lokr_adapter's (w1_factored,
#      w2_factored, out_l, out_k, in_m, in_n) EXACT vs REAL LokrModule
#      instances (incl. decompose_both, full_matrix, the odd-max /2 case).
#   T3 carrier algebra — the lokr_stack L1/L2/L3 Kronecker carrier pairs
#      reproduce lokr_forward, and the stack-contract grads (d_a_c, d_b_c)
#      chained through lokr_chain_carrier_grads match the upstream-parity
#      lokr_backward factor grads.
#   T4 perturbed-normal init — lokr_perturbed_normal_init matches the ST
#      helper's deterministic output stats (mean = org_mean*scale exactly,
#      std forced to org_std*scale, norm = f(stats)) vs the torch dump.
#   T5 reduced-dim e2e TRAINING repro — same bf16 init, same (x,t), 3 AdamW
#      steps: Mojo carrier-training (the exact per-step procedure the klein
#      trainer runs, on host) vs torch training through the lycoris wrapper's
#      own forward. Compares per-step losses, final factors, final delta-W.
#      D1 = upstream-default factored; D2 = ST flagship full_matrix +
#      init_lokr_norm.
#
# Build/run:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#   pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     serenitymojo/training/tests/lokr_st_parity.mojo -o /tmp/lokr_st_parity \
#   && /tmp/lokr_st_parity
from std.collections import List
from std.math import sqrt
from std.gpu.host import DeviceContext
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.dtype import STDtype
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.training.lokr_adapter import (
    LoKrAdapter, LoKrGrads, new_lokr_adapter, lokr_factorization,
    lokr_forward, lokr_backward, lokr_delta_weight, lokr_adamw,
    lokr_perturbed_normal_init,
)
from serenitymojo.training.lokr_stack import (
    lokr_carrier_adapter, lokr_carrier_r_eff, lokr_chain_carrier_grads,
)

comptime ORACLE = "/tmp/lokr_st_oracle.safetensors"


def _read_f32(st: SafeTensors, name: String) raises -> List[Float32]:
    var info = st.tensor_info(name)
    if info.dtype != STDtype.F32:
        raise Error(String("expected F32 oracle tensor: ") + name)
    var bytes = st.tensor_bytes(name)
    var fp = bytes.unsafe_ptr().bitcast[Float32]()
    var out = List[Float32]()
    for i in range(info.size // 4):
        out.append(fp[i])
    return out^


def _zeros(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(Float32(0.0))
    return out^


def _randn(n: Int, seed: UInt64, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    for _ in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
        out.append((u - Float32(0.5)) * scale)
    return out^


def _matmul(a: List[Float32], ra: Int, ca: Int, b: List[Float32], rb: Int, cb: Int) raises -> List[Float32]:
    if ca != rb:
        raise Error("gate _matmul: dim mismatch")
    var out = _zeros(ra * cb)
    for i in range(ra):
        for k in range(ca):
            var aik = a[i * ca + k]
            if aik == Float32(0.0):
                continue
            var brow = k * cb
            var orow = i * cb
            for j in range(cb):
                out[orow + j] = out[orow + j] + aik * b[brow + j]
    return out^


def _transpose(a: List[Float32], r: Int, c: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(r * c):
        out.append(Float32(0.0))
    for i in range(r):
        for j in range(c):
            out[j * r + i] = a[i * c + j]
    return out^


def _cos(a: List[Float32], b: List[Float32]) raises -> Float64:
    if len(a) != len(b):
        raise Error("cos: len mismatch")
    var dot = Float64(0.0)
    var na = Float64(0.0)
    var nb = Float64(0.0)
    for i in range(len(a)):
        dot += Float64(a[i]) * Float64(b[i])
        na += Float64(a[i]) * Float64(a[i])
        nb += Float64(b[i]) * Float64(b[i])
    if na == 0.0 and nb == 0.0:
        return 1.0
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (sqrt(na) * sqrt(nb))


def _nrel(a: List[Float32], b: List[Float32]) raises -> Float64:
    var d = Float64(0.0)
    var n = Float64(0.0)
    for i in range(len(a)):
        var dd = Float64(a[i]) - Float64(b[i])
        d += dd * dd
        n += Float64(b[i]) * Float64(b[i])
    if n == 0.0:
        return sqrt(d)
    return sqrt(d / n)


def _bf16_to_f32(v: List[BFloat16]) -> List[Float32]:
    var out = List[Float32]()
    for i in range(len(v)):
        out.append(v[i].cast[DType.float32]())
    return out^


def _relok(a: Float64, b: Float64, tol: Float64) -> Bool:
    var d = a - b
    if d < 0.0:
        d = -d
    var denom = b if b >= 0.0 else -b
    if denom < 1.0e-12:
        return d < 1.0e-9
    return d / denom <= tol


def _check(name: String, ok: Bool) raises:
    if ok:
        print("  PASS:", name)
    else:
        print("  FAIL:", name)
        raise Error(String("GATE FAIL: ") + name)


# Build a LoKrAdapter directly from explicit factor lists.
def _adapter_from_factors(
    var w1: List[Float32], var w1a: List[Float32], var w1b: List[Float32],
    w1_factored: Bool,
    var w2: List[Float32], var w2a: List[Float32], var w2b: List[Float32],
    w2_factored: Bool,
    rank: Int, in_f: Int, out_f: Int,
    out_l: Int, out_k: Int, in_m: Int, in_n: Int, alpha: Float32,
) raises -> LoKrAdapter:
    return LoKrAdapter(
        w1^, w1a^, w1b^, w1_factored, w2^, w2a^, w2b^, w2_factored,
        rank, in_f, out_f, out_l, out_k, in_m, in_n, alpha,
        _zeros(out_l * in_m), _zeros(out_l * in_m),
        _zeros(out_l * rank), _zeros(out_l * rank),
        _zeros(rank * in_m), _zeros(rank * in_m),
        _zeros(out_k * in_n), _zeros(out_k * in_n),
        _zeros(out_k * rank), _zeros(out_k * rank),
        _zeros(rank * in_n), _zeros(rank * in_n),
    )


# Simulate the klein stack contract on host for one carrier adapter:
#   y_delta = (x @ a^T) @ b^T          (carrier scale == 1)
#   d_b = d_y^T @ (x @ a^T)            d_a = (d_y @ b)^T @ x
struct StackSim(Movable):
    var y: List[Float32]
    var d_a: List[Float32]
    var d_b: List[Float32]

    def __init__(out self, var y: List[Float32], var d_a: List[Float32], var d_b: List[Float32]):
        self.y = y^
        self.d_a = d_a^
        self.d_b = d_b^


def _stack_sim(
    car: LoraAdapter, x: List[Float32], d_y: List[Float32], M: Int
) raises -> StackSim:
    var R = car.rank
    var IN = car.in_f
    var OUT = car.out_f
    var a = _bf16_to_f32(car.a)
    var b = _bf16_to_f32(car.b)
    var a_t = _transpose(a, R, IN)                       # [in,R]
    var xa = _matmul(x, M, IN, a_t, IN, R)               # [M,R]
    var b_t = _transpose(b, OUT, R)                      # [R,out]
    var y = _matmul(xa, M, R, b_t, R, OUT)               # [M,out]
    var dy_t = _transpose(d_y, M, OUT)                   # [out,M]
    var d_b = _matmul(dy_t, OUT, M, xa, M, R)            # [out,R]
    var dyb = _matmul(d_y, M, OUT, b, OUT, R)            # [M,R]
    var dyb_t = _transpose(dyb, M, R)                    # [R,M]
    var d_a = _matmul(dyb_t, R, M, x, M, IN)             # [R,in]
    return StackSim(y^, d_a^, d_b^)


# BARS (measured 2026-06-12): the carrier pair is stored BF16 in LoraAdapter —
# the SAME dtype class the klein stack stores its LoRA A/B in — so the carrier
# PRODUCTS take one extra bf16 rounding vs lokr_forward's f32-products-of-
# bf16-factors. That residual measures ~2e-3 nrel with cos 1-2e-6. An
# index-map/chain-rule error shows up as a GROSS mismatch (cos << 0.999), so
# the cos bar carries the structural check; nrel is pinned at the bf16 class.
comptime T3_COS_BAR = 0.99999
comptime T3_NREL_BAR = 8.0e-3


def _gate_carrier_case(
    label: String, lo: LoKrAdapter, M: Int, seed: UInt64
) raises:
    var x = _randn(M * lo.in_f, seed, Float32(1.0))
    var d_y = _randn(M * lo.out_f, seed + 7, Float32(1.0))
    var car = lokr_carrier_adapter(lo)
    var sim = _stack_sim(car, x, d_y, M)
    var y_ref = lokr_forward(x, lo, M)
    var cy = _cos(sim.y, y_ref)
    var ny = _nrel(sim.y, y_ref)
    print("  [", label, "] carrier fwd cos=", cy, " nrel=", ny)
    _check(label + " carrier forward", cy >= T3_COS_BAR and ny <= T3_NREL_BAR)
    var g_ref = lokr_backward(d_y, x, lo, M)
    var g_c = lokr_chain_carrier_grads(lo, sim.d_a, sim.d_b)
    if lo.w1_factored:
        var c1 = _cos(g_c.d_w1a, g_ref.d_w1a)
        var c2 = _cos(g_c.d_w1b, g_ref.d_w1b)
        print("    d_w1a cos=", c1, " nrel=", _nrel(g_c.d_w1a, g_ref.d_w1a))
        print("    d_w1b cos=", c2, " nrel=", _nrel(g_c.d_w1b, g_ref.d_w1b))
        _check(label + " d_w1a", c1 >= T3_COS_BAR and _nrel(g_c.d_w1a, g_ref.d_w1a) <= T3_NREL_BAR)
        _check(label + " d_w1b", c2 >= T3_COS_BAR and _nrel(g_c.d_w1b, g_ref.d_w1b) <= T3_NREL_BAR)
    else:
        var c1 = _cos(g_c.d_w1, g_ref.d_w1)
        print("    d_w1  cos=", c1, " nrel=", _nrel(g_c.d_w1, g_ref.d_w1))
        _check(label + " d_w1", c1 >= T3_COS_BAR and _nrel(g_c.d_w1, g_ref.d_w1) <= T3_NREL_BAR)
    if lo.w2_factored:
        var c3 = _cos(g_c.d_w2a, g_ref.d_w2a)
        var c4 = _cos(g_c.d_w2b, g_ref.d_w2b)
        print("    d_w2a cos=", c3, " nrel=", _nrel(g_c.d_w2a, g_ref.d_w2a))
        print("    d_w2b cos=", c4, " nrel=", _nrel(g_c.d_w2b, g_ref.d_w2b))
        _check(label + " d_w2a", c3 >= T3_COS_BAR and _nrel(g_c.d_w2a, g_ref.d_w2a) <= T3_NREL_BAR)
        _check(label + " d_w2b", c4 >= T3_COS_BAR and _nrel(g_c.d_w2b, g_ref.d_w2b) <= T3_NREL_BAR)
    else:
        var c3 = _cos(g_c.d_w2, g_ref.d_w2)
        print("    d_w2  cos=", c3, " nrel=", _nrel(g_c.d_w2, g_ref.d_w2))
        _check(label + " d_w2", c3 >= T3_COS_BAR and _nrel(g_c.d_w2, g_ref.d_w2) <= T3_NREL_BAR)


# Carrier of a FRESH (zero-leg) adapter must contribute EXACTLY zero (bit):
# the trainer's step-1 base loss must equal the no-adapter baseline.
def _gate_zero_at_init(label: String, lo: LoKrAdapter, M: Int, seed: UInt64) raises:
    var x = _randn(M * lo.in_f, seed, Float32(1.0))
    var car = lokr_carrier_adapter(lo)
    var d_y = _zeros(M * lo.out_f)
    var sim = _stack_sim(car, x, d_y, M)
    for i in range(len(sim.y)):
        if sim.y[i] != Float32(0.0):
            raise Error(String("GATE FAIL: ") + label + " carrier not exactly zero at init")
    print("  PASS:", label, "carrier delta EXACTLY zero at init")


def _has_key(st: SafeTensors, name: String) -> Bool:
    try:
        var _i = st.tensor_info(name)
        return True
    except:
        return False


# Load a train-case adapter from the oracle's {tag}_init_* tensors.
def _adapter_from_oracle(
    st: SafeTensors, tag: String, in_f: Int, out_f: Int, rank: Int,
    factor: Int, alpha: Float32,
) raises -> LoKrAdapter:
    var osplit = lokr_factorization(out_f, factor)
    var isplit = lokr_factorization(in_f, factor)
    var out_l = osplit[0]; var out_k = osplit[1]
    var in_m = isplit[0]; var in_n = isplit[1]
    var w1 = List[Float32]()
    var w1a = List[Float32]()
    var w1b = List[Float32]()
    var w2 = List[Float32]()
    var w2a = List[Float32]()
    var w2b = List[Float32]()
    var w1_fact = _has_key(st, tag + "_init_lokr_w1_a")
    var w2_fact = _has_key(st, tag + "_init_lokr_w2_a")
    if w1_fact:
        w1a = _read_f32(st, tag + "_init_lokr_w1_a")
        w1b = _read_f32(st, tag + "_init_lokr_w1_b")
    else:
        w1 = _read_f32(st, tag + "_init_lokr_w1")
    if w2_fact:
        w2a = _read_f32(st, tag + "_init_lokr_w2_a")
        w2b = _read_f32(st, tag + "_init_lokr_w2_b")
    else:
        w2 = _read_f32(st, tag + "_init_lokr_w2")
    return _adapter_from_factors(
        w1^, w1a^, w1b^, w1_fact, w2^, w2a^, w2b^, w2_fact,
        rank, in_f, out_f, out_l, out_k, in_m, in_n, alpha,
    )


def _gate_train_case(st: SafeTensors, tag: String) raises:
    var meta = _read_f32(st, tag + "_meta")
    var in_f = Int(meta[0]); var out_f = Int(meta[1]); var rank = Int(meta[2])
    var alpha = meta[3]; var factor = Int(meta[4])
    var steps = Int(meta[7]); var lr = meta[8]
    var oracle_scale = meta[9]
    var lo = _adapter_from_oracle(st, tag, in_f, out_f, rank, factor, alpha)
    if Float64(lo.scale) < Float64(oracle_scale) - 1e-6 or Float64(lo.scale) > Float64(oracle_scale) + 1e-6:
        raise Error(String("scale mismatch vs upstream for ") + tag)
    var x = _read_f32(st, tag + "_x")
    var t = _read_f32(st, tag + "_t")
    var base_w = _read_f32(st, tag + "_base_w")     # [out,in]
    var losses_ref = _read_f32(st, tag + "_losses")
    var M = 8
    var n_out = M * out_f
    var base_w_t = _transpose(base_w, out_f, in_f)  # [in,out]
    var y_base = _matmul(x, M, in_f, base_w_t, in_f, out_f)
    var max_loss_diff = Float64(0.0)
    for s in range(steps):
        var car = lokr_carrier_adapter(lo)
        # forward + MSE
        var a = _bf16_to_f32(car.a)
        var b = _bf16_to_f32(car.b)
        var a_t = _transpose(a, car.rank, in_f)
        var xa = _matmul(x, M, in_f, a_t, in_f, car.rank)
        var b_t = _transpose(b, out_f, car.rank)
        var y_d = _matmul(xa, M, car.rank, b_t, car.rank, out_f)
        var loss = Float64(0.0)
        var d_y = _zeros(n_out)
        for i in range(n_out):
            var y = y_base[i] + y_d[i]
            var diff = Float64(y) - Float64(t[i])
            loss += diff * diff
            d_y[i] = Float32(2.0 * diff / Float64(n_out))
        loss /= Float64(n_out)
        var ld = loss - Float64(losses_ref[s])
        if ld < 0.0:
            ld = -ld
        if ld > max_loss_diff:
            max_loss_diff = ld
        print("  [", tag, "] step", s + 1, " loss=", loss, " torch=", losses_ref[s], " |d|=", ld)
        # stack-contract grads + chain + masters AdamW
        var dy_t = _transpose(d_y, M, out_f)
        var d_b = _matmul(dy_t, out_f, M, xa, M, car.rank)
        var dyb = _matmul(d_y, M, out_f, b, out_f, car.rank)
        var dyb_t = _transpose(dyb, M, car.rank)
        var d_a = _matmul(dyb_t, car.rank, M, x, M, in_f)
        var g = lokr_chain_carrier_grads(lo, d_a, d_b)
        lokr_adamw(lo, g, s + 1, lr, Float32(0.9), Float32(0.999), Float32(1.0e-8), Float32(0.01))
    _check(tag + " per-step loss match (<=2e-3)", max_loss_diff <= 2.0e-3)
    # final factors + final delta vs torch
    var worst = Float64(1.0)
    if lo.w1_factored:
        var c = _cos(_bf16_to_f32(lo.w1a), _read_f32(st, tag + "_final_lokr_w1_a"))
        if c < worst: worst = c
        c = _cos(_bf16_to_f32(lo.w1b), _read_f32(st, tag + "_final_lokr_w1_b"))
        if c < worst: worst = c
    else:
        var c = _cos(_bf16_to_f32(lo.w1), _read_f32(st, tag + "_final_lokr_w1"))
        if c < worst: worst = c
    if lo.w2_factored:
        var c = _cos(_bf16_to_f32(lo.w2a), _read_f32(st, tag + "_final_lokr_w2_a"))
        if c < worst: worst = c
        c = _cos(_bf16_to_f32(lo.w2b), _read_f32(st, tag + "_final_lokr_w2_b"))
        if c < worst: worst = c
    else:
        var c = _cos(_bf16_to_f32(lo.w2), _read_f32(st, tag + "_final_lokr_w2"))
        if c < worst: worst = c
    var delta = lokr_delta_weight(lo)
    var delta_ref = _read_f32(st, tag + "_delta_final")
    var cd = _cos(delta, delta_ref)
    print("  [", tag, "] worst final-factor cos=", worst, " delta cos=", cd,
          " delta nrel=", _nrel(delta, delta_ref))
    _check(tag + " final factors cos>=0.999", worst >= 0.999)
    _check(tag + " final delta cos>=0.999", cd >= 0.999)


def main() raises:
    var ctx = DeviceContext()
    _ = ctx
    var st = SafeTensors.open(ORACLE)
    print("=== T2.G lokr_st_parity — SimpleTuner LoKr training-parity gate ===")

    # ── T1: factorization table EXACT ────────────────────────────────────────
    print("[T1] factorization table")
    var ft = _read_f32(st, "fact_table")
    var nrows = len(ft) // 4
    for i in range(nrows):
        var dim = Int(ft[i * 4 + 0])
        var fac = Int(ft[i * 4 + 1])
        var m_ref = Int(ft[i * 4 + 2])
        var n_ref = Int(ft[i * 4 + 3])
        var r = lokr_factorization(dim, fac)
        if r[0] != m_ref or r[1] != n_ref:
            print("  FAIL: factorization(", dim, ",", fac, ") = (", r[0], ",", r[1],
                  ") expected (", m_ref, ",", n_ref, ")")
            raise Error("GATE FAIL: T1 factorization")
        print("  ok factorization(", dim, ",", fac, ") = (", m_ref, ",", n_ref, ")")
    _check("T1 factorization table EXACT (" + String(nrows) + " cases)", True)

    # ── T2: leg-selection + split shapes EXACT vs real LokrModule ────────────
    print("[T2] leg-selection / shape table")
    var ncases = Int(_read_f32(st, "shape_case_count")[0])
    for i in range(ncases):
        var c = _read_f32(st, "shape_case_" + String(i))
        var din = Int(c[0]); var dout = Int(c[1]); var rank = Int(c[2])
        var fac = Int(c[3]); var dec = Int(c[4]) != 0; var full = Int(c[5]) != 0
        var w1_fact_ref = Int(c[6]) != 0
        var w2_fact_ref = Int(c[7]) != 0
        var out_l_ref = Int(c[8]); var out_k_ref = Int(c[9])
        var in_m_ref = Int(c[10]); var in_n_ref = Int(c[11])
        var lo = new_lokr_adapter(din, dout, rank, Float32(1.0), fac, UInt64(5 + i), dec, full)
        var ok = (
            lo.w1_factored == w1_fact_ref and lo.w2_factored == w2_fact_ref
            and lo.out_l == out_l_ref and lo.out_k == out_k_ref
            and lo.in_m == in_m_ref and lo.in_n == in_n_ref
        )
        if not ok:
            print("  FAIL case", i, ": (", din, dout, rank, fac, dec, full, ") mojo legs(",
                  lo.w1_factored, lo.w2_factored, ") split(", lo.out_l, lo.out_k,
                  lo.in_m, lo.in_n, ") expected legs(", w1_fact_ref, w2_fact_ref,
                  ") split(", out_l_ref, out_k_ref, in_m_ref, in_n_ref, ")")
            raise Error("GATE FAIL: T2 shape case")
        print("  ok case", i, " in=", din, " out=", dout, " rank=", rank, " factor=", fac,
              " decomp=", dec, " full=", full, " -> w1_fact=", lo.w1_factored,
              " w2_fact=", lo.w2_factored)
    _check("T2 leg/shape table EXACT (" + String(ncases) + " cases)", True)

    # ── T3: carrier algebra (L1/L2/L3) vs the parity-gated primitive ─────────
    print("[T3] carrier algebra vs lokr_forward/lokr_backward")
    # L2 (upstream default): W1 full + W2 factored, off-zero w2b for a live test
    var lo2 = new_lokr_adapter(64, 48, 2, Float32(1.0), -1, UInt64(21))
    if (not lo2.w2_factored) or lo2.w1_factored:
        raise Error("T3 L2 case did not produce W1-full + W2-factored")
    _gate_zero_at_init(String("L2 fresh"), lo2, 8, UInt64(2001))
    var w2b_live = _randn(lo2.rank * lo2.in_n, UInt64(77), Float32(0.2))
    for i in range(len(lo2.w2b)):
        lo2.w2b[i] = BFloat16(w2b_live[i])
    _gate_carrier_case(String("L2 W1full+W2fact"), lo2, 8, UInt64(1001))
    # L1 (ST full_matrix flagship): both full, off-zero w2
    var lo1 = new_lokr_adapter(24, 16, 4, Float32(1.0), 4, UInt64(22), False, True)
    if lo1.w1_factored or lo1.w2_factored:
        raise Error("T3 L1 case did not produce both-full")
    var w2_live = _randn(lo1.out_k * lo1.in_n, UInt64(78), Float32(0.2))
    for i in range(len(lo1.w2)):
        lo1.w2[i] = BFloat16(w2_live[i])
    _gate_carrier_case(String("L1 both-full"), lo1, 8, UInt64(1002))
    # L3 (decompose_both): both factored, off-zero w2b
    var lo3 = new_lokr_adapter(64, 48, 2, Float32(1.0), -1, UInt64(23), True, False)
    if not (lo3.w1_factored and lo3.w2_factored):
        raise Error("T3 L3 case did not produce both-factored")
    var w2b3 = _randn(lo3.rank * lo3.in_n, UInt64(79), Float32(0.2))
    for i in range(len(lo3.w2b)):
        lo3.w2b[i] = BFloat16(w2b3[i])
    _gate_carrier_case(String("L3 both-fact"), lo3, 8, UInt64(1003))

    # ── T4: perturbed-normal init stats vs the ST helper ─────────────────────
    print("[T4] init_lokr_norm perturbed-normal init")
    var pst = _read_f32(st, "pinit_stats")
    var org = _read_f32(st, "pinit_org")    # [48,64]
    # org stats in F64 (norm/mean/unbiased std) from the dumped org weight
    var n_org = len(org)
    var s_ = Float64(0.0)
    var ss_ = Float64(0.0)
    for i in range(n_org):
        s_ += Float64(org[i])
        ss_ += Float64(org[i]) * Float64(org[i])
    var org_mean = s_ / Float64(n_org)
    var org_var = (ss_ - Float64(n_org) * org_mean * org_mean) / Float64(n_org - 1)
    var org_norm = sqrt(ss_)
    var org_std = sqrt(org_var)
    print("  org stats mojo (norm,mean,std)=", org_norm, org_mean, org_std,
          " torch=", pst[0], pst[1], pst[2])
    var lop = new_lokr_adapter(64, 48, 4, Float32(1.0), -1, UInt64(31), False, True)
    lokr_perturbed_normal_init(lop, org_norm, org_mean, org_std, 1.0e-3, UInt64(4242))
    # w1 all ones
    var ones_ok = True
    for i in range(len(lop.w1)):
        if lop.w1[i].cast[DType.float32]() != Float32(1.0):
            ones_ok = False
    _check("T4 w1 == 1.0 everywhere", ones_ok)
    # w2 stats: mean/std/norm are DETERMINISTIC given (org stats, scale)
    var w2h = _bf16_to_f32(lop.w2)
    var nw = len(w2h)
    var sw = Float64(0.0)
    var ssw = Float64(0.0)
    for i in range(nw):
        sw += Float64(w2h[i])
        ssw += Float64(w2h[i]) * Float64(w2h[i])
    var w2_mean = sw / Float64(nw)
    var w2_std = sqrt((ssw - Float64(nw) * w2_mean * w2_mean) / Float64(nw - 1))
    var w2_norm = sqrt(ssw)
    print("  w2 stats mojo (norm,mean,std)=", w2_norm, w2_mean, w2_std,
          " torch=", pst[3], pst[4], pst[5])
    _check("T4 w2 norm vs torch (<=3%)", _relok(w2_norm, Float64(pst[3]), 0.03))
    _check("T4 w2 std vs torch (<=3%)", _relok(w2_std, Float64(pst[5]), 0.03))
    _check("T4 w2 mean vs torch (<=5% rel or tiny)", _relok(w2_mean, Float64(pst[4]), 0.05))

    # ── T5: reduced-dim e2e training repro ───────────────────────────────────
    print("[T5] e2e training repro (3 AdamW steps, bf16 masters)")
    _gate_train_case(st, String("train1"))
    _gate_train_case(st, String("train2"))

    print("ALL GATES PASS — lokr_st_parity")

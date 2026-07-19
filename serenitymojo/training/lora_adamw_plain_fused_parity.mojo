# training/lora_adamw_plain_fused_parity.mojo — GATE for the fused PLAIN
# AdamW (lora_adamw_plain_fused.mojo) vs the host loop it replaces
# (train_step._lora_adamw → _adamw_host_list), on identical data.
#
# EXPECTATION: identical per-element math; host scalar vs device F32 can differ
# by tiny absolute amounts because the device code may contract/reassociate FMA
# paths. The production invariant is the BF16 parameter writeback plus moment
# drift staying far below any BF16 quantum.
# BARS: params bit-equal except ≤1 bf16 quantum at rate < 1e-4; m/v max absolute
# drift below 1e-9/1e-10 respectively; zero NaN.
#
# Build (GPU):
#   pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#     serenitymojo/training/lora_adamw_plain_fused_parity.mojo -o /tmp/adamw_plain_par
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.sys import has_accelerator
from std.gpu.host import DeviceContext

from serenitymojo.training.train_step import LoraAdapter, LoraGrads, _lora_adamw
from serenitymojo.training.lora_adamw_plain_fused import fused_lora_adamw_plain_step


struct _Lcg(Movable):
    var state: UInt64

    def __init__(out self, seed: UInt64):
        self.state = seed

    def next_f32(mut self) -> Float32:
        self.state = (
            self.state * UInt64(6364136223846793005)
            + UInt64(1442695040888963407)
        )
        var bits = (self.state >> 33) % UInt64(2000000)
        return Float32(Int(bits)) / Float32(1.0e6) - Float32(1.0)


def _rand_list(mut rng: _Lcg, n: Int, amp: Float32) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n):
        o.append(rng.next_f32() * amp)
    return o^


def _abs_list(var x: List[Float32]) -> List[Float32]:
    # second moments must be ≥ 0 (sqrt(v_hat) in the update).
    for i in range(len(x)):
        if x[i] < 0:
            x[i] = -x[i]
    return x^


def _mk_adapter(mut rng: _Lcg, rank: Int, in_f: Int, out_f: Int) -> LoraAdapter:
    return LoraAdapter(
        _rand_list(rng, rank * in_f, 0.02),
        _rand_list(rng, out_f * rank, 0.02),
        rank, in_f, out_f, Float32(1.0) / Float32(rank),
        _rand_list(rng, rank * in_f, 0.001),
        _abs_list(_rand_list(rng, rank * in_f, 0.0001)),
        _rand_list(rng, out_f * rank, 0.001),
        _abs_list(_rand_list(rng, out_f * rank, 0.0001)),
    )


def _ulp_diff_f32(a: Float32, b: Float32) -> Int:
    # reinterpret-as-int distance (monotone for same-sign normal floats).
    var ia = Int(a.to_bits[DType.uint32]())
    var ib = Int(b.to_bits[DType.uint32]())
    var d = ia - ib
    if d < 0:
        d = -d
    return d


def _cmp_f32(name: String, x: List[Float32], y: List[Float32], n_total: Int,
             mut worst: Int, mut mism: Int, mut max_abs: Float32) raises:
    if len(x) != len(y):
        raise Error(name + ": length mismatch")
    for i in range(len(x)):
        if not (x[i] == x[i]) or not (y[i] == y[i]):
            raise Error(name + ": NaN at " + String(i))
        if x[i] != y[i]:
            mism += 1
            var ad = x[i] - y[i]
            if ad < Float32(0.0):
                ad = -ad
            if ad > max_abs:
                max_abs = ad
            var u = _ulp_diff_f32(x[i], y[i])
            if u > worst:
                worst = u


def _cmp_bf16(name: String, x: List[BFloat16], y: List[BFloat16],
              mut worst: Int, mut mism: Int) raises:
    if len(x) != len(y):
        raise Error(name + ": length mismatch")
    for i in range(len(x)):
        var xf = x[i].cast[DType.float32]()
        var yf = y[i].cast[DType.float32]()
        if not (xf == xf) or not (yf == yf):
            raise Error(name + ": NaN at " + String(i))
        if Int(x[i].to_bits[DType.uint16]()) != Int(y[i].to_bits[DType.uint16]()):
            mism += 1
            var d = Int(x[i].to_bits[DType.uint16]()) - Int(y[i].to_bits[DType.uint16]())
            if d < 0:
                d = -d
            if d > worst:
                worst = d


def main() raises:
    comptime if not has_accelerator():
        print("lora_adamw_plain_fused_parity: GPU required")
        raise Error("no accelerator")
    else:
        var ctx = DeviceContext()
        var rng = _Lcg(77)
        # zimage-class adapter shapes (q/k/v/o D=3840, mlp 10240) + odd sizes.
        var shapes_in = [3840, 3840, 10240, 3840, 130]
        var shapes_out = [3840, 10240, 3840, 3840, 70]

        var host_ads = List[LoraAdapter]()
        var fused_ads = List[LoraAdapter]()
        var d_a = List[List[Float32]]()
        var d_b = List[List[Float32]]()
        for s in range(len(shapes_in)):
            var ad = _mk_adapter(rng, 16, shapes_in[s], shapes_out[s])
            host_ads.append(ad.copy())
            fused_ads.append(ad.copy())
            d_a.append(_rand_list(rng, 16 * shapes_in[s], 0.005))
            d_b.append(_rand_list(rng, shapes_out[s] * 16, 0.005))

        var lr = Float32(3.0e-4)
        var beta1 = Float32(0.9)
        var beta2 = Float32(0.999)
        var eps = Float32(1.0e-8)
        var wd = Float32(0.01)

        # 3 optimizer steps with FRESH grads each step (reuse rng stream).
        var total_elems = 0
        for s in range(len(shapes_in)):
            total_elems += 16 * shapes_in[s] + shapes_out[s] * 16
        for t in range(1, 4):
            for i in range(len(host_ads)):
                var lg = LoraGrads(d_a[i].copy(), d_b[i].copy())
                _lora_adamw(host_ads[i], lg, t, lr, ctx, beta1, beta2, eps, wd)
            fused_lora_adamw_plain_step(
                fused_ads, d_a, d_b, 0, len(fused_ads),
                t, lr, beta1, beta2, eps, wd, ctx,
            )
            # next-step grads differ (fresh randoms, same for both sides)
            for i in range(len(d_a)):
                d_a[i] = _rand_list(rng, len(d_a[i]), 0.005)
                d_b[i] = _rand_list(rng, len(d_b[i]), 0.005)

        var p_worst = 0
        var p_mism = 0
        var m_worst = 0
        var m_mism = 0
        var m_max_abs = Float32(0.0)
        var v_worst = 0
        var v_mism = 0
        var v_max_abs = Float32(0.0)
        for i in range(len(host_ads)):
            _cmp_bf16("a", host_ads[i].a, fused_ads[i].a, p_worst, p_mism)
            _cmp_bf16("b", host_ads[i].b, fused_ads[i].b, p_worst, p_mism)
            _cmp_f32("ma", host_ads[i].ma, fused_ads[i].ma, total_elems, m_worst, m_mism, m_max_abs)
            _cmp_f32("va", host_ads[i].va, fused_ads[i].va, total_elems, v_worst, v_mism, v_max_abs)
            _cmp_f32("mb", host_ads[i].mb, fused_ads[i].mb, total_elems, m_worst, m_mism, m_max_abs)
            _cmp_f32("vb", host_ads[i].vb, fused_ads[i].vb, total_elems, v_worst, v_mism, v_max_abs)

        var p_rate = Float64(p_mism) / Float64(total_elems)
        var m_rate = Float64(m_mism) / Float64(total_elems)
        var v_rate = Float64(v_mism) / Float64(total_elems)
        print("params: mismatches=", p_mism, "/", total_elems,
              " rate=", p_rate, " worst_quanta=", p_worst)
        print("first moments: mismatches=", m_mism, "/", total_elems,
              " rate=", m_rate, " worst_ulp=", m_worst, " max_abs=", m_max_abs)
        print("second moments: mismatches=", v_mism, "/", total_elems,
              " rate=", v_rate, " worst_ulp=", v_worst, " max_abs=", v_max_abs)
        if p_worst > 1 or p_rate > 1.0e-4:
            raise Error("params outside ±1-quantum/rate bar")
        if m_max_abs > Float32(1.0e-9) or v_max_abs > Float32(1.0e-10):
            raise Error("moments outside absolute drift bar")
        print("lora_adamw_plain_fused_parity: PASS (3 steps, 5 adapters)")

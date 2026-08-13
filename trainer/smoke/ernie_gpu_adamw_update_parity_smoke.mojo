# ernie_gpu_adamw_update_parity_smoke.mojo — UPDATE-PARITY gate for the
# ERNIE_GPU_ADAMW trainer flag (trainer/train_ernie_real.mojo): old host
# scalar loop (ernie_lora_adamw_step -> _lora_adamw -> _adamw_host_list) vs
# the GPU fused path (fused_lora_adamw_plain_step) on ONE fixed synthetic
# ernie-shaped fixture — identical params, grads, AND pre-seeded nonzero
# moments on both sides; 3 optimizer steps with fresh grads per step.
#
# EXPECTATION (ledger MJ-1017 / training/lora_adamw_plain_fused_parity.mojo):
# per-element math is identical, but device codegen may contract/reassociate
# FMA chains, flipping RNE ties at the ulp level — so the bar is a WORST-CASE
# TOLERANCE, not exact equality:
#   * bf16 params a/b : worst diff <= 1 bf16 quantum (worst abs f32 reported)
#   * F32 moments m/v : worst abs diff <= 1e-6 (measured class is 1e-9/1e-10;
#     1e-6 is the audit's tolerance-class bar)
# Reports the worst abs diff per role either way.
#
# Run (GPU, from trainer):
#   pixi run mojo run -I /home/alex/mojodiffusion -I src \
#       smoke/ernie_gpu_adamw_update_parity_smoke.mojo
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.sys import has_accelerator
from max.gpu.host import DeviceContext

from serenitymojo.models.ernie.ernie_stack_lora import (
    ErnieLoraSet, ErnieLoraGrads, build_ernie_lora_set, ernie_lora_adamw_step,
)
from serenitymojo.training.lora_adamw_plain_fused import fused_lora_adamw_plain_step


# ── deterministic LCG fixture (same generator as the fused parity gate) ──────
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


def _fill_inplace(mut dst: List[Float32], src: List[Float32]) raises:
    if len(dst) != len(src):
        raise Error("_fill_inplace: length mismatch")
    for i in range(len(dst)):
        dst[i] = src[i]


def _abs_list(var x: List[Float32]) -> List[Float32]:
    # second moments must be >= 0 (sqrt(v_hat) in the update).
    for i in range(len(x)):
        if x[i] < 0:
            x[i] = -x[i]
    return x^


def _empty() -> List[Float32]:
    return List[Float32]()


# ── comparison: worst abs diff + worst bf16 quanta ───────────────────────────
def _cmp_bf16(
    name: String, x: List[BFloat16], y: List[BFloat16],
    mut worst_quanta: Int, mut worst_abs: Float32, mut mism: Int,
) raises:
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
            if d > worst_quanta:
                worst_quanta = d
            var ad = xf - yf
            if ad < Float32(0.0):
                ad = -ad
            if ad > worst_abs:
                worst_abs = ad


def _cmp_f32(
    name: String, x: List[Float32], y: List[Float32],
    mut worst_abs: Float32, mut mism: Int,
) raises:
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
            if ad > worst_abs:
                worst_abs = ad


def main() raises:
    comptime if not has_accelerator():
        print("ernie_gpu_adamw_update_parity_smoke: GPU required")
        raise Error("no accelerator")
    else:
        var ctx = DeviceContext()
        var rng = _Lcg(4242)

        # Small ernie-shaped stack: 2 layers x 7 slots (Q,K,V,O,gate,up,down),
        # D=128, F=256, rank=8. build_ernie_lora_set is seed-fixed, so two
        # calls yield IDENTICAL params (A randn, B=0).
        var num_layers = 2
        var host_set = build_ernie_lora_set(num_layers, 128, 256, 8, Float32(8.0))
        var gpu_set = build_ernie_lora_set(num_layers, 128, 256, 8, Float32(8.0))
        var n_ad = len(host_set.ad)

        # Pre-seed NONZERO moments, identical on both sides (v >= 0), so the
        # fixture exercises the beta1*m/beta2*v recurrence, not just m=v=0.
        for i in range(n_ad):
            var ma0 = _rand_list(rng, len(host_set.ad[i].ma), 0.001)
            var va0 = _abs_list(_rand_list(rng, len(host_set.ad[i].va), 0.0001))
            var mb0 = _rand_list(rng, len(host_set.ad[i].mb), 0.001)
            var vb0 = _abs_list(_rand_list(rng, len(host_set.ad[i].vb), 0.0001))
            _fill_inplace(host_set.ad[i].ma, ma0)
            _fill_inplace(gpu_set.ad[i].ma, ma0)
            _fill_inplace(host_set.ad[i].va, va0)
            _fill_inplace(gpu_set.ad[i].va, va0)
            _fill_inplace(host_set.ad[i].mb, mb0)
            _fill_inplace(gpu_set.ad[i].mb, mb0)
            _fill_inplace(host_set.ad[i].vb, vb0)
            _fill_inplace(gpu_set.ad[i].vb, vb0)

        # trainer-class hyperparameters
        var lr = Float32(3.0e-4)
        var beta1 = Float32(0.9)
        var beta2 = Float32(0.999)
        var eps = Float32(1.0e-8)
        var wd = Float32(0.01)

        var total_elems = 0
        for i in range(n_ad):
            total_elems += len(host_set.ad[i].a) + len(host_set.ad[i].b)

        # 3 optimizer steps, FRESH grads per step, byte-identical on both paths.
        for t in range(1, 4):
            var d_a = List[List[Float32]]()
            var d_b = List[List[Float32]]()
            for i in range(n_ad):
                d_a.append(_rand_list(rng, len(host_set.ad[i].a), 0.005))
                d_b.append(_rand_list(rng, len(host_set.ad[i].b), 0.005))

            # OLD path: host scalar loop via the retained ernie carrier step.
            var grads = ErnieLoraGrads(
                d_a.copy(), d_b.copy(),
                _empty(), _empty(), _empty(), _empty(), _empty(), _empty(), 0,
            )
            ernie_lora_adamw_step(
                host_set, grads, t, lr, ctx, beta1, beta2, eps, wd,
            )

            # NEW path: ONE fused GPU launch over the whole flat adapter list
            # (exactly the trainer's ERNIE_GPU_ADAMW call shape).
            fused_lora_adamw_plain_step(
                gpu_set.ad, d_a, d_b, 0, n_ad,
                t, lr, beta1, beta2, eps, wd, ctx,
            )

        # ── compare params + moments, worst-case-tolerance bars ─────────────
        var p_quanta = 0
        var p_abs = Float32(0.0)
        var p_mism = 0
        var m_abs = Float32(0.0)
        var m_mism = 0
        var v_abs = Float32(0.0)
        var v_mism = 0
        for i in range(n_ad):
            _cmp_bf16("a", host_set.ad[i].a, gpu_set.ad[i].a, p_quanta, p_abs, p_mism)
            _cmp_bf16("b", host_set.ad[i].b, gpu_set.ad[i].b, p_quanta, p_abs, p_mism)
            _cmp_f32("ma", host_set.ad[i].ma, gpu_set.ad[i].ma, m_abs, m_mism)
            _cmp_f32("mb", host_set.ad[i].mb, gpu_set.ad[i].mb, m_abs, m_mism)
            _cmp_f32("va", host_set.ad[i].va, gpu_set.ad[i].va, v_abs, v_mism)
            _cmp_f32("vb", host_set.ad[i].vb, gpu_set.ad[i].vb, v_abs, v_mism)

        print("params : mismatches=", p_mism, "/", total_elems,
              " worst_quanta=", p_quanta, " worst_abs=", p_abs)
        print("moment m: mismatches=", m_mism, "/", total_elems,
              " worst_abs=", m_abs)
        print("moment v: mismatches=", v_mism, "/", total_elems,
              " worst_abs=", v_abs)

        if p_quanta > 1:
            raise Error("params outside 1-bf16-quantum worst-case bar")
        if m_abs > Float32(1.0e-6) or v_abs > Float32(1.0e-6):
            raise Error("moments outside 1e-6 worst-abs bar")
        print(
            "ernie_gpu_adamw_update_parity_smoke: PASS (3 steps, ",
            n_ad, " adapters, ", total_elems, " elements)",
        )

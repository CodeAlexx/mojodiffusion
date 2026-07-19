# models/sd35/parity/sd35_gpu_adamw_update_parity_smoke.mojo — GATE for the
# SD35_GPU_ADAMW fused optimizer path (sd35_lora_adamw_step → ONE
# fused_lora_adamw_plain_step launch) vs the retained host scalar loop
# (sd35_lora_adamw_step_unfused → _lora_adamw → _adamw_host_list) on an
# identical fixed synthetic fixture (same params/grads/moments both sides).
#
# EXPECTATION (MJ-1017 / lora_adamw_plain_fused.mojo header): identical
# per-element math, but device codegen FMA contraction/reassociation can flip
# RNE ties vs the host F32 chain at the ulp level — so the gate is a
# worst-abs-diff bar per element, NOT exact equality:
#   params (bf16 writeback): worst quanta ≤ 1 AND worst |Δ| ≤ 2.5e-4
#                            (one bf16 quantum at the fixture's ~0.02 amp)
#   moments (F32 chain):     worst |Δ| ≤ 1e-6 (measured class is ≤1e-9;
#                            1e-6 is the campaign worst-case bar)
#   zero NaN anywhere.
#
# Build (GPU, from /home/alex/mojodiffusion; NEVER -O3):
#   rm -f serenitymojo.mojopkg
#   MEM_MAX=30G MEM_HIGH=26G SWAP_MAX=2G pixi run bash scripts/mem_safe.sh \
#     mojo build --optimization-level 2 --num-threads 4 -I . -I /home/alex/MOJO-libs \
#     -Xlinker -lm -Xlinker -Lserenitymojo/ops/cshim/lib \
#     -Xlinker -lserenity_cudnn_sdpa -Xlinker -rpath -Xlinker \
#     /home/alex/mojodiffusion/serenitymojo/ops/cshim/lib \
#     serenitymojo/models/sd35/parity/sd35_gpu_adamw_update_parity_smoke.mojo \
#     -o output/bin/sd35_gpu_adamw_update_parity_smoke
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.sys import has_accelerator
from std.gpu.host import DeviceContext

from serenitymojo.models.sd35.sd35_stack_lora import (
    SD35LoraSet,
    SD35LoraGradSet,
    SD35_GPU_ADAMW,
    sd35_lora_adamw_step,
    sd35_lora_adamw_step_unfused,
)
from serenitymojo.training.train_step import LoraAdapter


# sd35-shaped fixture dims (slot layout of build_sd35_lora_set at toy scale).
comptime DEPTH = 4
comptime D = 256
comptime MLP = 768
comptime RANK = 8


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
    # nonzero params AND nonzero warm moments: a step-0 B=0/m=0 fixture would
    # under-exercise the m_hat/v_hat chain the two paths must agree on.
    return LoraAdapter(
        _rand_list(rng, rank * in_f, 0.02),
        _rand_list(rng, out_f * rank, 0.02),
        rank, in_f, out_f, Float32(1.0) / Float32(rank),
        _rand_list(rng, rank * in_f, 0.001),
        _abs_list(_rand_list(rng, rank * in_f, 0.0001)),
        _rand_list(rng, out_f * rank, 0.001),
        _abs_list(_rand_list(rng, out_f * rank, 0.0001)),
    )


def _mk_stream(mut rng: _Lcg, mut ad: List[LoraAdapter]):
    # one stream = qkv, proj, fc1, fc2 (sd35 slot order, ctx then x per block).
    ad.append(_mk_adapter(rng, RANK, D, 3 * D))
    ad.append(_mk_adapter(rng, RANK, D, D))
    ad.append(_mk_adapter(rng, RANK, D, MLP))
    ad.append(_mk_adapter(rng, RANK, MLP, D))


def _ulp_diff_f32(a: Float32, b: Float32) -> Int:
    # reinterpret-as-int distance (monotone for same-sign normal floats).
    var ia = Int(a.to_bits[DType.uint32]())
    var ib = Int(b.to_bits[DType.uint32]())
    var d = ia - ib
    if d < 0:
        d = -d
    return d


def _cmp_f32(name: String, x: List[Float32], y: List[Float32],
             mut worst_ulp: Int, mut mism: Int, mut max_abs: Float32) raises:
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
            if u > worst_ulp:
                worst_ulp = u


def _cmp_bf16(name: String, x: List[BFloat16], y: List[BFloat16],
              mut worst_quanta: Int, mut mism: Int, mut max_abs: Float32) raises:
    if len(x) != len(y):
        raise Error(name + ": length mismatch")
    for i in range(len(x)):
        var xf = x[i].cast[DType.float32]()
        var yf = y[i].cast[DType.float32]()
        if not (xf == xf) or not (yf == yf):
            raise Error(name + ": NaN at " + String(i))
        if Int(x[i].to_bits[DType.uint16]()) != Int(y[i].to_bits[DType.uint16]()):
            mism += 1
            var ad = xf - yf
            if ad < Float32(0.0):
                ad = -ad
            if ad > max_abs:
                max_abs = ad
            var d = Int(x[i].to_bits[DType.uint16]()) - Int(y[i].to_bits[DType.uint16]())
            if d < 0:
                d = -d
            if d > worst_quanta:
                worst_quanta = d


def main() raises:
    comptime if not SD35_GPU_ADAMW:
        print("sd35_gpu_adamw_update_parity_smoke: SD35_GPU_ADAMW is False —")
        print("sd35_lora_adamw_step would run the host loop and this gate")
        print("would compare host vs host (vacuous). Flip the flag ON first.")
        raise Error("SD35_GPU_ADAMW off")
    else:
        comptime if not has_accelerator():
            print("sd35_gpu_adamw_update_parity_smoke: GPU required")
            raise Error("no accelerator")
        else:
            var ctx = DeviceContext()
            var rng = _Lcg(20260701)

            # identical fixed fixture on both sides (sd35 slot shapes).
            var ad = List[LoraAdapter]()
            for _ in range(DEPTH):
                _mk_stream(rng, ad)   # ctx stream
                _mk_stream(rng, ad)   # x stream
            var host_set = SD35LoraSet(ad.copy(), DEPTH, RANK)
            var gpu_set = SD35LoraSet(ad^, DEPTH, RANK)

            var total_elems = 0
            for i in range(len(host_set.ad)):
                total_elems += len(host_set.ad[i].a) + len(host_set.ad[i].b)

            var lr = Float32(3.0e-4)
            var beta1 = Float32(0.9)
            var beta2 = Float32(0.999)
            var eps = Float32(1.0e-8)
            var wd = Float32(0.01)

            # 3 optimizer steps, FRESH grads each step (same lists both sides).
            for t in range(1, 4):
                var d_a = List[List[Float32]]()
                var d_b = List[List[Float32]]()
                for i in range(len(host_set.ad)):
                    d_a.append(_rand_list(rng, len(host_set.ad[i].ma), 0.005))
                    d_b.append(_rand_list(rng, len(host_set.ad[i].mb), 0.005))
                var grads = SD35LoraGradSet(d_a^, d_b^, 0)
                # OLD path: retained host scalar loop.
                sd35_lora_adamw_step_unfused(
                    host_set, grads, t, lr, ctx, beta1, beta2, eps, wd,
                )
                # NEW path: SD35_GPU_ADAMW=True → ONE fused GPU launch.
                sd35_lora_adamw_step(
                    gpu_set, grads, t, lr, ctx, beta1, beta2, eps, wd,
                )

            var p_worst_q = 0
            var p_mism = 0
            var p_max_abs = Float32(0.0)
            var m_worst_ulp = 0
            var m_mism = 0
            var m_max_abs = Float32(0.0)
            var v_worst_ulp = 0
            var v_mism = 0
            var v_max_abs = Float32(0.0)
            for i in range(len(host_set.ad)):
                _cmp_bf16("a", host_set.ad[i].a, gpu_set.ad[i].a, p_worst_q, p_mism, p_max_abs)
                _cmp_bf16("b", host_set.ad[i].b, gpu_set.ad[i].b, p_worst_q, p_mism, p_max_abs)
                _cmp_f32("ma", host_set.ad[i].ma, gpu_set.ad[i].ma, m_worst_ulp, m_mism, m_max_abs)
                _cmp_f32("va", host_set.ad[i].va, gpu_set.ad[i].va, v_worst_ulp, v_mism, v_max_abs)
                _cmp_f32("mb", host_set.ad[i].mb, gpu_set.ad[i].mb, m_worst_ulp, m_mism, m_max_abs)
                _cmp_f32("vb", host_set.ad[i].vb, gpu_set.ad[i].vb, v_worst_ulp, v_mism, v_max_abs)

            print("params: mismatches=", p_mism, "/", total_elems,
                  " worst_quanta=", p_worst_q, " worst_abs=", p_max_abs)
            print("first moments: mismatches=", m_mism, "/", total_elems,
                  " worst_ulp=", m_worst_ulp, " worst_abs=", m_max_abs)
            print("second moments: mismatches=", v_mism, "/", total_elems,
                  " worst_ulp=", v_worst_ulp, " worst_abs=", v_max_abs)
            # worst-abs-diff bars (MJ-1017 tolerance class, NOT exact equality).
            if p_worst_q > 1:
                raise Error("params beyond ±1 bf16 quantum")
            if p_max_abs > Float32(2.5e-4):
                raise Error("params worst abs diff beyond 2.5e-4 bar")
            if m_max_abs > Float32(1.0e-6):
                raise Error("first moments worst abs diff beyond 1e-6 bar")
            if v_max_abs > Float32(1.0e-6):
                raise Error("second moments worst abs diff beyond 1e-6 bar")
            print("sd35_gpu_adamw_update_parity_smoke: PASS (3 steps, ",
                  len(host_set.ad), " adapters, ", total_elems, " elems)")

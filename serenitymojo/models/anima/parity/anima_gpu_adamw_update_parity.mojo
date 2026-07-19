# serenitymojo/models/anima/parity/anima_gpu_adamw_update_parity.mojo
#
# UPDATE-PARITY GATE for the anima GPU AdamW routing (ANIMA_GPU_ADAMW=True in
# anima_stack_lora.mojo): the OLD host scalar loop (anima_lora_adamw_step_unfused
# → _lora_adamw → _adamw_host_list) vs the NEW GPU path (anima_lora_adamw_step →
# fused_lora_adamw_plain_step) on a FIXED synthetic fixture — identical params /
# grads / moments on both sides, 3 optimizer steps with fresh grads each step.
#
# EXPECTATION (ledger MJ-1017 / lora_adamw_plain_fused.mojo header): identical
# per-element math, but device FMA contraction/reassociation can flip RNE ties
# by 1 ulp vs the host F32 chain — so the bar is a WORST-CASE TOLERANCE, NOT
# exact equality. BARS: bf16 params within ±1 bf16 quantum per element (worst
# abs diff reported); F32 moments worst abs diff per element <= 1e-6 (reported);
# zero NaN.
#
# Build (GPU, from /home/alex/mojodiffusion; -O2, NEVER -O3):
#   rm -f serenitymojo.mojopkg
#   MEM_MAX=30G MEM_HIGH=26G SWAP_MAX=2G pixi run bash scripts/mem_safe.sh \
#     mojo build --optimization-level 2 --num-threads 4 -I . -I /home/alex/MOJO-libs \
#     -Xlinker -lm -Xlinker -Lserenitymojo/ops/cshim/lib \
#     -Xlinker -lserenity_cudnn_sdpa \
#     -Xlinker -rpath -Xlinker /home/alex/mojodiffusion/serenitymojo/ops/cshim/lib \
#     serenitymojo/models/anima/parity/anima_gpu_adamw_update_parity.mojo \
#     -o output/bin/anima_gpu_adamw_update_parity
# Run:
#   LD_LIBRARY_PATH=.pixi/envs/default/lib:serenitymojo/ops/cshim/lib:$HOME/.local/lib/python3.12/site-packages/nvidia/cudnn/lib \
#     output/bin/anima_gpu_adamw_update_parity
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.sys import has_accelerator
from std.gpu.host import DeviceContext
from std.collections import List

from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.models.anima.lora_block import ANIMA_SLOTS
from serenitymojo.models.anima.anima_stack_lora import (
    AnimaLoraSet, AnimaLoraGrads, ANIMA_GPU_ADAMW,
    anima_lora_adamw_step, anima_lora_adamw_step_unfused,
)


comptime NUM_BLOCKS = 2
comptime RANK = 4
# small stand-ins for the anima per-block projection shapes (D=2048, JOINT=1024,
# F=8192 in production) — the optimizer math is shape-agnostic; small dims keep
# the gate cheap while still covering the ca_k/ca_v (JOINT-in) and mlp (F) cases.
comptime D_S = 64
comptime JOINT_S = 48
comptime F_S = 128


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
    # second moments must be >= 0 (sqrt(v_hat) in the update).
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


# per-block slot in/out shapes, flat slot order (anima_stack_lora.mojo carrier):
#   {sa_q, sa_k, sa_v, sa_out, ca_q, ca_k, ca_v, ca_out, mlp1, mlp2}
def _slot_in(slot: Int) -> Int:
    if slot == 5 or slot == 6:   # ca_k / ca_v take the frozen JOINT context
        return JOINT_S
    elif slot == 9:              # mlp2
        return F_S
    return D_S


def _slot_out(slot: Int) -> Int:
    if slot == 8:                # mlp1
        return F_S
    return D_S


def _empty() -> List[Float32]:
    return List[Float32]()


def _mk_grads(
    d_a: List[List[Float32]], d_b: List[List[Float32]]
) -> AnimaLoraGrads:
    return AnimaLoraGrads(
        d_a.copy(), d_b.copy(),
        _empty(), _empty(), _empty(),
        _empty(), _empty(), _empty(), _empty(),
        0,
    )


def _cmp_bf16(
    name: String, x: List[BFloat16], y: List[BFloat16],
    mut worst_quanta: Int, mut mism: Int, mut worst_abs: Float32,
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
    mut mism: Int, mut worst_abs: Float32,
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
        print("anima_gpu_adamw_update_parity: GPU required")
        raise Error("no accelerator")
    else:
        comptime if not ANIMA_GPU_ADAMW:
            # with the flag off anima_lora_adamw_step IS the host loop and this
            # gate would compare the host loop with itself — meaningless PASS.
            print("anima_gpu_adamw_update_parity: ANIMA_GPU_ADAMW is False —")
            print("both paths are the host loop; flip the flag to gate the GPU path")
            raise Error("ANIMA_GPU_ADAMW off: gate is vacuous")
        else:
            var ctx = DeviceContext()
            var rng = _Lcg(1017)  # fixed seed = fixed fixture (ledger MJ-1017)
            var n_adapters = NUM_BLOCKS * ANIMA_SLOTS

            # identical fixture on both sides: same params, same moments.
            var host_ads = List[LoraAdapter]()
            var gpu_ads = List[LoraAdapter]()
            var d_a = List[List[Float32]]()
            var d_b = List[List[Float32]]()
            var total_elems = 0
            for i in range(n_adapters):
                var slot = i % ANIMA_SLOTS
                var in_f = _slot_in(slot)
                var out_f = _slot_out(slot)
                var ad = _mk_adapter(rng, RANK, in_f, out_f)
                host_ads.append(ad.copy())
                gpu_ads.append(ad.copy())
                d_a.append(_rand_list(rng, RANK * in_f, 0.005))
                d_b.append(_rand_list(rng, out_f * RANK, 0.005))
                total_elems += RANK * in_f + out_f * RANK
            var host_set = AnimaLoraSet(host_ads^, NUM_BLOCKS, RANK)
            var gpu_set = AnimaLoraSet(gpu_ads^, NUM_BLOCKS, RANK)

            var lr = Float32(3.0e-4)
            var beta1 = Float32(0.9)
            var beta2 = Float32(0.999)
            var eps = Float32(1.0e-8)
            var wd = Float32(0.01)

            # 3 optimizer steps, FRESH grads each step (same grads both sides).
            for t in range(1, 4):
                var g = _mk_grads(d_a, d_b)
                anima_lora_adamw_step_unfused(
                    host_set, g, t, lr, ctx, beta1, beta2, eps, wd,
                )
                anima_lora_adamw_step(
                    gpu_set, g, t, lr, ctx, beta1, beta2, eps, wd,
                )
                for i in range(len(d_a)):
                    d_a[i] = _rand_list(rng, len(d_a[i]), 0.005)
                    d_b[i] = _rand_list(rng, len(d_b[i]), 0.005)

            var p_quanta = 0
            var p_mism = 0
            var p_abs = Float32(0.0)
            var m_mism = 0
            var m_abs = Float32(0.0)
            var v_mism = 0
            var v_abs = Float32(0.0)
            for i in range(n_adapters):
                _cmp_bf16("a", host_set.ad[i].a, gpu_set.ad[i].a, p_quanta, p_mism, p_abs)
                _cmp_bf16("b", host_set.ad[i].b, gpu_set.ad[i].b, p_quanta, p_mism, p_abs)
                _cmp_f32("ma", host_set.ad[i].ma, gpu_set.ad[i].ma, m_mism, m_abs)
                _cmp_f32("mb", host_set.ad[i].mb, gpu_set.ad[i].mb, m_mism, m_abs)
                _cmp_f32("va", host_set.ad[i].va, gpu_set.ad[i].va, v_mism, v_abs)
                _cmp_f32("vb", host_set.ad[i].vb, gpu_set.ad[i].vb, v_mism, v_abs)

            print("params:         mismatches=", p_mism, "/", total_elems,
                  " worst_abs_diff=", p_abs, " worst_bf16_quanta=", p_quanta)
            print("first moments:  mismatches=", m_mism, "/", total_elems,
                  " worst_abs_diff=", m_abs)
            print("second moments: mismatches=", v_mism, "/", total_elems,
                  " worst_abs_diff=", v_abs)

            # worst-case tolerance bars (NOT exact-equality — GPU FMA vs host
            # F32 differs at ulp level, MJ-1017 class).
            if p_quanta > 1:
                raise Error("params outside ±1 bf16-quantum worst-case bar")
            if m_abs > Float32(1.0e-6):
                raise Error("first moments worst abs diff > 1e-6")
            if v_abs > Float32(1.0e-6):
                raise Error("second moments worst abs diff > 1e-6")
            print(
                "anima_gpu_adamw_update_parity: PASS (3 steps, ",
                n_adapters, " adapters, host loop vs ANIMA_GPU_ADAMW fused path)",
            )

# serenitymojo/models/ltx2/parity/ltx2_gpu_adamw_update_parity.mojo
#
# UPDATE-PARITY gate for the LTX-2 GPU AdamW path (LTX2_GPU_ADAMW, campaign
# TRAINING_SPEED_AUDIT_2026-07-01 Finding 3): the fused-GPU optimizer entry
#   ltx2_lora_adamw_step            (-> fused_lora_adamw_plain_step, ONE launch)
# vs the retained host scalar loop
#   ltx2_lora_adamw_step_unfused    (-> _lora_adamw -> _adamw_host_list)
# on an IDENTICAL synthetic fixture: same bf16 params, same F32 grads, same
# nonzero F32 moments (mid-run state), production-class LTX-2 adapter shapes
# (D=4096, rank=16, 4 slots/block), 3 optimizer steps with fresh grads.
#
# EXPECTATION (ledger MJ-1017 / training/lora_adamw_plain_fused_parity.mojo):
# per-element math is identical, but device codegen may contract/reassociate
# FMA paths, flipping RNE ties at the ulp level. So the bar is WORST-CASE
# TOLERANCE, NOT exact equality:
#   - bf16 params a/b: bit-equal except <=1 bf16 quantum, mismatch rate < 1e-4
#     (worst abs diff reported).
#   - F32 moments m/v: worst abs diff per element <= 1e-6 (the ~1e-6 tolerance
#     class; the sibling training-level gate measured < 1e-9 in practice).
#   - zero NaN anywhere.
#
# Build (GPU; -O2 per repo rule, from /home/alex/mojodiffusion):
#   rm -f serenitymojo.mojopkg
#   MEM_MAX=30G MEM_HIGH=26G SWAP_MAX=2G pixi run bash scripts/mem_safe.sh \
#     mojo build --optimization-level 2 --num-threads 4 -I . -I /home/alex/MOJO-libs \
#     -Xlinker -lm -Xlinker -Lserenitymojo/ops/cshim/lib \
#     -Xlinker -lserenity_cudnn_sdpa -Xlinker -rpath -Xlinker \
#     /home/alex/mojodiffusion/serenitymojo/ops/cshim/lib \
#     serenitymojo/models/ltx2/parity/ltx2_gpu_adamw_update_parity.mojo \
#     -o output/bin/ltx2_gpu_adamw_update_parity
# Run:
#   LD_LIBRARY_PATH=.pixi/envs/default/lib:serenitymojo/ops/cshim/lib:$HOME/.local/lib/python3.12/site-packages/nvidia/cudnn/lib \
#     output/bin/ltx2_gpu_adamw_update_parity
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.sys import has_accelerator
from std.gpu.host import DeviceContext

from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.models.ltx2.ltx2_stack_lora import (
    LTX2_GPU_ADAMW,
    LTX2LoraGradSet,
    LTX2LoraSet,
    ltx2_lora_adamw_step,
    ltx2_lora_adamw_step_unfused,
    total_ltx2_adapters,
)

# production-class fixture: LTX-2 D=4096, rank=16, 4 slots/block, 2 blocks.
comptime FIX_D = 4096
comptime FIX_RANK = 16
comptime FIX_LAYERS = 2
comptime FIX_LR = Float32(3.0e-4)
comptime FIX_STEPS = 3


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


def _mk_adapter(mut rng: _Lcg) -> LoraAdapter:
    # mid-run state: nonzero A/B AND nonzero moments (NOT the B=0/zero-moment
    # init, which would under-exercise the m/v recurrence).
    return LoraAdapter(
        _rand_list(rng, FIX_RANK * FIX_D, 0.02),
        _rand_list(rng, FIX_D * FIX_RANK, 0.02),
        FIX_RANK, FIX_D, FIX_D, Float32(1.0) / Float32(FIX_RANK),
        _rand_list(rng, FIX_RANK * FIX_D, 0.001),
        _abs_list(_rand_list(rng, FIX_RANK * FIX_D, 0.0001)),
        _rand_list(rng, FIX_D * FIX_RANK, 0.001),
        _abs_list(_rand_list(rng, FIX_D * FIX_RANK, 0.0001)),
    )


def _mk_gradset(d_a: List[List[Float32]], d_b: List[List[Float32]]) -> LTX2LoraGradSet:
    var da = List[List[Float32]]()
    var db = List[List[Float32]]()
    for i in range(len(d_a)):
        da.append(d_a[i].copy())
        db.append(d_b[i].copy())
    return LTX2LoraGradSet(da^, db^, List[Float32](), 0)


def _cmp_f32(name: String, x: List[Float32], y: List[Float32],
             mut mism: Int, mut max_abs: Float32) raises:
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
            var d = Int(x[i].to_bits[DType.uint16]()) - Int(y[i].to_bits[DType.uint16]())
            if d < 0:
                d = -d
            if d > worst_quanta:
                worst_quanta = d
            var ad = xf - yf
            if ad < Float32(0.0):
                ad = -ad
            if ad > max_abs:
                max_abs = ad


def main() raises:
    comptime if not has_accelerator():
        print("ltx2_gpu_adamw_update_parity: GPU required")
        raise Error("no accelerator")
    else:
        # Vacuous-compare guard: with the flag off, ltx2_lora_adamw_step IS the
        # host loop and this gate would compare host vs host. Fail loudly.
        comptime if not LTX2_GPU_ADAMW:
            raise Error(
                "ltx2_gpu_adamw_update_parity: LTX2_GPU_ADAMW is False - gate is"
                + " meaningless (both sides would run the host loop)"
            )
        else:
            var ctx = DeviceContext()
            var rng = _Lcg(20260701)

            var n_adapters = FIX_LAYERS * 4  # LTX2_SLOTS
            var host_ads = List[LoraAdapter]()
            var gpu_ads = List[LoraAdapter]()
            var d_a = List[List[Float32]]()
            var d_b = List[List[Float32]]()
            for _ in range(n_adapters):
                var ad = _mk_adapter(rng)
                host_ads.append(ad.copy())
                gpu_ads.append(ad.copy())
                d_a.append(_rand_list(rng, FIX_RANK * FIX_D, 0.005))
                d_b.append(_rand_list(rng, FIX_D * FIX_RANK, 0.005))

            var host_set = LTX2LoraSet(host_ads^, FIX_LAYERS, FIX_RANK)
            var gpu_set = LTX2LoraSet(gpu_ads^, FIX_LAYERS, FIX_RANK)
            var total_elems = n_adapters * 2 * FIX_RANK * FIX_D

            # FIX_STEPS optimizer steps, FRESH grads each step (same fixture on
            # both sides), through the PRODUCT entry points.
            for t in range(1, FIX_STEPS + 1):
                var g_host = _mk_gradset(d_a, d_b)
                ltx2_lora_adamw_step_unfused(host_set, g_host, t, FIX_LR, ctx)
                var g_gpu = _mk_gradset(d_a, d_b)
                ltx2_lora_adamw_step(gpu_set, g_gpu, t, FIX_LR, ctx)
                for i in range(len(d_a)):
                    d_a[i] = _rand_list(rng, len(d_a[i]), 0.005)
                    d_b[i] = _rand_list(rng, len(d_b[i]), 0.005)

            var p_worst_q = 0
            var p_mism = 0
            var p_max_abs = Float32(0.0)
            var m_mism = 0
            var m_max_abs = Float32(0.0)
            var v_mism = 0
            var v_max_abs = Float32(0.0)
            for i in range(total_ltx2_adapters(host_set)):
                _cmp_bf16("a", host_set.ad[i].a, gpu_set.ad[i].a, p_worst_q, p_mism, p_max_abs)
                _cmp_bf16("b", host_set.ad[i].b, gpu_set.ad[i].b, p_worst_q, p_mism, p_max_abs)
                _cmp_f32("ma", host_set.ad[i].ma, gpu_set.ad[i].ma, m_mism, m_max_abs)
                _cmp_f32("mb", host_set.ad[i].mb, gpu_set.ad[i].mb, m_mism, m_max_abs)
                _cmp_f32("va", host_set.ad[i].va, gpu_set.ad[i].va, v_mism, v_max_abs)
                _cmp_f32("vb", host_set.ad[i].vb, gpu_set.ad[i].vb, v_mism, v_max_abs)

            var p_rate = Float64(p_mism) / Float64(total_elems)
            print("params: mismatches=", p_mism, "/", total_elems,
                  " rate=", p_rate, " worst_quanta=", p_worst_q,
                  " worst_abs=", p_max_abs)
            print("first moments: mismatches=", m_mism, "/", total_elems,
                  " worst_abs=", m_max_abs)
            print("second moments: mismatches=", v_mism, "/", total_elems,
                  " worst_abs=", v_max_abs)
            if p_worst_q > 1 or p_rate > 1.0e-4:
                raise Error("params outside ±1-bf16-quantum / 1e-4-rate bar")
            if m_max_abs > Float32(1.0e-6) or v_max_abs > Float32(1.0e-6):
                raise Error("moments outside 1e-6 worst-abs bar")
            print("ltx2_gpu_adamw_update_parity: PASS (", FIX_STEPS,
                  " steps, ", n_adapters, " adapters, D=", FIX_D,
                  " rank=", FIX_RANK, ")")

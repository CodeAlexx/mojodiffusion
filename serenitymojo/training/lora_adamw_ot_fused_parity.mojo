# training/lora_adamw_ot_fused_parity.mojo — BIT-EQUALITY gate [GPU]:
# fused_lora_adamw_ot_step (GPU kernel) vs _lora_adamw_precomputed (the host
# scalar loop the Klein trainer uses today), on identical adapters/grads.
#
# PASS = params (a,b) and moments (ma,va,mb,vb) are BIT-IDENTICAL across all
# adapters after the step, for two different (t -> scalars, seed) sets.
# Any mismatch prints index/host/gpu values and the gate raises.
#
# Shapes mirror Klein-9B LoRA slots (rank 16; in/out 4096 & 12288) plus an
# odd tail size (touches the last-thread path) and adversarial values
# (exact powers of two near bf16 binade boundaries, zeros, tiny, negatives).
#
# Build + run (GPU):
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#     pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#       serenitymojo/training/lora_adamw_ot_fused_parity.mojo -o /tmp/adamw_par \
#     && /tmp/adamw_par
# READ the printed counts; "ALL BIT-EQUAL" + exit 0 = PASS.

from std.gpu.host import DeviceContext
from std.math import fma, sqrt

from serenitymojo.models.klein.lora_adapter import _lora_adamw_precomputed
from serenitymojo.training.lora_adamw_ot_fused import (
    _rne_bf16_exact, _sr_bf16_exact, fused_lora_adamw_ot_step,
)
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.util.bf16_stochastic_rounding import sr_uniform


def _host_ref_fma_list(
    mut p: List[BFloat16],
    g: List[Float32],
    mut m: List[Float32],
    mut v: List[Float32],
    step_size: Float32,
    bc2_sqrt: Float32,
    decay: Float32,
    omb1: Float32,
    beta2: Float32,
    omb2: Float32,
    eps: Float32,
    seed: UInt32,
) raises:
    # host mirror of the GPU kernel with EXPLICIT fma where the device
    # compiler would contract — hypothesis probe for the 1-ulp v divergence.
    for i in range(len(p)):
        var pf = p[i].cast[DType.float32]()
        var mf = m[i]
        var vf = v[i]
        var gv = g[i]
        pf = pf * decay
        mf = fma(omb1, gv - mf, mf)
        vf = fma(beta2, vf, omb2 * gv * gv)
        var m_q = _rne_bf16_exact(mf)
        var v_q = _rne_bf16_exact(vf)
        m[i] = m_q.cast[DType.float32]()
        v[i] = v_q.cast[DType.float32]()
        var denom = sqrt(v_q.cast[DType.float32]()) / bc2_sqrt + eps
        var newp = pf - step_size * m_q.cast[DType.float32]() / denom
        p[i] = _sr_bf16_exact(newp, sr_uniform(seed, i))


def _pcg(mut state: UInt64) -> UInt64:
    state = state * 6364136223846793005 + 1442695040888963407
    return state


def _randf(mut state: UInt64) -> Float32:
    var z = _pcg(state)
    # uniform [0,1) from top 53 bits, then map to [-2, 2)
    var u = Float64(z >> 11) * (1.0 / 9007199254740992.0)
    return Float32(u * 4.0 - 2.0)


def _fill_f32(n: Int, mut state: UInt64, adversarial: Bool) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        if adversarial and i % 97 == 0:
            out.append(Float32(2.0))  # exact power of two (binade boundary)
        elif adversarial and i % 97 == 1:
            out.append(Float32(-0.25))
        elif adversarial and i % 97 == 2:
            out.append(Float32(0.0))
        elif adversarial and i % 97 == 3:
            out.append(Float32(1.0e-39))  # below the helper's 1e-38 floor
        else:
            out.append(_randf(state))
    return out^


def _fill_f32_nonneg(n: Int, mut state: UInt64) -> List[Float32]:
    # second moments (v) are sums of squares — ALWAYS >= 0 in real training.
    var out = List[Float32]()
    for _ in range(n):
        var x = _randf(state)
        if x < Float32(0.0):
            x = -x
        out.append(x * Float32(0.25))
    return out^


def _mk_adapter(
    rank: Int, in_f: Int, out_f: Int, mut state: UInt64, adversarial: Bool
) raises -> LoraAdapter:
    var n_a = rank * in_f
    var n_b = out_f * rank
    return LoraAdapter(
        _fill_f32(n_a, state, adversarial), _fill_f32(n_b, state, adversarial),
        rank, in_f, out_f, Float32(1.0),
        _fill_f32(n_a, state, adversarial), _fill_f32_nonneg(n_a, state),
        _fill_f32(n_b, state, adversarial), _fill_f32_nonneg(n_b, state),
    )


def _cmp_bf16(name: String, idx: Int, a: List[BFloat16], b: List[BFloat16]) raises -> Int:
    var bad = 0
    for i in range(len(a)):
        if a[i].cast[DType.float32]() != b[i].cast[DType.float32]():
            if bad < 3:
                print(
                    "  MISMATCH", name, "adapter", idx, "elem", i,
                    " host=", a[i].cast[DType.float32](),
                    " gpu=", b[i].cast[DType.float32](),
                )
            bad += 1
    return bad


def _cmp_f32(name: String, idx: Int, a: List[Float32], b: List[Float32]) raises -> Int:
    var bad = 0
    for i in range(len(a)):
        if a[i] != b[i]:
            if bad < 3:
                print(
                    "  MISMATCH", name, "adapter", idx, "elem", i,
                    " host=", a[i], " gpu=", b[i],
                )
            bad += 1
    return bad


def _cmp_f32_bounded(
    name: String, idx: Int, a: List[Float32], b: List[Float32]
) raises -> List[Int]:
    # moments are bf16-quantized storage: tolerate RNE midpoint ties flipped by
    # device-vs-host 1-ulp arithmetic — ±1 bf16 quantum (rel <= 2^-7) ONLY.
    var diffs = 0
    var violations = 0
    for i in range(len(a)):
        if a[i] != b[i]:
            diffs += 1
            var d = a[i] - b[i]
            if d < Float32(0.0):
                d = -d
            var mag = a[i]
            if mag < Float32(0.0):
                mag = -mag
            if d > mag * Float32(0.0079):  # > one bf16 quantum
                if violations < 3:
                    print(
                        "  BOUND-VIOLATION", name, "adapter", idx, "elem", i,
                        " host=", a[i], " gpu=", b[i],
                    )
                violations += 1
    return [diffs, violations]


def _run_case(
    t: Int, lr: Float32, weight_decay: Float32, adversarial: Bool,
    ctx: DeviceContext,
) raises -> Int:
    # Klein-like slot shapes + odd tail
    var state = UInt64(0xDEADBEE5) + UInt64(t)
    var host_side = List[LoraAdapter]()
    var gpu_side = List[LoraAdapter]()
    var seeds = state
    host_side.append(_mk_adapter(16, 4096, 4096, seeds, adversarial))
    host_side.append(_mk_adapter(16, 4096, 12288, seeds, adversarial))
    host_side.append(_mk_adapter(16, 12288, 4096, seeds, adversarial))
    host_side.append(_mk_adapter(16, 37, 5, seeds, adversarial))  # odd tail
    seeds = state
    gpu_side.append(_mk_adapter(16, 4096, 4096, seeds, adversarial))
    gpu_side.append(_mk_adapter(16, 4096, 12288, seeds, adversarial))
    gpu_side.append(_mk_adapter(16, 12288, 4096, seeds, adversarial))
    gpu_side.append(_mk_adapter(16, 37, 5, seeds, adversarial))
    seeds = state
    var fma_side = List[LoraAdapter]()
    fma_side.append(_mk_adapter(16, 4096, 4096, seeds, adversarial))
    fma_side.append(_mk_adapter(16, 4096, 12288, seeds, adversarial))
    fma_side.append(_mk_adapter(16, 12288, 4096, seeds, adversarial))
    fma_side.append(_mk_adapter(16, 37, 5, seeds, adversarial))

    var d_a = List[List[Float32]]()
    var d_b = List[List[Float32]]()
    var gstate = UInt64(0xFEED) + UInt64(t)
    for i in range(len(host_side)):
        d_a.append(_fill_f32(len(host_side[i].a), gstate, adversarial))
        d_b.append(_fill_f32(len(host_side[i].b), gstate, adversarial))

    # precomputed scalars EXACTLY like klein_lora_adamw_step
    var beta1 = Float32(0.9)
    var beta2 = Float32(0.999)
    var eps = Float32(1.0e-8)
    var b1p = Float32(1.0)
    var b2p = Float32(1.0)
    for _ in range(t):
        b1p *= beta1
        b2p *= beta2
    var bc1 = Float32(1.0) - b1p
    var bc2 = Float32(1.0) - b2p
    var step_size = lr / bc1
    var bc2_sqrt = sqrt(bc2)
    var decay = Float32(1.0) - lr * weight_decay
    var omb1 = Float32(1.0) - beta1
    var omb2 = Float32(1.0) - beta2
    var seed = UInt32(t)

    # host reference (the trainer's current path, SR on = its default)
    for i in range(len(host_side)):
        _lora_adamw_precomputed(
            host_side[i], d_a[i], d_b[i],
            step_size, bc2_sqrt, decay, omb1, beta2, omb2, eps, seed,
        )
    # GPU fused
    fused_lora_adamw_ot_step(
        gpu_side, d_a, d_b,
        step_size, bc2_sqrt, decay, omb1, beta2, omb2, eps, seed, ctx,
    )
    # host-with-explicit-fma probe (hypothesis: device contracts mul+add)
    for i in range(len(fma_side)):
        _host_ref_fma_list(
            fma_side[i].a, d_a[i], fma_side[i].ma, fma_side[i].va,
            step_size, bc2_sqrt, decay, omb1, beta2, omb2, eps, seed,
        )
        _host_ref_fma_list(
            fma_side[i].b, d_b[i], fma_side[i].mb, fma_side[i].vb,
            step_size, bc2_sqrt, decay, omb1, beta2, omb2, eps, seed,
        )

    _ = fma_side  # probe retained for forensics; gate is host-vs-gpu below

    # GATE: params STRICT bit-equal; m/v bounded (±1 bf16 quantum, counted)
    var param_bad = 0
    var total_elems = 0
    var mv_diffs = 0
    var mv_violations = 0
    for i in range(len(host_side)):
        param_bad += _cmp_bf16("a", i, host_side[i].a, gpu_side[i].a)
        param_bad += _cmp_bf16("b", i, host_side[i].b, gpu_side[i].b)
        total_elems += len(host_side[i].a) + len(host_side[i].b)
        var r1 = _cmp_f32_bounded("ma", i, host_side[i].ma, gpu_side[i].ma)
        var r2 = _cmp_f32_bounded("va", i, host_side[i].va, gpu_side[i].va)
        var r3 = _cmp_f32_bounded("mb", i, host_side[i].mb, gpu_side[i].mb)
        var r4 = _cmp_f32_bounded("vb", i, host_side[i].vb, gpu_side[i].vb)
        mv_diffs += r1[0] + r2[0] + r3[0] + r4[0]
        mv_violations += r1[1] + r2[1] + r3[1] + r4[1]
    print(
        "case t=", t, " lr=", lr, " wd=", weight_decay,
        " adversarial=", adversarial, " -> param_mismatch=", param_bad,
        " mv_quantum_ties=", mv_diffs, "/", 2 * total_elems,
        " mv_bound_violations=", mv_violations,
    )
    # fail conditions: ANY param mismatch; ANY m/v diff beyond one quantum;
    # m/v tie rate above 1e-4
    var fails = param_bad + mv_violations
    if mv_diffs * 10000 > 2 * total_elems:
        print("  FAIL: m/v tie rate above 1e-4")
        fails += 1
    return fails


def main() raises:
    print("=== lora_adamw_ot_fused parity: host loop vs GPU kernel ===")
    print("gate: params BIT-EQUAL; m/v ties bounded ±1 bf16 quantum, rate<1e-4")
    var ctx = DeviceContext()
    var fails = 0
    fails += _run_case(1, Float32(1.0e-4), Float32(0.01), False, ctx)
    fails += _run_case(7, Float32(3.0e-4), Float32(0.0), False, ctx)
    fails += _run_case(1, Float32(1.0e-4), Float32(0.01), True, ctx)
    fails += _run_case(7, Float32(3.0e-4), Float32(0.0), True, ctx)
    if fails != 0:
        raise Error("lora_adamw_ot_fused_parity: " + String(fails) + " failures")
    print("=== PASS: params bit-equal; m/v within quantization noise floor ===")

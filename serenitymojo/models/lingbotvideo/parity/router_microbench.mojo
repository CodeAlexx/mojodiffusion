# router_microbench.mojo — profile the host-side MoE stages at full res.
#
# Synthesizes logits [S,128] and times:
#   (1) lingbot_grouped_sigmoid_router (host)
#   (2) the deterministic combine loop (host DtoH + S*8*H multiply-add)
#   (3) grouped_expert_ffn routing-list build (host 128*n_slots scan)
# No 60GB stream — pure host-cost isolation.
#
#   cd /home/alex/mojodiffusion && \
#     pixi run mojo run -I . serenitymojo/models/lingbotvideo/parity/router_microbench.mojo

from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.lingbotvideo.moe import (
    lingbot_grouped_sigmoid_router,
    lingbot_grouped_sigmoid_router_gpu,
    TOP_K,
)

comptime S = 2017
comptime H = 2048
comptime E = 128


def _ms(dt: UInt) -> Float64:
    return Float64(dt) / 1.0e6


def main() raises:
    var ctx = DeviceContext()
    print("[BENCH] S =", S, " E =", E, " H =", H)

    # ── synth logits [S,E] via a cheap host LCG ──────────────────────────────
    var vals = List[Float32]()
    vals.resize(S * E, 0.0)
    var seed: UInt64 = 0x1234567
    for i in range(S * E):
        seed = seed * 6364136223846793005 + 1442695040888963407
        var u = Float32((seed >> 33) & 0x7FFFFF) / Float32(0x7FFFFF)
        vals[i] = (u - 0.5) * 6.0
    var logits = Tensor.from_host(vals, [S, E], STDtype.F32, ctx)

    var bvals = List[Float32]()
    bvals.resize(E, 0.0)
    for i in range(E):
        seed = seed * 6364136223846793005 + 1442695040888963407
        bvals[i] = (Float32((seed >> 40) & 0xFFFF) / Float32(0xFFFF) - 0.5) * 0.2
    var bias = Tensor.from_host(bvals, [E], STDtype.F32, ctx)

    # ── (1) HOST router ──────────────────────────────────────────────────────
    var t0 = perf_counter_ns()
    var plan_h = lingbot_grouped_sigmoid_router(logits, bias, ctx)
    var t1 = perf_counter_ns()
    print("[BENCH] (1) HOST router:", _ms(t1 - t0), "ms")

    # ── (1b) GPU router ──────────────────────────────────────────────────────
    var t2 = perf_counter_ns()
    var plan_g = lingbot_grouped_sigmoid_router_gpu(logits, bias, ctx)
    var t3 = perf_counter_ns()
    print("[BENCH] (1b) GPU router:", _ms(t3 - t2), "ms")

    # verify GPU == HOST (set + gating)
    var n = S * TOP_K
    var idmatch = 0
    var gdot: Float64 = 0.0
    var gna: Float64 = 0.0
    var gnb: Float64 = 0.0
    # set-match per token (order-invariant)
    var setmatch = 0
    for t in range(S):
        for j in range(TOP_K):
            if plan_h.expert_ids[t * TOP_K + j] == plan_g.expert_ids[t * TOP_K + j]:
                idmatch += 1
        # order-invariant set check
        for j in range(TOP_K):
            var eg = plan_g.expert_ids[t * TOP_K + j]
            var found = False
            for k in range(TOP_K):
                if plan_h.expert_ids[t * TOP_K + k] == eg:
                    found = True
            if found:
                setmatch += 1
    for i in range(n):
        var a = Float64(plan_h.gating[i])
        var b = Float64(plan_g.gating[i])
        gdot += a * b
        gna += a * a
        gnb += b * b
    print("[BENCH] GPU-vs-HOST id position match:", idmatch, "/", n)
    print("[BENCH] GPU-vs-HOST id SET match:", setmatch, "/", n)
    print("[BENCH] GPU-vs-HOST gating cos:", gdot / ((gna ** 0.5) * (gnb ** 0.5)))

    # ── (2) deterministic combine loop (host) ────────────────────────────────
    # simulate expert_out [S*TOP_K, H] on device, download + weighted sum.
    var eo_vals = List[Float32]()
    eo_vals.resize(S * TOP_K * H, 0.0)
    for i in range(S * TOP_K * H):
        eo_vals[i] = 0.001 * Float32(i % 97)
    var expert_out = Tensor.from_host(eo_vals, [S * TOP_K, H], STDtype.BF16, ctx)
    var t4 = perf_counter_ns()
    var eo_host = expert_out.to_host(ctx)
    var t5 = perf_counter_ns()
    var acc = List[Float32]()
    acc.resize(S * H, 0.0)
    for t in range(S):
        var ab = t * H
        for j in range(TOP_K):
            var slot = t * TOP_K + j
            var g = plan_h.gating[slot]
            var sb = slot * H
            for c in range(H):
                acc[ab + c] += eo_host[sb + c] * g
    var t6 = perf_counter_ns()
    print("[BENCH] (2a) combine DtoH download [S*8,H]:", _ms(t5 - t4), "ms")
    print("[BENCH] (2b) combine host mul-add S*8*H:", _ms(t6 - t5), "ms")

    # ── (3) grouped_expert_ffn routing-list build (host scan 128*n_slots) ────
    var n_slots = S * TOP_K
    var t7 = perf_counter_ns()
    var total_assigned = 0
    for ei in range(E):
        var src_tokens = List[Int]()
        var dst_slots = List[Int]()
        for s in range(n_slots):
            if plan_h.expert_ids[s] == ei:
                src_tokens.append(s // TOP_K)
                dst_slots.append(s)
        total_assigned += len(src_tokens)
    var t8 = perf_counter_ns()
    print("[BENCH] (3) grouped_ffn list build 128*n_slots scan:", _ms(t8 - t7), "ms (assigned", total_assigned, ")")

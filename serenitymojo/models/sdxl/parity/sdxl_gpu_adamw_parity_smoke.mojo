# serenitymojo/models/sdxl/parity/sdxl_gpu_adamw_parity_smoke.mojo
#
# UPDATE-PARITY smoke for the SDXL GPU AdamW dispatch: old host scalar loop
# (sdxl_lora_adamw_step_unfused -> _lora_adamw -> _adamw_host_list) vs the new
# fused GPU path (sdxl_lora_adamw_step with SDXL_GPU_ADAMW=True ->
# fused_lora_adamw_plain_step) on ONE fixed synthetic fixture: identical
# adapters (params + NONZERO moments), identical grads, 3 optimizer steps
# (t=1..3) with fresh grads per step, proj_in/proj_out adapters PRESENT so the
# ST-level Optional branch is exercised too.
#
# EXPECTATION (ledger MJ-1017 / lora_adamw_plain_fused_parity class): the math
# is per-element identical, but host scalar F32 vs device F32 can differ at ulp
# level (codegen FMA contraction/reassociation flipping RNE ties) — so the bar
# is WORST-CASE TOLERANCE, not exact equality:
#   * moments (F32 ma/va/mb/vb): worst abs diff per element <= 1e-6
#     (measured class in the fused gate is 1e-9/1e-10; 1e-6 is the hard bar).
#   * params (bf16 a/b): <=1 bf16 quantum worst, mismatch rate < 1e-4
#     (a 1e-6 absolute bar is meaningless for bf16 storage — one quantum near
#     |p|~0.02 is ~1e-4 absolute; report the abs diff, gate on quanta+rate).
#   * zero NaN anywhere.
# NOTE: this smoke is only a host-vs-GPU A/B while SDXL_GPU_ADAMW=True (the
# default). With the flag flipped off, both sides run the host loop and the
# comparison is trivially exact.
#
# Build (GPU; run from /home/alex/mojodiffusion; NEVER -O3):
#   rm -f serenitymojo.mojopkg
#   MEM_MAX=30G MEM_HIGH=26G SWAP_MAX=2G pixi run bash scripts/mem_safe.sh \
#     mojo build --optimization-level 2 --num-threads 4 -I . -I /home/alex/MOJO-libs \
#     -Xlinker -lm -Xlinker -Lserenitymojo/ops/cshim/lib \
#     -Xlinker -lserenity_cudnn_sdpa -Xlinker -rpath \
#     -Xlinker /home/alex/mojodiffusion/serenitymojo/ops/cshim/lib \
#     serenitymojo/models/sdxl/parity/sdxl_gpu_adamw_parity_smoke.mojo \
#     -o output/bin/sdxl_gpu_adamw_parity_smoke
#   LD_LIBRARY_PATH=.pixi/envs/default/lib:serenitymojo/ops/cshim/lib:$HOME/.local/lib/python3.12/site-packages/nvidia/cudnn/lib \
#     output/bin/sdxl_gpu_adamw_parity_smoke

from std.sys import has_accelerator
from std.gpu.host import DeviceContext
from std.collections import List, Optional

from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.models.sdxl.lora_block import (
    SDXL_SLOTS, SLOT_A2_K, SLOT_A2_V, SLOT_FF_PROJ, SLOT_FF_OUT,
)
from serenitymojo.models.sdxl.sdxl_unet_stack_lora import (
    SdxlLoraSet, SdxlStLoraGrads,
    sdxl_lora_adamw_step, sdxl_lora_adamw_step_unfused,
)

comptime DEPTH = 2
comptime C = 64
comptime CCTX = 16
comptime CFF = 32
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
    # second moments must be >= 0 (sqrt(v_hat) in the update).
    for i in range(len(x)):
        if x[i] < 0:
            x[i] = -x[i]
    return x^


# adapter with NONZERO params AND NONZERO moments (so step t=1 already
# exercises the full m/v/wd math, unlike the B=0/moments=0 train init).
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


# slot in/out shapes (mirrors sdxl_unet_stack_lora._slot_in/_slot_out).
def _slot_in(s: Int) -> Int:
    if s == SLOT_A2_K or s == SLOT_A2_V:
        return CCTX
    if s == SLOT_FF_OUT:
        return CFF
    return C


def _slot_out(s: Int) -> Int:
    if s == SLOT_FF_PROJ:
        return 2 * CFF
    return C


def _mk_grads(
    d_a: List[List[Float32]], d_b: List[List[Float32]],
    d_pi_a: List[Float32], d_pi_b: List[Float32],
    d_po_a: List[Float32], d_po_b: List[Float32],
) -> SdxlStLoraGrads:
    return SdxlStLoraGrads(
        d_a.copy(), d_b.copy(), List[Float32](), List[Float32](), 0,
        d_pi_a.copy(), d_pi_b.copy(), d_po_a.copy(), d_po_b.copy(),
    )


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
              mut worst_q: Int, mut mism: Int, mut max_abs: Float32) raises:
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
            if d > worst_q:
                worst_q = d


def _cmp_adapter(
    tag: String, h: LoraAdapter, g: LoraAdapter,
    mut p_worst_q: Int, mut p_mism: Int, mut p_max_abs: Float32,
    mut m_worst_ulp: Int, mut m_mism: Int, mut m_max_abs: Float32,
    mut v_worst_ulp: Int, mut v_mism: Int, mut v_max_abs: Float32,
) raises:
    _cmp_bf16(tag + ".a", h.a, g.a, p_worst_q, p_mism, p_max_abs)
    _cmp_bf16(tag + ".b", h.b, g.b, p_worst_q, p_mism, p_max_abs)
    _cmp_f32(tag + ".ma", h.ma, g.ma, m_worst_ulp, m_mism, m_max_abs)
    _cmp_f32(tag + ".mb", h.mb, g.mb, m_worst_ulp, m_mism, m_max_abs)
    _cmp_f32(tag + ".va", h.va, g.va, v_worst_ulp, v_mism, v_max_abs)
    _cmp_f32(tag + ".vb", h.vb, g.vb, v_worst_ulp, v_mism, v_max_abs)


def main() raises:
    comptime if not has_accelerator():
        print("sdxl_gpu_adamw_parity_smoke: GPU required")
        raise Error("no accelerator")
    else:
        var ctx = DeviceContext()
        print("==== sdxl GPU-AdamW update parity (host loop vs SDXL_GPU_ADAMW fused) ====")
        print("DEPTH=", DEPTH, " C=", C, " CCTX=", CCTX, " CFF=", CFF,
              " RANK=", RANK, " slots/block=", SDXL_SLOTS)

        var rng = _Lcg(20260701)
        var n = DEPTH * SDXL_SLOTS

        # ── ONE fixture, deep-copied into both sides ──
        var host_ad = List[LoraAdapter]()
        var gpu_ad = List[LoraAdapter]()
        var d_a = List[List[Float32]]()
        var d_b = List[List[Float32]]()
        var total_elems = 0
        for _ in range(DEPTH):
            for s in range(SDXL_SLOTS):
                var in_f = _slot_in(s)
                var out_f = _slot_out(s)
                var ad = _mk_adapter(rng, RANK, in_f, out_f)
                host_ad.append(ad.copy())
                gpu_ad.append(ad.copy())
                d_a.append(_rand_list(rng, RANK * in_f, 0.005))
                d_b.append(_rand_list(rng, out_f * RANK, 0.005))
                total_elems += RANK * in_f + out_f * RANK
        var pi_ad = _mk_adapter(rng, RANK, C, C)
        var po_ad = _mk_adapter(rng, RANK, C, C)
        total_elems += 2 * (RANK * C + C * RANK)
        var d_pi_a = _rand_list(rng, RANK * C, 0.005)
        var d_pi_b = _rand_list(rng, C * RANK, 0.005)
        var d_po_a = _rand_list(rng, RANK * C, 0.005)
        var d_po_b = _rand_list(rng, C * RANK, 0.005)

        var host_set = SdxlLoraSet(
            host_ad^, DEPTH, RANK,
            Optional[LoraAdapter](pi_ad.copy()), Optional[LoraAdapter](po_ad.copy()),
        )
        var gpu_set = SdxlLoraSet(
            gpu_ad^, DEPTH, RANK,
            Optional[LoraAdapter](pi_ad.copy()), Optional[LoraAdapter](po_ad.copy()),
        )

        var lr = Float32(3.0e-4)
        var beta1 = Float32(0.9)
        var beta2 = Float32(0.999)
        var eps = Float32(1.0e-8)
        var wd = Float32(0.01)

        # ── 3 optimizer steps, FRESH grads each step (same values both sides) ──
        for t in range(1, 4):
            var hg = _mk_grads(d_a, d_b, d_pi_a, d_pi_b, d_po_a, d_po_b)
            var gg = _mk_grads(d_a, d_b, d_pi_a, d_pi_b, d_po_a, d_po_b)
            sdxl_lora_adamw_step_unfused(
                host_set, hg, t, lr, ctx, beta1, beta2, eps, wd,
            )
            sdxl_lora_adamw_step(
                gpu_set, gg, t, lr, ctx, beta1, beta2, eps, wd,
            )
            for i in range(len(d_a)):
                d_a[i] = _rand_list(rng, len(d_a[i]), 0.005)
                d_b[i] = _rand_list(rng, len(d_b[i]), 0.005)
            d_pi_a = _rand_list(rng, len(d_pi_a), 0.005)
            d_pi_b = _rand_list(rng, len(d_pi_b), 0.005)
            d_po_a = _rand_list(rng, len(d_po_a), 0.005)
            d_po_b = _rand_list(rng, len(d_po_b), 0.005)

        # ── compare params (bf16) + moments (F32), block AND proj adapters ──
        var p_worst_q = 0
        var p_mism = 0
        var p_max_abs = Float32(0.0)
        var m_worst_ulp = 0
        var m_mism = 0
        var m_max_abs = Float32(0.0)
        var v_worst_ulp = 0
        var v_mism = 0
        var v_max_abs = Float32(0.0)
        for i in range(n):
            _cmp_adapter(
                String("ad[") + String(i) + String("]"),
                host_set.ad[i], gpu_set.ad[i],
                p_worst_q, p_mism, p_max_abs,
                m_worst_ulp, m_mism, m_max_abs,
                v_worst_ulp, v_mism, v_max_abs,
            )
        if not host_set.proj_in_ad or not gpu_set.proj_in_ad:
            raise Error("proj_in adapter went missing")
        if not host_set.proj_out_ad or not gpu_set.proj_out_ad:
            raise Error("proj_out adapter went missing")
        _cmp_adapter(
            String("proj_in"),
            host_set.proj_in_ad.value(), gpu_set.proj_in_ad.value(),
            p_worst_q, p_mism, p_max_abs,
            m_worst_ulp, m_mism, m_max_abs,
            v_worst_ulp, v_mism, v_max_abs,
        )
        _cmp_adapter(
            String("proj_out"),
            host_set.proj_out_ad.value(), gpu_set.proj_out_ad.value(),
            p_worst_q, p_mism, p_max_abs,
            m_worst_ulp, m_mism, m_max_abs,
            v_worst_ulp, v_mism, v_max_abs,
        )

        var p_rate = Float64(p_mism) / Float64(total_elems)
        print("")
        print("params (bf16 a/b): mismatches=", p_mism, "/", total_elems,
              " rate=", p_rate, " worst_quanta=", p_worst_q,
              " worst_abs_diff=", p_max_abs)
        print("first moments (F32 ma/mb): mismatches=", m_mism, "/", total_elems,
              " worst_ulp=", m_worst_ulp, " worst_abs_diff=", m_max_abs)
        print("second moments (F32 va/vb): mismatches=", v_mism, "/", total_elems,
              " worst_ulp=", v_worst_ulp, " worst_abs_diff=", v_max_abs)

        # ── the worst-case-tolerance bars (NOT exact equality; MJ-1017 class) ──
        var ok = True
        if p_worst_q > 1 or p_rate > 1.0e-4:
            print("FAIL: params outside +-1-bf16-quantum / 1e-4-rate bar")
            ok = False
        if m_max_abs > Float32(1.0e-6):
            print("FAIL: first moments worst abs diff above 1e-6 bar")
            ok = False
        if v_max_abs > Float32(1.0e-6):
            print("FAIL: second moments worst abs diff above 1e-6 bar")
            ok = False
        print("")
        if ok:
            print("VERDICT: PASS — sdxl GPU AdamW matches host loop within the",
                  " worst-case bars (3 steps, ", n, " block adapters + proj_in/proj_out)")
        else:
            raise Error("sdxl_gpu_adamw_parity_smoke: FAIL")

# flux_adamw_gpu_update_parity_smoke.mojo — GATE for FLUX_GPU_ADAMW: the old
# host scalar AdamW loop (flux_lora_adamw_step_host / flux_stack_lora_adamw_step_host,
# i.e. _lora_adamw → _adamw_host_list) vs the product dispatchers
# (flux_lora_adamw_step / flux_stack_lora_adamw_step), which route through the
# fused GPU kernel fused_lora_adamw_plain_step when FLUX_GPU_ADAMW=True
# (the default). Fixed synthetic fixture: SAME params/grads/moments on both
# sides, 3 optimizer steps with fresh grads per step.
#
# EXPECTATION (ledger MJ-1017 / klein fused-AdamW class): identical per-element
# math, but device codegen may contract/reassociate FMA paths, flipping RNE
# ties at ulp level — so the bar is WORST-CASE TOLERANCE, NOT exact equality:
#   * F32 moments (ma/va/mb/vb): worst per-element abs diff ≤ 1e-6 (measured
#     class for this kernel pair is ≤1e-9/1e-10 in
#     training/lora_adamw_plain_fused_parity.mojo; 1e-6 is the campaign bar).
#   * BF16 params (a/b): pre-round F32 differs at ulp level, so the rounded
#     bf16 is bit-equal or off by ≤1 bf16 quantum (worst abs diff reported).
#   * zero NaN anywhere.
# NOTE: the gate is only meaningful with FLUX_GPU_ADAMW=True (default) — with
# the flag off both sides run the same host loop and the compare is trivial.
# The flag value is printed so the log is honest.
#
# Build (GPU, campaign -O2 convention; run from /home/alex/mojodiffusion):
#   rm -f serenitymojo.mojopkg
#   MEM_MAX=30G MEM_HIGH=26G SWAP_MAX=2G pixi run bash scripts/mem_safe.sh \
#     mojo build --optimization-level 2 --num-threads 4 -I . -I /home/alex/MOJO-libs \
#     -Xlinker -lm -Xlinker -Lserenitymojo/ops/cshim/lib \
#     -Xlinker -lserenity_cudnn_sdpa -Xlinker -rpath \
#     -Xlinker /home/alex/mojodiffusion/serenitymojo/ops/cshim/lib \
#     serenitymojo/models/flux/parity/flux_adamw_gpu_update_parity_smoke.mojo \
#     -o output/bin/flux_adamw_gpu_update_parity_smoke
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.sys import has_accelerator
from std.gpu.host import DeviceContext
from std.collections import List, Optional

from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.models.flux.flux_stack_lora import (
    FLUX_GPU_ADAMW,
    FluxLoraSet, FluxLoraGradSet, FluxStackLoraSet,
    ST_LEVEL_SLOTS,
    flux_lora_adamw_step, flux_lora_adamw_step_host,
    flux_stack_lora_adamw_step, flux_stack_lora_adamw_step_host,
    total_adapters,
)


# ── deterministic fixture (same LCG as lora_adamw_plain_fused_parity.mojo) ────
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
    # nonzero params AND nonzero moments so step t>1 exercises the full chain.
    return LoraAdapter(
        _rand_list(rng, rank * in_f, 0.02),
        _rand_list(rng, out_f * rank, 0.02),
        rank, in_f, out_f, Float32(1.0) / Float32(rank),
        _rand_list(rng, rank * in_f, 0.001),
        _abs_list(_rand_list(rng, rank * in_f, 0.0001)),
        _rand_list(rng, out_f * rank, 0.001),
        _abs_list(_rand_list(rng, out_f * rank, 0.0001)),
    )


# ── comparison (worst abs diff per element; NaN fail-loud) ────────────────────
def _ulp_diff_f32(a: Float32, b: Float32) -> Int:
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


def _cmp_adapter(a: LoraAdapter, b: LoraAdapter, mut n_elems: Int,
                 mut p_quanta: Int, mut p_mism: Int, mut p_max_abs: Float32,
                 mut mv_ulp: Int, mut mv_mism: Int, mut mv_max_abs: Float32) raises:
    _cmp_bf16("a", a.a, b.a, p_quanta, p_mism, p_max_abs)
    _cmp_bf16("b", a.b, b.b, p_quanta, p_mism, p_max_abs)
    _cmp_f32("ma", a.ma, b.ma, mv_ulp, mv_mism, mv_max_abs)
    _cmp_f32("va", a.va, b.va, mv_ulp, mv_mism, mv_max_abs)
    _cmp_f32("mb", a.mb, b.mb, mv_ulp, mv_mism, mv_max_abs)
    _cmp_f32("vb", a.vb, b.vb, mv_ulp, mv_mism, mv_max_abs)
    n_elems += len(a.a) + len(a.b)


def main() raises:
    comptime if not has_accelerator():
        print("flux_adamw_gpu_update_parity_smoke: GPU required")
        raise Error("no accelerator")
    else:
        var ctx = DeviceContext()
        var rng = _Lcg(4242)
        print("flux_adamw_gpu_update_parity_smoke: FLUX_GPU_ADAMW=", FLUX_GPU_ADAMW)

        # ── fixture 1: flat block-projection FluxLoraSet (num_double=1 → 12
        # slots, num_single=1 → 5 slots) with the real flux slot shapes at
        # reduced D/Fmlp; identical copies for host oracle and GPU side.
        var rank = 8
        var D = 128
        var Fmlp = 256
        var ads = List[LoraAdapter]()
        for _stream in range(2):                     # img then txt
            ads.append(_mk_adapter(rng, rank, D, D))       # to_q
            ads.append(_mk_adapter(rng, rank, D, D))       # to_k
            ads.append(_mk_adapter(rng, rank, D, D))       # to_v
            ads.append(_mk_adapter(rng, rank, D, D))       # proj
            ads.append(_mk_adapter(rng, rank, D, Fmlp))    # mlp0
            ads.append(_mk_adapter(rng, rank, Fmlp, D))    # mlp2
        ads.append(_mk_adapter(rng, rank, D, D))           # sgl to_q
        ads.append(_mk_adapter(rng, rank, D, D))           # sgl to_k
        ads.append(_mk_adapter(rng, rank, D, D))           # sgl to_v
        ads.append(_mk_adapter(rng, rank, D, Fmlp))        # sgl proj_mlp
        ads.append(_mk_adapter(rng, rank, D + Fmlp, D))    # sgl linear2
        var host_set = FluxLoraSet(ads^, 1, 1, rank)
        var gpu_set = host_set.copy()
        var n_flat = total_adapters(host_set)

        # ── fixture 2: SPARSE stack-level set (some slots None) to exercise
        # the gather/scatter walk. Populated: level slots 0,1,3,8; double bi=0
        # img+txt, bi=1 img only; single bi=1 only → 8 populated adapters.
        var level = List[Optional[LoraAdapter]]()
        for slot in range(ST_LEVEL_SLOTS):
            if slot == 0 or slot == 1 or slot == 3 or slot == 8:
                level.append(Optional[LoraAdapter](_mk_adapter(rng, rank, 96, 64)))
            else:
                level.append(Optional[LoraAdapter](None))
        var dbl_img = List[Optional[LoraAdapter]]()
        var dbl_txt = List[Optional[LoraAdapter]]()
        dbl_img.append(Optional[LoraAdapter](_mk_adapter(rng, rank, 64, 96)))
        dbl_txt.append(Optional[LoraAdapter](_mk_adapter(rng, rank, 64, 96)))
        dbl_img.append(Optional[LoraAdapter](_mk_adapter(rng, rank, 64, 96)))
        dbl_txt.append(Optional[LoraAdapter](None))
        var sgl = List[Optional[LoraAdapter]]()
        sgl.append(Optional[LoraAdapter](None))
        sgl.append(Optional[LoraAdapter](_mk_adapter(rng, rank, 64, 48)))
        var host_sset = FluxStackLoraSet(level^, dbl_img^, dbl_txt^, sgl^, 2, 2, rank, True)
        var gpu_sset = host_sset.copy()
        var n_stack = 8

        # grad list shapes parallel to the fixtures (flat set; then populated
        # stack walk order: level 0,1,3,8 → dbl0 img, dbl0 txt, dbl1 img → sgl1).
        var d_a = List[List[Float32]]()
        var d_b = List[List[Float32]]()
        for i in range(n_flat):
            d_a.append(_rand_list(rng, len(host_set.ad[i].a), 0.005))
            d_b.append(_rand_list(rng, len(host_set.ad[i].b), 0.005))
        var st_d_a = List[List[Float32]]()
        var st_d_b = List[List[Float32]]()
        for _ in range(4):
            st_d_a.append(_rand_list(rng, rank * 96, 0.005))
            st_d_b.append(_rand_list(rng, 64 * rank, 0.005))
        for _ in range(3):
            st_d_a.append(_rand_list(rng, rank * 64, 0.005))
            st_d_b.append(_rand_list(rng, 96 * rank, 0.005))
        st_d_a.append(_rand_list(rng, rank * 64, 0.005))
        st_d_b.append(_rand_list(rng, 48 * rank, 0.005))

        var lr = Float32(3.0e-4)
        var beta1 = Float32(0.9)
        var beta2 = Float32(0.999)
        var eps = Float32(1.0e-8)
        var wd = Float32(0.01)

        # ── 3 optimizer steps, fresh grads each step (same on both sides) ────
        for t in range(1, 4):
            var g = FluxLoraGradSet(
                d_a.copy(), d_b.copy(),
                List[Float32](), List[Float32](), List[Float32](),
                List[Float32](), List[Float32](), List[Float32](),
                0,
                st_d_a.copy(), st_d_b.copy(),
            )
            flux_lora_adamw_step_host(host_set, g, t, lr, ctx, beta1, beta2, eps, wd)
            flux_lora_adamw_step(gpu_set, g, t, lr, ctx, beta1, beta2, eps, wd)
            flux_stack_lora_adamw_step_host(host_sset, g, t, lr, ctx, beta1, beta2, eps, wd)
            flux_stack_lora_adamw_step(gpu_sset, g, t, lr, ctx, beta1, beta2, eps, wd)
            for i in range(len(d_a)):
                d_a[i] = _rand_list(rng, len(d_a[i]), 0.005)
                d_b[i] = _rand_list(rng, len(d_b[i]), 0.005)
            for i in range(len(st_d_a)):
                st_d_a[i] = _rand_list(rng, len(st_d_a[i]), 0.005)
                st_d_b[i] = _rand_list(rng, len(st_d_b[i]), 0.005)

        # ── compare: worst abs diff per element (report), MJ-1017 bars ───────
        var n_elems = 0
        var p_quanta = 0
        var p_mism = 0
        var p_max_abs = Float32(0.0)
        var mv_ulp = 0
        var mv_mism = 0
        var mv_max_abs = Float32(0.0)
        for i in range(n_flat):
            _cmp_adapter(host_set.ad[i], gpu_set.ad[i], n_elems,
                         p_quanta, p_mism, p_max_abs, mv_ulp, mv_mism, mv_max_abs)
        for slot in range(ST_LEVEL_SLOTS):
            if host_sset.level[slot]:
                if not gpu_sset.level[slot]:
                    raise Error("stack level slot population mismatch")
                _cmp_adapter(host_sset.level[slot].value(), gpu_sset.level[slot].value(),
                             n_elems, p_quanta, p_mism, p_max_abs,
                             mv_ulp, mv_mism, mv_max_abs)
        for bi in range(host_sset.num_double):
            if host_sset.dbl_img_mod[bi]:
                _cmp_adapter(host_sset.dbl_img_mod[bi].value(), gpu_sset.dbl_img_mod[bi].value(),
                             n_elems, p_quanta, p_mism, p_max_abs,
                             mv_ulp, mv_mism, mv_max_abs)
            if host_sset.dbl_txt_mod[bi]:
                _cmp_adapter(host_sset.dbl_txt_mod[bi].value(), gpu_sset.dbl_txt_mod[bi].value(),
                             n_elems, p_quanta, p_mism, p_max_abs,
                             mv_ulp, mv_mism, mv_max_abs)
        for bi in range(host_sset.num_single):
            if host_sset.sgl_mod[bi]:
                _cmp_adapter(host_sset.sgl_mod[bi].value(), gpu_sset.sgl_mod[bi].value(),
                             n_elems, p_quanta, p_mism, p_max_abs,
                             mv_ulp, mv_mism, mv_max_abs)

        var p_rate = Float64(p_mism) / Float64(n_elems)
        print("adapters compared: flat=", n_flat, " stack=", n_stack,
              " param_elems=", n_elems)
        print("params(bf16): mismatches=", p_mism, " rate=", p_rate,
              " worst_quanta=", p_quanta, " worst_abs_diff=", p_max_abs)
        print("moments(f32): mismatches=", mv_mism, " worst_ulp=", mv_ulp,
              " worst_abs_diff=", mv_max_abs)
        if p_quanta > 1 or p_rate > 1.0e-4:
            raise Error("params outside ±1-bf16-quantum / 1e-4-rate bar")
        if mv_max_abs > Float32(1.0e-6):
            raise Error("moments worst abs diff above 1e-6 bar")
        print("flux_adamw_gpu_update_parity_smoke: PASS (3 steps, ",
              n_flat + n_stack, " adapters, host loop vs FLUX_GPU_ADAMW path)")

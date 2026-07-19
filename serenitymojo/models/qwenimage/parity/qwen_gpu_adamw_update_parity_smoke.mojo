# qwen_gpu_adamw_update_parity_smoke.mojo — GATE for QWEN_GPU_ADAMW: the fused
# GPU PLAIN-AdamW qwen optimizer path (qwen_offload_lora_adamw_step_gpu →
# fused_lora_adamw_plain_step) vs the retained host scalar loop
# (qwen_offload_lora_adamw_step_host → _lora_adamw → _adamw_host_list) on an
# identical fixed synthetic fixture (same params/grads/moments, LCG-seeded).
#
# EXPECTATION (ledger MJ-1017 / lora_adamw_plain_fused.mojo header): identical
# per-element math, but device codegen may contract FMA / reassociate, flipping
# RNE ties — so a WORST-CASE-TOLERANCE bar, NOT exact equality:
#   * F32 moments (m/v): worst per-element abs diff ≤ 1e-6 (measured class is
#     ~1e-9; 1e-6 is the campaign's tolerance-class bar).
#   * BF16 params (a/b): within ±1 bf16 quantum per element (a 1-quantum flip
#     at |p|~0.02 is ~1.5e-4 in F32, so an abs-diff-only 1e-6 bar cannot apply
#     to bf16 storage); worst abs diff is still REPORTED.
#   * A slot with EMPTY grads (old loop's `len(d_a)>0` skip) must stay
#     bit-identical to the fixture on BOTH paths.
#
# Build (GPU, from /home/alex/mojodiffusion — -O2 like every heavy target):
#   rm -f serenitymojo.mojopkg && MEM_MAX=30G MEM_HIGH=26G SWAP_MAX=2G \
#     pixi run bash scripts/mem_safe.sh mojo build --optimization-level 2 \
#     --num-threads 4 -I . -I /home/alex/MOJO-libs -Xlinker -lm \
#     serenitymojo/models/qwenimage/parity/qwen_gpu_adamw_update_parity_smoke.mojo \
#     -o output/bin/qwen_gpu_adamw_update_parity_smoke
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.builtin.dtype import DType
from std.gpu.host import DeviceContext
from std.sys import has_accelerator

from serenitymojo.models.klein.lora_block import LoraAdapter
from serenitymojo.models.qwenimage.qwenimage_stack_lora import (
    DBL_SLOTS,
    QwenLoraGradSet,
    QwenLoraSet,
    qwen_offload_lora_adamw_step_gpu,
    qwen_offload_lora_adamw_step_host,
)


# small qwen-shaped fixture: 2 double blocks, D=64, F=128, rank=4 → 24 adapters
comptime _ND = 2
comptime _D = 64
comptime _F = 128
comptime _RANK = 4
comptime _EMPTY_SLOT = 7   # block 0 txt.k left WITHOUT grads (skip semantics)
comptime _STEPS = 3
comptime _M_BAR = Float32(1.0e-6)   # F32 moment worst-abs bar (tolerance class)
comptime _P_QUANTA_BAR = 1          # bf16 param worst ±quanta bar (MJ-1017)


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


def _slot_in(slot: Int) -> Int:
    # per-block slot layout: img q,k,v,out,ffu,ffd then txt same six.
    var s = slot % 6
    if s == 4:
        return _D      # ff_up: [F, D] weight ⇒ in=D
    if s == 5:
        return _F      # ff_down: [D, F] weight ⇒ in=F
    return _D


def _slot_out(slot: Int) -> Int:
    var s = slot % 6
    if s == 4:
        return _F
    if s == 5:
        return _D
    return _D


def _mk_adapter(mut rng: _Lcg, in_f: Int, out_f: Int) -> LoraAdapter:
    # non-zero params AND non-zero moments (v ≥ 0): the fixture exercises the
    # full m/v recurrence from step 1, not just the zero-state special case.
    return LoraAdapter(
        _rand_list(rng, _RANK * in_f, 0.02),
        _rand_list(rng, out_f * _RANK, 0.02),
        _RANK, in_f, out_f, Float32(1.0) / Float32(_RANK),
        _rand_list(rng, _RANK * in_f, 0.001),
        _abs_list(_rand_list(rng, _RANK * in_f, 0.0001)),
        _rand_list(rng, out_f * _RANK, 0.001),
        _abs_list(_rand_list(rng, out_f * _RANK, 0.0001)),
    )


def _fresh_grads(mut rng: _Lcg) -> QwenLoraGradSet:
    var d_a = List[List[Float32]]()
    var d_b = List[List[Float32]]()
    for idx in range(_ND * DBL_SLOTS):
        if idx == _EMPTY_SLOT:
            d_a.append(List[Float32]())   # skipped slot: NO grads this run
            d_b.append(List[Float32]())
        else:
            d_a.append(_rand_list(rng, _RANK * _slot_in(idx), 0.005))
            d_b.append(_rand_list(rng, _slot_out(idx) * _RANK, 0.005))
    return QwenLoraGradSet(
        d_a^, d_b^, List[Float32](), List[Float32](), 0
    )


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


def _require_bits_equal_bf16(name: String, x: List[BFloat16], y: List[BFloat16]) raises:
    if len(x) != len(y):
        raise Error(name + ": length mismatch")
    for i in range(len(x)):
        if Int(x[i].to_bits[DType.uint16]()) != Int(y[i].to_bits[DType.uint16]()):
            raise Error(name + ": untouched slot changed at " + String(i))


def _require_bits_equal_f32(name: String, x: List[Float32], y: List[Float32]) raises:
    if len(x) != len(y):
        raise Error(name + ": length mismatch")
    for i in range(len(x)):
        if x[i] != y[i]:
            raise Error(name + ": untouched slot changed at " + String(i))


def _require_untouched(name: String, a0: LoraAdapter, a1: LoraAdapter) raises:
    _require_bits_equal_bf16(name + ".a", a0.a, a1.a)
    _require_bits_equal_bf16(name + ".b", a0.b, a1.b)
    _require_bits_equal_f32(name + ".ma", a0.ma, a1.ma)
    _require_bits_equal_f32(name + ".va", a0.va, a1.va)
    _require_bits_equal_f32(name + ".mb", a0.mb, a1.mb)
    _require_bits_equal_f32(name + ".vb", a0.vb, a1.vb)


def main() raises:
    comptime if not has_accelerator():
        print("qwen_gpu_adamw_update_parity_smoke: GPU required")
        raise Error("no accelerator")
    else:
        var ctx = DeviceContext()
        var rng = _Lcg(4242)

        # ONE fixture, duplicated bit-for-bit onto both paths.
        var dbl = List[LoraAdapter]()
        for idx in range(_ND * DBL_SLOTS):
            dbl.append(_mk_adapter(rng, _slot_in(idx), _slot_out(idx)))
        var host_set = QwenLoraSet(dbl^, _ND, _RANK)
        var gpu_set = host_set.copy()
        var empty_slot_init = host_set.dbl[_EMPTY_SLOT].copy()

        var lr = Float32(3.0e-4)
        var beta1 = Float32(0.9)
        var beta2 = Float32(0.999)
        var eps = Float32(1.0e-8)
        var wd = Float32(0.01)

        # _STEPS optimizer steps, FRESH grads each step (same for both sides).
        for t in range(1, _STEPS + 1):
            var grads = _fresh_grads(rng)
            qwen_offload_lora_adamw_step_host(
                host_set, grads, t, lr, ctx, beta1, beta2, eps, wd
            )
            qwen_offload_lora_adamw_step_gpu(
                gpu_set, grads, t, lr, ctx, beta1, beta2, eps, wd
            )

        var total_elems = 0
        for idx in range(_ND * DBL_SLOTS):
            total_elems += _RANK * _slot_in(idx) + _slot_out(idx) * _RANK

        var p_worst_quanta = 0
        var p_mism = 0
        var p_max_abs = Float32(0.0)
        var m_mism = 0
        var m_max_abs = Float32(0.0)
        var v_mism = 0
        var v_max_abs = Float32(0.0)
        for idx in range(_ND * DBL_SLOTS):
            _cmp_bf16("a", host_set.dbl[idx].a, gpu_set.dbl[idx].a,
                      p_worst_quanta, p_mism, p_max_abs)
            _cmp_bf16("b", host_set.dbl[idx].b, gpu_set.dbl[idx].b,
                      p_worst_quanta, p_mism, p_max_abs)
            _cmp_f32("ma", host_set.dbl[idx].ma, gpu_set.dbl[idx].ma, m_mism, m_max_abs)
            _cmp_f32("mb", host_set.dbl[idx].mb, gpu_set.dbl[idx].mb, m_mism, m_max_abs)
            _cmp_f32("va", host_set.dbl[idx].va, gpu_set.dbl[idx].va, v_mism, v_max_abs)
            _cmp_f32("vb", host_set.dbl[idx].vb, gpu_set.dbl[idx].vb, v_mism, v_max_abs)

        # the grad-less slot must be untouched on BOTH paths (skip semantics).
        _require_untouched("host_empty", empty_slot_init, host_set.dbl[_EMPTY_SLOT])
        _require_untouched("gpu_empty", empty_slot_init, gpu_set.dbl[_EMPTY_SLOT])

        print("params: mismatches=", p_mism, "/", total_elems,
              " worst_quanta=", p_worst_quanta, " worst_abs=", p_max_abs)
        print("first moments: mismatches=", m_mism, "/", total_elems,
              " worst_abs=", m_max_abs)
        print("second moments: mismatches=", v_mism, "/", total_elems,
              " worst_abs=", v_max_abs)
        if p_worst_quanta > _P_QUANTA_BAR:
            raise Error("params outside ±1 bf16 quantum bar")
        if m_max_abs > _M_BAR or v_max_abs > _M_BAR:
            raise Error("moments outside 1e-6 worst-abs bar")
        print("qwen_gpu_adamw_update_parity_smoke: PASS (",
              _STEPS, "steps,", _ND * DBL_SLOTS,
              "adapters, slot", _EMPTY_SLOT, "grad-less)")

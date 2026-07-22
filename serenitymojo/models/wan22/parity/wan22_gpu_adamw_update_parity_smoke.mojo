# serenitymojo/models/wan22/parity/wan22_gpu_adamw_update_parity_smoke.mojo
#
# UPDATE-PARITY gate for WAN22_GPU_ADAMW (wan22_stack_lora.mojo): the fused
# GPU PLAIN-AdamW path taken by wan22_lora_adamw_step (default True) vs the
# retained host scalar loop wan22_lora_adamw_step_host, on ONE fixed synthetic
# fixture (identical params / grads / moments on both sides, 3 optimizer
# steps, fresh grads per step). Also exercises the host loop's skip-empty
# semantics: one adapter gets EMPTY grad lists and must stay bit-untouched on
# BOTH paths.
#
# EXPECTATION: identical per-element math, but GPU FMA contraction /
# reassociation vs the host F32 chain flips RNE ties at ulp level (klein
# fused-AdamW class, ledger MJ-1017) — so the bar is WORST-CASE TOLERANCE,
# NOT exact equality:
#   params (bf16 writeback): worst diff ≤ 1 bf16 quantum (worst abs diff
#     reported alongside);
#   moments (F32 state):     worst abs diff ≤ 1e-6 per element (ulp-level
#     drift sits orders below this);
#   zero NaN anywhere; empty-grad adapter bit-equal to its initial state.
#
# Build (GPU, from /home/alex/mojodiffusion; rm -f serenitymojo.mojopkg first):
#   MEM_MAX=30G MEM_HIGH=26G SWAP_MAX=2G pixi run bash scripts/mem_safe.sh \
#     mojo build --optimization-level 2 --num-threads 4 -I . -I /home/alex/MOJO-libs \
#     -Xlinker -lm -Xlinker -Lserenitymojo/ops/cshim/lib \
#     -Xlinker -lserenity_cudnn_sdpa -Xlinker -rpath \
#     -Xlinker /home/alex/mojodiffusion/serenitymojo/ops/cshim/lib \
#     serenitymojo/models/wan22/parity/wan22_gpu_adamw_update_parity_smoke.mojo \
#     -o output/bin/wan22_gpu_adamw_update_parity_smoke
# Run:
#   LD_LIBRARY_PATH=.pixi/envs/default/lib:serenitymojo/ops/cshim/lib:$HOME/.local/lib/python3.12/site-packages/nvidia/cudnn/lib \
#     output/bin/wan22_gpu_adamw_update_parity_smoke
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.sys import has_accelerator
from std.gpu.host import DeviceContext

from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.models.wan22.wan22_stack_lora import (
    WAN22_GPU_ADAMW,
    Wan22LoraGradSet,
    Wan22LoraSet,
    wan22_lora_adamw_step,
    wan22_lora_adamw_step_host,
    wan22_total_adapters,
)


comptime NUM_BLOCKS = 2
comptime SLOTS = 10           # WAN_LORA_SLOTS: sa_{q,k,v,o}+ca_{q,k,v,o}+ffn.0+ffn.2
# (synthetic set uses square DIM×DIM adapters for all slots; the AdamW update is
# per-adapter shape-agnostic, so this stays a valid host-vs-GPU equivalence gate.)
comptime DIM = 96             # wan22 adapters are all in=out=dim
comptime RANK = 4
comptime EMPTY_ADAPTER = 5    # this one gets empty grads (skip-empty gate)
comptime STEPS = 3
comptime MOMENT_ABS_BAR = Float32(1.0e-6)   # worst-case per-element bar


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


def _mk_adapter(mut rng: _Lcg) -> LoraAdapter:
    # nonzero params AND nonzero moments so all update terms are live.
    return LoraAdapter(
        _rand_list(rng, RANK * DIM, 0.02),
        _rand_list(rng, DIM * RANK, 0.02),
        RANK, DIM, DIM, Float32(1.0) / Float32(RANK),
        _rand_list(rng, RANK * DIM, 0.001),
        _abs_list(_rand_list(rng, RANK * DIM, 0.0001)),
        _rand_list(rng, DIM * RANK, 0.001),
        _abs_list(_rand_list(rng, DIM * RANK, 0.0001)),
    )


def _mk_gradset(mut rng: _Lcg, n_adapters: Int) -> Wan22LoraGradSet:
    var d_a = List[List[Float32]]()
    var d_b = List[List[Float32]]()
    for i in range(n_adapters):
        if i == EMPTY_ADAPTER:
            d_a.append(List[Float32]())
            d_b.append(List[Float32]())
        else:
            d_a.append(_rand_list(rng, RANK * DIM, 0.005))
            d_b.append(_rand_list(rng, DIM * RANK, 0.005))
    return Wan22LoraGradSet(
        d_a^, d_b^, List[Float32](), List[Float32](), 0
    )


def _bf16_quanta(a: BFloat16, b: BFloat16) -> Int:
    var d = Int(a.to_bits[DType.uint16]()) - Int(b.to_bits[DType.uint16]())
    if d < 0:
        d = -d
    return d


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
            var ad = xf - yf
            if ad < Float32(0.0):
                ad = -ad
            if ad > worst_abs:
                worst_abs = ad
            var q = _bf16_quanta(x[i], y[i])
            if q > worst_quanta:
                worst_quanta = q


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


def _assert_bits_equal_bf16(name: String, x: List[BFloat16], y: List[BFloat16]) raises:
    if len(x) != len(y):
        raise Error(name + ": length mismatch")
    for i in range(len(x)):
        if Int(x[i].to_bits[DType.uint16]()) != Int(y[i].to_bits[DType.uint16]()):
            raise Error(name + ": empty-grad adapter param changed at " + String(i))


def _assert_bits_equal_f32(name: String, x: List[Float32], y: List[Float32]) raises:
    if len(x) != len(y):
        raise Error(name + ": length mismatch")
    for i in range(len(x)):
        if x[i].to_bits[DType.uint32]() != y[i].to_bits[DType.uint32]():
            raise Error(name + ": empty-grad adapter moment changed at " + String(i))


def main() raises:
    comptime if not has_accelerator():
        print("wan22_gpu_adamw_update_parity_smoke: GPU required")
        raise Error("no accelerator")
    else:
        comptime if not WAN22_GPU_ADAMW:
            print(
                "wan22_gpu_adamw_update_parity_smoke: WAN22_GPU_ADAMW=False —"
                " both sides would run the host loop, smoke vacuous"
            )
            raise Error("WAN22_GPU_ADAMW disabled")
        else:
            var ctx = DeviceContext()
            var rng = _Lcg(4242)

            # ── fixed fixture: identical adapters on host/gpu/init sides ──
            var n_adapters = NUM_BLOCKS * SLOTS
            var host_ads = List[LoraAdapter]()
            var gpu_ads = List[LoraAdapter]()
            var init_ads = List[LoraAdapter]()
            for _ in range(n_adapters):
                var ad = _mk_adapter(rng)
                host_ads.append(ad.copy())
                gpu_ads.append(ad.copy())
                init_ads.append(ad.copy())
            var host_lora = Wan22LoraSet(host_ads^, NUM_BLOCKS, RANK)
            var gpu_lora = Wan22LoraSet(gpu_ads^, NUM_BLOCKS, RANK)
            if wan22_total_adapters(host_lora) != n_adapters:
                raise Error("adapter count mismatch")

            var lr = Float32(3.0e-4)
            var beta1 = Float32(0.9)
            var beta2 = Float32(0.999)
            var eps = Float32(1.0e-8)
            var wd = Float32(0.01)

            # ── STEPS optimizer steps, fresh grads per step, same grads both
            # sides (one grad set, read-borrowed by both calls) ──
            for t in range(1, STEPS + 1):
                var gset = _mk_gradset(rng, n_adapters)
                wan22_lora_adamw_step_host(
                    host_lora, gset, t, lr, ctx, beta1, beta2, eps, wd
                )
                # WAN22_GPU_ADAMW=True (guarded above) → the fused GPU path.
                wan22_lora_adamw_step(
                    gpu_lora, gset, t, lr, ctx, beta1, beta2, eps, wd
                )

            # ── compare per element ──
            var p_worst_quanta = 0
            var p_worst_abs = Float32(0.0)
            var p_mism = 0
            var m_worst_abs = Float32(0.0)
            var m_mism = 0
            var v_worst_abs = Float32(0.0)
            var v_mism = 0
            var total_p = 0
            var total_mv = 0
            for i in range(n_adapters):
                total_p += len(host_lora.ad[i].a) + len(host_lora.ad[i].b)
                total_mv += len(host_lora.ad[i].ma) + len(host_lora.ad[i].mb)
                _cmp_bf16(
                    "a[" + String(i) + "]", host_lora.ad[i].a, gpu_lora.ad[i].a,
                    p_worst_quanta, p_worst_abs, p_mism,
                )
                _cmp_bf16(
                    "b[" + String(i) + "]", host_lora.ad[i].b, gpu_lora.ad[i].b,
                    p_worst_quanta, p_worst_abs, p_mism,
                )
                _cmp_f32(
                    "ma[" + String(i) + "]", host_lora.ad[i].ma, gpu_lora.ad[i].ma,
                    m_worst_abs, m_mism,
                )
                _cmp_f32(
                    "mb[" + String(i) + "]", host_lora.ad[i].mb, gpu_lora.ad[i].mb,
                    m_worst_abs, m_mism,
                )
                _cmp_f32(
                    "va[" + String(i) + "]", host_lora.ad[i].va, gpu_lora.ad[i].va,
                    v_worst_abs, v_mism,
                )
                _cmp_f32(
                    "vb[" + String(i) + "]", host_lora.ad[i].vb, gpu_lora.ad[i].vb,
                    v_worst_abs, v_mism,
                )

            # ── skip-empty gate: EMPTY_ADAPTER must be bit-identical to init
            # on BOTH paths ──
            _assert_bits_equal_bf16(
                "host.a", host_lora.ad[EMPTY_ADAPTER].a, init_ads[EMPTY_ADAPTER].a
            )
            _assert_bits_equal_bf16(
                "host.b", host_lora.ad[EMPTY_ADAPTER].b, init_ads[EMPTY_ADAPTER].b
            )
            _assert_bits_equal_f32(
                "host.ma", host_lora.ad[EMPTY_ADAPTER].ma, init_ads[EMPTY_ADAPTER].ma
            )
            _assert_bits_equal_f32(
                "host.va", host_lora.ad[EMPTY_ADAPTER].va, init_ads[EMPTY_ADAPTER].va
            )
            _assert_bits_equal_bf16(
                "gpu.a", gpu_lora.ad[EMPTY_ADAPTER].a, init_ads[EMPTY_ADAPTER].a
            )
            _assert_bits_equal_bf16(
                "gpu.b", gpu_lora.ad[EMPTY_ADAPTER].b, init_ads[EMPTY_ADAPTER].b
            )
            _assert_bits_equal_f32(
                "gpu.ma", gpu_lora.ad[EMPTY_ADAPTER].ma, init_ads[EMPTY_ADAPTER].ma
            )
            _assert_bits_equal_f32(
                "gpu.va", gpu_lora.ad[EMPTY_ADAPTER].va, init_ads[EMPTY_ADAPTER].va
            )

            # ── report + worst-case-tolerance bars (MJ-1017: NOT exact) ──
            print("params: mismatches=", p_mism, "/", total_p,
                  " worst_abs=", p_worst_abs,
                  " worst_bf16_quanta=", p_worst_quanta)
            print("first moments: mismatches=", m_mism, "/", total_mv,
                  " worst_abs=", m_worst_abs)
            print("second moments: mismatches=", v_mism, "/", total_mv,
                  " worst_abs=", v_worst_abs)
            if p_worst_quanta > 1:
                raise Error("params outside ±1 bf16 quantum worst-case bar")
            if m_worst_abs > MOMENT_ABS_BAR:
                raise Error("first moments outside 1e-6 worst-abs bar")
            if v_worst_abs > MOMENT_ABS_BAR:
                raise Error("second moments outside 1e-6 worst-abs bar")
            print(
                "wan22_gpu_adamw_update_parity_smoke: PASS (",
                STEPS, "steps,", n_adapters,
                "adapters, adapter", EMPTY_ADAPTER, "empty-grad untouched)",
            )

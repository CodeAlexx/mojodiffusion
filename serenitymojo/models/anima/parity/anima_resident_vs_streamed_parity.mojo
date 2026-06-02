# serenitymojo/models/anima/parity/anima_resident_vs_streamed_parity.mojo
#
# DEVICE-RESIDENT vs STREAMED parity gate for the Anima LoRA fast path.
#
# The device-resident path (anima_stack_lora_forward_device_resident /
# _backward_device_resident, all 28-block activations + LoRA A/B kept as DEVICE
# tensors) must compute the SAME thing as the proven host-streamed path
# (anima_stack_lora_forward_streamed / _backward_streamed). Speed must not change
# the math. This gate runs BOTH on identical inputs with a NON-ZERO LoRA B (so the
# LoRA branch is actually exercised — B=0 would make every adapter a no-op and
# hide a LoRA-grad bug) and compares:
#   * forward output (predicted flow)          cos >= 0.999
#   * every adapter d_A and d_B                 cos >= 0.999 (aggregate + worst)
#   * d_patches (load-bearing input grad)       cos >= 0.999
#   * d_t_silu (shared trained quantity)        cos >= 0.999
#
# Resident path uses F32 resident blocks + F32 device LoRA so it matches the F32
# streamed path bit-closely (BF16 base is a production-speed concern, separately
# gated by the trainer's loss curve, not this math gate).
#
# Run:
#   cd /home/alex/mojodiffusion
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#       serenitymojo/models/anima/parity/anima_resident_vs_streamed_parity.mojo \
#       -o /tmp/anima_resident_parity
#   /tmp/anima_resident_parity

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.math import log as flog, cos as fcos, sin as fsin, exp as fexp

from serenitymojo.parity import ParityHarness
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors

from serenitymojo.models.anima.config import anima
from serenitymojo.models.anima.weights import (
    AnimaBlockWeights, AnimaStackBase,
    load_anima_stack_base, load_anima_block_weights_f32, verify_anima_stack_shapes,
)
from serenitymojo.models.anima.lora_block import ANIMA_SLOTS
from serenitymojo.models.anima.anima_stack_lora import (
    AnimaLoraSet, AnimaLoraGrads, build_anima_lora_set,
    anima_stack_lora_forward_streamed, anima_stack_lora_backward_streamed,
    anima_stack_lora_forward_device_resident, anima_stack_lora_backward_device_resident,
    anima_lora_set_to_device,
)
from serenitymojo.models.dit.anima_contract import ANIMA_HIDDEN

comptime B = 1
comptime H = 16
comptime Dh = 128
comptime D = H * Dh
comptime JOINT = 1024
comptime F = 8192
comptime C = 16
comptime PS = 2
comptime IN_PATCH = (C + 1) * PS * PS
comptime OUT_PATCH = C * PS * PS
comptime EPS = Float32(1e-06)
comptime S_TXT = 512
comptime LATENT_HW = 16
comptime S_IMG = (LATENT_HW // PS) * (LATENT_HW // PS)   # 64
comptime L = 4                # gate depth (real blocks 0..L-1)
comptime RANK = 16
comptime ALPHA = Float32(16.0)


def _rng(n: Int, seed: UInt64, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    var s = seed
    for _ in range(n):
        s = s * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(s >> 40)) * Float32(1.0 / 16777216.0)
        out.append((u - Float32(0.5)) * scale)
    return out^


# seed every adapter B with small randn so the LoRA branch is non-degenerate.
def _seed_b(mut set: AnimaLoraSet, seed: UInt64):
    var s = seed
    for i in range(len(set.ad)):
        var nb = len(set.ad[i].b)
        var r = _rng(nb, s, 0.02)
        for j in range(nb):
            set.ad[i].b[j] = r[j]
        s += 17


def _ones(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n): o.append(1.0)
    return o^


def _zeros(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n): o.append(0.0)
    return o^


struct _Rope(Movable):
    var cos: Tensor
    var sin: Tensor
    def __init__(out self, var cos: Tensor, var sin: Tensor):
        self.cos = cos^; self.sin = sin^


def _rope_tables(s_img: Int, ctx: DeviceContext) raises -> _Rope:
    var half = Dh // 2
    var full_d = Dh
    var nh = LATENT_HW // PS
    var nw = LATENT_HW // PS
    var dim_h = full_d // 6 * 2
    var dim_w = dim_h
    var dim_t = full_d - 2 * dim_h
    var bins_t = dim_t // 2
    var bins_h = dim_h // 2
    var bins_w = dim_w // 2
    var base_theta: Float64 = 10000.0
    var h_exp = Float64(dim_h) / (Float64(dim_h) - 2.0)
    var w_exp = Float64(dim_w) / (Float64(dim_w) - 2.0)
    var theta_h = base_theta * fexp(flog(Float64(4.0)) * h_exp)
    var theta_w = base_theta * fexp(flog(Float64(4.0)) * w_exp)
    var theta_t = base_theta
    var ft = List[Float32]()
    for i in range(bins_t):
        ft.append(Float32(fexp(-flog(theta_t) * (Float64(2 * i) / Float64(dim_t)))))
    var fh = List[Float32]()
    for i in range(bins_h):
        fh.append(Float32(fexp(-flog(theta_h) * (Float64(2 * i) / Float64(dim_h)))))
    var fw = List[Float32]()
    for i in range(bins_w):
        fw.append(Float32(fexp(-flog(theta_w) * (Float64(2 * i) / Float64(dim_w)))))
    var cosl = List[Float32]()
    var sinl = List[Float32]()
    for _b in range(B):
        for ih in range(nh):
            for iw in range(nw):
                for _h in range(H):
                    for fi in range(bins_t):
                        cosl.append(fcos(Float32(0) * ft[fi])); sinl.append(fsin(Float32(0) * ft[fi]))
                    for fi in range(bins_h):
                        cosl.append(fcos(Float32(ih) * fh[fi])); sinl.append(fsin(Float32(ih) * fh[fi]))
                    for fi in range(bins_w):
                        cosl.append(fcos(Float32(iw) * fw[fi])); sinl.append(fsin(Float32(iw) * fw[fi]))
    var cos = Tensor.from_host(cosl, [B * s_img * H, half], STDtype.F32, ctx)
    var sin = Tensor.from_host(sinl, [B * s_img * H, half], STDtype.F32, ctx)
    return _Rope(cos^, sin^)


def _flatten(g: List[List[Float32]]) -> List[Float32]:
    var out = List[Float32]()
    for i in range(len(g)):
        for j in range(len(g[i])):
            out.append(g[i][j])
    return out^


def main() raises:
    var ctx = DeviceContext()
    print("==== anima_resident_vs_streamed_parity (device-resident vs streamed) ====")
    print("B=", B, " H=", H, " Dh=", Dh, " D=", D, " S_IMG=", S_IMG,
          " S_TXT=", S_TXT, " L=", L, " RANK=", RANK)

    var cfg = anima()
    var st = SafeTensors.open(cfg.checkpoint)
    verify_anima_stack_shapes(st, 28)
    var base = load_anima_stack_base(st, ctx)
    var blocks = List[AnimaBlockWeights]()
    for bi in range(L):
        blocks.append(load_anima_block_weights_f32(st, bi, ctx))
    print("loaded base +", L, "real blocks (F32 resident)")

    # ── fixed pseudo-random inputs ──
    var patches = _rng(B * S_IMG * IN_PATCH, UInt64(11), 1.0)
    var t_cond = _rng(B * ANIMA_HIDDEN, UInt64(22), 1.0)
    var base_adaln = _rng(B * 3 * D, UInt64(33), 0.5)
    var context = _rng(B * S_TXT * JOINT, UInt64(44), 1.0)
    var d_out = _rng(B * S_IMG * OUT_PATCH, UInt64(55), 1.0)
    var ropes = _rope_tables(S_IMG, ctx)

    # ── LoRA set with NON-ZERO B (exercise the LoRA branch) ──
    var lora = build_anima_lora_set(L, D, JOINT, F, RANK, ALPHA)
    _seed_b(lora, UInt64(7000))
    var lora_dev = anima_lora_set_to_device(lora, STDtype.F32, ctx)

    # ── STREAMED reference (host-list, proven path) ──
    var fwd_s = anima_stack_lora_forward_streamed[H, Dh, S_IMG, S_TXT](
        patches.copy(), t_cond.copy(), base_adaln.copy(), context.copy(),
        base, st, lora, ropes.cos, ropes.sin,
        B, D, JOINT, F, IN_PATCH, OUT_PATCH, EPS, ctx,
    )
    var grads_s = anima_stack_lora_backward_streamed[H, Dh, S_IMG, S_TXT](
        d_out.copy(), patches.copy(), t_cond.copy(), base_adaln.copy(), context.copy(),
        base, st, lora, ropes.cos, ropes.sin, fwd_s,
        B, D, JOINT, F, IN_PATCH, OUT_PATCH, EPS, ctx,
    )
    print("streamed forward+backward done")

    # ── DEVICE-RESIDENT path under test ──
    var fwd_r = anima_stack_lora_forward_device_resident[H, Dh, S_IMG, S_TXT](
        patches.copy(), t_cond.copy(), base_adaln.copy(), context.copy(),
        base, blocks, lora_dev, ropes.cos, ropes.sin,
        B, D, JOINT, F, IN_PATCH, OUT_PATCH, EPS, ctx,
    )
    var grads_r = anima_stack_lora_backward_device_resident[H, Dh, S_IMG, S_TXT](
        d_out.copy(), patches.copy(), t_cond.copy(), base_adaln.copy(), context.copy(),
        base, blocks, lora_dev, ropes.cos, ropes.sin, fwd_r,
        B, D, JOINT, F, IN_PATCH, OUT_PATCH, EPS, ctx, False,
    )
    print("device-resident forward+backward done")

    var harness = ParityHarness(Float64(0.999))
    var all_ok = True

    var ro = harness.compare_host(fwd_r.out.copy(), fwd_s.out.copy())
    print("  cos(out)       =", ro.cos, "  max_abs=", ro.max_abs,
          "  ", "PASS" if ro.passed else "FAIL")
    all_ok = all_ok and ro.passed

    var da = harness.compare_host(_flatten(grads_r.d_a), _flatten(grads_s.d_a))
    print("  cos(d_A all)   =", da.cos, "  max_abs=", da.max_abs,
          "  ", "PASS" if da.passed else "FAIL")
    all_ok = all_ok and da.passed

    var db = harness.compare_host(_flatten(grads_r.d_b), _flatten(grads_s.d_b))
    print("  cos(d_B all)   =", db.cos, "  max_abs=", db.max_abs,
          "  ", "PASS" if db.passed else "FAIL")
    all_ok = all_ok and db.passed

    var dp = harness.compare_host(grads_r.d_patches.copy(), grads_s.d_patches.copy())
    print("  cos(d_patches) =", dp.cos, "  max_abs=", dp.max_abs,
          "  ", "PASS" if dp.passed else "FAIL")
    all_ok = all_ok and dp.passed

    var dt = harness.compare_host(grads_r.d_t_silu.copy(), grads_s.d_t_silu.copy())
    print("  cos(d_t_silu)  =", dt.cos, "  max_abs=", dt.max_abs,
          "  ", "PASS" if dt.passed else "FAIL")
    all_ok = all_ok and dt.passed

    # worst per-adapter d_A/d_B cos (catch a single bad slot the aggregate hides).
    var worst = Float64(1.0)
    var worst_slot = -1
    for i in range(len(grads_r.d_a)):
        if len(grads_r.d_a[i]) == 0:
            continue
        var ca = harness.compare_host(grads_r.d_a[i].copy(), grads_s.d_a[i].copy())
        var cb = harness.compare_host(grads_r.d_b[i].copy(), grads_s.d_b[i].copy())
        if ca.cos < worst:
            worst = ca.cos; worst_slot = i
        if cb.cos < worst:
            worst = cb.cos; worst_slot = i
    print("  worst per-adapter cos =", worst, " at flat slot", worst_slot,
          "  ", "PASS" if worst >= 0.999 else "FAIL")
    all_ok = all_ok and (worst >= 0.999)

    print("  resident nonfinite =", grads_r.nonfinite_lora_grads,
          "  streamed nonfinite =", grads_s.nonfinite_lora_grads)

    if all_ok and grads_r.nonfinite_lora_grads == 0:
        print("VERDICT: PASS — device-resident == streamed (out + all LoRA grads + d_patches + d_t_silu, cos>=0.999)")
    else:
        print("VERDICT: FAIL")

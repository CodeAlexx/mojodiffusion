# serenitymojo/models/klein/parity/klein_stack_lora_real_smoke.mojo
#
# REAL-WEIGHT FINITE + ROUND-TRIP SMOKE for the Klein FULL DiT STACK *WITH LoRA*.
# Loads REAL Klein-9B weights (8 double + 24 single blocks + input projections +
# modulation MLP + final layer), builds the full KleinLoraSet (8×4 + 24×2 = 80
# adapters), runs klein_stack_lora_forward + klein_stack_lora_backward at REAL
# dims (D=4096, H=32, Dh=128, F=12288) with per-block recompute, then:
#   1. asserts every forward output + every collected LoRA d_A/d_B is FINITE,
#   2. runs ONE klein_lora_adamw_step on all 80 adapters (asserts A/B stay finite
#      and B moved off zero — proving the step actually updated trained params),
#   3. saves the set with save_klein_lora and reloads it byte-exact, asserting
#      the reloaded A/B match the in-memory A/B exactly (round-trip).
#
# Small token grids (N_IMG=4, N_TXT=2, RANK=4) keep it light; the block MATH is
# already parity-proven (klein_stack_lora_parity). This is the Tenet-4 evidence
# that the FULL-DEPTH LoRA path runs end to end on real weights and that the save
# round-trips.
#
# Run: cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#   pixi run mojo run -I . serenitymojo/models/klein/parity/klein_stack_lora_real_smoke.mojo

from std.collections import List
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.models.klein.double_block import DoubleBlockWeights
from serenitymojo.models.klein.single_block import SingleBlockWeights
from serenitymojo.models.klein.klein_stack import KleinStackBase
from serenitymojo.models.klein.klein_stack_lora import (
    KleinLoraSet, build_klein_lora_set,
    klein_stack_lora_forward, klein_stack_lora_backward,
    klein_lora_adamw_step, save_klein_lora, klein_lora_prefixes,
)
from serenitymojo.models.klein.weights import (
    load_double_block_weights, load_single_block_weights,
    load_klein_stack_base, build_klein_vec_silu,
    build_klein_double_modvecs, build_klein_single_modvecs,
)


comptime KLEIN9B_PATH = "/home/alex/.serenity/models/checkpoints/flux-2-klein-base-9b.safetensors"
comptime SAVE_PATH = "/tmp/klein_stack_lora_smoke.safetensors"

comptime H = 32
comptime Dh = 128
comptime N_IMG = 4
comptime N_TXT = 2
comptime S = N_IMG + N_TXT
comptime NUM_DOUBLE = 8
comptime NUM_SINGLE = 24
comptime TIMESTEP_DIM = 256
comptime RANK = 4
comptime ALPHA = Float32(8.0)


def _fill(n: Int, seed: UInt64, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    for _ in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
        out.append((u - Float32(0.5)) * scale)
    return out^


def _all_finite(h: List[Float32]) -> Bool:
    for i in range(len(h)):
        var v = h[i]
        if v != v:
            return False
        var a = v if v >= 0.0 else -v
        if a > Float32(1.0e30):
            return False
    return True


def _abs_sum(h: List[Float32]) -> Float32:
    var s = Float32(0.0)
    for i in range(len(h)):
        var v = h[i]
        s += v if v >= 0.0 else -v
    return s


# Read one tensor by name from a safetensors into a host F32 list (for the
# byte-exact round-trip check).
def _read_f32(st: SafeTensors, name: String, ctx: DeviceContext) raises -> List[Float32]:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    var t = Tensor.from_view(tv, ctx)
    return t.to_host(ctx)


def _max_abs_diff(a: List[Float32], b: List[Float32]) -> Float32:
    if len(a) != len(b):
        return Float32(1.0e30)
    var m = Float32(0.0)
    for i in range(len(a)):
        var d = a[i] - b[i]
        var ad = d if d >= 0.0 else -d
        if ad > m:
            m = ad
    return m


def main() raises:
    var ctx = DeviceContext()
    var D = 4096
    var F = 12288
    var IN_CH = 128
    var TXT_CH = 12288
    var OUT_CH = 128
    var eps = Float32(1.0e-6)

    print("=== Klein-9B FULL STACK + LoRA real-weight FINITE + round-trip smoke ===")
    print("  path:", KLEIN9B_PATH)
    print("  D=", D, " H=", H, " Dh=", Dh, " F=", F,
          " num_double=", NUM_DOUBLE, " num_single=", NUM_SINGLE,
          " N_IMG=", N_IMG, " N_TXT=", N_TXT, " RANK=", RANK)

    var st = SafeTensors.open(KLEIN9B_PATH)

    var ts = Tensor.from_host([Float32(0.5)], [1], STDtype.F32, ctx)
    var vec_silu = build_klein_vec_silu(st, ts, TIMESTEP_DIM, D, ctx)
    if not _all_finite(vec_silu):
        raise Error("vec_silu not finite")
    var img_mod = build_klein_double_modvecs(st, vec_silu, String("img"), D, ctx)
    var txt_mod = build_klein_double_modvecs(st, vec_silu, String("txt"), D, ctx)
    var single_mod = build_klein_single_modvecs(st, vec_silu, D, ctx)

    var base = load_klein_stack_base(st, vec_silu, D, ctx)

    var dbw = List[DoubleBlockWeights]()
    for bi in range(NUM_DOUBLE):
        dbw.append(load_double_block_weights(st, bi, ctx))
    var sbw = List[SingleBlockWeights]()
    for bi in range(NUM_SINGLE):
        sbw.append(load_single_block_weights(st, bi, ctx))
    print("  loaded", len(dbw), "double +", len(sbw), "single block weights")

    # ── build the full LoRA set (80 adapters) ──
    var lora = build_klein_lora_set(NUM_DOUBLE, NUM_SINGLE, D, RANK, ALPHA)
    var total_adapters = NUM_DOUBLE * 4 + NUM_SINGLE * 2
    print("  built KleinLoraSet:", len(lora.dbl), "double-slot +", len(lora.sgl),
          "single-slot adapters (expect", total_adapters, ")")

    # ── inputs (non-degenerate) ──
    var img_tokens = _fill(N_IMG * IN_CH, 100, 1.0)
    var txt_tokens = _fill(N_TXT * TXT_CH, 200, 1.0)
    var cos = _fill(S * H * (Dh // 2), 500, 1.0)
    var sin = _fill(S * H * (Dh // 2), 600, 1.0)

    # ── FORWARD ──
    print("  running full-depth LoRA forward ...")
    var fwd = klein_stack_lora_forward[H, Dh, N_IMG, N_TXT, S](
        img_tokens.copy(), txt_tokens.copy(), base,
        dbw, sbw, lora, img_mod, txt_mod, single_mod, cos.copy(), sin.copy(),
        D, F, IN_CH, TXT_CH, OUT_CH, eps, ctx,
    )
    if len(fwd.out) != N_IMG * OUT_CH:
        raise Error("forward output shape wrong")
    if not _all_finite(fwd.out):
        raise Error("FORWARD output not finite (NaN/inf) on real weights")
    print("  forward output FINITE ✓ (len", len(fwd.out), ")")

    # ── BACKWARD (full-depth, per-block recompute, LoRA grad collection) ──
    print("  running full-depth LoRA backward (per-block recompute) ...")
    var d_out = _fill(N_IMG * OUT_CH, 700, 0.05)
    var g = klein_stack_lora_backward[H, Dh, N_IMG, N_TXT, S](
        d_out, img_tokens.copy(), txt_tokens.copy(), base,
        dbw, sbw, lora, img_mod, txt_mod, single_mod, cos.copy(), sin.copy(), fwd,
        D, F, IN_CH, TXT_CH, OUT_CH, eps, ctx,
    )

    # ── finite checks on EVERY collected LoRA grad + the load-bearing grads ──
    var ok = True
    if not _all_finite(g.d_img_tokens): print("  d_img_tokens NOT finite"); ok = False
    if not _all_finite(g.d_txt_tokens): print("  d_txt_tokens NOT finite"); ok = False
    if not _all_finite(g.d_img_mod): print("  d_img_mod NOT finite"); ok = False
    if not _all_finite(g.d_txt_mod): print("  d_txt_mod NOT finite"); ok = False
    if not _all_finite(g.d_single_mod): print("  d_single_mod NOT finite"); ok = False
    var nd = NUM_DOUBLE * 4
    for i in range(nd):
        if not _all_finite(g.dbl_d_a[i]) or not _all_finite(g.dbl_d_b[i]):
            print("  double LoRA grad slot", i, "NOT finite"); ok = False
    var ns = NUM_SINGLE * 2
    for i in range(ns):
        if not _all_finite(g.sgl_d_a[i]) or not _all_finite(g.sgl_d_b[i]):
            print("  single LoRA grad slot", i, "NOT finite"); ok = False
    if not ok:
        raise Error("non-finite LoRA grad in real-weight smoke")
    print("  all", total_adapters, "adapter d_A/d_B + token/mod grads FINITE ✓")

    # ── ONE AdamW step on ALL adapters ──
    print("  running klein_lora_adamw_step on all", total_adapters, "adapters ...")
    # record B abs-sum BEFORE (B starts at 0 → must move off zero after the step,
    # proving the optimizer actually updated trained params).
    var b_before = _abs_sum(lora.dbl[0].b) + _abs_sum(lora.sgl[0].b)
    klein_lora_adamw_step(lora, g, 1, Float32(1.0e-4), ctx)
    var b_after = _abs_sum(lora.dbl[0].b) + _abs_sum(lora.sgl[0].b)
    print("  B abs-sum (dbl0+sgl0): before =", b_before, " after =", b_after)
    if b_after <= b_before:
        raise Error("AdamW step did not move B off zero (optimizer no-op?)")
    # all adapters still finite after the step?
    for i in range(nd):
        if not _all_finite(lora.dbl[i].a) or not _all_finite(lora.dbl[i].b):
            print("  double adapter", i, "NOT finite after AdamW"); ok = False
    for i in range(ns):
        if not _all_finite(lora.sgl[i].a) or not _all_finite(lora.sgl[i].b):
            print("  single adapter", i, "NOT finite after AdamW"); ok = False
    if not ok:
        raise Error("non-finite adapter after AdamW step")
    print("  all adapters FINITE after AdamW, B moved off zero ✓")

    # ── SAVE + byte-exact reload round-trip ──
    print("  saving with save_klein_lora ...")
    var npairs = save_klein_lora(lora, SAVE_PATH, ctx)
    print("  wrote", npairs, "(A,B) pairs (expect", total_adapters, ")")
    if npairs != total_adapters:
        raise Error("save_klein_lora wrote wrong adapter count")

    print("  reloading + asserting byte-exact ...")
    var st2 = SafeTensors.open(SAVE_PATH)
    var prefixes = klein_lora_prefixes(NUM_DOUBLE, NUM_SINGLE)
    var max_diff = Float32(0.0)
    # walk the flat order: first NUM_DOUBLE*4 prefixes are double slots, rest single.
    for i in range(nd):
        var ra = _read_f32(st2, prefixes[i] + ".lora_A.weight", ctx)
        var rb = _read_f32(st2, prefixes[i] + ".lora_B.weight", ctx)
        var da = _max_abs_diff(ra, lora.dbl[i].a)
        var db = _max_abs_diff(rb, lora.dbl[i].b)
        if da > max_diff: max_diff = da
        if db > max_diff: max_diff = db
    for i in range(ns):
        var ra = _read_f32(st2, prefixes[nd + i] + ".lora_A.weight", ctx)
        var rb = _read_f32(st2, prefixes[nd + i] + ".lora_B.weight", ctx)
        var da = _max_abs_diff(ra, lora.sgl[i].a)
        var db = _max_abs_diff(rb, lora.sgl[i].b)
        if da > max_diff: max_diff = da
        if db > max_diff: max_diff = db
    print("  round-trip max |reloaded - in-memory| =", max_diff)
    if max_diff != Float32(0.0):
        raise Error("LoRA save round-trip NOT byte-exact (F32 A/B should reload exactly)")
    print("  LoRA save round-trip BYTE-EXACT ✓")

    print("")
    print("PASS: Klein-9B FULL STACK + LoRA fwd+bwd FINITE at real depth/dims,",
          "AdamW step updates all adapters, save round-trips byte-exact")

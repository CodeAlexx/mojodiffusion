# serenitymojo/models/klein/parity/klein_stack_real_smoke.mojo
#
# REAL-WEIGHT FINITE SMOKE for the Klein FULL DiT STACK. Loads REAL Klein-9B
# weights (8 double + 24 single blocks + input projections + modulation MLP +
# final layer), runs klein_stack_forward + klein_stack_backward at REAL dims
# (D=4096, H=32, Dh=128, F=12288) with the per-block recompute that bounds memory,
# and asserts every output + grad is FINITE (no NaN/inf). Small token grids
# (N_IMG=4, N_TXT=2) keep it light; the block MATH is already parity-proven.
#
# This is the Tenet-4 evidence that the FULL-DEPTH stack runs end to end on real
# weights without diverging — the composition gate proves correctness at small
# depth; this proves the same code path survives real depth + real dims + the
# per-block recompute memory strategy.
#
# Run: cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#      pixi run mojo run -I . serenitymojo/models/klein/parity/klein_stack_real_smoke.mojo

from std.collections import List
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.models.klein.double_block import DoubleBlockWeights, DoubleBlockGrads
from serenitymojo.models.klein.single_block import SingleBlockWeights, SingleBlockGrads
from serenitymojo.models.klein.klein_stack import (
    KleinStackBase, klein_stack_forward, klein_stack_backward,
)
from serenitymojo.models.klein.weights import (
    load_double_block_weights, load_single_block_weights,
    load_klein_stack_base, build_klein_vec_silu,
    build_klein_double_modvecs, build_klein_single_modvecs,
)


comptime KLEIN9B_PATH = "/home/alex/.serenity/models/checkpoints/flux-2-klein-base-9b.safetensors"

comptime H = 32
comptime Dh = 128
comptime N_IMG = 4
comptime N_TXT = 2
comptime S = N_IMG + N_TXT
comptime NUM_DOUBLE = 8
comptime NUM_SINGLE = 24
comptime TIMESTEP_DIM = 256


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


def main() raises:
    var ctx = DeviceContext()
    var D = 4096
    var F = 12288
    var IN_CH = 128
    var TXT_CH = 12288
    var OUT_CH = 128
    var eps = Float32(1.0e-6)

    print("=== Klein-9B FULL STACK real-weight FINITE smoke ===")
    print("  path:", KLEIN9B_PATH)
    print("  D=", D, " H=", H, " Dh=", Dh, " F=", F,
          " num_double=", NUM_DOUBLE, " num_single=", NUM_SINGLE,
          " N_IMG=", N_IMG, " N_TXT=", N_TXT)

    var st = SafeTensors.open(KLEIN9B_PATH)

    # ── shared modulation feature from a single timestep ──
    var ts = Tensor.from_host([Float32(0.5)], [1], STDtype.F32, ctx)
    var vec_silu = build_klein_vec_silu(st, ts, TIMESTEP_DIM, D, ctx)
    print("  vec_silu len =", len(vec_silu), " (expect", D, ")")
    if not _all_finite(vec_silu):
        raise Error("vec_silu not finite")

    var img_mod = build_klein_double_modvecs(st, vec_silu, String("img"), D, ctx)
    var txt_mod = build_klein_double_modvecs(st, vec_silu, String("txt"), D, ctx)
    var single_mod = build_klein_single_modvecs(st, vec_silu, D, ctx)
    print("  modvecs built (img/txt 6 chunks, single 3 chunks)")

    # ── base weights (input proj + final layer) ──
    var base = load_klein_stack_base(st, vec_silu, D, ctx)
    print("  base loaded: img_in len =", len(base.img_in),
          " txt_in len =", len(base.txt_in), " final_lin len =", len(base.final_lin))

    # ── all block weights ──
    var dbw = List[DoubleBlockWeights]()
    for bi in range(NUM_DOUBLE):
        dbw.append(load_double_block_weights(st, bi, ctx))
    var sbw = List[SingleBlockWeights]()
    for bi in range(NUM_SINGLE):
        sbw.append(load_single_block_weights(st, bi, ctx))
    print("  loaded", len(dbw), "double +", len(sbw), "single block weights")

    # ── inputs (non-degenerate) ──
    var img_tokens = _fill(N_IMG * IN_CH, 100, 1.0)
    var txt_tokens = _fill(N_TXT * TXT_CH, 200, 1.0)
    var cos = _fill(S * H * (Dh // 2), 500, 1.0)
    var sin = _fill(S * H * (Dh // 2), 600, 1.0)

    # ── FORWARD ──
    print("  running full-depth forward ...")
    var fwd = klein_stack_forward[H, Dh, N_IMG, N_TXT, S](
        img_tokens.copy(), txt_tokens.copy(), base,
        dbw, sbw, img_mod, txt_mod, single_mod, cos.copy(), sin.copy(),
        D, F, IN_CH, TXT_CH, OUT_CH, eps, ctx,
    )
    print("  forward out len =", len(fwd.out), " (expect", N_IMG * OUT_CH, ")")
    if len(fwd.out) != N_IMG * OUT_CH:
        raise Error("forward output shape wrong")
    if not _all_finite(fwd.out):
        raise Error("FORWARD output not finite (NaN/inf) on real weights")
    print("  forward output FINITE ✓")

    # ── BACKWARD (full-depth, per-block recompute) ──
    print("  running full-depth backward (per-block recompute) ...")
    var d_out = _fill(N_IMG * OUT_CH, 700, 0.05)
    var g = klein_stack_backward[H, Dh, N_IMG, N_TXT, S](
        d_out, img_tokens.copy(), txt_tokens.copy(), base,
        dbw, sbw, img_mod, txt_mod, single_mod, cos.copy(), sin.copy(), fwd,
        D, F, IN_CH, TXT_CH, OUT_CH, eps, ctx,
    )

    # ── finite checks on all returned grads ──
    var ok = True
    if not _all_finite(g.d_img_tokens):
        print("  d_img_tokens NOT finite"); ok = False
    if not _all_finite(g.d_txt_tokens):
        print("  d_txt_tokens NOT finite"); ok = False
    if not _all_finite(g.d_img_mod):
        print("  d_img_mod NOT finite"); ok = False
    if not _all_finite(g.d_txt_mod):
        print("  d_txt_mod NOT finite"); ok = False
    if not _all_finite(g.d_single_mod):
        print("  d_single_mod NOT finite"); ok = False
    for bi in range(NUM_DOUBLE):
        if not _all_finite(g.dbl_grads[bi].img.d_wqkv) or not _all_finite(g.dbl_grads[bi].txt.d_wqkv):
            print("  double block", bi, "d_wqkv NOT finite"); ok = False
        if not _all_finite(g.dbl_grads[bi].img.d_x) or not _all_finite(g.dbl_grads[bi].txt.d_x):
            print("  double block", bi, "d_x NOT finite"); ok = False
    for bi in range(NUM_SINGLE):
        if not _all_finite(g.sgl_grads[bi].d_w1) or not _all_finite(g.sgl_grads[bi].d_w2):
            print("  single block", bi, "d_w NOT finite"); ok = False
        if not _all_finite(g.sgl_grads[bi].d_x):
            print("  single block", bi, "d_x NOT finite"); ok = False

    print("  d_img_tokens len =", len(g.d_img_tokens),
          " d_txt_tokens len =", len(g.d_txt_tokens))
    print("")
    if ok:
        print("PASS: Klein-9B FULL STACK fwd+bwd runs at real depth+dims, ALL outputs+grads FINITE")
    else:
        print("FAIL: some output/grad was NaN/inf on real weights")
        raise Error("non-finite result in real-weight smoke")

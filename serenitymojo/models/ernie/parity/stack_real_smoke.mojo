# serenitymojo/models/ernie/parity/stack_real_smoke.mojo
#
# REAL-DEPTH finiteness + memory smoke for the ERNIE-Image FULL STACK
# (models/ernie/ernie_stack.mojo) on the REAL 36-layer checkpoint at REAL hidden
# dims (D=4096, H=32, Dh=128, F=12288). Loads every block + the base projections
# from the real sharded safetensors (E1 weight loader), runs the full 36-layer
# fwd + full-depth bwd (per-block recompute), and asserts ALL outputs + a sample
# of grads are FINITE with NO OOM. Reports peak GPU memory + thermal.
#
# Sequence is REDUCED (small image grid + few text tokens) so the per-block SDPA
# scores [H, S, S] stay modest on the shared 3090; hidden/H/Dh/F are the REAL
# values, so this exercises the real 36-block composition at real per-op cost.
# This is NOT a parity gate (no torch oracle) — it is the OOM/finite smoke
# (deliverable 4). The shared modulation + f_scale/f_shift are synthesized at a
# believable finite scale (the real timestep/adaLN MLP source is the E5 link).
#
# Run (SEPARATE command, after build):
#   cd /home/alex/mojodiffusion
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build -I . -Xlinker -lm \
#       serenitymojo/models/ernie/parity/stack_real_smoke.mojo -o /tmp/ernie_stack_real_smoke
#   /tmp/ernie_stack_real_smoke

from std.gpu.host import DeviceContext
from std.collections import List
from std.math import sin as fsin, cos as fcos, isfinite
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.models.dit.ernie_contract import ERNIE_TRANSFORMER_DIR
from serenitymojo.models.dit.ernie_image import build_ernie_rope_tables
from serenitymojo.models.ernie.weights import (
    ErnieBlockWeights, ErnieStackBase,
    load_ernie_stack_base, verify_ernie_stack_shapes,
)
from serenitymojo.models.ernie.block import ErnieModVecs
from serenitymojo.models.ernie.ernie_stack import (
    ErnieStackForward, ErnieStackGradsLite,
    ernie_stack_forward_streamed, ernie_stack_backward_streamed,
)


# REAL ERNIE dims
comptime H = 32
comptime Dh = 128
comptime D = H * Dh         # 4096
comptime F = 12288          # REAL FFN hidden
comptime IN_CH = 128        # REAL latent channels
comptime TEXT_IN = 3072     # REAL Mistral hidden (text_in_dim)
comptime OUT_CH = 128       # REAL out channels
comptime NUM_LAYERS = 36    # REAL depth

# REDUCED sequence to keep SDPA scores [H,S,S] bounded on a shared 3090.
comptime IMG_H = 8
comptime IMG_W = 8
comptime N_IMG = IMG_H * IMG_W   # 64
comptime N_TXT = 16
comptime S = N_IMG + N_TXT       # 80
comptime EPS = Float32(1e-06)


def _synth(n: Int, a: Float32, b: Float32, c: Float32) -> List[Float32]:
    var o = List[Float32]()
    for i in range(n):
        o.append(fsin(a * Float32(i) + b) * c)
    return o^


def _synthc(n: Int, a: Float32, b: Float32, c: Float32) -> List[Float32]:
    var o = List[Float32]()
    for i in range(n):
        o.append(fcos(a * Float32(i) + b) * c)
    return o^


def _count_nonfinite(v: List[Float32]) -> Int:
    var bad = 0
    for i in range(len(v)):
        if not isfinite(v[i]):
            bad += 1
    return bad


def main() raises:
    var ctx = DeviceContext()
    print("==== ernie stack_real_smoke (36-layer REAL weights, finite + mem) ====")
    print("D=", D, " H=", H, " Dh=", Dh, " F=", F, " NUM_LAYERS=", NUM_LAYERS)
    print("N_IMG=", N_IMG, " N_TXT=", N_TXT, " S=", S, " (reduced seq, real hidden)")

    var mem0 = ctx.get_memory_info()
    var total = mem0[1]
    var min_free = mem0[0]
    print("free VRAM at start (bytes):", mem0[0], " total:", total)

    # ── open the real sharded checkpoint ──
    var st = ShardedSafeTensors.open(String(ERNIE_TRANSFORMER_DIR))
    print("opened transformer: num_shards =", st.num_shards(),
          " num_tensors =", st.num_tensors())

    # ── E1: verify a sample of real tensors load at the right shapes (RC=0) ──
    print("")
    print("---- weight-loader shape verification (sample) ----")
    verify_ernie_stack_shapes(st, NUM_LAYERS, D, Dh, F, IN_CH, TEXT_IN, OUT_CH)

    # ── load base (resident); blocks are STREAMED per layer (offload budget) ──
    # Full F32 residency of 36 blocks is ~31 GB > 24 GB; the streamed stack swaps
    # each block in on demand (load -> use -> drop), bounding resident weights to
    # ~one block. Mirrors the Rust ErnieImageSwapped/BlockOffloader contract.
    print("")
    print("loading base (input projections + final layer) ...")
    var base = load_ernie_stack_base(st, D, IN_CH, ctx)
    ctx.synchronize()
    var mem_loaded = ctx.get_memory_info()
    if mem_loaded[0] < min_free:
        min_free = mem_loaded[0]
    print("free VRAM after base load (bytes):", mem_loaded[0],
          " used by base:", mem0[0] - mem_loaded[0])
    print("blocks are STREAMED per layer (resident weights bounded to ~1 block)")

    # ── synthesize finite inputs + shared modulation + final-layer mod ──
    var img_tokens = _synth(N_IMG * IN_CH, 0.021, 0.05, 0.5)
    var txt_tokens = _synth(N_TXT * TEXT_IN, 0.018, 0.07, 0.4)
    var mv = ErnieModVecs(
        _synth(D, 0.013, 0.10, 0.30), _synthc(D, 0.017, 0.20, 0.20),
        _synth(D, 0.011, 0.30, 0.40), _synth(D, 0.015, 0.40, 0.25),
        _synthc(D, 0.019, 0.50, 0.15), _synth(D, 0.012, 0.60, 0.35),
    )
    var f_scale = _synthc(D, 0.015, 0.30, 0.10)
    var f_shift = _synth(D, 0.014, 0.40, 0.10)

    # REAL 3-axis interleaved-doubled half-split rope tables (the fixed primitive).
    var rope = build_ernie_rope_tables[N_IMG, N_TXT, H, Dh](
        IMG_H, IMG_W, N_TXT, ctx, STDtype.F32
    )

    # ── full 36-layer forward ──
    print("")
    print("running full", NUM_LAYERS, "-layer forward ...")
    var fwd = ernie_stack_forward_streamed[H, Dh, N_IMG, N_TXT, S](
        img_tokens.copy(), txt_tokens.copy(), base, st, NUM_LAYERS, mv,
        f_scale.copy(), f_shift.copy(), rope[0], rope[1],
        D, F, IN_CH, TEXT_IN, OUT_CH, EPS, ctx,
    )
    ctx.synchronize()
    var mem_fwd = ctx.get_memory_info()
    if mem_fwd[0] < min_free:
        min_free = mem_fwd[0]
    var out_bad = _count_nonfinite(fwd.out)
    print("forward out [N_IMG,out_ch] n =", len(fwd.out),
          " non-finite =", out_bad)
    print("free VRAM after forward (bytes):", mem_fwd[0])

    # ── full-depth backward (per-block recompute) ──
    print("")
    print("running full", NUM_LAYERS, "-layer backward (per-block recompute) ...")
    var d_out = _synth(N_IMG * OUT_CH, 0.027, 0.11, 0.05)
    var g = ernie_stack_backward_streamed[H, Dh, N_IMG, N_TXT, S](
        d_out, img_tokens.copy(), txt_tokens.copy(), base, st, NUM_LAYERS, mv,
        f_scale.copy(), f_shift.copy(), rope[0], rope[1], fwd,
        D, F, IN_CH, TEXT_IN, OUT_CH, EPS, ctx,
    )
    ctx.synchronize()
    var mem_bwd = ctx.get_memory_info()
    if mem_bwd[0] < min_free:
        min_free = mem_bwd[0]

    # ── finiteness checks on all returned host grads ──
    var bad_img = _count_nonfinite(g.d_img_tokens)
    var bad_txt = _count_nonfinite(g.d_txt_tokens)
    var bad_fs = _count_nonfinite(g.d_f_scale)
    var bad_fsh = _count_nonfinite(g.d_f_shift)
    var bad_fl = _count_nonfinite(g.d_final_lin)
    var bad_shared = _count_nonfinite(g.d_shared_mod)
    # per-block weight grads (deepest + shallowest)
    var bad_wq_deep = _count_nonfinite(g.d_wq_deep)
    var bad_wdown_deep = _count_nonfinite(g.d_wdown_deep)
    var bad_wq_shal = _count_nonfinite(g.d_wq_shallow)
    var bad_wdown_shal = _count_nonfinite(g.d_wdown_shallow)

    print("")
    print("---- backward grad finiteness (non-finite counts; 0 = all finite) ----")
    print("  d_img_tokens :", bad_img)
    print("  d_txt_tokens :", bad_txt)
    print("  d_f_scale    :", bad_fs)
    print("  d_f_shift    :", bad_fsh)
    print("  d_final_lin  :", bad_fl)
    print("  d_shared_mod :", bad_shared)
    print("  d_wq (deep)  :", bad_wq_deep)
    print("  d_wdown(deep):", bad_wdown_deep)
    print("  d_wq (shal)  :", bad_wq_shal)
    print("  d_wdown(shal):", bad_wdown_shal)
    print("  ALL 36 blocks' full grad sets non-finite count:", g.nonfinite_block_grads)

    var total_bad = (out_bad + bad_img + bad_txt + bad_fs + bad_fsh + bad_fl
        + bad_shared + bad_wq_deep + bad_wdown_deep + bad_wq_shal + bad_wdown_shal
        + g.nonfinite_block_grads)

    var peak_used = total - min_free
    print("")
    print("---- memory ----")
    print("  peak GPU mem used (bytes):", peak_used,
          " (", Float64(peak_used) / (1024.0 * 1024.0 * 1024.0), "GiB )")
    print("  total GPU mem (bytes):", total)

    print("")
    if total_bad == 0:
        print("VERDICT: PASS — 36-layer REAL fwd+bwd ALL FINITE, NO OOM. peak",
              Float64(peak_used) / (1024.0 * 1024.0 * 1024.0), "GiB")
    else:
        print("VERDICT: FAIL — non-finite values present (total", total_bad, ")")

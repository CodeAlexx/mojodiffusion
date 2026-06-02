# serenitymojo/models/anima/parity/stack_real_smoke.mojo
#
# REAL-DEPTH finiteness + memory smoke for the ANIMA FULL 28-block STACK
# (models/anima/anima_stack.mojo) on the REAL anima-base-v1.0.safetensors at REAL
# hidden dims (D=2048, H=16, Dh=128, F=8192). Loads every block (F32, resident) +
# the base projections, runs the full 28-block fwd + full-depth bwd (per-block
# recompute), and asserts ALL outputs + grads are FINITE with NO OOM. Reports peak
# GPU memory.
#
# Sequence is REDUCED (small image grid) so the per-block SDPA scores stay modest
# on the shared 3090; hidden/H/Dh/F are the REAL values, so this exercises the
# real 28-block composition at real per-op cost. NOT a parity gate (no oracle) —
# the OOM/finite smoke (deliverable 5). t_cond / base_adaln / context are
# synthesized at a believable finite scale (the real t_embedder / text-adapter
# source is the data-path phase D).
#
# Run (SEPARATE command, after build):
#   cd /home/alex/mojodiffusion
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build -I . -Xlinker -lm \
#       serenitymojo/models/anima/parity/stack_real_smoke.mojo -o /tmp/anima_real_smoke
#   /tmp/anima_real_smoke

from std.gpu.host import DeviceContext
from std.collections import List
from std.math import sin as fsin, cos as fcos, isfinite
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.models.anima.config import anima
from serenitymojo.models.anima.weights import (
    AnimaBlockWeights, AnimaStackBase,
    load_anima_all_blocks_f32, load_anima_stack_base, verify_anima_stack_shapes,
)
from serenitymojo.models.anima.anima_stack import (
    AnimaStackForward, AnimaStackGrads, anima_stack_forward, anima_stack_backward,
)
from serenitymojo.models.dit.anima_contract import (
    ANIMA_HIDDEN, ANIMA_NUM_HEADS, ANIMA_HEAD_DIM, ANIMA_DEPTH,
)


# REAL Anima dims
comptime B = 1
comptime H = ANIMA_NUM_HEADS     # 16
comptime Dh = ANIMA_HEAD_DIM     # 128
comptime D = ANIMA_HIDDEN        # 2048
comptime F = 8192                # REAL GELU MLP hidden
comptime JOINT = 1024            # cross-attn context dim
comptime IN_PATCH = 68           # (16+1)*2*2
comptime OUT_PATCH = 64          # 16*2*2

# REDUCED sequence: T=1, 16x16 latent, patch 2x2 -> nh=nw=8 -> S_IMG = 64.
comptime S_IMG = 64
comptime S_TXT = 16              # short text context for the smoke
comptime EPS = Float32(1e-06)
comptime MIB = 1024.0 * 1024.0
comptime GIB = MIB * 1024.0


def _synth(n: Int, a: Float32, b: Float32, c: Float32) -> List[Float32]:
    var o = List[Float32]()
    for i in range(n):
        o.append(fsin(a * Float32(i) + b) * c)
    return o^


def _count_nonfinite(v: List[Float32]) -> Int:
    var bad = 0
    for i in range(len(v)):
        if not isfinite(v[i]):
            bad += 1
    return bad


def main() raises:
    var ctx = DeviceContext()
    print("==== anima stack_real_smoke (28-layer REAL weights, finite + mem) ====")
    print("D=", D, " H=", H, " Dh=", Dh, " F=", F, " DEPTH=", ANIMA_DEPTH)
    print("S_IMG=", S_IMG, " S_TXT=", S_TXT, " (reduced seq, REAL hidden)")

    var mem0 = ctx.get_memory_info()
    var total = Int(mem0[1])
    var min_free = Int(mem0[0])
    print("free VRAM at start:", Int(Float64(min_free) / MIB), "MiB  total:",
          Int(Float64(total) / MIB), "MiB")

    var cfg = anima()
    print("opening checkpoint:", cfg.checkpoint)
    var st = SafeTensors.open(cfg.checkpoint)

    # ── E1: verify a sample of real tensors load at expected shapes (RC=0) ──
    print("")
    print("---- weight-loader shape verification (base + block 0 + block 27) ----")
    verify_anima_stack_shapes(st, ANIMA_DEPTH)

    # ── load base + ALL 28 blocks resident (F32). 28 x ~0.28GB + base ~ 7.8GB. ──
    print("")
    print("loading base + all", ANIMA_DEPTH, "blocks (F32 resident) ...")
    var base = load_anima_stack_base(st, ctx)
    var blocks = load_anima_all_blocks_f32(st, ANIMA_DEPTH, ctx)
    ctx.synchronize()
    var mem_loaded = ctx.get_memory_info()
    if Int(mem_loaded[0]) < min_free:
        min_free = Int(mem_loaded[0])
    print("free VRAM after weights load:", Int(Float64(mem_loaded[0]) / MIB), "MiB",
          " used by weights:", Int(Float64(mem0[0] - mem_loaded[0]) / MIB), "MiB")

    # ── synthesize finite inputs (real text/data path is phase D) ──
    var patches = _synth(B * S_IMG * IN_PATCH, 0.021, 0.05, 0.5)
    var t_cond = _synth(B * D, 0.013, 0.10, 0.30)
    var base_adaln = _synth(B * 3 * D, 0.017, 0.20, 0.20)
    var context = _synth(B * S_TXT * JOINT, 0.011, 0.30, 0.40)
    var d_out = _synth(B * S_IMG * OUT_PATCH, 0.027, 0.11, 0.05)

    # 3D-RoPE half-split tables [B*S_IMG*H, Dh/2], synthesized non-degenerate.
    var half = Dh // 2
    var cosl = List[Float32]()
    var sinl = List[Float32]()
    for _b in range(B):
        for s in range(S_IMG):
            for _h in range(H):
                for i in range(half):
                    var ang = Float32(s) / (Float32(10000.0) ** (Float32(2 * i) / Float32(Dh)))
                    cosl.append(fcos(ang))
                    sinl.append(fsin(ang))
    var cos = Tensor.from_host(cosl, [B * S_IMG * H, half], STDtype.F32, ctx)
    var sin = Tensor.from_host(sinl, [B * S_IMG * H, half], STDtype.F32, ctx)

    # ── full 28-layer forward ──
    print("")
    print("running full", ANIMA_DEPTH, "-layer forward ...")
    var fwd = anima_stack_forward[H, Dh, S_IMG, S_TXT](
        patches.copy(), t_cond.copy(), base_adaln.copy(), context.copy(),
        base, blocks, cos, sin, B, D, JOINT, F, IN_PATCH, OUT_PATCH, EPS, ctx,
    )
    ctx.synchronize()
    var mem_fwd = ctx.get_memory_info()
    if Int(mem_fwd[0]) < min_free:
        min_free = Int(mem_fwd[0])
    var out_bad = _count_nonfinite(fwd.out)
    print("forward out [S_IMG,64] n =", len(fwd.out), " non-finite =", out_bad)
    print("free VRAM after forward:", Int(Float64(mem_fwd[0]) / MIB), "MiB")

    # ── full-depth backward (per-block recompute) ──
    print("")
    print("running full", ANIMA_DEPTH, "-layer backward (per-block recompute) ...")
    var g = anima_stack_backward[H, Dh, S_IMG, S_TXT](
        d_out.copy(), patches.copy(), t_cond.copy(), base_adaln.copy(), context.copy(),
        base, blocks, cos, sin, fwd, B, D, JOINT, F, IN_PATCH, OUT_PATCH, EPS, ctx,
    )
    ctx.synchronize()
    var mem_bwd = ctx.get_memory_info()
    if Int(mem_bwd[0]) < min_free:
        min_free = Int(mem_bwd[0])

    # ── finiteness on returned grads + a sample of per-block grads ──
    var bad = _count_nonfinite(g.d_patches)
    bad += _count_nonfinite(g.d_t_silu)
    bad += _count_nonfinite(g.d_base_adaln)
    bad += _count_nonfinite(g.d_x_embed)
    bad += _count_nonfinite(g.d_fl_lin)
    bad += _count_nonfinite(g.d_fl_mod1)
    bad += _count_nonfinite(g.d_fl_mod2)
    # all 28 blocks' deepest+shallowest probe grads
    var blk_bad = 0
    for bi in range(ANIMA_DEPTH):
        blk_bad += _count_nonfinite(g.blk_grads[bi].d_sa_q)
        blk_bad += _count_nonfinite(g.blk_grads[bi].d_ca_v)
        blk_bad += _count_nonfinite(g.blk_grads[bi].d_mlp2)
        blk_bad += _count_nonfinite(g.blk_grads[bi].d_sa_mod1)

    print("")
    print("---- backward grad finiteness (0 = all finite) ----")
    print("  d_patches/d_t_silu/d_base/d_x_embed/d_fl_*:", bad)
    print("  ALL", ANIMA_DEPTH, "blocks (sa_q,ca_v,mlp2,sa_mod1) non-finite:", blk_bad)

    var total_bad = out_bad + bad + blk_bad
    var peak_used = total - min_free
    print("")
    print("---- memory ----")
    print("  peak GPU mem used:", Int(Float64(peak_used) / MIB), "MiB (",
          Float64(peak_used) / GIB, "GiB )")
    print("  total GPU mem:", Int(Float64(total) / MIB), "MiB")

    print("")
    if total_bad == 0:
        print("VERDICT: PASS —", ANIMA_DEPTH, "-layer REAL fwd+bwd ALL FINITE, NO OOM. peak",
              Float64(peak_used) / GIB, "GiB")
    else:
        print("VERDICT: FAIL — non-finite values present (total", total_bad, ")")

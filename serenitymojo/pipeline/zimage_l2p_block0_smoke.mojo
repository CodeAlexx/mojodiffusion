# Z-Image L2P block-0 compile-only smoke.
#
# Loads block 0 weights from the L2P checkpoint, builds the 3-axis RoPE tables,
# constructs a deterministic input and timestep cond, runs ONE transformer
# block, prints output shape + stats, and confirms the values are finite.
#
# Bounded sizes (NOT native 1024² yet — that's a follow-up):
#   CAP_LEN=32, PH=8, PW=8, IMG_PAD=0 -> S=96.
#
# Compile-verify only. The build command must produce a binary; do NOT run.

from std.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.io.dtype import STDtype
from serenitymojo.models.dit.zimage_l2p_contract import (
    ZIMAGE_L2P_HEAD_DIM,
    ZIMAGE_L2P_HIDDEN,
    ZIMAGE_L2P_NUM_HEADS,
    ZIMAGE_L2P_TIMESTEP_DIM,
    zimage_l2p_default_checkpoint_path,
)
from serenitymojo.models.dit.zimage_l2p_dit import (
    ZImageL2PBlockWeights,
    zimage_l2p_block_forward,
)
from serenitymojo.models.dit.zimage_l2p_rope import (
    build_zimage_l2p_3d_rope,
)
from serenitymojo.ops.random import randn
from serenitymojo.tensor import Tensor


# Block-0 smoke comptime knobs — keep small for fast compile.
comptime CAP_LEN = 32
comptime PH = 8
comptime PW = 8
comptime IMG_PAD = 0
comptime S = CAP_LEN + PH * PW + IMG_PAD  # 96
comptime DIM = ZIMAGE_L2P_HIDDEN          # 3840
comptime NUM_HEADS = ZIMAGE_L2P_NUM_HEADS  # 30
comptime HEAD_DIM = ZIMAGE_L2P_HEAD_DIM    # 128
comptime T_COND_DIM = ZIMAGE_L2P_TIMESTEP_DIM  # 256
comptime EPS = Float32(1.0e-5)
comptime SEED_X = UInt64(20260528)
comptime SEED_T = UInt64(20260529)


def _stats(name: String, t: Tensor, ctx: DeviceContext) raises:
    var h = t.to_host(ctx)
    var n = len(h)
    if n == 0:
        raise Error(String("empty tensor stats: ") + name)
    var s = Float64(0.0)
    var s2 = Float64(0.0)
    var amax = Float64(0.0)
    var any_nonfinite = False
    for i in range(n):
        var v = Float64(h[i])
        # NaN detection: NaN != NaN.
        if v != v:
            any_nonfinite = True
        # Inf detection: |v| > max finite f32.
        var av = v if v >= 0.0 else -v
        if av > Float64(3.4e38):
            any_nonfinite = True
        s += v
        s2 += v * v
        if av > amax:
            amax = av
    var mean = s / Float64(n)
    var var_ = s2 / Float64(n) - mean * mean
    if var_ < 0.0:
        var_ = 0.0
    print(
        "  [stat]", name,
        "n=", n,
        "mean=", Float32(mean),
        "std=", Float32(sqrt(Float32(var_))),
        "absmax=", Float32(amax),
    )
    if any_nonfinite:
        raise Error(String("non-finite values detected in ") + name)


def _shape_str(sh: List[Int]) -> String:
    var s = String("[")
    for i in range(len(sh)):
        if i > 0:
            s += String(", ")
        s += String(sh[i])
    s += String("]")
    return s^


def main() raises:
    var ctx = DeviceContext()
    print("== Z-Image L2P block-0 smoke ==")
    print("  CAP_LEN=", CAP_LEN)
    print("  PH=", PH, " PW=", PW, " IMG_PAD=", IMG_PAD)
    print("  S=", S)
    print("  DIM=", DIM)
    print("  NUM_HEADS=", NUM_HEADS, " HEAD_DIM=", HEAD_DIM)
    print("  T_COND_DIM=", T_COND_DIM)
    print("  EPS=", EPS)

    # ── Load layer-0 weights from the real L2P checkpoint ─────────────────
    print("[load] block 0 weights (layers.0)")
    var weights = ZImageL2PBlockWeights.load(
        zimage_l2p_default_checkpoint_path(), String("layers.0"), ctx
    )
    print("  weights loaded")

    # ── Build RoPE cos/sin tables ─────────────────────────────────────────
    print("[rope] 3-axis L2P RoPE tables")
    var rope = build_zimage_l2p_3d_rope[CAP_LEN, PH, PW, IMG_PAD](ctx)
    print("  rope_cos shape:", _shape_str(rope[0].shape()))
    print("  rope_sin shape:", _shape_str(rope[1].shape()))
    _stats("rope_cos", rope[0], ctx)
    _stats("rope_sin", rope[1], ctx)

    # ── Build deterministic input ─────────────────────────────────────────
    print("[input] random N(0,1) BF16 x [1, S, DIM] and t_cond [1, T_COND_DIM]")
    var x_shape = List[Int]()
    x_shape.append(1)
    x_shape.append(S)
    x_shape.append(DIM)
    var x = randn(x_shape^, SEED_X, STDtype.BF16, ctx)
    print("  x shape:", _shape_str(x.shape()))
    _stats("x", x, ctx)

    var t_shape = List[Int]()
    t_shape.append(1)
    t_shape.append(T_COND_DIM)
    var t_cond = randn(t_shape^, SEED_T, STDtype.BF16, ctx)
    print("  t_cond shape:", _shape_str(t_cond.shape()))
    _stats("t_cond", t_cond, ctx)

    # ── Block forward ─────────────────────────────────────────────────────
    print("[forward] zimage_l2p_block_forward[1, S=96]")
    var out = zimage_l2p_block_forward[1, S](
        weights, x, rope[0], rope[1], t_cond,
        NUM_HEADS, HEAD_DIM, EPS, ctx,
    )
    print("  out shape:", _shape_str(out.shape()))
    _stats("block0_out", out, ctx)

    var osh = out.shape()
    if len(osh) != 3 or osh[0] != 1 or osh[1] != S or osh[2] != DIM:
        raise Error("zimage_l2p_block0_smoke: output shape mismatch")
    print("[done] block-0 forward OK (compile + structural shape check)")

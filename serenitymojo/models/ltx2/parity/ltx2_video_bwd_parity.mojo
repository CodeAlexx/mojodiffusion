# serenitymojo/models/ltx2/parity/ltx2_video_bwd_parity.mojo
#
# LTX-2 VIDEO-ONLY (audio=None) block-0 BACKWARD parity gate.
#
# Compares the hand-chained Mojo video-only backward
#   serenitymojo/models/ltx2/ltx2_video_backward.mojo
#     ltx2_video_block_train_forward + ltx2_video_block_backward
# against the torch.autograd oracle
#   scripts/ltx2_video_block_bwd_oracle.py ->
#   output/ltx2_av_bwd/video_block0_bwd_ref.safetensors
#
# REAL block-0 weights from the dequant-bf16 export, REAL head counts
# (video 32x128), non-degenerate seeded inputs at S_V=128 / N_TXT=128.
# Factorized LoRA (rank 16, scale 0.5, A and B both random nonzero) attached on
# the 8 video-mode targets: {to_q,to_k,to_v,to_out.0} x {attn1, attn2}.
#
# Gate: cos >= 0.999 on EVERY grad — d_hidden and all 8x2 LoRA d_A/d_B — plus a
# forward cross-check (video_out cos >= 0.999). F32 compute (repo pattern for
# synthetic-dims gates; production is bf16).
#
# Run:
#   /home/alex/serenityflow-v2/.venv/bin/python scripts/ltx2_video_block_bwd_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build -O2 -I . \
#       serenitymojo/models/ltx2/parity/ltx2_video_bwd_parity.mojo \
#       -o /tmp/ltx2_video_bwd_parity && /tmp/ltx2_video_bwd_parity

from std.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.models.dit.ltx2_dit import LTX2Config, LTX2AVBlockWeights
from serenitymojo.models.ltx2.ltx2_video_backward import (
    ltx2_video_block_train_forward,
    ltx2_video_block_backward,
)

comptime CKPT = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled-fp8-dequant-bf16.safetensors"
comptime REF = "/home/alex/mojodiffusion/output/ltx2_av_bwd/video_block0_bwd_ref.safetensors"

comptime S_V = 128
comptime N_TXT = 128
comptime EPS = Float32(1e-6)
comptime LORA_SCALE = Float32(0.5)
comptime COS_GATE = 0.999


def _sh2(a: Int, b: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    s.append(b)
    return s^


def _load(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_view_as_f32(st.tensor_view(name), ctx)


# Oracle rope tables are [H, S, hrd]; the block consumes [(s,h), hrd] row order
# (same permute as ltx2_av_bwd_parity.mojo, exact F32 here).
def _load_rope(
    st: ShardedSafeTensors, name: String, ctx: DeviceContext
) raises -> Tensor:
    var t = Tensor.from_view_as_f32(st.tensor_view(name), ctx)
    var sh = t.shape()
    if len(sh) != 3:
        raise Error(String("rope table ") + name + " not rank-3")
    var H = sh[0]
    var S = sh[1]
    var hrd = sh[2]
    var host = t.to_host(ctx)
    var out = List[Float32]()
    for _ in range(S * H * hrd):
        out.append(Float32(0.0))
    for h in range(H):
        for s in range(S):
            for j in range(hrd):
                out[(s * H + h) * hrd + j] = host[(h * S + s) * hrd + j]
    return Tensor.from_host(out, _sh2(S * H, hrd), STDtype.F32, ctx)


struct _Cmp(Movable):
    var cos: Float64
    var max_abs: Float64

    def __init__(out self, cos: Float64, max_abs: Float64):
        self.cos = cos
        self.max_abs = max_abs


def _cmp_lists(a: List[Float32], b: List[Float32]) raises -> _Cmp:
    if len(a) != len(b):
        raise Error(
            String("cmp: length mismatch ") + String(len(a)) + " vs "
            + String(len(b)))
    var dot = 0.0
    var na = 0.0
    var nb = 0.0
    var mad = 0.0
    for i in range(len(a)):
        var x = Float64(a[i])
        var y = Float64(b[i])
        if x != x or y != y:
            raise Error("cmp: NaN")
        dot += x * y
        na += x * x
        nb += y * y
        var d = x - y
        if d < 0.0:
            d = -d
        if d > mad:
            mad = d
    return _Cmp(dot / (sqrt(na) * sqrt(nb) + 1e-30), mad)


def main() raises:
    var ctx = DeviceContext()
    var cfg = LTX2Config.ltx2()

    print("=== LTX-2 VIDEO-ONLY (audio=None) block-0 BACKWARD parity gate ===")
    print("  S_V/N_TXT:", S_V, N_TXT, " lora scale:", LORA_SCALE)
    print("  oracle:", REF)

    var dump = ShardedSafeTensors.open(String(REF))

    var hidden = _load(dump, "hs", ctx)
    var enc = _load(dump, "enc_hs", ctx)
    var v_temb = _load(dump, "v_timestep", ctx)
    var v_prompt_ts = _load(dump, "video_prompt_ts", ctx)

    var v_cos = _load_rope(dump, "v_cos", ctx)
    var v_sin = _load_rope(dump, "v_sin", ctx)

    var d_video = _load(dump, "d_video", ctx)

    print("  [load] block-0 AV weights (dequant-bf16 -> F32)")
    var weights = LTX2AVBlockWeights.load(String(CKPT), 0, cfg, ctx).to_f32(ctx)

    # Attach the 8 video-mode LoRA adapters (factorized, same scale as oracle).
    var mods = List[String]()
    mods.append(String("attn1"))
    mods.append(String("attn2"))
    var projs = List[String]()
    projs.append(String("to_q"))
    projs.append(String("to_k"))
    projs.append(String("to_v"))
    projs.append(String("to_out.0"))
    for ref m in mods:
        for ref p in projs:
            var a = _load(dump, String("lora.") + m + "." + p + ".A", ctx)
            var b = _load(dump, String("lora.") + m + "." + p + ".B", ctx)
            weights.add_lora_factor(m + "." + p + ".weight", a^, b^, LORA_SCALE)
    print("  [lora] attached", len(weights.lora_names), "adapters")

    print("  [forward] train-variant (activation-saving) VIDEO-ONLY block-0")
    var fwd = ltx2_video_block_train_forward[S_V, N_TXT](
        weights, hidden, enc, v_temb, v_prompt_ts, v_cos, v_sin, EPS, ctx,
    )

    var fails = 0

    # forward cross-check (catches forward drift before grading grads)
    var fv = _cmp_lists(fwd.video_out.to_host(ctx),
                        _load(dump, "video_out", ctx).to_host(ctx))
    print("  [fwd] video_out cos:", Float32(fv.cos), " max_abs:", Float32(fv.max_abs))
    if fv.cos < COS_GATE:
        fails += 1
        print("  FWD CROSS-CHECK FAIL")

    print("  [backward] hand-chained VIDEO-ONLY block-0")
    var grads = ltx2_video_block_backward[S_V, N_TXT](
        weights, fwd.acts, d_video, v_temb, v_cos, v_sin, EPS, ctx,
    )

    print("  --- grad parity table (cos >= 0.999 gates) ---")
    var dh = _cmp_lists(grads.d_hidden.to_host(ctx),
                        _load(dump, "g_d_hidden", ctx).to_host(ctx))
    print("  d_hidden          cos:", Float32(dh.cos), " max_abs:", Float32(dh.max_abs))
    if dh.cos < COS_GATE:
        fails += 1
        print("    ^^ FAIL")

    if len(grads.lora) != 8:
        raise Error(
            String("expected 8 LoRA pair grads, got ") + String(len(grads.lora)))

    for ref m in mods:
        for ref p in projs:
            var wkey = m + "." + p + ".weight"
            var found = False
            for ref g in grads.lora:
                if g.name == wkey:
                    found = True
                    var ra = _load(dump, String("g_dA.") + m + "." + p, ctx)
                    var rb = _load(dump, String("g_dB.") + m + "." + p, ctx)
                    var ca = _cmp_lists(g.d_a, ra.to_host(ctx))
                    var cb = _cmp_lists(g.d_b, rb.to_host(ctx))
                    print("  dA", m + "." + p,
                          " cos:", Float32(ca.cos), " max_abs:", Float32(ca.max_abs))
                    if ca.cos < COS_GATE:
                        fails += 1
                        print("    ^^ FAIL")
                    print("  dB", m + "." + p,
                          " cos:", Float32(cb.cos), " max_abs:", Float32(cb.max_abs))
                    if cb.cos < COS_GATE:
                        fails += 1
                        print("    ^^ FAIL")
            if not found:
                fails += 1
                print("  MISSING lora grad for", wkey)

    if fails > 0:
        raise Error(String("LTX-2 VIDEO BWD PARITY FAIL: ") + String(fails) + " gate(s)")
    print("LTX-2 VIDEO-ONLY block-0 BACKWARD PARITY PASS (1 d_x + 8x2 LoRA grads)")

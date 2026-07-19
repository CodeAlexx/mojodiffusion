# serenitymojo/models/ltx2/parity/ltx2_video_stack_parity.mojo
#
# LTX-2.3 VIDEO-MODE STACK (2 blocks + tail) parity gate.
#
# Compares the composed Mojo stack
#   serenitymojo/models/ltx2/ltx2_video_stack.mojo
#     ltx2_video_stack_lora_forward + ltx2_video_stack_lora_backward
# against the torch.autograd oracle
#   scripts/ltx2_video_stack_oracle.py ->
#   output/ltx2_video_stack/video_stack_bwd_ref.safetensors
#
# TWO parity stages against ONE fixture:
#   (A) HEAD cos-gate — run the Mojo head (LTX2VideoStackHead) on the fixture
#       PRIMITIVE latent+sigma and compare each head output (hidden0, v_temb,
#       v_embedded, v_prompt_ts, v_cos, v_sin) vs the oracle's, cos >= 0.999.
#       Gates the two flagged traps numerically: patchify token order f/h/w
#       (VideoLatentPatchifier) and rope inv_freq staying F64 (MJ-0815).
#   (B) STACK cos-gate — the fixture-driven composed BLOCKS + F32 TAIL forward
#       and full backward chain, using the fixture head-outputs as exact block
#       inputs: fwd pred cos >= 0.999; d_input cos >= 0.999; every LoRA d_A/d_B
#       (2 blocks x 8 pairs = 16 pairs) cos >= 0.999.
# REAL block-0/1 + head + tail weights from the distilled dequant-bf16 export,
# F32 compute (bf16-roundtripped, matching the oracle).
#
# Run:
#   /home/alex/serenityflow-v2/.venv/bin/python scripts/ltx2_video_stack_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build -O2 -I . -Xlinker -lm \
#       serenitymojo/models/ltx2/parity/ltx2_video_stack_parity.mojo \
#       -o /tmp/ltx2_video_stack_parity && /tmp/ltx2_video_stack_parity

from std.gpu.host import DeviceContext
from std.collections import List
from std.memory import ArcPointer
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.models.dit.ltx2_dit import LTX2Config
from serenitymojo.models.ltx2.ltx2_video_stack import (
    LTX2VideoBlockSource, LTX2VideoTail, LTX2VideoStackHead,
    ltx2_video_stack_lora_forward, ltx2_video_stack_lora_backward,
)

comptime CKPT = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled-fp8-dequant-bf16.safetensors"
comptime REF = "/home/alex/mojodiffusion/output/ltx2_video_stack/video_stack_bwd_ref.safetensors"

comptime S_V = 24
comptime N_TXT = 32
comptime NUM_BLOCKS = 2
comptime NF = 2          # reduced latent grid (S_V = NF*NH*NW = 24)
comptime NH = 3
comptime NW = 4
comptime FRAME_RATE = Float64(25.0)
comptime EPS = Float32(1e-6)
comptime LORA_SCALE = Float32(0.5)
comptime COS_GATE = 0.999


def _sh2(a: Int, b: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); return s^


def _load(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_view_as_f32(st.tensor_view(name), ctx)


# Oracle rope tables are [H,S,hrd]; the block consumes [(s,h),hrd] row order.
def _load_rope(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    var t = Tensor.from_view_as_f32(st.tensor_view(name), ctx)
    var sh = t.shape()
    if len(sh) != 3:
        raise Error(String("rope table ") + name + " not rank-3")
    var H = sh[0]; var S = sh[1]; var hrd = sh[2]
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


def _cmp(a: List[Float32], b: List[Float32]) raises -> _Cmp:
    if len(a) != len(b):
        raise Error(String("cmp: length mismatch ") + String(len(a)) + " vs " + String(len(b)))
    var dot = 0.0; var na = 0.0; var nb = 0.0; var mad = 0.0
    for i in range(len(a)):
        var x = Float64(a[i]); var y = Float64(b[i])
        if x != x or y != y:
            raise Error("cmp: NaN")
        dot += x * y; na += x * x; nb += y * y
        var d = x - y
        if d < 0.0: d = -d
        if d > mad: mad = d
    return _Cmp(dot / (sqrt(na) * sqrt(nb) + 1e-30), mad)


def _gate(name: String, a: List[Float32], b: List[Float32]) raises -> Int:
    var c = _cmp(a, b)
    print("  ", name, " cos:", Float32(c.cos), " max_abs:", Float32(c.max_abs))
    if c.cos < COS_GATE:
        print("    ^^ FAIL")
        return 1
    return 0


def _mods() -> List[String]:
    var m = List[String](); m.append("attn1"); m.append("attn2"); return m^

def _projs() -> List[String]:
    var p = List[String]()
    p.append("to_q"); p.append("to_k"); p.append("to_v"); p.append("to_out.0")
    return p^


def main() raises:
    var ctx = DeviceContext()
    var cfg = LTX2Config.ltx2()

    print("=== LTX-2.3 VIDEO-MODE STACK (2 blocks + tail) parity gate ===")
    print("  S_V/N_TXT:", S_V, N_TXT, " blocks:", NUM_BLOCKS, " lora scale:", LORA_SCALE)
    print("  oracle:", REF)

    var dump = ShardedSafeTensors.open(String(REF))

    var hidden = _load(dump, "hidden0", ctx)
    var enc = _load(dump, "enc_hs", ctx)
    var v_temb = _load(dump, "v_temb", ctx)
    var v_embedded = _load(dump, "v_embedded", ctx)
    var v_prompt_ts = _load(dump, "v_prompt_ts", ctx)
    var v_cos = _load_rope(dump, "v_cos", ctx)
    var v_sin = _load_rope(dump, "v_sin", ctx)
    var d_pred = _load(dump, "d_pred", ctx)

    # flat LoRA A/B (F32), block*8 + slot order = video_lora_names().
    var lora_a = List[ArcPointer[Tensor]]()
    var lora_b = List[ArcPointer[Tensor]]()
    for bi in range(NUM_BLOCKS):
        for ref m in _mods():
            for ref p in _projs():
                var key = String("lora.") + String(bi) + "." + m + "." + p
                lora_a.append(ArcPointer[Tensor](_load(dump, key + ".A", ctx)))
                lora_b.append(ArcPointer[Tensor](_load(dump, key + ".B", ctx)))
    print("  [lora] loaded", len(lora_a), "A/B pairs (F32)")

    var fails = 0

    # ── HEAD cos-gate: run the Mojo head on the fixture latent+sigma and gate
    # each head output vs the fixture BEFORE the fixture-driven stack compare.
    # (patchify+patchify_proj, adaln_single -> v_temb/v_embedded, prompt_adaln ->
    # v_prompt_ts, 3D RoPE). F32 to match the oracle. Traps gated NUMERICALLY:
    # token order f-outer/h/w-inner (VideoLatentPatchifier); rope inv_freq built
    # in F64, never bf16-rounded (MJ-0815 class).
    print("  --- HEAD parity (cos >= 0.999) ---")
    var latent = _load(dump, "latent", ctx)
    var sig_h = _load(dump, "sigma", ctx).to_host(ctx)
    var sigma = sig_h[0]
    print("  [head] load (F32) + forward, sigma:", sigma)
    var head = LTX2VideoStackHead.load(String(CKPT), ctx, True)
    var ho = head.forward[S_V, N_TXT](latent, enc, sigma, NF, NH, NW, FRAME_RATE, ctx)
    fails += _gate("hidden0    ", ho.hidden.to_host(ctx), hidden.to_host(ctx))
    fails += _gate("v_temb     ", ho.v_temb.to_host(ctx), v_temb.to_host(ctx))
    fails += _gate("v_embedded ", ho.v_embedded.to_host(ctx), v_embedded.to_host(ctx))
    fails += _gate("v_prompt_ts", ho.v_prompt_ts.to_host(ctx), v_prompt_ts.to_host(ctx))
    fails += _gate("v_cos      ", ho.v_cos.to_host(ctx), v_cos.to_host(ctx))
    fails += _gate("v_sin      ", ho.v_sin.to_host(ctx), v_sin.to_host(ctx))

    print("  [load] block source (dequant-bf16 -> F32) + tail")
    var src = LTX2VideoBlockSource.open(String(CKPT), cfg, True)
    var tail = LTX2VideoTail.load(String(CKPT), True, ctx)

    print("  [forward] stack (2 blocks + F32 tail)")
    var fwd = ltx2_video_stack_lora_forward[S_V, N_TXT](
        hidden, enc, v_temb, v_embedded, v_prompt_ts, v_cos, v_sin,
        tail, src, lora_a, lora_b, LORA_SCALE, NUM_BLOCKS, EPS, ctx,
    )

    var fp = _cmp(fwd.pred.to_host(ctx), _load(dump, "pred", ctx).to_host(ctx))
    print("  [fwd] pred cos:", Float32(fp.cos), " max_abs:", Float32(fp.max_abs))
    if fp.cos < COS_GATE:
        fails += 1
        print("    ^^ FWD FAIL")

    print("  [backward] stack (tail bwd -> per-block recompute+bwd)")
    var grads = ltx2_video_stack_lora_backward[S_V, N_TXT](
        d_pred, fwd.saved_inputs, fwd.x_last,
        enc, v_temb, v_embedded, v_prompt_ts, v_cos, v_sin,
        tail, src, lora_a, lora_b, LORA_SCALE, NUM_BLOCKS, EPS, ctx,
    )
    if grads.nonfinite != 0:
        fails += 1
        print("  NONFINITE grads:", grads.nonfinite)

    print("  --- grad parity table (cos >= 0.999 gates) ---")
    var di = _cmp(grads.d_input, _load(dump, "g_d_input", ctx).to_host(ctx))
    print("  d_input           cos:", Float32(di.cos), " max_abs:", Float32(di.max_abs))
    if di.cos < COS_GATE:
        fails += 1
        print("    ^^ FAIL")

    for bi in range(NUM_BLOCKS):
        var mi = 0
        for ref m in _mods():
            var pi = 0
            for ref p in _projs():
                var slot = bi * 8 + mi * 4 + pi
                var key = String(bi) + "." + m + "." + p
                var ra = _load(dump, String("g_dA.") + key, ctx)
                var rb = _load(dump, String("g_dB.") + key, ctx)
                var ca = _cmp(grads.d_a[slot], ra.to_host(ctx))
                var cb = _cmp(grads.d_b[slot], rb.to_host(ctx))
                print("  b" + String(bi), "dA", m + "." + p,
                      " cos:", Float32(ca.cos), " max_abs:", Float32(ca.max_abs))
                if ca.cos < COS_GATE:
                    fails += 1
                    print("    ^^ FAIL")
                print("  b" + String(bi), "dB", m + "." + p,
                      " cos:", Float32(cb.cos), " max_abs:", Float32(cb.max_abs))
                if cb.cos < COS_GATE:
                    fails += 1
                    print("    ^^ FAIL")
                pi += 1
            mi += 1

    if fails > 0:
        raise Error(String("LTX-2 VIDEO STACK PARITY FAIL: ") + String(fails) + " gate(s)")
    print("LTX-2.3 VIDEO-MODE STACK PARITY PASS (head 6 + fwd + d_input + 16x2 LoRA grads)")

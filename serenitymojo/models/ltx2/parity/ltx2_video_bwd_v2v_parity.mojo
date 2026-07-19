# serenitymojo/models/ltx2/parity/ltx2_video_bwd_v2v_parity.mojo
#
# LTX-2 VIDEO-ONLY V2V block-0 BACKWARD parity gate (P3 task 1).
#
# Sibling of ltx2_video_bwd_parity.mojo (the t2v 8-slot gate, LEFT UNTOUCHED so
# it keeps validating the attention backward with the FFN LoRA path OFF). This
# gate attaches the v2v surface = 10 adapters/block ({to_q,to_k,to_v,to_out.0} x
# {attn1,attn2} PLUS ff.net.0.proj + ff.net.2) and checks EVERY grad — d_hidden +
# all 10x2 LoRA d_A/d_B — cos >= 0.999 against the torch.autograd oracle's v2v
# reference, proving the FFN LoRA backward (the two new _lora_pair_bwd calls in
# ltx2_video_backward.mojo).
#
# Run:
#   /home/alex/serenityflow-v2/.venv/bin/python scripts/ltx2_video_block_bwd_oracle.py --v2v
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build -O2 -I . \
#       serenitymojo/models/ltx2/parity/ltx2_video_bwd_v2v_parity.mojo \
#       -o /tmp/ltx2_video_bwd_v2v_parity && /tmp/ltx2_video_bwd_v2v_parity

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
comptime REF = "/home/alex/mojodiffusion/output/ltx2_av_bwd/video_block0_bwd_v2v_ref.safetensors"

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


# The 10 v2v LoRA ref-suffixes (weight key = suffix + ".weight"; oracle dumps
# lora.<suffix>.A/.B + g_dA.<suffix>/g_dB.<suffix>).
def _v2v_keys() -> List[String]:
    var keys = List[String]()
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
            keys.append(m + "." + p)
    keys.append(String("ff.net.0.proj"))
    keys.append(String("ff.net.2"))
    return keys^


def main() raises:
    var ctx = DeviceContext()
    var cfg = LTX2Config.ltx2()

    print("=== LTX-2 VIDEO V2V block-0 BACKWARD parity gate (10 slots) ===")
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

    var keys = _v2v_keys()
    for ref k in keys:
        var a = _load(dump, String("lora.") + k + ".A", ctx)
        var b = _load(dump, String("lora.") + k + ".B", ctx)
        weights.add_lora_factor(k + ".weight", a^, b^, LORA_SCALE)
    print("  [lora] attached", len(weights.lora_names), "adapters (v2v = 10)")

    print("  [forward] train-variant (activation-saving) VIDEO-ONLY block-0")
    var fwd = ltx2_video_block_train_forward[S_V, N_TXT](
        weights, hidden, enc, v_temb, v_prompt_ts, v_cos, v_sin, EPS, ctx,
    )

    var fails = 0

    var fv = _cmp_lists(fwd.video_out.to_host(ctx),
                        _load(dump, "video_out", ctx).to_host(ctx))
    print("  [fwd] video_out cos:", Float32(fv.cos), " max_abs:", Float32(fv.max_abs))
    if fv.cos < COS_GATE:
        fails += 1
        print("  FWD CROSS-CHECK FAIL")

    print("  [backward] hand-chained VIDEO-ONLY block-0 (v2v)")
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

    if len(grads.lora) != 10:
        raise Error(
            String("expected 10 LoRA pair grads, got ") + String(len(grads.lora)))

    for ref k in keys:
        var wkey = k + ".weight"
        var found = False
        for ref g in grads.lora:
            if g.name == wkey:
                found = True
                var ra = _load(dump, String("g_dA.") + k, ctx)
                var rb = _load(dump, String("g_dB.") + k, ctx)
                var ca = _cmp_lists(g.d_a, ra.to_host(ctx))
                var cb = _cmp_lists(g.d_b, rb.to_host(ctx))
                print("  dA", k, " cos:", Float32(ca.cos), " max_abs:", Float32(ca.max_abs))
                if ca.cos < COS_GATE:
                    fails += 1
                    print("    ^^ FAIL")
                print("  dB", k, " cos:", Float32(cb.cos), " max_abs:", Float32(cb.max_abs))
                if cb.cos < COS_GATE:
                    fails += 1
                    print("    ^^ FAIL")
        if not found:
            fails += 1
            print("  MISSING lora grad for", wkey)

    if fails > 0:
        raise Error(String("LTX-2 VIDEO V2V BWD PARITY FAIL: ") + String(fails) + " gate(s)")
    print("LTX-2 VIDEO V2V block-0 BACKWARD PARITY PASS (1 d_x + 10x2 LoRA grads)")

# serenitymojo/models/ltx2/parity/ltx2_ic_v2v_block_parity.mojo
#
# LTX-2 IC-LoRA / V2V COMBINED-SEQUENCE block-0 forward+backward parity gate
# (P5 unit 2). Sibling of ltx2_video_bwd_v2v_parity.mojo (the t2v-grid v2v-surface
# gate) but on the COMBINED [S_ref + S_target] sequence: it proves the video block
# fwd/bwd handle the ref-prepended sequence + the ref-sliced-off loss cotangent
# exactly as the IC/V2V torch oracle (ltx2_ic_v2v_oracle.py) does.
#
# What this validates (against ic_v2v_block0_ref_image512.safetensors):
#   * fwd cross-check: block_out (combined S=320) cos >= 0.999 vs the oracle,
#   * bwd from d_block_out (= [zeros on the ref rows ; d(masked_loss)/d(target_pred)]
#     -- the ref-slice IS the zeros on the ref rows): d_hidden + all 10x2 v2v LoRA
#     grads cos >= 0.999.
# The block is comptime-generic on S_V, so S=320 is one new instantiation; no
# block code changed. The two-grid RoPE that the TRAINER feeds here is gated
# separately (ltx2_v2v_rope_coords_gate + the spine _compute_rope); THIS gate
# consumes the oracle's dumped v_cos/v_sin so it isolates the block math (same
# convention as the existing t2v block-gate family).
#
# image512 preset: ref 1x8x8 (64) + target 1x16x16 (256) -> S_V = 320, N_TXT=128.
#
# Run (GPU window):
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/ltx2/parity/ltx2_ic_v2v_oracle.py            # dump image512
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build -O2 -I . -Xlinker -lm -Xlinker -lcuda \
#       serenitymojo/models/ltx2/parity/ltx2_ic_v2v_block_parity.mojo \
#       -o /tmp/ltx2_ic_v2v_block_parity && /tmp/ltx2_ic_v2v_block_parity

from std.gpu.host import DeviceContext
from std.math import sqrt
from std.sys import argv

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.models.dit.ltx2_dit import LTX2Config, LTX2AVBlockWeights
from serenitymojo.models.ltx2.ltx2_video_backward import (
    ltx2_video_block_train_forward,
    ltx2_video_block_backward,
)
from serenitymojo.models.ltx2.ltx2_video_stack import _build_v2v_rope

comptime FR = Float64(25.0)

comptime CKPT = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled-fp8-dequant-bf16.safetensors"
comptime OUT_DIR = "/home/alex/mojodiffusion/output/ltx2_ic_v2v/"
comptime REF_IMAGE512 = OUT_DIR + "ic_v2v_block0_ref_image512.safetensors"    # S=320 (av rope)
comptime REF_VIDEO512 = OUT_DIR + "ic_v2v_block0_ref_video512.safetensors"    # S=608 (av rope)
comptime REF_IMAGE512_SPINE = OUT_DIR + "ic_v2v_block0_ref_image512_spine.safetensors"  # variant B
comptime REF_VIDEO512_SPINE = OUT_DIR + "ic_v2v_block0_ref_video512_spine.safetensors"  # variant B

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


def _run_gate[S_V: Int, N_TXT: Int](
    REF: String, build_rope: Bool,
    rnf: Int, rnh: Int, rnw: Int, tnf: Int, tnh: Int, tnw: Int, ds: Int,
) raises:
    var ctx = DeviceContext()
    var cfg = LTX2Config.ltx2()

    print("=== LTX-2 IC/V2V COMBINED-SEQUENCE block-0 parity gate (S=", S_V,
          " rope=", "BUILT(_build_v2v_rope)" if build_rope else "DUMPED", ") ===")
    print("  N_TXT:", N_TXT, " lora scale:", LORA_SCALE)
    print("  oracle:", REF)

    var dump = ShardedSafeTensors.open(String(REF))

    var hidden = _load(dump, "hs", ctx)                 # [1,S,4096]
    var enc = _load(dump, "enc_hs", ctx)                # [1,128,4096]
    var v_temb = _load(dump, "v_timestep", ctx)         # [1,S,9*4096]
    var v_prompt_ts = _load(dump, "video_prompt_ts", ctx)
    # VARIANT B: build the TRAINER's rope (_build_v2v_rope) rather than consuming
    # the oracle's dumped cos/sin — gates the rope BUILDER + block together at the
    # trainer operating point vs the musubi-spine oracle. _build_v2v_rope emits
    # [P*H,hrd] token-major directly (same layout _load_rope produces).
    var v_cos: Tensor
    var v_sin: Tensor
    if build_rope:
        var rope = _build_v2v_rope(rnf, rnh, rnw, tnf, tnh, tnw, ds, FR, STDtype.F32, ctx)
        v_cos = rope[0].clone(ctx)
        v_sin = rope[1].clone(ctx)
    else:
        v_cos = _load_rope(dump, "v_cos", ctx)
        v_sin = _load_rope(dump, "v_sin", ctx)
    var d_block = _load(dump, "d_block_out", ctx)       # [1,S,4096] (ref rows = 0)

    print("  [load] block-0 AV weights (dequant-bf16 -> F32)")
    var weights = LTX2AVBlockWeights.load(String(CKPT), 0, cfg, ctx).to_f32(ctx)

    var keys = _v2v_keys()
    for ref k in keys:
        var a = _load(dump, String("lora.") + k + ".A", ctx)
        var b = _load(dump, String("lora.") + k + ".B", ctx)
        weights.add_lora_factor(k + ".weight", a^, b^, LORA_SCALE)
    print("  [lora] attached", len(weights.lora_names), "adapters (v2v = 10)")

    print("  [forward] combined-sequence VIDEO block-0 (S=", S_V, ")")
    var fwd = ltx2_video_block_train_forward[S_V, N_TXT](
        weights, hidden, enc, v_temb, v_prompt_ts, v_cos, v_sin, EPS, ctx,
    )

    var fails = 0

    var fv = _cmp_lists(fwd.video_out.to_host(ctx),
                        _load(dump, "block_out", ctx).to_host(ctx))
    print("  [fwd] block_out cos:", Float32(fv.cos), " max_abs:", Float32(fv.max_abs))
    if fv.cos < COS_GATE:
        fails += 1
        print("  FWD CROSS-CHECK FAIL")

    print("  [backward] hand-chained combined block-0, cotangent = d_block_out")
    var grads = ltx2_video_block_backward[S_V, N_TXT](
        weights, fwd.acts, d_block, v_temb, v_cos, v_sin, EPS, ctx,
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
        raise Error(String("LTX-2 IC/V2V BLOCK PARITY FAIL: ") + String(fails) + " gate(s)")
    print("LTX-2 IC/V2V COMBINED-SEQUENCE block-0 PARITY PASS (1 d_x + 10x2 LoRA grads @ S=", S_V, ")")


# --grid image512 (S=320) | video512 (S=608, target 4x9x16 + ref 1x4x8).
# --build-rope = VARIANT B: build the trainer rope (_build_v2v_rope) + block and
#   gate vs the SPINE oracle dump (musubi precompute rope, trainer operating
#   point). Default (no flag) = the block-gate-family rope (dumped av rope).
# The chosen grid's oracle dump must exist (run ltx2_ic_v2v_oracle.py --grid <g>
#   [--spine-rope] for variant B).
def main() raises:
    var grid = String("image512")
    var build_rope = False
    var args = argv()
    for i in range(len(args)):
        if String(args[i]) == "--grid" and i + 1 < len(args):
            grid = String(args[i + 1])
        if String(args[i]) == "--build-rope":
            build_rope = True
    if grid == "image512":
        var ref_path = String(REF_IMAGE512)
        if build_rope:
            ref_path = String(REF_IMAGE512_SPINE)
        _run_gate[320, N_TXT](ref_path, build_rope, 1, 8, 8, 1, 16, 16, 2)
    elif grid == "video512":
        var ref_path = String(REF_VIDEO512)
        if build_rope:
            ref_path = String(REF_VIDEO512_SPINE)
        _run_gate[608, N_TXT](ref_path, build_rope, 1, 4, 8, 4, 9, 16, 2)
    else:
        raise Error(String("unknown --grid: ") + grid + " (image512 | video512)")

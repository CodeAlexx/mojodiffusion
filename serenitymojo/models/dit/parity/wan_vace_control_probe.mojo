# wan_vace_control_probe.mojo — block-level VACE control-injection MATH gate.
#
# NO Wan-VACE weights exist on disk (only a structure dump of
# wan2.1_vace_14B_fp16.safetensors). Per the port gate, this probe gates the
# VACE control DELTAS — before_proj / wan22 block body / after_proj / scaled hint
# injection — using BASE Wan2.2 TI2V-5B block weights for the (already-gated at
# cos 0.99963) WanAttentionBlock body, plus SYNTHETIC before_proj (identity),
# after_proj (0.5*identity) and a synthetic control hidden. The full WanVaceDit
# forward stays BLOCKED-pending-weights.
#
# Reference math (VaceWanModel, model.py:33-44, 63-67):
#   block0:  c0   = before_proj(c) + img
#            cout = base_block(c0)
#            hint = after_proj(cout)
#   inject:  out  = img + hint * scale
#
# In-probe ORACLE (independent of wan_vace_block) recomputes each step from the
# trusted primitives (linear, wan22_block_forward, add, mul_scalar) and compares
# to wan_vace_block + wan_vace_inject. Gate: cos >= 0.999 on cout, hint, inject.
#
# Run:
#   cd /home/alex/mojodiffusion
#   pixi run mojo run -I . serenitymojo/models/dit/parity/wan_vace_control_probe.mojo

from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.linear import linear
from serenitymojo.ops.tensor_algebra import add, mul_scalar
from serenitymojo.models.dit.wan22_dit import (
    Wan22Config, wan22_build_rope, wan22_block_forward,
)
from serenitymojo.models.dit.wan_vace_dit import (
    wan_vace_block, wan_vace_inject, wan_vace_layer_for_base,
)


comptime CKPT = "/home/alex/.serenity/models/checkpoints/Wan2.2-TI2V-5B-bf16"

# Small grid (reuse the gated wan22 block path: S=16 <=512 keeps math-mode SDPA).
comptime S = 16
comptime TXT = 512
comptime NH = 24
comptime HD = 128
comptime DIM = 3072
comptime F_G = 1
comptime H_G = 4
comptime W_G = 4
comptime SCALE = 1.5


def _randish(n: Int, seed: Int) -> List[Float32]:
    # Deterministic pseudo-random fill in ~[-1,1] (host-side, probe-only).
    var out = List[Float32]()
    var st = UInt64(seed * 2654435761 + 1013904223)
    for _ in range(n):
        st = st * 6364136223846793005 + 1442695040888963407
        var u = Float32((st >> 33) & 0x7FFFFF) / Float32(0x7FFFFF)  # [0,1)
        out.append(u * 2.0 - 1.0)
    return out^


# Identity matrix [d,d] (linear computes x@wᵀ; identityᵀ == identity).
def _identity(d: Int, ctx: DeviceContext) raises -> Tensor:
    var h = List[Float32]()
    for r in range(d):
        for c in range(d):
            var v: Float32 = 1.0 if r == c else 0.0
            h.append(v)
    return Tensor.from_host(h^, [d, d], STDtype.BF16, ctx)


def _zeros1(d: Int, ctx: DeviceContext) raises -> Tensor:
    var h = List[Float32]()
    for _ in range(d):
        h.append(0.0)
    return Tensor.from_host(h^, [d], STDtype.BF16, ctx)


def _load_w(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    var tv = st.tensor_view(name)
    return Tensor.from_view(tv, ctx)


def main() raises:
    var ctx = DeviceContext()
    print("=== Wan-VACE control-injection MATH gate (S=", S, ", bf16 GPU) ===")
    print("    weights: NONE on disk -> block body uses base Wan2.2 TI2V-5B")
    print("    before_proj=identity, after_proj=0.5*identity, scale=", SCALE)

    var cfg = Wan22Config.ti2v_5b()

    # sanity: vace-layer mapping (model.py vace_layers = [0,2,4,...,14]).
    if wan_vace_layer_for_base(0) != 0 or wan_vace_layer_for_base(14) != 7 \
       or wan_vace_layer_for_base(1) != -1 or wan_vace_layer_for_base(16) != -1:
        raise Error("vace layer mapping wrong")
    print("    vace-layer mapping OK: base[0]->0, base[14]->7, base[1,16]->none")

    # ── Synthetic inputs ──
    var c_in_f32 = Tensor.from_host(_randish(S * DIM, 1), [1, S, DIM], STDtype.F32, ctx)
    var c_in = cast_tensor(c_in_f32, STDtype.BF16, ctx)
    var img_f32 = Tensor.from_host(_randish(S * DIM, 2), [1, S, DIM], STDtype.F32, ctx)
    var img = cast_tensor(img_f32, STDtype.BF16, ctx)
    var e0 = Tensor.from_host(_randish(S * 6 * DIM, 3), [1, S, 6, DIM], STDtype.F32, ctx)
    var ctxt_f32 = Tensor.from_host(_randish(TXT * DIM, 4), [1, TXT, DIM], STDtype.F32, ctx)
    var ctxt = cast_tensor(ctxt_f32, STDtype.BF16, ctx)
    var cs = wan22_build_rope(F_G, H_G, W_G, HD, cfg.rope_theta, STDtype.BF16, ctx)

    # ── Block weights = base blocks.0.* + synthetic before/after_proj ──
    var st = ShardedSafeTensors.open(CKPT)
    var w = Dict[String, ArcPointer[Tensor]]()
    var keys = [
        "modulation",
        "self_attn.q.weight", "self_attn.q.bias",
        "self_attn.k.weight", "self_attn.k.bias",
        "self_attn.v.weight", "self_attn.v.bias",
        "self_attn.o.weight", "self_attn.o.bias",
        "self_attn.norm_q.weight", "self_attn.norm_k.weight",
        "cross_attn.q.weight", "cross_attn.q.bias",
        "cross_attn.k.weight", "cross_attn.k.bias",
        "cross_attn.v.weight", "cross_attn.v.bias",
        "cross_attn.o.weight", "cross_attn.o.bias",
        "cross_attn.norm_q.weight", "cross_attn.norm_k.weight",
        "norm3.weight", "norm3.bias",
        "ffn.0.weight", "ffn.0.bias",
        "ffn.2.weight", "ffn.2.bias",
    ]
    for kk in keys:
        var key = String(kk)
        w[key] = ArcPointer(_load_w(st, String("blocks.0.") + key, ctx))

    # before_proj = identity (zero bias). after_proj = 0.5*identity (zero bias).
    var ident = _identity(DIM, ctx)
    var half_ident_h = List[Float32]()
    for r in range(DIM):
        for c in range(DIM):
            var hv: Float32 = 0.5 if r == c else 0.0
            half_ident_h.append(hv)
    var half_ident = Tensor.from_host(half_ident_h^, [DIM, DIM], STDtype.BF16, ctx)
    w[String("before_proj.weight")] = ArcPointer(ident.clone(ctx))
    w[String("before_proj.bias")] = ArcPointer(_zeros1(DIM, ctx))
    w[String("after_proj.weight")] = ArcPointer(half_ident.clone(ctx))
    w[String("after_proj.bias")] = ArcPointer(_zeros1(DIM, ctx))

    # ── DUT: wan_vace_block (block0) + wan_vace_inject ──
    var res = wan_vace_block[S, TXT, NH, HD](
        True, c_in, img, e0, ctxt, cs[0], cs[1], w, cfg, ctx,
    )
    var dut_cout = res[0].clone(ctx)
    var dut_hint = res[1].clone(ctx)
    var dut_inject = wan_vace_inject(img, dut_hint, SCALE, ctx)

    # ── In-probe ORACLE (independent recompute) ──
    # c0 = before_proj(c_in)+img ; before_proj=identity so == c_in+img
    var bp = linear(c_in, ident, Optional(_zeros1(DIM, ctx)), ctx)
    var c0_f32 = add(cast_tensor(bp, STDtype.F32, ctx), cast_tensor(img, STDtype.F32, ctx), ctx)
    var c0 = cast_tensor(c0_f32, STDtype.BF16, ctx)
    var ref_cout = wan22_block_forward[S, TXT, NH, HD](
        c0, e0, ctxt, cs[0], cs[1], w, cfg, ctx,
    )
    # hint = after_proj(cout) = 0.5*cout
    var ref_hint = linear(ref_cout, half_ident, Optional(_zeros1(DIM, ctx)), ctx)
    # inject = img + hint*scale
    var ref_inject_f32 = add(
        cast_tensor(img, STDtype.F32, ctx),
        mul_scalar(cast_tensor(ref_hint, STDtype.F32, ctx), SCALE, ctx), ctx,
    )

    # ── Compare ──
    var harness = ParityHarness(0.999)
    var ref_cout_h = cast_tensor(ref_cout, STDtype.F32, ctx).to_host(ctx)
    var ref_hint_h = cast_tensor(ref_hint, STDtype.F32, ctx).to_host(ctx)
    var ref_inject_h = ref_inject_f32.to_host(ctx)

    var r_cout = harness.compare(cast_tensor(dut_cout, STDtype.F32, ctx), ref_cout_h, ctx)
    var r_hint = harness.compare(cast_tensor(dut_hint, STDtype.F32, ctx), ref_hint_h, ctx)
    var r_inject = harness.compare(cast_tensor(dut_inject, STDtype.F32, ctx), ref_inject_h, ctx)

    print("    vace block cout  :", r_cout)
    print("    vace hint (aproj):", r_hint)
    print("    vace inject      :", r_inject)

    var lo = r_cout.cos
    if r_hint.cos < lo:
        lo = r_hint.cos
    if r_inject.cos < lo:
        lo = r_inject.cos

    if r_cout.passed and r_hint.passed and r_inject.passed:
        print("GATE PASS controlMathCos=", lo)
    else:
        print("GATE FAIL controlMathCos=", lo)

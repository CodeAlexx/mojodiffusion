# serenitymojo/models/flux/parity/flux_direct_stack_offload_smoke.mojo
#
# Focused Flux/Chroma direct DoRA/OFT stack gate. Runs one synthetic double block
# and one synthetic single block through the streamed offload wrappers. Direct
# DoRA/OFT initialize as identity substitutions for the frozen projection
# weights, so forward output should match the resident base stack while backward
# should produce compact direct grads without nonfinite values.

from std.collections import List, Optional
from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.models.flux.block import (
    StreamWeights, DoubleBlockWeights, SingleBlockWeights,
)
from serenitymojo.models.flux.flux_stack import (
    FluxStackBase, EmbedMlp, ModLin, DoubleModLin, flux_stack_forward,
)
from serenitymojo.models.flux.flux_stack_lora import (
    build_flux_direct_dora_set_from_offload,
    build_flux_direct_oft_set_for_stack,
    flux_stack_direct_dora_forward_offload,
    flux_stack_direct_dora_backward_offload,
    flux_stack_direct_oft_forward_offload,
    flux_stack_direct_oft_backward_offload,
)
from serenitymojo.models.flux.flux_lycoris_stack import FLUX_LYCORIS_TGT_ALL
from serenitymojo.models.flux.flux_direct_lycoris_stack import (
    flux_direct_active_slot_count,
)
from serenitymojo.offload.plan import build_flux_block_plan, OffloadConfig
from serenitymojo.offload.turbo_planned_loader import TurboPlannedLoader


comptime TArc = ArcPointer[Tensor]
comptime CKPT_PATH = "/tmp/flux_direct_stack_offload_smoke.safetensors"
comptime H = 2
comptime Dh = 8
comptime D = H * Dh
comptime N_IMG = 3
comptime N_TXT = 2
comptime S = N_TXT + N_IMG
comptime FMLP = 32
comptime IN_CH = 8
comptime TXT_CH = 12
comptime OUT_CH = 8
comptime T_DIM = 8
comptime VEC_DIM = 10
comptime NUM_DOUBLE = 1
comptime NUM_SINGLE = 1
comptime EPS = Float32(1.0e-06)
comptime MAX_PERIOD = Float32(10000.0)
comptime RANK = 4
comptime ALPHA = Float32(4.0)
comptime OFT_BLOCK = 4
comptime COS_BAR = 0.99999


def _rand(n: Int, seed: UInt64, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    for _ in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
        out.append((u - Float32(0.5)) * scale)
    return out^


def _ones(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(Float32(1.0))
    return out^


def _take_rows(src: List[Float32], start: Int, rows: Int, cols: Int) -> List[Float32]:
    var out = List[Float32]()
    var base = start * cols
    for r in range(rows):
        for c in range(cols):
            out.append(src[base + r * cols + c])
    return out^


def _cos(a: List[Float32], b: List[Float32]) raises -> Float64:
    if len(a) != len(b):
        raise Error("cos: length mismatch")
    var dot = Float64(0.0)
    var na = Float64(0.0)
    var nb = Float64(0.0)
    for i in range(len(a)):
        dot += Float64(a[i]) * Float64(b[i])
        na += Float64(a[i]) * Float64(a[i])
        nb += Float64(b[i]) * Float64(b[i])
    if na == 0.0 or nb == 0.0:
        raise Error("cos: zero vector")
    return dot / (sqrt(na) * sqrt(nb))


def _nonfinite(v: List[Float32]) -> Int:
    var bad = 0
    for i in range(len(v)):
        var x = v[i]
        if (x != x) or (x - x != Float32(0.0)):
            bad += 1
    return bad


def _stream_weights(seed: UInt64, ctx: DeviceContext) raises -> StreamWeights:
    return StreamWeights(
        _rand(3 * D * D, seed + 1, Float32(0.04)),
        _rand(3 * D, seed + 2, Float32(0.01)),
        _rand(D * D, seed + 3, Float32(0.04)),
        _rand(D, seed + 4, Float32(0.01)),
        _rand(FMLP * D, seed + 5, Float32(0.04)),
        _rand(FMLP, seed + 6, Float32(0.01)),
        _rand(D * FMLP, seed + 7, Float32(0.04)),
        _rand(D, seed + 8, Float32(0.01)),
        _ones(Dh), _ones(Dh),
        D, FMLP, Dh, ctx,
    )


def _single_weights(seed: UInt64, ctx: DeviceContext) raises -> SingleBlockWeights:
    return SingleBlockWeights(
        _rand((3 * D + FMLP) * D, seed + 1, Float32(0.04)),
        _rand(3 * D + FMLP, seed + 2, Float32(0.01)),
        _rand(D * (D + FMLP), seed + 3, Float32(0.04)),
        _rand(D, seed + 4, Float32(0.01)),
        _ones(Dh), _ones(Dh), D, FMLP, Dh, ctx,
    )


def _add(
    mut names: List[String], mut tensors: List[TArc],
    name: String, vals: List[Float32], var shape: List[Int], ctx: DeviceContext,
) raises:
    names.append(name)
    tensors.append(TArc(Tensor.from_host(vals.copy(), shape^, STDtype.BF16, ctx)))


def main() raises:
    var ctx = DeviceContext()
    print("==== flux_direct_stack_offload_smoke ====")
    print("D=", D, " FMLP=", FMLP, " double=", NUM_DOUBLE, " single=", NUM_SINGLE)

    var time_in = EmbedMlp(
        _rand(D * T_DIM, UInt64(100), Float32(0.03)), _rand(D, UInt64(101), Float32(0.01)),
        _rand(D * D, UInt64(102), Float32(0.03)), _rand(D, UInt64(103), Float32(0.01)),
        T_DIM, D, ctx,
    )
    var guid_in = EmbedMlp(
        _rand(D * T_DIM, UInt64(110), Float32(0.03)), _rand(D, UInt64(111), Float32(0.01)),
        _rand(D * D, UInt64(112), Float32(0.03)), _rand(D, UInt64(113), Float32(0.01)),
        T_DIM, D, ctx,
    )
    var vec_in = EmbedMlp(
        _rand(D * VEC_DIM, UInt64(120), Float32(0.03)), _rand(D, UInt64(121), Float32(0.01)),
        _rand(D * D, UInt64(122), Float32(0.03)), _rand(D, UInt64(123), Float32(0.01)),
        VEC_DIM, D, ctx,
    )

    var dbl_mod = List[DoubleModLin]()
    var im = ModLin(
        _rand(6 * D * D, UInt64(200), Float32(0.03)),
        _rand(6 * D, UInt64(201), Float32(0.01)), 6 * D, D, ctx,
    )
    var tm = ModLin(
        _rand(6 * D * D, UInt64(202), Float32(0.03)),
        _rand(6 * D, UInt64(203), Float32(0.01)), 6 * D, D, ctx,
    )
    dbl_mod.append(DoubleModLin(im^, tm^))
    var sgl_mod = List[ModLin]()
    sgl_mod.append(ModLin(
        _rand(3 * D * D, UInt64(210), Float32(0.03)),
        _rand(3 * D, UInt64(211), Float32(0.01)), 3 * D, D, ctx,
    ))

    var base = FluxStackBase(
        _rand(D * IN_CH, UInt64(300), Float32(0.03)), _rand(D, UInt64(301), Float32(0.01)),
        _rand(D * TXT_CH, UInt64(302), Float32(0.03)), _rand(D, UInt64(303), Float32(0.01)),
        time_in^, True, guid_in^, vec_in^,
        dbl_mod^, sgl_mod^,
        _rand(2 * D * D, UInt64(304), Float32(0.03)), _rand(2 * D, UInt64(305), Float32(0.01)),
        _rand(OUT_CH * D, UInt64(306), Float32(0.03)), _rand(OUT_CH, UInt64(307), Float32(0.01)),
        D, IN_CH, TXT_CH, OUT_CH, ctx,
    )

    var iw = _stream_weights(UInt64(1000), ctx)
    var tw = _stream_weights(UInt64(1100), ctx)
    var sw = _single_weights(UInt64(1200), ctx)
    var dbw = List[DoubleBlockWeights]()
    dbw.append(DoubleBlockWeights(iw.copy(), tw.copy()))
    var sbw = List[SingleBlockWeights]()
    sbw.append(sw.copy())

    var s_w1 = sw.w1[].to_host(ctx)
    var s_w2 = sw.w2[].to_host(ctx)

    var names = List[String]()
    var tensors = List[TArc]()
    _add(names, tensors, "double_blocks.0.img_attn.qkv.weight", iw.wqkv[].to_host(ctx), [3 * D, D], ctx)
    _add(names, tensors, "double_blocks.0.img_attn.qkv.bias", iw.bqkv[].to_host(ctx), [3 * D], ctx)
    _add(names, tensors, "double_blocks.0.img_attn.proj.weight", iw.wproj[].to_host(ctx), [D, D], ctx)
    _add(names, tensors, "double_blocks.0.img_attn.proj.bias", iw.bproj[].to_host(ctx), [D], ctx)
    _add(names, tensors, "double_blocks.0.img_mlp.0.weight", iw.wmlp0[].to_host(ctx), [FMLP, D], ctx)
    _add(names, tensors, "double_blocks.0.img_mlp.0.bias", iw.bmlp0[].to_host(ctx), [FMLP], ctx)
    _add(names, tensors, "double_blocks.0.img_mlp.2.weight", iw.wmlp2[].to_host(ctx), [D, FMLP], ctx)
    _add(names, tensors, "double_blocks.0.img_mlp.2.bias", iw.bmlp2[].to_host(ctx), [D], ctx)
    _add(names, tensors, "double_blocks.0.img_attn.norm.query_norm.scale", iw.q_norm[].to_host(ctx), [Dh], ctx)
    _add(names, tensors, "double_blocks.0.img_attn.norm.key_norm.scale", iw.k_norm[].to_host(ctx), [Dh], ctx)
    _add(names, tensors, "double_blocks.0.txt_attn.qkv.weight", tw.wqkv[].to_host(ctx), [3 * D, D], ctx)
    _add(names, tensors, "double_blocks.0.txt_attn.qkv.bias", tw.bqkv[].to_host(ctx), [3 * D], ctx)
    _add(names, tensors, "double_blocks.0.txt_attn.proj.weight", tw.wproj[].to_host(ctx), [D, D], ctx)
    _add(names, tensors, "double_blocks.0.txt_attn.proj.bias", tw.bproj[].to_host(ctx), [D], ctx)
    _add(names, tensors, "double_blocks.0.txt_mlp.0.weight", tw.wmlp0[].to_host(ctx), [FMLP, D], ctx)
    _add(names, tensors, "double_blocks.0.txt_mlp.0.bias", tw.bmlp0[].to_host(ctx), [FMLP], ctx)
    _add(names, tensors, "double_blocks.0.txt_mlp.2.weight", tw.wmlp2[].to_host(ctx), [D, FMLP], ctx)
    _add(names, tensors, "double_blocks.0.txt_mlp.2.bias", tw.bmlp2[].to_host(ctx), [D], ctx)
    _add(names, tensors, "double_blocks.0.txt_attn.norm.query_norm.scale", tw.q_norm[].to_host(ctx), [Dh], ctx)
    _add(names, tensors, "double_blocks.0.txt_attn.norm.key_norm.scale", tw.k_norm[].to_host(ctx), [Dh], ctx)
    _add(names, tensors, "single_blocks.0.linear1.weight", s_w1.copy(), [3 * D + FMLP, D], ctx)
    _add(names, tensors, "single_blocks.0.linear1.bias", sw.b1[].to_host(ctx), [3 * D + FMLP], ctx)
    _add(names, tensors, "single_blocks.0.linear2.weight", s_w2.copy(), [D, D + FMLP], ctx)
    _add(names, tensors, "single_blocks.0.linear2.bias", sw.b2[].to_host(ctx), [D], ctx)
    _add(names, tensors, "single_blocks.0.norm.query_norm.scale", sw.q_norm[].to_host(ctx), [Dh], ctx)
    _add(names, tensors, "single_blocks.0.norm.key_norm.scale", sw.k_norm[].to_host(ctx), [Dh], ctx)
    save_safetensors(names, tensors, String(CKPT_PATH), ctx)

    var plan_init = build_flux_block_plan(NUM_DOUBLE, NUM_SINGLE)
    var cfg_init = OffloadConfig.synchronous_single()
    var loader_init = TurboPlannedLoader.open(String(CKPT_PATH), plan_init^, cfg_init, ctx)
    var dora = build_flux_direct_dora_set_from_offload(
        loader_init, NUM_DOUBLE, NUM_SINGLE, D, FMLP, Dh,
        RANK, ALPHA, FLUX_LYCORIS_TGT_ALL, UInt64(7000), False, ctx,
    )
    var oft = build_flux_direct_oft_set_for_stack(
        NUM_DOUBLE, NUM_SINGLE, D, FMLP, OFT_BLOCK, FLUX_LYCORIS_TGT_ALL,
    )
    var expected_slots = flux_direct_active_slot_count(NUM_DOUBLE, NUM_SINGLE, FLUX_LYCORIS_TGT_ALL)
    if len(dora.ad) != expected_slots or len(oft.ad) != expected_slots:
        raise Error("direct stack smoke: direct slot count mismatch")

    var img_tokens = _rand(N_IMG * IN_CH, UInt64(800), Float32(0.50))
    var txt_tokens = _rand(N_TXT * TXT_CH, UInt64(801), Float32(0.50))
    var timestep = _rand(1, UInt64(802), Float32(0.50))
    var guidance = Optional[List[Float32]](_rand(1, UInt64(803), Float32(0.50)))
    var vector = _rand(VEC_DIM, UInt64(804), Float32(0.50))
    var cos = _rand(S * H * (Dh // 2), UInt64(805), Float32(0.50))
    var sin = _rand(S * H * (Dh // 2), UInt64(806), Float32(0.50))
    var d_out = _rand(N_IMG * OUT_CH, UInt64(807), Float32(0.25))

    var fwd_base = flux_stack_forward[H, Dh, N_IMG, N_TXT, S](
        img_tokens.copy(), txt_tokens.copy(), timestep.copy(), guidance, vector.copy(),
        base, dbw, sbw, cos.copy(), sin.copy(),
        D, FMLP, IN_CH, TXT_CH, OUT_CH, T_DIM, VEC_DIM, EPS, ctx,
    )

    var plan_d = build_flux_block_plan(NUM_DOUBLE, NUM_SINGLE)
    var cfg_d = OffloadConfig.synchronous_single()
    var loader_d = TurboPlannedLoader.open(String(CKPT_PATH), plan_d^, cfg_d, ctx)
    var fwd_dora = flux_stack_direct_dora_forward_offload[H, Dh, N_IMG, N_TXT, S](
        img_tokens.copy(), txt_tokens.copy(), timestep.copy(), guidance, vector.copy(),
        base, loader_d, dora, NUM_DOUBLE, NUM_SINGLE, FLUX_LYCORIS_TGT_ALL,
        cos.copy(), sin.copy(),
        D, FMLP, IN_CH, TXT_CH, OUT_CH, T_DIM, VEC_DIM, EPS, ctx,
    )
    var gd = flux_stack_direct_dora_backward_offload[H, Dh, N_IMG, N_TXT, S](
        d_out.copy(), img_tokens.copy(), txt_tokens.copy(), base, loader_d, dora,
        NUM_DOUBLE, NUM_SINGLE, FLUX_LYCORIS_TGT_ALL, cos.copy(), sin.copy(),
        fwd_dora, D, FMLP, IN_CH, TXT_CH, OUT_CH, T_DIM, VEC_DIM, MAX_PERIOD, EPS, ctx,
    )

    var plan_o = build_flux_block_plan(NUM_DOUBLE, NUM_SINGLE)
    var cfg_o = OffloadConfig.synchronous_single()
    var loader_o = TurboPlannedLoader.open(String(CKPT_PATH), plan_o^, cfg_o, ctx)
    var fwd_oft = flux_stack_direct_oft_forward_offload[H, Dh, N_IMG, N_TXT, S](
        img_tokens.copy(), txt_tokens.copy(), timestep.copy(), guidance, vector.copy(),
        base, loader_o, oft, NUM_DOUBLE, NUM_SINGLE, FLUX_LYCORIS_TGT_ALL,
        cos.copy(), sin.copy(),
        D, FMLP, IN_CH, TXT_CH, OUT_CH, T_DIM, VEC_DIM, EPS, ctx,
    )
    var go = flux_stack_direct_oft_backward_offload[H, Dh, N_IMG, N_TXT, S](
        d_out.copy(), img_tokens.copy(), txt_tokens.copy(), base, loader_o, oft,
        NUM_DOUBLE, NUM_SINGLE, FLUX_LYCORIS_TGT_ALL, cos.copy(), sin.copy(),
        fwd_oft, D, FMLP, IN_CH, TXT_CH, OUT_CH, T_DIM, VEC_DIM, MAX_PERIOD, EPS, ctx,
    )

    var cd = _cos(fwd_dora.out, fwd_base.out)
    var co = _cos(fwd_oft.out, fwd_base.out)
    print("  direct DoRA slots=", len(dora.ad), " forward_cos=", cd,
          " nonfinite=", gd.nonfinite_lora_grads)
    print("  direct OFT slots=", len(oft.ad), " forward_cos=", co,
          " nonfinite=", go.nonfinite_lora_grads)
    if cd < COS_BAR or co < COS_BAR:
        raise Error("direct stack smoke: initialized direct forward diverged from base stack")
    if gd.nonfinite_lora_grads != 0 or go.nonfinite_lora_grads != 0:
        raise Error("direct stack smoke: direct backward produced nonfinite grads")
    if len(gd.grads.g) != expected_slots or len(go.grads.d_vec) != expected_slots:
        raise Error("direct stack smoke: compact grad count mismatch")
    if _nonfinite(gd.d_img_tokens) != 0 or _nonfinite(go.d_img_tokens) != 0:
        raise Error("direct stack smoke: nonfinite input-token grads")

    print("ALL GATES PASS -- flux_direct_stack_offload_smoke")

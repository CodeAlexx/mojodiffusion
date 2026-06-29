# serenitymojo/models/chroma/parity/chroma_direct_stack_offload_smoke.mojo
#
# Chroma direct DoRA/OFT stack gate. Chroma shares Flux block math but has its
# own stack modulation/final path, so this explicitly runs the Chroma offload
# wrappers over Diffusers-style Chroma block keys.

from std.collections import List
from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.models.chroma.chroma_stack_lora import (
    ChromaStackBase,
    build_chroma_direct_dora_set_from_offload,
    build_chroma_direct_oft_set_for_stack,
    chroma_stack_lora_forward_offload,
    chroma_stack_direct_dora_forward_offload,
    chroma_stack_direct_dora_backward_offload,
    chroma_stack_direct_oft_forward_offload,
    chroma_stack_direct_oft_backward_offload,
)
from serenitymojo.models.flux.flux_stack_lora import build_flux_lora_set
from serenitymojo.models.flux.flux_lycoris_stack import FLUX_LYCORIS_TGT_ALL
from serenitymojo.models.flux.flux_direct_lycoris_stack import (
    flux_direct_active_slot_count,
)
from serenitymojo.offload.plan import build_chroma_block_plan, OffloadConfig
from serenitymojo.offload.turbo_planned_loader import TurboPlannedLoader


comptime TArc = ArcPointer[Tensor]
comptime CKPT_PATH = "/tmp/chroma_direct_stack_offload_smoke.safetensors"
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
comptime NUM_DOUBLE = 1
comptime NUM_SINGLE = 1
comptime MOD_INDEX = 3 * NUM_SINGLE + 12 * NUM_DOUBLE + 2
comptime EPS = Float32(1.0e-06)
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


def _arc(vals: List[Float32], var shape: List[Int], ctx: DeviceContext) raises -> TArc:
    return TArc(Tensor.from_host(vals.copy(), shape^, STDtype.BF16, ctx))


def _add(
    mut names: List[String], mut tensors: List[TArc],
    name: String, vals: List[Float32], var shape: List[Int], ctx: DeviceContext,
) raises:
    names.append(name)
    tensors.append(_arc(vals, shape^, ctx))


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


def main() raises:
    var ctx = DeviceContext()
    print("==== chroma_direct_stack_offload_smoke ====")
    print("D=", D, " FMLP=", FMLP, " double=", NUM_DOUBLE, " single=", NUM_SINGLE)

    var base = ChromaStackBase(
        _arc(_rand(D * IN_CH, UInt64(100), Float32(0.03)), [D, IN_CH], ctx),
        _arc(_rand(D, UInt64(101), Float32(0.01)), [D], ctx),
        _arc(_rand(D * TXT_CH, UInt64(102), Float32(0.03)), [D, TXT_CH], ctx),
        _arc(_rand(D, UInt64(103), Float32(0.01)), [D], ctx),
        _arc(_rand(OUT_CH * D, UInt64(104), Float32(0.03)), [OUT_CH, D], ctx),
        _arc(_rand(OUT_CH, UInt64(105), Float32(0.01)), [OUT_CH], ctx),
        NUM_DOUBLE, NUM_SINGLE,
    )

    var iwq = _rand(D * D, UInt64(1000), Float32(0.04))
    var iwk = _rand(D * D, UInt64(1001), Float32(0.04))
    var iwv = _rand(D * D, UInt64(1002), Float32(0.04))
    var ibq = _rand(D, UInt64(1003), Float32(0.01))
    var ibk = _rand(D, UInt64(1004), Float32(0.01))
    var ibv = _rand(D, UInt64(1005), Float32(0.01))
    var iwp = _rand(D * D, UInt64(1006), Float32(0.04))
    var ibp = _rand(D, UInt64(1007), Float32(0.01))
    var iw0 = _rand(FMLP * D, UInt64(1008), Float32(0.04))
    var ib0 = _rand(FMLP, UInt64(1009), Float32(0.01))
    var iw2 = _rand(D * FMLP, UInt64(1010), Float32(0.04))
    var ib2 = _rand(D, UInt64(1011), Float32(0.01))

    var twq = _rand(D * D, UInt64(1100), Float32(0.04))
    var twk = _rand(D * D, UInt64(1101), Float32(0.04))
    var twv = _rand(D * D, UInt64(1102), Float32(0.04))
    var tbq = _rand(D, UInt64(1103), Float32(0.01))
    var tbk = _rand(D, UInt64(1104), Float32(0.01))
    var tbv = _rand(D, UInt64(1105), Float32(0.01))
    var twp = _rand(D * D, UInt64(1106), Float32(0.04))
    var tbp = _rand(D, UInt64(1107), Float32(0.01))
    var tw0 = _rand(FMLP * D, UInt64(1108), Float32(0.04))
    var tb0 = _rand(FMLP, UInt64(1109), Float32(0.01))
    var tw2 = _rand(D * FMLP, UInt64(1110), Float32(0.04))
    var tb2 = _rand(D, UInt64(1111), Float32(0.01))

    var swq = _rand(D * D, UInt64(1200), Float32(0.04))
    var swk = _rand(D * D, UInt64(1201), Float32(0.04))
    var swv = _rand(D * D, UInt64(1202), Float32(0.04))
    var sbq = _rand(D, UInt64(1203), Float32(0.01))
    var sbk = _rand(D, UInt64(1204), Float32(0.01))
    var sbv = _rand(D, UInt64(1205), Float32(0.01))
    var swpm = _rand(FMLP * D, UInt64(1206), Float32(0.04))
    var sbpm = _rand(FMLP, UInt64(1207), Float32(0.01))
    var swo = _rand(D * (D + FMLP), UInt64(1208), Float32(0.04))
    var sbo = _rand(D, UInt64(1209), Float32(0.01))

    var names = List[String]()
    var tensors = List[TArc]()
    var bp = String("transformer_blocks.0.")
    _add(names, tensors, bp + "attn.to_q.weight", iwq, [D, D], ctx)
    _add(names, tensors, bp + "attn.to_q.bias", ibq, [D], ctx)
    _add(names, tensors, bp + "attn.to_k.weight", iwk, [D, D], ctx)
    _add(names, tensors, bp + "attn.to_k.bias", ibk, [D], ctx)
    _add(names, tensors, bp + "attn.to_v.weight", iwv, [D, D], ctx)
    _add(names, tensors, bp + "attn.to_v.bias", ibv, [D], ctx)
    _add(names, tensors, bp + "attn.to_out.0.weight", iwp, [D, D], ctx)
    _add(names, tensors, bp + "attn.to_out.0.bias", ibp, [D], ctx)
    _add(names, tensors, bp + "ff.net.0.proj.weight", iw0, [FMLP, D], ctx)
    _add(names, tensors, bp + "ff.net.0.proj.bias", ib0, [FMLP], ctx)
    _add(names, tensors, bp + "ff.net.2.weight", iw2, [D, FMLP], ctx)
    _add(names, tensors, bp + "ff.net.2.bias", ib2, [D], ctx)
    _add(names, tensors, bp + "attn.norm_q.weight", _ones(Dh), [Dh], ctx)
    _add(names, tensors, bp + "attn.norm_k.weight", _ones(Dh), [Dh], ctx)
    _add(names, tensors, bp + "attn.add_q_proj.weight", twq, [D, D], ctx)
    _add(names, tensors, bp + "attn.add_q_proj.bias", tbq, [D], ctx)
    _add(names, tensors, bp + "attn.add_k_proj.weight", twk, [D, D], ctx)
    _add(names, tensors, bp + "attn.add_k_proj.bias", tbk, [D], ctx)
    _add(names, tensors, bp + "attn.add_v_proj.weight", twv, [D, D], ctx)
    _add(names, tensors, bp + "attn.add_v_proj.bias", tbv, [D], ctx)
    _add(names, tensors, bp + "attn.to_add_out.weight", twp, [D, D], ctx)
    _add(names, tensors, bp + "attn.to_add_out.bias", tbp, [D], ctx)
    _add(names, tensors, bp + "ff_context.net.0.proj.weight", tw0, [FMLP, D], ctx)
    _add(names, tensors, bp + "ff_context.net.0.proj.bias", tb0, [FMLP], ctx)
    _add(names, tensors, bp + "ff_context.net.2.weight", tw2, [D, FMLP], ctx)
    _add(names, tensors, bp + "ff_context.net.2.bias", tb2, [D], ctx)
    _add(names, tensors, bp + "attn.norm_added_q.weight", _ones(Dh), [Dh], ctx)
    _add(names, tensors, bp + "attn.norm_added_k.weight", _ones(Dh), [Dh], ctx)
    var sp = String("single_transformer_blocks.0.")
    _add(names, tensors, sp + "attn.to_q.weight", swq, [D, D], ctx)
    _add(names, tensors, sp + "attn.to_q.bias", sbq, [D], ctx)
    _add(names, tensors, sp + "attn.to_k.weight", swk, [D, D], ctx)
    _add(names, tensors, sp + "attn.to_k.bias", sbk, [D], ctx)
    _add(names, tensors, sp + "attn.to_v.weight", swv, [D, D], ctx)
    _add(names, tensors, sp + "attn.to_v.bias", sbv, [D], ctx)
    _add(names, tensors, sp + "proj_mlp.weight", swpm, [FMLP, D], ctx)
    _add(names, tensors, sp + "proj_mlp.bias", sbpm, [FMLP], ctx)
    _add(names, tensors, sp + "proj_out.weight", swo, [D, D + FMLP], ctx)
    _add(names, tensors, sp + "proj_out.bias", sbo, [D], ctx)
    _add(names, tensors, sp + "attn.norm_q.weight", _ones(Dh), [Dh], ctx)
    _add(names, tensors, sp + "attn.norm_k.weight", _ones(Dh), [Dh], ctx)
    save_safetensors(names, tensors, String(CKPT_PATH), ctx)

    var plan_init = build_chroma_block_plan(NUM_DOUBLE, NUM_SINGLE)
    var cfg_init = OffloadConfig.synchronous_single()
    var loader_init = TurboPlannedLoader.open(String(CKPT_PATH), plan_init^, cfg_init, ctx)
    var dora = build_chroma_direct_dora_set_from_offload(
        loader_init, NUM_DOUBLE, NUM_SINGLE, D, FMLP,
        RANK, ALPHA, FLUX_LYCORIS_TGT_ALL, UInt64(7000), False, ctx,
    )
    var oft = build_chroma_direct_oft_set_for_stack(
        NUM_DOUBLE, NUM_SINGLE, D, FMLP, OFT_BLOCK, FLUX_LYCORIS_TGT_ALL,
    )
    var expected_slots = flux_direct_active_slot_count(NUM_DOUBLE, NUM_SINGLE, FLUX_LYCORIS_TGT_ALL)
    if len(dora.ad) != expected_slots or len(oft.ad) != expected_slots:
        raise Error("chroma direct stack smoke: direct slot count mismatch")

    var img_tokens = _rand(N_IMG * IN_CH, UInt64(800), Float32(0.50))
    var txt_tokens = _rand(N_TXT * TXT_CH, UInt64(801), Float32(0.50))
    var pooled = _rand(MOD_INDEX * D, UInt64(802), Float32(0.05))
    var cos = _rand(S * H * (Dh // 2), UInt64(803), Float32(0.50))
    var sin = _rand(S * H * (Dh // 2), UInt64(804), Float32(0.50))
    var d_out = _rand(N_IMG * OUT_CH, UInt64(805), Float32(0.25))

    var lora = build_flux_lora_set(NUM_DOUBLE, NUM_SINGLE, D, FMLP, RANK, ALPHA)
    var plan_b = build_chroma_block_plan(NUM_DOUBLE, NUM_SINGLE)
    var cfg_b = OffloadConfig.synchronous_single()
    var loader_b = TurboPlannedLoader.open(String(CKPT_PATH), plan_b^, cfg_b, ctx)
    var fwd_base = chroma_stack_lora_forward_offload[H, Dh, N_IMG, N_TXT, S](
        img_tokens.copy(), txt_tokens.copy(), pooled.copy(), MOD_INDEX,
        base, loader_b, lora, cos.copy(), sin.copy(),
        D, FMLP, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
    )

    var plan_d = build_chroma_block_plan(NUM_DOUBLE, NUM_SINGLE)
    var cfg_d = OffloadConfig.synchronous_single()
    var loader_d = TurboPlannedLoader.open(String(CKPT_PATH), plan_d^, cfg_d, ctx)
    var fwd_dora = chroma_stack_direct_dora_forward_offload[H, Dh, N_IMG, N_TXT, S](
        img_tokens.copy(), txt_tokens.copy(), pooled.copy(), MOD_INDEX,
        base, loader_d, dora, NUM_DOUBLE, NUM_SINGLE, FLUX_LYCORIS_TGT_ALL,
        cos.copy(), sin.copy(), D, FMLP, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
    )
    var gd = chroma_stack_direct_dora_backward_offload[H, Dh, N_IMG, N_TXT, S](
        d_out.copy(), img_tokens.copy(), txt_tokens.copy(), base, loader_d, dora,
        NUM_DOUBLE, NUM_SINGLE, FLUX_LYCORIS_TGT_ALL, cos.copy(), sin.copy(),
        fwd_dora, D, FMLP, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
    )

    var plan_o = build_chroma_block_plan(NUM_DOUBLE, NUM_SINGLE)
    var cfg_o = OffloadConfig.synchronous_single()
    var loader_o = TurboPlannedLoader.open(String(CKPT_PATH), plan_o^, cfg_o, ctx)
    var fwd_oft = chroma_stack_direct_oft_forward_offload[H, Dh, N_IMG, N_TXT, S](
        img_tokens.copy(), txt_tokens.copy(), pooled.copy(), MOD_INDEX,
        base, loader_o, oft, NUM_DOUBLE, NUM_SINGLE, FLUX_LYCORIS_TGT_ALL,
        cos.copy(), sin.copy(), D, FMLP, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
    )
    var go = chroma_stack_direct_oft_backward_offload[H, Dh, N_IMG, N_TXT, S](
        d_out.copy(), img_tokens.copy(), txt_tokens.copy(), base, loader_o, oft,
        NUM_DOUBLE, NUM_SINGLE, FLUX_LYCORIS_TGT_ALL, cos.copy(), sin.copy(),
        fwd_oft, D, FMLP, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
    )

    var cd = _cos(fwd_dora.out, fwd_base.out)
    var co = _cos(fwd_oft.out, fwd_base.out)
    print("  direct DoRA slots=", len(dora.ad), " forward_cos=", cd,
          " nonfinite=", gd.nonfinite_lora_grads)
    print("  direct OFT slots=", len(oft.ad), " forward_cos=", co,
          " nonfinite=", go.nonfinite_lora_grads)
    if cd < COS_BAR or co < COS_BAR:
        raise Error("chroma direct stack smoke: initialized direct forward diverged from base stack")
    if gd.nonfinite_lora_grads != 0 or go.nonfinite_lora_grads != 0:
        raise Error("chroma direct stack smoke: direct backward produced nonfinite grads")
    if len(gd.grads.g) != expected_slots or len(go.grads.d_vec) != expected_slots:
        raise Error("chroma direct stack smoke: compact grad count mismatch")
    if _nonfinite(gd.d_img_tokens) != 0 or _nonfinite(go.d_img_tokens) != 0:
        raise Error("chroma direct stack smoke: nonfinite input-token grads")

    print("ALL GATES PASS -- chroma_direct_stack_offload_smoke")

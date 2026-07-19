# serenitymojo/models/chroma/parity/chroma_stack_device_parity.mojo
#
# STACK-LEVEL parity gate: DEVICE-RESIDENT chroma stack (activations as TArc, via
# the gated device block) vs the HOST chroma stack (the parity oracle), fwd+bwd,
# on byte-identical inputs. Mirrors the synthetic setup of
# chroma_direct_stack_offload_smoke.mojo (random base weights -> temp
# safetensors -> TurboPlannedLoader + ChromaStackBase), but runs the LoRA
# forward/backward on BOTH arms and compares:
#   forward `out`                          : cos >= 0.999
#   every LoRA d_a / d_b slot              : cos >= 0.999 AND rel-L2 <= 1e-3
#
# NOTE: build_flux_lora_set inits B = 0 (PEFT identity), which would zero the
# forward LoRA delta AND d_a. To exercise BOTH d_a and d_b the set is rebuilt with
# NONZERO random A and B (moments zeroed; AdamW never runs here).
#
# NOTE: the loader is CONSUMED by forward (awaits all blocks); a single loader
# instance serves fwd THEN bwd of one arm (exactly the trainer's use). Each arm
# gets its OWN loader instance (rebuilt between arms), as the direct smoke does.

from std.collections import List, Optional
from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.models.chroma.chroma_stack_lora import (
    ChromaStackBase,
    chroma_stack_lora_forward_offload,
    chroma_stack_lora_backward_offload,
    chroma_stack_lora_forward_device_offload,
    chroma_stack_lora_backward_device_offload,
)
from serenitymojo.models.flux.flux_stack_lora import (
    FluxLoraSet, build_flux_lora_set, total_adapters,
)
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.offload.plan import build_chroma_block_plan, OffloadConfig
from serenitymojo.offload.turbo_planned_loader import TurboPlannedLoader


comptime TArc = ArcPointer[Tensor]
comptime CKPT_PATH = "/tmp/chroma_stack_device_parity.safetensors"
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
comptime NUM_DOUBLE = 2
comptime NUM_SINGLE = 2
comptime MOD_INDEX = 3 * NUM_SINGLE + 12 * NUM_DOUBLE + 2
comptime EPS = Float32(1.0e-06)
comptime RANK = 4
comptime ALPHA = Float32(4.0)
comptime COS_BAR = 0.999
comptime L2_BAR = 1.0e-03


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


def _zeros(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(Float32(0.0))
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


def _rel_l2(a: List[Float32], b: List[Float32]) raises -> Float64:
    if len(a) != len(b):
        raise Error("rel_l2: length mismatch")
    var num = Float64(0.0)
    var den = Float64(0.0)
    for i in range(len(a)):
        var diff = Float64(a[i]) - Float64(b[i])
        num += diff * diff
        den += Float64(b[i]) * Float64(b[i])
    if den == 0.0:
        raise Error("rel_l2: zero reference")
    return sqrt(num) / sqrt(den)


# rebuild the LoRA set with NONZERO random A and B (moments zeroed).
def _nonzero_lora_set(src: FluxLoraSet, seed0: UInt64) -> FluxLoraSet:
    var ad = List[LoraAdapter]()
    var seed = seed0
    for i in range(len(src.ad)):
        var rank = src.ad[i].rank
        var in_f = src.ad[i].in_f
        var out_f = src.ad[i].out_f
        var scale = src.ad[i].scale
        var na = rank * in_f
        var nb = out_f * rank
        var a = _rand(na, seed, Float32(0.30)); seed += 1
        var b = _rand(nb, seed, Float32(0.30)); seed += 1
        ad.append(LoraAdapter(
            a^, b^, rank, in_f, out_f, scale,
            _zeros(na), _zeros(na), _zeros(nb), _zeros(nb),
        ))
    return FluxLoraSet(ad^, src.num_double, src.num_single, src.rank)


def _write_double_block(
    mut names: List[String], mut tensors: List[TArc], bp: String, seed: UInt64, ctx: DeviceContext
) raises:
    # img stream
    _add(names, tensors, bp + "attn.to_q.weight", _rand(D * D, seed + 0, 0.04), [D, D], ctx)
    _add(names, tensors, bp + "attn.to_q.bias", _rand(D, seed + 1, 0.01), [D], ctx)
    _add(names, tensors, bp + "attn.to_k.weight", _rand(D * D, seed + 2, 0.04), [D, D], ctx)
    _add(names, tensors, bp + "attn.to_k.bias", _rand(D, seed + 3, 0.01), [D], ctx)
    _add(names, tensors, bp + "attn.to_v.weight", _rand(D * D, seed + 4, 0.04), [D, D], ctx)
    _add(names, tensors, bp + "attn.to_v.bias", _rand(D, seed + 5, 0.01), [D], ctx)
    _add(names, tensors, bp + "attn.to_out.0.weight", _rand(D * D, seed + 6, 0.04), [D, D], ctx)
    _add(names, tensors, bp + "attn.to_out.0.bias", _rand(D, seed + 7, 0.01), [D], ctx)
    _add(names, tensors, bp + "ff.net.0.proj.weight", _rand(FMLP * D, seed + 8, 0.04), [FMLP, D], ctx)
    _add(names, tensors, bp + "ff.net.0.proj.bias", _rand(FMLP, seed + 9, 0.01), [FMLP], ctx)
    _add(names, tensors, bp + "ff.net.2.weight", _rand(D * FMLP, seed + 10, 0.04), [D, FMLP], ctx)
    _add(names, tensors, bp + "ff.net.2.bias", _rand(D, seed + 11, 0.01), [D], ctx)
    _add(names, tensors, bp + "attn.norm_q.weight", _ones(Dh), [Dh], ctx)
    _add(names, tensors, bp + "attn.norm_k.weight", _ones(Dh), [Dh], ctx)
    # txt stream
    _add(names, tensors, bp + "attn.add_q_proj.weight", _rand(D * D, seed + 12, 0.04), [D, D], ctx)
    _add(names, tensors, bp + "attn.add_q_proj.bias", _rand(D, seed + 13, 0.01), [D], ctx)
    _add(names, tensors, bp + "attn.add_k_proj.weight", _rand(D * D, seed + 14, 0.04), [D, D], ctx)
    _add(names, tensors, bp + "attn.add_k_proj.bias", _rand(D, seed + 15, 0.01), [D], ctx)
    _add(names, tensors, bp + "attn.add_v_proj.weight", _rand(D * D, seed + 16, 0.04), [D, D], ctx)
    _add(names, tensors, bp + "attn.add_v_proj.bias", _rand(D, seed + 17, 0.01), [D], ctx)
    _add(names, tensors, bp + "attn.to_add_out.weight", _rand(D * D, seed + 18, 0.04), [D, D], ctx)
    _add(names, tensors, bp + "attn.to_add_out.bias", _rand(D, seed + 19, 0.01), [D], ctx)
    _add(names, tensors, bp + "ff_context.net.0.proj.weight", _rand(FMLP * D, seed + 20, 0.04), [FMLP, D], ctx)
    _add(names, tensors, bp + "ff_context.net.0.proj.bias", _rand(FMLP, seed + 21, 0.01), [FMLP], ctx)
    _add(names, tensors, bp + "ff_context.net.2.weight", _rand(D * FMLP, seed + 22, 0.04), [D, FMLP], ctx)
    _add(names, tensors, bp + "ff_context.net.2.bias", _rand(D, seed + 23, 0.01), [D], ctx)
    _add(names, tensors, bp + "attn.norm_added_q.weight", _ones(Dh), [Dh], ctx)
    _add(names, tensors, bp + "attn.norm_added_k.weight", _ones(Dh), [Dh], ctx)


def _write_single_block(
    mut names: List[String], mut tensors: List[TArc], sp: String, seed: UInt64, ctx: DeviceContext
) raises:
    _add(names, tensors, sp + "attn.to_q.weight", _rand(D * D, seed + 0, 0.04), [D, D], ctx)
    _add(names, tensors, sp + "attn.to_q.bias", _rand(D, seed + 1, 0.01), [D], ctx)
    _add(names, tensors, sp + "attn.to_k.weight", _rand(D * D, seed + 2, 0.04), [D, D], ctx)
    _add(names, tensors, sp + "attn.to_k.bias", _rand(D, seed + 3, 0.01), [D], ctx)
    _add(names, tensors, sp + "attn.to_v.weight", _rand(D * D, seed + 4, 0.04), [D, D], ctx)
    _add(names, tensors, sp + "attn.to_v.bias", _rand(D, seed + 5, 0.01), [D], ctx)
    _add(names, tensors, sp + "proj_mlp.weight", _rand(FMLP * D, seed + 6, 0.04), [FMLP, D], ctx)
    _add(names, tensors, sp + "proj_mlp.bias", _rand(FMLP, seed + 7, 0.01), [FMLP], ctx)
    _add(names, tensors, sp + "proj_out.weight", _rand(D * (D + FMLP), seed + 8, 0.04), [D, D + FMLP], ctx)
    _add(names, tensors, sp + "proj_out.bias", _rand(D, seed + 9, 0.01), [D], ctx)
    _add(names, tensors, sp + "attn.norm_q.weight", _ones(Dh), [Dh], ctx)
    _add(names, tensors, sp + "attn.norm_k.weight", _ones(Dh), [Dh], ctx)


def _open_loader(ctx: DeviceContext) raises -> TurboPlannedLoader:
    var plan = build_chroma_block_plan(NUM_DOUBLE, NUM_SINGLE)
    var cfg = OffloadConfig.synchronous_single()
    return TurboPlannedLoader.open(String(CKPT_PATH), plan^, cfg, ctx)


def main() raises:
    var ctx = DeviceContext()
    print("==== chroma_stack_device_parity ====")
    print("D=", D, " FMLP=", FMLP, " double=", NUM_DOUBLE, " single=", NUM_SINGLE,
          " N_IMG=", N_IMG, " N_TXT=", N_TXT, " RANK=", RANK)

    var base = ChromaStackBase(
        _arc(_rand(D * IN_CH, UInt64(100), Float32(0.03)), [D, IN_CH], ctx),
        _arc(_rand(D, UInt64(101), Float32(0.01)), [D], ctx),
        _arc(_rand(D * TXT_CH, UInt64(102), Float32(0.03)), [D, TXT_CH], ctx),
        _arc(_rand(D, UInt64(103), Float32(0.01)), [D], ctx),
        _arc(_rand(OUT_CH * D, UInt64(104), Float32(0.03)), [OUT_CH, D], ctx),
        _arc(_rand(OUT_CH, UInt64(105), Float32(0.01)), [OUT_CH], ctx),
        NUM_DOUBLE, NUM_SINGLE,
    )

    var names = List[String]()
    var tensors = List[TArc]()
    for bi in range(NUM_DOUBLE):
        _write_double_block(
            names, tensors,
            String("transformer_blocks.") + String(bi) + String("."),
            UInt64(1000 + bi * 100), ctx,
        )
    for bi in range(NUM_SINGLE):
        _write_single_block(
            names, tensors,
            String("single_transformer_blocks.") + String(bi) + String("."),
            UInt64(2000 + bi * 100), ctx,
        )
    save_safetensors(names, tensors, String(CKPT_PATH), ctx)

    # byte-identical inputs.
    var img_tokens = _rand(N_IMG * IN_CH, UInt64(800), Float32(0.50))
    var txt_tokens = _rand(N_TXT * TXT_CH, UInt64(801), Float32(0.50))
    var pooled = _rand(MOD_INDEX * D, UInt64(802), Float32(0.05))
    var cos = _rand(S * H * (Dh // 2), UInt64(803), Float32(0.50))
    var sin = _rand(S * H * (Dh // 2), UInt64(804), Float32(0.50))
    var d_out = _rand(N_IMG * OUT_CH, UInt64(805), Float32(0.25))

    var lora0 = build_flux_lora_set(NUM_DOUBLE, NUM_SINGLE, D, FMLP, RANK, ALPHA)
    var lora = _nonzero_lora_set(lora0, UInt64(9000))
    var n = total_adapters(lora)

    # ── HOST arm (oracle) ──
    var loader_h = _open_loader(ctx)
    var fwd_h = chroma_stack_lora_forward_offload[H, Dh, N_IMG, N_TXT, S](
        img_tokens.copy(), txt_tokens.copy(), pooled.copy(), MOD_INDEX,
        base, loader_h, lora, cos.copy(), sin.copy(),
        D, FMLP, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
    )
    var g_h = chroma_stack_lora_backward_offload[H, Dh, N_IMG, N_TXT, S](
        d_out.copy(), img_tokens.copy(), txt_tokens.copy(),
        base, loader_h, lora, cos.copy(), sin.copy(),
        fwd_h, D, FMLP, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
    )

    # ── DEVICE arm (new; recompute-in-backward, activations as TArc) ──
    #    FLASH=False PINS the math SDPA arm so the bit bars below compare
    #    device-math vs host-math (production default is FLASH=True cuDNN flash;
    #    its value-class numerics are exercised in the FLASH-arm section).
    var loader_d = _open_loader(ctx)
    var fwd_d = chroma_stack_lora_forward_device_offload[H, Dh, N_IMG, N_TXT, S, False](
        img_tokens.copy(), txt_tokens.copy(), pooled.copy(), MOD_INDEX,
        base, loader_d, lora, cos.copy(), sin.copy(),
        D, FMLP, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
    )
    var g_d = chroma_stack_lora_backward_device_offload[H, Dh, N_IMG, N_TXT, S, False](
        d_out.copy(), img_tokens.copy(), txt_tokens.copy(),
        base, loader_d, lora, cos.copy(), sin.copy(),
        fwd_d, D, FMLP, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
    )

    var allok = True

    # forward out.
    var fout = _cos(fwd_d.out, fwd_h.out)
    var okf = fout >= COS_BAR
    if not okf:
        allok = False
    print("---- forward out ----")
    print("  out cos=", fout, "  ", ("PASS" if okf else "FAIL"))
    print("  device nonfinite_lora_grads=", g_d.nonfinite_lora_grads,
          "  host nonfinite_lora_grads=", g_h.nonfinite_lora_grads)

    # per-slot LoRA d_a / d_b.
    print("---- LoRA grad slots (device vs host) ----")
    var worst_cos = Float64(1.0)
    var worst_l2 = Float64(0.0)
    for i in range(n):
        var ca = _cos(g_d.d_a[i], g_h.d_a[i])
        var la = _rel_l2(g_d.d_a[i], g_h.d_a[i])
        var cb = _cos(g_d.d_b[i], g_h.d_b[i])
        var lb = _rel_l2(g_d.d_b[i], g_h.d_b[i])
        var ok = ca >= COS_BAR and la <= L2_BAR and cb >= COS_BAR and lb <= L2_BAR
        if not ok:
            allok = False
        if ca < worst_cos:
            worst_cos = ca
        if cb < worst_cos:
            worst_cos = cb
        if la > worst_l2:
            worst_l2 = la
        if lb > worst_l2:
            worst_l2 = lb
        print("  slot", i,
              " d_a cos=", ca, " l2=", la,
              " | d_b cos=", cb, " l2=", lb,
              "  ", ("PASS" if ok else "FAIL"))

    print("---- summary ----")
    print("  slots=", n, " worst_cos=", worst_cos, " worst_rel_l2=", worst_l2)
    if g_d.nonfinite_lora_grads != 0:
        allok = False
        print("  FAIL: device produced nonfinite LoRA grads")
    if len(g_d.d_a) != n or len(g_d.d_b) != n:
        allok = False
        print("  FAIL: device grad flat count mismatch")

    # ── FLASH arm (value-class vs the math host oracle; production default is
    #    FLASH=True cuDNN flash). Bars are the qwen/sd35 flash precedent:
    #    stack out cos >= 0.999, a representative LoRA d_b cos >= 0.995. ──
    print("")
    print("################ FLASH ARM (value-class) ################")
    var loader_f = _open_loader(ctx)
    var fwd_f = chroma_stack_lora_forward_device_offload[H, Dh, N_IMG, N_TXT, S, True](
        img_tokens.copy(), txt_tokens.copy(), pooled.copy(), MOD_INDEX,
        base, loader_f, lora, cos.copy(), sin.copy(),
        D, FMLP, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
    )
    var g_f = chroma_stack_lora_backward_device_offload[H, Dh, N_IMG, N_TXT, S, True](
        d_out.copy(), img_tokens.copy(), txt_tokens.copy(),
        base, loader_f, lora, cos.copy(), sin.copy(),
        fwd_f, D, FMLP, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
    )
    var f_out = _cos(fwd_f.out, fwd_h.out)
    var f_db = _cos(g_f.d_b[0], g_h.d_b[0])
    var flash_ok = f_out >= 0.999 and f_db >= 0.995
    print("  flash cos(out) =", f_out, " cos(slot0 d_b) =", f_db,
          "  ", ("PASS" if flash_ok else "FAIL"))
    if not flash_ok:
        allok = False

    if allok:
        print("VERDICT: ALL GATES PASS -- chroma_stack_device_parity")
    else:
        raise Error("VERDICT: FAIL -- chroma_stack_device_parity diverged")

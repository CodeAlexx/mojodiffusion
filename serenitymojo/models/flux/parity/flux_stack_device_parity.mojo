# serenitymojo/models/flux/parity/flux_stack_device_parity.mojo
#
# STACK-LEVEL parity gate: DEVICE-RESIDENT flux stack (recompute-in-backward,
# activations as TArc via the gated device block) vs the HOST flux stack (the
# parity oracle), fwd+bwd, WITH stack-level LoRA ENABLED (the real 504-adapter
# recipe: block LoRA + per-block modulation Linears + embedders + input/final).
# Both arms run flux_stack_lora_*_offload_full over their OWN streamed loader on
# byte-identical inputs; the device arm additionally recomputes each block's
# forward in backward. Compares:
#   forward `out`                              : cos >= 0.9999
#   load-bearing input/embed grads             : cos >= 0.9999
#   every BLOCK LoRA d_a/d_b                    : cos >= 0.9999
#   every STACK LoRA st_d_a/st_d_b              : cos >= 0.9999
# (device vs host is expected BIT-IDENTICAL — the block is bit-identical and the
#  cold stack-level math is the same host code.)
#
# Synthetic reduced-depth checkpoint (REAL per-block H/Dh/D). Adapted from
# flux_offload_equiv_parity.mojo.
#
# Run (GPU; check `nvidia-smi` is idle first):
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg
#   pixi run mojo build --optimization-level 2 -I . -I /home/alex/MOJO-libs \
#     -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     -Xlinker -rpath -Xlinker /home/alex/mojodiffusion/serenitymojo/ops/cshim/lib \
#     serenitymojo/models/flux/parity/flux_stack_device_parity.mojo -o /tmp/flux_stack_dev_parity
#   /tmp/flux_stack_dev_parity

from std.collections import List, Optional
from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.parity import ParityHarness
from serenitymojo.training.train_step import LoraAdapter

from serenitymojo.models.flux.block import (
    StreamWeights, DoubleBlockWeights, SingleBlockWeights,
)
from serenitymojo.models.flux.flux_stack import (
    FluxStackBase, EmbedMlp, ModLin, DoubleModLin,
)
from serenitymojo.models.flux.flux_stack_lora import (
    FluxLoraSet, FluxStackLoraSet, total_adapters, total_stack_adapters,
    flux_stack_lora_forward_offload_full, flux_stack_lora_backward_offload_full,
    flux_stack_lora_forward_device_offload_full, flux_stack_lora_backward_device_offload_full,
)
from serenitymojo.offload.plan import build_flux_block_plan, OffloadConfig
from serenitymojo.offload.turbo_planned_loader import TurboPlannedLoader


comptime TArc = ArcPointer[Tensor]
comptime CKPT_PATH = "/tmp/flux_stack_device_parity_model.safetensors"

comptime H = 24
comptime Dh = 128
comptime D = H * Dh            # 3072
comptime N_IMG = 4
comptime N_TXT = 3
comptime S = N_TXT + N_IMG
comptime FMLP = 32
comptime IN_CH = 64
comptime TXT_CH = 40
comptime OUT_CH = 64
comptime T_DIM = 16
comptime VEC_DIM = 20
comptime NUM_DOUBLE = 2
comptime NUM_SINGLE = 2
comptime EPS = Float32(1e-06)
comptime MAX_PERIOD = Float32(10000.0)
comptime RANK = 4
comptime ALPHA = Float32(2.0)
comptime LSCALE = ALPHA / Float32(RANK)


def _rand(n: Int, seed: UInt64, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    for _ in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
        out.append((u - Float32(0.5)) * scale)
    return out^


def _zeros(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n):
        o.append(0.0)
    return o^


# LoRA adapter with NONZERO A and B (so every grad arm is exercised).
def _adapter(in_f: Int, out_f: Int, seed: UInt64) -> LoraAdapter:
    return LoraAdapter(
        _rand(RANK * in_f, seed, 0.05), _rand(out_f * RANK, seed + 7777, 0.05),
        RANK, in_f, out_f, LSCALE,
        _zeros(RANK * in_f), _zeros(RANK * in_f),
        _zeros(out_f * RANK), _zeros(out_f * RANK),
    )


def _opt_ad(in_f: Int, out_f: Int, seed: UInt64) -> Optional[LoraAdapter]:
    return Optional[LoraAdapter](_adapter(in_f, out_f, seed))


# ── per-block weight lists ───────────────────────────────────────────────────
struct _StreamLists(Copyable, Movable):
    var wqkv: List[Float32]
    var bqkv: List[Float32]
    var wproj: List[Float32]
    var bproj: List[Float32]
    var wmlp0: List[Float32]
    var bmlp0: List[Float32]
    var wmlp2: List[Float32]
    var bmlp2: List[Float32]
    var q_norm: List[Float32]
    var k_norm: List[Float32]

    def __init__(
        out self,
        var wqkv: List[Float32], var bqkv: List[Float32],
        var wproj: List[Float32], var bproj: List[Float32],
        var wmlp0: List[Float32], var bmlp0: List[Float32],
        var wmlp2: List[Float32], var bmlp2: List[Float32],
        var q_norm: List[Float32], var k_norm: List[Float32],
    ):
        self.wqkv = wqkv^; self.bqkv = bqkv^
        self.wproj = wproj^; self.bproj = bproj^
        self.wmlp0 = wmlp0^; self.bmlp0 = bmlp0^
        self.wmlp2 = wmlp2^; self.bmlp2 = bmlp2^
        self.q_norm = q_norm^; self.k_norm = k_norm^


def _gen_stream(seed: UInt64) -> _StreamLists:
    return _StreamLists(
        _rand(3 * D * D, seed + 1, 0.02), _rand(3 * D, seed + 2, 0.02),
        _rand(D * D, seed + 3, 0.02), _rand(D, seed + 4, 0.02),
        _rand(FMLP * D, seed + 5, 0.02), _rand(FMLP, seed + 6, 0.02),
        _rand(D * FMLP, seed + 7, 0.02), _rand(D, seed + 8, 0.02),
        _rand(Dh, seed + 9, 0.1), _rand(Dh, seed + 10, 0.1),
    )


struct _SingleLists(Copyable, Movable):
    var w1: List[Float32]
    var b1: List[Float32]
    var w2: List[Float32]
    var b2: List[Float32]
    var q_norm: List[Float32]
    var k_norm: List[Float32]

    def __init__(
        out self,
        var w1: List[Float32], var b1: List[Float32],
        var w2: List[Float32], var b2: List[Float32],
        var q_norm: List[Float32], var k_norm: List[Float32],
    ):
        self.w1 = w1^; self.b1 = b1^; self.w2 = w2^; self.b2 = b2^
        self.q_norm = q_norm^; self.k_norm = k_norm^


def _gen_single(seed: UInt64) -> _SingleLists:
    return _SingleLists(
        _rand((3 * D + FMLP) * D, seed + 1, 0.02), _rand(3 * D + FMLP, seed + 2, 0.02),
        _rand(D * (D + FMLP), seed + 3, 0.02), _rand(D, seed + 4, 0.02),
        _rand(Dh, seed + 5, 0.1), _rand(Dh, seed + 6, 0.1),
    )


def _add(
    mut names: List[String], mut tensors: List[TArc],
    name: String, vals: List[Float32], var shape: List[Int], ctx: DeviceContext,
) raises:
    # Real flux1-dev blocks are bf16; write bf16 so the streamed block weights
    # match the bf16 base + bf16 activations (the block forward is bf16-native).
    names.append(name)
    tensors.append(TArc(Tensor.from_host(vals.copy(), shape^, STDtype.BF16, ctx)))


def _check(
    mut harness: ParityHarness, name: String,
    actual: List[Float32], expected: List[Float32],
    mut allok: Bool, mut npass: Int, mut nfail: Int,
) raises:
    var r = harness.compare_host(actual, expected)
    if not r.passed:
        print("  cos(", name, ") =", r.cos, "  max_abs =", r.max_abs, "  n =", r.n, "   FAIL")
        allok = False
        nfail += 1
    else:
        npass += 1


def _build_sset() -> FluxStackLoraSet:
    # nonzero A/B; ST_* level order + per-block mod order.
    var seed = UInt64(600000)
    var level = List[Optional[LoraAdapter]]()
    level.append(_opt_ad(TXT_CH, D, seed)); seed += 1     # ST_CTX_EMB
    level.append(_opt_ad(IN_CH, D, seed)); seed += 1      # ST_X_EMB
    level.append(_opt_ad(T_DIM, D, seed)); seed += 1      # ST_TIME_1
    level.append(_opt_ad(D, D, seed)); seed += 1          # ST_TIME_2
    level.append(_opt_ad(VEC_DIM, D, seed)); seed += 1    # ST_TEXT_1
    level.append(_opt_ad(D, D, seed)); seed += 1          # ST_TEXT_2
    level.append(_opt_ad(T_DIM, D, seed)); seed += 1      # ST_GUID_1
    level.append(_opt_ad(D, D, seed)); seed += 1          # ST_GUID_2
    level.append(_opt_ad(D, 2 * D, seed)); seed += 1      # ST_NORM_OUT
    level.append(_opt_ad(D, OUT_CH, seed)); seed += 1     # ST_PROJ_OUT
    var dbl_img = List[Optional[LoraAdapter]]()
    var dbl_txt = List[Optional[LoraAdapter]]()
    for _ in range(NUM_DOUBLE):
        dbl_img.append(_opt_ad(D, 6 * D, seed)); seed += 1
        dbl_txt.append(_opt_ad(D, 6 * D, seed)); seed += 1
    var sgl = List[Optional[LoraAdapter]]()
    for _ in range(NUM_SINGLE):
        sgl.append(_opt_ad(D, 3 * D, seed)); seed += 1
    return FluxStackLoraSet(level^, dbl_img^, dbl_txt^, sgl^, NUM_DOUBLE, NUM_SINGLE, RANK, True)


def main() raises:
    var ctx = DeviceContext()
    print("==== flux_stack_device_parity (DEVICE recompute vs HOST offload; stack LoRA ON) ====")
    print("H=", H, " Dh=", Dh, " D=", D, " N_IMG=", N_IMG, " N_TXT=", N_TXT,
          " FMLP=", FMLP, " RANK=", RANK, " num_double=", NUM_DOUBLE, " num_single=", NUM_SINGLE)

    # ── stack-level base ──
    var time_in = EmbedMlp(_rand(D * T_DIM, 100, 0.02), _rand(D, 101, 0.02),
                           _rand(D * D, 102, 0.02), _rand(D, 103, 0.02), T_DIM, D, ctx)
    var guid_in = EmbedMlp(_rand(D * T_DIM, 110, 0.02), _rand(D, 111, 0.02),
                           _rand(D * D, 112, 0.02), _rand(D, 113, 0.02), T_DIM, D, ctx)
    var vec_in = EmbedMlp(_rand(D * VEC_DIM, 120, 0.02), _rand(D, 121, 0.02),
                          _rand(D * D, 122, 0.02), _rand(D, 123, 0.02), VEC_DIM, D, ctx)
    var dbl_mod = List[DoubleModLin]()
    for bi in range(NUM_DOUBLE):
        var sd = UInt64(200 + bi * 10)
        var im = ModLin(_rand(6 * D * D, sd + 1, 0.02), _rand(6 * D, sd + 2, 0.02), 6 * D, D, ctx)
        var tm = ModLin(_rand(6 * D * D, sd + 3, 0.02), _rand(6 * D, sd + 4, 0.02), 6 * D, D, ctx)
        dbl_mod.append(DoubleModLin(im^, tm^))
    var sgl_mod = List[ModLin]()
    for bi in range(NUM_SINGLE):
        var sd = UInt64(300 + bi * 10)
        sgl_mod.append(ModLin(_rand(3 * D * D, sd + 1, 0.02), _rand(3 * D, sd + 2, 0.02), 3 * D, D, ctx))
    var base = FluxStackBase(
        _rand(D * IN_CH, 400, 0.02), _rand(D, 401, 0.02),
        _rand(D * TXT_CH, 402, 0.02), _rand(D, 403, 0.02),
        time_in^, True, guid_in^, vec_in^,
        dbl_mod^, sgl_mod^,
        _rand(2 * D * D, 404, 0.02), _rand(2 * D, 405, 0.02),
        _rand(OUT_CH * D, 406, 0.02), _rand(OUT_CH, 407, 0.02),
        D, IN_CH, TXT_CH, OUT_CH, ctx,
    )

    # ── per-block weight lists ──
    var dbl_lists = List[_StreamLists]()
    for bi in range(NUM_DOUBLE):
        dbl_lists.append(_gen_stream(UInt64(1000 + bi * 100)))       # img
        dbl_lists.append(_gen_stream(UInt64(1000 + bi * 100 + 50)))  # txt
    var sgl_lists = List[_SingleLists]()
    for bi in range(NUM_SINGLE):
        sgl_lists.append(_gen_single(UInt64(5000 + bi * 100)))

    # ── block LoRA set (NONZERO B; flat order) ──
    var ad = List[LoraAdapter]()
    var seed = UInt64(70000)
    for _ in range(NUM_DOUBLE):
        for _stream in range(2):
            ad.append(_adapter(D, D, seed)); seed += 1     # to_q
            ad.append(_adapter(D, D, seed)); seed += 1     # to_k
            ad.append(_adapter(D, D, seed)); seed += 1     # to_v
            ad.append(_adapter(D, D, seed)); seed += 1     # proj
            ad.append(_adapter(D, FMLP, seed)); seed += 1  # mlp0
            ad.append(_adapter(FMLP, D, seed)); seed += 1  # mlp2
    for _ in range(NUM_SINGLE):
        ad.append(_adapter(D, D, seed)); seed += 1         # to_q
        ad.append(_adapter(D, D, seed)); seed += 1         # to_k
        ad.append(_adapter(D, D, seed)); seed += 1         # to_v
        ad.append(_adapter(D, FMLP, seed)); seed += 1      # proj_mlp
        ad.append(_adapter(D + FMLP, D, seed)); seed += 1  # linear2
    var lora = FluxLoraSet(ad^, NUM_DOUBLE, NUM_SINGLE, RANK)
    var sset = _build_sset()

    # ── inputs ──
    var img_tokens = _rand(N_IMG * IN_CH, 800, 1.0)
    var txt_tokens = _rand(N_TXT * TXT_CH, 801, 1.0)
    var timestep = _rand(1, 802, 1.0)
    var guidance = Optional[List[Float32]](_rand(1, 803, 1.0))
    var vector = _rand(VEC_DIM, 804, 1.0)
    var cos = _rand(S * H * (Dh // 2), 805, 1.0)
    var sin = _rand(S * H * (Dh // 2), 806, 1.0)
    var d_out = _rand(N_IMG * OUT_CH, 807, 1.0)

    # ── write reduced-depth checkpoint (BFL block keys) ──
    print("---- writing reduced-depth checkpoint to ", CKPT_PATH, " ----")
    var names = List[String]()
    var tensors = List[TArc]()
    for bi in range(NUM_DOUBLE):
        var dp = String("double_blocks.") + String(bi)
        var streams: List[String] = ["img", "txt"]
        for si in range(2):
            ref sl = dbl_lists[bi * 2 + si]
            var ap = dp + "." + streams[si] + "_attn"
            var mp = dp + "." + streams[si] + "_mlp"
            _add(names, tensors, ap + ".qkv.weight", sl.wqkv, [3 * D, D], ctx)
            _add(names, tensors, ap + ".qkv.bias", sl.bqkv, [3 * D], ctx)
            _add(names, tensors, ap + ".proj.weight", sl.wproj, [D, D], ctx)
            _add(names, tensors, ap + ".proj.bias", sl.bproj, [D], ctx)
            _add(names, tensors, mp + ".0.weight", sl.wmlp0, [FMLP, D], ctx)
            _add(names, tensors, mp + ".0.bias", sl.bmlp0, [FMLP], ctx)
            _add(names, tensors, mp + ".2.weight", sl.wmlp2, [D, FMLP], ctx)
            _add(names, tensors, mp + ".2.bias", sl.bmlp2, [D], ctx)
            _add(names, tensors, ap + ".norm.query_norm.scale", sl.q_norm, [Dh], ctx)
            _add(names, tensors, ap + ".norm.key_norm.scale", sl.k_norm, [Dh], ctx)
    for bi in range(NUM_SINGLE):
        var sp = String("single_blocks.") + String(bi)
        ref sl = sgl_lists[bi]
        _add(names, tensors, sp + ".linear1.weight", sl.w1, [3 * D + FMLP, D], ctx)
        _add(names, tensors, sp + ".linear1.bias", sl.b1, [3 * D + FMLP], ctx)
        _add(names, tensors, sp + ".linear2.weight", sl.w2, [D, D + FMLP], ctx)
        _add(names, tensors, sp + ".linear2.bias", sl.b2, [D], ctx)
        _add(names, tensors, sp + ".norm.query_norm.scale", sl.q_norm, [Dh], ctx)
        _add(names, tensors, sp + ".norm.key_norm.scale", sl.k_norm, [Dh], ctx)
    save_safetensors(names, tensors, String(CKPT_PATH), ctx)

    var plan_cfg = OffloadConfig.synchronous_single()

    # ════════════════════════ HOST arm (oracle) ════════════════════════
    print("---- HOST offload_full arm (oracle) ----")
    var plan_h = build_flux_block_plan(NUM_DOUBLE, NUM_SINGLE)
    var loader_h = TurboPlannedLoader.open(String(CKPT_PATH), plan_h^, plan_cfg, ctx)
    var fwd_h = flux_stack_lora_forward_offload_full[H, Dh, N_IMG, N_TXT, S](
        img_tokens.copy(), txt_tokens.copy(), timestep.copy(), guidance, vector.copy(),
        base, loader_h, lora, sset, cos.copy(), sin.copy(),
        D, FMLP, IN_CH, TXT_CH, OUT_CH, T_DIM, VEC_DIM, EPS, ctx,
    )
    var g_h = flux_stack_lora_backward_offload_full[H, Dh, N_IMG, N_TXT, S](
        d_out.copy(), img_tokens.copy(), txt_tokens.copy(), base, loader_h, lora, sset, vector.copy(),
        cos.copy(), sin.copy(), fwd_h,
        D, FMLP, IN_CH, TXT_CH, OUT_CH, T_DIM, VEC_DIM, MAX_PERIOD, EPS, ctx,
    )

    # ════════════════════════ DEVICE arm (under test) ════════════════════════
    #    FLASH=False PINS the math SDPA arm so the bit bars below compare
    #    device-math vs host-math (production default is FLASH=True cuDNN flash;
    #    its value-class numerics are exercised in the FLASH-arm section).
    print("---- DEVICE recompute arm (under test) ----")
    var plan_d = build_flux_block_plan(NUM_DOUBLE, NUM_SINGLE)
    var loader_d = TurboPlannedLoader.open(String(CKPT_PATH), plan_d^, plan_cfg, ctx)
    var fwd_d = flux_stack_lora_forward_device_offload_full[H, Dh, N_IMG, N_TXT, S, False](
        img_tokens.copy(), txt_tokens.copy(), timestep.copy(), guidance, vector.copy(),
        base, loader_d, lora, sset, cos.copy(), sin.copy(),
        D, FMLP, IN_CH, TXT_CH, OUT_CH, T_DIM, VEC_DIM, EPS, ctx,
    )
    var g_d = flux_stack_lora_backward_device_offload_full[H, Dh, N_IMG, N_TXT, S, False](
        d_out.copy(), img_tokens.copy(), txt_tokens.copy(), base, loader_d, lora, sset, vector.copy(),
        cos.copy(), sin.copy(), fwd_d,
        D, FMLP, IN_CH, TXT_CH, OUT_CH, T_DIM, VEC_DIM, MAX_PERIOD, EPS, ctx,
    )

    # ════════════════════════ COMPARE ════════════════════════
    var harness = ParityHarness(0.9999)
    var allok = True
    var npass = 0
    var nfail = 0

    print("")
    print("---- forward output ----")
    _check(harness, "out", fwd_d.out, fwd_h.out, allok, npass, nfail)

    print("---- load-bearing input/embed grads ----")
    _check(harness, "d_img_tokens", g_d.d_img_tokens, g_h.d_img_tokens, allok, npass, nfail)
    _check(harness, "d_txt_tokens", g_d.d_txt_tokens, g_h.d_txt_tokens, allok, npass, nfail)
    _check(harness, "d_vec", g_d.d_vec, g_h.d_vec, allok, npass, nfail)
    _check(harness, "d_timestep", g_d.d_timestep, g_h.d_timestep, allok, npass, nfail)
    _check(harness, "d_guidance", g_d.d_guidance, g_h.d_guidance, allok, npass, nfail)
    _check(harness, "d_vector", g_d.d_vector, g_h.d_vector, allok, npass, nfail)

    print("---- ALL block LoRA d_A/d_B (only FAILs printed) ----")
    var n = total_adapters(lora)
    for i in range(n):
        _check(harness, String("d_a[") + String(i) + "]", g_d.d_a[i], g_h.d_a[i], allok, npass, nfail)
        _check(harness, String("d_b[") + String(i) + "]", g_d.d_b[i], g_h.d_b[i], allok, npass, nfail)

    print("---- ALL stack LoRA st_d_A/st_d_B (only FAILs printed) ----")
    var ns = total_stack_adapters(sset)
    for i in range(ns):
        _check(harness, String("st_d_a[") + String(i) + "]", g_d.st_d_a[i], g_h.st_d_a[i], allok, npass, nfail)
        _check(harness, String("st_d_b[") + String(i) + "]", g_d.st_d_b[i], g_h.st_d_b[i], allok, npass, nfail)

    # ════════════════════════ FLASH arm (value-class) ════════════════════════
    #    Production default is FLASH=True cuDNN flash. Bars are the qwen/sd35
    #    flash precedent: stack out cos >= 0.999, representative LoRA d_b >= 0.995.
    print("")
    print("---- FLASH arm (value-class vs host math oracle) ----")
    var plan_f = build_flux_block_plan(NUM_DOUBLE, NUM_SINGLE)
    var loader_f = TurboPlannedLoader.open(String(CKPT_PATH), plan_f^, plan_cfg, ctx)
    var fwd_f = flux_stack_lora_forward_device_offload_full[H, Dh, N_IMG, N_TXT, S, True](
        img_tokens.copy(), txt_tokens.copy(), timestep.copy(), guidance, vector.copy(),
        base, loader_f, lora, sset, cos.copy(), sin.copy(),
        D, FMLP, IN_CH, TXT_CH, OUT_CH, T_DIM, VEC_DIM, EPS, ctx,
    )
    var g_f = flux_stack_lora_backward_device_offload_full[H, Dh, N_IMG, N_TXT, S, True](
        d_out.copy(), img_tokens.copy(), txt_tokens.copy(), base, loader_f, lora, sset, vector.copy(),
        cos.copy(), sin.copy(), fwd_f,
        D, FMLP, IN_CH, TXT_CH, OUT_CH, T_DIM, VEC_DIM, MAX_PERIOD, EPS, ctx,
    )
    var fharness = ParityHarness(0.999)
    var f_out = fharness.compare_host(fwd_f.out, fwd_h.out).cos
    var f_db = fharness.compare_host(g_f.d_b[0], g_h.d_b[0]).cos
    var flash_ok = f_out >= 0.999 and f_db >= 0.995 and g_f.nonfinite_lora_grads == 0
    print("  flash cos(out) =", f_out, " cos(d_b[0]) =", f_db,
          " nonfinite =", g_f.nonfinite_lora_grads,
          "  ", ("PASS" if flash_ok else "FAIL"))
    if not flash_ok:
        allok = False

    print("")
    print("checks: PASS =", npass, " FAIL =", nfail, " (block adapters =", n, " stack adapters =", ns, ")")
    print("device nonfinite_lora_grads =", g_d.nonfinite_lora_grads)
    if allok and g_d.nonfinite_lora_grads == 0:
        print("VERDICT: PASS — device stack bit-faithful to host (out + all block+stack grads cos>=0.9999, 0 nonfinite)")
    else:
        print("VERDICT: FAIL — at least one tensor diverged (see FAIL lines)")

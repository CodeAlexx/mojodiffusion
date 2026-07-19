# serenitymojo/models/klein/parity/klein_stack_loha_real_smoke.mojo
#
# REAL-WEIGHT FINITE + STEP + SAVE SMOKE for the Klein FULL DiT STACK driven by a
# LoHa ADAPTER SET through the SHARED (a,b) CARRIER (the dispatch path).
# Twin of klein_stack_lokr_real_smoke.mojo for the LoHa family. The LoHa
# Hadamard delta factors into an R_eff=R² carrier (loha_stack.mojo), so it runs
# through klein_stack_lora_forward/backward with NO stack/kernel change.
#
# Run: cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#   pixi run mojo build --optimization-level 2 -I . -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     serenitymojo/models/klein/parity/klein_stack_loha_real_smoke.mojo \
#     -o /tmp/klein_stack_loha_real_smoke && \
#   LD_LIBRARY_PATH="serenitymojo/ops/cshim/lib:$HOME/.local/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH" \
#     /tmp/klein_stack_loha_real_smoke

from std.collections import List
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.models.klein.double_block import DoubleBlockWeights
from serenitymojo.models.klein.single_block import SingleBlockWeights
from serenitymojo.models.klein.klein_stack import KleinStackBase
from serenitymojo.models.klein.klein_stack_lora import (
    KleinLoraSet,
    klein_stack_lora_forward, klein_stack_lora_backward,
    DBL_SLOTS,
)
from serenitymojo.models.klein.weights import (
    load_double_block_weights, load_single_block_weights,
    load_klein_stack_base, build_klein_vec_silu,
    build_klein_double_modvecs, build_klein_single_modvecs,
)
from serenitymojo.training.loha_stack import (
    KleinLoHaSet, build_klein_loha_set, klein_loha_carrier_lists,
    klein_loha_chain_all, klein_loha_adamw_step, save_klein_loha,
    lokr_loha_carrier_total_bytes, klein_loha_zero_leg_l1, klein_loha_trainable_l1,
)


comptime KLEIN9B_PATH = "/home/alex/.serenity/models/checkpoints/flux-2-klein-base-9b.safetensors"
comptime SAVE_PATH = "/tmp/klein_stack_loha_smoke.safetensors"

comptime H = 32
comptime Dh = 128
comptime N_IMG = 4
comptime N_TXT = 2
comptime S = N_IMG + N_TXT
comptime NUM_DOUBLE = 8
comptime NUM_SINGLE = 24
comptime TIMESTEP_DIM = 256
comptime RANK = 4
comptime ALPHA = Float32(8.0)
comptime LOHA_TARGETS = 3          # 1=attn, 2=attn+ff, 3=all


def _fill(n: Int, seed: UInt64, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    for _ in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
        out.append((u - Float32(0.5)) * scale)
    return out^


def _all_finite(h: List[Float32]) -> Bool:
    for i in range(len(h)):
        var v = h[i]
        if v != v:
            return False
        var a = v if v >= 0.0 else -v
        if a > Float32(1.0e30):
            return False
    return True


def main() raises:
    var ctx = DeviceContext()
    var D = 4096
    var F = 12288
    var IN_CH = 128
    var TXT_CH = 12288
    var OUT_CH = 128
    var eps = Float32(1.0e-6)

    print("=== Klein-9B FULL STACK + LoHa-CARRIER real-weight FINITE + step + save smoke ===")
    print("  path:", KLEIN9B_PATH)
    print("  D=", D, " F=", F, " RANK=", RANK, " ALPHA=", ALPHA, " targets=", LOHA_TARGETS)

    var st = SafeTensors.open(KLEIN9B_PATH)

    var ts = Tensor.from_host([Float32(0.5)], [1], STDtype.F32, ctx)
    var vec_silu = build_klein_vec_silu(st, ts, TIMESTEP_DIM, D, ctx)
    if not _all_finite(vec_silu):
        raise Error("vec_silu not finite")
    var img_mod = build_klein_double_modvecs(st, vec_silu, String("img"), D, ctx)
    var txt_mod = build_klein_double_modvecs(st, vec_silu, String("txt"), D, ctx)
    var single_mod = build_klein_single_modvecs(st, vec_silu, D, ctx)

    var base = load_klein_stack_base(st, vec_silu, D, ctx)

    var dbw = List[DoubleBlockWeights]()
    for bi in range(NUM_DOUBLE):
        dbw.append(load_double_block_weights(st, bi, ctx))
    var sbw = List[SingleBlockWeights]()
    for bi in range(NUM_SINGLE):
        sbw.append(load_single_block_weights(st, bi, ctx))
    print("  loaded", len(dbw), "double +", len(sbw), "single block weights")

    # ── build LoHa MASTER set, materialize (a,b) carriers ──
    var masters = build_klein_loha_set(NUM_DOUBLE, NUM_SINGLE, D, F, RANK, ALPHA, LOHA_TARGETS, 12345)
    var carrier_bytes = lokr_loha_carrier_total_bytes(masters)
    print("  built KleinLoHaSet masters; carrier set =", carrier_bytes // (1024 * 1024), "MB bf16")
    var carriers = klein_loha_carrier_lists(masters, D, F)
    var carrier_set = KleinLoraSet(carriers[0].copy(), carriers[1].copy(),
                                   NUM_DOUBLE, NUM_SINGLE, RANK)
    var total_adapters = NUM_DOUBLE * DBL_SLOTS + NUM_SINGLE * 2
    print("  materialized carriers:", len(carrier_set.dbl), "dbl +",
          len(carrier_set.sgl), "sgl (expect", total_adapters, ")")

    var img_tokens = _fill(N_IMG * IN_CH, 100, 1.0)
    var txt_tokens = _fill(N_TXT * TXT_CH, 200, 1.0)
    var cos = _fill(S * H * (Dh // 2), 500, 1.0)
    var sin = _fill(S * H * (Dh // 2), 600, 1.0)

    print("  running full-depth carrier forward ...")
    var fwd = klein_stack_lora_forward[H, Dh, N_IMG, N_TXT, S](
        img_tokens.copy(), txt_tokens.copy(), base,
        dbw, sbw, carrier_set, img_mod, txt_mod, single_mod, cos.copy(), sin.copy(),
        D, F, IN_CH, TXT_CH, OUT_CH, eps, ctx,
    )
    if len(fwd.out) != N_IMG * OUT_CH:
        raise Error("forward output shape wrong")
    if not _all_finite(fwd.out):
        raise Error("FORWARD output not finite on real weights")
    print("  forward output FINITE ✓ (len", len(fwd.out), ")")

    print("  running full-depth carrier backward ...")
    var d_out = _fill(N_IMG * OUT_CH, 700, 0.05)
    var g = klein_stack_lora_backward[H, Dh, N_IMG, N_TXT, S](
        d_out, img_tokens.copy(), txt_tokens.copy(), base,
        dbw, sbw, carrier_set, img_mod, txt_mod, single_mod, cos.copy(), sin.copy(), fwd,
        D, F, IN_CH, TXT_CH, OUT_CH, eps, ctx,
    )
    var ok = True
    var nd = NUM_DOUBLE * DBL_SLOTS
    var ns = NUM_SINGLE * 2
    for i in range(nd):
        if not _all_finite(g.dbl_d_a[i]) or not _all_finite(g.dbl_d_b[i]):
            print("  double carrier grad slot", i, "NOT finite"); ok = False
    for i in range(ns):
        if not _all_finite(g.sgl_d_a[i]) or not _all_finite(g.sgl_d_b[i]):
            print("  single carrier grad slot", i, "NOT finite"); ok = False
    if not ok:
        raise Error("non-finite carrier grad in real-weight smoke")
    print("  all", total_adapters, "carrier d_A/d_B FINITE ✓")

    print("  chaining carrier grads → LoHa master grads ...")
    var loha_grads = klein_loha_chain_all(masters, g.dbl_d_a, g.dbl_d_b, g.sgl_d_a, g.sgl_d_b)
    var zero_before = klein_loha_zero_leg_l1(masters)
    var train_before = klein_loha_trainable_l1(masters)
    klein_loha_adamw_step(masters, loha_grads, 1, Float32(1.0e-4),
                          Float32(0.9), Float32(0.999), Float32(1.0e-8), Float32(0.01))
    var zero_after = klein_loha_zero_leg_l1(masters)
    var train_after = klein_loha_trainable_l1(masters)
    print("  w2a zero-leg L1: before =", zero_before, " after =", zero_after)
    print("  trainable L1:     before =", train_before, " after =", train_after)
    if zero_after <= zero_before:
        raise Error("AdamW did not move the w2a zero-leg off zero (master no-op?)")
    print("  master AdamW updated the zero-leg ✓")

    print("  saving with save_klein_loha ...")
    var npairs = save_klein_loha(masters, SAVE_PATH, ctx)
    print("  save_klein_loha wrote", npairs, "modules")
    if npairs <= 0:
        raise Error("save_klein_loha wrote nothing")
    var st2 = SafeTensors.open(SAVE_PATH)
    print("  reopened", SAVE_PATH, "✓")

    print("ALL GATES PASS — klein_stack_loha_real_smoke (LoHa dispatch e2e on real Klein-9B)")

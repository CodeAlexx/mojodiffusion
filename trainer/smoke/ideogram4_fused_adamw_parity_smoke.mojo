# ideogram4_fused_adamw_parity_smoke.mojo — MJ-1038 fixture smoke: OLD
# per-tensor adamw_step loop vs the ONE fused multitensor shared-ABI update.
#
# WHAT IT PROVES (when run): starting from bit-identical BF16 params and
# BF16 grads,
#   branch OLD   = the pre-MJ-1038 ideogram4 optimizer (per-tensor
#                  adamw_extensions.adamw_step, BF16 m/v moments, torch-RNE
#                  param write-back, stochastic_rounding=False here so the
#                  compare is deterministic),
#   branch FUSED = the IDEOGRAM4_FUSED_ADAMW path (shared
#                  DeviceTrainableSet/DeviceGradSet/DeviceAdamWState ->
#                  device_adamw_train_step_update -> fused_adamw_step; F32
#                  m/v moments, plain RNE cast write-back, max_grad_norm=0.0
#                  = clip disabled, matching IDEOGRAM4_GRAD_CLIP=False),
# produce post-step params equal within ULP-CLASS tolerance. The paths are
# intentionally NOT bit-identical (documented in Ideogram4StackTrain.mojo:
# BF16- vs F32-quantized moments, torch-RNE vs plain RNE cast); the expected
# per-element gap is a few BF16 ulps, so the gate is
#   worst_rel = max |p_old - p_fused| / max(|p_old|, |p_fused|, 1e-8)
#             <= 2^-5  (= 0.03125 = 4 BF16 ulps at the BOTTOM of a binade,
#   where BF16's relative ulp peaks at 2^-7; first run measured worst_abs
#   1.9073486e-06 = exactly 4 ulps of the [2^-14,2^-13) binade on an element
#   of magnitude ~9.9e-5, rel 0.01932 — the original 2^-6 constant assumed
#   the top-of-binade 2^-8 relative ulp and rejected that in-spec 4-ulp gap).
# It also proves the buf.copy() ALIASING packing actually updates the live
# param tensors in place (changed-element count > 0) and that the fused path
# returns a positive device grad_norm scalar.
#
# NOTE stochastic_rounding: the real recipe default is True; the fused path
# does not implement SR (flip IDEOGRAM4_FUSED_ADAMW=False to restore it). SR
# would add +/-1 BF16 ulp of unbiased rounding noise per element, which is
# inside the same ulp-class bound but makes the compare non-deterministic —
# hence SR=False here.
#
# BUILD (from trainer, the ideogram4 smoke pattern in
# pixi.toml ideogram4-live-trainer-build; no cudnn shim needed — this smoke
# touches no SDPA):
#   mojo build -I . -I src -I /home/alex/mojodiffusion \
#     -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -L/home/alex/mojodiffusion/.pixi/envs/default/lib \
#     smoke/ideogram4_fused_adamw_parity_smoke.mojo \
#     -o target/ideogram4_fused_adamw_parity_smoke
# WRITTEN, NOT BUILT/RUN in the MJ-1038 edit session (campaign rule).

from max.gpu.host import DeviceContext
from std.math import max
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.tensor_algebra import zeros_device
from serenitymojo.training.device_train_step import (
    DeviceAdamWState,
    DeviceGradSet,
    DeviceTrainableSet,
    device_adamw_train_step_update,
)

from serenity_trainer.util.optimizer.adamw_extensions import adamw_step


comptime TArc = ArcPointer[Tensor]

# The ideogram4 recipe scalars (TrainConfig.adamw_lora_defaults magnitudes).
comptime LR = Float32(1.0e-4)
comptime BETA1 = Float32(0.9)
comptime BETA2 = Float32(0.99)
comptime EPS = Float32(1.0e-8)
comptime WD = Float32(0.01)
comptime STEP_T = 1
# ulp-class gate: 2^-5 relative = 4 BF16 ulps at bottom-of-binade (relative
# ulp 2^-7 there; see header — the first run measured a genuine 4-ulp gap at
# rel 0.01932 that the old 2^-6 constant mis-rejected).
comptime REL_TOL = Float32(0.03125)


def _numel(shape: List[Int]) -> Int:
    var n = 1
    for i in range(len(shape)):
        n *= shape[i]
    return n


# Deterministic pseudo-values in (-scale/2, +scale/2); scale=0 -> zero-init
# (the real LoRA-B start state).
def _fill(n: Int, scale: Float32, salt: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        var h = (i * 37 + salt * 101 + 13) % 997
        out.append((Float32(h) / Float32(997.0) - Float32(0.5)) * scale)
    return out^


def main() raises:
    var ctx = DeviceContext()

    # 3 synthetic adapters x (a, b) = 6 tensors of varied shape (exercises the
    # fused kernel's per-tensor offset table). b-like tensors at indices 1 and
    # 3 are ZERO-INIT, matching the real LoRA-B start state.
    var shapes = List[List[Int]]()
    var s0: List[Int] = [8, 64]
    var s1: List[Int] = [96, 8]
    var s2: List[Int] = [4, 160]
    var s3: List[Int] = [80, 4]
    var s4: List[Int] = [16, 48]
    var s5: List[Int] = [32, 16]
    shapes.append(s0^)
    shapes.append(s1^)
    shapes.append(s2^)
    shapes.append(s3^)
    shapes.append(s4^)
    shapes.append(s5^)
    var p_scales: List[Float32] = [0.05, 0.0, 0.05, 0.0, 0.05, 0.02]

    var p_old = List[TArc]()
    var p_fused = List[TArc]()
    var p_init_host = List[List[Float32]]()
    var g_shared = List[TArc]()
    var m_old = List[TArc]()
    var v_old = List[TArc]()
    var m_f = List[TArc]()
    var v_f = List[TArc]()

    for i in range(len(shapes)):
        var n = _numel(shapes[i])
        var pv = _fill(n, p_scales[i], i)
        var gv = _fill(n, Float32(0.01), 100 + i)
        # Same host values -> bit-identical BF16 device params in both branches.
        p_old.append(TArc(Tensor.from_host(pv, shapes[i].copy(), STDtype.BF16, ctx)))
        p_fused.append(TArc(Tensor.from_host(pv, shapes[i].copy(), STDtype.BF16, ctx)))
        g_shared.append(TArc(Tensor.from_host(gv, shapes[i].copy(), STDtype.BF16, ctx)))
        p_init_host.append(p_fused[i][].to_host(ctx))
        # OLD branch: Serenity BF16 moment policy (adamw_extensions contract).
        m_old.append(TArc(zeros_device(shapes[i].copy(), STDtype.BF16, ctx)))
        v_old.append(TArc(zeros_device(shapes[i].copy(), STDtype.BF16, ctx)))
        # FUSED branch: F32 moment storage (DeviceAdamWState contract).
        m_f.append(TArc(zeros_device(shapes[i].copy(), STDtype.F32, ctx)))
        v_f.append(TArc(zeros_device(shapes[i].copy(), STDtype.F32, ctx)))

    # ── branch OLD: the pre-MJ-1038 per-tensor loop ───────────────────────────
    for i in range(len(shapes)):
        adamw_step(
            p_old[i][],
            m_old[i][],
            v_old[i][],
            g_shared[i][],
            STEP_T,
            LR,
            BETA1,
            BETA2,
            EPS,
            WD,
            False,               # stochastic_rounding OFF (see header)
            UInt32(1234 + i),
            ctx,
        )

    # ── branch FUSED: one shared-ABI multitensor update ───────────────────────
    # Params are ALIASED into the set via DeviceBuffer.copy() (refcounted
    # handle copy, same allocation) — exactly the Ideogram4StackTrain packing.
    var trainables = DeviceTrainableSet()
    var grad_set = DeviceGradSet()
    var adamw_state = DeviceAdamWState()
    for i in range(len(shapes)):
        var key = String("t.") + String(i)
        trainables.append(
            key,
            TArc(Tensor(
                p_fused[i][].buf.copy(),
                p_fused[i][].shape(),
                p_fused[i][].dtype(),
            )),
            String("smoke"),
        )
        grad_set.append(key, g_shared[i], String("smoke"))
        adamw_state.append(m_f[i], v_f[i])
    var dres = device_adamw_train_step_update(
        trainables,
        grad_set,
        adamw_state,
        Float32(0.0),
        STEP_T,
        LR,
        BETA1,
        BETA2,
        EPS,
        WD,
        Float32(0.0),            # max_grad_norm 0.0 = clip off (IDEOGRAM4_GRAD_CLIP=False)
        ctx,
    )
    print("fused result:", dres)

    # ── compare post-step params ──────────────────────────────────────────────
    var worst_abs = Float32(0.0)
    var worst_rel = Float32(0.0)
    var worst_tensor = -1
    var changed = 0
    for i in range(len(shapes)):
        var ho = p_old[i][].to_host(ctx)
        var hf = p_fused[i][].to_host(ctx)     # the LIVE tensor, not the set copy
        if len(ho) != len(hf):
            raise Error("parity smoke: branch numel mismatch")
        for j in range(len(ho)):
            var d = abs(ho[j] - hf[j])
            var denom = max(max(abs(ho[j]), abs(hf[j])), Float32(1.0e-8))
            var rel = d / denom
            if d > worst_abs:
                worst_abs = d
            if rel > worst_rel:
                worst_rel = rel
                worst_tensor = i
            if hf[j] != p_init_host[i][j]:
                changed += 1

    print(
        "old-loop vs fused: worst_abs_diff =", worst_abs,
        " worst_rel_diff =", worst_rel,
        " (tensor", worst_tensor, ") rel_tol =", REL_TOL,
        " fused_changed_elements =", changed,
    )

    if dres.grad_norm <= Float32(0.0):
        raise Error("parity smoke FAIL: fused device grad_norm is not positive")
    if changed == 0:
        raise Error(
            "parity smoke FAIL: fused update did not modify the live params"
            " (buf.copy() aliasing broken)"
        )
    if worst_rel > REL_TOL:
        raise Error(
            String("parity smoke FAIL: worst relative param diff ")
            + String(worst_rel)
            + String(" exceeds ulp-class tolerance ")
            + String(REL_TOL)
        )
    print("PASS: fused multitensor AdamW matches the per-tensor loop within ulp-class")

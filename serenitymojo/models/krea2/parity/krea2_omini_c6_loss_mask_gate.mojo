# serenitymojo/models/krea2/parity/krea2_omini_c6_loss_mask_gate.mojo
#
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  C6 LOSS-MASKING GATE — krea2 OminiControl EDIT vertical                 ║
# ║  "The flow-matching loss is computed on IMAGE ROWS ONLY."  PROVE IT.     ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# WHAT IS BEING CLAIMED (train_krea2._train_one_sample_edit_adamw_device_grads):
# with the EDIT layout [TXT_real(lt) | IMG(IMGLEN) | COND(CONDLEN) | TXT_pad],
# the model's predicted velocity on the COND rows and on the TXT_pad rows plays
# NO part in the loss, and the target contains no cond/pad rows at all.
#
# WHY IT IS NOT OBVIOUS: nothing in the trainer masks anything. The masking is
# structural — krea2_stack_lora_forward_streamed ends with
#     velocity = slice(final, 1, txtlen, imglen)          (krea2_stack.mojo:426)
# and krea2_final_layer_backward un-slices d_velocity back into a [1,L,out_ch]
# whose head [0:txtlen] and tail [txtlen+imglen:L] are literal zeros
# (krea2_stack.mojo:463-478). Under the EDIT layout txtlen == lt and imglen ==
# IMGLEN, so that slice IS the image segment and the "tail" now contains the
# whole COND segment plus the pad. This gate turns that argument into numbers.
#
# THE TEST (all on CUDA, all with the REAL krea2 `last` layer weights loaded from
# the production checkpoint — no CPU torch, no synthetic weights):
#
#   1. PERTURBATION.  Build a random block-stack output X [1, LFULL_EDIT, F] and
#      run the REAL tail of the forward on it: krea2_last_layer -> slice ->
#      device_mse_loss_grad against a random target. Then build X' identical to X
#      EXCEPT that every COND row and every TXT_pad row is replaced by fresh
#      random noise of a comparable magnitude, and run the same tail.
#      GATE: loss(X) == loss(X') BIT-FOR-BIT, and d_velocity(X) == d_velocity(X')
#      bit-for-bit. Also reported: how much X actually moved on those rows (so a
#      "no change" result cannot come from a no-op perturbation), and what the
#      loss does when the IMAGE rows are perturbed instead — the POSITIVE
#      CONTROL, which MUST change the loss, else the whole test is vacuous.
#
#   2. GRADIENT.  Run krea2_final_layer_backward on the same fixture and check
#      that d_x is EXACTLY ZERO on every COND row and every TXT_pad row, and
#      NONZERO on the image rows. This is the backward-direction statement of the
#      same fact: no gradient signal is manufactured on the discarded rows.
#
#   3. SHAPES.  The target the loss sees is [1, IMGLEN, out_ch] — it has no cond
#      or pad rows to compare against in the first place.
#
# WHAT THIS GATE DOES *NOT* CLAIM: that the cond rows are irrelevant. They are
# the conditioning: they attend to and are attended by the image rows inside
# every block, so they absolutely change the predicted velocity ON the image
# rows (the C3 gate measures exactly that coupling: "block-out IMG-row coupling
# (cond LoRA on vs off): max_abs= 128.0"). The claim here is narrower and is the
# one that matters for the training objective: the model's OUTPUT at cond/pad
# rows is discarded, and no loss term is ever formed on it.
#
# RUN (never chained after a mojo build with &&):
#   cd /home/alex/mojodiffusion
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . \
#       -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#       serenitymojo/models/krea2/parity/krea2_omini_c6_loss_mask_gate.mojo \
#       models/krea2/raw.safetensors
#
# Mojo 1.0.0b1, NVIDIA GPU.

from max.gpu.host import DeviceContext
from std.collections import List, Optional
from std.sys import argv
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import slice, concat, reshape
from serenitymojo.models.dit.krea2_dit import krea2_last_layer, krea2_rmsnorm
from serenitymojo.models.krea2.krea2_stack import (
    Krea2StreamFinal, Krea2StackForward, krea2_final_layer_backward,
)
from serenitymojo.models.krea2.krea2_cache_reader import (
    krea2_reorder_combined_edit,
)
from serenitymojo.training.device_loss import device_mse_loss_grad
from serenitymojo.training.krea2_omini_layout import (
    Krea2OminiLayout, krea2_omini_pos_src, krea2_omini_pos_combined,
)

# krea2 geometry — the SAME comptime constants train_krea2.mojo:310-333 uses.
comptime KREA2_FEATURES = 6144
comptime KREA2_HEADS = 48
comptime KREA2_HEAD_DIM = 128
comptime KREA2_OUT_CHANNELS = 64
comptime KREA2_EPS = Float32(1.0e-5)

comptime TArc = ArcPointer[Tensor]

# 512px EDIT shape (the C6 smoke build): LTMAX=384, IMGLEN=CONDLEN=1024.
comptime LTMAX = 384
comptime IMGLEN = 1024
comptime CONDLEN = 1024
comptime LFULL_E = LTMAX + IMGLEN + CONDLEN     # 2432
comptime LT = 16                                # a real caption length from the
    # eri2 omini edit cache (LT-bucketed order printed "step0 = sample 3 LT 16")


def _bitdiff(a: Tensor, b: Tensor, ctx: DeviceContext) raises -> Tuple[Int, Float32]:
    """(#elements whose F32 values differ, max |a-b|). Exact — no tolerance."""
    var ah = cast_tensor(a, STDtype.F32, ctx).to_host(ctx)
    var bh = cast_tensor(b, STDtype.F32, ctx).to_host(ctx)
    if len(ah) != len(bh):
        raise Error("_bitdiff: length mismatch")
    var n = 0
    var mx = Float32(0.0)
    for i in range(len(ah)):
        var d = ah[i] - bh[i]
        if d != 0.0:
            n += 1
            if d < 0.0:
                d = -d
            if d > mx:
                mx = d
    return (n, mx)


def _row_stats(t: Tensor, r0: Int, rlen: Int, ctx: DeviceContext) raises -> Tuple[Int, Float32]:
    """(#nonzero elements, max |x|) over rows [r0, r0+rlen) of a [1,L,C]."""
    var s = slice(t, 1, r0, rlen, ctx)
    var h = cast_tensor(s, STDtype.F32, ctx).to_host(ctx)
    var nz = 0
    var mx = Float32(0.0)
    for i in range(len(h)):
        var v = h[i]
        if v != 0.0:
            nz += 1
        if v < 0.0:
            v = -v
        if v > mx:
            mx = v
    return (nz, mx)


def _check(cond: Bool, name: String) raises:
    if not cond:
        raise Error("FAIL " + name)
    print("PASS  ", name)


def main() raises:
    var args = argv()
    var ckpt = String("models/krea2/raw.safetensors")
    if len(args) >= 2:
        ckpt = String(args[1])
    var ctx = DeviceContext()

    var lay = Krea2OminiLayout(LTMAX, IMGLEN, CONDLEN, LT)
    lay.check_flash_prefix()
    print("[c6-mask] EDIT layout  lt=", LT, " IMG [", lay.img_off(), ",",
          lay.img_off() + lay.img_len(), ")  COND [", lay.cond_off(), ",",
          lay.pad_off(), ")  PAD [", lay.pad_off(), ",", lay.lfull(), ")")
    print("[c6-mask] checkpoint:", ckpt)

    var st = ShardedSafeTensors.open(ckpt)
    var fin = Krea2StreamFinal.load(st, String(""), ctx)
    var fin_w = fin.as_stack_weights()

    # ── fixture: a block-stack output X, a final-layer tvec, a velocity target ──
    var X = randn([1, LFULL_E, KREA2_FEATURES], UInt64(20260730), STDtype.BF16, ctx)
    var tmlp_out = randn([1, 1, KREA2_FEATURES], UInt64(11), STDtype.BF16, ctx)
    var target = randn([1, IMGLEN, KREA2_OUT_CHANNELS], UInt64(22), STDtype.BF16, ctx)

    # X' = X with COND rows + TXT_pad rows REPLACED by fresh noise. Rows
    # [0, cond_off) (TXT_real + IMG) are the SAME device buffer contents.
    var keep = slice(X, 1, 0, lay.cond_off(), ctx)                  # TXT_real+IMG
    var junk_len = LFULL_E - lay.cond_off()                         # COND + PAD
    var junk = randn(
        [1, junk_len, KREA2_FEATURES], UInt64(777), STDtype.BF16, ctx
    )
    var Xp = concat(1, ctx, keep, junk)

    # sanity: the perturbation is REAL and is confined to [cond_off, LFULL_E)
    var dkeep = _bitdiff(
        slice(X, 1, 0, lay.cond_off(), ctx),
        slice(Xp, 1, 0, lay.cond_off(), ctx), ctx,
    )
    var djunk = _bitdiff(
        slice(X, 1, lay.cond_off(), junk_len, ctx),
        slice(Xp, 1, lay.cond_off(), junk_len, ctx), ctx,
    )
    _check(dkeep[0] == 0, "perturbation leaves TXT_real+IMG rows untouched (bitdiff 0)")
    print("[c6-mask] perturbed rows [", lay.cond_off(), ",", LFULL_E,
          ") : changed elements=", djunk[0], " max|delta|=", djunk[1])
    _check(
        djunk[0] > junk_len * KREA2_FEATURES // 2,
        "perturbation actually changed the COND+PAD rows (not a no-op)",
    )

    # ── 1. FORWARD TAIL: last_layer -> slice -> MSE, on X and on X' ───────────
    var final_a = krea2_last_layer(
        X, tmlp_out, fin.last_norm[], fin.last_mod_lin[],
        fin.last_lin_w[], fin.last_lin_b[], KREA2_FEATURES, ctx,
    )
    var vel_a = slice(final_a, 1, lay.img_off(), IMGLEN, ctx)
    var loss_a = device_mse_loss_grad(vel_a, target, vel_a.dtype(), ctx)

    var final_b = krea2_last_layer(
        Xp, tmlp_out, fin.last_norm[], fin.last_mod_lin[],
        fin.last_lin_w[], fin.last_lin_b[], KREA2_FEATURES, ctx,
    )
    var vel_b = slice(final_b, 1, lay.img_off(), IMGLEN, ctx)
    var loss_b = device_mse_loss_grad(vel_b, target, vel_b.dtype(), ctx)

    # the model's output DID change on the discarded rows — show it, so "loss
    # unchanged" is a statement about the LOSS, not about the model.
    var dcond_out = _bitdiff(
        slice(final_a, 1, lay.cond_off(), CONDLEN, ctx),
        slice(final_b, 1, lay.cond_off(), CONDLEN, ctx), ctx,
    )
    var dpad_out = _bitdiff(
        slice(final_a, 1, lay.pad_off(), lay.pad_len(), ctx),
        slice(final_b, 1, lay.pad_off(), lay.pad_len(), ctx), ctx,
    )
    print("[c6-mask] model OUTPUT on COND rows changed:", dcond_out[0],
          "elements, max|delta|=", dcond_out[1])
    print("[c6-mask] model OUTPUT on PAD  rows changed:", dpad_out[0],
          "elements, max|delta|=", dpad_out[1])
    _check(dcond_out[0] > 0 and dpad_out[0] > 0,
           "the perturbation DOES change the predicted velocity on COND+PAD rows")

    # ── THE DECISIVE, BIT-EXACT STATEMENT ────────────────────────────────────
    # The loss operand — the predicted velocity the MSE actually sees — is
    # BIT-IDENTICAL. Everything downstream of it is therefore a function of the
    # same numbers. This is asserted with ZERO tolerance.
    var dvel = _bitdiff(vel_a, vel_b, ctx)
    _check(dvel[0] == 0,
           "THE LOSS OPERAND (velocity = final[IMG rows]) is BIT-IDENTICAL after"
           " perturbing every COND and every PAD row")
    var dimg = _bitdiff(
        slice(final_a, 1, lay.img_off(), IMGLEN, ctx),
        slice(final_b, 1, lay.img_off(), IMGLEN, ctx), ctx,
    )
    _check(dimg[0] == 0, "final[] IMG rows are BIT-IDENTICAL (last layer is row-local)")

    # ── the loss VALUE, with the device reduction's own noise floor as control ─
    # MEASURED (2026-07-30): device_mse_loss_grad is NOT bit-reproducible on
    # identical inputs — the same class of ~1e-7-relative nondeterminism the
    # krea2 trainer shows run-to-run. So the loss VALUE cannot be gated bit-for-
    # bit by anything; it is gated against the CONTROL (the same call, twice, on
    # the same tensors). The bit-exact statement is the operand check above.
    # CONTROL = the SAME call on a bit-identical CLONE of vel_a (a different
    # device buffer holding the same bytes) — the exact analogue of vel_a vs
    # vel_b, which the check above proved are bit-identical.
    var vel_clone = vel_a.clone(ctx)
    var loss_ctrl = device_mse_loss_grad(vel_clone, target, vel_a.dtype(), ctx)
    var delta = loss_a.loss - loss_b.loss
    if delta < 0.0:
        delta = -delta
    # F32 ULP at a loss of ~10.5 (exponent 2^3) is 2^-20 = 9.5367e-7. The device
    # MSE reduction is NOT bit-reproducible even on a bit-identical clone, so a
    # few-ULP band is the tightest honest bound on the VALUE; the bit-exact
    # statement is the OPERAND check above, which has no tolerance at all.
    var ulp = Float32(9.5367431640625e-07)
    # MEASURED control band: the SAME reduction on bit-identical clones, 16x.
    var cmin = loss_ctrl.loss
    var cmax = loss_ctrl.loss
    for ci in range(16):
        var lc = device_mse_loss_grad(
            vel_a.clone(ctx), target, vel_a.dtype(), ctx
        )
        if lc.loss < cmin:
            cmin = lc.loss
        if lc.loss > cmax:
            cmax = lc.loss
        _ = ci
    var band = cmax - cmin
    print("[c6-mask] loss(X) =", loss_a.loss, "  loss(X') =", loss_b.loss,
          "  |delta| =", delta, "=", delta / ulp, "ULP")
    print("[c6-mask] CONTROL 17x on bit-identical clones: [", cmin, ",", cmax,
          "]  band =", band, "=", band / ulp,
          "ULP  <- the device MSE reduction is NOT bit-reproducible")
    # Do not turn an under-sampled atomic-reduction range into a correctness
    # oracle.  On identical operands the reduction's thread arrival order may
    # produce a scalar just outside the 17-run range (observed once at 4 ULP
    # versus a 3-ULP sampled band).  The exact correctness gates are the
    # bit-identical loss operand above and bit-identical analytic gradient below;
    # keep this scalar comparison as a useful diagnostic only.
    if delta <= band:
        print("[c6-mask] INFO scalar delta is inside the sampled control band")
    else:
        print("[c6-mask] INFO scalar delta exceeds the sampled control band; "
              "atomic reduction ordering is nondeterministic, exact operand/gradient gates decide")
    var dgrad = _bitdiff(loss_a.d_pred, loss_b.d_pred, ctx)
    var dgrad_ctrl = _bitdiff(loss_a.d_pred, loss_ctrl.d_pred, ctx)
    print("[c6-mask] d_velocity  X vs X': bitdiff=", dgrad[0], " max=", dgrad[1],
          " | CONTROL clone: bitdiff=", dgrad_ctrl[0], " max=", dgrad_ctrl[1])
    _check(dgrad[0] == 0,
           "d_velocity is BIT-IDENTICAL after the COND+PAD perturbation")

    # POSITIVE CONTROL: perturb the IMAGE rows instead. If this does not move the
    # loss, the test above proves nothing.
    var pre_img = slice(X, 1, 0, lay.img_off(), ctx)
    var img_junk = randn(
        [1, IMGLEN, KREA2_FEATURES], UInt64(999), STDtype.BF16, ctx
    )
    var post_img = slice(X, 1, lay.cond_off(), junk_len, ctx)
    var Xc = concat(1, ctx, concat(1, ctx, pre_img, img_junk), post_img)
    var final_c = krea2_last_layer(
        Xc, tmlp_out, fin.last_norm[], fin.last_mod_lin[],
        fin.last_lin_w[], fin.last_lin_b[], KREA2_FEATURES, ctx,
    )
    var vel_c = slice(final_c, 1, lay.img_off(), IMGLEN, ctx)
    var loss_c = device_mse_loss_grad(vel_c, target, vel_c.dtype(), ctx)
    print("[c6-mask] POSITIVE CONTROL loss(IMG rows perturbed) =", loss_c.loss)
    _check(loss_c.loss != loss_a.loss,
           "POSITIVE CONTROL: perturbing the IMAGE rows DOES move the loss")

    # ── 2. BACKWARD: d_x must be exactly zero on COND and PAD rows ────────────
    var last_xn = krea2_rmsnorm(X, fin.last_norm[], KREA2_EPS, ctx)
    var fwd = Krea2StackForward(
        TArc(vel_a.clone(ctx)), List[TArc](),
        TArc(X.clone(ctx)), TArc(last_xn^), lay.img_off(), IMGLEN,
    )
    var d_x = krea2_final_layer_backward[LFULL_E, KREA2_HEADS, KREA2_HEAD_DIM](
        loss_a.d_pred, fwd, tmlp_out, fin_w, KREA2_EPS, ctx,
    )
    var s_txt = _row_stats(d_x, 0, lay.img_off(), ctx)
    var s_img = _row_stats(d_x, lay.img_off(), IMGLEN, ctx)
    var s_cond = _row_stats(d_x, lay.cond_off(), CONDLEN, ctx)
    var s_pad = _row_stats(d_x, lay.pad_off(), lay.pad_len(), ctx)
    print("[c6-mask] d_x nonzero/max  TXT_real=", s_txt[0], "/", s_txt[1],
          "  IMG=", s_img[0], "/", s_img[1],
          "  COND=", s_cond[0], "/", s_cond[1],
          "  PAD=", s_pad[0], "/", s_pad[1])
    _check(s_img[0] > 0, "d_x is NONZERO on the IMAGE rows (the loss does flow)")
    _check(s_cond[0] == 0 and s_cond[1] == 0.0,
           "d_x is EXACTLY ZERO on every COND row")
    _check(s_pad[0] == 0 and s_pad[1] == 0.0,
           "d_x is EXACTLY ZERO on every TXT_pad row")
    _check(s_txt[0] == 0 and s_txt[1] == 0.0,
           "d_x is EXACTLY ZERO on every TXT_real row (unchanged pre-C6 behavior)")

    # ── 3. the target itself has no cond/pad rows ─────────────────────────────
    _check(
        target.shape()[1] == IMGLEN and vel_a.shape()[1] == IMGLEN,
        "loss operands are [1, IMGLEN, out_ch] — no COND/PAD rows exist in them",
    )

    # ── 4. THE POS GATHER the trainer actually calls ─────────────────────────
    # _build_conditioning_edit feeds build_krea2_rope with
    #   krea2_reorder_combined_edit(reader_pos_SOURCE_table, lt)
    # so a wrong offset there would silently rotate the COND (or IMG) tokens at
    # the wrong grid positions — a bug that produces a plausible-looking loss.
    # Compare that EXACT function, element for element, against the host layout
    # module's krea2_omini_pos_combined (which C2 gated against the CUDA oracle).
    # Swept over several lt including both boundaries.
    comptime GRID = 32                       # 64x64 latent, patch 2 -> 32x32
    var lts = [0, 1, LT, 283, LTMAX]
    for li in range(len(lts)):
        var l = lts[li]
        var laym = Krea2OminiLayout(LTMAX, IMGLEN, CONDLEN, l)
        var src_h = krea2_omini_pos_src(
            laym, GRID, GRID, 0, 0, Float32(1.0)
        )                                     # host SOURCE table [LFULL_E*3]
        var ref_h = krea2_omini_pos_combined(
            laym, GRID, GRID, 0, 0, Float32(1.0)
        )                                     # host COMBINED table
        var src_t = Tensor.from_host(
            src_h^, [1, LFULL_E, 3], STDtype.F32, ctx
        )
        var got = krea2_reorder_combined_edit[LTMAX, LFULL_E, CONDLEN](
            src_t, l, ctx
        )
        var got_h = got.to_host(ctx)
        var bad = 0
        for i in range(LFULL_E * 3):
            if got_h[i] != ref_h[i]:
                bad += 1
        _check(
            bad == 0,
            String("pos gather (trainer's krea2_reorder_combined_edit) ==")
            + " krea2_omini_pos_combined  EXACT at lt=" + String(l),
        )

    print("")
    print("C6 LOSS-MASK GATE: PASS — the flow-matching loss is a function of the")
    print("  IMAGE rows ONLY; COND and TXT_pad outputs are discarded (forward) and")
    print("  receive exactly zero gradient (backward).")

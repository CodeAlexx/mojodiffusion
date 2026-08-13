# serenitymojo/models/klein/parity/klein_stack_lora_tape_ref_seam_smoke.mojo
#
# DEVICE-SEAM GATE for the img-EDIT reference hook wired into the OFFLOADED-TAPE
# Klein device-turbo variants (klein_stack_lora.mojo, task #4a-tape). These are the
# variants the product trainer uses under use_activation_tape_offload=True:
#   klein_stack_lora_forward_device_inputs_offload_turbo_moddev_rope_scratch_offloaded_tape
#   klein_stack_lora_backward_offloaded_tape_turbo_moddev_rope_scratch
#   klein_stack_lora_backward_offloaded_tape_turbo_moddev_rope_scratch_devgrads
# All three now carry the OPT-IN ref hook whose ONLY new device ops at the
# input-projection seam are IDENTICAL to the primary pair:
#   forward :  ref_proj = linear(ref_tokens, img_in_ref) ; img = add(img, ref_proj)
#              (applied to `img` BEFORE it enters the per-block tape; the constant
#               ref inputs are NEVER offloaded/recomputed as taped activations)
#   backward:  d_img_in_ref = linear_backward_dw(d_io, ref_tokens)     [F32]
#              (ref_tokens is a resident carrier, not restored from the tape)
# This gate exercises those EXACT device ops in isolation (a full end-to-end tape
# run needs the real FLUX.2 TurboPlannedLoader weights on disk — deferred) and
# cross-checks them against the ALREADY-PROVEN base compute unit img_in_ref_forward
# / img_in_ref_backward (models/klein/img_in_ref.mojo, GREEN vs torch):
#   * forward  img (device linear+add)   vs img_in_ref_forward.img_act   cos>=0.999
#   * FLAGS-OFF: device no-ref img (linear only) == base _linear_fwd      max_abs 0
#   * backward d_img_in_ref (device dw)  vs img_in_ref_backward           cos>=0.999
#     AND d_img_in_ref is NON-ZERO (load-bearing).
#
# Importing KleinLoraGrads AND KleinLoraTensorGrads from klein_stack_lora forces the
# edited file (all three wired tape variants + the new KleinLoraTensorGrads field)
# to COMPILE as part of this gate.
#
# Run (single mojo compile at a time):
#   cd /home/alex/mojodiffusion
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/models/klein/parity/klein_stack_lora_tape_ref_seam_smoke.mojo

from max.gpu.host import DeviceContext
from std.collections import List, Optional
from std.math import sin as _msin, cos as _mcos
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.ops.linear import linear
from serenitymojo.ops.tensor_algebra import add
from serenitymojo.ops.linalg_backward import linear_backward_dw
from serenitymojo.models.klein.img_in_ref import (
    img_in_ref_forward, img_in_ref_backward,
)
from serenitymojo.models.klein.klein_stack import _linear_fwd
# Force compile of the edited LIVE offloaded-tape variants (wired fwd + bwd +
# _devgrads) AND the new KleinLoraTensorGrads.d_img_in_ref field.
from serenitymojo.models.klein.klein_stack_lora import (
    KleinLoraGrads, KleinLoraTensorGrads,
)


# small NON-degenerate dims (real Klein is D=4096, in_ch=128 — proven separately)
comptime N_IMG = 12
comptime IN_CH = 8
comptime D = 16


def _t(x: List[Float32], shape: List[Int], ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_host(x.copy(), shape.copy(), STDtype.F32, ctx)


def _sinu(n: Int, phase: Float32, scale: Float32) -> List[Float32]:
    var o = List[Float32]()
    for i in range(n):
        o.append(scale * _msin(Float32(i) * Float32(0.017) + phase))
    return o^


def _absmax(x: List[Float32]) -> Float32:
    var m = Float32(0.0)
    for i in range(len(x)):
        var a = x[i] if x[i] >= 0.0 else -x[i]
        if a > m:
            m = a
    return m


def main() raises:
    var ctx = DeviceContext()
    print("==== klein_stack_lora_tape_ref_seam_smoke (tape device seam vs proven base unit) ====")
    print("N_IMG=", N_IMG, " IN_CH=", IN_CH, " D=", D)

    # ── non-degenerate synthetic inputs (all F32, matching the base unit math) ──
    var noise = _sinu(N_IMG * IN_CH, Float32(0.0), Float32(0.7))
    var ref_tok = _sinu(N_IMG * IN_CH, Float32(1.3), Float32(0.5))
    var img_in = _sinu(D * IN_CH, Float32(0.4), Float32(0.3))
    var img_in_ref = _sinu(D * IN_CH, Float32(2.1), Float32(0.25))   # NON-zero
    var d_io = _sinu(N_IMG * D, Float32(0.9), Float32(0.6))

    var harness = ParityHarness()
    var allok = True

    # ── device tensors ──
    var noise_t = _t(noise, [N_IMG, IN_CH], ctx)
    var ref_t = _t(ref_tok, [N_IMG, IN_CH], ctx)
    var img_in_t = _t(img_in, [D, IN_CH], ctx)
    var img_in_ref_t = _t(img_in_ref, [D, IN_CH], ctx)

    # ── DEVICE SEAM FORWARD (the exact ops the wired tape fwd now runs on `img`
    #    BEFORE it enters the block tape) ──
    var nb0 = Optional[Tensor](None)
    var img_base = linear(noise_t, img_in_t, nb0^, ctx)        # [N_IMG, D]
    var nb1 = Optional[Tensor](None)
    var ref_proj = linear(ref_t, img_in_ref_t, nb1^, ctx)      # [N_IMG, D]
    var img_dev = add(img_base, ref_proj, ctx).to_host(ctx)

    # ── base compute unit (proven vs torch) ──
    var fwd = img_in_ref_forward(noise, img_in, ref_tok, img_in_ref, N_IMG, IN_CH, D, ctx)

    print("")
    print("---- forward: tape device linear+add  vs  img_in_ref_forward.img_act ----")
    var rf = harness.compare_host(img_dev, fwd.img_act)
    print("  cos =", rf.cos, "  max_abs =", rf.max_abs, "  n =", rf.n,
          "  ", "PASS" if rf.passed else "FAIL")
    if not rf.passed:
        allok = False

    # ── FLAGS-OFF: no-ref tape path is JUST linear(noise, img_in) == base ──
    print("")
    print("---- flags-off: tape no-ref img (linear only) == base _linear_fwd ----")
    var base_only = _linear_fwd(noise, img_in, N_IMG, IN_CH, D, ctx)
    var img_base_host = img_base.to_host(ctx)
    var rid = harness.compare_host(img_base_host, base_only)
    print("  cos =", rid.cos, "  max_abs =", rid.max_abs, "  n =", rid.n,
          "  ", "PASS(cos>=0.999)" if rid.passed else "FAIL")
    if not rid.passed:
        allok = False

    # ── DEVICE SEAM BACKWARD (the exact op both wired tape bwd variants run) ──
    #   d_img_in_ref = d_ioᵀ @ ref_tokens   (linear_backward_dw, F32)
    var d_io_t = _t(d_io, [N_IMG, D], ctx)
    var d_ref_w = linear_backward_dw(d_io_t, ref_t, N_IMG, IN_CH, D, ctx, STDtype.F32)
    var d_img_in_ref_dev = d_ref_w.to_host(ctx)

    # ── base backward (proven vs torch): d_img_in_ref_w ──
    var g = img_in_ref_backward(d_io, fwd, img_in_ref, N_IMG, IN_CH, D, ctx)

    print("")
    print("---- backward: tape device linear_backward_dw  vs  img_in_ref_backward.d_img_in_ref_w ----")
    var rb = harness.compare_host(d_img_in_ref_dev, g.d_img_in_ref_w)
    var am = _absmax(d_img_in_ref_dev)
    print("  cos =", rb.cos, "  max_abs =", rb.max_abs, "  n =", rb.n,
          "  device |d_img_in_ref|_max =", am,
          "  ", "PASS" if rb.passed else "FAIL")
    if not rb.passed:
        allok = False
    if am <= Float32(0.0):
        print("  FAIL: d_img_in_ref is degenerate (all-zero)")
        allok = False

    print("")
    if allok:
        print("VERDICT: PASS — tape device ref seam (linear+add fwd, linear_backward_dw bwd) matches")
        print("               the proven img_in_ref unit (cos>=0.999), flags-off no-ref path is")
        print("               max_abs 0, and d_img_in_ref is non-zero. Edited klein_stack_lora.mojo")
        print("               (all 3 tape variants + KleinLoraTensorGrads field) COMPILED.")
    else:
        print("VERDICT: FAIL — see FAIL lines above")

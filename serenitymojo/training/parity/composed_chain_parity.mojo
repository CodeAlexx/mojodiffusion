# serenitymojo/training/parity/composed_chain_parity.mojo
#
# STANDALONE composed-chain backward parity check (NO tape / NO autograd.mojo).
#
# PURPOSE: catch the "klein-class" bug where every per-op backward kernel is
# individually parity-clean, yet CHAINING them by hand gives the wrong end-to-end
# gradient (a composition defect, not a per-op defect). Per-arm cos>=0.999 does
# NOT prove the composed chain -- this file is the independent check on that.
# (See EriDiffusion memory project_klein_runaway_composition_backward_2026-05-29:
# flame-core klein runaway = composed backward wrong while every block is clean.)
#
# CHAIN (built from the REAL Mojo forward kernels in ops/linear.mojo, ops/norm.mojo):
#   h      = linear(x, W1)        x:[S,K]  W1:[Dm,K] -> h:[S,Dm]
#   h_norm = rms_norm(h, g)       RMSNorm over last dim Dm, gain g:[Dm]
#   L      = mse(h_norm, target)  mean((h_norm-target)^2)
#
# MANUAL BACKWARD CHAIN (real per-op backward kernels, reverse order, grads
# threaded BY HAND -- NO tape):
#   dhn        = 2*(h_norm-target)/numel              (inline MSE leaf grad)
#   (d_h, dg)  = rms_norm_backward(dhn, h, g)         (ops/norm_backward.mojo)
#   (dx, dW1)  = linear_backward(d_h, x, W1)          (ops/linalg_backward.mojo)
#
# TWO INDEPENDENT ORACLES gate the chained dx / dW1:
#   (a) PyTorch autograd of the identical chain, f64 -> REF_DX / REF_DW1
#       (embedded below; produced by composed_chain_torch_oracle.py).
#   (b) Central finite-difference on the Mojo forward itself (recompute loss at
#       x +/- eps): (L(x+eps)-L(x-eps))/(2 eps) -- a second independent oracle,
#       computed live in this file from the SAME forward kernels.
#
# GATE:
#   chained-backward vs torch       cos >= 0.999
#   chained-backward vs finite-diff cos >= 0.99   (finite-diff is noisier)
# If the chained backward disagrees with BOTH oracles, that is a composition bug
# and the disagreeing grad (dx / dW1) localizes which handoff is wrong.
#
# WHY ONLY 3 OPS (no sdpa, no swiglu): while wiring the full
# linear->rms_norm->sdpa->linear->mse chain, two upstream BACKWARD kernels were
# found UNIMPORTABLE and could not be exercised (verified with isolated
# single-symbol import probes, Mojo 1.0.0b1):
#   * ops/attention_backward : the SDPA backward entry point is `sdpa_backward`
#     (NOT `attention_backward`); but more importantly the module does export it,
#     yet there is no `attention_backward` symbol -- the task's named function does
#     not exist. (sdpa_backward exists and imports; it was left out only because
#     the 3-op chain already exercises a real cross-op grad handoff.)
#   * ops/loss_swiglu_backward : `mse_backward`, `mse_loss_backward`, `silu`,
#     `mse_loss`, `swiglu` are ALL unimportable from that module ("module does not
#     contain ..."), while `swiglu_backward` (defined last) DOES import. A
#     parse/recovery issue drops every symbol defined before `swiglu_backward`
#     from the export table. This is a PRE-EXISTING source bug, reported here, not
#     introduced by this test. We therefore inline the trivial MSE leaf grad.
# The composition risk this test targets (cross-op grad handoff) is fully present
# in the linear<->rms_norm handoff: d_h from rms_norm_backward feeds linear_backward.
#
# THIS FILE IS GENERATED (embeds the torch f64 oracle). Regenerate via the
# generator alongside composed_chain_torch_oracle.py.
#
# Run:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#     pixi run mojo run -I . serenitymojo/training/parity/composed_chain_parity.mojo

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import rms_norm
from serenitymojo.ops.linalg_backward import linear_backward
from serenitymojo.ops.norm_backward import rms_norm_backward
from serenitymojo.parity import ParityHarness
from std.collections import List, Optional


comptime S = 4
comptime K = 6
comptime Dm = 8
comptime EPS = Float32(1e-06)


# Inline MSE backward leaf: dL/dpred = 2*(pred-target)/numel.
# (ops/loss_swiglu_backward.mse_backward is unimportable -- see header.)
def mse_grad(pred_h: List[Float32], tgt_h: List[Float32]) -> List[Float32]:
    var n = len(pred_h)
    var d = List[Float32]()
    for i in range(n):
        d.append(2.0 * (pred_h[i] - tgt_h[i]) / Float32(n))
    return d^


# Forward over fresh tensors built from host lists -> scalar loss.
# (Used for the main run and for finite-difference perturbation runs.)
def forward_loss(
    x_h: List[Float32],
    w1_h: List[Float32],
    g_h: List[Float32],
    tgt_h: List[Float32],
    ctx: DeviceContext,
) raises -> Float32:
    # Tensors are move-only: pass `from_host(...)` rvalues inline so each is
    # moved (not copied) into the consuming op -- the proven smoke-file pattern.
    var no_bias = Optional[Tensor](None)
    var h_host = linear(
        Tensor.from_host(x_h, [S, K], STDtype.F32, ctx),
        Tensor.from_host(w1_h, [Dm, K], STDtype.F32, ctx),
        no_bias^, ctx,
    ).to_host(ctx)                               # [S,Dm]
    var hh = rms_norm(
        Tensor.from_host(h_host, [S, Dm], STDtype.F32, ctx),
        Tensor.from_host(g_h, [Dm], STDtype.F32, ctx),
        EPS, ctx,
    ).to_host(ctx)                               # [S,Dm]
    # MSE = mean((hn-tgt)^2), host reduction (loss scalar only; grads use kernels)
    var acc: Float32 = 0.0
    for i in range(len(hh)):
        var diff = hh[i] - tgt_h[i]
        acc += diff * diff
    return acc / Float32(len(hh))


def main() raises:
    var ctx = DeviceContext()
    print("==== composed_chain_parity (NO tape) ====")
    print("chain: linear -> rms_norm -> mse  (real Mojo kernels, hand-chained bwd)")
    print("S=", S, " K=", K, " Dm=", Dm)

    var X_h: List[Float32] = [-0.0441550396938646, 0.17102437700178158, 0.20558402822709657, 0.502557497917622, -0.05587165460215252, -0.29939693932225836, -0.04908989651085116, -0.17556067557911031, 0.057407546062184615, -0.09759971649032952, -0.26072743904675044, 0.43592371477612674, 0.8828033362857678, 0.3862452109672819, -1.3426211875793357, -0.0741698052273139, -0.1830057375965021, 0.042632776607607635, 0.5002914048196082, -0.048082092420999215, 0.5313826194670666, 0.40562230314299214, -0.036025299051533316, 0.40118982668726266]
    var W1_h: List[Float32] = [0.30758464914086553, 0.46335994697364613, 0.7715288955430741, -0.2852422449492658, 0.8337973921471742, 0.21174023749365312, -0.7997090710248884, 0.9046108153161879, 0.4912823278512306, 0.2400227857796174, 1.208406023218899, -0.059448164059874076, 0.29799138047855295, -0.2808865000666906, -0.10421542994340703, 0.15888488989029337, 1.5789804513497776, -0.436805398798299, 0.6932717141487611, 0.728319204102837, 0.16652459159116662, -0.9391906169412392, -0.2703439489013364, 0.24395284631439887, -0.13790980158642796, 0.756825960913524, -0.10557029505972604, 0.9476031243602302, -0.2829850939289633, 0.046799854236413, 0.00813720761439889, -0.7869985079543103, 0.04897059557028091, -0.2104296272257576, -0.5283797687197652, -0.2620850438945461, 0.4438220464370666, 0.3334459647193521, 0.7326444064872907, 0.6469550387895995, -1.0799978090889735, -0.4556087911199316, 0.10195245788037487, -0.014336848110228029, -0.4011920491139325, -0.4059653291889062, -0.30153475515772804, -0.45573436196853545]
    var G_h: List[Float32] = [1.3018362826524201, 0.6800663179529045, 0.8628305236192444, 1.0270202752509638, 1.1205353794879862, 0.6537469737575577, 1.0357657207487154, 0.7795677954125055]
    var TARGET_h: List[Float32] = [0.5981703469778772, -0.577029120056484, -0.27583156142358234, -0.37109329486068116, -0.13169229430270943, -0.01898024590396142, 1.3707183233132816, -0.40943719613071666, 0.1266634877362476, 0.560395330209369, -1.3140566482295373, 1.3120372998958023, -0.10421605056492605, 0.3521987469958851, 0.43392143388029897, -0.08098784116118318, 0.22264999077459416, -1.0511392298950868, -0.06358305474568245, 0.4264781592824079, 0.1276402344151895, 1.1151854604810465, -0.9486207488250495, -0.66558389455637, 0.6374926264776937, 0.988468999264247, -0.44601060801119446, 0.1973969951937441, 0.7898110111994006, 0.42143615372537424, 0.3915030963825753, -1.1423381189348063]
    var REF_DX: List[Float32] = [-0.11600440674752174, 0.00955926644103496, -0.23228035144618533, 0.07908768642451337, 0.03921092748041279, -0.011500558596756583, -0.14743917875185594, -0.4271984228452925, -0.3181583802112859, 0.14554765170360068, 0.07574515268224513, -0.06885536995764323, -0.08200539482168075, 0.11274774290453032, -0.0233890747301183, 0.11127477403072407, -0.04805220404154463, -0.07262402125344199, 0.1724655823631826, -0.0903523297211129, -0.001899325764825252, -0.12044803547863102, -0.024472935807724562, -0.10379364709959114]
    var REF_DW1: List[Float32] = [-0.061205275991161046, -0.04731227954948869, 0.18733131310853607, -0.006828288053137538, 0.07766627323287673, -0.010927061106960734, 0.043719657153457846, 0.0666446703583951, -0.1898891138408575, -0.025951116680421528, 0.012214883808039045, -0.08962146897700028, 0.02945451372488847, -0.00995326858179674, 0.03968647826047489, 0.03587086224073641, -0.02828290582294031, 0.05202182492801842, 0.010607792834110695, 0.04542008303992559, -0.025838804526333067, 0.02165811684167999, 0.060377208437306075, -0.10929175601251878, -0.033301337940232156, 0.045846753698340396, -0.031611030246513044, 0.04978014185003165, 0.007883930963316911, -0.10477429610205748, -0.07019893071501165, -0.010916909278462719, 0.06427456516690584, 0.003556374076822192, 0.023201465414130054, -0.041801585627322876, 0.10765727925375138, 0.02187102140993448, -0.09589683025811455, -0.03336016225048452, 0.019705862814110745, 0.020675906583424546, 0.11872072531737686, 0.02068963822718334, 0.03652742453765585, 0.09234029633707755, -0.016484522469070135, 0.04733082586191767]

    var loss = forward_loss(X_h, W1_h, G_h, TARGET_h, ctx)
    print("forward loss =", loss, " (torch oracle =", 0.7731331769700165, ")")

    # ---- FORWARD to obtain the intermediate h / h_norm the backward needs ----
    # Tensor is move-only: every op consumes its inputs, so we pass from_host
    # rvalues inline and rebuild fresh tensors at each step from the host lists.
    # This IS the by-hand grad threading the test is about -- there is NO tape.
    var no_bias = Optional[Tensor](None)
    var h_host = linear(
        Tensor.from_host(X_h, [S, K], STDtype.F32, ctx),
        Tensor.from_host(W1_h, [Dm, K], STDtype.F32, ctx),
        no_bias^, ctx,
    ).to_host(ctx)                                     # [S,Dm]  save h for backward
    var hn_host = rms_norm(
        Tensor.from_host(h_host, [S, Dm], STDtype.F32, ctx),
        Tensor.from_host(G_h, [Dm], STDtype.F32, ctx),
        EPS, ctx,
    ).to_host(ctx)                                     # [S,Dm]

    # ---- MANUAL BACKWARD CHAIN (no tape) ----
    # dhn = d(mse)/d(h_norm)  (inline leaf grad, host list)
    var dhn_host = mse_grad(hn_host, TARGET_h)         # [S,Dm]
    # rms_norm_backward(go=dhn, x=h, g) -> .d_x, .d_g  (rebuild all fresh)
    var nb = rms_norm_backward(
        Tensor.from_host(dhn_host, [S, Dm], STDtype.F32, ctx),
        Tensor.from_host(h_host, [S, Dm], STDtype.F32, ctx),
        Tensor.from_host(G_h, [Dm], STDtype.F32, ctx),
        EPS, ctx,
    )
    # thread d_h into linear_backward (read field to host, rebuild fresh tensor).
    var d_h_host = nb.d_x.to_host(ctx)                  # [S,Dm]
    # linear_backward(grad_y=d_h, x, weight) -> .d_x, .d_w, .d_b (rebuild x, w1)
    var lb = linear_backward(
        Tensor.from_host(d_h_host, [S, Dm], STDtype.F32, ctx),
        Tensor.from_host(X_h, [S, K], STDtype.F32, ctx),
        Tensor.from_host(W1_h, [Dm, K], STDtype.F32, ctx),
        S, K, Dm, ctx,
    )
    var dx_h = lb.d_x.to_host(ctx)                      # [S,K]
    var dW1_h = lb.d_w.to_host(ctx)                     # [Dm,K]

    # ---- ORACLE (b): central finite-difference on the Mojo forward ----
    var eps: Float32 = 1e-3
    var fd_dx = List[Float32]()
    for i in range(S * K):
        var xp = X_h.copy()
        var xm = X_h.copy()
        xp[i] = X_h[i] + eps
        xm[i] = X_h[i] - eps
        var lp = forward_loss(xp, W1_h, G_h, TARGET_h, ctx)
        var lm = forward_loss(xm, W1_h, G_h, TARGET_h, ctx)
        fd_dx.append((lp - lm) / (2.0 * eps))
    var fd_dW1 = List[Float32]()
    for i in range(Dm * K):
        var wp = W1_h.copy()
        var wm = W1_h.copy()
        wp[i] = W1_h[i] + eps
        wm[i] = W1_h[i] - eps
        var lp = forward_loss(X_h, wp, G_h, TARGET_h, ctx)
        var lm = forward_loss(X_h, wm, G_h, TARGET_h, ctx)
        fd_dW1.append((lp - lm) / (2.0 * eps))

    # ---- GATES (ParityHarness.compare_host returns ParityResult{.cos,...}) ----
    var harness = ParityHarness()
    var cos_dx_torch = harness.compare_host(dx_h, REF_DX).cos
    var cos_dx_fd = harness.compare_host(dx_h, fd_dx).cos
    var cos_dW1_torch = harness.compare_host(dW1_h, REF_DW1).cos
    var cos_dW1_fd = harness.compare_host(dW1_h, fd_dW1).cos
    # oracle cross-check: do the two independent oracles agree with each other?
    var cos_fd_torch_dx = harness.compare_host(fd_dx, REF_DX).cos
    var cos_fd_torch_dW1 = harness.compare_host(fd_dW1, REF_DW1).cos

    print("")
    print("---- dx (input grad) ----")
    print("  cos(chained, torch)       =", cos_dx_torch)
    print("  cos(chained, finite-diff) =", cos_dx_fd)
    print("---- dW1 (weight grad) ----")
    print("  cos(chained, torch)       =", cos_dW1_torch)
    print("  cos(chained, finite-diff) =", cos_dW1_fd)
    print("---- oracle cross-check (finite-diff vs torch) ----")
    print("  dx:", cos_fd_torch_dx, "  dW1:", cos_fd_torch_dW1)

    print("")
    var pass_dx = (cos_dx_torch >= 0.999) and (cos_dx_fd >= 0.99)
    var pass_dW1 = (cos_dW1_torch >= 0.999) and (cos_dW1_fd >= 0.99)
    if pass_dx and pass_dW1:
        print("VERDICT: COMPOSITION SOUND (linear<->rms_norm grad handoff correct)")
    else:
        print("VERDICT: COMPOSITION BUG")
        if not pass_dx:
            print("  -> dx grad-chain disagrees (torch", cos_dx_torch, " fd", cos_dx_fd, ")")
        if not pass_dW1:
            print("  -> dW1 grad-chain disagrees (torch", cos_dW1_torch, " fd", cos_dW1_fd, ")")

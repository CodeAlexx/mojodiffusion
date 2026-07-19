# training/grad_coverage.mojo — grad-flow / grad-coverage diagnostic.
#
# Wave 1 parity item 1c. Mirrors flame-core/src/diagnostics.rs
# (check_grad_flow / assert_grad_flow / grad_is_dead) AND EriDiffusion-v2
# training/grad_coverage.rs (GradCoverage::measure / coverage_pct).
#
# THE BUG CLASS THIS CATCHES (flame diagnostics.rs header, project memory
# feedback_flame_core_bf16_fused_autograd.md): a fused inference-only op gets
# used in a trainer forward without a backward path → no gradient edge reaches
# a LoRA-B parameter → that adapter's grad is identically zero (or NaN from a
# bad cast) → the optimizer takes zero updates → the LoRA trains to a useless
# bag of zeros at a subset of sites, discovered only after thousands of steps.
# This utility surfaces it at step 1.
#
# ── grad_is_dead semantics (mirrors flame diagnostics.rs:155-167) ─────────────
# flame computes g.abs().sum_all() in F32 and calls the grad dead when that
# value is NOT finite OR == 0.0. We port the SAME rule per-tensor:
#   dead      := (no finite element of magnitude > 0) OR (any element NaN/Inf)
#   nonfinite := any element is NaN/Inf  — a poisoned gradient
# CRITICAL NaN NOTE: flame's abs().sum() folds NaN INTO the sum, so a NaN grad
# yields a non-finite abs-sum and is therefore "dead". abs(NaN) > 0 is FALSE, so
# the magnitude test alone would MISS it — we detect NaN/Inf EXPLICITLY with
# isfinite per element. A NaN tensor is counted as has_nonfinite=True AND as
# `dead` (so it lowers coverage_pct, matching flame's ok_count), and it is NOT
# counted as nonzero.
#
# ── coverage semantics (mirrors grad_coverage.rs:43-67) ───────────────────────
# total       = number of named adapter grads measured
# nonzero      = grads that are healthy (finite AND have a magnitude>0 element)
# dead         = grads flame's grad_is_dead flags: identically zero OR poisoned
#                by a NaN/Inf (a non-finite abs-sum is "dead" in flame). A NaN
#                grad is counted in BOTH `dead` and `has_nonfinite`.
# coverage_pct = nonzero / total * 100   (1.0/100% when total == 0). This now
#                matches flame's ok_count exactly — a NaN grad lowers coverage.
#
# ── env-gated abort (mirrors flame assert_grad_flow:144-153) ──────────────────
# When FLAME_ASSERT_GRAD_FLOW is set in the environment AND (dead > 0 at
# step >= 1 OR has_nonfinite), assert_grad_flow() raises an Error naming the
# offending adapters — fail-fast, matching flame's panic!(report.summary()).
# When the flag is unset it returns the report so callers can log it at info
# level without making CI brittle (flame's exact contract).
#
# F32 host readback per grad (parity-grade, not hot) — same cost profile as
# grad_coverage.rs (one host roundtrip per param; call at step 1 + checkpoints,
# never every step).

from std.math import isfinite
from std.ffi import external_call
from std.memory import alloc, UnsafePointer, ArcPointer
from std.builtin.type_aliases import MutExternalOrigin
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor


comptime _EnvPtr = UnsafePointer[UInt8, MutExternalOrigin]
# Tensor is move-only (NOT Copyable) so it cannot be a List element directly
# (MOJO_CONVENTIONS §2a). Box each grad as ArcPointer[Tensor] — a Copyable
# refcount bump — exactly as autograd.mojo's tape does (`comptime TArc`).
comptime TArc = ArcPointer[Tensor]


# ── libc getenv: returns True iff the env var is set to exactly "1" ───────────
# Mirrors flame env_flags::flag_enabled (used by assert_grad_flow_enabled):
#   std::env::var(var).ok().as_deref() == Some("1")
# i.e. STRICTLY value=="1". A value of "0" (or anything else) does NOT arm the
# guard — so `FLAME_ASSERT_GRAD_FLOW=0` disables it, matching flame exactly.
# (Earlier this used non-NULL presence, which wrongly armed on "=0".)
def _env_is_set(name: String) -> Bool:
    var n = name.byte_length()
    var buf = alloc[UInt8](n + 1)
    var src = name.as_bytes()
    for i in range(n):
        buf[i] = src[i]
    buf[n] = 0
    var cname = _EnvPtr(unsafe_from_address=Int(buf))
    var ret = external_call["getenv", _EnvPtr](cname)
    buf.free()
    if Int(ret) == 0:
        return False  # unset → disabled
    # Compare the C string to exactly "1": ret[0]=='1' (0x31) and ret[1]==NUL.
    return ret[0] == UInt8(49) and ret[1] == UInt8(0)


# ── the report (mirrors flame GradFlowReport + EDv2 GradCoverage) ─────────────
@fieldwise_init
struct GradCoverage(Copyable, Movable):
    var total: Int          # adapter grads measured
    var nonzero: Int        # grads with a finite element of magnitude > 0
    var dead: Int           # grads identically zero (no finite nonzero element)
    var has_nonfinite: Bool # any grad contains a NaN/Inf element

    def coverage_pct(self) -> Float64:
        """nonzero / total * 100. 100% when total == 0 (mirrors
        grad_coverage.rs coverage_pct returning 1.0 for the empty set)."""
        if self.total == 0:
            return Float64(100.0)
        return Float64(self.nonzero) / Float64(self.total) * 100.0

    def is_clean(self) -> Bool:
        """True when every grad is finite and nonzero (flame is_clean)."""
        return self.dead == 0 and not self.has_nonfinite


# ── classify ONE host grad: (is_dead, is_nonfinite) ───────────────────────────
# Faithful port of flame grad_is_dead (diagnostics.rs:155-167): flame computes
# v = abs(g).sum_all() in F32 and returns dead := (!v.is_finite() || v == 0.0).
# A NaN/Inf anywhere makes the abs-sum non-finite, so flame folds a poisoned
# grad INTO `dead`. We replicate exactly:
#   is_nonfinite := any element is NaN/Inf  (detected via isfinite — abs(NaN)>0
#                   is False, so the magnitude test alone would MISS it)
#   is_dead      := is_nonfinite  OR  (no finite element of magnitude > 0)
# Thus a grad with one NaN and some finite nonzeros is DEAD (matches flame),
# not counted toward `nonzero` — so coverage_pct matches flame's ok_count.
def _classify(h: List[Float32]) -> Tuple[Bool, Bool]:
    var has_finite_nonzero = False
    var nonfinite = False
    for j in range(len(h)):
        var x = h[j]
        if not isfinite(x):
            nonfinite = True
        elif x != Float32(0.0):
            has_finite_nonzero = True
    var is_dead = nonfinite or not has_finite_nonzero
    return (is_dead, nonfinite)


# ── measure: count total / nonzero / dead + detect NaN/Inf over named grads ───
# Mirrors flame check_grad_flow + EDv2 GradCoverage::measure. `names` parallels
# `grads` (names used only for the env-gated abort message). A grad is "dead"
# when it is neither nonzero nor merely nonfinite-but-otherwise (we follow
# flame: a nonfinite grad is ALSO not healthy, but we account it under
# has_nonfinite, and it is NOT counted as nonzero so it lands in `dead` too —
# a NaN grad is both dead and nonfinite, which is the worst case and both
# trigger the abort).
def measure(names: List[String], grads: List[TArc], ctx: DeviceContext) raises -> GradCoverage:
    if len(names) != len(grads):
        raise Error(
            String("grad_coverage.measure: names/grads length mismatch (")
            + String(len(names)) + " != " + String(len(grads)) + ")"
        )
    var total = len(grads)
    var nonzero = 0
    var dead = 0
    var has_nonfinite = False
    for i in range(total):
        var h = grads[i][].to_host(ctx)
        var cls = _classify(h)
        var is_dead = cls[0]
        var is_nonfinite = cls[1]
        if is_nonfinite:
            has_nonfinite = True
        if is_dead:
            dead += 1
        else:
            nonzero += 1
    return GradCoverage(total, nonzero, dead, has_nonfinite)


# ── env-gated fail-fast (mirrors flame assert_grad_flow) ──────────────────────
# Measures, then — only if FLAME_ASSERT_GRAD_FLOW is set AND step >= 1 AND
# (dead > 0 OR has_nonfinite) — raises a clear Error naming the offending
# adapters. Otherwise returns the report. `step` gates the abort so a legitimate
# step-0 state (LoRA-B starts at zeros → its grad can be ~0 before any update)
# does not trip the guard; the dead-adapter contract is "dead at step >= 1".
def assert_grad_flow(
    names: List[String],
    grads: List[TArc],
    step: Int,
    ctx: DeviceContext,
) raises -> GradCoverage:
    var report = measure(names, grads, ctx)
    var armed = _env_is_set(String("FLAME_ASSERT_GRAD_FLOW"))
    var bad = (report.dead > 0 and step >= 1) or report.has_nonfinite
    if armed and bad:
        # Build the offender list naming each dead / nonfinite adapter.
        var msg = String("[grad-flow] FAILED at step ") + String(step)
        msg += String(": ") + String(report.dead) + String(" dead / ")
        msg += String(report.total) + String(" total")
        if report.has_nonfinite:
            msg += String(" (NON-FINITE grad present)")
        msg += String("\n")
        for i in range(len(grads)):
            var h = grads[i][].to_host(ctx)
            var cls = _classify(h)
            var is_dead = cls[0]
            var is_nonfinite = cls[1]
            if is_nonfinite:
                msg += String("    - ") + names[i] + String(" : NON-FINITE (NaN/Inf)\n")
            elif is_dead:
                msg += String("    - ") + names[i] + String(" : DEAD (grad identically zero)\n")
        raise Error(msg)
    return report^

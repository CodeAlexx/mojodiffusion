# training/grad_coverage_smoke.mojo — gate for grad_coverage.mojo (item 1c).
#
# Builds 3 toy grads:
#   g0 = all zeros        → DEAD (identically zero), finite
#   g1 = [1, NaN, 2]      → NON-FINITE (NaN) AND DEAD. Flame's grad_is_dead
#                            folds NaN into the abs-sum → non-finite → dead, so a
#                            NaN-poisoned grad is dead even though it has finite
#                            nonzero elements. NaN detected explicitly via
#                            isfinite (abs(NaN) > 0 is False).
#   g2 = [0.5, -0.5, 3]   → healthy (finite, nonzero)
#
# Asserts (the deliverable's gate — EXACT values, flame-faithful semantics):
#   total == 3
#   nonzero == 1          (only g2 healthy)
#   dead == 2             (g0 zero + g1 NaN-poisoned, matching flame grad_is_dead)
#   has_nonfinite == True (g1 has a NaN)
#   coverage_pct() == 1/3*100 ≈ 33.33  (matches flame ok_count: NaN lowers it)
#
# Then a SECOND, env-aware path proves the FLAME_ASSERT_GRAD_FLOW abort fires:
# assert_grad_flow over a dead grad at step 1. When the smoke is run WITH the
# env var set, that call MUST raise (nonzero exit). When run without it, it
# returns the report (exit 0). The driver script runs the file twice — once
# bare (PASS, exit 0) and once with FLAME_ASSERT_GRAD_FLOW=1 (abort, exit != 0).
#
# Build/run (JIT):
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/training/grad_coverage_smoke.mojo
# Abort path:
#   FLAME_ASSERT_GRAD_FLOW=1 pixi run mojo run -I . serenitymojo/training/grad_coverage_smoke.mojo

from std.math import nan
from std.ffi import external_call
from std.memory import alloc, UnsafePointer
from std.builtin.type_aliases import MutExternalOrigin
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.training.grad_coverage import (
    GradCoverage,
    measure,
    assert_grad_flow,
    _env_is_set,
    TArc,
)


comptime _EnvPtr = UnsafePointer[UInt8, MutExternalOrigin]


# libc setenv(name, value, overwrite=1) for the in-process strict-"1" check.
def _setenv(name: String, value: String):
    var nn = name.byte_length()
    var nb = alloc[UInt8](nn + 1)
    var nsrc = name.as_bytes()
    for i in range(nn):
        nb[i] = nsrc[i]
    nb[nn] = 0
    var vn = value.byte_length()
    var vb = alloc[UInt8](vn + 1)
    var vsrc = value.as_bytes()
    for i in range(vn):
        vb[i] = vsrc[i]
    vb[vn] = 0
    var cn = _EnvPtr(unsafe_from_address=Int(nb))
    var cv = _EnvPtr(unsafe_from_address=Int(vb))
    _ = external_call["setenv", Int32](cn, cv, Int32(1))
    nb.free()
    vb.free()


def _grad(vals: List[Float32], ctx: DeviceContext) raises -> TArc:
    var shape = List[Int]()
    shape.append(len(vals))
    return TArc(Tensor.from_host(vals, shape^, STDtype.F32, ctx))


def main() raises:
    var ctx = DeviceContext()

    # g0: all zeros (DEAD)
    var v0 = List[Float32]()
    v0.append(0.0); v0.append(0.0); v0.append(0.0)
    # g1: contains a NaN (NON-FINITE)
    var v1 = List[Float32]()
    v1.append(1.0); v1.append(nan[DType.float32]()); v1.append(2.0)
    # g2: healthy
    var v2 = List[Float32]()
    v2.append(0.5); v2.append(-0.5); v2.append(3.0)

    var names = List[String]()
    names.append(String("adapter.dead"))
    names.append(String("adapter.nan"))
    names.append(String("adapter.healthy"))

    var grads = List[TArc]()
    grads.append(_grad(v0, ctx))
    grads.append(_grad(v1, ctx))
    grads.append(_grad(v2, ctx))

    var rep = measure(names, grads, ctx)
    print("measure: total=", rep.total, " nonzero=", rep.nonzero,
          " dead=", rep.dead, " has_nonfinite=", rep.has_nonfinite,
          " coverage_pct=", rep.coverage_pct())

    var ok = True
    if rep.total != 3:
        print("FAIL total != 3 (got ", rep.total, ")"); ok = False
    else:
        print("PASS total == 3")
    if rep.nonzero != 1:
        print("FAIL nonzero != 1 (got ", rep.nonzero, ")"); ok = False
    else:
        print("PASS nonzero == 1 (only g2 healthy)")
    if rep.dead != 2:
        print("FAIL dead != 2 (got ", rep.dead, ")"); ok = False
    else:
        print("PASS dead == 2 (g0 zero + g1 NaN-poisoned, flame grad_is_dead)")
    if not rep.has_nonfinite:
        print("FAIL has_nonfinite != True"); ok = False
    else:
        print("PASS has_nonfinite == True")
    # coverage_pct must equal 1/3*100 ≈ 33.333… (flame ok_count parity).
    var cov = rep.coverage_pct()
    if cov < 33.0 or cov > 33.7:
        print("FAIL coverage_pct != ~33.33 (got ", cov, ")"); ok = False
    else:
        print("PASS coverage_pct == ~33.33 (got ", cov, ") — NaN lowers coverage")

    if not ok:
        raise Error("grad_coverage_smoke FAILED")
    print("grad_coverage_smoke numeric gate PASS")

    # ── strict-"1" env semantics gate (finding #1) ────────────────────────────
    # flame flag_enabled arms ONLY on value=="1". Prove _env_is_set matches:
    # "1" → True, "0" → False, "true"/"" → False. Uses a probe var, not the
    # real FLAME_ASSERT_GRAD_FLOW, so we don't perturb the abort-path test.
    var probe = String("SM_GRADCOV_ENV_PROBE")
    _setenv(probe, String("1"))
    if not _env_is_set(probe):
        print("FAIL env probe: value '1' should arm"); ok = False
    else:
        print("PASS env '1' arms")
    _setenv(probe, String("0"))
    if _env_is_set(probe):
        print("FAIL env probe: value '0' must NOT arm (flame disables on =0)"); ok = False
    else:
        print("PASS env '0' does NOT arm (matches flame flag_enabled)")
    _setenv(probe, String("true"))
    if _env_is_set(probe):
        print("FAIL env probe: value 'true' must NOT arm (flame is strict ==\"1\")"); ok = False
    else:
        print("PASS env 'true' does NOT arm")
    if not ok:
        raise Error("grad_coverage_smoke env-semantics gate FAILED")
    print("grad_coverage_smoke env-semantics gate PASS")

    # ── env-gated abort path ──────────────────────────────────────────────────
    # Re-run assert_grad_flow over a DEAD grad at step 1. Rebuild the grads
    # (to_host does not consume, but measure already read them; build fresh to
    # be unambiguous about ownership).
    var armed = _env_is_set(String("FLAME_ASSERT_GRAD_FLOW"))
    print("FLAME_ASSERT_GRAD_FLOW armed =", armed)

    var dn = List[String]()
    dn.append(String("adapter.dead2"))
    var dg = List[TArc]()
    var vd = List[Float32]()
    vd.append(0.0); vd.append(0.0)
    dg.append(_grad(vd, ctx))

    # When armed, this MUST raise (dead > 0 at step 1). When unarmed, returns.
    var rep2 = assert_grad_flow(dn, dg, 1, ctx)
    # Only reached when NOT armed.
    print("assert_grad_flow returned (env unset path): dead=", rep2.dead)
    print("grad_coverage_smoke ABORT-PATH did NOT fire (FLAME_ASSERT_GRAD_FLOW unset) — expected when bare")

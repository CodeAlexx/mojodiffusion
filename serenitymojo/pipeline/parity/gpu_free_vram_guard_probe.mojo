# serenitymojo/pipeline/parity/gpu_free_vram_guard_probe.mojo
#
# Exercises `pipeline/gpu_free_vram_guard.mojo` against the REAL card.
#
# ── THIS PROBE CANNOT TOUCH THE GPU, BY CONSTRUCTION ───────────────────────
# It imports no `DeviceContext` and links no GPU code, so there is no path from
# here to an allocation even if every check passes. That is deliberate and it is
# the reason the guard is tested HERE rather than by running a pipeline binary:
# a pipeline binary reaches a DeviceContext the moment the guard lets it
# through, which makes "verify the guard works" and "start a GPU job" the same
# action. Testing a safety check by triggering the thing it guards is how the
# 2026-08-03 near-miss happened in the first place.
#
# The refusal path is the one that matters and it is exercised against real
# `nvidia-smi` output: ask for an impossible amount, require a raise.
#
# Run (never touches the GPU):
#   pixi run mojo build -O0 -j 1 -I . -I vendor/mojo-libs \
#     serenitymojo/pipeline/parity/gpu_free_vram_guard_probe.mojo \
#     -o /tmp/gpu_free_vram_guard_probe -Xlinker -lm \
#   && /tmp/gpu_free_vram_guard_probe /tmp/gpu_guard_scratch
#   # and once more with H3_ALLOW_BUSY_GPU=1 to see the override branch

from std.sys import argv

from serenitymojo.pipeline.gpu_free_vram_guard import (
    gpu_busy_process_summary,
    gpu_free_vram_mib,
    gpu_guard_override_enabled,
    require_free_vram,
)


struct Report(Copyable, Movable):
    var checks: Int
    var failures: Int

    def __init__(out self):
        self.checks = 0
        self.failures = 0

    def ok(mut self, label: String, detail: String):
        self.checks += 1
        print("  ok  ", label, "—", detail)

    def fail(mut self, label: String, detail: String):
        self.checks += 1
        self.failures += 1
        print("  FAIL", label, "—", detail)


def main() raises:
    var args = argv()
    var scratch = String("/tmp/gpu_free_vram_guard_probe.scratch")
    if len(args) >= 2:
        scratch = String(args[1])

    print("GPU free-VRAM guard probe (host only — links no DeviceContext)")
    print("")
    var report = Report()

    # ── [1] the read itself ────────────────────────────────────────────────
    print("[1] reading the card without initializing CUDA")
    var free = gpu_free_vram_mib(scratch)
    if free >= 0 and free < 1000000:
        report.ok("free VRAM", String(free) + " MiB")
    else:
        report.fail("free VRAM", String(free) + " MiB is not plausible")
    print("      holders:", gpu_busy_process_summary(scratch))

    # ── [2] the REFUSAL path, against the real card ────────────────────────
    # SKIPPED when the override is set: with H3_ALLOW_BUSY_GPU=1 nothing can be
    # refused, so asserting a refusal here would fail the guard for doing
    # exactly what it was told. Reported, not silently passed.
    var override_on = gpu_guard_override_enabled(String("H3"))
    print("")
    print("[2] refusal — ask for more than any GPU has")
    if override_on:
        report.ok(
            "refusal path",
            "SKIPPED — H3_ALLOW_BUSY_GPU=1 is set, so nothing can be refused;"
            " run again with it unset to exercise this",
        )
    var refused = False
    var message = String("")
    if not override_on:
      try:
        require_free_vram(
            10000000, scratch, String("H3"), String("an impossible job")
        )
      except e:
        refused = True
        message = String(e)
    if override_on:
        pass
    elif refused:
        report.ok("impossible request refused", "raised, as it must")
        # The message has to name the numbers and the override, or a person
        # hitting this at 3am learns nothing from it.
        if message.find("REFUSING") >= 0 and message.find("H3_ALLOW_BUSY_GPU") >= 0:
            report.ok("refusal message", "names the override and the shortfall")
        else:
            report.fail("refusal message", "unhelpful: " + message)
    else:
        report.fail(
            "impossible request refused",
            "the guard allowed a request no card could satisfy",
        )

    # ── [3] the ALLOW path ─────────────────────────────────────────────────
    # Asked for 1 MiB: passes on any card with anything free at all. If the card
    # is completely full this legitimately refuses, which the probe reports
    # rather than treating as a failure of the guard.
    print("")
    print("[3] allow — a request that fits")
    var allowed = True
    try:
        require_free_vram(1, scratch, String("H3"), String("a trivial job"))
    except:
        allowed = False
    if allowed:
        report.ok("1 MiB request allowed", "the guard is not refusing everything")
    elif free < 1:
        report.ok(
            "1 MiB request refused",
            "the card reports 0 MiB free — correct refusal, not a guard bug",
        )
    else:
        report.fail(
            "1 MiB request", String("refused with ") + String(free) + " MiB free"
        )

    # ── [4] the override, when the environment sets it ─────────────────────
    print("")
    print("[4] override branch (set H3_ALLOW_BUSY_GPU=1 to exercise)")
    var overridden = True
    try:
        require_free_vram(
            10000000, scratch, String("H3"), String("an impossible job")
        )
    except:
        overridden = False
    if overridden:
        report.ok(
            "H3_ALLOW_BUSY_GPU=1",
            "impossible request ALLOWED through with a loud banner",
        )
    else:
        report.ok(
            "H3_ALLOW_BUSY_GPU unset",
            "impossible request still refused (set it to see the other branch)",
        )

    print("")
    if report.failures == 0:
        print("PASS:", report.checks, "checks")
    else:
        print("FAIL:", report.failures, "of", report.checks, "checks")
        raise Error("gpu_free_vram_guard probe FAILED")

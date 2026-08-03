# serenitymojo/pipeline/gpu_free_vram_guard.mojo
#
# Refuse to start a GPU job that will not fit alongside what is already on the
# card. Host only — this runs BEFORE `DeviceContext()` and must never itself
# touch the GPU.
#
# ── WHY THIS EXISTS (a real near-miss, 2026-08-03) ─────────────────────────
# `scripts/minimax_h3_i2va_smoke.sh` already had this check and it did not help,
# because the check was in the WRAPPER: invoking the binary directly walked
# straight past it. A keyframe run then created a DeviceContext while an
# overnight hero render owned ~21.4 GiB, held memory for ~32 seconds during that
# job's own allocation ramp, and died with CUDA_ERROR_OUT_OF_MEMORY. The hero
# survived, but only because the OOM happened to land on the newcomer; with
# slightly different timing it would have killed the long render instead.
#
# The lesson, and the reason this is a module and not another shell line: A
# GUARD THAT LIVES OUTSIDE THE BINARY IS NOT A GUARD. It protects the invocation
# someone remembered to route through it, which is never the invocation that
# causes the incident. This belongs in every pipeline's preflight, next to the
# checkpoint checks that already fail loud there.
#
# ── HOW IT READS THE CARD, AND WHY NOT NVML ───────────────────────────────
# `nvidia-smi --query-gpu=memory.free`, through the same
# `io/ffi.sys_system` + `components/artifacts.shell_quote` + scratch-file route
# `pipeline/minimax_h3_media_in.mojo` and `pipeline/scail2_decode.mojo` already
# use for ffprobe. NVML over FFI would avoid the subprocess, but it means
# dlopen'ing libnvidia-ml and managing its init/shutdown from a process whose
# entire purpose is to NOT initialize the GPU stack yet — the subprocess cannot
# perturb this process's CUDA context because there isn't one.
#
# ── FAIL-OPEN vs FAIL-CLOSED, chosen deliberately ─────────────────────────
# If `nvidia-smi` cannot be run or its output cannot be parsed, this RAISES
# rather than assuming the card is free. A guard that silently disables itself
# when the tool is missing is the same failure as having no guard, and the
# incident above is exactly the case where "assume it's fine" is wrong. A
# machine legitimately without nvidia-smi is a machine that cannot run these
# pipelines anyway.
#
# ── THE OVERRIDE ───────────────────────────────────────────────────────────
# `<PREFIX>_ALLOW_BUSY_GPU=1` skips the refusal, prints what it is overriding,
# and continues. Deliberately an ENV VAR and not a flag: it is a thing a human
# types on purpose for one run, not something a script quietly inherits from a
# config file.

from std.ffi import external_call
from std.memory import alloc

from serenitymojo.components.artifacts import shell_quote
from serenitymojo.io.ffi import sys_system

comptime _EnvPtr = UnsafePointer[UInt8, MutExternalOrigin]


def _env_str(name: String) -> String:
    """libc `getenv` -> String, empty when unset.

    The `external_call` form rather than `std.os.getenv`, following
    `models/wan22/wan22_stack_lora.mojo`'s note that the stdlib overload does
    not lower in this build."""
    var n = name.byte_length()
    var buf = alloc[UInt8](n + 1)
    var src = name.as_bytes()
    for i in range(n):
        buf[i] = src[i]
    buf[n] = 0
    var ret = external_call["getenv", _EnvPtr](_EnvPtr(unsafe_from_address=Int(buf)))
    buf.free()
    var out = String("")
    if Int(ret) == 0:
        return out
    var i = 0
    while ret[i] != UInt8(0):
        out += chr(Int(ret[i]))
        i += 1
    return out^


def _read_text(path: String) raises -> String:
    var text: String
    with open(path, "r") as f:
        text = f.read()
    return text^


def _first_int(text: String) raises -> Int:
    """The first run of ASCII digits in `text`.

    `nvidia-smi --format=csv,noheader,nounits` emits one bare integer per line;
    taking the first digit run tolerates a stray unit suffix or trailing
    whitespace without pulling in a number parser."""
    var bytes = text.as_bytes()
    var i = 0
    var n = text.byte_length()
    while i < n and (bytes[i] < UInt8(48) or bytes[i] > UInt8(57)):
        i += 1
    if i >= n:
        raise Error(
            String("gpu_free_vram_guard: no integer in nvidia-smi output: '")
            + text + "'"
        )
    var value = 0
    while i < n and bytes[i] >= UInt8(48) and bytes[i] <= UInt8(57):
        value = value * 10 + Int(bytes[i] - UInt8(48))
        i += 1
    return value


def gpu_free_vram_mib(scratch_path: String) raises -> Int:
    """Free VRAM on GPU 0, in MiB, without initializing CUDA in this process."""
    var command = (
        String("nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits")
        + " | head -1 > " + shell_quote(scratch_path)
    )
    if sys_system(command) != 0:
        raise Error(
            "gpu_free_vram_guard: `nvidia-smi` failed. This guard FAILS CLOSED"
            " rather than assume the card is free — see this module's header."
            " Set <PREFIX>_ALLOW_BUSY_GPU=1 to run anyway."
        )
    return _first_int(_read_text(scratch_path))


def gpu_busy_process_summary(scratch_path: String) raises -> String:
    """Who is holding memory, for the refusal message.

    Diagnostics only: a refusal that says "not enough free VRAM" sends someone
    to `nvidia-smi` anyway, so it may as well answer the next question."""
    var command = (
        String("nvidia-smi --query-compute-apps=pid,used_memory")
        + " --format=csv,noheader 2>/dev/null | head -8 > "
        + shell_quote(scratch_path)
    )
    if sys_system(command) != 0:
        return String("(could not list compute apps)")
    var text = _read_text(scratch_path)
    if text.byte_length() == 0:
        return String("(no compute apps reported)")
    var out = String("")
    var bytes = text.as_bytes()
    for i in range(text.byte_length()):
        if bytes[i] == UInt8(10):
            out += "; "
        else:
            out += chr(Int(bytes[i]))
    return out^


def gpu_guard_override_enabled(env_prefix: String) -> Bool:
    """Whether `<PREFIX>_ALLOW_BUSY_GPU=1` is set.

    Public because two callers need it and neither should re-read the
    environment itself: the guard's own probe, to know which branch to expect,
    and a pipeline writing its result manifest — a render that overrode the
    guard is not the same artifact as one that did not, and the manifest should
    say so."""
    return _env_str(env_prefix + "_ALLOW_BUSY_GPU") == String("1")


def require_free_vram(
    needed_mib: Int,
    scratch_path: String,
    env_prefix: String,
    what: String,
) raises:
    """Refuse to proceed unless at least `needed_mib` MiB are free.

    CALL THIS BEFORE `DeviceContext()`, not after — the whole point is that no
    allocation has happened yet when the decision is made.

    `env_prefix` names the override variable (`H3` -> `H3_ALLOW_BUSY_GPU`), so
    each pipeline's escape hatch is its own and enabling one does not silently
    enable the others. `what` is named in the message so a refusal says which
    job was stopped."""
    if needed_mib <= 0:
        raise Error("gpu_free_vram_guard: needed_mib must be positive")

    var override_name = env_prefix + "_ALLOW_BUSY_GPU"
    var free = gpu_free_vram_mib(scratch_path)

    if free >= needed_mib:
        print(
            "  gpu guard:", free, "MiB free >=", needed_mib,
            "MiB needed for", what, "— proceeding",
        )
        return

    if gpu_guard_override_enabled(env_prefix):
        print("")
        print("  ################################################################")
        print("  # " + override_name + "=1 — GPU GUARD OVERRIDDEN.")
        print("  # Only", free, "MiB free;", what, "wants", needed_mib, "MiB.")
        print("  # Holders:", gpu_busy_process_summary(scratch_path))
        print("  # If another job is mid-run, one of you is about to OOM.")
        print("  ################################################################")
        print("")
        return

    raise Error(
        String("gpu_free_vram_guard: REFUSING to start ") + what + " — only "
        + String(free) + " MiB free, need " + String(needed_mib) + " MiB."
        " Holders: " + gpu_busy_process_summary(scratch_path)
        + ". Wait for the card, or set " + override_name + "=1 to run anyway"
        " (which risks OOM-ing whichever job loses the race — a keyframe run"
        " did exactly this to an overnight render on 2026-08-03)."
    )

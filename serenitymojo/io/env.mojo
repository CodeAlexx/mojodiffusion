# io/env.mojo — tiny getenv wrapper for runtime string config.
#
# `env_or(name, default)` reads an environment variable as a String, returning
# `default` when it is unset or empty. Used to make hardcoded ckpt-dir literals
# in the parity/pipeline drivers runtime-switchable (e.g. LINGBOT_CKPT to point
# the LingBot-Video MoE loader at transformer_fp8/ instead of the bf16 dir).
#
# getenv idiom copied from io/disk_check_smoke.mojo:33-45 (external_call getenv +
# NUL-terminated request buffer), extended to read the full returned C string.

from std.ffi import external_call
from std.memory import alloc, UnsafePointer


comptime _EnvPtr = UnsafePointer[UInt8, MutExternalOrigin]


def env_or(name: String, default: String) -> String:
    """Return the value of environment variable `name`, or `default` if unset/empty."""
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
        return default
    var out = String("")
    var i = 0
    while ret[i] != 0:
        out += chr(Int(ret[i]))
        i += 1
    if out.byte_length() == 0:
        return default
    return out


def env_int(name: String, default: Int) -> Int:
    """Return env var `name` parsed as a non-negative base-10 Int, or `default`
    if unset/empty/non-numeric. Used for LINGBOT_RESIDENT_BLOCKS."""
    var s = env_or(name, String(""))
    if s.byte_length() == 0:
        return default
    var v = 0
    var bytes = s.as_bytes()
    for i in range(len(bytes)):
        var c = Int(bytes[i])
        if c < 48 or c > 57:
            return default
        v = v * 10 + (c - 48)
    return v


def serenity_home() -> String:
    """Return Serenity's writable state root without assuming a user name."""
    return env_or(
        "SERENITY_HOME",
        env_or("HOME", String(".")) + String("/.serenity"),
    )


def serenity_model_root() -> String:
    """Return the shared model root used by trainers, inference, and probes."""
    return env_or(
        "SERENITY_MODEL_ROOT",
        serenity_home() + String("/models"),
    )


def serenity_repo_root() -> String:
    """Return the MojoDiffusion checkout root for developer-only artifacts."""
    return env_or("SERENITY_REPO_ROOT", String("."))


def serenity_checkpoint(name: String) -> String:
    """Resolve a checkpoint filename below the shared model root."""
    return serenity_model_root() + String("/checkpoints/") + name


def serenity_output(path: String) -> String:
    """Resolve a developer artifact below the checkout's ignored output tree."""
    return serenity_repo_root() + String("/output/") + path

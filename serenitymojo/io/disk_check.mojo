# io/disk_check.mojo — pre-checkpoint free-disk-space probe + guard.
#
# Pure-Mojo port of EriDiffusion-v2
# crates/eridiffusion-core/src/training/features/disk_check.rs (check_free_space).
#
# ── Approach: `df` shellout (matches the Rust reference) ──────────────────────
# The reference uses `df --output=avail -B1 <path>` and parses the available-bytes
# integer from line 2 of stdout. We do the SAME: run df, redirecting stdout to a
# temp file, then read+parse the temp file with the proven sys_open/sys_pread FFI
# helpers in io/ffi.mojo. statvfs(2) would give free bytes directly, but it
# requires passing the platform-specific `struct statvfs` layout across the FFI —
# the same hazard io/ffi.mojo's file_size() deliberately avoids for `struct stat`
# ("avoids needing the platform-specific struct stat layout that fstat requires").
# The df shellout is portable by construction and byte-for-byte the reference's
# semantics.
#
# ── Semantics (mirrors check_free_space) ──────────────────────────────────────
# free_bytes_for(path):
#   - probes `path`, or — if it does not exist — walks up to the closest existing
#     ancestor (df on a missing dir errors on some systems), exactly as the rs.
#   - returns the available bytes as an Int, or -1 when free space cannot be
#     determined (df missing / non-zero status / unparseable output). The -1
#     sentinel is the "best-effort: don't block training on a failed probe"
#     contract — the rs returns Ok in that case; here a caller treats -1 as "skip".
# guard_free_space(path, required):
#   - RAISES when free space IS determinable AND < required (the rs Err arm).
#   - returns silently (best-effort skip) when free space is undeterminable (-1),
#     matching the rs Ok fallback. A 0-byte requirement always passes.
#
# Mojo 0.26.x: `def` not `fn`; String c-string staging via io/ffi helpers.

from std.collections import List
from std.memory import alloc
from serenitymojo.io.ffi import (
    BytePtr,
    O_RDONLY,
    sys_open,
    sys_pread,
    sys_close,
    sys_system,
    file_size,
)


# A process-unique-ish temp path for df output. PID would be ideal; absent a
# cheap getpid here, a fixed name in /tmp is adequate for a one-shot init probe
# (the file is fully overwritten each call via `>`).
comptime _DF_TMP = "/tmp/.serenitymojo_disk_check.txt"


# ── does `path` exist? (open O_RDONLY; dirs open read-only on Linux) ──────────
# A successful open → exists. We do NOT use stat (struct-layout hazard). Opening
# a directory O_RDONLY succeeds on Linux, which is all we need for the ancestor
# walk. Returns True on a valid fd (and closes it).
def _path_exists(path: String) -> Bool:
    var fd = sys_open(path, O_RDONLY, 0)
    if fd < 0:
        return False
    _ = sys_close(fd)
    return True


# ── build s[start:end] over BYTES (range-slice `s[a:b]` is not supported in
# 1.0.0b1; the codebase builds substrings byte-by-byte — see lora._substr_bytes).
def _substr_bytes(s: String, start: Int, end: Int) -> String:
    var out = String("")
    var bytes = s.as_bytes()
    for i in range(start, end):
        out += chr(Int(bytes[i]))
    return out^


# ── closest existing ancestor of `path` (mirrors the rs parent-walk) ──────────
# Strips the last `/`-segment repeatedly until an existing dir is found; falls
# back to "/" (always exists). Used so df is never called on a missing dir.
def _existing_ancestor(path: String) -> String:
    if _path_exists(path):
        return path
    var p = path
    while True:
        # find the last '/'
        var cut = -1
        var bytes = p.as_bytes()
        var nb = len(bytes)
        for i in range(nb):
            if bytes[i] == UInt8(47):  # '/'
                cut = i
        if cut <= 0:
            return String("/")
        p = _substr_bytes(p, 0, cut)
        if _path_exists(p):
            return p


# ── read a small text file fully into a host String ──────────────────────────
def _read_text(path: String) raises -> String:
    var fd = sys_open(path, O_RDONLY, 0)
    if fd < 0:
        raise Error("disk_check: cannot open df output file")
    var sz = file_size(fd)
    if sz <= 0:
        _ = sys_close(fd)
        return String("")
    var buf = alloc[UInt8](sz)
    var bp = BytePtr(unsafe_from_address=Int(buf))
    var got = sys_pread(fd, bp, sz, 0)
    _ = sys_close(fd)
    var out = String("")
    var ncopy = got if got > 0 else 0
    for i in range(ncopy):
        out += chr(Int(buf[i]))
    buf.free()
    return out^


# ── parse the available-bytes integer from `df --output=avail -B1` stdout ─────
# Line 1 is the "Avail" header; line 2 is the integer. Returns the parsed Int, or
# -1 if the output is malformed (best-effort). Whitespace-trims the value line.
def _parse_df_avail(text: String) -> Int:
    # split into lines
    var lines = List[String]()
    var cur = String("")
    var bytes = text.as_bytes()
    for i in range(len(bytes)):
        if bytes[i] == UInt8(10):  # '\n'
            lines.append(cur)
            cur = String("")
        else:
            cur += chr(Int(bytes[i]))
    if cur.byte_length() > 0:
        lines.append(cur)
    if len(lines) < 2:
        return -1
    # trim spaces/tabs from line[1]
    var raw = lines[1]
    var rb = raw.as_bytes()
    var start = 0
    var end = len(rb)
    while start < end and (rb[start] == UInt8(32) or rb[start] == UInt8(9)):
        start += 1
    while end > start and (rb[end - 1] == UInt8(32) or rb[end - 1] == UInt8(9)):
        end -= 1
    if end <= start:
        return -1
    var val = 0
    for i in range(start, end):
        var c = rb[i]
        if c < UInt8(48) or c > UInt8(57):  # not 0-9
            return -1
        val = val * 10 + Int(c - UInt8(48))
    return val


# ── free_bytes_for: available bytes at `path` (or its ancestor), -1 if unknown ─
def free_bytes_for(path: String) raises -> Int:
    var probe = _existing_ancestor(path)
    # df --output=avail -B1 <probe> > tmp   (-B1 → bytes; --output=avail → just the column)
    var cmd = String("df --output=avail -B1 '") + probe + String("' > ") + String(_DF_TMP) + String(" 2>/dev/null")
    var status = sys_system(cmd)
    if status != 0:
        return -1   # df missing or errored → undeterminable (best-effort skip)
    var text = _read_text(String(_DF_TMP))
    return _parse_df_avail(text)


# ── guard_free_space: RAISE iff free is KNOWN and < required ───────────────────
# Best-effort: a -1 (undeterminable) probe does NOT raise (mirrors the rs Ok
# fallback — "treat as skip this save, not abort"). required==0 always passes.
def guard_free_space(path: String, required: Int) raises:
    if required <= 0:
        return
    var avail = free_bytes_for(path)
    if avail < 0:
        # undeterminable → best-effort skip (do not block).
        return
    if avail < required:
        raise Error(
            String("[disk-check] insufficient disk space: ") + String(avail)
            + String(" bytes available at '") + path + String("', need ")
            + String(required)
        )

# clipboard.clipboard - Linux desktop clipboard helpers for Mojo apps.
#
# The library intentionally keeps payload bytes out of shell command strings:
# desktop provider commands are static, while clipboard data flows over stdin or
# stdout through libc popen/fread/fwrite. This avoids shell injection from app
# content and keeps the Mojo side dependency-light. Runtime providers:
#   * Wayland: wl-copy / wl-paste
#   * X11:     xclip or xsel
#   * OSC52:   explicit write-only terminal escape fallback

from std.ffi import external_call
from std.memory import alloc, UnsafePointer

comptime BytePtr = UnsafePointer[UInt8, MutExternalOrigin]

comptime BACKEND_AUTO = 0
comptime BACKEND_WAYLAND = 1
comptime BACKEND_XCLIP = 2
comptime BACKEND_XSEL = 3
comptime BACKEND_OSC52 = 4

comptime SELECTION_CLIPBOARD = 0
comptime SELECTION_PRIMARY = 1

comptime PIPE_CHUNK = 65536
comptime DEFAULT_MAX_READ_BYTES = 64 * 1024 * 1024
comptime B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"


def _cbuf(s: String) -> BytePtr:
    """NUL-terminated copy of `s` for libc calls."""
    var n = s.byte_length()
    var b = alloc[UInt8](n + 1)
    var src = s.as_bytes()
    for i in range(n):
        b[i] = src[i]
    b[n] = 0
    return BytePtr(unsafe_from_address=Int(b))


def _bytes_to_string(data: List[UInt8]) -> String:
    var n = len(data)
    if n == 0:
        return String("")
    var buf = alloc[UInt8](n)
    for i in range(n):
        buf[i] = data[i]
    var out = String(StringSlice(unsafe_from_utf8=Span(unsafe_ptr=BytePtr(unsafe_from_address=Int(buf)), length=n)))
    buf.free()
    return out^


def _string_to_bytes(s: String) -> List[UInt8]:
    var out = List[UInt8]()
    var b = s.as_bytes()
    for i in range(s.byte_length()):
        out.append(b[i])
    return out^


def env_nonempty(name: String) -> Bool:
    """True when an environment variable exists and is not empty."""
    var np = _cbuf(name)
    var p = external_call["getenv", BytePtr](np)
    np.free()
    if Int(p) == 0:
        return False
    return p[0] != 0


def _run_status(cmd: String) -> Int:
    var cp = _cbuf(cmd)
    var rc = Int(external_call["system", Int32](cp))
    cp.free()
    return rc


def _safe_command_name(name: String) -> Bool:
    if name.byte_length() == 0:
        return False
    var b = name.as_bytes()
    for i in range(name.byte_length()):
        var c = Int(b[i])
        var alpha = (c >= 65 and c <= 90) or (c >= 97 and c <= 122)
        var digit = c >= 48 and c <= 57
        var punct = c == 45 or c == 46 or c == 95
        if not (alpha or digit or punct):
            return False
    return True


def command_exists(name: String) -> Bool:
    """PATH probe for simple command names.

    Rejects shell metacharacters before calling `command -v`.
    """
    if not _safe_command_name(name):
        return False
    return _run_status(String("command -v ") + name + String(" >/dev/null 2>&1")) == 0


def _popen(cmd: String, mode: String) -> Int:
    var cp = _cbuf(cmd)
    var mp = _cbuf(mode)
    var fp = external_call["popen", BytePtr](cp, mp)
    cp.free()
    mp.free()
    return Int(fp)


def _pclose(fp_addr: Int) -> Int:
    if fp_addr == 0:
        return -1
    return Int(external_call["pclose", Int32](BytePtr(unsafe_from_address=fp_addr)))


def _read_cmd_bytes(cmd: String, max_bytes: Int = DEFAULT_MAX_READ_BYTES) raises -> List[UInt8]:
    var fp_addr = _popen(cmd, String("r"))
    if fp_addr == 0:
        raise Error("clipboard: popen failed for read provider")
    var fp = BytePtr(unsafe_from_address=fp_addr)
    var buf = alloc[UInt8](PIPE_CHUNK)
    var bp = BytePtr(unsafe_from_address=Int(buf))
    var out = List[UInt8]()
    while True:
        var n = external_call["fread", Int](bp, Int(1), Int(PIPE_CHUNK), fp)
        if n <= 0:
            break
        if len(out) + n > max_bytes:
            buf.free()
            _ = _pclose(fp_addr)
            raise Error(
                "clipboard: provider output exceeded max_bytes="
                + String(max_bytes)
            )
        for i in range(n):
            out.append(buf[i])
    buf.free()
    var rc = _pclose(fp_addr)
    if rc != 0:
        raise Error("clipboard: read provider exited with status " + String(rc))
    return out^


def _write_cmd_bytes(cmd: String, data: List[UInt8]) raises:
    var fp_addr = _popen(cmd, String("w"))
    if fp_addr == 0:
        raise Error("clipboard: popen failed for write provider")
    var fp = BytePtr(unsafe_from_address=fp_addr)
    var n = len(data)
    var total = 0
    if n > 0:
        var src = BytePtr(unsafe_from_address=Int(data.unsafe_ptr()))
        while total < n:
            var wrote = external_call["fwrite", Int](
                src + total, Int(1), Int(n - total), fp
            )
            if wrote <= 0:
                _ = _pclose(fp_addr)
                raise Error("clipboard: provider stdin write failed")
            total += wrote
    var rc = _pclose(fp_addr)
    if rc != 0:
        raise Error(
            "clipboard: write provider failed (status=" + String(rc) + ")"
        )


def _validate_selection(selection: Int) raises:
    if selection != SELECTION_CLIPBOARD and selection != SELECTION_PRIMARY:
        raise Error("clipboard: invalid selection " + String(selection))


def backend_name(backend: Int) -> String:
    if backend == BACKEND_AUTO:
        return String("auto")
    if backend == BACKEND_WAYLAND:
        return String("wayland")
    if backend == BACKEND_XCLIP:
        return String("xclip")
    if backend == BACKEND_XSEL:
        return String("xsel")
    if backend == BACKEND_OSC52:
        return String("osc52")
    return String("unknown")


def selection_name(selection: Int) -> String:
    if selection == SELECTION_CLIPBOARD:
        return String("clipboard")
    if selection == SELECTION_PRIMARY:
        return String("primary")
    return String("unknown")


def backend_available(backend: Int) -> Bool:
    """True when the named backend is usable in the current environment.

    OSC52 is write-only. It reports available when TERM is set; read_text() will
    still reject it because terminal OSC52 has no portable read path.
    """
    if backend == BACKEND_WAYLAND:
        return (
            env_nonempty(String("WAYLAND_DISPLAY"))
            and command_exists(String("wl-copy"))
            and command_exists(String("wl-paste"))
        )
    if backend == BACKEND_XCLIP:
        return (
            env_nonempty(String("DISPLAY"))
            and command_exists(String("xclip"))
        )
    if backend == BACKEND_XSEL:
        return (
            env_nonempty(String("DISPLAY"))
            and command_exists(String("xsel"))
        )
    if backend == BACKEND_OSC52:
        return env_nonempty(String("TERM"))
    return False


def detect_backend(selection: Int = SELECTION_CLIPBOARD) raises -> Int:
    """Select the best full read/write provider for this process.

    Prefers Wayland when WAYLAND_DISPLAY is present, then X11 providers. OSC52 is
    not auto-selected because it is write-only and visibly emits terminal escape
    sequences.
    """
    _validate_selection(selection)
    if backend_available(BACKEND_WAYLAND):
        return BACKEND_WAYLAND
    if backend_available(BACKEND_XCLIP):
        return BACKEND_XCLIP
    if backend_available(BACKEND_XSEL):
        return BACKEND_XSEL
    return -1


def availability_report() -> String:
    var out = String("clipboard backends:")
    out += String(" wayland=") + String(backend_available(BACKEND_WAYLAND))
    out += String(" xclip=") + String(backend_available(BACKEND_XCLIP))
    out += String(" xsel=") + String(backend_available(BACKEND_XSEL))
    out += String(" osc52=") + String(backend_available(BACKEND_OSC52))
    if not env_nonempty(String("WAYLAND_DISPLAY")) and not env_nonempty(String("DISPLAY")):
        out += String(" (no WAYLAND_DISPLAY or DISPLAY)")
    return out^


def _resolve_backend(backend: Int, selection: Int) raises -> Int:
    _validate_selection(selection)
    if backend == BACKEND_AUTO:
        var detected = detect_backend(selection)
        if detected < 0:
            raise Error(
                "clipboard: no full read/write backend available; install "
                + "wl-clipboard, xclip, or xsel and ensure WAYLAND_DISPLAY or "
                + "DISPLAY is set. " + availability_report()
            )
        return detected
    var valid = (
        backend == BACKEND_WAYLAND
        or backend == BACKEND_XCLIP
        or backend == BACKEND_XSEL
        or backend == BACKEND_OSC52
    )
    if not valid:
        raise Error("clipboard: invalid backend " + String(backend))
    if not backend_available(backend):
        raise Error(
            "clipboard: backend " + backend_name(backend)
            + " is not available. " + availability_report()
        )
    return backend


def _write_cmd(backend: Int, selection: Int) raises -> String:
    _validate_selection(selection)
    if backend == BACKEND_WAYLAND:
        var cmd = String("wl-copy --type text/plain")
        if selection == SELECTION_PRIMARY:
            cmd += String(" --primary")
        return cmd^
    if backend == BACKEND_XCLIP:
        if selection == SELECTION_PRIMARY:
            return String("xclip -selection primary -in")
        return String("xclip -selection clipboard -in")
    if backend == BACKEND_XSEL:
        if selection == SELECTION_PRIMARY:
            return String("xsel --primary --input")
        return String("xsel --clipboard --input")
    raise Error("clipboard: backend " + backend_name(backend) + " is not a pipe writer")


def _read_cmd(backend: Int, selection: Int) raises -> String:
    _validate_selection(selection)
    if backend == BACKEND_WAYLAND:
        var cmd = String("wl-paste --no-newline --type text/plain")
        if selection == SELECTION_PRIMARY:
            cmd += String(" --primary")
        return cmd^
    if backend == BACKEND_XCLIP:
        if selection == SELECTION_PRIMARY:
            return String("xclip -selection primary -out")
        return String("xclip -selection clipboard -out")
    if backend == BACKEND_XSEL:
        if selection == SELECTION_PRIMARY:
            return String("xsel --primary --output")
        return String("xsel --clipboard --output")
    raise Error("clipboard: backend " + backend_name(backend) + " is not a pipe reader")


def write_text(
    text: String,
    selection: Int = SELECTION_CLIPBOARD,
    backend: Int = BACKEND_AUTO,
) raises:
    """Write UTF-8 text to the desktop clipboard.

    The payload is sent to the provider over stdin, never interpolated into a
    shell command.
    """
    var b = _resolve_backend(backend, selection)
    if b == BACKEND_OSC52:
        write_text_osc52(text, selection)
        return
    var data = _string_to_bytes(text)
    _write_cmd_bytes(_write_cmd(b, selection), data)


def read_text(
    selection: Int = SELECTION_CLIPBOARD,
    backend: Int = BACKEND_AUTO,
    max_bytes: Int = DEFAULT_MAX_READ_BYTES,
) raises -> String:
    """Read UTF-8 text from the desktop clipboard."""
    var b = _resolve_backend(backend, selection)
    if b == BACKEND_OSC52:
        raise Error("clipboard: OSC52 is write-only; read_text is unsupported")
    return _bytes_to_string(_read_cmd_bytes(_read_cmd(b, selection), max_bytes))


def clear(selection: Int = SELECTION_CLIPBOARD, backend: Int = BACKEND_AUTO) raises:
    """Clear the clipboard by making the selected backend own an empty string."""
    write_text(String(""), selection, backend)


def base64_encode(data: List[UInt8]) -> String:
    """Base64 encoder used by OSC52. Kept local to avoid HTTP package coupling."""
    var b64s = String(B64)
    var alpha = b64s.as_bytes()
    var out = List[UInt8]()
    var n = len(data)
    var i = 0
    while i + 3 <= n:
        var b0 = Int(data[i])
        var b1 = Int(data[i + 1])
        var b2 = Int(data[i + 2])
        out.append(alpha[(b0 >> 2) & 0x3F])
        out.append(alpha[((b0 & 3) << 4) | (b1 >> 4)])
        out.append(alpha[((b1 & 15) << 2) | (b2 >> 6)])
        out.append(alpha[b2 & 0x3F])
        i += 3
    var rem = n - i
    if rem == 1:
        var b0 = Int(data[i])
        out.append(alpha[(b0 >> 2) & 0x3F])
        out.append(alpha[(b0 & 3) << 4])
        out.append(UInt8(61))
        out.append(UInt8(61))
    elif rem == 2:
        var b0 = Int(data[i])
        var b1 = Int(data[i + 1])
        out.append(alpha[(b0 >> 2) & 0x3F])
        out.append(alpha[((b0 & 3) << 4) | (b1 >> 4)])
        out.append(alpha[(b1 & 15) << 2])
        out.append(UInt8(61))
    return _bytes_to_string(out)


def base64_text(text: String) -> String:
    return base64_encode(_string_to_bytes(text))


def osc52_sequence(text: String, selection: Int = SELECTION_CLIPBOARD) raises -> String:
    """Return an OSC52 escape sequence for terminals that support clipboard set.

    Clipboard selection uses target 'c'; primary selection uses target 'p'.
    """
    _validate_selection(selection)
    var target = String("c")
    if selection == SELECTION_PRIMARY:
        target = String("p")
    return (
        String(chr(0x1B)) + String("]52;") + target + String(";")
        + base64_text(text) + String(chr(0x07))
    )


def _write_fd(fd: Int32, data: String) raises:
    if fd != 1:
        raise Error(
            "clipboard: custom OSC52 fd writes are not available in this build; "
            + "use osc52_sequence() with the app's own terminal writer"
        )
    print(data)


def write_text_osc52(
    text: String,
    selection: Int = SELECTION_CLIPBOARD,
    fd: Int32 = 1,
) raises:
    """Emit an OSC52 clipboard write sequence to stdout by default.

    For exact byte control or non-stdout destinations, call osc52_sequence() and
    write the returned string through the app's terminal layer.
    """
    _write_fd(fd, osc52_sequence(text, selection))

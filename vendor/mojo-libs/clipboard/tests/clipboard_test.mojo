from clipboard.clipboard import (
    BACKEND_AUTO,
    BACKEND_OSC52,
    BACKEND_WAYLAND,
    BACKEND_XCLIP,
    BACKEND_XSEL,
    SELECTION_CLIPBOARD,
    SELECTION_PRIMARY,
    availability_report,
    backend_available,
    backend_name,
    base64_text,
    clear,
    detect_backend,
    env_nonempty,
    osc52_sequence,
    read_text,
    selection_name,
    write_text,
)


struct Tally(Movable):
    var p: Int
    var f: Int

    def __init__(out self):
        self.p = 0
        self.f = 0


def chk(mut t: Tally, cond: Bool, label: String):
    if cond:
        t.p += 1
    else:
        t.f += 1
        print("  FAIL", label)


def _read_text_or_empty() -> String:
    try:
        return read_text()
    except:
        return String("")


def main() raises:
    var t = Tally()

    chk(t, backend_name(BACKEND_AUTO) == "auto", "backend auto name")
    chk(t, backend_name(BACKEND_WAYLAND) == "wayland", "backend wayland name")
    chk(t, backend_name(BACKEND_XCLIP) == "xclip", "backend xclip name")
    chk(t, backend_name(BACKEND_XSEL) == "xsel", "backend xsel name")
    chk(t, backend_name(BACKEND_OSC52) == "osc52", "backend osc52 name")
    chk(t, selection_name(SELECTION_CLIPBOARD) == "clipboard", "clipboard selection name")
    chk(t, selection_name(SELECTION_PRIMARY) == "primary", "primary selection name")

    chk(t, base64_text(String("")) == "", "base64 empty")
    chk(t, base64_text(String("M")) == "TQ==", "base64 one byte")
    chk(t, base64_text(String("Ma")) == "TWE=", "base64 two bytes")
    chk(t, base64_text(String("Man")) == "TWFu", "base64 three bytes")

    var seq = osc52_sequence(String("hello"), SELECTION_CLIPBOARD)
    var expected = (
        String(chr(0x1B)) + String("]52;c;aGVsbG8=") + String(chr(0x07))
    )
    chk(t, seq == expected, "OSC52 clipboard sequence")

    var primary_seq = osc52_sequence(String("hi"), SELECTION_PRIMARY)
    var expected_primary = (
        String(chr(0x1B)) + String("]52;p;aGk=") + String(chr(0x07))
    )
    chk(t, primary_seq == expected_primary, "OSC52 primary sequence")

    var report = availability_report()
    chk(t, report.byte_length() > 0, "availability report nonempty")
    print(report)
    print("wayland", backend_available(BACKEND_WAYLAND),
          "xclip", backend_available(BACKEND_XCLIP),
          "xsel", backend_available(BACKEND_XSEL))

    if env_nonempty(String("CLIPBOARD_TEST_REAL")):
        var backend = detect_backend()
        if backend < 0:
            print("SKIP: no real clipboard backend available")
        else:
            print("real backend:", backend_name(backend))
            var before = _read_text_or_empty()
            var payload = String("mojo-clipboard-roundtrip-2026-06-12\nline2")
            write_text(payload)
            var got = read_text()
            chk(t, got == payload, "real clipboard round-trip")

            clear()
            var empty = read_text()
            chk(t, empty.byte_length() == 0, "real clipboard clear")
            if before.byte_length() > 0:
                write_text(before)

    print("")
    print("clipboard:", t.p, "passed,", t.f, "failed")
    if t.f != 0:
        raise Error("clipboard tests FAILED")

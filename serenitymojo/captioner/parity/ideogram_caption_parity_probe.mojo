# ideogram_caption_parity_probe.mojo — byte-exact parity gate for the pure-Mojo
# port of ai-toolkit's toolkit/ideogram_caption.py (module A).
#
# Loads the 126-case fixture set produced from ai-toolkit's OWN code (no GPU,
# no LLM) and asserts our Mojo fn(in) == out BYTE-IDENTICAL for every case.
#
# Fixture file: /home/alex/EriTrainer/trainer/parity/ideogram_caption/fixtures.json
#   top object with 5 keys, each a list of {"in": ..., "out": ...}:
#     digest_caption_string   (101 cases; in:str  out:str)
#     swap_bbox_xy_in_text    (5 cases;   in:str  out:str)
#     canon_medium            (7 cases;   in:str  out:str)
#     normalize_hex           (7 cases;   in:str  out:str|null)
#     is_ideogram_caption_str (6 cases;   in:str  out:bool)
#
# CPU-only string probe (no GPU kernels) — safe to run anytime.
#   cd /home/alex/mojodiffusion && pixi run mojo run -I . \
#     serenitymojo/captioner/parity/ideogram_caption_parity_probe.mojo

from std.memory import alloc

from serenitymojo.io.ffi import (
    sys_open,
    sys_close,
    sys_pread,
    file_size,
    BytePtr,
    O_RDONLY,
)
from serenitymojo.captioner.ideogram_caption import (
    JValue,
    parse_json,
    digest_caption_string,
    swap_bbox_xy_in_text,
    canon_medium,
    normalize_hex,
    is_ideogram_caption_str,
)

comptime FIXTURES = "/home/alex/EriTrainer/trainer/parity/ideogram_caption/fixtures.json"


def _read_file(path: String) raises -> String:
    # Length-aware read — build the String from the byte COUNT the OS returned,
    # never from NUL termination, so a NUL-bearing fixture would survive intact
    # (the original NUL-terminated reader was the structural blind spot that kept
    # the 126-case set from ever carrying a 0x00).
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error("cannot open fixtures: " + path)
    var sz = file_size(fd)
    var tmp = alloc[UInt8](sz)
    var bp = BytePtr(unsafe_from_address=Int(tmp))
    var done = 0
    while done < sz:
        var got = sys_pread(fd, bp + done, sz - done, done)
        if got <= 0:
            break
        done += got
    _ = sys_close(fd)
    var buf = List[UInt8]()
    for i in range(done):
        buf.append(tmp[i])
    tmp.free()
    return String(unsafe_from_utf8=buf^)


def main() raises:
    var raw = _read_file(FIXTURES)
    var root = parse_json(raw)

    var total = 0
    var passed = 0
    var fail_reports = List[String]()

    # ── digest_caption_string ──────────────────────────────────────────────────
    var dcs = root.get("digest_caption_string")
    for i in range(len(dcs.arr)):
        var c = dcs.arr[i].copy()
        var inp = c.get("in").s
        var exp = c.get("out").s
        var got = digest_caption_string(inp)
        total += 1
        if got == exp:
            passed += 1
        else:
            fail_reports.append(
                "digest_caption_string["
                + String(i)
                + "]\n  in : "
                + _trunc(inp)
                + "\n  exp: "
                + _trunc(exp)
                + "\n  got: "
                + _trunc(got)
            )

    # ── swap_bbox_xy_in_text ────────────────────────────────────────────────────
    var sbx = root.get("swap_bbox_xy_in_text")
    for i in range(len(sbx.arr)):
        var c = sbx.arr[i].copy()
        var inp = c.get("in").s
        var exp = c.get("out").s
        var got = swap_bbox_xy_in_text(inp)
        total += 1
        if got == exp:
            passed += 1
        else:
            fail_reports.append(
                "swap_bbox_xy_in_text["
                + String(i)
                + "]\n  in : "
                + _trunc(inp)
                + "\n  exp: "
                + _trunc(exp)
                + "\n  got: "
                + _trunc(got)
            )

    # ── canon_medium ────────────────────────────────────────────────────────────
    var cm = root.get("canon_medium")
    for i in range(len(cm.arr)):
        var c = cm.arr[i].copy()
        var inp = c.get("in").s
        var exp = c.get("out").s
        var got = canon_medium(inp)
        total += 1
        if got == exp:
            passed += 1
        else:
            fail_reports.append(
                "canon_medium["
                + String(i)
                + "]\n  in : "
                + _trunc(inp)
                + "\n  exp: "
                + _trunc(exp)
                + "\n  got: "
                + _trunc(got)
            )

    # ── normalize_hex (out is str | null) ───────────────────────────────────────
    var nh = root.get("normalize_hex")
    for i in range(len(nh.arr)):
        var c = nh.arr[i].copy()
        var inp = c.get("in").s
        var out_v = c.get("out")
        var hr = normalize_hex(inp)
        # Expected null  ⇔ ok must be False; expected string ⇔ value must match.
        var ok_case: Bool
        var exp_repr: String
        var got_repr: String
        if out_v.is_null():
            ok_case = not hr.ok
            exp_repr = String("None")
            got_repr = String("None") if not hr.ok else hr.value
        else:
            ok_case = hr.ok and (hr.value == out_v.s)
            exp_repr = out_v.s
            got_repr = hr.value if hr.ok else String("None")
        total += 1
        if ok_case:
            passed += 1
        else:
            fail_reports.append(
                "normalize_hex["
                + String(i)
                + "]\n  in : "
                + _trunc(inp)
                + "\n  exp: "
                + exp_repr
                + "\n  got: "
                + got_repr
            )

    # ── is_ideogram_caption_str (out is bool) ───────────────────────────────────
    var iic = root.get("is_ideogram_caption_str")
    for i in range(len(iic.arr)):
        var c = iic.arr[i].copy()
        var inp = c.get("in").s
        var exp_b = c.get("out").b  # JSON true/false → JValue bool
        var got_b = is_ideogram_caption_str(inp)
        total += 1
        if got_b == exp_b:
            passed += 1
        else:
            fail_reports.append(
                "is_ideogram_caption_str["
                + String(i)
                + "]\n  in : "
                + _trunc(inp)
                + "\n  exp: "
                + _bstr(exp_b)
                + "\n  got: "
                + _bstr(got_b)
            )

    # ── report ──────────────────────────────────────────────────────────────────
    print("ideogram_caption parity:", passed, "/", total, "PASS")
    if len(fail_reports) > 0:
        print("")
        print("FAILURES (", len(fail_reports), "):")
        for i in range(len(fail_reports)):
            print("----------------------------------------")
            print(fail_reports[i])
    if passed != total:
        raise Error("PARITY FAILED: " + String(passed) + "/" + String(total))
    print("ALL CASES BYTE-IDENTICAL")


def _trunc(s: String) -> String:
    # Truncate long strings for readable failure reports (length-aware build).
    var b = s.as_bytes()
    if len(b) <= 200:
        return s
    var out = List[UInt8]()
    for i in range(200):
        out.append(b[i])
    return String(unsafe_from_utf8=out^) + " …(" + String(len(b)) + "b)"


def _bstr(v: Bool) -> String:
    return String("True") if v else String("False")

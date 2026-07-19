# ideogram_caption_adv_probe.mojo — ADVERSARIAL byte-exact parity gate.
#
# Companion to ideogram_caption_parity_probe.mojo (the committed 126-case gate).
# This one exercises the high-risk surfaces the original gate was BLIND to (the
# skeptic's 5 blockers + verified-correct edges), regenerated from the REAL
# ai-toolkit Python (no GPU, no LLM):
#   - embedded NUL (0x00) and every control char 0x00-0x1f (BLOCKER 1) — carried
#     in the fixture file as \uXXXX ESCAPES, decoded to raw bytes by the parser;
#   - float reserialize 1e2→100.0 / 1.5e3→1500.0 / -0.0 (BLOCKER 2);
#   - duplicate keys last-wins / single-key / first-position (BLOCKER 3);
#   - strict-parse REJECTS → passthrough: leading-zero 01, trailing-dot 5., bare
#     .5, leading +5, bareword, single-quotes, trailing comma, RAW ctrl in string
#     (BLOCKERS 4+5);
#   - verified-correct: non-ASCII passthrough, surrogate pairs, big ints, prose.
#
# NaN/Infinity/-Infinity are EXCLUDED by design (documented deliberate divergence:
# Python minifies, the strict Mojo parser passes through — see ideogram_caption.mojo
# parse_value comment). They are not in the fixture set; this gate does not test them.
#
# Fixtures (NUL-free, control-byte-free at the file level — all special bytes are
# escapes, so this gate's own length-aware reader is not the blind spot the
# original was):
#   serenitymojo/captioner/parity/ideogram_caption_adv_fixtures.json
#
# CPU-only string probe (no GPU). FAIL-LOUD: non-zero exit on ANY mismatch.
#   cd /home/alex/mojodiffusion && pixi run mojo run -I . \
#     serenitymojo/captioner/parity/ideogram_caption_adv_probe.mojo

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

comptime FIXTURES = "/home/alex/mojodiffusion/serenitymojo/captioner/parity/ideogram_caption_adv_fixtures.json"


def _read_file(path: String) raises -> String:
    # Length-aware read — never relies on NUL termination. (The fixture file is
    # itself NUL-free since all special bytes ride as escapes, but reading by the
    # byte count returned from the OS is the correct, robust contract regardless.)
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error("cannot open adv fixtures: " + path)
    var sz = file_size(fd)
    var buf = List[UInt8]()
    for _i in range(sz):
        buf.append(0)
    var tmp = alloc[UInt8](sz)
    var bp = BytePtr(unsafe_from_address=Int(tmp))
    var done = 0
    while done < sz:
        var got = sys_pread(fd, bp + done, sz - done, done)
        if got <= 0:
            break
        done += got
    _ = sys_close(fd)
    for i in range(done):
        buf[i] = tmp[i]
    tmp.free()
    # truncate to the bytes actually read
    while len(buf) > done:
        _ = buf.pop()
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
                "digest_caption_string[" + String(i) + "]\n  in : " + _trunc(inp)
                + "\n  exp: " + _trunc(exp) + "\n  got: " + _trunc(got)
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
                "swap_bbox_xy_in_text[" + String(i) + "]\n  in : " + _trunc(inp)
                + "\n  exp: " + _trunc(exp) + "\n  got: " + _trunc(got)
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
                "canon_medium[" + String(i) + "]\n  in : " + _trunc(inp)
                + "\n  exp: " + _trunc(exp) + "\n  got: " + _trunc(got)
            )

    # ── normalize_hex (out is str | null) ───────────────────────────────────────
    var nh = root.get("normalize_hex")
    for i in range(len(nh.arr)):
        var c = nh.arr[i].copy()
        var inp = c.get("in").s
        var out_v = c.get("out")
        var hr = normalize_hex(inp)
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
                "normalize_hex[" + String(i) + "]\n  in : " + _trunc(inp)
                + "\n  exp: " + exp_repr + "\n  got: " + got_repr
            )

    # ── is_ideogram_caption_str (out is bool) ───────────────────────────────────
    var iic = root.get("is_ideogram_caption_str")
    for i in range(len(iic.arr)):
        var c = iic.arr[i].copy()
        var inp = c.get("in").s
        var exp_b = c.get("out").b
        var got_b = is_ideogram_caption_str(inp)
        total += 1
        if got_b == exp_b:
            passed += 1
        else:
            fail_reports.append(
                "is_ideogram_caption_str[" + String(i) + "]\n  in : " + _trunc(inp)
                + "\n  exp: " + _bstr(exp_b) + "\n  got: " + _bstr(got_b)
            )

    # ── report ──────────────────────────────────────────────────────────────────
    print("ideogram_caption ADVERSARIAL parity:", passed, "/", total, "PASS")
    if len(fail_reports) > 0:
        print("")
        print("FAILURES (", len(fail_reports), "):")
        for i in range(len(fail_reports)):
            print("----------------------------------------")
            print(fail_reports[i])
    if passed != total:
        raise Error("ADVERSARIAL PARITY FAILED: " + String(passed) + "/" + String(total))
    print("ALL ADVERSARIAL CASES BYTE-IDENTICAL")


def _trunc(s: String) -> String:
    var b = s.as_bytes()
    if len(b) <= 240:
        return s
    var out = List[UInt8]()
    for i in range(240):
        out.append(b[i])
    return String(unsafe_from_utf8=out^) + " …(" + String(len(b)) + "b)"


def _bstr(v: Bool) -> String:
    return String("True") if v else String("False")

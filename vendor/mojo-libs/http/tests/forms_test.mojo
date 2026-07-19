# Tests for http.cookies and http.multipart. Pure Mojo, no sockets.
# Run from the repo root:
#   pixi run --manifest-path /home/alex/rill/pixi.toml mojo run -I . http/tests/forms_test.mojo

from std.memory import UnsafePointer, alloc
from std.builtin.type_aliases import MutExternalOrigin
from std.ffi import external_call

from http.cookies import (
    Cookie, parse_cookie_header, serialize_cookie,
    SetCookie, parse_set_cookie, serialize_set_cookie,
    CookieJar,
    FormField, parse_form_urlencoded, build_form_urlencoded,
)
from http.multipart import (
    Part, parse_multipart, boundary_from_content_type,
    build_multipart_part, build_multipart_end,
)

comptime TBytePtr = UnsafePointer[UInt8, MutExternalOrigin]


struct TT(Copyable, Movable):
    var p: Int
    var f: Int
    def __init__(out self):
        self.p = 0
        self.f = 0
    def ck(mut self, cond: Bool, name: String):
        if cond:
            self.p += 1
        else:
            self.f += 1
            print("  FAIL:", name)


def _str_to_bytes(s: String) -> List[UInt8]:
    var out = List[UInt8]()
    var sp = s.as_bytes()
    for i in range(s.byte_length()):
        out.append(sp[i])
    return out^


def main() raises:
    var t = TT()

    # ---- request Cookie header --------------------------------------------
    var cks = parse_cookie_header(String("a=1; b=2; c=hello"))
    t.ck(len(cks) == 3, "cookie header -> 3 cookies")
    t.ck(cks[0].name == "a" and cks[0].value == "1", "cookie a=1")
    t.ck(cks[1].name == "b" and cks[1].value == "2", "cookie b=2")
    t.ck(cks[2].name == "c" and cks[2].value == "hello", "cookie c=hello")
    print("cookies parsed:")
    for i in range(len(cks)):
        print("  ", cks[i].name, "=", cks[i].value)
    t.ck(serialize_cookie(String("x"), String("y")) == "x=y", "serialize_cookie")

    # ---- Set-Cookie -------------------------------------------------------
    var sc = parse_set_cookie(String("sid=abc; Path=/; Max-Age=3600; HttpOnly; SameSite=Strict"))
    t.ck(sc.name == "sid" and sc.value == "abc", "set-cookie name/value")
    t.ck(sc.path == "/", "set-cookie path=/")
    t.ck(sc.max_age == 3600, "set-cookie max_age=3600")
    t.ck(sc.http_only == True, "set-cookie HttpOnly")
    t.ck(sc.secure == False, "set-cookie Secure absent")
    t.ck(sc.same_site == "Strict", "set-cookie SameSite=Strict")
    print("set-cookie parsed: name=", sc.name, " value=", sc.value,
          " path=", sc.path, " max_age=", sc.max_age,
          " http_only=", sc.http_only, " same_site=", sc.same_site)

    # also test a fuller Set-Cookie with Secure + Domain
    var sc2 = parse_set_cookie(String("t=v; Domain=example.com; Secure; SameSite=Lax; Max-Age=0"))
    t.ck(sc2.domain == "example.com", "set-cookie2 domain")
    t.ck(sc2.secure == True, "set-cookie2 Secure")
    t.ck(sc2.max_age == 0, "set-cookie2 Max-Age=0 (delete)")
    var sc2_ser = serialize_set_cookie(sc2)
    print("set-cookie2 round-trip:", sc2_ser)
    t.ck(parse_set_cookie(sc2_ser).domain == "example.com", "set-cookie2 ser/parse domain")

    # ---- CookieJar --------------------------------------------------------
    var jar = CookieJar()
    jar.set(String("session"), String("xyz"))
    jar.set(String("theme"), String("dark"))
    t.ck(jar.get(String("session")) == "xyz", "jar get session")
    t.ck(jar.get(String("theme")) == "dark", "jar get theme")
    t.ck(jar.get(String("missing")) == "", "jar get missing -> empty")
    var jar_hdr = jar.header()
    print("jar header:", jar_hdr)
    # round-trip: parse the emitted header back, ensure both cookies survive
    var jar_back = parse_cookie_header(jar_hdr)
    var found_session = False
    var found_theme = False
    for i in range(len(jar_back)):
        if jar_back[i].name == "session" and jar_back[i].value == "xyz":
            found_session = True
        if jar_back[i].name == "theme" and jar_back[i].value == "dark":
            found_theme = True
    t.ck(found_session and found_theme, "jar header round-trip")

    # ---- form-urlencoded --------------------------------------------------
    var ff = parse_form_urlencoded(String("a=1&b=two+words&c=%2F"))
    t.ck(len(ff) == 3, "form -> 3 fields")
    t.ck(ff[0].name == "a" and ff[0].value == "1", "form a=1")
    t.ck(ff[1].name == "b" and ff[1].value == "two words", "form b=two words")
    t.ck(ff[2].name == "c" and ff[2].value == "/", "form c=/")
    print("form fields:")
    for i in range(len(ff)):
        print("  ", ff[i].name, "=", ff[i].value)
    var built = build_form_urlencoded(ff)
    print("form rebuilt:", built)
    var ff2 = parse_form_urlencoded(built)
    t.ck(len(ff2) == 3 and ff2[1].value == "two words" and ff2[2].value == "/",
         "form round-trip")

    # ---- multipart --------------------------------------------------------
    var boundary = String("----WebKitFormBoundary7MA4YWxkTrZu0gW")

    # binary payload: includes 0x00 and a boundary-LIKE-but-not byte run
    # ("--" + boundary without a leading CRLF, mid-data) to stress the scanner.
    var bin = List[UInt8]()
    bin.append(0x00)
    bin.append(0xFF)
    bin.append(0x01)
    bin.append(0x00)
    bin.append(0x42)
    # boundary-like bytes embedded in the data (NOT a real delimiter: no CRLF--)
    var fake = String("--") + boundary + String("X")
    var fb = fake.as_bytes()
    for i in range(fake.byte_length()):
        bin.append(fb[i])
    bin.append(0x00)
    bin.append(0x99)

    # build a 2-part body
    var body = List[UInt8]()
    build_multipart_part(body, boundary, String("title"), String(""), String(""),
                         _str_to_bytes(String("Hi")))
    build_multipart_part(body, boundary, String("f"), String("a.bin"),
                         String("application/octet-stream"), bin)
    build_multipart_end(body, boundary)

    # boundary_from_content_type
    var ct = String("multipart/form-data; boundary=") + boundary
    t.ck(boundary_from_content_type(ct) == boundary, "boundary_from_content_type")
    t.ck(boundary_from_content_type(String("multipart/form-data; boundary=\"abc\"")) == "abc",
         "boundary_from_content_type quoted")

    # parse it back
    var got = parse_multipart(body, boundary)
    t.ck(len(got) == 2, "multipart -> 2 parts")
    t.ck(got[0].name == "title" and got[0].filename == "", "part0 name=title, no filename")
    t.ck(_bytes_eq_str(got[0].data, String("Hi")), "part0 data == 'Hi'")
    t.ck(got[1].name == "f", "part1 name=f")
    t.ck(got[1].filename == "a.bin", "part1 filename=a.bin")
    t.ck(got[1].content_type == "application/octet-stream", "part1 content_type")
    # byte-for-byte binary check (incl. the 0x00s)
    var bin_ok = len(got[1].data) == len(bin)
    if bin_ok:
        for i in range(len(bin)):
            if got[1].data[i] != bin[i]:
                bin_ok = False
                break
    t.ck(bin_ok, "part1 binary data byte-for-byte (incl 0x00 and boundary-like run)")
    print("part1 data length:", len(got[1].data), " expected:", len(bin))
    print("part1 first bytes:", Int(got[1].data[0]), Int(got[1].data[1]),
          Int(got[1].data[2]), Int(got[1].data[3]), Int(got[1].data[4]))

    # write the generated body to disk (as hex) for the Python interop cross-check
    _write_hex(String("/tmp/forms_mp_body.hex"), body)
    _write_file_str(String("/tmp/forms_mp_boundary.txt"), boundary)
    print("wrote /tmp/forms_mp_body.hex (", len(body), " body bytes ) for Python cross-check")

    # ---- reverse interop: parse a Python-BUILT multipart body ------------
    # If the Python helper has produced /tmp/py_mp_body.hex, parse it here and
    # confirm the Mojo parser reads Python's framing (incl. a 0x00 file part).
    var py_hex = _read_text(String("/tmp/py_mp_body.hex"))
    if py_hex.byte_length() > 0:
        var py_boundary = _strip_nl(_read_text(String("/tmp/py_mp_boundary.txt")))
        var py_body = _hex_to_bytes(py_hex)
        var py_parts = parse_multipart(py_body, py_boundary)
        print("reverse interop: parsed Python-built body ->", len(py_parts), "parts")
        t.ck(len(py_parts) == 2, "reverse: 2 parts from Python body")
        if len(py_parts) == 2:
            t.ck(py_parts[0].name == "greeting"
                 and _bytes_eq_str(py_parts[0].data, String("hello world")),
                 "reverse: part0 greeting=hello world")
            t.ck(py_parts[1].name == "up" and py_parts[1].filename == "x.dat",
                 "reverse: part1 name/filename")
            # expected binary: 0x00 0x10 0x00 0xAB 0xCD 0x00
            var exp = List[UInt8]()
            exp.append(0x00); exp.append(0x10); exp.append(0x00)
            exp.append(0xAB); exp.append(0xCD); exp.append(0x00)
            var rok = len(py_parts[1].data) == len(exp)
            if rok:
                for i in range(len(exp)):
                    if py_parts[1].data[i] != exp[i]:
                        rok = False
                        break
            t.ck(rok, "reverse: part1 binary (incl 0x00) byte-for-byte")
    else:
        print("reverse interop: /tmp/py_mp_body.hex not present (run Python helper first)")

    print("---")
    print("passed:", t.p, " failed:", t.f)
    if t.f == 0:
        print("ALL FORMS TESTS PASSED")


def _bytes_eq_str(b: List[UInt8], s: String) -> Bool:
    if len(b) != s.byte_length():
        return False
    var sp = s.as_bytes()
    for i in range(len(b)):
        if b[i] != sp[i]:
            return False
    return True


def _write_hex(path: String, data: List[UInt8]) raises:
    """Write the body as a hex string to `path` (text-safe, binary-faithful).
    The Python cross-check reads this and does bytes.fromhex() to reconstruct the
    exact bytes — avoids any binary text-mode encoding issues."""
    var hexc = String("0123456789abcdef")
    var hb = hexc.as_bytes()
    var out = alloc[UInt8](len(data) * 2) if len(data) > 0 else alloc[UInt8](1)
    for i in range(len(data)):
        var b = Int(data[i])
        out[i * 2] = hb[(b >> 4) & 0xF]
        out[i * 2 + 1] = hb[b & 0xF]
    var s = String(StringSlice(ptr=TBytePtr(unsafe_from_address=Int(out)), length=len(data) * 2))
    out.free()
    var f = open(path, "w")
    f.write(s)
    f.close()


def _write_file_str(path: String, s: String) raises:
    var f = open(path, "w")
    f.write(s)
    f.close()


def _read_text(path: String) -> String:
    """Read a text file; "" if it doesn't exist (so reverse interop is optional)."""
    try:
        var f = open(path, "r")
        var s = f.read()
        f.close()
        return s
    except:
        return String("")


def _strip_nl(s: String) -> String:
    return String(s.strip())


fn _hx(c: Int) -> Int:
    if c >= 48 and c <= 57:
        return c - 48
    if c >= 97 and c <= 102:
        return c - 87
    if c >= 65 and c <= 70:
        return c - 55
    return 0


def _hex_to_bytes(s: String) -> List[UInt8]:
    """Reconstruct raw bytes from a hex string (companion to _write_hex)."""
    var out = List[UInt8]()
    var t = String(s.strip())
    var sp = t.as_bytes()
    var n = t.byte_length()
    var i = 0
    while i + 1 < n:
        out.append(UInt8(_hx(Int(sp[i])) * 16 + _hx(Int(sp[i + 1]))))
        i += 2
    return out^

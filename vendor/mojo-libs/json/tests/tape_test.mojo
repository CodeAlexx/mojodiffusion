from json.tape import parse, JSONDoc

struct TT(Copyable, Movable):
    var p: Int
    var f: Int
    def __init__(out self):
        self.p = 0; self.f = 0
    def ck(mut self, cond: Bool, name: String):
        if cond: self.p += 1
        else:
            self.f += 1
            print("  FAIL:", name)

def main() raises:
    var t = TT()
    var doc = parse(String('{"name":"Mojo","ver":3,"pi":3.5,"ok":true,"nil":null,"tags":["a","b","c"],"nested":{"x":-5,"y":1e3}}'))
    t.ck(doc.get_str("name") == "Mojo", "string field")
    t.ck(doc.get_int("ver") == 3, "int field")
    t.ck(doc.get_float("pi") == 3.5, "float field")
    t.ck(doc.get_bool("ok") == True, "bool field")
    t.ck(doc.has("nil") and doc.kind("nil") == 0, "null field present")
    t.ck(doc.length("tags") == 3, "array length")
    t.ck(doc.get_str("tags.0") == "a", "array idx 0")
    t.ck(doc.get_str("tags.2") == "c", "array idx 2")
    t.ck(doc.get_int("nested.x") == -5, "nested negative int")
    t.ck(doc.get_float("nested.y") == 1000.0, "exponent float")
    t.ck(doc.has("missing") == False, "missing key absent")
    t.ck(doc.has("nested.nope") == False, "missing nested absent")

    var d2 = parse(String('"a\\"b\\nA\\u00e9\\u4e2d"'))
    var expect = String('a"b') + "\n" + "A" + "é" + "中"
    t.ck(d2.get_str("") == expect, "string escapes + unicode (root)")
    var d3 = parse(String('"\\ud83d\\ude00"'))
    t.ck(d3.get_str("").byte_length() == 4, "surrogate pair -> 4 bytes")

    var d4 = parse(String("42"))
    t.ck(d4.get_int("") == 42, "bare int root")
    var d5 = parse(String("[10,20,30]"))
    t.ck(d5.length("") == 3 and d5.get_int("1") == 20, "array root + index")
    var d6 = parse(String("  true "))
    t.ck(d6.get_bool(""), "ws + bare bool")

    # regression canary: parse() returns by move; accessing a tiny (sub-SSO)
    # doc through paths that read the source buffer (object-key compare, string
    # unescape) must still work — i.e. the source pointer is derived live, not a
    # cached pointer that dangles when the doc is moved.
    var tiny = parse(String('{"a":1}'))
    t.ck(tiny.get_int("a") == 1, "tiny obj int after move (no dangling src ptr)")
    var tiny2 = parse(String('{"k":"v"}'))
    t.ck(tiny2.get_str("k") == "v", "tiny obj str after move")
    var tiny3 = parse(String('{"x":"y"}'))
    var tiny3b = tiny3^  # force a second explicit move, then read
    t.ck(tiny3b.get_str("x") == "y", "tiny obj str after extra explicit move")

    # integer-overflow guard: a value beyond Int64 must not silently wrap. It is
    # promoted to a float (JS-like) so get_float is right and kind() is honest;
    # values that DO fit Int64 (incl. the max) stay exact integers.
    var ov = parse(String('{"big":123456789012345678901234567890}'))
    t.ck(ov.kind("big") == 3, "oversized int promoted to float (no silent wrap)")
    t.ck(ov.get_float("big") > 1.0e29 and ov.get_float("big") < 2.0e29, "oversized int reads as ~1.2e29")
    var mx = parse(String('{"m":9223372036854775807,"n":42}'))
    t.ck(mx.kind("m") == 2 and mx.get_int("m") == 9223372036854775807, "max Int64 stays exact int")
    t.ck(mx.kind("n") == 2 and mx.get_int("n") == 42, "small int unaffected")

    # correctly-rounded float fast path: for the common range the parser must
    # match the compiler's own correctly-rounded literal (bit-exact), not the
    # last-ULP-off naive result. (Verified separately bit-identical to Python's
    # strtod for 0.1/0.2/0.3 etc.)
    var fr = parse(String('{"a":0.1,"b":0.3,"c":3.141592653589793,"d":123.456,"e":1.5e-3,"f":-273.15}'))
    t.ck(fr.get_float("a") == 0.1, "0.1 correctly rounded")
    t.ck(fr.get_float("b") == 0.3, "0.3 correctly rounded")
    t.ck(fr.get_float("c") == 3.141592653589793, "pi (16 sig digits) correctly rounded")
    t.ck(fr.get_float("d") == 123.456, "123.456 correctly rounded")
    t.ck(fr.get_float("e") == 1.5e-3, "1.5e-3 correctly rounded")
    t.ck(fr.get_float("f") == -273.15, "negative correctly rounded")

    print("---")
    print("passed:", t.p, " failed:", t.f)
    if t.f == 0:
        print("ALL TAPE TESTS PASSED")

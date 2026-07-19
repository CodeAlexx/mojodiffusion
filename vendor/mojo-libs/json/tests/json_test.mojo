from json.parser import loads
from json.serialize import dumps, dumps_pretty
from json.value import JSONValue

struct T(Copyable, Movable):
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

def main() raises:
    var t = T()
    var doc = String('{"name":"Mojo","ver":3,"pi":3.14,"ok":true,"nil":null,"tags":["a","b","c"],"nested":{"x":-5,"y":1e3}}')
    var v = loads(doc)
    t.ck(v.is_object(), "root is object")
    t.ck(v["name"].as_string() == "Mojo", "string field")
    t.ck(v["ver"].as_int() == 3, "int field")
    t.ck(v["pi"].as_float() == 3.14, "float field")
    t.ck(v["ok"].as_bool() == True, "bool field")
    t.ck(v["nil"].is_null(), "null field")
    t.ck(v["tags"].is_array() and v["tags"].length() == 3, "array length")
    t.ck(v["tags"][1].as_string() == "b", "array index")
    t.ck(v["nested"]["x"].as_int() == -5, "nested negative int")
    t.ck(v["nested"]["y"].as_float() == 1000.0, "exponent float")
    t.ck(v["missing"].is_null(), "missing key -> null")

    var s = loads(String('"a\\"b\\\\c\\n\\u0041\\u00e9\\u4e2d"'))
    t.ck(s.is_string(), "escape: is string")
    var expect = String('a"b\\c') + "\n" + "A" + "é" + "中"
    t.ck(s.as_string() == expect, "escapes + unicode")

    var emoji = loads(String('"\\ud83d\\ude00"'))
    t.ck(emoji.as_string().byte_length() == 4, "surrogate pair -> 4 UTF-8 bytes")

    t.ck(loads(String("42")).as_int() == 42, "bare int")
    t.ck(loads(String("-7")).as_int() == -7, "negative int")
    t.ck(loads(String("2.5")).as_float() == 2.5, "bare float")
    t.ck(loads(String("[]")).length() == 0, "empty array")
    t.ck(loads(String("{}")).length() == 0, "empty object")
    t.ck(loads(String("  true  ")).as_bool() == True, "whitespace around")

    var rt = loads(dumps(v))
    t.ck(rt["nested"]["x"].as_int() == -5, "round-trip nested int")
    t.ck(rt["tags"][2].as_string() == "c", "round-trip array")

    var esc = JSONValue.from_string(String("tab\there\"quote"))
    t.ck(dumps(esc) == String('"tab\\there\\"quote"'), "serialize escapes")

    var got_err = False
    try:
        _ = loads(String('{"a":}'))
    except:
        got_err = True
    t.ck(got_err, "malformed value raises")
    got_err = False
    try:
        _ = loads(String('{"a":1'))
    except:
        got_err = True
    t.ck(got_err, "unterminated object raises")
    got_err = False
    try:
        _ = loads(String('[1,2] garbage'))
    except:
        got_err = True
    t.ck(got_err, "trailing data raises")

    print("---")
    print("passed:", t.p, " failed:", t.f)
    if t.f == 0:
        print("ALL JSON TESTS PASSED")

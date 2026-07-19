# Tests for json.pointer (RFC 6901), json.patch merge (RFC 7386) and
# json.patch JSON Patch (RFC 6902). Expected strings are the verbatim outputs
# of a reference Python implementation of each RFC (see commit notes).

from json.parser import loads
from json.serialize import dumps
from json.value import JSONValue
from json.pointer import resolve, split_pointer
from json.patch import apply_merge_patch, apply_patch


struct T(Copyable, Movable):
    var p: Int
    var f: Int

    def __init__(out self):
        self.p = 0
        self.f = 0

    def ck(mut self, cond: Bool, name: String):
        if cond:
            self.p += 1
            print("  PASS:", name)
        else:
            self.f += 1
            print("  FAIL:", name)


def ck_resolve(mut t: T, doc: JSONValue, ptr: String, expect_json: String) raises:
    var got = dumps(resolve(doc, ptr))
    t.ck(got == expect_json, "resolve " + repr_ptr(ptr) + " => " + got + " (want " + expect_json + ")")


def ck_resolve_missing(mut t: T, doc: JSONValue, ptr: String) raises:
    var raised = False
    try:
        _ = resolve(doc, ptr)
    except:
        raised = True
    t.ck(raised, "resolve missing raises: " + repr_ptr(ptr))


def repr_ptr(ptr: String) -> String:
    return "'" + ptr + "'"


def ck_merge(mut t: T, name: String, target: String, patch: String, expect: String) raises:
    var got = dumps(apply_merge_patch(loads(target), loads(patch)))
    t.ck(got == expect, "merge " + name + " => " + got + " (want " + expect + ")")


def ck_patch(mut t: T, name: String, doc: String, ops: String, expect: String) raises:
    var got = dumps(apply_patch(loads(doc), loads(ops)))
    t.ck(got == expect, "patch " + name + " => " + got + " (want " + expect + ")")


def ck_patch_fails(mut t: T, name: String, doc: String, ops: String) raises:
    var raised = False
    try:
        _ = apply_patch(loads(doc), loads(ops))
    except:
        raised = True
    t.ck(raised, "patch " + name + " raises (correct)")


def main() raises:
    var t = T()

    # ───────────────────────────── RFC 6901 ─────────────────────────────
    print("=== RFC 6901 JSON Pointer ===")
    var doc6901 = loads(String(
        '{"foo":["bar","baz"],"":0,"a/b":1,"c%d":2,"e^f":3,"g|h":4,'
        + '"i\\\\j":5,"k\\"l":6," ":7,"m~n":8}'
    ))
    # whole document
    ck_resolve(t, doc6901, "",
        '{"foo":["bar","baz"],"":0,"a/b":1,"c%d":2,"e^f":3,"g|h":4,"i\\\\j":5,"k\\"l":6," ":7,"m~n":8}')
    ck_resolve(t, doc6901, "/foo", '["bar","baz"]')
    ck_resolve(t, doc6901, "/foo/0", '"bar"')
    ck_resolve(t, doc6901, "/", "0")          # key "" -> 0
    ck_resolve(t, doc6901, "/a~1b", "1")      # ~1 -> "/"
    ck_resolve(t, doc6901, "/c%d", "2")
    ck_resolve(t, doc6901, "/e^f", "3")
    ck_resolve(t, doc6901, "/g|h", "4")
    ck_resolve(t, doc6901, "/i\\j", "5")
    ck_resolve(t, doc6901, "/k\"l", "6")
    ck_resolve(t, doc6901, "/ ", "7")
    ck_resolve(t, doc6901, "/m~0n", "8")      # ~0 -> "~"
    # split_pointer escaping
    var toks = split_pointer("/a~1b/m~0n")
    t.ck(len(toks) == 2 and toks[0] == "a/b" and toks[1] == "m~n", "split_pointer unescapes ~1/~0")
    var toks0 = split_pointer("")
    t.ck(len(toks0) == 0, "split_pointer empty -> []")
    # missing targets
    ck_resolve_missing(t, doc6901, "/nope")
    ck_resolve_missing(t, doc6901, "/foo/5")
    ck_resolve_missing(t, doc6901, "/foo/-")

    # ───────────────────────────── RFC 7386 ─────────────────────────────
    print("=== RFC 7386 JSON Merge Patch ===")
    ck_merge(t, "replace member", '{"a":"b"}', '{"a":"c"}', '{"a":"c"}')
    ck_merge(t, "add member", '{"a":"b"}', '{"b":"c"}', '{"a":"b","b":"c"}')
    ck_merge(t, "null deletes", '{"a":"b"}', '{"a":null}', '{}')
    ck_merge(t, "null deletes one of two", '{"a":"b","b":"c"}', '{"a":null}', '{"b":"c"}')
    ck_merge(t, "replace array w/ scalar", '{"a":["b"]}', '{"a":"c"}', '{"a":"c"}')
    ck_merge(t, "replace scalar w/ array", '{"a":"c"}', '{"a":["b"]}', '{"a":["b"]}')
    ck_merge(t, "nested merge + delete", '{"a":{"b":"c"}}', '{"a":{"b":"d","c":null}}', '{"a":{"b":"d"}}')
    ck_merge(t, "array of obj replaced", '{"a":[{"b":"c"}]}', '{"a":[1]}', '{"a":[1]}')
    ck_merge(t, "array replaces array", '["a","b"]', '["c","d"]', '["c","d"]')
    ck_merge(t, "non-obj patch replaces", '{"a":"b"}', '["c"]', '["c"]')
    ck_merge(t, "null patch replaces", '{"a":"foo"}', "null", "null")
    ck_merge(t, "string patch replaces", '{"a":"foo"}', '"bar"', '"bar"')
    ck_merge(t, "null member kept if patch adds other", '{"e":null}', '{"a":1}', '{"e":null,"a":1}')
    ck_merge(t, "array target becomes obj", "[1,2]", '{"a":"b","c":null}', '{"a":"b"}')
    ck_merge(t, "deep null creates empty", "{}", '{"a":{"bb":{"ccc":null}}}', '{"a":{"bb":{}}}')

    # ───────────────────────────── RFC 6902 ─────────────────────────────
    print("=== RFC 6902 JSON Patch (Appendix A) ===")
    ck_patch(t, "A1 add member", '{"foo":"bar"}',
        '[{"op":"add","path":"/baz","value":"qux"}]', '{"foo":"bar","baz":"qux"}')
    ck_patch(t, "A2 add into array", '{"foo":["bar","baz"]}',
        '[{"op":"add","path":"/foo/1","value":"qux"}]', '{"foo":["bar","qux","baz"]}')
    ck_patch(t, "A3 remove member", '{"baz":"qux","foo":"bar"}',
        '[{"op":"remove","path":"/baz"}]', '{"foo":"bar"}')
    ck_patch(t, "A4 remove array elt", '{"foo":["bar","qux","baz"]}',
        '[{"op":"remove","path":"/foo/1"}]', '{"foo":["bar","baz"]}')
    ck_patch(t, "A5 replace", '{"baz":"qux","foo":"bar"}',
        '[{"op":"replace","path":"/baz","value":"boo"}]', '{"baz":"boo","foo":"bar"}')
    ck_patch(t, "A6 move", '{"foo":{"bar":"baz","waldo":"fred"},"qux":{"corge":"grault"}}',
        '[{"op":"move","from":"/foo/waldo","path":"/qux/thud"}]',
        '{"foo":{"bar":"baz"},"qux":{"corge":"grault","thud":"fred"}}')
    ck_patch(t, "A7 move within array", '{"foo":["all","grass","cows","eat"]}',
        '[{"op":"move","from":"/foo/1","path":"/foo/3"}]',
        '{"foo":["all","cows","eat","grass"]}')
    ck_patch(t, "A8 test ok", '{"baz":"qux","foo":["a",2,"c"]}',
        '[{"op":"test","path":"/baz","value":"qux"},{"op":"test","path":"/foo/1","value":2}]',
        '{"baz":"qux","foo":["a",2,"c"]}')
    ck_patch_fails(t, "A9 test fail", '{"baz":"qux"}',
        '[{"op":"test","path":"/baz","value":"bar"}]')
    ck_patch(t, "A10 add nested obj", '{"foo":"bar"}',
        '[{"op":"add","path":"/child","value":{"grandchild":{}}}]',
        '{"foo":"bar","child":{"grandchild":{}}}')
    ck_patch(t, "A14 escaped ~1 test", '{"/":9,"~1":10}',
        '[{"op":"test","path":"/~01","value":10}]', '{"/":9,"~1":10}')
    ck_patch(t, "A16 add to end with -", '{"foo":["bar"]}',
        '[{"op":"add","path":"/foo/-","value":["abc","def"]}]',
        '{"foo":["bar",["abc","def"]]}')
    ck_patch(t, "copy", '{"foo":{"bar":"baz"},"qux":{}}',
        '[{"op":"copy","from":"/foo/bar","path":"/qux/thud"}]',
        '{"foo":{"bar":"baz"},"qux":{"thud":"baz"}}')
    # extra: replace at root, remove failure on missing
    ck_patch(t, "replace root", '{"a":1}', '[{"op":"replace","path":"","value":[1,2]}]', "[1,2]")
    ck_patch_fails(t, "remove missing key", '{"a":1}', '[{"op":"remove","path":"/b"}]')
    ck_patch_fails(t, "add into missing parent", '{"a":1}',
        '[{"op":"add","path":"/b/c","value":1}]')

    print("---")
    print("passed:", t.p, " failed:", t.f)
    if t.f == 0:
        print("ALL POINTER/PATCH TESTS PASSED")

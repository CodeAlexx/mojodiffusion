# Tests for json.stream (pull/SAX parser) and json.ndjson (JSON Lines).
# Run:
#   pixi run --manifest-path /home/alex/rill/pixi.toml mojo run -I . json/tests/stream_test.mojo

from json.stream import (
    StreamParser, StreamEvent, stream, event_name,
    EV_EOF, EV_START_OBJECT, EV_END_OBJECT, EV_START_ARRAY, EV_END_ARRAY,
    EV_KEY, EV_STRING, EV_INT, EV_FLOAT, EV_BOOL, EV_NULL,
)
from json.ndjson import parse_ndjson, write_ndjson, NdjsonReader, reader
from json.value import JSONValue
from json.parser import loads
from json.serialize import dumps


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


def _ev_to_str(ev: StreamEvent) -> String:
    # render an event as "kind[:payload]" for printing/comparison
    var k = event_name(ev.kind)
    if ev.kind == EV_KEY or ev.kind == EV_STRING:
        return k + ":" + ev.str_val
    if ev.kind == EV_INT:
        return k + ":" + String(ev.int_val)
    if ev.kind == EV_FLOAT:
        return k + ":" + String(ev.float_val)
    if ev.kind == EV_BOOL:
        return k + ":" + ("true" if ev.bool_val else "false")
    return k


def main() raises:
    var t = TT()

    # ── 1. pull-parse a mixed document, print + check the event sequence ───────
    print("=== Test 1: event sequence ===")
    var doc = String('{"a":1,"b":[true,null,"x"],"c":{"d":2.5}}')
    var sp = stream(doc)
    var seq = List[String]()
    # capture path at the leaves we care about
    var path_at_int_1 = String("")     # value of "a"
    var path_at_str_x = String("")     # the "x" element in b
    var path_at_float = String("")     # value of "c"."d"
    while True:
        var ev = sp.next_event()
        if ev.kind == EV_EOF:
            seq.append(String("eof"))
            break
        seq.append(_ev_to_str(ev))
        if ev.kind == EV_INT and ev.int_val == 1:
            path_at_int_1 = sp.current_path()
        if ev.kind == EV_STRING and ev.str_val == "x":
            path_at_str_x = sp.current_path()
        if ev.kind == EV_FLOAT:
            path_at_float = sp.current_path()

    # print full sequence verbatim
    var line = String("")
    for i in range(len(seq)):
        if i > 0:
            line += " "
        line += seq[i]
    print("events:", line)
    print("path(a=1):", path_at_int_1)
    print("path(b[2]='x'):", path_at_str_x)
    print("path(c.d=2.5):", path_at_float)

    # expected ordered events
    var expect = List[String]()
    expect.append(String("start_object"))
    expect.append(String("key:a"))
    expect.append(String("int:1"))
    expect.append(String("key:b"))
    expect.append(String("start_array"))
    expect.append(String("bool:true"))
    expect.append(String("null"))
    expect.append(String("string:x"))
    expect.append(String("end_array"))
    expect.append(String("key:c"))
    expect.append(String("start_object"))
    expect.append(String("key:d"))
    expect.append(String("float:2.5"))
    expect.append(String("end_object"))
    expect.append(String("end_object"))
    expect.append(String("eof"))

    var seq_ok = len(seq) == len(expect)
    if seq_ok:
        for i in range(len(seq)):
            if seq[i] != expect[i]:
                seq_ok = False
                print("  mismatch at", i, ":", seq[i], "!=", expect[i])
    t.ck(seq_ok, "event sequence matches expected")
    t.ck(path_at_int_1 == "/a", "current_path at a=1 is /a")
    t.ck(path_at_str_x == "/b/2", "current_path at b[2]='x' is /b/2")
    t.ck(path_at_float == "/c/d", "current_path at c.d=2.5 is /c/d")

    # ── 2. LARGE synthetic doc — array of 100_000 small objects ────────────────
    print()
    print("=== Test 2: large doc (streaming, no tree) ===")
    var nbig = 100000
    var big = String("[")
    for i in range(nbig):
        if i > 0:
            big += ","
        # small object: {"id":<i>,"v":true}
        big += '{"id":'
        big += String(i)
        big += ',"v":true}'
    big += "]"
    print("input bytes:", big.byte_length())

    var sp2 = stream(big)
    var n_start_obj = 0
    var n_end_obj = 0
    var n_int = 0
    var n_bool = 0
    var max_depth_seen = 0
    var depth = 0
    while True:
        var ev = sp2.next_event()
        if ev.kind == EV_EOF:
            break
        if ev.kind == EV_START_OBJECT or ev.kind == EV_START_ARRAY:
            depth += 1
            if depth > max_depth_seen:
                max_depth_seen = depth
        if ev.kind == EV_END_OBJECT or ev.kind == EV_END_ARRAY:
            depth -= 1
        if ev.kind == EV_START_OBJECT:
            n_start_obj += 1
        if ev.kind == EV_END_OBJECT:
            n_end_obj += 1
        if ev.kind == EV_INT:
            n_int += 1
        if ev.kind == EV_BOOL:
            n_bool += 1
    print("element objects:", n_start_obj, "ints:", n_int, "bools:", n_bool)
    print("max nesting depth seen:", max_depth_seen)
    t.ck(n_start_obj == nbig, "large: 100000 start_object events")
    t.ck(n_end_obj == nbig, "large: 100000 end_object events")
    t.ck(n_int == nbig, "large: 100000 int values")
    t.ck(n_bool == nbig, "large: 100000 bool values")
    t.ck(max_depth_seen == 2, "large: peak nesting depth bounded at 2 (array+obj)")

    # ── 3. depth limit: 600-deep nested array must RAISE ───────────────────────
    print()
    print("=== Test 3: depth limit ===")
    var ddeep = 600
    var deep = String("")
    for _ in range(ddeep):
        deep += "["
    for _ in range(ddeep):
        deep += "]"
    var raised = False
    try:
        var sp3 = stream(deep)
        while True:
            var ev = sp3.next_event()
            if ev.kind == EV_EOF:
                break
    except e:
        raised = True
        print("raised as expected:", String(e))
    t.ck(raised, "depth 600 raises (does not overflow)")

    # ── 4. NDJSON round-trip + blank-line tolerance + python cross-check ───────
    print()
    print("=== Test 4: NDJSON ===")
    var vs = List[JSONValue]()
    vs.append(loads(String('{"id":1,"name":"a"}')))
    vs.append(loads(String('[1,2,3]')))
    vs.append(loads(String('"hello"')))
    vs.append(loads(String('42')))
    vs.append(loads(String('{"nested":{"x":true,"y":null}}')))

    var nd = write_ndjson(vs)
    print("write_ndjson output:")
    print(nd)

    var back = parse_ndjson(nd)
    var rt_ok = len(back) == len(vs)
    if rt_ok:
        for i in range(len(vs)):
            if dumps(back[i]) != dumps(vs[i]):
                rt_ok = False
                print("  rt mismatch", i, ":", dumps(back[i]), "!=", dumps(vs[i]))
    t.ck(rt_ok, "NDJSON round-trip parse(write(vs)) == vs (via dumps)")

    # blank-line tolerance
    var withblanks = String('\n{"a":1}\n\n   \n[2,3]\n\n')
    var bb = parse_ndjson(withblanks)
    print("blank-tolerant parsed count:", len(bb))
    t.ck(len(bb) == 2, "blank lines skipped (2 values from padded input)")
    t.ck(dumps(bb[0]) == '{"a":1}', "blank-tolerant value 0 correct")
    t.ck(dumps(bb[1]) == "[2,3]", "blank-tolerant value 1 correct")

    # streaming NdjsonReader yields one value at a time
    var rd = reader(nd)
    var rcount = 0
    var stream_ok = True
    while rd.has_next():
        var v = rd.next_value()
        if dumps(v) != dumps(vs[rcount]):
            stream_ok = False
        rcount += 1
    print("NdjsonReader yielded:", rcount)
    t.ck(rcount == len(vs) and stream_ok, "NdjsonReader yields all values in order")

    # cross-check one line against the known Python json.loads canonical form.
    # python: json.dumps(json.loads('{"id":1,"name":"a"}'), separators=(",",":"))
    #         -> '{"id":1,"name":"a"}'
    t.ck(dumps(vs[0]) == '{"id":1,"name":"a"}', "line 0 matches python json.loads canonical")

    print()
    print("---")
    print("passed:", t.p, " failed:", t.f)
    if t.f == 0:
        print("ALL STREAM/NDJSON TESTS PASSED")

from json.parser import loads
from json.serialize import dumps
from json.value import JSONValue
from json.codec import (
    validate_required, reject_unknown, field_list,
    req_int, req_str, req_bool, req_float, opt_str,
)

# mixed model: id/name/active required, role optional
struct User(Copyable, Movable):
    var id: Int
    var name: String
    var active: Bool
    var role: String
    def __init__(out self):
        self.id = 0
        self.name = String("")
        self.active = False
        self.role = String("user")

def parse_user(text: String) raises -> User:
    var obj = loads(text)
    if not obj.is_object():
        raise Error("expected a JSON object")
    var u = User()
    u.id = req_int(obj, "id")            # raises if missing/not-number
    u.name = req_str(obj, "name")
    u.active = req_bool(obj, "active")
    u.role = opt_str(obj, "role", String("user"))   # optional, defaulted
    return u^

def user_to_json(u: User) raises -> String:
    var o = JSONValue.new_object()
    o.set("id", JSONValue.from_int(u.id))
    o.set("name", JSONValue.from_string(u.name))
    o.set("active", JSONValue.from_bool(u.active))
    o.set("role", JSONValue.from_string(u.role))
    return dumps(o)

# strict model: ALL fields required -> validate_required[Point] (reflection)
struct Point(Copyable, Movable):
    var x: Int
    var y: Int
    def __init__(out self):
        self.x = 0
        self.y = 0

struct TT(Copyable, Movable):
    var p: Int
    var f: Int
    def __init__(out self):
        self.p = 0
        self.f = 0
    def ck(mut self, cond: Bool, name: String):
        if cond: self.p += 1
        else:
            self.f += 1
            print("  FAIL:", name)

def main() raises:
    var t = TT()
    print("User fields (reflection):", field_list[User]())
    t.ck(field_list[User]() == '["id","name","active","role"]', "field_list from reflection")

    var u = parse_user(String('{"id":7,"name":"alice","active":true,"role":"admin"}'))
    t.ck(u.id == 7 and u.name == "alice" and u.active and u.role == "admin", "valid typed parse")

    var u2 = parse_user(String('{"id":1,"name":"bob","active":false}'))
    t.ck(u2.role == "user", "optional field defaulted")

    var e1 = False
    try: _ = parse_user(String('{"id":1,"active":true}'))
    except e: e1 = True
    t.ck(e1, "missing required field rejected")

    var e2 = False
    try: _ = parse_user(String('{"id":"x","name":"a","active":true}'))
    except e: e2 = True
    t.ck(e2, "type mismatch (id) rejected")

    var e3 = False
    try: _ = parse_user(String('{"id":1,"name":2,"active":true}'))
    except e: e3 = True
    t.ck(e3, "type mismatch (name) rejected")

    var e4 = False
    try: reject_unknown[User](loads(String('{"id":1,"name":"x","active":true,"hacker":1}')))
    except e: e4 = True
    t.ck(e4, "unknown field rejected (reflection)")

    # validate_required on a STRICT all-required model
    validate_required[Point](loads(String('{"x":1,"y":2}')))
    t.ck(True, "validate_required passes when all present")
    var e5 = False
    try: validate_required[Point](loads(String('{"x":1}')))
    except e: e5 = True
    t.ck(e5, "validate_required rejects missing (reflection)")

    var j = user_to_json(u)
    var back = parse_user(j)
    t.ck(back.id == 7 and back.name == "alice" and back.role == "admin", "round-trip model<->json")

    print("---")
    print("passed:", t.p, " failed:", t.f)
    if t.f == 0:
        print("ALL CODEC TESTS PASSED")

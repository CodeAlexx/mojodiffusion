from config.value import ConfigValue, ConfigTree

def check(mut p: Int, mut f: Int, cond: Bool, name: String):
    if cond: p += 1
    else:
        f += 1; print("  FAIL:", name)

def main() raises:
    var p = 0
    var f = 0
    # coercions
    check(p, f, ConfigValue.str_("42").as_int() == 42, "str->int")
    check(p, f, ConfigValue.str_("3.5").as_float() == 3.5, "str->float")
    check(p, f, ConfigValue.str_("true").as_bool() == True, "str->bool true")
    check(p, f, ConfigValue.str_("off").as_bool() == False, "str->bool off")
    check(p, f, ConfigValue.int_(7).as_str() == "7", "int->str")
    check(p, f, ConfigValue.int_(-13).as_int() == -13, "neg int")
    check(p, f, ConfigValue.float_(2.5).as_float() == 2.5, "float")
    var arr = List[ConfigValue](); arr.append(ConfigValue.int_(1)); arr.append(ConfigValue.int_(2))
    var av = ConfigValue.array_(arr^)
    check(p, f, len(av.as_array()) == 2 and av.as_array()[1].as_int() == 2, "array")
    var raised = False
    try: _ = ConfigValue.str_("abc").as_int()
    except: raised = True
    check(p, f, raised, "bad int coercion raises")

    # tree set/get/has/sections/keys
    var t = ConfigTree()
    t.set("server", "host", ConfigValue.str_("localhost"))
    t.set("server", "port", ConfigValue.int_(8080))
    t.set("", "name", ConfigValue.str_("app"))
    check(p, f, t.get("server", "host").as_str() == "localhost", "tree get host")
    check(p, f, t.get("server", "port").as_int() == 8080, "tree get port")
    check(p, f, t.has("server", "host") and not t.has("server", "nope"), "tree has")
    check(p, f, len(t.keys("server")) == 2, "tree keys count")
    var miss = False
    try: _ = t.get("x", "y")
    except: miss = True
    check(p, f, miss, "missing get raises")

    # merge_over (other wins)
    var o = ConfigTree()
    o.set("server", "port", ConfigValue.int_(9090))
    o.set("server", "tls", ConfigValue.bool_(True))
    t.merge_over(o)
    check(p, f, t.get("server", "port").as_int() == 9090, "merge overrides port")
    check(p, f, t.get("server", "host").as_str() == "localhost", "merge keeps host")
    check(p, f, t.get("server", "tls").as_bool() == True, "merge adds tls")

    print("passed:", p, " failed:", f)
    if f == 0: print("ALL VALUE TESTS PASSED")

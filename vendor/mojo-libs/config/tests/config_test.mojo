from config.config import Config
from config.value import ConfigTree, ConfigValue
from std.collections import Dict

def check(mut p: Int, mut f: Int, cond: Bool, name: String):
    if cond:
        p += 1
    else:
        f += 1
        print("  FAIL:", name)


def main() raises:
    var p = 0
    var f = 0

    # ── 1) TOML typed gets (cross-checked vs Python tomllib) ──────────────────
    var toml_text = String(
        'title = "MojoApp"\n'
        + "\n"
        + "[server]\n"
        + 'host = "0.0.0.0"\n'
        + "port = 8080\n"
        + "timeout = 1.5\n"
        + "debug = true\n"
        + 'tags = ["a", "b", "c"]\n'
        + "ints = [1, 2, 3]\n"
        + "\n"
        + "[db]\n"
        + 'name = "main"\n'
        + "replicas = 3\n"
    )
    var ct = Config.from_toml(toml_text)
    check(p, f, ct.get_str("", "title") == "MojoApp", "toml title (global section)")
    check(p, f, ct.get_str("server", "host") == "0.0.0.0", "toml get_str host")
    check(p, f, ct.get_int("server", "port") == 8080, "toml get_int port==8080")
    check(p, f, ct.get_float("server", "timeout") == 1.5, "toml get_float timeout==1.5")
    check(p, f, ct.get_bool("server", "debug") == True, "toml get_bool debug==true")
    check(p, f, ct.get_int("db", "replicas") == 3, "toml get_int db.replicas==3")
    var tags = ct.get_list("server", "tags")
    check(p, f, len(tags) == 3 and tags[0] == "a" and tags[2] == "c", "toml get_list tags")
    var ints = ct.get_list("server", "ints")
    check(p, f, len(ints) == 3 and ints[1] == "2", "toml get_list ints (coerced to str)")

    # ── 2) INI: get_str works; get_int coerces a numeric string ───────────────
    var ini_text = String(
        "[server]\n"
        + "host = 127.0.0.1\n"
        + "port = 9090\n"
        + "debug = false\n"
    )
    var ci = Config.from_ini(ini_text)
    check(p, f, ci.get_str("server", "host") == "127.0.0.1", "ini get_str host")
    check(p, f, ci.get_int("server", "port") == 9090, "ini get_int coerces '9090'")
    check(p, f, ci.get_bool("server", "debug") == False, "ini get_bool coerces 'false'")

    # ── 3) *_or defaults, bad coercion raises, dotted path ────────────────────
    check(p, f, ct.get_str_or("server", "missing", "DFLT") == "DFLT", "get_str_or missing -> default")
    check(p, f, ct.get_int_or("server", "missing", 42) == 42, "get_int_or missing -> default")
    check(p, f, ct.get_int("server", "port") == 8080, "get_int present (sanity for _or)")
    # bad coercion: host="0.0.0.0" cannot be an int
    var raised = False
    try:
        _ = ct.get_int("server", "host")
    except:
        raised = True
    check(p, f, raised, "bad int coercion raises")
    # dotted path
    check(p, f, ct.get_int_path("server.port") == 8080, "dotted path get_int_path")
    check(p, f, ct.get_str_path("server.host") == "0.0.0.0", "dotted path get_str_path")
    check(p, f, ct.get_float_path("server.timeout") == 1.5, "dotted path get_float_path")

    # ── 4) .env parsing into section "env" ────────────────────────────────────
    var env_path = String("/tmp/cfgtest/.env")
    var ce = Config()
    ce.load_dotenv(env_path)
    check(p, f, ce.get_str("env", "TOKEN") == "abc123", "dotenv export TOKEN=abc123")
    check(p, f, ce.get_str("env", "NAME") == "quoted name", 'dotenv NAME="quoted name" (quotes stripped)')
    check(p, f, ce.get_str("env", "PLAIN") == "unquoted", "dotenv PLAIN = unquoted (spaces trimmed)")
    check(p, f, ce.get_str("env", "EMPTYISH") == "", "dotenv empty value")
    check(p, f, not ce.has("env", "# a comment line"), "dotenv '#' comment ignored")

    # ── 5) overlay_env_dict ───────────────────────────────────────────────────
    var od = Config()
    od.set("server", "port", "1000")
    var envd = Dict[String, String]()
    envd["APP_SERVER_PORT"] = String("2000")
    od.overlay_env_dict("APP", envd)
    check(p, f, od.get_int("server", "port") == 2000, "overlay_env_dict APP_SERVER_PORT -> 2000")

    # ══ HEADLINE: layered precedence (later wins) ═════════════════════════════
    # Layers all target [server] port; assert the value after EACH layer.
    print("")
    print("=== PRECEDENCE TEST (defaults -> file -> .env/env -> set) ===")
    var cfg = Config()

    # Layer 1: defaults
    cfg.set("server", "port", "1")  # baseline default via explicit-style seed
    var v1 = cfg.get_int("server", "port")
    print("  after defaults        : server.port =", v1)
    check(p, f, v1 == 1, "L1 defaults port==1")

    # Layer 2: file overrides port
    cfg.merge_toml(String("[server]\nport = 2\n"))
    var v2 = cfg.get_int("server", "port")
    print("  after file merge      : server.port =", v2)
    check(p, f, v2 == 2, "L2 file overrides port==2")

    # Layer 3: process env (injected dict) overrides port
    var envd2 = Dict[String, String]()
    envd2["MYAPP_SERVER_PORT"] = String("3")
    cfg.overlay_env_dict("MYAPP", envd2)
    var v3 = cfg.get_int("server", "port")
    print("  after env overlay     : server.port =", v3)
    check(p, f, v3 == 3, "L3 env overrides port==3")

    # Layer 4: explicit set() — highest precedence
    cfg.set("server", "port", "4")
    var v4 = cfg.get_int("server", "port")
    print("  after explicit set()  : server.port =", v4)
    check(p, f, v4 == 4, "L4 explicit set wins port==4")
    print("============================================================")
    print("")

    # Also prove a SEPARATE key set only at defaults survives all layers.
    var cfg2 = Config()
    cfg2.set("server", "name", "default-name")
    cfg2.merge_toml(String("[server]\nport = 2\n"))
    check(p, f, cfg2.get_str("server", "name") == "default-name", "untouched default key survives")
    check(p, f, cfg2.get_int("server", "port") == 2, "file key applied alongside")

    # introspection sanity
    check(p, f, len(ct.sections()) >= 2, "sections() lists tables")

    print("passed:", p, " failed:", f)
    if f == 0:
        print("ALL CONFIG TESTS PASSED")

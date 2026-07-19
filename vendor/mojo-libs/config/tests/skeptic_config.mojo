# Adversarial probes for config/config.mojo (layered Config).

from config.config import Config
from config.value import ConfigTree, ConfigValue


def _showkey(label: String, c: Config, section: String, key: String) raises:
    if c.has(section, key):
        print(label, ": [", section, "]", key, "='", c.get_str(section, key), "'")
    else:
        print(label, ": MISSING [", section, "]", key)


def main() raises:
    # ── .env parsing ──
    var c = Config()
    c.load_dotenv(String("/tmp/probe.env"))
    _showkey(String("E1 export FOO"), c, String("env"), String("FOO"))
    _showkey(String("E2 QUOTED spaces"), c, String("env"), String("QUOTED"))
    _showkey(String("E3 single-quote"), c, String("env"), String("SQ"))
    _showkey(String("E4 BLANKNEXT"), c, String("env"), String("BLANKNEXT"))
    _showkey(String("E5 EMPTY"), c, String("env"), String("EMPTY"))
    _showkey(String("E6 NOEQUALS (no =)"), c, String("env"), String("NOEQUALS"))
    _showkey(String("E7 HASH_IN_VAL"), c, String("env"), String("HASH_IN_VAL"))
    # comment lines must not appear as keys
    print("E8 comment-not-key: has 'a comment'? ", c.has(String("env"), String("a comment")))

    # ── env overlay for a key that exists ONLY in env (NEW key) ──
    var c2 = Config.from_ini(String("[srv]\nhost = local\n"))
    var env2 = Dict[String, String]()
    env2[String("APP_SRV_HOST")] = String("override")   # exists in tree -> should override
    env2[String("APP_SRV_NEWKEY")] = String("newval")    # NOT in tree -> added or ignored?
    c2.overlay_env_dict(String("APP"), env2)
    _showkey(String("O1 overlay-existing"), c2, String("srv"), String("host"))
    print("O2 overlay-new-key added? ", c2.has(String("srv"), String("newkey")), " (or NEWKEY?) ", c2.has(String("srv"), String("NEWKEY")))

    # ── dotted-table section env mapping [a.b] key -> PREFIX_A_B_KEY ──
    var c3 = Config.from_toml(String("[a.b]\nkey = \"orig\"\n"))
    print("O3 toml-section-name: has [a.b] key? ", c3.has(String("a.b"), String("key")))
    var env3 = Dict[String, String]()
    env3[String("APP_A_B_KEY")] = String("envwon")
    c3.overlay_env_dict(String("APP"), env3)
    _showkey(String("O3b overlay-dotted"), c3, String("a.b"), String("key"))

    # ── collision ambiguity: [a.b] key vs [a] b.key both -> APP_A_B_KEY ──
    var c4 = Config()
    c4.set(String("a.b"), String("key"), String("v1"))
    c4.set(String("a"), String("b.key"), String("v2"))   # b.key as literal key under [a]
    var env4 = Dict[String, String]()
    env4[String("APP_A_B_KEY")] = String("collide")
    c4.overlay_env_dict(String("APP"), env4)
    _showkey(String("O4a collision a.b/key"), c4, String("a.b"), String("key"))
    _showkey(String("O4b collision a/b.key"), c4, String("a"), String("b.key"))

    # ── *_or defaults on missing ──
    var c5 = Config.from_ini(String("[s]\nn = notanint\n"))
    print("D1 int_or on bad coercion: ", c5.get_int_or(String("s"), String("n"), Int64(-1)))
    print("D2 int_or on missing: ", c5.get_int_or(String("s"), String("missing"), Int64(99)))
    print("D3 str_or missing: '", c5.get_str_or(String("s"), String("missing"), String("dflt")), "'")

    # ── bad coercion raises (non-_or) ──
    try:
        var x = c5.get_int(String("s"), String("n"))
        print("D4 get_int bad: NO RAISE ", x)
    except e:
        print("D4 get_int bad: RAISED:", String(e))

    # ── dotted get_*_path on missing path ──
    try:
        var x = c5.get_str_path(String("s.missing"))
        print("D5 path missing: NO RAISE '", x, "'")
    except e:
        print("D5 path missing: RAISED:", String(e))
    # path with no dot
    try:
        var x = c5.get_str_path(String("nodot"))
        print("D6 path no-dot: NO RAISE '", x, "'")
    except e:
        print("D6 path no-dot: RAISED:", String(e))

    print("PROBES DONE")

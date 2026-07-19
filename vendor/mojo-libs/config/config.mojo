# config/config.mojo — the layered Config front-end. 100% Mojo, no FFI.
#
# This module OWNS only the layered `Config` wrapper. The shared value/tree
# types live in config/value.mojo (`ConfigValue`, `ConfigTree`); the parsers
# live in config/ini.mojo and config/toml.mojo. This file reuses all four and
# edits none of them.
#
# ── What Config is ────────────────────────────────────────────────────────────
# `Config` wraps a single merged `ConfigTree` and layers multiple sources with
# LATER-WINS precedence:
#
#     defaults  →  file(s)  →  .env file  →  process env  →  explicit set()
#
# Every overlay method (`merge_*`, `load_dotenv`, `overlay_env*`, `set`) writes
# ON TOP of the merged tree, so the *call order* establishes precedence. The
# headline precedence test drives exactly that sequence and asserts the value
# after each layer.
#
# ── Typed access + coercion ───────────────────────────────────────────────────
# Getters return typed values, coercing the stored `ConfigValue`. INI stores
# strings, so get_int/get_float/get_bool parse the string; TOML stores typed
# values but the getters still coerce (e.g. get_str on an int renders it). An
# impossible coercion raises (delegated to ConfigValue.as_* in value.mojo).
# Dotted-path variants accept "section.key" (split on the FIRST '.').
# `*_or` variants never raise: they return the supplied default on miss/bad coercion.
#
# ── .env section / env mapping (documented choices) ────────────────────────────
#   * load_dotenv(path) parses KEY=VALUE lines into section "env" (override via
#     load_dotenv_into(path, section)). Handles '#'/';' comment lines, optional
#     leading `export `, and surrounding single/double quotes around the value.
#   * overlay_env(prefix): for each (section, key) ALREADY present in the tree,
#     the process env var named  PREFIX_SECTION_KEY  (uppercased; '.' '-' ' ' →
#     '_') overrides the existing value. Anchoring on the tree's own names makes
#     the split rule unambiguous, including dotted TOML tables: [a.b] key →
#     PREFIX_A_B_KEY. If std.os.getenv is unavailable on a host, use the
#     overlay_env_dict(prefix, Dict) injected-environment variant instead.
#
# ── JSON ─────────────────────────────────────────────────────────────────────
# JSON config is NOT supported. from_file/merge_file reject a ".json" path. The
# json/ library yields a nested JSONValue tree that does not map onto the flat
# section/key ConfigTree without lossy flattening — out of scope for this lib.

from std.os import getenv
from std.collections import Dict
from config.value import ConfigTree, ConfigValue
from config.ini import parse_ini
from config.toml import parse_toml


struct Config(Movable):
    """Layered, typed configuration over a merged ConfigTree.

    Precedence (later wins): defaults → file(s) → .env → process env → set().
    See the module docstring for the env-var mapping and .env parsing rules.
    """

    var tree: ConfigTree

    def __init__(out self):
        self.tree = ConfigTree()

    def __init__(out self, var tree: ConfigTree):
        self.tree = tree^

    # ── constructors / loaders ────────────────────────────────────────────────
    @staticmethod
    def from_ini(text: String) raises -> Config:
        var t = parse_ini(text)
        return Config(t^)

    @staticmethod
    def from_toml(text: String) raises -> Config:
        var t = parse_toml(text)
        return Config(t^)

    @staticmethod
    def from_file(path: String) raises -> Config:
        var c = Config()
        c.merge_file(path)
        return c^

    # ── overlay sources (each wins over what is already present) ──────────────
    def merge_ini(mut self, text: String) raises:
        var t = parse_ini(text)
        self.tree.merge_over(t)

    def merge_toml(mut self, text: String) raises:
        var t = parse_toml(text)
        self.tree.merge_over(t)

    def merge_tree(mut self, other: ConfigTree) raises:
        self.tree.merge_over(other)

    def merge_file(mut self, path: String) raises:
        var text = _read_file(path)
        if _ends_with(path, ".ini"):
            self.merge_ini(text)
        elif _ends_with(path, ".toml"):
            self.merge_toml(text)
        elif _ends_with(path, ".json"):
            raise Error("JSON config is not supported: " + path)
        else:
            raise Error("unknown config extension (expected .ini/.toml): " + path)

    # ── .env loading ──────────────────────────────────────────────────────────
    # Parse KEY=VALUE lines into section "env" (default), overlaying existing
    # values. '#'/';' comment lines, optional leading `export `, and surrounding
    # single/double quotes on the value are handled.
    def load_dotenv(mut self, path: String) raises:
        self.load_dotenv_into(path, String("env"))

    def load_dotenv_into(mut self, path: String, section: String) raises:
        var text = _read_file(path)
        var lines = _split_lines(text)
        for li in range(len(lines)):
            var line = _trim(lines[li])
            if line.byte_length() == 0:
                continue
            var fb = line.as_bytes()
            var f0 = Int(fb[0])
            if f0 == 0x23 or f0 == 0x3B:  # '#' or ';' comment line
                continue
            if _starts_with(line, "export "):
                line = _trim(_substr(line, 7, line.byte_length()))
            var b = line.as_bytes()
            var n = line.byte_length()
            var eq = -1
            for i in range(n):
                if Int(b[i]) == 0x3D:  # '='
                    eq = i
                    break
            if eq < 0:
                continue  # not an assignment; skip
            var key = _trim(_substr(line, 0, eq))
            var rawval = _trim(_substr(line, eq + 1, n))
            if key.byte_length() == 0:
                continue
            var val = _strip_quotes(rawval)
            self.tree.set(section, key, ConfigValue.str_(val))

    # ── process-env overlay ───────────────────────────────────────────────────
    # For each (section, key) already present, look up env var PREFIX_SECTION_KEY
    # (uppercased; '.' '-' ' ' -> '_'). If set, it overrides the value.
    def overlay_env(mut self, prefix: String) raises:
        for sname in self.tree.sections():
            for k in self.tree.keys(sname):
                var var_name = _env_var_name(prefix, sname, k)
                var sentinel = String(chr(1))
                var got = getenv(var_name, sentinel)
                if got != sentinel:
                    self.tree.set(sname, k, ConfigValue.str_(got))

    # Injected-environment variant (tests / hosts without getenv). `env` keys
    # are raw PREFIX_SECTION_KEY names.
    def overlay_env_dict(mut self, prefix: String, env: Dict[String, String]) raises:
        for sname in self.tree.sections():
            for k in self.tree.keys(sname):
                var var_name = _env_var_name(prefix, sname, k)
                if var_name in env:
                    self.tree.set(sname, k, ConfigValue.str_(env[var_name]))

    # ── explicit override (highest precedence) ────────────────────────────────
    def set(mut self, section: String, key: String, value_string: String) raises:
        self.tree.set(section, key, ConfigValue.str_(value_string))

    def set_value(mut self, section: String, key: String, var v: ConfigValue) raises:
        self.tree.set(section, key, v^)

    # ── typed access ──────────────────────────────────────────────────────────
    def has(self, section: String, key: String) raises -> Bool:
        return self.tree.has(section, key)

    def get_value(self, section: String, key: String) raises -> ConfigValue:
        return self.tree.get(section, key)

    def get_str(self, section: String, key: String) raises -> String:
        return self.tree.get(section, key).as_str()

    def get_int(self, section: String, key: String) raises -> Int64:
        return self.tree.get(section, key).as_int()

    def get_float(self, section: String, key: String) raises -> Float64:
        return self.tree.get(section, key).as_float()

    def get_bool(self, section: String, key: String) raises -> Bool:
        return self.tree.get(section, key).as_bool()

    # List of strings: each element of the stored array coerced via as_str().
    def get_list(self, section: String, key: String) raises -> List[String]:
        var arr = self.tree.get(section, key).as_array()
        var out = List[String]()
        for i in range(len(arr)):
            out.append(arr[i].as_str())
        return out^

    # ── dotted-path access ("section.key"; only the FIRST '.' splits) ─────────
    def get_str_path(self, path: String) raises -> String:
        var sk = _split_path(path)
        return self.get_str(sk[0], sk[1])

    def get_int_path(self, path: String) raises -> Int64:
        var sk = _split_path(path)
        return self.get_int(sk[0], sk[1])

    def get_float_path(self, path: String) raises -> Float64:
        var sk = _split_path(path)
        return self.get_float(sk[0], sk[1])

    def get_bool_path(self, path: String) raises -> Bool:
        var sk = _split_path(path)
        return self.get_bool(sk[0], sk[1])

    def get_list_path(self, path: String) raises -> List[String]:
        var sk = _split_path(path)
        return self.get_list(sk[0], sk[1])

    # ── non-raising *_or variants ─────────────────────────────────────────────
    def get_str_or(self, section: String, key: String, default: String) -> String:
        try:
            return self.tree.get(section, key).as_str()
        except:
            return default

    def get_int_or(self, section: String, key: String, default: Int64) -> Int64:
        try:
            return self.tree.get(section, key).as_int()
        except:
            return default

    def get_float_or(self, section: String, key: String, default: Float64) -> Float64:
        try:
            return self.tree.get(section, key).as_float()
        except:
            return default

    def get_bool_or(self, section: String, key: String, default: Bool) -> Bool:
        try:
            return self.tree.get(section, key).as_bool()
        except:
            return default

    # ── introspection ─────────────────────────────────────────────────────────
    def sections(self) raises -> List[String]:
        return self.tree.sections()

    def keys(self, section: String) raises -> List[String]:
        return self.tree.keys(section)


# ── module-level helpers (private to config.mojo) ─────────────────────────────
def _split_path(path: String) raises -> List[String]:
    var b = path.as_bytes()
    var n = path.byte_length()
    var dot = -1
    for i in range(n):
        if Int(b[i]) == 0x2E:  # '.'
            dot = i
            break
    if dot < 0:
        raise Error("dotted path must be 'section.key': " + path)
    var out = List[String]()
    out.append(_substr(path, 0, dot))
    out.append(_substr(path, dot + 1, n))
    return out^


def _env_var_name(prefix: String, section: String, key: String) -> String:
    var raw = prefix + "_" + section + "_" + key
    return _to_env_token(raw)


def _to_env_token(s: String) -> String:
    # Uppercase ASCII letters; map '.' '-' ' ' to '_'; keep digits/'_'/other.
    var b = s.as_bytes()
    var n = s.byte_length()
    var out = String("")
    for i in range(n):
        var c = Int(b[i])
        if c >= 97 and c <= 122:  # a-z
            out += chr(c - 32)
        elif c == 0x2E or c == 0x2D or c == 0x20:  # . - space
            out += "_"
        else:
            out += chr(c)
    return out


def _trim(s: String) -> String:
    var b = s.as_bytes()
    var n = s.byte_length()
    var start = 0
    while start < n:
        var c = Int(b[start])
        if c == 0x20 or c == 0x09 or c == 0x0D or c == 0x0A:
            start += 1
        else:
            break
    var end = n
    while end > start:
        var c = Int(b[end - 1])
        if c == 0x20 or c == 0x09 or c == 0x0D or c == 0x0A:
            end -= 1
        else:
            break
    var out = String("")
    for i in range(start, end):
        out += chr(Int(b[i]))
    return out


def _substr(s: String, start: Int, end: Int) -> String:
    var b = s.as_bytes()
    var out = String("")
    for i in range(start, end):
        out += chr(Int(b[i]))
    return out


def _starts_with(s: String, prefix: String) -> Bool:
    var sn = s.byte_length()
    var pn = prefix.byte_length()
    if pn > sn:
        return False
    var sb = s.as_bytes()
    var pb = prefix.as_bytes()
    for i in range(pn):
        if Int(sb[i]) != Int(pb[i]):
            return False
    return True


def _ends_with(s: String, suffix: String) -> Bool:
    var sn = s.byte_length()
    var fn = suffix.byte_length()
    if fn > sn:
        return False
    var sb = s.as_bytes()
    var fb = suffix.as_bytes()
    for i in range(fn):
        if Int(sb[sn - fn + i]) != Int(fb[i]):
            return False
    return True


def _strip_quotes(s: String) -> String:
    var n = s.byte_length()
    if n < 2:
        return s
    var b = s.as_bytes()
    var first = Int(b[0])
    var last = Int(b[n - 1])
    if (first == 0x22 and last == 0x22) or (first == 0x27 and last == 0x27):
        return _substr(s, 1, n - 1)
    return s


def _split_lines(text: String) -> List[String]:
    var out = List[String]()
    var b = text.as_bytes()
    var n = text.byte_length()
    var cur = String("")
    for i in range(n):
        var c = Int(b[i])
        if c == 0x0A:  # '\n'
            out.append(cur)
            cur = String("")
        elif c == 0x0D:  # '\r' dropped
            continue
        else:
            cur += chr(c)
    out.append(cur)
    return out^


def _read_file(path: String) raises -> String:
    var f = open(path, "r")
    var s = f.read()
    f.close()
    return s

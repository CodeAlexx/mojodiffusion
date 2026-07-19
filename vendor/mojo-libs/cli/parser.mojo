# cli.parser — an enterprise-grade builder + parser for command-line arguments.
# 100% Mojo, no FFI.
#
# Surface supported:
#   flags (bool):    --verbose / -v ; combined short flags -abc == -a -b -c
#   count flags:     -vvv -> 3 (get_count); also --verbose --verbose -> 2
#   options (value): --name X | --name=X | -n X | -nX | -n=X
#                    typed: str | int | float ; default + required
#                    choices/enum: restrict value to a fixed set
#                    multi-value/repeatable: --inc a --inc b -> [a, b]
#                    comma-split: a single value "a,b,c" -> [a, b, c]
#   positionals:     ordered, named
#   rest (variadic): a final catch-all collecting leftover positionals
#   --:              everything after is positional (stops option parsing)
#   subcommands:     `app build --opt` ; each sub has its own parser
#   env fallback:    an option may name an env var; used if not on the cmdline
#   exclusive grps:  set_exclusive(a, b) -> error if both given
#   version:         set_version("1.2.3") ; --version reported via wants_version
#   help:            --help / -h reported via wants_help; see cli/help.mojo
#
# Env vars are read via std.os.getenv (verified available on this toolchain).
# parse() also accepts an injected env Dict so tests are deterministic; when a
# key is present in that Dict it takes precedence over the process environment.

from std.os import getenv


# ── spec kinds ───────────────────────────────────────────────────────────────
comptime KIND_FLAG = 0
comptime KIND_OPTION = 1
comptime KIND_POSITIONAL = 2
comptime KIND_COUNT = 3        # repeatable presence flag (-vvv)
comptime KIND_REST = 4         # variadic final positional

# ── value types for options ──────────────────────────────────────────────────
comptime TYPE_STR = 0
comptime TYPE_INT = 1
comptime TYPE_FLOAT = 2


# A single registered argument definition.
struct ArgSpec(Copyable, Movable):
    var kind: Int          # KIND_*
    var long: String       # canonical name (no leading --); also the lookup key
    var short: String      # single char (no leading -), or "" if none
    var help: String       # human description
    var default: String    # default value for an option ("" if none)
    var has_default: Bool   # whether a default was supplied
    var required: Bool     # option must be present (or satisfied by env)
    var env: String        # env var name to fall back to, or "" if none
    var value_type: Int    # TYPE_STR | TYPE_INT | TYPE_FLOAT (options only)
    var choices: List[String]  # allowed values (empty = unrestricted)
    var multi: Bool        # repeatable / multi-value option -> list
    var comma_split: Bool   # split a single value on commas into a list

    def __init__(out self, kind: Int, long: String, short: String, help: String,
                 default: String, has_default: Bool, required: Bool, env: String,
                 value_type: Int = TYPE_STR, var choices: List[String] = List[String](),
                 multi: Bool = False, comma_split: Bool = False):
        self.kind = kind
        self.long = long
        self.short = short
        self.help = help
        self.default = default
        self.has_default = has_default
        self.required = required
        self.env = env
        self.value_type = value_type
        self.choices = choices^
        self.multi = multi
        self.comma_split = comma_split


# Result of parsing.
struct ParseResult(Copyable, Movable):
    var bools: Dict[String, Bool]
    var counts: Dict[String, Int]
    var strs: Dict[String, String]
    var lists: Dict[String, List[String]]
    var present: Dict[String, Bool]   # which named args actually resolved
    var _positionals: List[String]
    var _rest: List[String]
    var _subcommand: String
    var _sub: List[ParseResult]       # 0 or 1 element (the sub-result)
    var _wants_help: Bool
    var _wants_version: Bool

    def __init__(out self):
        self.bools = Dict[String, Bool]()
        self.counts = Dict[String, Int]()
        self.strs = Dict[String, String]()
        self.lists = Dict[String, List[String]]()
        self.present = Dict[String, Bool]()
        self._positionals = List[String]()
        self._rest = List[String]()
        self._subcommand = String("")
        self._sub = List[ParseResult]()
        self._wants_help = False
        self._wants_version = False

    def get_bool(self, name: String) raises -> Bool:
        if name in self.bools:
            return self.bools[name]
        return False

    def get_count(self, name: String) raises -> Int:
        if name in self.counts:
            return self.counts[name]
        return 0

    def get_str(self, name: String) raises -> String:
        if name in self.strs:
            return self.strs[name].copy()
        return String("")

    def get_int(self, name: String) raises -> Int:
        return Int(self.get_str(name))

    def get_float(self, name: String) raises -> Float64:
        return Float64(self.get_str(name))

    # Multi-value / comma-split / repeatable option values.
    def get_list(self, name: String) raises -> List[String]:
        if name in self.lists:
            return self.lists[name].copy()
        # fall back to a single scalar value if present
        if name in self.strs:
            var out_list = List[String]()
            out_list.append(self.strs[name].copy())
            return out_list^
        return List[String]()

    def has(self, name: String) raises -> Bool:
        if name in self.present:
            return self.present[name]
        return False

    def positionals(self) -> List[String]:
        return self._positionals.copy()

    # The variadic rest (everything beyond named positionals, if add_rest used).
    def rest(self) -> List[String]:
        return self._rest.copy()

    def subcommand(self) -> String:
        return self._subcommand

    def wants_help(self) -> Bool:
        return self._wants_help

    def wants_version(self) -> Bool:
        return self._wants_version

    # Sub-result for the chosen subcommand. Raises if there was none.
    def sub(self) raises -> ParseResult:
        if len(self._sub) == 0:
            raise Error("no subcommand was parsed")
        ref r = self._sub[0]
        return r.copy()


# Small helper: byte-range substring (slice syntax not available here).
def _substr(s: String, start: Int, end: Int) -> String:
    var b = s.as_bytes()
    var r = String("")
    var i = start
    while i < end:
        r += chr(Int(b[i]))
        i += 1
    return r^


# True if `s` looks like a negative number: "-" followed by a digit or "."
# (e.g. "-5", "-3.14", "-.5"). Used so option values and positionals can be
# negative numbers rather than being mistaken for flags.
def _is_neg_number(s: String) -> Bool:
    var n = s.byte_length()
    if n < 2:
        return False
    var b = s.as_bytes()
    if chr(Int(b[0])) != "-":
        return False
    var c = chr(Int(b[1]))
    var is_digit = c >= "0" and c <= "9"
    return is_digit or c == "."


# Split a string on a single-char delimiter.
def _split_commas(s: String) -> List[String]:
    var out_list = List[String]()
    var b = s.as_bytes()
    var n = s.byte_length()
    var cur = String("")
    var i = 0
    while i < n:
        var ch = chr(Int(b[i]))
        if ch == ",":
            out_list.append(cur.copy())
            cur = String("")
        else:
            cur += ch
        i += 1
    out_list.append(cur^)
    return out_list^


# ── the parser ─────────────────────────────────────────────────────────────
struct ArgParser(Copyable, Movable):
    var name: String
    var about: String
    var version: String
    var specs: List[ArgSpec]
    var _sub_names: List[String]
    var _sub_parsers: List[ArgParser]
    # mutually exclusive groups: each entry is a list of canonical long names.
    var _excl_groups: List[List[String]]

    def __init__(out self, name: String = String("app"), about: String = String("")):
        self.name = name
        self.about = about
        self.version = String("")
        self.specs = List[ArgSpec]()
        self._sub_names = List[String]()
        self._sub_parsers = List[ArgParser]()
        self._excl_groups = List[List[String]]()

    # ── builder API ──────────────────────────────────────────────────────────
    def add_flag(mut self, long: String, short: String = String(""),
                 help: String = String("")):
        self.specs.append(
            ArgSpec(KIND_FLAG, long, short, help, String(""), False, False, String(""))
        )

    def add_count_flag(mut self, long: String, short: String = String(""),
                       help: String = String("")):
        self.specs.append(
            ArgSpec(KIND_COUNT, long, short, help, String(""), False, False, String(""))
        )

    def add_option(mut self, long: String, short: String = String(""),
                   help: String = String(""), default: String = String(""),
                   has_default: Bool = False, required: Bool = False,
                   env: String = String(""), value_type: Int = TYPE_STR,
                   var choices: List[String] = List[String](),
                   multi: Bool = False, comma_split: Bool = False):
        self.specs.append(
            ArgSpec(KIND_OPTION, long, short, help, default, has_default, required,
                    env, value_type, choices^, multi, comma_split)
        )

    def add_positional(mut self, name: String, help: String = String("")):
        self.specs.append(
            ArgSpec(KIND_POSITIONAL, name, String(""), help, String(""), False, False, String(""))
        )

    # A final variadic catch-all collecting leftover positionals.
    def add_rest(mut self, name: String, help: String = String("")):
        self.specs.append(
            ArgSpec(KIND_REST, name, String(""), help, String(""), False, False, String(""))
        )

    # Register a fully-built sub-parser under `name`.
    def add_subcommand(mut self, name: String, var sub: ArgParser):
        self._sub_names.append(name)
        self._sub_parsers.append(sub^)

    # Mark a set of (canonical long) names as mutually exclusive.
    def set_exclusive(mut self, var names: List[String]):
        self._excl_groups.append(names^)

    def set_version(mut self, version: String):
        self.version = version

    # ── lookups ────────────────────────────────────────────────────────────────
    def _find_long(self, long: String) -> Int:
        for i in range(len(self.specs)):
            ref s = self.specs[i]
            if (s.kind == KIND_FLAG or s.kind == KIND_OPTION or s.kind == KIND_COUNT) \
               and s.long == long:
                return i
        return -1

    def _find_short(self, short: String) -> Int:
        for i in range(len(self.specs)):
            ref s = self.specs[i]
            if (s.kind == KIND_FLAG or s.kind == KIND_OPTION or s.kind == KIND_COUNT) \
               and s.short != "" and s.short == short:
                return i
        return -1

    def _find_sub(self, name: String) -> Int:
        for i in range(len(self._sub_names)):
            ref n = self._sub_names[i]
            if n == name:
                return i
        return -1

    # ── value handling ───────────────────────────────────────────────────────
    def _validate_choice(self, spec: ArgSpec, value: String) raises:
        if len(spec.choices) == 0:
            return
        for i in range(len(spec.choices)):
            ref c = spec.choices[i]
            if c == value:
                return
        var allowed = String("")
        for i in range(len(spec.choices)):
            if i > 0:
                allowed += ", "
            allowed += spec.choices[i]
        raise Error("invalid value '" + value + "' for option --" + spec.long
                    + " (choices: " + allowed + ")")

    def _coerce_check(self, spec: ArgSpec, value: String) raises:
        if spec.value_type == TYPE_INT:
            try:
                var _i = Int(value)
            except e:
                raise Error("option --" + spec.long + " expects an integer, got '"
                            + value + "'")
        elif spec.value_type == TYPE_FLOAT:
            try:
                var _f = Float64(value)
            except e:
                raise Error("option --" + spec.long + " expects a float, got '"
                            + value + "'")

    # Record a parsed value for an option spec, honoring multi/comma-split.
    def _set_option_value(self, idx: Int, value: String, mut res: ParseResult) raises:
        ref spec = self.specs[idx]
        var key = spec.long.copy()
        # Determine the concrete value list to add.
        var vals = List[String]()
        if spec.comma_split:
            var parts = _split_commas(value)
            for j in range(len(parts)):
                vals.append(parts[j].copy())
        else:
            vals.append(value.copy())
        # Validate each value.
        for j in range(len(vals)):
            self._validate_choice(spec, vals[j])
            self._coerce_check(spec, vals[j])
        if spec.multi or spec.comma_split:
            if key not in res.lists:
                res.lists[key] = List[String]()
            ref cur = res.lists[key]
            for j in range(len(vals)):
                cur.append(vals[j].copy())
            # last value also lands in strs for get_str convenience
            res.strs[key] = vals[len(vals) - 1].copy()
        else:
            res.strs[key] = vals[0].copy()
        res.present[key] = True

    # ── parse ──────────────────────────────────────────────────────────────────
    def parse(self, args: List[String]) raises -> ParseResult:
        var empty = Dict[String, String]()
        return self.parse(args, empty)

    def parse(self, args: List[String], env: Dict[String, String]) raises -> ParseResult:
        var res = ParseResult()
        var positionals = List[String]()
        var no_more_options = False

        var i = 0
        while i < len(args):
            var arg = args[i].copy()

            # "--" : stop option parsing, rest are positionals.
            if not no_more_options and arg == "--":
                no_more_options = True
                i += 1
                continue

            # subcommand check: first non-option token that matches a sub name.
            if not no_more_options and not arg.startswith("-"):
                var si = self._find_sub(arg)
                if si >= 0:
                    var rest = List[String]()
                    var j = i + 1
                    while j < len(args):
                        rest.append(args[j].copy())
                        j += 1
                    ref subp = self._sub_parsers[si]
                    var subres = subp.parse(rest, env)
                    res._subcommand = arg.copy()
                    res._sub.append(subres^)
                    return self.parse_finish(res^, env)

            # --help / --version (long), checked before unknown-option error.
            if not no_more_options and (arg == "--help"):
                res._wants_help = True
                i += 1
                continue
            if not no_more_options and (arg == "--version"):
                res._wants_version = True
                i += 1
                continue

            if not no_more_options and arg.startswith("--") and arg.byte_length() > 2:
                # long flag/option: --name | --name=value
                var body = _substr(arg, 2, arg.byte_length())
                var eq = body.find("=")
                var key: String
                var inline_val: String
                var has_inline = False
                if eq >= 0:
                    key = _substr(body, 0, eq)
                    inline_val = _substr(body, eq + 1, body.byte_length())
                    has_inline = True
                else:
                    key = body.copy()
                    inline_val = String("")
                var idx = self._find_long(key)
                if idx < 0:
                    raise Error("unknown option: --" + key)
                ref spec = self.specs[idx]
                if spec.kind == KIND_FLAG:
                    if has_inline:
                        raise Error("flag --" + key + " does not take a value")
                    res.bools[spec.long] = True
                    res.present[spec.long] = True
                elif spec.kind == KIND_COUNT:
                    if has_inline:
                        raise Error("flag --" + key + " does not take a value")
                    var cur = 0
                    if spec.long in res.counts:
                        cur = res.counts[spec.long]
                    res.counts[spec.long] = cur + 1
                    res.present[spec.long] = True
                else:
                    var value: String
                    if has_inline:
                        value = inline_val.copy()
                    else:
                        if i + 1 >= len(args):
                            raise Error("missing value for option: --" + key)
                        ref nxt = args[i + 1]
                        # A flag-looking next token is NOT a value (matches
                        # argparse: "expected one argument"). A lone "-" and a
                        # negative number ARE accepted as values.
                        if nxt.byte_length() >= 2 and nxt.startswith("-") \
                           and not _is_neg_number(nxt):
                            raise Error("option --" + key + ": expected a value")
                        value = args[i + 1].copy()
                        i += 1
                    self._set_option_value(idx, value, res)
                i += 1
                continue

            if not no_more_options and arg.startswith("-") and arg.byte_length() > 1 \
               and arg != "-" and not _is_neg_number(arg):
                # short cluster: -v | -abc | -vvv | -n value | -nValue | -n=value | -h
                var body = _substr(arg, 1, arg.byte_length())
                var consumed_next = self._parse_short_cluster(body, args, i, res)
                i += 1 + consumed_next
                continue

            # plain positional
            positionals.append(arg.copy())
            i += 1

        # assign named positionals in order; leftover -> rest (if declared)
        for k in range(len(positionals)):
            res._positionals.append(positionals[k].copy())
        var p = 0
        var rest_idx = -1
        for si in range(len(self.specs)):
            ref s = self.specs[si]
            if s.kind == KIND_POSITIONAL:
                if p < len(positionals):
                    res.strs[s.long] = positionals[p].copy()
                    res.present[s.long] = True
                    p += 1
            elif s.kind == KIND_REST:
                rest_idx = si
        if rest_idx >= 0:
            while p < len(positionals):
                res._rest.append(positionals[p].copy())
                p += 1

        return self.parse_finish(res^, env)

    # Apply env fallback + defaults + required + exclusive-group checks.
    def parse_finish(self, var res: ParseResult, env: Dict[String, String]) raises -> ParseResult:
        # If help/version requested, skip required/exclusive validation entirely.
        if res._wants_help or res._wants_version:
            return res^

        for si in range(len(self.specs)):
            ref s = self.specs[si]
            if s.kind != KIND_OPTION:
                continue
            var got = s.long in res.present and res.present[s.long]
            if not got:
                # env fallback
                if s.env != "":
                    var ev = String("")
                    var found = False
                    if s.env in env:
                        ev = env[s.env].copy()
                        found = True
                    else:
                        var pe = getenv(s.env, String("\x00__cli_unset__"))
                        if pe != "\x00__cli_unset__":
                            ev = pe.copy()
                            found = True
                    if found:
                        self._set_option_value(si, ev, res)
                        got = True
            if not got and s.has_default:
                res.strs[s.long] = s.default.copy()
                # default does not count as "present" for has(); value still readable
            if not got and (not s.has_default) and s.required:
                raise Error("missing required option: --" + s.long)

        # mutually exclusive groups
        for gi in range(len(self._excl_groups)):
            ref grp = self._excl_groups[gi]
            var seen = String("")
            var conflict = String("")
            for ni in range(len(grp)):
                ref nm = grp[ni]
                var present_here = nm in res.present and res.present[nm]
                if present_here:
                    if seen != "":
                        conflict = nm.copy()
                        break
                    seen = nm.copy()
            if conflict != "":
                raise Error("options --" + seen + " and --" + conflict
                            + " are mutually exclusive")
        return res^

    # Returns how many *extra* args were consumed (0 or 1, for -n value form).
    def _parse_short_cluster(self, body: String, args: List[String], i: Int,
                             mut res: ParseResult) raises -> Int:
        var b = body.as_bytes()
        var k = 0
        var n = body.byte_length()
        while k < n:
            var ch = chr(Int(b[k]))
            # short help
            if ch == "h" and self._find_short(ch) < 0:
                res._wants_help = True
                k += 1
                continue
            var idx = self._find_short(ch)
            if idx < 0:
                raise Error("unknown flag: -" + ch)
            ref spec = self.specs[idx]
            if spec.kind == KIND_FLAG:
                res.bools[spec.long] = True
                res.present[spec.long] = True
                k += 1
                continue
            if spec.kind == KIND_COUNT:
                var cur = 0
                if spec.long in res.counts:
                    cur = res.counts[spec.long]
                res.counts[spec.long] = cur + 1
                res.present[spec.long] = True
                k += 1
                continue
            # option: value is the rest of the cluster, or the next arg.
            var rest = _substr(body, k + 1, n)
            if rest != "":
                if rest.startswith("="):
                    rest = _substr(rest, 1, rest.byte_length())
                self._set_option_value(idx, rest, res)
                return 0
            else:
                if i + 1 >= len(args):
                    raise Error("missing value for option: -" + ch)
                ref nxt = args[i + 1]
                if nxt.byte_length() >= 2 and nxt.startswith("-") \
                   and not _is_neg_number(nxt):
                    raise Error("option -" + ch + ": expected a value")
                self._set_option_value(idx, args[i + 1].copy(), res)
                return 1
        return 0

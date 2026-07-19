# cli.help — auto usage/help and version text generation for an ArgParser.
#
#   from cli.help import render_help, render_version
#
# render_help(p) returns a formatted usage block: a usage line, the optional
# "about" paragraph, then aligned Flags / Options / Arguments / Commands
# sections. Each option line shows name, short alias, help, [required],
# [default: X], [choices: ...], [env: VAR]. render_version(p) returns the
# program name + the version set via set_version().

from cli.parser import (
    ArgParser,
    ArgSpec,
    KIND_FLAG,
    KIND_OPTION,
    KIND_POSITIONAL,
    KIND_COUNT,
    KIND_REST,
    TYPE_INT,
    TYPE_FLOAT,
)


# Right-pad `s` to `width` with spaces (for aligned columns).
def _pad(s: String, width: Int) -> String:
    var r = s.copy()
    var n = s.byte_length()
    var i = n
    while i < width:
        r += " "
        i += 1
    return r^


def _type_name(t: Int) -> String:
    if t == TYPE_INT:
        return String("int")
    if t == TYPE_FLOAT:
        return String("float")
    return String("str")


# The left-column label for a flag/option/count spec, e.g. "-v, --verbose".
def _spec_label(s: ArgSpec) -> String:
    var label = String("")
    if s.short != "":
        label += "-" + s.short + ", "
    else:
        label += "    "
    label += "--" + s.long
    if s.kind == KIND_OPTION:
        label += " <" + _type_name(s.value_type) + ">"
    return label^


def render_version(p: ArgParser) -> String:
    var v = p.version
    if v == "":
        v = String("(unset)")
    return p.name + " " + v


def render_help(p: ArgParser) -> String:
    var h = String("")
    # ── usage line ──
    h += "Usage: " + p.name
    var has_flagopt = False
    for i in range(len(p.specs)):
        ref s = p.specs[i]
        if s.kind == KIND_FLAG or s.kind == KIND_OPTION or s.kind == KIND_COUNT:
            has_flagopt = True
            break
    if has_flagopt:
        h += " [OPTIONS]"
    for i in range(len(p.specs)):
        ref s = p.specs[i]
        if s.kind == KIND_POSITIONAL:
            h += " <" + s.long + ">"
    for i in range(len(p.specs)):
        ref s = p.specs[i]
        if s.kind == KIND_REST:
            h += " [" + s.long + "...]"
    if len(p._sub_names) > 0:
        h += " <COMMAND>"
    h += "\n"
    if p.about != "":
        h += "\n" + p.about + "\n"

    # ── compute alignment width across flags/options ──
    var col = 0
    for i in range(len(p.specs)):
        ref s = p.specs[i]
        if s.kind == KIND_FLAG or s.kind == KIND_OPTION or s.kind == KIND_COUNT:
            var w = _spec_label(s).byte_length()
            if w > col:
                col = w
    col += 2

    # ── flags (bool + count) ──
    var any_flag = False
    for i in range(len(p.specs)):
        ref s = p.specs[i]
        if s.kind == KIND_FLAG or s.kind == KIND_COUNT:
            if not any_flag:
                h += "\nFlags:\n"
                any_flag = True
            h += "  " + _pad(_spec_label(s), col)
            if s.help != "":
                h += s.help
            if s.kind == KIND_COUNT:
                h += " (repeatable)"
            h += "\n"

    # ── options ──
    var any_opt = False
    for i in range(len(p.specs)):
        ref s = p.specs[i]
        if s.kind == KIND_OPTION:
            if not any_opt:
                h += "\nOptions:\n"
                any_opt = True
            h += "  " + _pad(_spec_label(s), col)
            if s.help != "":
                h += s.help
            if s.required:
                h += " [required]"
            if s.has_default:
                h += " [default: " + s.default + "]"
            if len(s.choices) > 0:
                h += " [choices: "
                for c in range(len(s.choices)):
                    if c > 0:
                        h += ", "
                    h += s.choices[c]
                h += "]"
            if s.multi:
                h += " [repeatable]"
            if s.env != "":
                h += " [env: " + s.env + "]"
            h += "\n"

    # ── positionals + rest ──
    var any_pos = False
    for i in range(len(p.specs)):
        ref s = p.specs[i]
        if s.kind == KIND_POSITIONAL or s.kind == KIND_REST:
            if not any_pos:
                h += "\nArguments:\n"
                any_pos = True
            var nm = s.long.copy()
            if s.kind == KIND_REST:
                nm += "..."
            h += "  " + _pad(nm, col)
            if s.help != "":
                h += s.help
            h += "\n"

    # ── subcommands ──
    if len(p._sub_names) > 0:
        h += "\nCommands:\n"
        for i in range(len(p._sub_names)):
            ref nm = p._sub_names[i]
            ref sp = p._sub_parsers[i]
            h += "  " + _pad(nm, col)
            if sp.about != "":
                h += sp.about
            h += "\n"

    # ── version footer ──
    if p.version != "":
        h += "\nVersion: " + p.version + "\n"
    return h^

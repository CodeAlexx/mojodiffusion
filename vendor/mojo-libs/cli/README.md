# cli — command-line argument parsing for Mojo (100% Mojo, no FFI)

A clap/argparse-class argument parser: a fluent `ArgParser` builder, typed
options, choices, repeatable/comma-split lists, count flags, positionals + a
variadic rest, single-level subcommands, env-var fallback, mutually-exclusive
groups, and auto usage/help/version text. Zero C, zero dependencies.

## Modules

| Module | What it is |
|---|---|
| `parser.mojo` | `ArgParser` (the builder) + `ParseResult` (the parsed values) + `ArgSpec` (one registered argument). All parsing logic; raises `Error` with a clear message on any user error. |
| `help.mojo` | `render_help(parser)` → aligned usage + Flags/Options/Arguments/Commands block; `render_version(parser)` → `"<name> <version>"`. |
| `tests/parser_test.mojo` | Exhaustive feature checks via `check(...)` + a 3-input cross-check against Python `argparse`. Prints `passed: N failed: M` and `ALL CLI TESTS PASSED`. |
| `tests/argparse_oracle.py` | Python `argparse` driver mirroring the test spec; the oracle for the cross-check. |
| `tests/run_oracle.sh` | Regenerates the oracle output for the three representative inputs. |

## Builder API (`ArgParser`)

| Method | Registers |
|---|---|
| `add_flag(long, short, help)` | Boolean presence flag (`--verbose` / `-v`). |
| `add_count_flag(long, short, help)` | Repeatable flag; `-vvv` → 3 (read via `get_count`). |
| `add_option(long, short, help, default, has_default, required, env, value_type, choices, multi, comma_split)` | A value option. `value_type` ∈ `TYPE_STR`/`TYPE_INT`/`TYPE_FLOAT`; `choices` restricts to a set; `multi` makes it repeatable into a list; `comma_split` splits one value on commas; `env` names a fallback env var. |
| `add_positional(name, help)` | A named, ordered positional. |
| `add_rest(name, help)` | A final variadic catch-all for leftover positionals. |
| `add_subcommand(name, sub)` | Attach a fully-built sub-`ArgParser` under `name`. |
| `set_exclusive(names)` | Mark a list of canonical long names mutually exclusive. |
| `set_version(version)` | Version string reported by `--version` / `render_version`. |

## Reading results (`ParseResult`)

`get_bool(name)`, `get_count(name)`, `get_str(name)`, `get_int(name) raises`,
`get_float(name) raises`, `get_list(name) raises`, `has(name) raises`,
`positionals()`, `rest()`, `subcommand()`, `sub() raises`, `wants_help()`,
`wants_version()`.

`parse(args) raises -> ParseResult` (process env used for fallback), or
`parse(args, env: Dict[String,String]) raises` (injected env — keys here win
over the process environment; absent keys fall through to `std.os.getenv`).

## Value forms parsed

`--name X` · `--name=X` · `-n X` · `-nX` · `-n=X` · combined short `-abc` ·
count `-vvv` · `--` stops option parsing (rest become positionals) ·
`--inc a --inc b` (repeatable) · `a,b,c` (comma-split).

## Errors raised

Unknown flag/option, missing option value, missing required option, bad
`int`/`float` coercion (typed options, at parse time), invalid choice, and
mutually-exclusive-group violation — each a `raise Error("…")` with context.

An option will **not** swallow a flag-looking next token as its value:
`--name --verbose` raises `option --name: expected a value` (matching argparse's
"expected one argument") rather than silently setting `name="--verbose"`. A
**negative number** (`-5`, `-3.14`) IS accepted as a value, and a bare
`-<digit>` token with no numeric option defined is treated as a positional (also
matching argparse).

## Example

```mojo
from cli.parser import ArgParser, TYPE_INT
from cli.help import render_help, render_version

def main() raises:
    # Root parser with a global flag + version.
    var app = ArgParser(String("git-ish"), String("a tiny VCS"))
    app.set_version(String("1.0.0"))
    app.add_count_flag(String("verbose"), String("v"), String("more verbose; repeatable"))

    # A `build` subcommand with its own typed option, choice, and positional.
    var build = ArgParser(String("build"), String("build the project"))
    build.add_option(String("jobs"), String("j"), String("parallel jobs"),
                     String("1"), True, False, String("BUILD_JOBS"), TYPE_INT)
    var profiles = List[String]()
    profiles.append(String("debug"))
    profiles.append(String("release"))
    build.add_option(String("profile"), String("p"), String("build profile"),
                     String("debug"), True, False, String(""), 0, profiles^)
    build.add_positional(String("target"), String("what to build"))
    app.add_subcommand(String("build"), build^)

    # Root flags come before the subcommand; sub options come after it.
    var argv: List[String] = [String("-vv"), String("build"), String("-j"),
                              String("8"), String("--profile"), String("release"),
                              String("server")]
    var r = app.parse(argv)

    if r.wants_help():
        print(render_help(app))
        return
    if r.wants_version():
        print(render_version(app))
        return

    print("command:", r.subcommand())          # build
    var sub = r.sub()
    print("jobs:", sub.get_int(String("jobs"))) # 8  (typed int)
    print("profile:", sub.get_str(String("profile")))   # release
    print("target:", sub.get_str(String("target")))     # server
```

## Cross-checked against Python argparse

`tests/parser_test.mojo` builds a parser mirroring `tests/argparse_oracle.py`
and asserts byte-identical results for three representative argv inputs
(flags+name+count+positionals; choice+append+rest; short-opt+defaults). The
oracle output is regenerated by `tests/run_oracle.sh`.

## Honest scope — what is NOT supported

- **Nested subcommands beyond one level.** A subcommand is itself an `ArgParser`,
  so `set_exclusive`/options/positionals work inside it, but `sub()` returns a
  single level; `app a b --opt` deeper than one hop is not routed.
- **Config-file binding.** No TOML/INI/JSON layering; only command line + a
  single env-var fallback per option.
- **Shell completion generation.** No bash/zsh/fish completion scripts.
- **Negatable flags** (`--no-verbose`), **prefix abbreviation** (`--verb` for
  `--verbose`), and **interspersed argparse-style `nargs=N`** (only the single
  variadic `add_rest` is provided).
- **Per-subcommand `--help`/`--version` rendering dispatch.** The intent bits
  (`wants_help`/`wants_version`) are set; the caller chooses what to print.

## Env handling

Env fallback uses `std.os.getenv` (verified to compile/run on this toolchain).
`parse(args, env)` additionally accepts an injected `Dict[String, String]` so
tests are deterministic; injected keys take precedence over the process
environment, and absent keys fall through to `getenv`.

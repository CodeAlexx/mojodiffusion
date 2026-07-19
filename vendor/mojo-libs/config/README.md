# config — configuration for Mojo (INI + TOML + .env + env layering, 100% Mojo)

Parse **INI** and **TOML** config files, layer them with `.env` files and process
environment variables (defined precedence), and read values back **typed**. Pure
Mojo, no FFI. Every parser is verified against a Python oracle — INI vs
`configparser`, TOML vs `tomllib`.

## Modules

| Module | What it is |
|---|---|
| `value.mojo` | `ConfigValue` — tagged value (null/str/int/float/bool/array) with coercions (`as_str`/`as_int`/`as_float`/`as_bool`/`as_array`); `ConfigTree` — sections → key → value, with `set`/`get`/`has`/`sections`/`keys`/`merge_over`. The shared type every parser produces. |
| `ini.mojo` | `parse_ini(text)` / `parse_ini_file(path)` → `ConfigTree`. `[section]` headers, `key = value` **and** `key : value` (splits on the first separator), `;`/`#` full-line comments, empty values, last-key-wins. `[DEFAULT]` is a plain section; pre-header keys go to the global scope `""`. **Matches Python `configparser` byte-for-byte** on the tested doc. |
| `toml.mojo` | `parse_toml(text)` / `parse_toml_file(path)` → `ConfigTree`. `[table]` + dotted `[a.b]`, dotted keys, basic + literal strings (escapes incl. `\uXXXX`, **UTF-8 correct**), integers (`_` separators, validated), floats (exp, `inf`/`nan`), bools, arrays (mixed/nested/multi-line). Rejects TOML-invalid input (leading zeros, bad `_`, duplicate keys, Int64 overflow) and out-of-scope constructs with clear errors. **Matches Python `tomllib`** on typed values. |
| `config.mojo` | `Config` — layered front-end. Load from INI/TOML (by extension), overlay `.env` files and process env, plus explicit overrides; **precedence (later wins): defaults → file(s) → `.env` → process env → `set()`**. Typed access `get_str`/`get_int`/`get_float`/`get_bool`/`get_list`, dotted-path `get_*_path("section.key")`, `*_or(...)` defaults, `has`, `sections`. |

## Example

```mojo
from config.config import Config

var cfg = Config.from_file("app.toml")    # [server] host="0.0.0.0", port=8080
cfg.load_dotenv(".env")                    # -> section "env"
cfg.overlay_env("APP")                     # APP_SERVER_PORT=9090 overrides [server] port
var port = cfg.get_int("server", "port")   # typed; env override wins
var host = cfg.get_str_or("server", "host", String("127.0.0.1"))
```

## Verified (Python oracles, measured — not asserted)

```bash
# regenerate oracle fixtures, then run (from repo root, -I .)
python3 config/tests/ini_oracle.py   && pixi run mojo run -I . config/tests/ini_test.mojo     # 24/24 vs configparser
python3 config/tests/toml_oracle.py /tmp/toml_oracle.txt && pixi run mojo run -I . config/tests/toml_test.mojo  # 40/40 vs tomllib
pixi run mojo run -I . config/tests/value_test.mojo     # 17/17
pixi run mojo run -I . config/tests/config_test.mojo    # 31/31 (precedence layers verified)
```
Adversarial skeptic probes (`config/tests/skeptic_*.mojo`) confirm the hardened
edges: non-ASCII strings round-trip byte-exact (`café`), and TOML-invalid input
(`01`, `1__0`, `1_`, duplicate keys, Int64 overflow) plus out-of-scope constructs
(inline tables `{}`, array-of-tables `[[x]]`, multiline `"""..."""`, datetimes)
all **raise** rather than silently mis-parsing.

## Scope / limitations (honest)

- **TOML subset**: no inline tables `{}`, array-of-tables `[[x]]`, multiline strings,
  or datetimes (each raises a clear error — quote a datetime to keep it as a string).
  Hex/octal/binary int literals are not supported.
- **INI**: no value interpolation (`%(x)s`/`${x}`), no line continuations; inline
  `;`/`#` are **not** stripped (matches `configparser` default). `[DEFAULT]` is a
  plain section (not auto-merged as fallback — the layered `Config` decides policy).
- **`Config`**: `.json` loading is not supported (the `json/` tree doesn't map onto
  the flat section/key model without lossy flattening). Env mapping is
  `PREFIX_SECTION_KEY` (uppercased, `.`/`-`/space → `_`); collisions between
  `[a.b] key` and `[a] b.key` are ambiguous (documented).

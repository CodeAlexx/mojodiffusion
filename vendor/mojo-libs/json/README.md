# json — JSON for Mojo (100% Mojo, no FFI)

A complete JSON library: an ergonomic tree DOM, a high-performance flat-tape
parser, and reflection-driven validation. RFC 8259. Zero C, zero dependencies.

## Modules

| Module | What it is |
|---|---|
| `value.mojo` | `JSONValue` — the tagged-union value tree (null/bool/int/float/string/array/object), with accessors and builders. |
| `parser.mojo` | `loads(text)` → `JSONValue`. Recursive-descent; full string escapes incl. `\uXXXX` + surrogate pairs, int/float, whitespace, precise errors. |
| `serialize.mojo` | `dumps(v)` / `dumps_pretty(v)` — compact or indented; builds into a byte buffer (linear, not quadratic). |
| `tape.mojo` | High-performance parser: one flat `List[Node]` (no per-node heap alloc), strings as zero-copy source slices, path-based accessor. Floats in the common range are **correctly rounded** (exact mantissa × exact power-of-ten — bit-identical to strtod); integers beyond Int64 promote to float instead of silently wrapping. |
| `codec.mojo` | Reflection-driven validation + typed extraction for mapping JSON ↔ Mojo structs. |
| `pointer.mojo` | **JSON Pointer (RFC 6901)** — `resolve(doc, "/a/b/0")` with `~0`/`~1` unescaping; `split_pointer`. |
| `patch.mojo` | **JSON Patch (RFC 6902)** `apply_patch` (add/remove/replace/move/copy/test, functional — returns a new doc) + **JSON Merge Patch (RFC 7386)** `apply_merge_patch`. Verified against the RFC appendix vectors. |
| `stream.mojo` | **Streaming / SAX pull-parser** — `StreamParser.next_event()` walks JSON without building a tree (O(depth) memory, 512-depth limit), with `current_path()`. Parses a 100k-element array at peak nesting depth 2. |
| `ndjson.mojo` | **NDJSON / JSON Lines** — `parse_ndjson` / `write_ndjson` + a one-value-at-a-time `NdjsonReader`. |
| `schema.mojo` | **JSON Schema (Draft-07 subset)** — `validate(instance, schema)` collecting all errors with JSON-Pointer locations: type/required/properties/enum/const/min-max/multipleOf/length/pattern/items/uniqueItems/allOf-anyOf-oneOf-not. **60/60 verdicts match Python `jsonschema`.** |
| `canonical.mojo` | **Canonical / minified / sorted** output — `dumps_sorted` (recursive key sort, RFC 8785 JCS ordering), `minify`, `canonicalize(text)`. **Byte-identical to Python `json.dumps(sort_keys=True, separators=(",",":"))`** on the tested docs. |

## Tree DOM

```mojo
from json.parser import loads
from json.serialize import dumps, dumps_pretty

var v = loads(String('{"name":"Mojo","tags":["a","b"],"n":42}'))
print(v["name"].as_string())        # Mojo
print(v["tags"][1].as_string())     # b
print(v["n"].as_int())              # 42
print(dumps(v))
```

## Tape parser (high performance)

The tree DOM allocates a fat node (Dict+List+String) per value (~19 MB/s). The
tape parses into one contiguous array of small trivially-copyable nodes — strings
are `(offset,len)` into the source, materialized lazily — and pre-reserves
capacity. Result: **~800–860 MB/s on an 8 MB document** (~43× the tree, ~0.6× yyjson).

Each node stores `skip` (index past its whole subtree), so the **path accessor**
walks without recursion. Accessors live on the document (the live receiver) and
return owned values, so nothing dangles:

```mojo
from json.tape import parse

var doc = parse(text)               # keep `doc` alive while you query it
print(doc.get_str("user.name"))
print(doc.get_int("items.0.id"))    # numeric segment = array index
print(doc.length("tags"))
if doc.has("meta.author"): ...
```

How the speedup was found: a scan-only pass (validate bytes, build nothing) runs
at **~18 GB/s** — faster than yyjson. So Mojo isn't the bottleneck; per-node heap
allocation in the tree was 99.9% of the cost. The tape removes it.

## Reflection-driven codec/validation

`codec.mojo` uses Mojo's compile-time reflection (`std.reflection`) to derive
validation from a struct's fields — no schema duplication:

```mojo
from json.codec import validate_required, reject_unknown, field_list, req_int, req_str, opt_str

# generic over ANY struct T (iterates reflect[T]().field_names() at comptime):
validate_required[User](obj)        # all fields present?
reject_unknown[User](obj)           # no extra keys?
var names = field_list[User]()      # ["id","name",...] — OpenAPI building block

# typed extractors -> FastAPI-style 422 messages on missing/wrong-type:
var id = req_int(obj, "id")
var role = opt_str(obj, "role", String("user"))
```

**Limitation (Mojo 1.0.0b1):** a fully-automatic per-field-type *value* codec
(loop fields, coerce each by its own type with no per-field code) isn't
expressible — inside a `comptime for` the field type stays symbolic, so
type-specific dispatch won't resolve. Validation over field *names* is fully
generic; value binding is one `req_`/`opt_` line per field. Newer Mojo reflection
(field types usable in type position) is expected to lift this.

## Tests

```bash
pixi run mojo run -I .. json/tests/tape_test.mojo    # from repo root: -I .
pixi run mojo run -I .. json/tests/codec_test.mojo
pixi run mojo run -I .. json/tests/json_test.mojo
```
(26 tree tests, 10 codec tests, 30 tape tests — all passing.)

## Enterprise modules (verified)

```bash
pixi run mojo run -I . json/tests/pointer_patch_test.mojo   # 48/48 — RFC 6901/6902/7386 vectors
pixi run mojo run -I . json/tests/stream_test.mojo          # 16/16 — SAX events, 100k-doc, NDJSON
pixi run mojo run -I . json/tests/schema_test.mojo          # 61/61 — 60/60 match Python jsonschema
pixi run mojo run -I . json/tests/canonical_test.mojo       # 12/12 — byte-match Python sort_keys
```

- **Pointer/Patch/Merge** — every RFC appendix example, cross-checked against reference RFC implementations.
- **Streaming** — bounded O(depth) memory (does not build a tree); the source text itself is still held in memory (no byte-source/file-streaming abstraction yet).
- **Schema** — Draft-07 *subset*; `pattern` is a small regex engine (literals, `.`, `*+?`, anchors, char classes — no groups/alternation/`{m,n}`/`\d\w\s`); `$ref`/`if-then-else`/`format` not implemented.
- **Canonical** — integers + ordinary decimals match Python exactly; full ECMAScript number canonicalization (exotic floats) is not done; key sort diverges from strict RFC 8785 only for mixed astral+BMP keys.

# snapshot — pure-Mojo project snapshots

A compact, structured **index of a project directory** — built so an agent (or a
human) can grasp and track a codebase without re-reading every file, and can
diff the state of a tree over time.

100% Mojo + libc. No third-party dependencies. Filesystem via `std.os`, current
time via libc `time(2)` (`std.ffi`); everything else is pure Mojo string work.

## What a snapshot captures

- **File tree** — every file (excluding `.git`, `.pixi`, `node_modules`,
  `target`, `build`, `__pycache__`, … ), with byte size and line count.
- **Language breakdown** — files / lines / size per extension, sorted by size.
- **Symbol map** — top-level declarations per source file:
  - Mojo: `struct` / `trait` / `def` / `fn`
  - Python: `class` / `def` / `async def`
  - Rust: `fn` / `struct` / `enum` / `trait` / `impl` / `macro_rules!` (+`pub`)
  - (top-level only — indentation 0 — so it stays a clean module-level API map)
- **Content hash** — FNV-1a 64 per source file, so two snapshots can be diffed.

Binary / non-source files (and any text file over `max_text_bytes`, default
2 MiB) are recorded with size only (no line count, no hash).

## Usage

Build the CLI once (from the repo root, with `-I .`):

```sh
mojo build -I . snapshot/cli.mojo -o snapshot/snapshot
```

Or run it directly:

```sh
# print a markdown map to stdout (lands straight in a tool result)
mojo run -I . snapshot/cli.mojo <dir>

# brief = tree + language table only (skip the symbol map)
mojo run -I . snapshot/cli.mojo <dir> --brief

# write PREFIX.md (human map) + PREFIX.tsv (machine manifest)
mojo run -I . snapshot/cli.mojo <dir> -o PREFIX

# diff two manifests: added / removed / changed
mojo run -I . snapshot/cli.mojo --diff OLD.tsv NEW.tsv
```

Typical agent workflow: snapshot a project to `-o /tmp/proj`, do work, snapshot
again to `-o /tmp/proj2`, then `--diff /tmp/proj.tsv /tmp/proj2.tsv` to see
exactly what changed.

## Output

**Markdown** (stdout / `PREFIX.md`):

```
# Snapshot — json
_epoch 1781465771 · 20 files (20 source) · 4458 lines · 163.2K_

## languages
| ext | files | lines | size |
|---|--:|--:|--:|
| mojo | 19 | 4352 | 157.0K |
| md   | 1  | 106  | 6.2K  |

## files
`README.md`  6.2K  106L
`parser.mojo`  9.9K  297L
...

## symbols
### serialize.mojo
- def dumps
- def dumps_pretty
...
```

**TSV manifest** (`PREFIX.tsv`) — one line per file, easy to re-parse:

```
#snapshot	<root>	<epoch>	<nfiles>
#path	size	lines	hash	kind
README.md	6325	106	8d1e822ded6785ba	T
parser.mojo	10140	297	...	T
graphics/images/text.png	5841	-1	-	B
```

## Library API

```mojo
from snapshot import scan, default_config, render_markdown, render_tsv, diff

var cfg = default_config()        # max_text_bytes=2MiB, max_depth=64, brief=False
var snap = scan("myproject", cfg) # Snapshot{ root, epoch, files: List[FileEntry] }
print(render_markdown(snap, cfg))
write_file("snap.tsv", render_tsv(snap))
print(diff("old.tsv", "snap.tsv"))
```

Also exported: `FileEntry`, `Config`, `Snapshot`, `symbols_for`, `fnv1a_hex`,
`human_size`, `parse_manifest`.

## Tests

```sh
mojo run -I . snapshot/tests/snapshot_test.mojo
```

9 unit tests (hashing determinism, line counting, extension parsing,
classification, size formatting, symbol extraction for Mojo/Python).

# snapshot.snapshot — pure-Mojo project snapshots.
#
# A "snapshot" is a compact, structured index of a project directory:
#   * the file tree (sizes + line counts)
#   * a per-extension language breakdown
#   * a top-level symbol map (struct/trait/def/fn for Mojo, class/def for
#     Python, fn/struct/enum/trait/impl for Rust)
#   * a content hash per source file (FNV-1a 64) so two snapshots can be diffed
#
# Designed to be RUN, not just imported: `cli.mojo` prints a markdown map to
# stdout (so the snapshot lands directly in a tool result) and can write a TSV
# manifest for change-detection across time.
#
# 100% Mojo + libc (no third-party deps). Filesystem via std.os; current time
# via libc time(2) (std.ffi); everything else is pure Mojo string work.

from std.os import listdir
from std.os.path import isdir, getsize
from std.ffi import external_call

# ── FNV-1a 64-bit (content hashing) ──────────────────────────────────────────
comptime FNV_OFFSET: UInt64 = 14695981039346656037
comptime FNV_PRIME: UInt64 = 1099511628211

# ── language ids ──────────────────────────────────────────────────────────────
comptime LANG_NONE = 0
comptime LANG_MOJO = 1
comptime LANG_PY = 2
comptime LANG_RUST = 3


@fieldwise_init
struct FileEntry(ImplicitlyCopyable, Movable):
    var path: String  # relative to scan root
    var size: Int  # bytes
    var lines: Int  # -1 when not counted (non-text / too big)
    var hash: String  # 16-hex FNV-1a, or "-" when not hashed
    var is_text: Bool


@fieldwise_init
struct Config(ImplicitlyCopyable, Movable):
    var max_text_bytes: Int  # files larger than this are recorded as "other"
    var max_depth: Int
    var brief: Bool  # skip the symbol map


@fieldwise_init
struct LangStat(ImplicitlyCopyable, Movable):
    var files: Int
    var lines: Int
    var bytes: Int


@fieldwise_init
struct Snapshot(Copyable, Movable):
    var root: String
    var epoch: Int
    var files: List[FileEntry]


def default_config() -> Config:
    return Config(2 * 1024 * 1024, 64, False)


def now_epoch() -> Int:
    return external_call["time", Int](Int(0))


# ── small string helpers ─────────────────────────────────────────────────────
def to_hex16(v: UInt64) -> String:
    var chars = String("0123456789abcdef")
    var res = String("")
    var shift = 60
    while shift >= 0:
        var nib = Int((v >> UInt64(shift)) & UInt64(15))
        res += chars[byte=nib]
        shift -= 4
    return res


def fnv1a_hex(data: String) -> String:
    var bs = data.as_bytes()
    var h = FNV_OFFSET
    for i in range(len(bs)):
        h = (h ^ UInt64(bs[i])) * FNV_PRIME
    return to_hex16(h)


def count_lines(data: String) -> Int:
    return data.count("\n")


def ext_of(name: String) -> String:
    # lowercase extension after the final '.', else "".
    var parts = name.split(".")
    if len(parts) < 2:
        return String("")
    return parts[len(parts) - 1].lower()


def is_text_ext(ext: String) -> Bool:
    var t = [
        "mojo", "py", "pyi", "rs", "md", "txt", "rst", "toml", "json", "jsonl",
        "ndjson", "yaml", "yml", "ini", "cfg", "conf", "sh", "bash", "zsh", "c",
        "h", "cc", "cpp", "cxx", "hpp", "hxx", "js", "mjs", "ts", "tsx", "jsx",
        "html", "htm", "css", "scss", "go", "java", "kt", "rb", "lua", "sql",
        "csv", "tsv", "xml", "svg", "make", "mk", "cmake", "gradle", "proto",
        "graphql", "vue", "env", "gitignore", "dockerignore", "lock",
    ]
    for i in range(len(t)):
        if ext == t[i]:
            return True
    return False


def is_text_name(name: String) -> Bool:
    var t = [
        "README", "LICENSE", "Makefile", "Dockerfile", "CHANGELOG", "TODO",
        "NOTICE", "AUTHORS", "COPYING", "MANIFEST",
    ]
    for i in range(len(t)):
        if name == t[i]:
            return True
    return False


def is_excluded_dir(name: String) -> Bool:
    var t = [
        ".git", ".pixi", ".magic", ".venv", "venv", "node_modules", "target",
        "build", "dist", "__pycache__", ".mypy_cache", ".pytest_cache",
        ".ruff_cache", ".cache", ".idea", ".gradle", ".next", ".turbo",
        "site-packages", ".eggs",
    ]
    for i in range(len(t)):
        if name == t[i]:
            return True
    return False


def lang_of(ext: String) -> Int:
    if ext == "mojo":
        return LANG_MOJO
    if ext == "py" or ext == "pyi":
        return LANG_PY
    if ext == "rs":
        return LANG_RUST
    return LANG_NONE


# ── top-level symbol extraction (cheap line-prefix scan, no real parser) ──────
def _is_ident_byte(c: UInt8) -> Bool:
    var i = Int(c)
    return (
        (i >= ord("a") and i <= ord("z"))
        or (i >= ord("A") and i <= ord("Z"))
        or (i >= ord("0") and i <= ord("9"))
        or i == ord("_")
    )


def _ident_after(line: String, start: Int) -> String:
    var bs = line.as_bytes()
    var n = len(bs)
    var i = start
    while i < n and Int(bs[i]) == ord(" "):
        i += 1
    var s = i
    while i < n and _is_ident_byte(bs[i]):
        i += 1
    if i == s:
        return String("")
    return String(line[byte=s:i])


def _match_kw(line: String, kws: List[String]) -> Int:
    # returns the length of the matched keyword, else -1. kws longest-first.
    for i in range(len(kws)):
        if line.startswith(kws[i]):
            return kws[i].byte_length()
    return -1


def _kws_for(lang: Int) -> List[String]:
    if lang == LANG_MOJO:
        return ["struct ", "trait ", "def ", "fn "]
    if lang == LANG_PY:
        return ["async def ", "class ", "def "]
    if lang == LANG_RUST:
        return [
            "pub struct ", "pub enum ", "pub trait ", "pub fn ", "pub async fn ",
            "struct ", "enum ", "trait ", "impl ", "fn ", "async fn ",
            "macro_rules! ",
        ]
    return List[String]()


def symbols_for(data: String, ext: String) -> List[String]:
    var syms = List[String]()
    var lang = lang_of(ext)
    if lang == LANG_NONE:
        return syms^
    var kws = _kws_for(lang)
    var lines = data.split("\n")
    for li in range(len(lines)):
        var line = String(lines[li])
        if line.byte_length() == 0:
            continue
        var b0 = Int(line.as_bytes()[0])
        if b0 == ord(" ") or b0 == ord("\t"):
            continue  # top-level declarations only
        var kwlen = _match_kw(line, kws)
        if kwlen < 0:
            continue
        var name = _ident_after(line, kwlen)
        if name.byte_length() == 0:
            continue
        # label = the keyword (trimmed) for kind context
        var kind = String(line[byte=0:kwlen])
        syms.append(String(kind + name))
    return syms^


# ── directory walk ───────────────────────────────────────────────────────────
def _walk(
    root: String, rel: String, depth: Int, mut files: List[FileEntry], cfg: Config
) raises:
    if depth > cfg.max_depth:
        return
    var dirpath = root if rel.byte_length() == 0 else String(root + "/" + rel)
    var names = listdir(dirpath)
    for ni in range(len(names)):
        var name = names[ni]
        var full = String(dirpath + "/" + name)
        var childrel = name if rel.byte_length() == 0 else String(rel + "/" + name)
        if isdir(full):
            if is_excluded_dir(name):
                continue
            _walk(root, childrel, depth + 1, files, cfg)
        else:
            var ext = ext_of(name)
            var size = getsize(full)
            var text = is_text_ext(ext) or is_text_name(name)
            if text and size <= cfg.max_text_bytes:
                var f = open(full, "r")
                var data = f.read()
                f.close()
                files.append(
                    FileEntry(
                        childrel, size, count_lines(data), fnv1a_hex(data), True
                    )
                )
            else:
                files.append(FileEntry(childrel, size, -1, String("-"), False))


def _sort_entries(mut entries: List[FileEntry]):
    # insertion sort by path (small N; deterministic output for diff-friendliness)
    var n = len(entries)
    for i in range(1, n):
        var j = i
        while j > 0 and entries[j - 1].path > entries[j].path:
            var tmp = entries[j].copy()
            entries[j] = entries[j - 1].copy()
            entries[j - 1] = tmp^
            j -= 1


def scan(root: String, cfg: Config) raises -> Snapshot:
    var files = List[FileEntry]()
    _walk(root, String(""), 0, files, cfg)
    _sort_entries(files)
    return Snapshot(root, now_epoch(), files^)


# ── rendering ────────────────────────────────────────────────────────────────
def _fmt1(value: Float64) -> String:
    var x10 = Int(value * 10.0 + 0.5)
    return String(String(x10 // 10) + "." + String(x10 % 10))


def human_size(n: Int) -> String:
    if n < 1024:
        return String(String(n) + "B")
    var kb = Float64(n) / 1024.0
    if kb < 1024.0:
        return String(_fmt1(kb) + "K")
    var mb = kb / 1024.0
    if mb < 1024.0:
        return String(_fmt1(mb) + "M")
    var gb = mb / 1024.0
    return String(_fmt1(gb) + "G")


def render_markdown(snap: Snapshot, cfg: Config) raises -> String:
    var total_lines = 0
    var total_bytes = 0
    var text_files = 0
    var stats = Dict[String, LangStat]()
    for i in range(len(snap.files)):
        var e = snap.files[i]
        total_bytes += e.size
        if e.is_text:
            text_files += 1
            if e.lines > 0:
                total_lines += e.lines
        var key = ext_of(e.path)
        if key.byte_length() == 0:
            key = String("(noext)")
        var ln = e.lines if e.lines > 0 else 0
        if key in stats:
            var s = stats[key]
            stats[key] = LangStat(s.files + 1, s.lines + ln, s.bytes + e.size)
        else:
            stats[key] = LangStat(1, ln, e.size)

    var out = String("")
    out += "# Snapshot — " + snap.root + "\n"
    out += (
        "_epoch "
        + String(snap.epoch)
        + " · "
        + String(len(snap.files))
        + " files ("
        + String(text_files)
        + " source) · "
        + String(total_lines)
        + " lines · "
        + human_size(total_bytes)
        + "_\n\n"
    )

    # language table, sorted by bytes desc
    var keys = List[String]()
    for entry in stats.items():
        keys.append(entry.key)
    for i in range(1, len(keys)):
        var j = i
        while j > 0 and stats[keys[j - 1]].bytes < stats[keys[j]].bytes:
            var tmp = keys[j]
            keys[j] = keys[j - 1]
            keys[j - 1] = tmp
            j -= 1
    out += "## languages\n\n"
    out += "| ext | files | lines | size |\n|---|--:|--:|--:|\n"
    for i in range(len(keys)):
        var k = keys[i]
        var s = stats[k]
        out += (
            "| "
            + k
            + " | "
            + String(s.files)
            + " | "
            + String(s.lines)
            + " | "
            + human_size(s.bytes)
            + " |\n"
        )
    out += "\n"

    # file tree
    out += "## files\n\n"
    for i in range(len(snap.files)):
        var e = snap.files[i]
        out += "`" + e.path + "`  " + human_size(e.size)
        if e.is_text:
            out += "  " + String(e.lines) + "L"
        out += "\n"
    out += "\n"

    if cfg.brief:
        return out

    # symbol map
    out += "## symbols\n\n"
    for i in range(len(snap.files)):
        var e = snap.files[i]
        if not e.is_text:
            continue
        var ext = ext_of(e.path)
        if lang_of(ext) == LANG_NONE:
            continue
        var f = open(String(snap.root + "/" + e.path), "r")
        var data = f.read()
        f.close()
        var syms = symbols_for(data, ext)
        if len(syms) == 0:
            continue
        out += "### " + e.path + "\n"
        for s in range(len(syms)):
            out += "- " + syms[s] + "\n"
        out += "\n"
    return out


def render_tsv(snap: Snapshot) -> String:
    # machine manifest for diffing. header + one line/file.
    var out = String("")
    out += (
        "#snapshot\t"
        + snap.root
        + "\t"
        + String(snap.epoch)
        + "\t"
        + String(len(snap.files))
        + "\n"
    )
    out += "#path\tsize\tlines\thash\tkind\n"
    for i in range(len(snap.files)):
        var e = snap.files[i]
        var kind = String("T") if e.is_text else String("B")
        out += (
            e.path
            + "\t"
            + String(e.size)
            + "\t"
            + String(e.lines)
            + "\t"
            + e.hash
            + "\t"
            + kind
            + "\n"
        )
    return out


def write_file(path: String, content: String) raises:
    with open(path, "w") as f:
        f.write(content)

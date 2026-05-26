# sharded.mojo — ShardedSafeTensors: a multi-shard safetensors loader.
#
# Chunk 2 of the serenity-safetensors -> Mojo port (BUILD_PLAN.md row
# "sharded.mojo"; Rust ref: serenity-safetensors src/diffusers.rs weight_map).
# Linux x86-64, Mojo 1.0.0b1. Pure Mojo, no Python in the runtime path.
#
# open(dir):
#   * If `dir` contains an index file
#     (`diffusion_pytorch_model.safetensors.index.json` or
#      `model.safetensors.index.json`), parse its "weight_map" (tensor-name ->
#     shard-filename), open each UNIQUE shard file as a chunk-1 `SafeTensors`,
#     and build a unified tensor-name -> shard-index map.
#   * Else, fall back to the single `*.safetensors` in `dir` as one shard
#     (probed by the known diffusers filename, then `model.safetensors`).
#
# Index JSON schema (DIFFERENT from the safetensors header schema):
#   {"metadata":{...}, "weight_map":{"<tensor>":"<shard-file>", ...}}
# i.e. weight_map is a flat string->string map. The chunk-1 json_header parser
# is shaped for the tensor-header schema (values are objects with
# dtype/shape/data_offsets), so it does NOT fit here. We write a SMALL DEDICATED
# parser (`_parse_weight_map`) for the string->string map instead of bending
# json_header — the two schemas are genuinely different and a dedicated 60-line
# scanner is clearer and lower-risk than overloading the header parser. FLAGGED
# per task: dedicated parser, reason = schema mismatch.
#
# Ownership / lifetime contract (HEADLINE RISK):
#   Shards are stored as `List[ArcPointer[SafeTensors]]`. `SafeTensors` is
#   Movable-but-not-Copyable (it uniquely owns its MmapRegion), and Mojo
#   1.0.0b1's `List[T]` REQUIRES `T: Copyable`, so a bare `List[SafeTensors]`
#   (and `List[OwnedPointer[SafeTensors]]`) do NOT compile. `ArcPointer` is
#   Copyable (copy == refcount bump), so `List[ArcPointer[SafeTensors]]` works
#   and each shard's mmap stays alive as long as the Arc is held.
#
#   Accessors return a `Span` / `TensorView` whose origin is
#   `origin_of(self.shards)` — a field of `self`. That ties the view's lifetime
#   to the whole `ShardedSafeTensors`: the compiler keeps `self` (and thus the
#   owning shard's mmap) alive while the view is used. No bare pointer, no
#   `unsafe_origin_cast`.
#
#   Scope of the compile-time guarantee (chunk-2 skeptic F2 — accurate claim):
#   the origin binding REJECTS (compile error, verified) the two common
#   footguns — escape-returning a view past `self`, and explicitly destroying
#   `self` (`self^.__del__()`) while a view is live. It does NOT catch
#   *reassigning/overwriting* the source binding while a view borrows it
#   (`var v = sh.tensor_bytes(n); sh = ShardedSafeTensors.open(other)` compiles
#   and yields a use-after-free) — a Mojo 1.0.0b1 origin-tracking limitation.
#   Not reassigning the source while a view is live is the caller's contract.

from std.memory import ArcPointer
from serenitymojo.io.safetensors import SafeTensors, TensorRef
from serenitymojo.io.tensor_view import TensorView, from_parts
from serenitymojo.io.ffi import sys_open, sys_close, O_RDONLY


comptime MAX_INDEX_LEN = 256 * 1024 * 1024  # generous cap for an index file


def _index_names() -> List[String]:
    """Candidate index filenames, in priority order (diffusers, then plain)."""
    var out = List[String]()
    out.append(String("diffusion_pytorch_model.safetensors.index.json"))
    out.append(String("model.safetensors.index.json"))
    return out^


def _single_names() -> List[String]:
    """Candidate single-file safetensors names (no-index fallback)."""
    var out = List[String]()
    out.append(String("diffusion_pytorch_model.safetensors"))
    out.append(String("model.safetensors"))
    return out^


# ── small filesystem helpers (FFI; no std.os dependency) ──────────────────────


def _path_exists(path: String) -> Bool:
    """True if `path` can be opened O_RDONLY. Uses the chunk-1 ffi open/close —
    avoids depending on an uncertain std.os API surface."""
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        return False
    _ = sys_close(fd)
    return True


def _join(dir: String, name: String) -> String:
    """Join a directory and filename with a single '/'."""
    if dir.byte_length() == 0:
        return name
    if dir[byte = dir.byte_length() - 1] == "/":
        return dir + name
    return dir + "/" + name


def _looks_safetensors_file(path: String) -> Bool:
    var suffix = String(".safetensors")
    if path.byte_length() < suffix.byte_length():
        return False
    var off = path.byte_length() - suffix.byte_length()
    for i in range(suffix.byte_length()):
        if path[byte = off + i] != suffix[byte = i]:
            return False
    return True


def _read_file_bytes(path: String) raises -> List[UInt8]:
    """Read an entire (small) file into a List[UInt8] via chunk-1 ffi pread.
    Used for the index JSON only (bounded by MAX_INDEX_LEN)."""
    from serenitymojo.io.ffi import sys_pread, file_size, BytePtr
    from std.memory import alloc, UnsafePointer

    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("failed to open index: ") + path)
    var n = file_size(fd)
    if n <= 0:
        _ = sys_close(fd)
        raise Error(String("empty or unreadable index: ") + path)
    if n > MAX_INDEX_LEN:
        _ = sys_close(fd)
        raise Error("index file too large (>256MB)")
    var buf = alloc[UInt8](n)
    var done = 0
    while done < n:
        var got = sys_pread(fd, buf + done, n - done, done)
        if got < 0:
            buf.free()
            _ = sys_close(fd)
            raise Error("pread failed reading index")
        if got == 0:
            break
        done += got
    _ = sys_close(fd)
    var out = List[UInt8]()
    for i in range(done):
        out.append(buf[i])
    buf.free()
    return out^


# ── dedicated weight_map parser (string -> string) ────────────────────────────
#
# We scan for the top-level "weight_map" key and parse its object as a flat
# string->string map. Other top-level keys ("metadata", ...) are skipped by
# balanced-brace scanning. This reuses the same byte-scanning discipline as the
# chunk-1 header parser (quote-delimited strings with escapes, balanced
# containers) but for a different VALUE shape (string, not tensor-object).


struct _Scan:
    var data: List[UInt8]
    var pos: Int
    var n: Int

    def __init__(out self, var data: List[UInt8]):
        self.n = len(data)
        self.data = data^
        self.pos = 0

    def at_end(self) -> Bool:
        return self.pos >= self.n

    def peek(self) -> Int:
        if self.pos >= self.n:
            return -1
        return Int(self.data[self.pos])

    def skip_ws(mut self):
        while self.pos < self.n:
            var c = Int(self.data[self.pos])
            if c == 0x20 or c == 0x09 or c == 0x0A or c == 0x0D:
                self.pos += 1
            else:
                break

    def expect(mut self, ch: Int) raises:
        self.skip_ws()
        if self.peek() != ch:
            raise Error(
                String("index JSON: expected byte ")
                + String(ch)
                + " at "
                + String(self.pos)
            )
        self.pos += 1


def _scan_string(mut s: _Scan) raises -> String:
    """Parse a JSON string at the current position (must be at the opening
    quote). Handles the standard escapes. Mirrors json_header._parse_string but
    local to this parser."""
    s.skip_ws()
    if s.peek() != 0x22:
        raise Error(
            String("index JSON: expected string at ") + String(s.pos)
        )
    s.pos += 1
    var out = List[UInt8]()
    while not s.at_end():
        var c = Int(s.data[s.pos])
        s.pos += 1
        if c == 0x22:  # closing quote
            return String(from_utf8=out)
        if c == 0x5C:  # backslash
            if s.at_end():
                raise Error("index JSON: dangling escape")
            var e = Int(s.data[s.pos])
            s.pos += 1
            if e == 0x22:
                out.append(0x22)
            elif e == 0x5C:
                out.append(0x5C)
            elif e == 0x2F:
                out.append(0x2F)
            elif e == 0x6E:
                out.append(0x0A)
            elif e == 0x74:
                out.append(0x09)
            elif e == 0x72:
                out.append(0x0D)
            elif e == 0x62:
                out.append(0x08)
            elif e == 0x66:
                out.append(0x0C)
            elif e == 0x75:  # \uXXXX (BMP)
                if s.pos + 4 > s.n:
                    raise Error("index JSON: truncated \\u escape")
                var cp = 0
                for _i in range(4):
                    var h = Int(s.data[s.pos])
                    var hv: Int
                    if h >= 0x30 and h <= 0x39:
                        hv = h - 0x30
                    elif h >= 0x41 and h <= 0x46:
                        hv = h - 0x41 + 10
                    elif h >= 0x61 and h <= 0x66:
                        hv = h - 0x61 + 10
                    else:
                        raise Error("index JSON: bad hex digit")
                    cp = cp * 16 + hv
                    s.pos += 1
                if cp < 0x80:
                    out.append(UInt8(cp))
                elif cp < 0x800:
                    out.append(UInt8(0xC0 | (cp >> 6)))
                    out.append(UInt8(0x80 | (cp & 0x3F)))
                else:
                    out.append(UInt8(0xE0 | (cp >> 12)))
                    out.append(UInt8(0x80 | ((cp >> 6) & 0x3F)))
                    out.append(UInt8(0x80 | (cp & 0x3F)))
            else:
                raise Error("index JSON: bad escape")
        else:
            out.append(UInt8(c))
    raise Error("index JSON: unterminated string")


def _skip_value(mut s: _Scan) raises:
    """Skip an arbitrary JSON value (object/array/string/number/literal) by
    balanced scanning. Used for top-level keys other than weight_map."""
    s.skip_ws()
    var c = s.peek()
    if c == 0x22:
        _ = _scan_string(s)
        return
    if c == 0x7B or c == 0x5B:  # object or array
        var open_ch = c
        var close_ch = 0x7D if c == 0x7B else 0x5D
        var depth = 0
        while not s.at_end():
            var ch = Int(s.data[s.pos])
            if ch == 0x22:
                _ = _scan_string(s)
                continue
            if ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                s.pos += 1
                if depth == 0:
                    return
                continue
            s.pos += 1
        raise Error("index JSON: unbalanced container")
    # number / true / false / null
    while not s.at_end():
        var ch = Int(s.data[s.pos])
        if (
            ch == 0x2C
            or ch == 0x7D
            or ch == 0x5D
            or ch == 0x20
            or ch == 0x09
            or ch == 0x0A
            or ch == 0x0D
        ):
            break
        s.pos += 1


def _parse_weight_map(
    var data: List[UInt8],
) raises -> Dict[String, String]:
    """Parse an index JSON, returning the "weight_map" as tensor-name ->
    shard-filename. Top-level keys other than "weight_map" are skipped."""
    var s = _Scan(data^)
    var out = Dict[String, String]()
    s.expect(0x7B)  # top-level '{'
    s.skip_ws()
    if s.peek() == 0x7D:  # empty object
        s.pos += 1
        raise Error("index JSON: no weight_map")
    var found = False
    while True:
        var key = _scan_string(s)
        s.expect(0x3A)  # ':'
        if key == "weight_map":
            found = True
            # parse the weight_map object: { "<name>": "<file>", ... }
            s.expect(0x7B)
            s.skip_ws()
            if s.peek() != 0x7D:  # non-empty
                while True:
                    var tname = _scan_string(s)
                    s.expect(0x3A)
                    var fname = _scan_string(s)
                    out[tname] = fname
                    s.skip_ws()
                    var cc = s.peek()
                    if cc == 0x2C:
                        s.pos += 1
                        continue
                    if cc == 0x7D:
                        break
                    raise Error(
                        String("index JSON: expected ',' or '}' in weight_map"
                        " at ") + String(s.pos)
                    )
            s.expect(0x7D)  # close weight_map
        else:
            _skip_value(s)
        s.skip_ws()
        var c = s.peek()
        if c == 0x2C:
            s.pos += 1
            continue
        if c == 0x7D:
            s.pos += 1
            break
        raise Error(
            String("index JSON: expected ',' or '}' at top level at ")
            + String(s.pos)
        )
    if not found:
        raise Error("index JSON: no weight_map key")
    return out^


# ── ShardedSafeTensors ────────────────────────────────────────────────────────


struct ShardedSafeTensors(Movable):
    """A multi-shard safetensors collection. Owns N `SafeTensors` (each an
    mmap'd shard) and a unified tensor-name -> shard-index map. Single-file
    inputs are represented as a 1-shard collection.

    Lifetime: accessors return origin-bound views tied to `self.shards`, so the
    whole collection (and the owning shard's mmap) is kept alive while a view is
    used. See module header HEADLINE RISK."""

    var shards: List[ArcPointer[SafeTensors]]
    var name_to_shard: Dict[String, Int]  # tensor-name -> index into shards

    def __init__(
        out self,
        var shards: List[ArcPointer[SafeTensors]],
        var name_to_shard: Dict[String, Int],
    ):
        self.shards = shards^
        self.name_to_shard = name_to_shard^

    @staticmethod
    def open(dir: String) raises -> ShardedSafeTensors:
        """Open a directory of shards. Detects an index file and parses its
        weight_map; otherwise falls back to a single `*.safetensors`."""
        # Direct single-file input. This keeps call sites simple for local files
        # such as `.serenity/models/vaes/flux2-vae.safetensors`.
        if _looks_safetensors_file(dir) and _path_exists(dir):
            var direct = SafeTensors.open(dir)
            var direct_shards = List[ArcPointer[SafeTensors]]()
            var direct_map = Dict[String, Int]()
            for ref nm in direct.names():
                direct_map[nm] = 0
            direct_shards.append(ArcPointer(direct^))
            return ShardedSafeTensors(direct_shards^, direct_map^)

        # 1) Look for an index file.
        var index_path = String("")
        for ref nm in _index_names():
            var p = _join(dir, nm)
            if _path_exists(p):
                index_path = p
                break

        var shards = List[ArcPointer[SafeTensors]]()
        var name_to_shard = Dict[String, Int]()

        if index_path.byte_length() > 0:
            # Indexed (sharded) path.
            var raw = _read_file_bytes(index_path)
            var wmap = _parse_weight_map(raw^)
            if len(wmap) == 0:
                raise Error(
                    String("index JSON has empty weight_map (0 tensors): ")
                    + index_path
                )

            # Open each UNIQUE shard file once; map filename -> shard index.
            var file_to_idx = Dict[String, Int]()
            for ref e in wmap.items():
                var tensor = e.key
                var fname = e.value
                if fname not in file_to_idx:
                    var shard_path = _join(dir, fname)
                    var st = SafeTensors.open(shard_path)
                    var idx = len(shards)
                    shards.append(ArcPointer(st^))
                    file_to_idx[fname] = idx
                name_to_shard[tensor] = file_to_idx[fname]
            return ShardedSafeTensors(shards^, name_to_shard^)

        # 2) No index: single-file fallback.
        var single_path = String("")
        for ref nm in _single_names():
            var p = _join(dir, nm)
            if _path_exists(p):
                single_path = p
                break
        if single_path.byte_length() == 0:
            raise Error(
                String("no index and no known single-file safetensors in ")
                + dir
            )
        var st = SafeTensors.open(single_path)
        # Map every tensor in the single shard to index 0.
        for ref nm in st.names():
            name_to_shard[nm] = 0
        shards.append(ArcPointer(st^))
        return ShardedSafeTensors(shards^, name_to_shard^)

    def num_shards(self) -> Int:
        return len(self.shards)

    def num_tensors(self) -> Int:
        """Total number of tensors across all shards (== weight_map size)."""
        return len(self.name_to_shard)

    def names(self) -> List[String]:
        """All tensor names (order unspecified)."""
        var out = List[String]()
        for ref e in self.name_to_shard.items():
            out.append(e.key)
        return out^

    def shard_index(self, name: String) raises -> Int:
        """The shard index that owns `name`."""
        if name not in self.name_to_shard:
            raise Error(String("Tensor '") + name + "' not found")
        return self.name_to_shard[name]

    def tensor_info(self, name: String) raises -> TensorRef:
        """(offset, size, dtype, shape) for `name`, from its owning shard."""
        var idx = self.shard_index(name)
        return self.shards[idx][].tensor_info(name)

    def tensor_bytes(
        self, name: String
    ) raises -> Span[UInt8, origin_of(self.shards)]:
        """Origin-bound view of `name`'s raw bytes in its owning shard.

        The returned `Span` carries `origin_of(self.shards)` — a field of
        `self` — so the compiler keeps the whole `ShardedSafeTensors` (and the
        owning shard's mmap) alive while the Span is used, and rejects letting
        it escape past `self` or outlive an explicit `self^.__del__()`. It does
        NOT catch reassigning the source binding while the Span is live (see
        module header F2); that is the caller's contract. No bare pointer; no
        unsafe cast. The inner
        `SafeTensors.tensor_bytes` returns `Span[UInt8, origin_of(shard)]`, and
        because `shard` is reached *through* `self.shards`, that origin is a
        sub-origin of `self.shards` and unifies with the declared return
        origin (verified clean — see chunk-2 report SKEPTIC-BAIT)."""
        var idx = self.shard_index(name)
        ref shard = self.shards[idx][]
        return shard.tensor_bytes(name)

    def tensor_view(
        self, name: String
    ) raises -> TensorView[origin_of(self.shards)]:
        """Origin-bound `TensorView` (dtype + shape + byte span) for `name`.
        Same lifetime contract as `tensor_bytes`. The origin is inferred from
        the span at the call site via `from_parts`, sidestepping the Mojo
        named-origin unification quirk (see tensor_view.mojo)."""
        var idx = self.shard_index(name)
        ref shard = self.shards[idx][]
        var info = shard.tensor_info(name)
        var bytes = shard.tensor_bytes(name)
        return from_parts(info.dtype, info.shape.copy(), bytes)


# ── Test: open the VAE *directory* (single-file fallback path) ────────────────
def test_single_file_fallback() raises:
    comptime VAE_DIR = (
        "/home/alex/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image/"
        "snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021/vae"
    )
    var s = ShardedSafeTensors.open(String(VAE_DIR))
    print("VAE dir (single-file fallback):")
    print("  num_shards =", s.num_shards())
    print("  num_tensors =", s.num_tensors())
    var b = s.tensor_bytes(String("decoder.conv_in.bias"))
    print("  decoder.conv_in.bias first_byte =", Int(b[0]))
    _ = s.num_shards()


def main() raises:
    test_single_file_fallback()

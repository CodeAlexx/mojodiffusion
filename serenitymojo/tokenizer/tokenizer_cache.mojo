# serenitymojo/tokenizer/tokenizer_cache.mojo
#
# Binary sidecar for `Qwen3Tokenizer`, so a process start does not re-parse a
# 7 MB tokenizer.json.
#
# MEASURED on the MiniMax-H3 conditioner's tokenizer.json (7,032,403 bytes,
# 151643 vocab entries, 151387 merges), -O2:
#
#     construct from JSON        0.180 s
#       of which raw file read   0.0085 s
#       of which Dict fill floor 0.0251 s   <- irreducible, measured separately
#       of which JSON parsing    ~0.146 s   <- what this file removes
#
# So the ceiling on this optimization is about 5x, not 100x: filling two
# 151k-entry Dicts costs 25 ms no matter where the bytes came from. Beating
# THAT needs a different lookup structure, not a different file format, and is
# not attempted here.
#
# Why this matters at all: encoding a typical prompt takes 0.074 ms. Construction
# is ~2400x an encode, so for a per-request process the tokenizer's cost is
# entirely its constructor.
#
# STALENESS: the header stores the source's byte length AND an FNV-1a 64 hash of
# its full contents, and `qwen3_tokenizer_load_cached` verifies both before
# trusting the cache. That costs the 8.5 ms source read, which is why the win is
# ~5x rather than ~30x. The alternative — trusting size and mtime — would be
# faster and would silently return wrong token ids the day a tokenizer.json is
# swapped for a same-sized one. Wrong ids mean wrong conditioning, which does not
# crash and does not look obviously wrong. Not a trade worth 8 ms.
#
# FORMAT (little-endian, all counts u64):
#   magic   8 bytes  "MJQ3TOK1"
#   version u64      1
#   source_size u64
#   source_fnv  u64
#   vocab_count u64
#   merge_count u64
#   special_count u64
#   pre_o200k u64    (0 or 1)
#   then vocab_count  x [u32 key_len][key bytes][i64 id]
#   then merge_count  x [u32 key_len][key bytes][i64 rank]
#   then special_count x [u32 key_len][key bytes][i64 id]

from std.collections import List
from std.memory import alloc

from serenitymojo.io.ffi import (
    BytePtr,
    O_CREAT,
    O_RDONLY,
    O_TRUNC,
    O_WRONLY,
    file_size,
    sys_close,
    sys_open,
    sys_pread,
    sys_pwrite,
    sys_rename,
)
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer, _read_utf8_file

comptime _MAGIC = "MJQ3TOK1"
comptime _VERSION = 1
comptime _HEADER_BYTES = 8 + 7 * 8


def qwen3_fnv1a64(bytes: Span[Byte, _]) -> UInt64:
    """FNV-1a over the whole buffer — the cache's content check."""
    var hash = UInt64(0xCBF29CE484222325)
    for i in range(len(bytes)):
        hash = hash ^ UInt64(bytes[i])
        hash = hash * UInt64(0x100000001B3)
    return hash


def _put_u64(mut out: List[UInt8], value: UInt64):
    var v = value
    for _ in range(8):
        out.append(UInt8(v & 0xFF))
        v = v >> 8


def _put_u32(mut out: List[UInt8], value: UInt32):
    var v = value
    for _ in range(4):
        out.append(UInt8(v & 0xFF))
        v = v >> 8


def _get_u64(bytes: List[UInt8], offset: Int) -> UInt64:
    var v = UInt64(0)
    for i in range(8):
        v = v | (UInt64(bytes[offset + i]) << UInt64(8 * i))
    return v


def _get_u32(bytes: List[UInt8], offset: Int) -> UInt32:
    var v = UInt32(0)
    for i in range(4):
        v = v | (UInt32(bytes[offset + i]) << UInt32(8 * i))
    return v


def _put_entry(mut out: List[UInt8], key: String, value: Int):
    var key_bytes = key.as_bytes()
    _put_u32(out, UInt32(len(key_bytes)))
    for i in range(len(key_bytes)):
        out.append(key_bytes[i])
    _put_u64(out, UInt64(value))


def qwen3_tokenizer_cache_write(
    ref tokenizer: Qwen3Tokenizer,
    source_path: String,
    cache_path: String,
) raises:
    """Serialize a built tokenizer next to its source.

    Written to `cache_path + ".tmp"` and renamed, so a crash mid-write cannot
    leave a half-file that later loads as valid."""
    var source = _read_utf8_file(source_path)
    var source_bytes = source.as_bytes()

    var out = List[UInt8]()
    var magic = String(_MAGIC).as_bytes()
    for i in range(len(magic)):
        out.append(magic[i])
    _put_u64(out, UInt64(_VERSION))
    _put_u64(out, UInt64(len(source_bytes)))
    _put_u64(out, qwen3_fnv1a64(source_bytes))
    _put_u64(out, UInt64(len(tokenizer.vocab)))
    _put_u64(out, UInt64(len(tokenizer.merge_rank)))
    _put_u64(out, UInt64(len(tokenizer.special_tokens)))
    _put_u64(out, UInt64(1) if tokenizer.pre_o200k else UInt64(0))

    for entry in tokenizer.vocab.items():
        _put_entry(out, entry.key, entry.value)
    for entry in tokenizer.merge_rank.items():
        _put_entry(out, entry.key, entry.value)
    for i in range(len(tokenizer.special_tokens)):
        _put_entry(out, tokenizer.special_tokens[i], tokenizer.special_ids[i])

    var tmp_path = cache_path + ".tmp"
    var fd = sys_open(tmp_path, O_WRONLY | O_CREAT | O_TRUNC, 0o644)
    if fd < 0:
        raise Error(String("tokenizer cache: cannot open for write: ") + tmp_path)
    # As in `_read_all`: the syscall wrapper needs a MutExternalOrigin pointer.
    var raw = alloc[UInt8](len(out))
    for i in range(len(out)):
        raw[i] = out[i]
    var written = sys_pwrite(fd, raw, len(out), 0)
    raw.free()
    _ = sys_close(fd)
    if written != len(out):
        raise Error("tokenizer cache: short write")
    if sys_rename(tmp_path, cache_path) != 0:
        raise Error(String("tokenizer cache: rename failed: ") + cache_path)


def _read_all(path: String) raises -> List[UInt8]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("tokenizer cache: cannot open: ") + path)
    var size = file_size(fd)
    # `sys_pread` wants a MutExternalOrigin pointer, which is what `alloc`
    # gives; a List's own pointer carries the List's origin and is rejected.
    # Same shape as `_read_utf8_file` in tokenizer.mojo.
    var raw = alloc[UInt8](size)
    var done = 0
    while done < size:
        var got = sys_pread(fd, raw + done, size - done, done)
        if got <= 0:
            break
        done += got
    _ = sys_close(fd)
    if done != size:
        raw.free()
        raise Error(String("tokenizer cache: short read: ") + path)
    var buffer = List[UInt8](capacity=size)
    for i in range(size):
        buffer.append(raw[i])
    raw.free()
    return buffer^


def qwen3_tokenizer_load_cached(
    source_path: String, cache_path: String
) raises -> Qwen3Tokenizer:
    """Load from the sidecar, verifying it against the source's contents.

    Raises if the cache is missing, truncated, a different version, or does not
    match the source — every one of which means "rebuild from JSON", never
    "use it anyway"."""
    var bytes = _read_all(cache_path)
    if len(bytes) < _HEADER_BYTES:
        raise Error("tokenizer cache: truncated header")

    var magic = String(_MAGIC).as_bytes()
    for i in range(len(magic)):
        if bytes[i] != magic[i]:
            raise Error("tokenizer cache: bad magic")
    if _get_u64(bytes, 8) != UInt64(_VERSION):
        raise Error("tokenizer cache: version mismatch")

    var source = _read_utf8_file(source_path)
    var source_bytes = source.as_bytes()
    if _get_u64(bytes, 16) != UInt64(len(source_bytes)):
        raise Error("tokenizer cache: source size changed")
    if _get_u64(bytes, 24) != qwen3_fnv1a64(source_bytes):
        raise Error("tokenizer cache: source contents changed")

    var vocab_count = Int(_get_u64(bytes, 32))
    var merge_count = Int(_get_u64(bytes, 40))
    var special_count = Int(_get_u64(bytes, 48))
    var pre_o200k = _get_u64(bytes, 56) == UInt64(1)

    var offset = _HEADER_BYTES
    var vocab = Dict[String, Int]()
    var merge_rank = Dict[String, Int]()
    var special_tokens = List[String]()
    var special_ids = List[Int]()

    for section in range(3):
        var count = vocab_count
        if section == 1:
            count = merge_count
        elif section == 2:
            count = special_count
        for _ in range(count):
            if offset + 4 > len(bytes):
                raise Error("tokenizer cache: truncated entry")
            var key_len = Int(_get_u32(bytes, offset))
            offset += 4
            if offset + key_len + 8 > len(bytes):
                raise Error("tokenizer cache: truncated entry")
            var key = String(
                StringSlice(unsafe_from_utf8=Span(bytes)[offset : offset + key_len])
            )
            offset += key_len
            var value = Int(_get_u64(bytes, offset))
            offset += 8
            if section == 0:
                vocab[key] = value
            elif section == 1:
                merge_rank[key] = value
            else:
                special_tokens.append(key)
                special_ids.append(value)

    return Qwen3Tokenizer(
        vocab^, merge_rank^, special_tokens^, special_ids^, pre_o200k
    )


def qwen3_tokenizer_open(
    source_path: String, cache_path: String, pre_o200k: Bool = False
) raises -> Qwen3Tokenizer:
    """Load from the cache when it is valid, otherwise parse the JSON and write
    one. The call a runtime should make. `pre_o200k` only matters on the miss
    path (the cache stores and restores the flag itself); pass the same value
    the direct constructor would get."""
    try:
        return qwen3_tokenizer_load_cached(source_path, cache_path)
    except:
        var tokenizer = Qwen3Tokenizer(source_path, pre_o200k)
        qwen3_tokenizer_cache_write(tokenizer, source_path, cache_path)
        return tokenizer^

# sharded_full_parity.mojo — SKEPTIC chunk-2 full byte-parity (2026-05-25, fresh).
#
# For a sharded model dir, for EVERY tensor in the weight_map:
#   (A) hash the bytes returned by ShardedSafeTensors.tensor_bytes(name)
#   (B) independently open the RESOLVED shard file via chunk-1 SafeTensors
#       directly, hash the same tensor's bytes there (oracle-correct chunk-1).
#   compare full-length FNV-1a 64-bit, AND length, AND dtype, AND shape.
# ANY mismatch is a BLOCKER and is printed with the tensor name.
#
# Also emits a per-tensor line "<name> <len> <fnv>" so a Python safetensors
# oracle can confirm the bytes match the official library (compare.py).
#
# Run: pixi run mojo run -I . serenitymojo/io/parity/sharded_full_parity.mojo <dir> [--dump]
# If <dir> omitted, runs both transformer and text_encoder.

from std.sys import argv
from serenitymojo.io.sharded import ShardedSafeTensors, _read_file_bytes, _parse_weight_map, _join
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.ffi import sys_open, sys_close, O_RDONLY


comptime SNAP = (
    "/home/alex/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image/"
    "snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021"
)


comptime WINDOW = 65536


def fnv1a(span: Span[UInt8, _]) -> UInt64:
    # 64-bit FNV-1a over the FULL span.
    var h: UInt64 = 0xCBF29CE484222325
    var n = len(span)
    for i in range(n):
        h = h ^ UInt64(span[i])
        h = h * 0x00000100000001B3
    return h


def fnv1a_windowed(span: Span[UInt8, _]) -> UInt64:
    # FNV over first WINDOW bytes then last WINDOW bytes; full if small.
    # Matches sharded_oracle.py fnv1a_windowed exactly.
    var n = len(span)
    var h: UInt64 = 0xCBF29CE484222325
    if n <= 2 * WINDOW:
        for i in range(n):
            h = h ^ UInt64(span[i])
            h = h * 0x00000100000001B3
        return h
    for i in range(WINDOW):
        h = h ^ UInt64(span[i])
        h = h * 0x00000100000001B3
    for i in range(n - WINDOW, n):
        h = h ^ UInt64(span[i])
        h = h * 0x00000100000001B3
    return h


def _index_path(dir: String) raises -> String:
    var cands = List[String]()
    cands.append(String("diffusion_pytorch_model.safetensors.index.json"))
    cands.append(String("model.safetensors.index.json"))
    for ref nm in cands:
        var p = _join(dir, nm)
        var fd = sys_open(p, O_RDONLY)
        if fd >= 0:
            _ = sys_close(fd)
            return p
    raise Error(String("no index in ") + dir)


def check_dir(dir: String, label: String, dump: Bool) raises -> Int:
    print("=========================================================")
    print("DIR:", label, "->", dir)
    # Resolve weight_map ourselves (independent of the loader's internal copy)
    var ip = _index_path(dir)
    var raw = _read_file_bytes(ip)
    var wmap = _parse_weight_map(raw^)
    print("weight_map entries (our parse):", len(wmap))

    var sh = ShardedSafeTensors.open(dir)
    print("loader num_shards:", sh.num_shards(), "num_tensors:", sh.num_tensors())

    # Open each unique resolved shard directly via chunk-1 SafeTensors (oracle).
    var shard_files = List[String]()
    for ref e in wmap.items():
        var f = e.value
        var seen = False
        for ref sf in shard_files:
            if sf == f:
                seen = True
                break
        if not seen:
            shard_files.append(f)

    # Build a parallel structure: open each shard file once.
    # We can't store List[SafeTensors] (not Copyable); open per-tensor on demand
    # would be O(N * shard_open). Instead group tensors by shard file, open the
    # shard once, iterate its tensors.
    var total = 0
    var matched = 0
    var mismatches = List[String]()

    for ref fname in shard_files:
        var shard_path = _join(dir, fname)
        var direct = SafeTensors.open(shard_path)
        # iterate every weight_map tensor that lives in this shard
        for ref we in wmap.items():
            if we.value != fname:
                continue
            var name = we.key
            total += 1

            # (A) via the sharded loader
            var a_info = sh.tensor_info(name)
            var a_bytes = sh.tensor_bytes(name)
            var a_len = len(a_bytes)
            var a_hash = fnv1a(a_bytes)

            # (B) via direct chunk-1 open of the resolved shard
            var b_info = direct.tensor_info(name)
            var b_bytes = direct.tensor_bytes(name)
            var b_len = len(b_bytes)
            var b_hash = fnv1a(b_bytes)

            var ok = True
            if a_len != b_len:
                ok = False
            if a_hash != b_hash:
                ok = False
            if a_info.dtype.name() != b_info.dtype.name():
                ok = False
            if len(a_info.shape) != len(b_info.shape):
                ok = False
            else:
                for k in range(len(a_info.shape)):
                    if a_info.shape[k] != b_info.shape[k]:
                        ok = False

            if ok:
                matched += 1
            else:
                mismatches.append(
                    name + " A(len=" + String(a_len) + ",h=" + String(a_hash)
                    + ",dt=" + a_info.dtype.name() + ") B(len=" + String(b_len)
                    + ",h=" + String(b_hash) + ",dt=" + b_info.dtype.name() + ")"
                )

            if dump:
                # windowed hash for cross-check vs Python oracle (fast)
                var a_win = fnv1a_windowed(a_bytes)
                print("DUMP", name, a_len, a_win, a_info.dtype.name())

    print("TOTAL:", total, " MATCHED:", matched, " MISMATCH:", len(mismatches))
    for ref m in mismatches:
        print("  MISMATCH:", m)
    return total - matched


def main() raises:
    var args = argv()
    var dump = False
    var dir = String("")
    for i in range(1, len(args)):
        var a = String(args[i])
        if a == "--dump":
            dump = True
        else:
            dir = a

    var fails = 0
    if dir != "":
        fails += check_dir(dir, "explicit", dump)
    else:
        fails += check_dir(String(SNAP) + "/transformer", "transformer", dump)
        fails += check_dir(String(SNAP) + "/text_encoder", "text_encoder", dump)

    if fails == 0:
        print("ALL PARITY CHECKS PASSED")
    else:
        print("PARITY FAILURES:", fails)
        raise Error("parity mismatch")

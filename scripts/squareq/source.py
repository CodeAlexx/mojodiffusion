"""Checkpoint source adapter: one .safetensors file OR a sharded directory.

The slab builder was written against a single-file checkpoint. MiniMax-H3 ships
its transformer as 13 shards (66 GB), so `--src` now also accepts a directory
(or its model.safetensors.index.json). The adapter exposes the same three
methods the builder uses — keys() / get_slice() / get_tensor() — and keeps at
most ONE shard handle open at a time so peak mmap/RSS stays at one shard,
matching the bounded-RAM contract of the rest of the builder.
"""

import json
import os
import struct

from safetensors import safe_open


def _shard_files(path: str) -> list:
    idx = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx) as f:
            wm = json.load(f)["weight_map"]
        return sorted({os.path.join(path, v) for v in wm.values()})
    files = sorted(
        os.path.join(path, f) for f in os.listdir(path) if f.endswith(".safetensors")
    )
    if not files:
        raise SystemExit(f"open_source: no .safetensors under {path}")
    return files


def _header_keys(fpath: str) -> dict:
    """Key -> (shape, dtype_tag) straight from the safetensors header. Reads the
    header bytes only; no tensor payload is touched, so nothing is paged in."""
    with open(fpath, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return {
        k: (v["shape"], v["dtype"]) for k, v in hdr.items() if k != "__metadata__"
    }


class _Slice:
    """The two get_slice() accessors the builder uses, from header metadata."""

    def __init__(self, shape, dtype):
        self._shape, self._dtype = shape, dtype

    def get_shape(self):
        return self._shape

    def get_dtype(self):
        return self._dtype


class ShardedSource:
    def __init__(self, files: list):
        self._files = files
        self._meta = {}  # key -> (file, shape, dtype)
        for fp in files:
            for k, (shape, dt) in _header_keys(fp).items():
                if k in self._meta:
                    raise SystemExit(f"ShardedSource: duplicate key {k}")
                self._meta[k] = (fp, shape, dt)
        self._open_path = None
        self._open_h = None

    def keys(self):
        return list(self._meta)

    def get_slice(self, k: str):
        _, shape, dt = self._meta[k]
        return _Slice(shape, dt)

    def get_tensor(self, k: str):
        fp = self._meta[k][0]
        if fp != self._open_path:
            self._close()
            self._open_h = safe_open(fp, "pt")
            self._open_path = fp
        return self._open_h.get_tensor(k)

    def _close(self):
        self._open_h = None
        self._open_path = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._close()
        return False


def open_source(path: str):
    """Context manager over a single file or a sharded directory."""
    if os.path.isdir(path):
        return ShardedSource(_shard_files(path))
    return safe_open(path, "pt")


def source_identity(path: str) -> tuple:
    """(bytes, mtime) resume identity. For a directory: summed size and the
    newest mtime across shards, so touching any shard invalidates the resume."""
    if not os.path.isdir(path):
        st = os.stat(path)
        return st.st_size, int(st.st_mtime)
    total = 0
    newest = 0
    for fp in _shard_files(path):
        st = os.stat(fp)
        total += st.st_size
        newest = max(newest, int(st.st_mtime))
    return total, newest

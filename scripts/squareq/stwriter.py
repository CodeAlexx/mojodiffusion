"""Streaming safetensors writer — bounded-RAM, two-pass.

safetensors layout: u64 header_len | JSON header | raw payload. Offsets depend
only on dtypes/shapes, so pass 1 declares every tensor (name, dtype, shape),
the header is written once, and pass 2 streams payloads one tensor at a time
with positional pwrites. Peak RAM = one tensor. Never accumulate a model dict.

Write to `<path>.tmp` and rename on close for crash atomicity.
"""

import json
import os
import struct

import torch

_DTYPE_TAG = {
    torch.float32: ("F32", 4),
    torch.bfloat16: ("BF16", 2),
    torch.float16: ("F16", 2),
    torch.uint8: ("U8", 1),
    torch.int8: ("I8", 1),
    torch.int16: ("I16", 2),
    torch.int32: ("I32", 4),
    torch.int64: ("I64", 8),
}


def _numel(shape) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n


class StreamingSafetensorsWriter:
    def __init__(self, path: str, metadata: dict | None = None):
        self.path = path
        self.tmp = path + ".tmp"
        self._decl: list = []  # (name, tag, shape, offset, nbytes)
        self._offsets: dict = {}
        self._cursor = 0
        self._metadata = metadata or {}
        self._fd = None
        self._data_base = 0
        self._written: set = set()

    def declare(self, name: str, dtype: torch.dtype, shape) -> None:
        if self._fd is not None:
            raise RuntimeError("declare() after header written")
        if name in self._offsets:
            raise ValueError(f"duplicate tensor: {name}")
        tag, esize = _DTYPE_TAG[dtype]
        nbytes = _numel(shape) * esize
        self._decl.append((name, tag, list(shape), self._cursor, nbytes))
        self._offsets[name] = (self._cursor, nbytes, tag, list(shape))
        self._cursor += nbytes

    def write_header(self) -> None:
        header = {}
        if self._metadata:
            header["__metadata__"] = {k: str(v) for k, v in self._metadata.items()}
        for name, tag, shape, off, nbytes in self._decl:
            header[name] = {
                "dtype": tag,
                "shape": shape,
                "data_offsets": [off, off + nbytes],
            }
        hjson = json.dumps(header, separators=(",", ":")).encode("utf-8")
        pad = (-(8 + len(hjson))) % 8  # align payload to 8
        hjson += b" " * pad
        self._fd = os.open(self.tmp, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
        os.pwrite(self._fd, struct.pack("<Q", len(hjson)), 0)
        os.pwrite(self._fd, hjson, 8)
        self._data_base = 8 + len(hjson)
        os.ftruncate(self._fd, self._data_base + self._cursor)

    def write_tensor(self, name: str, t: torch.Tensor) -> None:
        if self._fd is None:
            raise RuntimeError("write_header() first")
        off, nbytes, tag, shape = self._offsets[name]
        tag2, _ = _DTYPE_TAG[t.dtype]
        if tag2 != tag or list(t.shape) != shape:
            raise ValueError(
                f"{name}: declared {tag}{shape}, got {tag2}{list(t.shape)}"
            )
        t = t.contiguous()
        buf = t.view(torch.uint8) if t.dtype == torch.bfloat16 else t
        data = buf.reshape(-1).numpy().tobytes() if t.dtype != torch.bfloat16 else bytes(
            buf.reshape(-1).numpy()
        )
        if len(data) != nbytes:
            raise ValueError(f"{name}: payload {len(data)} != declared {nbytes}")
        os.pwrite(self._fd, data, self._data_base + off)
        self._written.add(name)

    def close(self) -> None:
        if self._fd is None:
            raise RuntimeError("nothing written")
        missing = set(self._offsets) - self._written
        if missing:
            os.close(self._fd)
            raise RuntimeError(f"unwritten tensors: {sorted(missing)[:8]} ...")
        os.fsync(self._fd)
        os.close(self._fd)
        self._fd = None
        os.replace(self.tmp, self.path)

    def abort(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        if os.path.exists(self.tmp):
            os.unlink(self.tmp)

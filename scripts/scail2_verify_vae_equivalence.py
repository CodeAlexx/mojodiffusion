#!/usr/bin/env python3
"""Prove that a native SCAIL-2 Wan VAE and a Mojo cache have identical tensors.

The two formats use different key names. This development-only gate compares
the complete multiset of dtype/shape/content fingerprints, so renamed tensors
must still have exactly identical bytes and multiplicities.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
from pathlib import Path

import torch
from safetensors import safe_open


def fingerprint(tensor: torch.Tensor) -> tuple[str, tuple[int, ...], str]:
    tensor = tensor.detach().cpu().contiguous()
    payload = tensor.view(torch.uint8).numpy()
    return str(tensor.dtype), tuple(tensor.shape), hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("native_pth", type=Path)
    parser.add_argument("mojo_safetensors", type=Path)
    args = parser.parse_args()

    native = torch.load(
        args.native_pth.resolve(), map_location="cpu", mmap=True, weights_only=True
    )
    if not isinstance(native, dict) or len(native) != 194:
        raise RuntimeError(f"native Wan VAE must contain 194 tensors, found {len(native)}")
    native_fingerprints = collections.Counter(fingerprint(value) for value in native.values())

    with safe_open(args.mojo_safetensors.resolve(), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        if len(keys) != 194:
            raise RuntimeError(f"Mojo Wan VAE must contain 194 tensors, found {len(keys)}")
        mojo_fingerprints = collections.Counter(
            fingerprint(handle.get_tensor(name)) for name in keys
        )

    if native_fingerprints != mojo_fingerprints:
        only_native = native_fingerprints - mojo_fingerprints
        only_mojo = mojo_fingerprints - native_fingerprints
        raise RuntimeError(
            f"VAE tensor contents differ: only_native={sum(only_native.values())} "
            f"only_mojo={sum(only_mojo.values())}"
        )
    print("SCAIL-2 Wan VAE equivalence PASS: 194/194 tensors byte-identical")


if __name__ == "__main__":
    main()

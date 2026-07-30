#!/usr/bin/env python3
"""Run the pinned Wan creator CLI without its unused DashScope dependency.

Wan's generate.py imports `dashscope` at module import time even when prompt
extension is disabled. This development-only wrapper supplies an empty module
for that unused optional dependency, verifies the creator checkout pin, and
then executes the unmodified creator CLI with the remaining arguments.
"""

from __future__ import annotations

import runpy
import subprocess
import sys
import types
from pathlib import Path

import torch


ORACLE_ROOT = Path("/home/alex/Wan2.2")
ORACLE_REVISION = "42bf4cfaa384bc21833865abc2f9e6c0e67233dc"


def check_oracle() -> None:
    head = subprocess.check_output(
        ["git", "-C", str(ORACLE_ROOT), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(ORACLE_ROOT), "status", "--porcelain"],
        text=True,
    ).strip()
    if head != ORACLE_REVISION or dirty:
        raise RuntimeError(
            "Wan creator checkout must be clean at "
            f"{ORACLE_REVISION}; head={head!r} dirty={bool(dirty)}"
        )


def main() -> None:
    check_oracle()
    sys.path.insert(0, str(ORACLE_ROOT))
    sys.modules.setdefault("dashscope", types.ModuleType("dashscope"))

    # The creator checkout requires the optional flash_attn package for its
    # attention wrapper. This machine already has PyTorch/cuDNN SDPA, which is
    # math-equivalent for the batch-1, unmasked TI2V path. Patch only the module
    # symbol used by WanModel; creator weights, blocks, scheduler, and VAE remain
    # unchanged.
    import wan.modules.model as wan_model

    def torch_sdpa_attention(
        q,
        k,
        v,
        q_lens=None,
        k_lens=None,
        dropout_p=0.0,
        softmax_scale=None,
        q_scale=None,
        causal=False,
        window_size=(-1, -1),
        deterministic=False,
        dtype=torch.bfloat16,
        version=None,
    ):
        del q_lens, causal, window_size, deterministic, version
        output_dtype = q.dtype
        q = q.to(dtype)
        k = k.to(dtype)
        v = v.to(dtype)
        if q_scale is not None:
            q = q * q_scale
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attention_mask = None
        if k_lens is not None:
            batch, _, key_length, _ = k.shape
            invalid = torch.zeros(
                batch,
                1,
                1,
                key_length,
                dtype=torch.bool,
                device=k.device,
            )
            for index, valid in enumerate(k_lens.tolist()):
                invalid[index, :, :, int(valid) :] = True
            if invalid.any():
                attention_mask = torch.zeros(
                    batch,
                    1,
                    1,
                    key_length,
                    dtype=q.dtype,
                    device=k.device,
                )
                attention_mask.masked_fill_(invalid, float("-inf"))
        output = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=dropout_p,
            scale=softmax_scale,
        )
        return output.transpose(1, 2).contiguous().to(output_dtype)

    wan_model.flash_attention = torch_sdpa_attention
    generate = ORACLE_ROOT / "generate.py"
    sys.argv = [str(generate), *sys.argv[1:]]
    runpy.run_path(str(generate), run_name="__main__")


if __name__ == "__main__":
    main()

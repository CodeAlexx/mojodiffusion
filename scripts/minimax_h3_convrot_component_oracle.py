#!/usr/bin/env python3
"""Pinned Musubi/Torch oracle for the bounded MiniMax-H3 ConvRot component.

This is development-only fixture generation.  Product code remains pure Mojo.
The oracle executes the immutable Musubi ConvRot module directly from git object
``b871786...`` for the regular H256 rotation and Triton BF16 row quantizer.  It
also pins the exact BF16 transient weight-dequant expression used by Musubi's
``ConvRotInt8LinearFn.backward``.

This fixture does *not* cover an INT8 projection, a DiT block, or a trainer.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import triton


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "serenitymojo/training/parity/fixtures/minimax_h3_convrot_component_v1.json"
)
COMMIT = "b8717864713c9e4e7ef3d56eba1fc695a9b626a5"
KERNEL_SOURCE = "src/musubi_tuner/modules/convrot_int8_kernels.py"
UTIL_SOURCE = "src/musubi_tuner/modules/convrot_int8_utils.py"
KERNEL_SHA256 = "47af7db50bd4017df8fedbf2bf3de726064076fc4fb662dd7bb43a9c9c268978"
UTIL_SHA256 = "8f864adfb204c6115ae7237ef0a0debe86f56d3b6c129014e59bb4f055b5c22b"
SCHEMA = "serenity.minimax_h3.convrot_component_oracle.v1"
GROUP_SIZE = 256


def _git_source(repo: Path, path: str, expected_sha: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), "show", f"{COMMIT}:{path}"],
        check=True,
        stdout=subprocess.PIPE,
    )
    digest = hashlib.sha256(proc.stdout).hexdigest()
    if digest != expected_sha:
        raise RuntimeError(f"pinned source digest mismatch for {path}: {digest}")
    return proc.stdout.decode("utf-8")


def _load_upstream(
    kernel_source: str,
) -> tuple[dict[str, object], tempfile.TemporaryDirectory[str]]:
    # Triton 3.7 requires @jit functions to live in an inspectable real file.
    # Materialize only the already SHA-verified immutable git object in /tmp.
    temp_dir = tempfile.TemporaryDirectory(prefix="h3-convrot-oracle-")
    module_path = Path(temp_dir.name) / "pinned_musubi_convrot_int8_kernels.py"
    module_path.write_text(kernel_source, encoding="utf-8")
    module_name = "pinned_musubi_convrot_int8_kernels"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not create pinned Musubi module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    namespace = vars(module)
    for name in ("_build_hadamard", "_rotate_activation", "triton_quantize_rowwise"):
        if name not in namespace:
            raise RuntimeError(f"pinned Musubi source did not define {name}")
    return namespace, temp_dir


def _as_f32_list(tensor: torch.Tensor) -> list[float]:
    return tensor.detach().float().cpu().reshape(-1).tolist()


def _as_i8_list(tensor: torch.Tensor) -> list[int]:
    return tensor.detach().cpu().reshape(-1).tolist()


def _manual_quant_contract(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Literal lines 202-231 of pinned Triton BF16 branch."""
    scale = (x.abs().amax(dim=-1, keepdim=True).float() / 127.0).clamp(min=1.0e-30)
    quotient_bf16 = (x / scale.to(torch.bfloat16)).to(torch.bfloat16)
    q = torch.round(quotient_bf16.float()).clamp(-128.0, 127.0).to(torch.int8)
    return q, scale


def _sylvester(size: int, device: torch.device) -> torch.Tensor:
    h = torch.ones((1, 1), dtype=torch.float32, device=device)
    h2 = torch.tensor([[1.0, 1.0], [1.0, -1.0]], device=device)
    while h.shape[0] < size:
        h = torch.kron(h, h2)
    return (h / math.sqrt(size)).to(torch.bfloat16)


def _rotation_input(device: torch.device) -> torch.Tensor:
    values: list[float] = []
    for row in range(3):
        for col in range(2 * GROUP_SIZE):
            numerator = ((col * (37 + 6 * row) + row * 53) % 521) - 260
            value = numerator / float(16 << row)
            if col % 97 == 0:
                value = 0.0
            elif col % 89 == 0:
                value = -value * 3.0
            values.append(value)
    return torch.tensor(values, dtype=torch.bfloat16, device=device).reshape(3, 512)


def _quant_input(device: torch.device) -> torch.Tensor:
    rows = [
        [0.0] * 16,
        [
            127.0,
            -127.0,
            0.5,
            -0.5,
            1.5,
            -1.5,
            2.5,
            -2.5,
            3.5,
            -3.5,
            126.5,
            -126.5,
            63.5,
            -63.5,
            1.0,
            -1.0,
        ],
        [
            3.859375,
            1.546875,
            -1.828125,
            3.59375,
            0.2109375,
            2.78125,
            -3.265625,
            1.1875,
            -2.515625,
            3.453125,
            -2.40625,
            -3.796875,
            0.482421875,
            -3.09375,
            -3.59375,
            -0.05859375,
        ],
        [
            5.46875,
            -2.21875,
            4.5,
            -5.3125,
            -2.921875,
            0.439453125,
            -5.46875,
            -0.369140625,
            -1.6640625,
            0.5234375,
            -1.796875,
            0.96484375,
            2.078125,
            -3.859375,
            3.53125,
            -4.875,
        ],
        [
            1.0e-31,
            -1.0e-31,
            5.0e-31,
            -5.0e-31,
            9.0e-31,
            -9.0e-31,
            0.0,
            0.0,
            2.0e-31,
            -2.0e-31,
            3.0e-31,
            -3.0e-31,
            7.0e-31,
            -7.0e-31,
            8.0e-31,
            -8.0e-31,
        ],
    ]
    return torch.tensor(rows, dtype=torch.bfloat16, device=device)


def _weight_inputs(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    q = torch.tensor(
        [
            [-128, -127, -65, -3, -2, -1, 0, 1, 2, 3, 64, 65, 126, 127, 17, -19],
            [127, -128, 93, -91, 33, -31, 7, -5, 1, 0, -1, 5, -7, 31, -33, 91],
            [0, 1, -1, 2, -2, 4, -4, 8, -8, 16, -16, 32, -32, 64, -64, 127],
            [-128, 127, -126, 125, -64, 63, -17, 16, -3, 2, -1, 0, 1, 3, 65, -66],
        ],
        dtype=torch.int8,
        device=device,
    )
    scale = torch.tensor(
        [[0.030388779938220978], [0.0430610254406929], [1.0e-30], [math.pi]],
        dtype=torch.float32,
        device=device,
    )
    return q, scale


def _build_fixture(source_repo: Path) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the pinned Musubi Triton/BF16 oracle")
    kernel_source = _git_source(source_repo, KERNEL_SOURCE, KERNEL_SHA256)
    util_source = _git_source(source_repo, UTIL_SOURCE, UTIL_SHA256)
    required_dequant = (
        "w_rot = wq.to(grad_out.dtype) * "
        "w_scale.reshape(-1, 1).to(grad_out.dtype)"
    )
    if required_dequant not in util_source:
        raise RuntimeError("pinned Musubi BF16 backward dequant expression changed")

    upstream, source_temp = _load_upstream(kernel_source)
    device = torch.device("cuda")
    torch.cuda.manual_seed_all(0)

    h = upstream["_build_hadamard"](  # type: ignore[index,operator]
        GROUP_SIZE, device=device, dtype=torch.bfloat16
    )
    x = _rotation_input(device)
    rotated = upstream["_rotate_activation"](x, h, GROUP_SIZE)  # type: ignore[index,operator]
    involuted = upstream["_rotate_activation"](rotated, h, GROUP_SIZE)  # type: ignore[index,operator]
    wrong_h = _sylvester(GROUP_SIZE, device)
    wrong_rotated = torch.matmul(x.reshape(-1, 2, GROUP_SIZE), wrong_h).reshape_as(x)

    qx = _quant_input(device)
    q_actual, scale_actual = upstream["triton_quantize_rowwise"](qx)  # type: ignore[index,operator]
    q_manual, scale_manual = _manual_quant_contract(qx)
    torch.cuda.synchronize()
    if not torch.equal(q_actual, q_manual) or not torch.equal(scale_actual, scale_manual):
        raise RuntimeError("manual BF16 boundary contract disagrees with pinned Musubi Triton")

    wrong_f32_scale = torch.round(qx.float() / scale_actual).clamp(-128, 127).to(torch.int8)
    wrong_no_q_bf16 = torch.round(
        qx.float() / scale_actual.to(torch.bfloat16).float()
    ).clamp(-128, 127).to(torch.int8)
    wrong_scale_mismatches = int((wrong_f32_scale != q_actual).sum().item())
    wrong_quotient_mismatches = int((wrong_no_q_bf16 != q_actual).sum().item())
    if wrong_scale_mismatches == 0 or wrong_quotient_mismatches == 0:
        raise RuntimeError("quant fixture lost BF16-boundary sensitivity")
    if -128 not in q_actual or 127 not in q_actual:
        raise RuntimeError("quant fixture lost signed INT8 extrema")

    wq, w_scale = _weight_inputs(device)
    # Literal pinned ConvRotInt8LinearFn.backward BF16 branch semantics.
    w_dequant = wq.to(torch.bfloat16) * w_scale.reshape(-1, 1).to(torch.bfloat16)
    wrong_w_dequant = (wq.float() * w_scale).to(torch.bfloat16)
    wrong_weight_mismatches = int((w_dequant != wrong_w_dequant).sum().item())
    if wrong_weight_mismatches == 0:
        raise RuntimeError("weight fixture lost BF16-scale/product sensitivity")

    regular_wrong_max_abs = float((rotated.float() - wrong_rotated.float()).abs().max().item())
    regular_wrong_rel_l2 = float(
        torch.linalg.vector_norm(rotated.float() - wrong_rotated.float()).item()
        / max(torch.linalg.vector_norm(rotated.float()).item(), 1.0e-30)
    )
    if regular_wrong_rel_l2 < 0.25:
        raise RuntimeError("rotation fixture is not sensitive to Sylvester ordering")

    h_bytes = bytes(h.detach().cpu().view(torch.uint8).reshape(-1).tolist())
    fixture = {
        "schema": SCHEMA,
        "oracle_commit": COMMIT,
        "source_files": {
            KERNEL_SOURCE: KERNEL_SHA256,
            UTIL_SOURCE: UTIL_SHA256,
        },
        "oracle_runtime": {
            "torch": torch.__version__,
            "triton": triton.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
        },
        "claim": "H256 rotation/involution + BF16 activation quant + BF16 backward weight dequant component parity only",
        "dtype_contract": {
            "rotation_input_h_output": "torch.bfloat16",
            "activation_codes": "torch.int8",
            "activation_scale": "torch.float32 [rows,1]",
            "activation_quant_math": "scale F32; denominator BF16; quotient BF16; RNE; clamp [-128,127]",
            "weight_codes": "torch.int8",
            "weight_scale": "torch.float32 [rows,1] cast to BF16 before multiply",
            "weight_dequant_output": "torch.bfloat16",
        },
        "geometry": {
            "group_size": GROUP_SIZE,
            "kronecker_base": 4,
            "kronecker_factors": 4,
            "rotation_shape": list(x.shape),
            "quant_shape": list(qx.shape),
            "weight_shape": list(wq.shape),
        },
        "hadamard_bf16_bytes_sha256": hashlib.sha256(h_bytes).hexdigest(),
        "sensitivity": {
            "regular_vs_sylvester_max_abs": regular_wrong_max_abs,
            "regular_vs_sylvester_rel_l2": regular_wrong_rel_l2,
            "wrong_f32_scale_code_mismatches": wrong_scale_mismatches,
            "wrong_no_quotient_bf16_code_mismatches": wrong_quotient_mismatches,
            "wrong_f32_weight_dequant_mismatches": wrong_weight_mismatches,
        },
        "inputs": {
            "rotation_x_bf16_as_f32": _as_f32_list(x),
            "activation_x_bf16_as_f32": _as_f32_list(qx),
            "weight_q_int8": _as_i8_list(wq),
            "weight_scale_f32": _as_f32_list(w_scale),
        },
        "outputs": {
            "rotation_y_bf16_as_f32": _as_f32_list(rotated),
            "involution_y_bf16_as_f32": _as_f32_list(involuted),
            "wrong_sylvester_y_bf16_as_f32": _as_f32_list(wrong_rotated),
            "activation_q_int8": _as_i8_list(q_actual),
            "activation_scale_f32": _as_f32_list(scale_actual),
            "weight_dequant_bf16_as_f32": _as_f32_list(w_dequant),
        },
    }
    source_temp.cleanup()
    return fixture


def _serialize(doc: dict[str, object]) -> bytes:
    return (json.dumps(doc, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sidecar_path(output: Path) -> Path:
    return output.with_suffix(".sha256")


def _write(output: Path, payload: bytes) -> str:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    _sidecar_path(output).write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    return digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", type=Path, default=Path("/home/alex/musubi-tuner"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    payload = _serialize(_build_fixture(args.source_repo))
    digest = hashlib.sha256(payload).hexdigest()
    if args.check:
        if not args.output.is_file():
            raise SystemExit(f"missing canonical fixture: {args.output}")
        if args.output.read_bytes() != payload:
            raise SystemExit("canonical ConvRot fixture is stale or nondeterministic")
        sidecar = _sidecar_path(args.output)
        expected_sidecar = f"{digest}  {args.output.name}\n"
        if not sidecar.is_file() or sidecar.read_text(encoding="utf-8") != expected_sidecar:
            raise SystemExit("canonical ConvRot SHA256 sidecar is stale")
        print(f"ConvRot oracle check PASS sha256={digest}")
        return

    written = _write(args.output, payload)
    print(f"wrote {args.output}")
    print(f"sha256={written}")


if __name__ == "__main__":
    main()

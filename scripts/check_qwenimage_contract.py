#!/usr/bin/env python3
"""Static dtype guard for the Qwen-Image checkpoint loader path.

Default mode keeps the accepted loader gate strict and reports train-boundary
host-F32 carriers as warnings. Use --strict-train-boundaries to make those
non-production activation/grad carriers fatal.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
WEIGHTS = REPO / "serenitymojo/models/qwenimage/weights.mojo"
STACK = REPO / "serenitymojo/models/qwenimage/qwenimage_stack_lora.mojo"
TRAIN = REPO / "serenitymojo/training/train_qwenimage_real.mojo"
SMOKE = REPO / "serenitymojo/models/qwenimage/parity/qwen_fp8_loader_smoke.mojo"


@dataclass(frozen=True)
class BoundaryBlocker:
    category: str
    path: Path
    line: int
    symbol: str
    detail: str
    code: str


def read(path: Path) -> str:
    if not path.exists():
        raise SystemExit(f"missing required file: {path}")
    return path.read_text(encoding="utf-8")


def function_body(text: str, name: str) -> str:
    return top_level_span(text, "def", name)[0]


def struct_body(text: str, name: str) -> str:
    return top_level_span(text, "struct", name)[0]


def top_level_body(text: str, keyword: str, name: str) -> str:
    return top_level_span(text, keyword, name)[0]


def top_level_span(text: str, keyword: str, name: str) -> tuple[str, int]:
    match = re.search(rf"^{keyword} {re.escape(name)}\b", text, flags=re.MULTILINE)
    if not match:
        raise SystemExit(f"missing {keyword} {name}")
    next_match = re.search(r"^def \w+\b|^struct \w+\b|^comptime \w+\b", text[match.end() :], flags=re.MULTILINE)
    end = len(text) if next_match is None else match.end() + next_match.start()
    start_line = text.count("\n", 0, match.start()) + 1
    return text[match.start() : end], start_line


def has(pattern: str, text: str) -> bool:
    return re.search(pattern, text, flags=re.DOTALL | re.MULTILINE) is not None


def strip_comments(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        lines.append(line.split("#", 1)[0])
    return "\n".join(lines)


def code_part(line: str) -> str:
    return line.split("#", 1)[0].rstrip()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def add_if(blockers: list[str], condition: bool, message: str) -> None:
    if condition:
        blockers.append(message)


def add_boundary_hits(
    blockers: list[BoundaryBlocker],
    *,
    category: str,
    path: Path,
    text: str,
    pattern: str | re.Pattern[str],
    symbol: str,
    detail: str,
    scope: tuple[str, str] | None = None,
    max_hits: int | None = None,
) -> None:
    compiled = re.compile(pattern) if isinstance(pattern, str) else pattern
    if scope is None:
        body = text
        start_line = 1
    else:
        body, start_line = top_level_span(text, scope[0], scope[1])

    hits = 0
    for offset, raw_line in enumerate(body.splitlines()):
        code = code_part(raw_line)
        if not code.strip():
            continue
        if not compiled.search(code):
            continue
        blockers.append(
            BoundaryBlocker(
                category=category,
                path=path,
                line=start_line + offset,
                symbol=symbol,
                detail=detail,
                code=code.strip(),
            )
        )
        hits += 1
        if max_hits is not None and hits >= max_hits:
            return


def grouped_boundary_blockers(
    blockers: list[BoundaryBlocker],
) -> list[tuple[str, list[BoundaryBlocker]]]:
    groups: dict[str, list[BoundaryBlocker]] = {}
    order: list[str] = []
    for blocker in blockers:
        if blocker.category not in groups:
            groups[blocker.category] = []
            order.append(blocker.category)
        groups[blocker.category].append(blocker)
    return [(category, groups[category]) for category in order]


def loader_blockers(weights: str, stack: str, smoke: str) -> list[str]:
    blockers: list[str] = []
    qwen_loader = function_body(weights, "load_qwen_tensor_bf16")
    qwen_host = function_body(weights, "load_qwen_host_bf16")
    qwen_private = function_body(weights, "_load_tensor")
    stack_block_loader = function_body(stack, "_block_tensor_bf16")
    qwen_loader_code = strip_comments(qwen_loader)
    qwen_host_code = strip_comments(qwen_host)
    qwen_private_code = strip_comments(qwen_private)
    stack_block_loader_code = strip_comments(stack_block_loader)

    add_if(
        blockers,
        "Tensor.from_view_raw" not in qwen_loader_code or "fp8_e4m3_dequant_to_bf16" not in qwen_loader_code,
        f"{WEIGHTS}: load_qwen_tensor_bf16 no longer raw-loads FP8 and dequants to BF16",
    )
    add_if(
        blockers,
        "Tensor.from_view_as_bf16" not in qwen_loader_code,
        f"{WEIGHTS}: load_qwen_tensor_bf16 no longer routes non-FP8 checkpoint tensors through BF16",
    )
    add_if(
        blockers,
        "Tensor.from_view(" in qwen_loader_code or "from_view_as_f32" in qwen_loader_code,
        f"{WEIGHTS}: load_qwen_tensor_bf16 uses a generic/F32 checkpoint tensor boundary",
    )
    add_if(
        blockers,
        ".to_host(ctx)" in qwen_loader_code or "List[Float32]" in qwen_loader_code,
        f"{WEIGHTS}: load_qwen_tensor_bf16 materializes checkpoint tensors as host F32",
    )
    add_if(
        blockers,
        "to_host_bf16" not in qwen_host_code or "List[BFloat16]" not in qwen_host_code,
        f"{WEIGHTS}: load_qwen_host_bf16 no longer exposes host BF16",
    )
    add_if(
        blockers,
        "load_qwen_tensor_bf16" not in qwen_private_code,
        f"{WEIGHTS}: _load_tensor no longer delegates to the BF16 Qwen loader",
    )

    add_if(
        blockers,
        "fp8_e4m3_dequant_to_bf16" not in stack_block_loader_code,
        f"{STACK}: streamed block helper no longer dequants FP8 block tensors to BF16",
    )
    add_if(
        blockers,
        "_block_tensor_bf16" not in stack or "_stream_weights_from_block_offload" not in stack,
        f"{STACK}: streamed block loader helpers are missing",
    )
    add_if(
        blockers,
        ".to_host(ctx)" in stack_block_loader_code or "List[Float32]" in stack_block_loader_code,
        f"{STACK}: _block_tensor_bf16 materializes block weights as host F32",
    )
    add_if(
        blockers,
        has(r"cast_tensor\([^)]*STDtype\.F32", stack_block_loader_code)
        or has(r"Tensor\.from_host\([^)]*STDtype\.F32", stack_block_loader_code),
        f"{STACK}: _block_tensor_bf16 creates an F32 tensor boundary",
    )
    add_if(
        blockers,
        "load_block_as_f32" in stack or "_block_tensor_f32" in stack,
        f"{STACK}: streamed block path contains an explicit F32 block-weight loader",
    )

    add_if(
        blockers,
        "load_qwen_tensor_bf16" not in smoke or "to_host_bf16" not in smoke,
        f"{SMOKE}: FP8 loader smoke no longer checks BF16 device/host boundary",
    )
    return blockers


def train_boundary_blockers(weights: str, stack: str, train: str) -> list[BoundaryBlocker]:
    blockers: list[BoundaryBlocker] = []

    # Cache/input tensors: these are the Qwen hits reported by
    # scripts/check_train_loop_cache_contract_bindings.py, narrowed to actual
    # train-boundary carriers rather than scalar helper math.
    add_boundary_hits(
        blockers,
        category="cache-inputs",
        path=TRAIN,
        text=train,
        pattern=r"^\s*def _load_host_f32\b",
        symbol="_load_host_f32",
        detail="cache tensor loader exposes sample tensors as host List[Float32]",
    )
    add_boundary_hits(
        blockers,
        category="cache-inputs",
        path=TRAIN,
        text=train,
        pattern=r"^\s*def _load_host_f32_sharded\b",
        symbol="_load_host_f32_sharded",
        detail="sharded helper exposes checkpoint/cache tensors as host List[Float32]",
    )
    add_boundary_hits(
        blockers,
        category="cache-inputs",
        path=TRAIN,
        text=train,
        pattern=r"\breturn t\.to_host\(\s*ctx\s*\)",
        symbol="_load_host_f32*.return",
        detail="loader returns a device tensor through a host-F32 boundary",
        scope=("def", "_load_host_f32"),
    )
    add_boundary_hits(
        blockers,
        category="cache-inputs",
        path=TRAIN,
        text=train,
        pattern=r"\breturn t\.to_host\(\s*ctx\s*\)",
        symbol="_load_host_f32_sharded.return",
        detail="sharded loader returns a device tensor through a host-F32 boundary",
        scope=("def", "_load_host_f32_sharded"),
    )
    add_boundary_hits(
        blockers,
        category="cache-inputs",
        path=TRAIN,
        text=train,
        pattern=r"\bvar img_tokens = List\[Float32\]\(\)",
        symbol="main.img_tokens",
        detail="latent/image tokens are assembled as host F32 before the forward pass",
        scope=("def", "main"),
    )
    add_boundary_hits(
        blockers,
        category="cache-inputs",
        path=TRAIN,
        text=train,
        pattern=r"\bvar txt_tokens = List\[Float32\]\(\)",
        symbol="main.txt_tokens",
        detail="text tokens are assembled as host F32 before the forward pass",
        scope=("def", "main"),
    )
    add_boundary_hits(
        blockers,
        category="cache-inputs",
        path=TRAIN,
        text=train,
        pattern=r"\bimg_tokens = _load_host_f32\b",
        symbol="main.latent_cache_read",
        detail="cached latent is read directly into host F32 tokens",
        scope=("def", "main"),
    )
    add_boundary_hits(
        blockers,
        category="cache-inputs",
        path=TRAIN,
        text=train,
        pattern=r"\btxt_flat = _load_host_f32\b",
        symbol="main.text_cache_read",
        detail="cached text embedding is read directly into a host F32 staging list",
        scope=("def", "main"),
    )

    add_boundary_hits(
        blockers,
        category="noise-target-loss",
        path=TRAIN,
        text=train,
        pattern=r"^\s*def _host_noise\b.*List\[Float32\]",
        symbol="_host_noise",
        detail="training noise is generated and stored as host List[Float32]",
    )
    add_boundary_hits(
        blockers,
        category="noise-target-loss",
        path=TRAIN,
        text=train,
        pattern=r"\bvar noise = _host_noise\b",
        symbol="main.noise",
        detail="per-step noise remains a host F32 carrier",
        scope=("def", "main"),
    )
    add_boundary_hits(
        blockers,
        category="noise-target-loss",
        path=TRAIN,
        text=train,
        pattern=r"\bvar noisy = List\[Float32\]\(\)",
        symbol="main.noisy",
        detail="flow-match noisy latents are assembled as host F32",
        scope=("def", "main"),
    )
    add_boundary_hits(
        blockers,
        category="noise-target-loss",
        path=TRAIN,
        text=train,
        pattern=r"\bvar target = List\[Float32\]\(\)",
        symbol="main.target",
        detail="flow-match target is assembled as host F32",
        scope=("def", "main"),
    )
    add_boundary_hits(
        blockers,
        category="noise-target-loss",
        path=TRAIN,
        text=train,
        pattern=r"\bvar d_loss = List\[Float32\]\(\)",
        symbol="main.d_loss",
        detail="loss gradient is carried as a host F32 list into backward",
        scope=("def", "main"),
    )

    add_boundary_hits(
        blockers,
        category="timestep-rope",
        path=TRAIN,
        text=train,
        pattern=r"^\s*def _sinusoidal_temb\b.*List\[Float32\]",
        symbol="_sinusoidal_temb",
        detail="sinusoidal timestep embedding returns host F32",
    )
    add_boundary_hits(
        blockers,
        category="timestep-rope",
        path=TRAIN,
        text=train,
        pattern=r"\bSTDtype\.F32\b",
        symbol="_sinusoidal_temb.STDtype.F32",
        detail="timestep embedding tensor is explicitly created/computed as F32",
        scope=("def", "_sinusoidal_temb"),
    )
    add_boundary_hits(
        blockers,
        category="timestep-rope",
        path=TRAIN,
        text=train,
        pattern=r"\breturn t_emb\.to_host\(\s*ctx\s*\)",
        symbol="_sinusoidal_temb.return",
        detail="timestep embedding crosses back to host F32",
        scope=("def", "_sinusoidal_temb"),
    )
    add_boundary_hits(
        blockers,
        category="timestep-rope",
        path=TRAIN,
        text=train,
        pattern=r"^\s*\) raises -> List\[Float32\]",
        symbol="_build_silu_temb",
        detail="timestep MLP output is exposed as host List[Float32]",
        scope=("def", "_build_silu_temb"),
    )
    add_boundary_hits(
        blockers,
        category="timestep-rope",
        path=TRAIN,
        text=train,
        pattern=r"\breturn silu\(temb_out, ctx\)\.to_host\(\s*ctx\s*\)",
        symbol="_build_silu_temb.return",
        detail="silu timestep MLP activation crosses back to host F32",
        scope=("def", "_build_silu_temb"),
    )
    add_boundary_hits(
        blockers,
        category="timestep-rope",
        path=TRAIN,
        text=train,
        pattern=r"\bvar cos_h = rope\[0\]\.to_host\(\s*ctx\s*\)",
        symbol="main.rope_cos",
        detail="RoPE cosine table is staged as host F32",
        scope=("def", "main"),
    )
    add_boundary_hits(
        blockers,
        category="timestep-rope",
        path=TRAIN,
        text=train,
        pattern=r"\bvar sin_h = rope\[1\]\.to_host\(\s*ctx\s*\)",
        symbol="main.rope_sin",
        detail="RoPE sine table is staged as host F32",
        scope=("def", "main"),
    )

    add_boundary_hits(
        blockers,
        category="modulation",
        path=STACK,
        text=stack,
        pattern=r"\btemb_h: List\[Float32\]",
        symbol="_modvecs_from_block.temb_h",
        detail="per-block modulation input is a host F32 list",
        scope=("def", "_modvecs_from_block"),
        max_hits=1,
    )
    add_boundary_hits(
        blockers,
        category="modulation",
        path=STACK,
        text=stack,
        pattern=r"\.to_host\(\s*ctx\s*\)",
        symbol="_modvecs_from_block.mods",
        detail="per-block modulation linear output crosses back to host F32",
        scope=("def", "_modvecs_from_block"),
    )
    add_boundary_hits(
        blockers,
        category="modulation",
        path=STACK,
        text=stack,
        pattern=r"\bvar out = List\[List\[Float32\]\]\(\)",
        symbol="_modvecs_from_block.out",
        detail="modulation chunks are stored as nested host F32 lists",
        scope=("def", "_modvecs_from_block"),
    )
    add_boundary_hits(
        blockers,
        category="modulation",
        path=STACK,
        text=stack,
        pattern=r"^\s*\) raises -> List\[List\[Float32\]\]",
        symbol="_compute_final_modvecs",
        detail="final modulation helper returns nested host F32 lists",
        scope=("def", "_compute_final_modvecs"),
    )
    add_boundary_hits(
        blockers,
        category="modulation",
        path=STACK,
        text=stack,
        pattern=r"\.to_host\(\s*ctx\s*\)",
        symbol="_compute_final_modvecs.fmods",
        detail="final scale/shift linear output crosses back to host F32",
        scope=("def", "_compute_final_modvecs"),
    )
    add_boundary_hits(
        blockers,
        category="modulation",
        path=STACK,
        text=stack,
        pattern=r"\bvar f(?:scale|shift) = List\[Float32\]\(\)",
        symbol="_compute_final_modvecs.final_scale_shift",
        detail="final scale/shift are stored as host F32 lists",
        scope=("def", "_compute_final_modvecs"),
    )
    add_boundary_hits(
        blockers,
        category="modulation",
        path=WEIGHTS,
        text=weights,
        pattern=r"\btemb_h: List\[Float32\]",
        symbol="modvecs_from_temb.temb_h",
        detail="non-offload modulation helper accepts host F32 timestep activations",
        scope=("def", "modvecs_from_temb"),
        max_hits=1,
    )
    add_boundary_hits(
        blockers,
        category="modulation",
        path=WEIGHTS,
        text=weights,
        pattern=r"\bSTDtype\.F32\b",
        symbol="modvecs_from_temb.STDtype.F32",
        detail="non-offload modulation helper creates an explicit F32 tensor boundary",
        scope=("def", "modvecs_from_temb"),
    )
    add_boundary_hits(
        blockers,
        category="modulation",
        path=WEIGHTS,
        text=weights,
        pattern=r"\.to_host\(\s*ctx\s*\)",
        symbol="modvecs_from_temb.mods",
        detail="non-offload modulation output crosses back to host F32",
        scope=("def", "modvecs_from_temb"),
    )
    add_boundary_hits(
        blockers,
        category="modulation",
        path=WEIGHTS,
        text=weights,
        pattern=r"^\s{4}var final_(?:scale|shift): List\[Float32\]",
        symbol="QwenPerBlockMods.final_scale_shift",
        detail="precomputed final modulation vectors are stored as host F32 lists",
        scope=("struct", "QwenPerBlockMods"),
    )
    add_boundary_hits(
        blockers,
        category="modulation",
        path=WEIGHTS,
        text=weights,
        pattern=r"\bSTDtype\.F32\b",
        symbol="build_qwen_per_block_mods.STDtype.F32",
        detail="precomputed final modulation helper creates an explicit F32 tensor boundary",
        scope=("def", "build_qwen_per_block_mods"),
    )
    add_boundary_hits(
        blockers,
        category="modulation",
        path=WEIGHTS,
        text=weights,
        pattern=r"\.to_host\(\s*ctx\s*\)",
        symbol="build_qwen_per_block_mods.fmods",
        detail="precomputed final modulation helper returns F32 host vectors",
        scope=("def", "build_qwen_per_block_mods"),
    )

    add_boundary_hits(
        blockers,
        category="activation-tape",
        path=STACK,
        text=stack,
        pattern=r"\bvar out: List\[Float32\]",
        symbol="QwenOffloadForward.out",
        detail="final forward prediction is stored as host List[Float32]",
        scope=("struct", "QwenOffloadForward"),
        max_hits=1,
    )
    add_boundary_hits(
        blockers,
        category="activation-tape",
        path=STACK,
        text=stack,
        pattern=r"\bvar dbl_img_in: List\[List\[Float32\]\]",
        symbol="QwenOffloadForward.dbl_img_in",
        detail="per-block image inputs are stored as nested host F32 lists",
        scope=("struct", "QwenOffloadForward"),
        max_hits=1,
    )
    add_boundary_hits(
        blockers,
        category="activation-tape",
        path=STACK,
        text=stack,
        pattern=r"\bvar dbl_txt_in: List\[List\[Float32\]\]",
        symbol="QwenOffloadForward.dbl_txt_in",
        detail="per-block text inputs are stored as nested host F32 lists",
        scope=("struct", "QwenOffloadForward"),
        max_hits=1,
    )
    add_boundary_hits(
        blockers,
        category="activation-tape",
        path=STACK,
        text=stack,
        pattern=r"^\s{4}var final_(?:scale|shift): List\[Float32\]",
        symbol="QwenOffloadForward.final_scale_shift",
        detail="saved final modulation vectors are stored as host F32 lists",
        scope=("struct", "QwenOffloadForward"),
    )
    add_boundary_hits(
        blockers,
        category="activation-tape",
        path=STACK,
        text=stack,
        pattern=r"\bvar dbl_img_in = List\[List\[Float32\]\]\(\)",
        symbol="forward_offload.dbl_img_in",
        detail="forward allocates host F32 block-input tape",
        scope=("def", "qwenimage_stack_lora_forward_offload"),
    )
    add_boundary_hits(
        blockers,
        category="activation-tape",
        path=STACK,
        text=stack,
        pattern=r"\bvar dbl_txt_in = List\[List\[Float32\]\]\(\)",
        symbol="forward_offload.dbl_txt_in",
        detail="forward allocates host F32 text-input tape",
        scope=("def", "qwenimage_stack_lora_forward_offload"),
    )
    add_boundary_hits(
        blockers,
        category="activation-tape",
        path=STACK,
        text=stack,
        pattern=r"\bdbl_(?:img|txt)_in\.append\(",
        symbol="forward_offload.saved_block_inputs",
        detail="forward stores per-block inputs as host F32 copies",
        scope=("def", "qwenimage_stack_lora_forward_offload"),
    )
    add_boundary_hits(
        blockers,
        category="activation-tape",
        path=STACK,
        text=stack,
        pattern=r"\.to_host\(\s*ctx\s*\)",
        symbol="forward_offload.D2H",
        detail="forward still stages activations through host F32",
        scope=("def", "qwenimage_stack_lora_forward_offload"),
    )

    add_boundary_hits(
        blockers,
        category="gradient-tape",
        path=STACK,
        text=stack,
        pattern=r"\bvar d_a: List\[List\[Float32\]\]",
        symbol="QwenLoraGradSet.d_a",
        detail="LoRA A gradients are stored as nested host F32 lists",
        scope=("struct", "QwenLoraGradSet"),
        max_hits=1,
    )
    add_boundary_hits(
        blockers,
        category="gradient-tape",
        path=STACK,
        text=stack,
        pattern=r"\bvar d_b: List\[List\[Float32\]\]",
        symbol="QwenLoraGradSet.d_b",
        detail="LoRA B gradients are stored as nested host F32 lists",
        scope=("struct", "QwenLoraGradSet"),
        max_hits=1,
    )
    add_boundary_hits(
        blockers,
        category="gradient-tape",
        path=STACK,
        text=stack,
        pattern=r"\bvar d_img_tokens: List\[Float32\]",
        symbol="QwenLoraGradSet.d_img_tokens",
        detail="image-token gradients are stored as host F32 lists",
        scope=("struct", "QwenLoraGradSet"),
        max_hits=1,
    )
    add_boundary_hits(
        blockers,
        category="gradient-tape",
        path=STACK,
        text=stack,
        pattern=r"\bvar d_txt_tokens: List\[Float32\]",
        symbol="QwenLoraGradSet.d_txt_tokens",
        detail="text-token gradients are stored as host F32 lists",
        scope=("struct", "QwenLoraGradSet"),
        max_hits=1,
    )
    add_boundary_hits(
        blockers,
        category="gradient-tape",
        path=STACK,
        text=stack,
        pattern=r"\bvar d_[ab]_flat = List\[List\[Float32\]\]\(\)",
        symbol="backward_offload.d_a_b_flat",
        detail="backward allocates nested host F32 LoRA-gradient accumulators",
        scope=("def", "qwenimage_stack_lora_backward_offload"),
    )
    add_boundary_hits(
        blockers,
        category="gradient-tape",
        path=STACK,
        text=stack,
        pattern=r"\.to_host\(\s*ctx\s*\)",
        symbol="backward_offload.D2H",
        detail="backward still stages activations/gradients through host F32",
        scope=("def", "qwenimage_stack_lora_backward_offload"),
    )

    return blockers


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strict-train-boundaries",
        action="store_true",
        help="fail while known Qwen train/offload activation or grad carriers remain host-F32",
    )
    args = parser.parse_args()

    weights = read(WEIGHTS)
    stack = read(STACK)
    train = read(TRAIN)
    smoke = read(SMOKE)

    blockers = loader_blockers(weights, stack, smoke)
    train_blockers = train_boundary_blockers(weights, stack, train)

    print("[qwenimage-dtype-contract] loader blockers:", len(blockers))
    for blocker in blockers:
        print("[qwenimage-dtype-contract] FAIL:", blocker)
    if blockers:
        return 1

    print("[qwenimage-dtype-contract] train-boundary blockers:", len(train_blockers))
    for category, category_blockers in grouped_boundary_blockers(train_blockers):
        print(
            "[qwenimage-dtype-contract] WARN train-boundary "
            f"{category}: {len(category_blockers)}"
        )
        for blocker in category_blockers:
            print(
                "[qwenimage-dtype-contract]   "
                f"{rel(blocker.path)}:{blocker.line} "
                f"{blocker.symbol}: {blocker.detail} :: {blocker.code}"
            )
    if args.strict_train_boundaries and train_blockers:
        print("[qwenimage-dtype-contract] FAIL strict train-boundaries")
        return 1
    if train_blockers:
        print("[qwenimage-dtype-contract] PASS loader gate; train-boundaries are report-only")
    else:
        print("[qwenimage-dtype-contract] PASS loader gate and train-boundary audit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

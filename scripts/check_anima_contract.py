#!/usr/bin/env python3
"""Static dtype guard for the Anima checkpoint loader and train boundaries.

Default mode keeps the accepted loader/storage gate strict and reports known
Anima train-boundary host-F32 carriers as warnings. Use
--strict-train-boundaries to make those non-production activation/grad carriers
fatal.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]

WEIGHTS = REPO / "serenitymojo/models/anima/weights.mojo"
STACK = REPO / "serenitymojo/models/anima/anima_stack.mojo"
LORA_STACK = REPO / "serenitymojo/models/anima/anima_stack_lora.mojo"
BLOCK = REPO / "serenitymojo/models/anima/block.mojo"
LORA_BLOCK = REPO / "serenitymojo/models/anima/lora_block.mojo"

TRAIN_FILES = tuple(sorted((REPO / "../serenity-trainer/src/serenity_trainer/trainer").glob("train_anima*.mojo")))
PIPELINE_FILES = tuple(sorted((REPO / "serenitymojo/pipeline").glob("anima*.mojo")))
MODEL_FILES = tuple(sorted((REPO / "serenitymojo/models/anima").glob("*.mojo")))


@dataclass(frozen=True)
class Source:
    path: Path
    text: str

    @property
    def rel(self) -> str:
        return str(self.path.relative_to(REPO))

    def line_for_offset(self, offset: int) -> int:
        return self.text.count("\n", 0, offset) + 1

    def scope_for_offset(self, offset: int) -> str:
        last: tuple[int, str, str] | None = None
        for match in re.finditer(r"^(def|struct)\s+(\w+)", self.text, flags=re.MULTILINE):
            if match.start() > offset:
                break
            last = (match.start(), match.group(1), match.group(2))
        if last is None:
            return "module"
        return f"{last[1]} {last[2]}"


@dataclass(frozen=True)
class Block:
    source: Source
    keyword: str
    name: str
    text: str
    code: str
    start_offset: int

    @property
    def start_line(self) -> int:
        return self.source.line_for_offset(self.start_offset)

    @property
    def scope(self) -> str:
        return f"{self.keyword} {self.name}"

    def line_for(self, pattern: str, *, regex: bool = False) -> int:
        if regex:
            match = re.search(pattern, self.text, flags=re.MULTILINE | re.DOTALL)
            if match:
                return self.source.line_for_offset(self.start_offset + match.start())
        else:
            idx = self.text.find(pattern)
            if idx >= 0:
                return self.source.line_for_offset(self.start_offset + idx)
        return self.start_line


@dataclass(frozen=True)
class Finding:
    category: str
    rel: str
    line: int
    scope: str
    message: str

    def format(self) -> str:
        return f"{self.rel}:{self.line}: {self.scope}: {self.message}"


def read_source(path: Path) -> Source:
    if not path.exists():
        raise SystemExit(f"missing required file: {path}")
    return Source(path, path.read_text(encoding="utf-8"))


def strip_comments(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        lines.append(line.split("#", 1)[0])
    return "\n".join(lines)


def top_level_span(text: str, keyword: str, name: str) -> tuple[int, int] | None:
    match = re.search(rf"^{keyword}\s+{re.escape(name)}\b", text, flags=re.MULTILINE)
    if not match:
        return None
    boundary = re.search(
        r"^(?:def|struct|alias|comptime|from|import)\s+\w+\b",
        text[match.end() :],
        flags=re.MULTILINE,
    )
    end = len(text) if boundary is None else match.end() + boundary.start()
    return match.start(), end


def maybe_top_level_body(source: Source, keyword: str, name: str) -> Block | None:
    span = top_level_span(source.text, keyword, name)
    if span is None:
        return None
    start, end = span
    text = source.text[start:end]
    return Block(source, keyword, name, text, strip_comments(text), start)


def top_level_body(source: Source, keyword: str, name: str) -> Block:
    block = maybe_top_level_body(source, keyword, name)
    if block is None:
        raise SystemExit(f"missing {keyword} {name} in {source.rel}")
    return block


def maybe_function_body(source: Source, name: str) -> Block | None:
    return maybe_top_level_body(source, "def", name)


def maybe_struct_body(source: Source, name: str) -> Block | None:
    return maybe_top_level_body(source, "struct", name)


def function_body(source: Source, name: str) -> Block:
    return top_level_body(source, "def", name)


def has(pattern: str, text: str) -> bool:
    return re.search(pattern, text, flags=re.MULTILINE | re.DOTALL) is not None


def add(
    findings: list[Finding],
    category: str,
    source: Source,
    line: int,
    scope: str,
    message: str,
) -> None:
    finding = Finding(category, source.rel, line, scope, message)
    if finding not in findings:
        findings.append(finding)


def add_absent(
    findings: list[Finding],
    block: Block,
    pattern: str,
    message: str,
    *,
    category: str = "loader",
) -> None:
    if pattern not in block.code:
        add(findings, category, block.source, block.start_line, block.scope, message)


def add_count_less_than(
    findings: list[Finding],
    block: Block,
    pattern: str,
    minimum: int,
    message: str,
    *,
    category: str = "loader",
) -> None:
    if block.code.count(pattern) < minimum:
        add(findings, category, block.source, block.start_line, block.scope, message)


def add_if_present(
    findings: list[Finding],
    block: Block | None,
    patterns: tuple[str, ...],
    message: str,
    *,
    category: str,
) -> None:
    if block is None:
        return
    for pattern in patterns:
        if pattern in block.code:
            add(findings, category, block.source, block.line_for(pattern), block.scope, message)
            return


def add_if_regex(
    findings: list[Finding],
    block: Block | None,
    pattern: str,
    message: str,
    *,
    category: str,
) -> None:
    if block is None:
        return
    match = re.search(pattern, block.code, flags=re.MULTILINE | re.DOTALL)
    if match:
        add(findings, category, block.source, block.line_for(pattern, regex=True), block.scope, message)


def add_source_match(
    findings: list[Finding],
    category: str,
    source: Source,
    pattern: str,
    message: str,
) -> None:
    match = re.search(pattern, strip_comments(source.text), flags=re.MULTILINE | re.DOTALL)
    if not match:
        return
    line = source.line_for_offset(match.start())
    add(findings, category, source, line, source.scope_for_offset(match.start()), message)


def add_loader_caller_guard(
    blockers: list[Finding],
    block: Block | None,
    message: str,
) -> None:
    if block is None:
        return
    if not has(r"\bload_anima_(?:all_blocks|block_weights)", block.code):
        return
    if (
        "load_anima_block_weights_bf16_normf32" not in block.code
        and "load_anima_all_blocks_bf16_normf32" not in block.code
    ):
        add(blockers, "loader", block.source, block.start_line, block.scope, message)


def loader_blockers(
    weights: Source,
    lora_stack: Source,
    train_sources: list[Source],
    pipeline_sources: list[Source],
) -> list[Finding]:
    blockers: list[Finding] = []

    stored = function_body(weights, "_load_tensor")
    add_absent(
        blockers,
        stored,
        "from_parts(info.dtype, info.shape.copy(), bytes)",
        "_load_tensor no longer builds tensor views from checkpoint dtype metadata",
    )
    add_absent(
        blockers,
        stored,
        "Tensor.from_view(tv, ctx)",
        "_load_tensor no longer uploads checkpoint storage through Tensor.from_view",
    )
    add_if_present(
        blockers,
        stored,
        ("cast_tensor", "STDtype.F32", ".to_host(ctx)", "Tensor.from_host"),
        "_load_tensor materializes checkpoint tensors across an F32 or host boundary",
        category="loader",
    )

    for name in (
        "load_anima_block_weights",
        "load_anima_block_weights_f32",
        "load_anima_block_weights_bf16_normf32",
    ):
        block = function_body(weights, name)
        add_count_less_than(
            blockers,
            block,
            "_load_tensor(",
            20,
            f"{name} no longer routes all 20 block tensors through the stored-dtype loader",
        )
        add_if_present(
            blockers,
            block,
            ("cast_tensor", "STDtype.F32", ".to_host(ctx)", "Tensor.from_host"),
            f"{name} creates an F32 or host checkpoint boundary",
            category="loader",
        )

    base_loader = function_body(weights, "load_anima_stack_base")
    add_count_less_than(
        blockers,
        base_loader,
        "_load_tensor(",
        7,
        "load_anima_stack_base no longer routes all base tensors through the stored-dtype loader",
    )
    add_if_present(
        blockers,
        base_loader,
        ("cast_tensor", "STDtype.F32", ".to_host(ctx)", "Tensor.from_host"),
        "load_anima_stack_base creates an F32 or host checkpoint boundary",
        category="loader",
    )

    all_bf16 = function_body(weights, "load_anima_all_blocks_bf16_normf32")
    add_absent(
        blockers,
        all_bf16,
        "load_anima_block_weights_bf16_normf32",
        "load_anima_all_blocks_bf16_normf32 no longer delegates to the accepted dtype-preserving block loader",
    )

    for name in ("anima_stack_lora_forward_streamed", "anima_stack_lora_backward_streamed"):
        block = maybe_function_body(lora_stack, name)
        if block is None:
            continue
        if "load_anima_block_weights" in block.code:
            add_absent(
                blockers,
                block,
                "load_anima_block_weights_f32",
                f"{name} calls a streamed block loader but not the accepted dtype-preserving legacy symbol",
            )

    for train in train_sources:
        add_loader_caller_guard(
            blockers,
            maybe_function_body(train, "main"),
            "train main calls an Anima block loader but not the accepted BF16/norm dtype loader",
        )

    for pipe in pipeline_sources:
        add_loader_caller_guard(
            blockers,
            maybe_function_body(pipe, "main"),
            "pipeline main calls an Anima block loader but not the accepted BF16/norm dtype loader",
        )
        add_loader_caller_guard(
            blockers,
            maybe_function_body(pipe, "_load_and_denoise"),
            "pipeline denoise path calls an Anima block loader but not the accepted BF16/norm dtype loader",
        )

    return blockers


def train_boundary_blockers(sources: dict[Path, Source]) -> list[Finding]:
    blockers: list[Finding] = []

    stack = sources[STACK]
    lora_stack = sources[LORA_STACK]
    block = sources[BLOCK]
    lora_block = sources[LORA_BLOCK]

    add_if_present(
        blockers,
        maybe_struct_body(block, "AnimaBlockForward"),
        ("var out: List[Float32]",),
        "AnimaBlockForward stores block outputs as host List[Float32]",
        category="saved-activation",
    )
    add_if_present(
        blockers,
        maybe_struct_body(block, "AnimaBlockGrads"),
        ("var d_x: List[Float32]", "var d_t_silu: List[Float32]"),
        "AnimaBlockGrads stores activation/shared gradients as host List[Float32]",
        category="grad-boundary",
    )
    add_if_present(
        blockers,
        maybe_struct_body(block, "AnimaBlockGrads"),
        ("var d_sa_q: List[Float32]", "var d_mlp2: List[Float32]"),
        "AnimaBlockGrads stores full-finetune/base weight gradients as host List[Float32]",
        category="grad-boundary",
    )
    for name in ("anima_block_forward", "anima_block_backward"):
        add_if_present(
            blockers,
            maybe_function_body(block, name),
            (".to_host(ctx)",),
            f"{name} stages block activations/grads through host F32 readback",
            category="host-f32-boundary",
        )
    add_if_present(
        blockers,
        maybe_struct_body(lora_block, "AnimaLoraGrads"),
        ("var d_a: List[Float32]", "var d_b: List[Float32]"),
        "AnimaLoraGrads stores adapter gradients as host List[Float32]",
        category="lora-boundary",
    )
    add_if_present(
        blockers,
        maybe_struct_body(lora_block, "AnimaBlockLoraGrads"),
        ("var d_a: List[List[Float32]]", "var d_b: List[List[Float32]]"),
        "AnimaBlockLoraGrads stores per-slot LoRA gradients as host List[List[Float32]]",
        category="lora-boundary",
    )
    for name in (
        "anima_lora_fwd",
        "anima_lora_apply",
        "anima_lora_bwd",
        "anima_block_lora_forward",
        "anima_block_lora_backward",
    ):
        add_if_regex(
            blockers,
            maybe_function_body(lora_block, name),
            r"\bList\[Float32\]|\.to_host\(ctx\)",
            f"{name} exposes LoRA/activation values through host F32 carriers",
            category="lora-boundary",
        )
    add_if_present(
        blockers,
        maybe_function_body(lora_block, "_proj_lora_grads"),
        ("d_y_h: List[Float32]", "x_in_h: List[Float32]", "base_dx_h: List[Float32]"),
        "_proj_lora_grads requires projection activations/grads as host F32 lists",
        category="lora-boundary",
    )

    add_if_present(
        blockers,
        maybe_struct_body(stack, "AnimaStackForward"),
        ("var out: List[Float32]", "var x_emb: List[Float32]"),
        "AnimaStackForward stores stack output/input activations as host List[Float32]",
        category="saved-activation",
    )
    add_if_present(
        blockers,
        maybe_struct_body(stack, "AnimaStackGrads"),
        ("var d_patches: List[Float32]", "var d_t_silu: List[Float32]"),
        "AnimaStackGrads stores activation/shared gradients as host List[Float32]",
        category="grad-boundary",
    )
    add_if_present(
        blockers,
        maybe_function_body(stack, "_t"),
        ("Tensor.from_host(vals, shape^, STDtype.F32, ctx)",),
        "_t uploads train activations from host List[Float32] as STDtype.F32",
        category="host-f32-boundary",
    )
    add_if_present(
        blockers,
        maybe_function_body(stack, "_linear_wdev"),
        (".to_host(ctx)", "-> List[Float32]"),
        "_linear_wdev returns projection activations through host F32 readback",
        category="host-f32-boundary",
    )
    for name in ("anima_stack_forward", "anima_stack_backward"):
        add_if_regex(
            blockers,
            maybe_function_body(stack, name),
            r"\bList\[Float32\]|\.to_host\(ctx\)",
            f"{name} exposes stack train activations/grads through host F32 carriers",
            category="host-f32-boundary",
        )

    add_if_present(
        blockers,
        maybe_struct_body(lora_stack, "AnimaLoraSet"),
        ("var ad: List[LoraAdapter]",),
        "AnimaLoraSet keeps adapter factors and AdamW moments in host Float32 lists",
        category="lora-boundary",
    )
    add_if_present(
        blockers,
        maybe_struct_body(lora_stack, "AnimaLoraGrads"),
        ("var d_a: List[List[Float32]]", "var d_b: List[List[Float32]]"),
        "AnimaLoraGrads stores all LoRA gradients as host List[List[Float32]]",
        category="lora-boundary",
    )
    add_if_present(
        blockers,
        maybe_struct_body(lora_stack, "_AnimaHostGradLists"),
        ("var d_a: List[List[Float32]]", "var d_b: List[List[Float32]]"),
        "_AnimaHostGradLists materializes device LoRA grads as host F32 lists",
        category="lora-boundary",
    )
    add_if_present(
        blockers,
        maybe_function_body(lora_stack, "_host_grad_slice"),
        ("bitcast[Float32]()", "-> List[Float32]"),
        "_host_grad_slice decodes device grad buffers into host Float32 lists",
        category="lora-boundary",
    )
    add_if_present(
        blockers,
        maybe_function_body(lora_stack, "_grad_arc_f32"),
        ("cast_tensor(t[], STDtype.F32, ctx)",),
        "_grad_arc_f32 casts device LoRA gradients to F32 before host readback",
        category="lora-boundary",
    )
    add_if_present(
        blockers,
        maybe_function_body(lora_stack, "_anima_tensor_grads_to_host"),
        ("var d_a_flat = List[List[Float32]]()", "ctx.enqueue_create_host_buffer"),
        "_anima_tensor_grads_to_host bulk-copies LoRA grad tensors into host F32 carriers",
        category="lora-boundary",
    )
    for name in (
        "anima_stack_lora_forward",
        "anima_stack_lora_forward_streamed",
        "anima_stack_lora_forward_device_resident",
        "anima_stack_lora_forward_device_resident_nosave",
        "anima_stack_lora_predict_device_resident",
    ):
        add_if_regex(
            blockers,
            maybe_function_body(lora_stack, name),
            r"\bList\[Float32\]|\.to_host\(ctx\)",
            f"{name} exposes LoRA forward activations/output through host F32 carriers",
            category="host-f32-boundary",
        )
    for name in (
        "anima_stack_lora_backward",
        "anima_stack_lora_backward_streamed",
        "anima_stack_lora_backward_device_resident",
    ):
        add_if_regex(
            blockers,
            maybe_function_body(lora_stack, name),
            r"\bList\[Float32\]|\.to_host\(ctx\)|_anima_tensor_grads_to_host",
            f"{name} stages LoRA backward grads/activations through host F32 boundaries",
            category="lora-boundary",
        )
    add_if_present(
        blockers,
        maybe_function_body(lora_stack, "anima_lora_adamw_step"),
        ("LoraGrads(grads.d_a[i].copy(), grads.d_b[i].copy())", "_lora_adamw"),
        "anima_lora_adamw_step steps adapters through host F32 gradient carriers",
        category="optimizer-boundary",
    )

    for path in TRAIN_FILES:
        train = sources[path]
        add_if_present(
            blockers,
            maybe_function_body(train, "_cache_f32"),
            ("cast_tensor(t^, STDtype.F32, ctx)", "return cast_tensor"),
            "_cache_f32 widens cached latent/context tensors to F32",
            category="cache-boundary",
        )
        add_if_present(
            blockers,
            maybe_struct_body(train, "_TEmb"),
            ("var t_cond: List[Float32]", "var base_adaln: List[Float32]"),
            "_TEmb stores timestep/base AdaLN activations as host List[Float32]",
            category="host-f32-boundary",
        )
        add_if_present(
            blockers,
            maybe_function_body(train, "_prepare_timestep"),
            ("Tensor.from_host(emb_l, [B, D], STDtype.F32, ctx)", ".to_host(ctx)"),
            "_prepare_timestep returns t_cond/base_adaln through host F32 boundaries",
            category="host-f32-boundary",
        )
        add_if_present(
            blockers,
            maybe_function_body(train, "_host_noise"),
            ("-> List[Float32]", "var out = List[Float32]()"),
            "_host_noise generates training noise as host List[Float32]",
            category="cache-boundary",
        )
        for name in ("_patchify_in", "_patchify_out", "_load_giger3_sample"):
            add_if_regex(
                blockers,
                maybe_function_body(train, name),
                r"\bList\[Float32\]",
                f"{name} carries latent/image/context train data as host List[Float32]",
                category="cache-boundary",
            )
        main = maybe_function_body(train, "main")
        add_if_present(
            blockers,
            main,
            ("_cache_f32(cache", "_cache_f32(ctx_st"),
            "main reads train cache tensors through _cache_f32 host-F32 path",
            category="cache-boundary",
        )
        add_if_present(
            blockers,
            main,
            ("var noisy = List[Float32]()", "var target = List[Float32]()"),
            "main stores noisy latents/targets as host List[Float32]",
            category="cache-boundary",
        )
        add_if_present(
            blockers,
            main,
            ("var d_out = List[Float32]()", "var pred = fwd.out.copy()"),
            "main computes loss/upstream gradients on host List[Float32]",
            category="grad-boundary",
        )

    for path in PIPELINE_FILES:
        pipe = sources[path]
        if "prepare" in path.name:
            main = maybe_function_body(pipe, "main")
            add_if_present(
                blockers,
                main,
                ("lat.to_host(ctx)", "Tensor.from_host(lh.copy()"),
                "prepare path stages VAE latents through host list before cache write",
                category="prepare-boundary",
            )
        add_if_present(
            blockers,
            maybe_function_body(pipe, "anima_text_context_from_tokens"),
            ("cast_tensor(last_hidden, STDtype.F32, ctx)", "context.to_host(ctx)"),
            "text-context pipeline widens/stages conditioning activations as host F32",
            category="pipeline-boundary",
        )
        add_if_present(
            blockers,
            maybe_function_body(pipe, "_load_context_512"),
            ("cast_tensor(t^, STDtype.F32, ctx).to_host(ctx)",),
            "_load_context_512 loads cached context through host F32",
            category="pipeline-boundary",
        )
        add_if_regex(
            blockers,
            maybe_function_body(pipe, "_denoise"),
            r"\bList\[Float32\]|\.to_host\(ctx\)",
            "_denoise carries denoise latents/velocity through host F32 lists",
            category="pipeline-boundary",
        )

    for path in (*MODEL_FILES, *TRAIN_FILES, *PIPELINE_FILES):
        src = sources[path]
        if path.name in {"weights.mojo", "config.mojo", "__init__.mojo"}:
            continue
        add_source_match(
            blockers,
            "typed-f32-boundary",
            src,
            r"Tensor\[DType\.float32\]",
            "production Anima file declares an explicit Tensor[DType.float32] boundary",
        )

    return blockers


def print_categories(prefix: str, findings: list[Finding]) -> None:
    counts: dict[str, int] = {}
    for finding in findings:
        counts[finding.category] = counts.get(finding.category, 0) + 1
    if not counts:
        return
    print(f"{prefix} blocker categories:")
    for category in sorted(counts):
        print(f"{prefix}   {category}: {counts[category]}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strict-train-boundaries",
        action="store_true",
        help="fail while known Anima train/offload activation or grad carriers remain host-F32",
    )
    args = parser.parse_args()

    required_paths = {WEIGHTS, STACK, LORA_STACK, BLOCK, LORA_BLOCK, *TRAIN_FILES, *PIPELINE_FILES, *MODEL_FILES}
    sources = {path: read_source(path) for path in sorted(required_paths)}

    train_sources = [sources[path] for path in TRAIN_FILES]
    pipeline_sources = [sources[path] for path in PIPELINE_FILES]
    loader = loader_blockers(sources[WEIGHTS], sources[LORA_STACK], train_sources, pipeline_sources)
    train = train_boundary_blockers(sources)

    prefix = "[anima-dtype-contract]"
    print(prefix, "scanned model files:", len(MODEL_FILES))
    print(prefix, "scanned train files:", len(TRAIN_FILES))
    print(prefix, "scanned pipeline files:", len(PIPELINE_FILES))
    print(prefix, "loader blockers:", len(loader))
    for blocker in loader:
        print(prefix, "FAIL:", blocker.format())
    if loader:
        print_categories(prefix, loader)
        return 1

    print(prefix, "train-boundary blockers:", len(train))
    for blocker in train:
        print(prefix, "WARN train-boundary:", blocker.format())
    print_categories(prefix, train)
    if args.strict_train_boundaries and train:
        print(prefix, "FAIL strict train-boundaries")
        return 1
    if train:
        print(prefix, "PASS loader gate; train-boundaries are report-only")
    else:
        print(prefix, "PASS loader gate and train-boundary audit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

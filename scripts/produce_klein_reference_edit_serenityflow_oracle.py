#!/usr/bin/env python3
"""Produce SerenityFlow oracle artifacts for Klein ReferenceLatent edit parity.

This script is intentionally a Python oracle bridge, not a production path.
It imports SerenityFlow's FLUX.2/Klein helpers, writes Mojo cap-cache tensor
bins, and records enough artifacts to compare the Mojo ReferenceLatent edit
dump against the source-of-truth Python path.

The useful no-model mode dumps VAE/reference/noise packing artifacts. Passing
--run-model with model/precomputed-text/VAE inputs also records the Euler
trajectory, final latents, decoded tensor, and PNG. Live text encoder loading is
opt-in because Klein's Qwen text path is large and has caused machine reboots.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


MAGIC = 0x4B4C4E4341505631
DTYPE_TAGS: dict[torch.dtype, int] = {
    torch.bool: 0,
    torch.uint8: 1,
    torch.int8: 2,
    torch.int16: 5,
    torch.float16: 7,
    torch.bfloat16: 8,
    torch.int32: 9,
    torch.float32: 11,
    torch.float64: 12,
    torch.int64: 13,
}
TAG_DTYPES: dict[int, torch.dtype] = {tag: dtype for dtype, tag in DTYPE_TAGS.items()}
DTYPE_NAMES: dict[torch.dtype, str] = {
    torch.bool: "BOOL",
    torch.uint8: "U8",
    torch.int8: "I8",
    torch.int16: "I16",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.int32: "I32",
    torch.float32: "F32",
    torch.float64: "F64",
    torch.int64: "I64",
}


@dataclass
class SerenityFlowAPI:
    patchify_flux2_latents: Any
    unpatchify_flux2_latents: Any
    pack_flux2_latents: Any
    unpack_flux2_latents: Any
    prepare_flux2_image_ids: Any
    prepare_flux2_text_ids: Any
    create_noise: Any
    encode_text: Any
    vae_encode: Any
    vae_decode: Any
    load_diffusion_model: Any
    load_vae: Any
    load_clip: Any
    apply_lora: Any
    extract_model_output: Any


def _dtype_name(dtype: torch.dtype) -> str:
    return DTYPE_NAMES.get(dtype, str(dtype).replace("torch.", "").upper())


def _dtype_from_name(name: str) -> torch.dtype:
    lowered = name.lower()
    if lowered in {"f32", "float32", "fp32"}:
        return torch.float32
    if lowered in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if lowered in {"f16", "float16", "fp16"}:
        return torch.float16
    raise ValueError(f"unsupported output dtype {name!r}")


def save_tensor_bin(path: Path, tensor: torch.Tensor, *, dtype: torch.dtype | None = None) -> dict[str, Any]:
    """Write Mojo's KLNCAPV1 raw tensor-bin format."""
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    tensor = tensor.detach().contiguous().cpu()
    if tensor.dtype not in DTYPE_TAGS:
        raise ValueError(f"unsupported tensor-bin dtype {tensor.dtype}")

    path.parent.mkdir(parents=True, exist_ok=True)
    shape = list(tensor.shape)
    raw = tensor.view(torch.uint8).numpy().tobytes()
    with path.open("wb") as f:
        f.write(struct.pack("<qqq", MAGIC, DTYPE_TAGS[tensor.dtype], len(shape)))
        if shape:
            f.write(struct.pack("<" + "q" * len(shape), *shape))
        f.write(raw)
    return {
        "path": str(path),
        "dtype": _dtype_name(tensor.dtype),
        "shape": shape,
        "bytes": path.stat().st_size,
    }


def load_tensor_bin(path: Path) -> torch.Tensor:
    """Read Mojo's KLNCAPV1 raw tensor-bin format into a CPU tensor."""
    data = path.read_bytes()
    if len(data) < 24:
        raise ValueError(f"{path}: tensor-bin header is too short")
    magic, dtype_tag, rank = struct.unpack_from("<qqq", data, 0)
    if magic != MAGIC:
        raise ValueError(f"{path}: bad magic 0x{magic:x}")
    if dtype_tag not in TAG_DTYPES:
        raise ValueError(f"{path}: unsupported dtype tag {dtype_tag}")
    dims_off = 24
    dims_end = dims_off + rank * 8
    if len(data) < dims_end:
        raise ValueError(f"{path}: dim header is truncated")
    shape = list(struct.unpack_from("<" + "q" * rank, data, dims_off)) if rank else []
    body = data[dims_end:]
    dtype = TAG_DTYPES[dtype_tag]
    byte_tensor = torch.frombuffer(bytearray(body), dtype=torch.uint8)
    return byte_tensor.view(dtype).reshape(shape).clone()


def import_serenityflow(root: Path) -> SerenityFlowAPI:
    root = root.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from serenityflow.bridge.sampling import (
        _extract_model_output,
        _pack_flux2_latents,
        _patchify_flux2_latents,
        _prepare_flux2_image_ids,
        _prepare_flux2_text_ids,
        _unpack_flux2_latents,
        _unpatchify_flux2_latents,
        encode_text,
        vae_decode,
        vae_encode,
    )
    from serenityflow.bridge.sampling_math import create_noise
    from serenityflow.bridge.serenity_api import (
        apply_lora,
        load_clip,
        load_diffusion_model,
        load_vae,
    )

    return SerenityFlowAPI(
        patchify_flux2_latents=_patchify_flux2_latents,
        unpatchify_flux2_latents=_unpatchify_flux2_latents,
        pack_flux2_latents=_pack_flux2_latents,
        unpack_flux2_latents=_unpack_flux2_latents,
        prepare_flux2_image_ids=_prepare_flux2_image_ids,
        prepare_flux2_text_ids=_prepare_flux2_text_ids,
        create_noise=create_noise,
        encode_text=encode_text,
        vae_encode=vae_encode,
        vae_decode=vae_decode,
        load_diffusion_model=load_diffusion_model,
        load_vae=load_vae,
        load_clip=load_clip,
        apply_lora=apply_lora,
        extract_model_output=_extract_model_output,
    )


def build_mojo_fixed_shift_sigmas(steps: int, shift: float, denoise: float) -> torch.Tensor:
    """Match serenitymojo.sampling.flux2_klein.build_flux2_img2img_sigmas."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    if denoise <= 0:
        raise ValueError("denoise must be positive")

    def fixed(num_steps: int) -> list[float]:
        n_sigmas = 10000
        exp_mu = math.exp(float(shift))
        ss = float(n_sigmas) / float(num_steps)
        out: list[float] = []
        for x in range(num_steps):
            idx = n_sigmas - 1 - int(float(x) * ss)
            t = float(idx + 1) / float(n_sigmas)
            out.append(exp_mu / (exp_mu + (1.0 / t - 1.0)))
        out.append(0.0)
        return out

    if denoise >= 0.9999:
        values = fixed(steps)
    else:
        full = fixed(int(float(steps) / float(denoise)))
        values = full[-(steps + 1):]
    return torch.tensor(values, dtype=torch.float32)


def build_serenityflow_flux2_sigmas(steps: int, shift: float, denoise: float) -> torch.Tensor:
    """Match SerenityFlow's Flux2Scheduler node output before _run_sampling."""
    total_steps = steps if denoise >= 1.0 else int(steps / denoise)
    sigmas = torch.linspace(1.0, 0.0, total_steps + 1, dtype=torch.float32)
    if shift != 1.0:
        sigmas = shift * sigmas / (1.0 + (shift - 1.0) * sigmas)
    if denoise < 1.0:
        sigmas = sigmas[-(steps + 1):]
    return sigmas


def flux_mu(seq_len: int) -> float:
    base_seq_len = 256
    max_seq_len = 4096
    base_shift = 0.5
    max_shift = 1.15
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return float(m * seq_len + b)


def apply_flux_exponential_shift(sigmas: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Match SerenityFlow FluxPrediction.apply_sigma_shift for flow_flux."""
    mu = flux_mu(seq_len)
    body = sigmas[:-1]
    shifted = math.exp(mu) / (math.exp(mu) + (1.0 / body - 1.0))
    return torch.cat([shifted, sigmas[-1:]])


def load_image_bchw(path: Path) -> torch.Tensor:
    from PIL import Image
    import numpy as np

    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype="float32") / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()


def save_png(path: Path, tensor_bchw: torch.Tensor) -> None:
    from PIL import Image
    import numpy as np

    t = tensor_bchw.detach().float().cpu().clamp(0.0, 1.0)
    if t.ndim != 4 or t.shape[0] != 1:
        raise ValueError(f"expected decoded image [1,C,H,W], got {tuple(t.shape)}")
    arr = (t[0].permute(1, 2, 0).numpy() * 255.0 + 0.5).astype("uint8")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def resolve_reference_latent(args: argparse.Namespace, api: SerenityFlowAPI) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Return raw [1,32,H/8,W/8], patchified [1,128,H/16,W/16], metadata."""
    raw_h = args.height // 8
    raw_w = args.width // 8
    patch_h = args.height // 16
    patch_w = args.width // 16
    source: dict[str, Any] = {}

    if args.synthetic_reference:
        raw = torch.zeros(1, 32, raw_h, raw_w, dtype=torch.float32)
        source["kind"] = "synthetic_zero_reference_not_oracle"
    elif args.reference_latent_bin:
        raw = load_tensor_bin(Path(args.reference_latent_bin)).float()
        source["kind"] = "raw_reference_latent_bin"
        source["path"] = str(args.reference_latent_bin)
    elif args.reference_patchified_bin:
        patch = load_tensor_bin(Path(args.reference_patchified_bin)).float()
        if list(patch.shape) != [1, 128, patch_h, patch_w]:
            raise ValueError(
                "--reference-patchified-bin must be [1,128,H/16,W/16], "
                f"got {tuple(patch.shape)}"
            )
        raw = api.unpatchify_flux2_latents(patch)
        source["kind"] = "patchified_reference_latent_bin"
        source["path"] = str(args.reference_patchified_bin)
    elif args.reference_image:
        if not args.vae:
            raise ValueError("--reference-image requires --vae so SerenityFlow can encode it")
        vae = api.load_vae(str(args.vae))
        image = load_image_bchw(Path(args.reference_image))
        if list(image.shape[-2:]) != [args.height, args.width]:
            raise ValueError(
                f"--reference-image size must match --width/--height; got {tuple(image.shape[-2:])}"
            )
        with torch.no_grad():
            raw = api.vae_encode(vae, image).detach().float().cpu()
        source["kind"] = "serenityflow_vae_encode_reference_image"
        source["path"] = str(args.reference_image)
        source["vae"] = str(args.vae)
    else:
        raise ValueError(
            "provide one of --reference-image, --reference-latent-bin, "
            "--reference-patchified-bin, or --synthetic-reference"
        )

    if list(raw.shape) != [1, 32, raw_h, raw_w]:
        raise ValueError(f"raw reference latent must be [1,32,{raw_h},{raw_w}], got {tuple(raw.shape)}")
    patch = api.patchify_flux2_latents(raw)
    return raw.contiguous(), patch.contiguous(), source


def normalize_token_shape(tokens_batched: torch.Tensor) -> torch.Tensor:
    if tokens_batched.ndim != 3 or tokens_batched.shape[0] != 1:
        raise ValueError(f"expected [1,N,C] tokens, got {tuple(tokens_batched.shape)}")
    return tokens_batched[0].contiguous()


def load_text_tokens(path: Path) -> torch.Tensor:
    tokens = load_tensor_bin(path).float()
    if tokens.ndim == 2:
        tokens = tokens.unsqueeze(0)
    if tokens.ndim != 3 or tokens.shape[0] != 1:
        raise ValueError(f"{path}: expected text tokens [512,D] or [1,512,D], got {tuple(tokens.shape)}")
    return tokens.contiguous()


def load_text_conditioning(args: argparse.Namespace, api: SerenityFlowAPI) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, str]:
    """Resolve conditioning without loading Qwen unless explicitly allowed."""
    if args.positive_text_bin:
        pos = load_text_tokens(args.positive_text_bin)
        if args.cfg > 1.0:
            if not args.negative_text_bin:
                raise ValueError("--cfg > 1 requires --negative-text-bin when using precomputed text")
            neg = load_text_tokens(args.negative_text_bin)
            if list(neg.shape) != list(pos.shape):
                raise ValueError(
                    "--negative-text-bin shape must match --positive-text-bin; "
                    f"got {tuple(neg.shape)} vs {tuple(pos.shape)}"
                )
            negative = [{"cross_attn": neg}]
        else:
            negative = []
        return ([{"cross_attn": pos}], negative, int(pos.shape[1]), "precomputed_text_bin")

    if not args.allow_live_text_encoder:
        raise ValueError(
            "--run-model requires --positive-text-bin by default. "
            "Live text encoder loading is disabled unless --allow-live-text-encoder is supplied."
        )
    if not args.clip:
        raise ValueError("--allow-live-text-encoder requires --clip")

    clip = api.load_clip(str(args.clip), clip_type=args.clip_type)
    positive = api.encode_text(clip, args.prompt)
    negative = api.encode_text(clip, args.negative_prompt) if args.cfg > 1.0 else []
    txt_len = (
        int(positive[0].get("cross_attn").shape[1])
        if positive and positive[0].get("cross_attn") is not None
        else 512
    )
    return positive, negative, txt_len, "live_text_encoder_opt_in"


def call_model_with_retry(fn: Any, *args: Any, **kwargs: Any) -> Any:
    attempted: set[str] = set()
    while True:
        try:
            return fn(*args, **kwargs)
        except TypeError as exc:
            msg = str(exc)
            if "unexpected keyword argument" not in msg:
                raise
            parts = msg.split("'")
            bad_kwarg = parts[1] if len(parts) >= 2 else None
            if bad_kwarg is None or bad_kwarg not in kwargs or bad_kwarg in attempted:
                raise
            attempted.add(bad_kwarg)
            kwargs = dict(kwargs)
            kwargs.pop(bad_kwarg, None)


def first_parameter_dtype(model: torch.nn.Module) -> torch.dtype:
    try:
        return next(model.parameters()).dtype
    except StopIteration:
        return torch.float32


def build_flux_kwargs(
    *,
    cond_entry: dict[str, Any],
    txt_ids: torch.Tensor,
    img_ids: torch.Tensor,
    model_dtype: torch.dtype,
    device: torch.device,
    guidance: float,
    model: torch.nn.Module,
) -> dict[str, Any]:
    hidden = cond_entry.get("cross_attn")
    pooled = cond_entry.get("pooled_output")
    kwargs: dict[str, Any] = {}
    if hidden is not None:
        kwargs["encoder_hidden_states"] = hidden.to(device=device, dtype=model_dtype)
    if pooled is not None:
        kwargs["pooled_projections"] = pooled.to(device=device, dtype=model_dtype)
    else:
        kwargs["pooled_projections"] = torch.zeros(1, 768, device=device, dtype=model_dtype)
    if hasattr(model, "config") and getattr(model.config, "guidance_embeds", True) is False:
        guidance = 0.0
    kwargs["guidance"] = torch.tensor([guidance], device=device, dtype=model_dtype)
    kwargs["img_ids"] = img_ids.to(device=device, dtype=model_dtype)
    kwargs["txt_ids"] = txt_ids.to(device=device, dtype=model_dtype)
    return kwargs


@torch.no_grad()
def run_reference_edit_euler(
    *,
    api: SerenityFlowAPI,
    model: torch.nn.Module,
    positive: list[dict[str, Any]],
    negative: list[dict[str, Any]],
    reference_tokens_batched: torch.Tensor,
    effective_initial_batched: torch.Tensor,
    img_ids: torch.Tensor,
    txt_ids: torch.Tensor,
    sigmas: torch.Tensor,
    cfg: float,
    guidance: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = first_parameter_dtype(model)
    x = effective_initial_batched.to(device=device, dtype=torch.float32)
    ref_tokens = reference_tokens_batched.to(device=device, dtype=x.dtype)
    sigmas = sigmas.to(device=device, dtype=torch.float32)

    cond_entry = positive[0] if positive else {}
    uncond_entry = negative[0] if negative else {}
    use_cfg = bool(negative and uncond_entry.get("cross_attn") is not None and cfg > 1.0)
    trajectory: list[torch.Tensor] = [normalize_token_shape(x.detach().cpu())]
    first_step: dict[str, Any] = {}

    denoise_t0 = time.perf_counter()
    for i in range(len(sigmas) - 1):
        sigma = sigmas[i]
        dt = sigmas[i + 1] - sigma
        model_input = x
        inp = torch.cat([model_input.to(dtype=model_dtype), ref_tokens.to(dtype=model_dtype)], dim=1)
        timestep = sigma.reshape(1).to(device=device, dtype=model_dtype)

        cond_kwargs = build_flux_kwargs(
            cond_entry=cond_entry,
            txt_ids=txt_ids,
            img_ids=img_ids,
            model_dtype=model_dtype,
            device=device,
            guidance=guidance,
            model=model,
        )
        raw_cond = call_model_with_retry(model, inp, timestep=timestep, **cond_kwargs)
        cond_out = api.extract_model_output(raw_cond).to(device=device, dtype=x.dtype)[:, : x.shape[1], :]

        if use_cfg:
            uncond_kwargs = build_flux_kwargs(
                cond_entry=uncond_entry,
                txt_ids=txt_ids,
                img_ids=img_ids,
                model_dtype=model_dtype,
                device=device,
                guidance=guidance,
                model=model,
            )
            raw_uncond = call_model_with_retry(model, inp, timestep=timestep, **uncond_kwargs)
            uncond_out = api.extract_model_output(raw_uncond).to(device=device, dtype=x.dtype)[:, : x.shape[1], :]
            velocity = uncond_out + (cond_out - uncond_out) * float(cfg)
        else:
            uncond_out = None
            velocity = cond_out

        denoised = x - velocity * sigma.reshape(1, 1, 1)
        derivative = (x - denoised) / sigma.reshape(1, 1, 1)
        if i == 0:
            first_step = {
                "python_first_step_cond_model_output": normalize_token_shape(cond_out.detach().cpu()),
                "python_first_step_cfg_velocity": normalize_token_shape(velocity.detach().cpu()),
                "python_first_step_denoised": normalize_token_shape(denoised.detach().cpu()),
                "python_first_step_euler_derivative": normalize_token_shape(derivative.detach().cpu()),
            }
            if uncond_out is not None:
                first_step["python_first_step_uncond_model_output"] = normalize_token_shape(
                    uncond_out.detach().cpu()
                )

        x = x + derivative * dt.reshape(1, 1, 1)
        trajectory.append(normalize_token_shape(x.detach().cpu()))

    denoise_seconds = time.perf_counter() - denoise_t0
    metrics = {
        "denoise_seconds": denoise_seconds,
        "denoise_seconds_per_step": denoise_seconds / max(len(sigmas) - 1, 1),
        "device": str(device),
        "model_dtype": _dtype_name(model_dtype),
        "use_cfg": use_cfg,
    }
    return x.detach().cpu(), torch.stack(trajectory, dim=0).contiguous(), {**metrics, **first_step}


def artifact_path(out_dir: Path, name: str) -> Path:
    return out_dir / name


def add_artifact(
    artifacts: dict[str, dict[str, Any]],
    out_dir: Path,
    key: str,
    tensor: torch.Tensor,
    *,
    filename: str | None = None,
    dtype: torch.dtype | None = torch.float32,
) -> None:
    path = artifact_path(out_dir, filename or f"{key}.bin")
    artifacts[key] = save_tensor_bin(path, tensor, dtype=dtype)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serenityflow-root", type=Path, default=Path("/home/alex/serenityflow-v2"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--edit-shift", type=float, default=2.02)
    parser.add_argument("--edit-denoise", type=float, default=1.0)
    parser.add_argument("--reference-t-offset", type=float, default=10.0)
    parser.add_argument(
        "--schedule-mode",
        choices=("mojo_fixed_shift", "serenityflow_flux2", "serenityflow_flux2_with_flow_shift"),
        default="mojo_fixed_shift",
    )
    parser.add_argument("--output-dtype", default="f32", choices=("f32", "bf16", "f16"))
    parser.add_argument("--reference-image", type=Path)
    parser.add_argument("--reference-latent-bin", type=Path)
    parser.add_argument("--reference-patchified-bin", type=Path)
    parser.add_argument("--synthetic-reference", action="store_true")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--vae", type=Path)
    parser.add_argument("--clip", type=Path)
    parser.add_argument("--clip-type", default="klein")
    parser.add_argument("--positive-text-bin", type=Path)
    parser.add_argument("--negative-text-bin", type=Path)
    parser.add_argument(
        "--allow-live-text-encoder",
        action="store_true",
        help="Opt in to SerenityFlow load_clip/encode_text. Disabled by default to avoid accidental Qwen loads.",
    )
    parser.add_argument("--lora", type=Path)
    parser.add_argument("--lora-strength", type=float, default=1.0)
    parser.add_argument("--model-dtype", default="default")
    parser.add_argument("--run-model", action="store_true")
    parser.add_argument("--decode", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.width % 16 != 0 or args.height % 16 != 0:
        raise ValueError("--width and --height must be multiples of 16 for FLUX.2/Klein patching")
    if args.run_model and not args.model:
        raise ValueError("--run-model requires --model")
    if args.decode and not args.vae:
        raise ValueError("--decode requires --vae")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or (out_dir / "klein_reference_edit_serenityflow_oracle_manifest.json")
    output_dtype = _dtype_from_name(args.output_dtype)
    api = import_serenityflow(args.serenityflow_root)

    raw_ref, patch_ref, reference_source = resolve_reference_latent(args, api)
    raw_h = args.height // 8
    raw_w = args.width // 8
    patch_h = args.height // 16
    patch_w = args.width // 16
    n_tokens = patch_h * patch_w

    if args.schedule_mode == "mojo_fixed_shift":
        sigmas = build_mojo_fixed_shift_sigmas(args.steps, args.edit_shift, args.edit_denoise)
        effective_sigmas = sigmas
        schedule_name = "FlowMatchEulerReferenceEdit"
    elif args.schedule_mode == "serenityflow_flux2":
        sigmas = build_serenityflow_flux2_sigmas(args.steps, args.edit_shift, args.edit_denoise)
        effective_sigmas = sigmas
        schedule_name = "SerenityFlowFlux2SchedulerReferenceEdit"
    else:
        sigmas = build_serenityflow_flux2_sigmas(args.steps, args.edit_shift, args.edit_denoise)
        effective_sigmas = apply_flux_exponential_shift(sigmas, n_tokens)
        schedule_name = "SerenityFlowFlux2SchedulerReferenceEditWithFlowShift"

    noise_raw = api.create_noise(
        seed=args.seed,
        shape=raw_ref.shape,
        device="cpu",
        dtype=torch.float32,
    ).float()
    noise_patch = api.patchify_flux2_latents(noise_raw)
    noise_tokens_b = api.pack_flux2_latents(noise_patch)
    ref_tokens_b = api.pack_flux2_latents(patch_ref)
    sigma0 = float(effective_sigmas[0].item())
    effective_initial_b = noise_tokens_b * sigma0 + ref_tokens_b * (1.0 - sigma0)
    combined_step0_b = torch.cat([effective_initial_b, ref_tokens_b], dim=1)
    combined_ids = torch.cat(
        [
            api.prepare_flux2_image_ids(patch_h, patch_w, "cpu", torch.float32, t_offset=0.0),
            api.prepare_flux2_image_ids(
                patch_h,
                patch_w,
                "cpu",
                torch.float32,
                t_offset=float(args.reference_t_offset),
            ),
        ],
        dim=0,
    )

    artifacts: dict[str, dict[str, Any]] = {}
    add_artifact(artifacts, out_dir, "python_reference_vae_latent_raw_nchw", raw_ref, dtype=output_dtype)
    add_artifact(artifacts, out_dir, "python_reference_patchified_nchw", patch_ref, dtype=output_dtype)
    add_artifact(artifacts, out_dir, "python_reference_tokens", normalize_token_shape(ref_tokens_b), dtype=output_dtype)
    add_artifact(artifacts, out_dir, "python_reference_combined_img_ids", combined_ids, dtype=torch.float32)
    add_artifact(artifacts, out_dir, "python_initial_noise_raw_nchw", noise_raw, dtype=output_dtype)
    add_artifact(artifacts, out_dir, "python_initial_noise_patchified_nchw", noise_patch, dtype=output_dtype)
    add_artifact(
        artifacts,
        out_dir,
        "python_initial_noise_post_pack",
        normalize_token_shape(noise_tokens_b),
        dtype=output_dtype,
    )
    add_artifact(
        artifacts,
        out_dir,
        "python_edit_initial_noise_target_tokens",
        normalize_token_shape(noise_tokens_b),
        dtype=output_dtype,
    )
    add_artifact(
        artifacts,
        out_dir,
        "python_edit_effective_initial_target_tokens",
        normalize_token_shape(effective_initial_b),
        dtype=output_dtype,
    )
    add_artifact(
        artifacts,
        out_dir,
        "python_edit_combined_tokens_step0",
        normalize_token_shape(combined_step0_b),
        dtype=output_dtype,
    )

    metrics: dict[str, Any] = {}
    notes: list[str] = []
    if args.synthetic_reference:
        notes.append("synthetic reference used for format smoke only; not an oracle parity artifact")

    model = None
    positive: list[dict[str, Any]] = []
    negative: list[dict[str, Any]] = []
    txt_len = 512
    if args.run_model:
        model = api.load_diffusion_model(str(args.model), dtype=args.model_dtype)
        if args.lora:
            api.apply_lora(model, str(args.lora), strength=float(args.lora_strength))
        positive, negative, txt_len, text_source = load_text_conditioning(args, api)

        txt_ids = api.prepare_flux2_text_ids(txt_len, "cpu", torch.float32)
        add_artifact(artifacts, out_dir, "python_txt_ids", txt_ids, dtype=torch.float32)
        metrics["text_source"] = text_source

        final_tokens_b, trajectory, run_metrics = run_reference_edit_euler(
            api=api,
            model=model,
            positive=positive,
            negative=negative,
            reference_tokens_batched=ref_tokens_b,
            effective_initial_batched=effective_initial_b,
            img_ids=combined_ids,
            txt_ids=txt_ids,
            sigmas=effective_sigmas,
            cfg=float(args.cfg),
            guidance=float(args.guidance),
        )
        for key, value in list(run_metrics.items()):
            if isinstance(value, torch.Tensor):
                add_artifact(artifacts, out_dir, key, value, dtype=output_dtype)
                run_metrics.pop(key)
        metrics.update(run_metrics)

        final_packed = api.unpack_flux2_latents(final_tokens_b, patch_h, patch_w)
        add_artifact(artifacts, out_dir, "python_edit_target_latent_trajectory", trajectory, dtype=output_dtype)
        add_artifact(artifacts, out_dir, "python_final_packed_latent", final_packed, dtype=output_dtype)

        final_for_vae = final_packed
        bn_mean = getattr(model, "_serenity_flux2_bn_mean", None)
        bn_var = getattr(model, "_serenity_flux2_bn_var", None)
        bn_eps = float(getattr(model, "_serenity_flux2_bn_eps", 1e-4))
        if isinstance(bn_mean, torch.Tensor) and isinstance(bn_var, torch.Tensor):
            mean = bn_mean.view(1, -1, 1, 1).to(dtype=final_for_vae.dtype)
            std = torch.sqrt(bn_var.view(1, -1, 1, 1).to(dtype=final_for_vae.dtype) + bn_eps)
            if mean.shape[1] == final_for_vae.shape[1]:
                final_for_vae = final_for_vae * std + mean
        final_unpatchified = api.unpatchify_flux2_latents(final_for_vae)
        if list(final_unpatchified.shape) != [1, 32, raw_h, raw_w]:
            raise ValueError(f"final unpatchified latent shape drifted: {tuple(final_unpatchified.shape)}")
        add_artifact(
            artifacts,
            out_dir,
            "python_final_unscaled_unpatchified_latent",
            final_unpatchified,
            dtype=output_dtype,
        )

        if args.decode:
            vae = api.load_vae(str(args.vae))
            vae_t0 = time.perf_counter()
            decoded = api.vae_decode(vae, final_unpatchified).detach().float().cpu()
            metrics["vae_decode_seconds"] = time.perf_counter() - vae_t0
            add_artifact(artifacts, out_dir, "python_vae_decoded_tensor", decoded, dtype=torch.float32)
            png_path = out_dir / "python_png.png"
            save_png(png_path, decoded)
            artifacts["python_png"] = {
                "path": str(png_path),
                "shape": [args.height, args.width],
                "dtype": "PNG",
                "bytes": png_path.stat().st_size,
            }
    else:
        notes.append("--run-model not supplied; trajectory/final/decoded artifacts intentionally omitted")

    manifest: dict[str, Any] = {
        "producer": "produce_klein_reference_edit_serenityflow_oracle.py",
        "mode": "reference_latent_edit",
        "parity_claimed": False,
        "python_oracle_claimed": bool(args.run_model and not args.synthetic_reference),
        "reference_source": reference_source,
        "inputs": {
            "width": args.width,
            "height": args.height,
            "steps": args.steps,
            "seed": args.seed,
            "cfg": args.cfg,
            "guidance": args.guidance,
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "model": str(args.model) if args.model else None,
            "vae": str(args.vae) if args.vae else None,
            "clip": str(args.clip) if args.clip else None,
            "positive_text_bin": str(args.positive_text_bin) if args.positive_text_bin else None,
            "negative_text_bin": str(args.negative_text_bin) if args.negative_text_bin else None,
            "allow_live_text_encoder": bool(args.allow_live_text_encoder),
            "lora": str(args.lora) if args.lora else None,
            "lora_strength": args.lora_strength,
        },
        "scheduler": {
            "name": schedule_name,
            "schedule_mode": args.schedule_mode,
            "edit_shift": args.edit_shift,
            "edit_denoise": args.edit_denoise,
            "reference_t_offset": args.reference_t_offset,
            "sigmas": [float(v) for v in effective_sigmas.tolist()],
            "scheduler_timestep": [float(v) * 1000.0 for v in effective_sigmas.tolist()],
        },
        "layout": {
            "raw_reference_shape": [1, 32, raw_h, raw_w],
            "patchified_reference_shape": [1, 128, patch_h, patch_w],
            "target_token_shape": [n_tokens, 128],
            "combined_token_shape": [2 * n_tokens, 128],
            "combined_img_ids_shape": [2 * n_tokens, 4],
        },
        "artifacts": artifacts,
        "comparison": {
            "reference_tokens": [
                "python_reference_tokens",
                "mojo_reference_tokens",
            ],
            "initial_noise": [
                "python_initial_noise_post_pack",
                "mojo_edit_initial_noise_target_tokens",
            ],
            "effective_initial": [
                "python_edit_effective_initial_target_tokens",
                "mojo_edit_effective_initial_target_tokens",
            ],
            "trajectory": [
                "python_edit_target_latent_trajectory",
                "mojo_edit_target_latent_trajectory",
            ],
            "final_latent": [
                "python_final_unscaled_unpatchified_latent",
                "mojo_final_unscaled_unpatchified_latent",
            ],
            "vae_png": [
                "python_png",
                "mojo_png",
            ],
        },
        "metrics": {"python": metrics},
        "notes": notes,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[klein-serenityflow-oracle] manifest: {manifest_path}")
    print("[klein-serenityflow-oracle] parity_claimed: false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

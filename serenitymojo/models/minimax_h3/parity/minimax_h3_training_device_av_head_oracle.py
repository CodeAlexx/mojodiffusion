#!/usr/bin/env python3
"""Pinned Torch oracle for H3 device AV objective + frozen final head.

Reduced hidden geometry, exact released video/audio channel widths. This is a
synthetic training seam gate, not a released-checkpoint or trainer claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file


HERE = Path(__file__).resolve().parent
OUT = HERE / "fixtures/minimax_h3_training_device_av_head.safetensors"
SHA_FILE = HERE / "fixtures/minimax_h3_training_device_av_head.sha256"
COMMIT = "b8717864713c9e4e7ef3d56eba1fc695a9b626a5"
SOURCE_SHA = {
    "model.py": "500fcacf93b40fac49b1ccbb21d8b382cb1f1b9fbd7954d1ac08155b2d0d243a",
    "minimax_h3_train_network.py": "4554196e24d5d85e9703b79602e8e4e64efc2c9ec2812011e9416192f2b1b99a",
    "packing.py": "464371faca4f156de883ce37022533c0fd3e0965723648c904d2f1cc09be2cc3",
    "audio_loss.py": "d266bc227d89a840829ce2e6cc941324e1b33672894e3ca815834090020370eb",
}

S, D, VIDEO_C, AUDIO_C, VIDEO_F, VIDEO_H, VIDEO_W, AUDIO_T = 8, 8, 24, 32, 1, 2, 4, 2
VIDEO_ROWS = VIDEO_F * (VIDEO_H // 2) * (VIDEO_W // 2)
AUDIO_ROWS = 2 * AUDIO_T
VIDEO_WIDTH = 4 * VIDEO_C
BASE_SIGMA, VIDEO_SHIFT, AUDIO_SHIFT, AUDIO_WEIGHT = 0.37, 12.0, 3.0, 0.65
VIDEO_INDICES = (4, 7)
AUDIO_INDICES = (0, 2, 3, 6)
TIMESTEP_INDICES = (0, 1, 0, 1, 0, 1, 0, 1)


def expected_metadata() -> dict[str, str]:
    return {
        "oracle": "Musubi MiniMax-H3 dual-shift native-flow AV objective and training-autocast final head",
        "oracle_commit": COMMIT,
        "source_sha256": json.dumps(SOURCE_SHA, sort_keys=True),
        "seed": "31029",
        "scope": "reduced hidden synthetic AV rows; ordinary frozen final head; no block/base/dataset/trainer",
        "dtype": "F32 latent/noise/x_t/targets/predictions/loss roots; BF16 hidden/modulation/head compute/d_hidden; F32 frozen head storage",
        "geometry": json.dumps({"S": S, "D": D, "video": [VIDEO_C, VIDEO_F, VIDEO_H, VIDEO_W], "audio": [AUDIO_C, 2, AUDIO_T]}, sort_keys=True),
        "schedule": json.dumps({"base_sigma": BASE_SIGMA, "video_shift": VIDEO_SHIFT, "audio_shift": AUDIO_SHIFT}, sort_keys=True),
        "loss": "independent packed-row means; total=video+audio_present*configured_weight*audio",
        "head": "F32 parameters cast by CUDA BF16 autocast; prediction cast F32 for loss; frozen d_hidden only",
    }


def specs() -> dict[str, tuple[torch.dtype, tuple[int, ...]]]:
    return {
        "in.video_latent": (torch.float32, (VIDEO_C, VIDEO_F, VIDEO_H, VIDEO_W)),
        "in.video_noise": (torch.float32, (VIDEO_C, VIDEO_F, VIDEO_H, VIDEO_W)),
        "in.audio_latent": (torch.float32, (AUDIO_C, 2, AUDIO_T)),
        "in.audio_noise": (torch.float32, (AUDIO_C, 2, AUDIO_T)),
        "out.video_x_t_rows": (torch.float32, (VIDEO_ROWS, VIDEO_WIDTH)),
        "out.video_target_rows": (torch.float32, (VIDEO_ROWS, VIDEO_WIDTH)),
        "out.audio_x_t_rows": (torch.float32, (AUDIO_ROWS, AUDIO_C)),
        "out.audio_target_rows": (torch.float32, (AUDIO_ROWS, AUDIO_C)),
        "head.hidden": (torch.bfloat16, (S, D)),
        "head.final_modulation": (torch.bfloat16, (2, 2 * D)),
        "head.norm_weight": (torch.bfloat16, (D,)),
        "head.video_weight": (torch.float32, (VIDEO_WIDTH, D)),
        "head.video_bias": (torch.float32, (VIDEO_WIDTH,)),
        "head.audio_weight": (torch.float32, (AUDIO_C, D)),
        "head.audio_bias": (torch.float32, (AUDIO_C,)),
        "head.video_prediction": (torch.float32, (VIDEO_ROWS, VIDEO_WIDTH)),
        "head.audio_prediction": (torch.float32, (AUDIO_ROWS, AUDIO_C)),
        "loss.d_video_present": (torch.float32, (VIDEO_ROWS, VIDEO_WIDTH)),
        "loss.d_audio_present": (torch.float32, (AUDIO_ROWS, AUDIO_C)),
        "loss.d_hidden_present": (torch.bfloat16, (S, D)),
        "loss.d_video_absent": (torch.float32, (VIDEO_ROWS, VIDEO_WIDTH)),
        "loss.d_audio_absent": (torch.float32, (AUDIO_ROWS, AUDIO_C)),
        "meta.scalars": (torch.float32, (12,)),
    }


def shifted(s: float, shift: float) -> float:
    return shift * s / (1.0 + (shift - 1.0) * s)


def pack_video(x: torch.Tensor) -> torch.Tensor:
    # [C,F,H,W] -> [F,H/2,W/2,C,1,2,2] -> rows, c-slowest within patch.
    return x.reshape(VIDEO_C, VIDEO_F, VIDEO_H // 2, 2, VIDEO_W // 2, 2).permute(1, 2, 4, 0, 3, 5).reshape(VIDEO_ROWS, VIDEO_WIDTH)


def pack_audio(x: torch.Tensor) -> torch.Tensor:
    return x.permute(1, 2, 0).reshape(AUDIO_ROWS, AUDIO_C)


def rms_norm_bf16(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (D,), weight, 1.0e-5)


def forward_head(hidden, modulation, norm_weight, vw, vb, aw, ab):
    row_mod = modulation[torch.tensor(TIMESTEP_INDICES, device="cuda")]
    shift, scale = row_mod.chunk(2, dim=-1)
    modulated = rms_norm_bf16(hidden, norm_weight) * (1 + scale) + shift
    with torch.autocast("cuda", dtype=torch.bfloat16):
        video_all = F.linear(modulated, vw, vb)
        audio_all = F.linear(modulated, aw, ab)
    vi = torch.tensor(VIDEO_INDICES, device="cuda")
    ai = torch.tensor(AUDIO_INDICES, device="cuda")
    return video_all[vi].float(), audio_all[ai].float()


def check_fixture(path: Path, sha_path: Path) -> None:
    if not path.is_file() or not sha_path.is_file():
        raise RuntimeError("missing H3 AV/head fixture or SHA sidecar")
    fields = sha_path.read_text().strip().split()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if len(fields) != 2 or fields[0] != digest or fields[1] != path.name:
        raise RuntimeError("H3 AV/head fixture SHA sidecar mismatch")
    expected = specs()
    with safe_open(path, framework="pt", device="cpu") as fixture:
        metadata = fixture.metadata()
        for key, value in expected_metadata().items():
            if metadata.get(key) != value:
                raise RuntimeError(f"metadata mismatch {key}: {metadata.get(key)!r}")
        for key in ("torch", "cuda", "gpu"):
            if not metadata.get(key):
                raise RuntimeError(f"missing runtime provenance {key}")
        if set(fixture.keys()) != set(expected):
            raise RuntimeError("fixture tensor inventory mismatch")
        for name, (dtype, shape) in expected.items():
            tensor = fixture.get_tensor(name)
            if tensor.dtype != dtype or tuple(tensor.shape) != shape:
                raise RuntimeError(f"fixture spec mismatch {name}")
    print(f"PASS H3 AV/head metadata+shape+dtype+sha256 preflight: {digest}")


def canonicalize_safetensors_header(path: Path) -> None:
    """Sort/re-pad the JSON header so fresh processes have one file digest."""
    blob = path.read_bytes()
    if len(blob) < 8:
        raise RuntimeError("truncated H3 AV/head fixture")
    header_len = struct.unpack("<Q", blob[:8])[0]
    data_start = 8 + header_len
    if data_start > len(blob):
        raise RuntimeError("invalid H3 AV/head safetensors header")
    header = json.loads(blob[8:data_start].decode("utf-8"))
    canonical = json.dumps(
        header, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    canonical += b" " * ((-len(canonical)) % 8)
    path.write_bytes(
        struct.pack("<Q", len(canonical)) + canonical + blob[data_start:]
    )


def generate(path: Path, sha_path: Path) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("H3 AV/head oracle requires CUDA")
    gen = torch.Generator(device="cpu").manual_seed(31029)
    f32 = lambda shape, scale: (torch.randn(shape, generator=gen) * scale).cuda()
    bf16 = lambda shape, scale: f32(shape, scale).to(torch.bfloat16)
    vl, vn = f32((VIDEO_C, VIDEO_F, VIDEO_H, VIDEO_W), 0.7), f32((VIDEO_C, VIDEO_F, VIDEO_H, VIDEO_W), 0.9)
    al, an = f32((AUDIO_C, 2, AUDIO_T), 0.6), f32((AUDIO_C, 2, AUDIO_T), 0.8)
    sv, sa = shifted(BASE_SIGMA, VIDEO_SHIFT), shifted(BASE_SIGMA, AUDIO_SHIFT)
    vx, ax = (1 - sv) * vl + sv * vn, (1 - sa) * al + sa * an
    vt, at = vl - vn, al - an
    vxr, vtr, axr, atr = pack_video(vx), pack_video(vt), pack_audio(ax), pack_audio(at)

    hidden = bf16((S, D), 0.4).requires_grad_()
    modulation = bf16((2, 2 * D), 0.15)
    norm_weight = (bf16((D,), 0.04).float() + 1.0).to(torch.bfloat16)
    vw, vb = f32((VIDEO_WIDTH, D), 0.12), f32((VIDEO_WIDTH,), 0.03)
    aw, ab = f32((AUDIO_C, D), 0.11), f32((AUDIO_C,), 0.03)
    vp, ap = forward_head(hidden, modulation, norm_weight, vw, vb, aw, ab)
    vp.retain_grad(); ap.retain_grad()
    video_loss = F.mse_loss(vp, vtr)
    audio_loss = F.mse_loss(ap, atr)
    total = video_loss + AUDIO_WEIGHT * audio_loss
    total.backward()
    dv_present, da_present, dh_present = vp.grad.detach(), ap.grad.detach(), hidden.grad.detach()
    dv_absent = 2.0 * (vp.detach() - vtr) / vtr.numel()
    da_absent = torch.zeros_like(ap)
    scalars = torch.tensor([
        BASE_SIGMA, sv, sa, 1 - sv, 1 - sa, AUDIO_WEIGHT,
        video_loss.item(), audio_loss.item(), total.item(), video_loss.item(),
        2.0, 1.0,
    ], dtype=torch.float32)
    tensors = {
        "in.video_latent": vl, "in.video_noise": vn,
        "in.audio_latent": al, "in.audio_noise": an,
        "out.video_x_t_rows": vxr, "out.video_target_rows": vtr,
        "out.audio_x_t_rows": axr, "out.audio_target_rows": atr,
        "head.hidden": hidden.detach(), "head.final_modulation": modulation,
        "head.norm_weight": norm_weight, "head.video_weight": vw,
        "head.video_bias": vb, "head.audio_weight": aw, "head.audio_bias": ab,
        "head.video_prediction": vp.detach(), "head.audio_prediction": ap.detach(),
        "loss.d_video_present": dv_present, "loss.d_audio_present": da_present,
        "loss.d_hidden_present": dh_present,
        "loss.d_video_absent": dv_absent, "loss.d_audio_absent": da_absent,
        "meta.scalars": scalars,
    }
    metadata = {**expected_metadata(), "torch": torch.__version__, "cuda": torch.version.cuda or "", "gpu": torch.cuda.get_device_name(0)}
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in tensors.items()}, path, metadata=metadata)
    canonicalize_safetensors_header(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    sha_path.write_text(f"{digest}  {path.name}\n")
    check_fixture(path, sha_path)


def check_generation_determinism(canonical: Path) -> None:
    digests: list[str] = []
    with tempfile.TemporaryDirectory(prefix="h3-av-head-oracle-") as directory:
        root = Path(directory)
        for run in range(3):
            path = root / f"fixture_{run}.safetensors"
            generate(path, path.with_suffix(".sha256"))
            digests.append(hashlib.sha256(path.read_bytes()).hexdigest())
    if len(set(digests)) != 1:
        raise RuntimeError(f"fresh-process-equivalent generation mismatch: {digests}")
    if canonical.is_file():
        canonical_digest = hashlib.sha256(canonical.read_bytes()).hexdigest()
        if digests[0] != canonical_digest:
            raise RuntimeError(
                f"fresh generation {digests[0]} != canonical {canonical_digest}"
            )
    print(f"PASS H3 AV/head 3-generation byte identity: {digests[0]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--determinism-check", action="store_true")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    sha = args.output.with_suffix(".sha256")
    if args.check:
        check_fixture(args.output, sha)
    elif args.determinism_check:
        check_generation_determinism(args.output)
    else:
        generate(args.output, sha)


if __name__ == "__main__":
    main()

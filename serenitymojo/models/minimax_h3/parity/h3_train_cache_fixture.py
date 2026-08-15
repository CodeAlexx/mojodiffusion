# h3_train_cache_fixture.py — write MiniMax-H3 training cache files through
# MUSUBI'S OWN save functions (the format oracle) with seeded tensors, plus an
# expected-stats sidecar the Mojo reader gate checks against.
#
# musubi-tuner akane/minimax-h3 @ 04324c28 is THE oracle: the files here are
# produced by save_latent_cache_minimax_h3 / save_text_encoder_output_cache_
# minimax_h3 exactly as cache_latents / cache_text_encoder_outputs would,
# with deterministic tensors standing in for VAE/conditioner outputs (the
# reader gate is about FORMAT fidelity; encoder numerics live in their own
# parity gates).
#
# Run:  /home/alex/musubi-tuner/.venv/bin/python h3_train_cache_fixture.py
import os
import sys

sys.path.insert(0, "/home/alex/musubi-h3/src")

import torch

from musubi_tuner.dataset.image_video_dataset import ItemInfo
from musubi_tuner.minimax_h3.cache import (
    H3_AUDIO_LOSS_MASK_KEY,
    H3_CONDITIONING_TASK_IDS,
    H3_CONDITIONING_TASK_KEY,
    H3_EMPTY_TEXT_HIDDEN_KEY,
    H3_EMPTY_TEXT_TOKEN_TAGS_KEY,
    H3_KEYFRAME_VIDEO_ROWS_KEY,
    H3_TEXT_HIDDEN_KEY,
    H3_TEXT_TOKEN_TAGS_KEY,
    save_latent_cache_minimax_h3,
    save_text_encoder_output_cache_minimax_h3,
)

OUT_DIR = "/home/alex/mojodiffusion/output/checks/h3_cache_fixture"
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(0)
stats = {}


def f64_sum(t: torch.Tensor) -> float:
    return float(t.to(torch.float32).sum(dtype=torch.float64))


def write_item(
    name: str,
    wh: tuple[int, int],
    frame_count: int,
    video_shape: tuple[int, int, int],   # (F, H, W) latent
    audio_t: int,
    tokens: int,
    task: str,
    with_empty: bool,
    empty_tokens: int,
    with_video_loss_mask: bool,
):
    w, h = wh
    item = ItemInfo(
        item_key=f"/fake/data/{name}.mp4",
        caption="fixture caption",
        original_size=(w, h),
        frame_count=frame_count,
        latent_cache_path=os.path.join(OUT_DIR, f"{name}_{w:04d}x{h:04d}_mmh3.safetensors"),
    )
    item.text_encoder_output_cache_path = os.path.join(OUT_DIR, f"{name}_mmh3_te.safetensors")

    F, H, W = video_shape
    video = torch.randn(24, F, H, W, dtype=torch.float32).to(torch.bfloat16)
    audio = torch.randn(2, 32, audio_t, dtype=torch.float32).to(torch.bfloat16)
    audio_mask = torch.ones(audio_t, dtype=torch.bool)
    audio_mask[-2:] = False  # exercise a padded tail
    rows = (H // 2) * (W // 2)
    keyframes = torch.randn(2 * rows, 24 * 2 * 2, dtype=torch.float32).to(torch.bfloat16)

    latent_tensors = {
        f"latents_{F}x{H}x{W}_bfloat16": video,
        f"latents_audio_2x32x{audio_t}_bfloat16": audio,
        H3_AUDIO_LOSS_MASK_KEY: audio_mask,
        f"varlen_{H3_KEYFRAME_VIDEO_ROWS_KEY}_bfloat16": keyframes,
    }
    if with_video_loss_mask:
        vlm = torch.ones(F, H, W, dtype=torch.bool)
        vlm[:, :2, :] = False
        latent_tensors["video_loss_mask"] = vlm
    save_latent_cache_minimax_h3(item, latent_tensors)

    hidden = torch.randn(tokens, 5120, dtype=torch.float32).to(torch.bfloat16)
    tags = torch.arange(tokens, dtype=torch.long) % 7
    te_tensors = {
        f"varlen_{H3_TEXT_HIDDEN_KEY}_bfloat16": hidden,
        f"varlen_{H3_TEXT_TOKEN_TAGS_KEY}_int64": tags,
        H3_CONDITIONING_TASK_KEY: torch.tensor(H3_CONDITIONING_TASK_IDS[task], dtype=torch.long),
    }
    empty_hidden = empty_tags = None
    if with_empty:
        empty_hidden = torch.randn(empty_tokens, 5120, dtype=torch.float32).to(torch.bfloat16)
        empty_tags = torch.arange(empty_tokens, dtype=torch.long) % 3
        te_tensors[f"varlen_{H3_EMPTY_TEXT_HIDDEN_KEY}_bfloat16"] = empty_hidden
        te_tensors[f"varlen_{H3_EMPTY_TEXT_TOKEN_TAGS_KEY}_int64"] = empty_tags
    save_text_encoder_output_cache_minimax_h3(item, te_tensors)

    s = {
        "lat_f": F, "lat_h": H, "lat_w": W, "audio_t": audio_t,
        "video_sum": f64_sum(video),
        "audio_sum": f64_sum(audio),
        "mask_true": int(audio_mask.sum()),
        "kf_rows": 2 * rows, "kf_width": 24 * 4,
        "kf_sum": f64_sum(keyframes),
        "has_video_loss_mask": int(with_video_loss_mask),
        "tokens": tokens,
        "hidden_sum": f64_sum(hidden),
        "tags_sum": int(tags.sum()),
        "task_id": H3_CONDITIONING_TASK_IDS[task],
        "has_empty": int(with_empty),
        "empty_tokens": empty_tokens if with_empty else 0,
        "empty_hidden_sum": f64_sum(empty_hidden) if with_empty else 0.0,
        "empty_tags_sum": int(empty_tags.sum()) if with_empty else 0,
    }
    stats[name] = s


write_item(
    "clip_alpha", wh=(832, 480), frame_count=33, video_shape=(9, 30, 52),
    audio_t=94, tokens=87, task="t2va", with_empty=True, empty_tokens=12,
    with_video_loss_mask=False,
)
write_item(
    "clip_beta", wh=(640, 384), frame_count=17, video_shape=(5, 24, 40),
    audio_t=54, tokens=33, task="t2va", with_empty=False, empty_tokens=0,
    with_video_loss_mask=True,
)

with open(os.path.join(OUT_DIR, "expected.txt"), "w") as f:
    for name, s in stats.items():
        for k, v in s.items():
            if isinstance(v, float):
                f.write(f"{name}.{k}={v:.17g}\n")
            else:
                f.write(f"{name}.{k}={v}\n")

print("wrote fixture to", OUT_DIR)
for name in stats:
    print(" ", name, "->", {k: stats[name][k] for k in ("lat_f", "audio_t", "tokens", "task_id")})

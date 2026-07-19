#!/usr/bin/env python
# prep_dance_i2v.py — prep a custom i2v run from an uploaded image + dance prompt.
#   (1) cover-resize + center-crop the image to HxW -> [1,3,1,H,W] in [0,1]
#   (2) tokenize the dance prompt + the video negative (LingBot template) via the
#       processor ONLY (no 8.9GB encoder) -> ids + crop_start + post-crop lengths.
# The Mojo probe then VAE-encodes the image and Mojo-encodes these ids.
import os, sys, json
import numpy as np, torch
from PIL import Image
from safetensors.torch import save_file
from transformers import AutoProcessor

OUT = "/home/alex/mojodiffusion/serenitymojo/models/lingbotvideo/parity"
sys.path.insert(0, OUT)
sys.path.insert(0, "/mnt/disk1/lingbot-src/lingbot-video")
from oracle_e_t2i import PROMPT_TEMPLATE, _apply_template, _compute_crop_start, TOKEN_LENGTH, PROC_DIR
from lingbot_video.pipeline_lingbot_video import DEFAULT_NEGATIVE_PROMPT

IMG = "/home/alex/.claude/uploads/6fd4828e-ac2b-4c91-a10b-c346b1b61e18/8280dc5c-1000004560.webp"
H, W = 576, 320   # portrait full-body (aspect 0.556 ~ image 0.577)
import json as _json
DANCE = _json.dumps({
    "comprehensive_description": (
        "A dark-haired man in a brown tweed jacket, white shirt, red tie and brown "
        "trousers performs a silly, energetic comedic full-body dance from the first "
        "frame to the last, standing centered against a plain white studio background: "
        "he bounces on the balls of his feet and steps side to side, swinging his arms "
        "and pumping his elbows to a beat, throwing playful disco-style points with his "
        "hands, wiggling his hips and shoulders, bobbing his head and doing exaggerated "
        "goofy moves with a deadpan expression, alternating between arm swings, shoulder "
        "shimmies, little kicks and spins as the dance keeps changing. His whole body "
        "stays in constant lively motion, the tweed jacket and loose trousers swaying "
        "with every move. The camera stays static at eye level, holding the full-body "
        "figure centered so the entire comedic dance clearly reads - arms, legs, hips "
        "and head all in continuous, varied motion throughout the clip."
    ),
    "camera_info": {
        "color": "Neutral", "frame_size": "Full Shot", "shot_type_angle": "Eye level",
        "lens_size": "Medium", "composition": "Center", "lighting": "Soft light",
        "lighting_type": "Artificial light",
        "camera_motion": "static shot, camera holds still while the subject dances",
    },
    "world_knowledge": [],
    "prominent_elements": [
        {
            "name": "man in a tweed suit",
            "description": "A dark-haired man in a brown tweed jacket and brown trousers doing a comedic full-body twerking dance with an exaggerated deadpan face.",
            "location": "center of the frame, full body facing the camera",
            "relative_size": "dominant",
            "shape_and_color": "a man in a brown herringbone tweed jacket, white shirt and red tie, dark brown trousers and black shoes, with short dark hair",
            "texture": "woven tweed fabric, crisp white shirt, smooth trousers",
            "appearance_details": "knees bending into a low squat, upper body leaning forward, hips and rear bouncing up and down rapidly to the beat, arms swinging out and bending for balance, shoulders shimmying, head bobbing, jacket and trousers jiggling with each bounce, deadpan comedic expression",
            "relationship": "stands and twerks alone at the center of the frame",
            "orientation": "upright and facing the camera, crouching and bouncing rhythmically",
        },
        {
            "name": "plain white background",
            "description": "A clean seamless white studio backdrop behind the dancing man.",
            "location": "fills the entire frame behind the subject",
            "relative_size": "large",
            "shape_and_color": "flat, evenly lit pure white",
            "texture": "smooth seamless studio backdrop",
            "appearance_details": "stays clean and static while the man dances in front of it",
            "relationship": "isolates the twerking man as the sole subject",
            "orientation": "flat vertical backdrop",
        },
    ],
})


def preprocess(path, h, w):
    im = Image.open(path).convert("RGB")
    iw, ih = im.size
    s = max(w / iw, h / ih)                       # cover
    im = im.resize((round(iw * s), round(ih * s)), Image.BICUBIC)
    rw, rh = im.size
    left, top = (rw - w) // 2, (rh - h) // 2
    im = im.crop((left, top, left + w, top + h))  # center-crop to WxH
    arr = np.asarray(im, np.float32) / 255.0      # HWC [0,1]
    chw = arr.transpose(2, 0, 1)[None, :, None]   # [1,3,1,H,W]
    return chw


def tok(processor, crop_start, text):
    inputs = processor(text=[_apply_template(text)], images=None, videos=None,
                       do_resize=False, truncation=True, max_length=TOKEN_LENGTH,
                       padding="longest", return_tensors="pt")
    ids = inputs["input_ids"][0].tolist()
    assert 151643 not in ids, "pad id present in real tokens"
    return ids, len(ids), len(ids) - crop_start


def main():
    cond = preprocess(IMG, H, W)
    save_file({"image": torch.from_numpy(np.ascontiguousarray(cond))}, os.path.join(OUT, "dance_condition.safetensors"))
    print(f"[prep] condition image {cond.shape} range [{cond.min():.3f},{cond.max():.3f}]")

    processor = AutoProcessor.from_pretrained(PROC_DIR, trust_remote_code=True)
    crop = _compute_crop_start(processor)
    p_ids, p_len, p_post = tok(processor, crop, DANCE)
    n_ids, n_len, n_post = tok(processor, crop, DEFAULT_NEGATIVE_PROMPT)
    save_file({
        "prompt_ids": torch.tensor(p_ids, dtype=torch.int32),
        "neg_ids": torch.tensor(n_ids, dtype=torch.int32),
        "crop_start": torch.tensor([crop], dtype=torch.int32),
        "prompt_true_len": torch.tensor([p_len], dtype=torch.int32),
        "neg_true_len": torch.tensor([n_len], dtype=torch.int32),
    }, os.path.join(OUT, "dance_ids.safetensors"))
    json.dump({"crop_start": crop, "prompt_post": p_post, "neg_post": n_post,
               "prompt_true_len": p_len, "neg_true_len": n_len,
               "H": H, "W": W, "LH": H // 8, "LW": W // 8, "GH": H // 16, "GW": W // 16},
              open(os.path.join(OUT, "dance_meta.json"), "w"), indent=2)
    print(f"[prep] crop_start={crop}  PROMPT post-crop L_COND={p_post}  NEG post-crop L_UNCOND={n_post}")
    print(f"[prep] geometry H={H} W={W} LH={H//8} LW={W//8} GH={H//16} GW={W//16}")


if __name__ == "__main__":
    main()

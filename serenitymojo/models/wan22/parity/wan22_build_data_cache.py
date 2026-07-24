#!/usr/bin/env python3
# wan22_build_data_cache.py — OFFLINE dev tool (like the parity oracles).
#
# Builds a REAL data cache for the pure-Mojo Wan2.2 trainer by running Musubi's
# OWN Wan2.1 VAE + umt5-xxl encoders over the first 8 image/caption pairs of the
# 40_woman dataset. The Mojo shipped trainer path stays pure Mojo; this tool is
# run offline with the ai-toolkit venv:
#
#   /home/alex/ai-toolkit/venv/bin/python \
#       serenitymojo/models/wan22/parity/wan22_build_data_cache.py
#
# OUTPUT (one file per sample):
#   /home/alex/.serenity/wan22_cache/40_woman/sample_{i}.safetensors
# with a SIMPLE, documented format (safetensors save):
#   key "latent"   : f32,  shape [16, 1, 32, 32]  (Wan2.1-VAE 16ch, F=1, 256/8=32)
#                    ALREADY normalized by Wan2.1 per-channel mean/std (scale is
#                    applied INSIDE WanVAE.model.encode) — used as-is (Wan
#                    scale_shift_latents is identity).
#   key "text"     : bf16, shape [512, 4096]      (umt5-xxl, pad/trunc to 512)
#   key "text_len" : i32,  shape [1]              (true token count before pad)
#
# MEMORY (5080 = 16 GB): ONE model at a time. Encode all 8 latents with the VAE,
# free the VAE + empty_cache, THEN load umt5 and encode all 8 texts.

import os
import sys
import torch
import numpy as np
from PIL import Image
from safetensors.torch import save_file

# ── I2V MODE (--i2v) ─────────────────────────────────────────────────────────
# Wan2.2-I2V-A14B conditions the DiT on the input image. The model input is
#   concat([noisy_latent(16ch), y(20ch)]) = 36ch, where y = concat([mask(4),
#   image_latent(16)]). musubi wan_cache_latents.py:63-80:
#     msk = ones(1,F,lat_h,lat_w); msk[:,1:]=0 ; repeat_interleave frame0 x4 ;
#     view -> [1,F',4,H,W] ; transpose -> [1,4,F,H,W]  (4-ch temporal mask)
#     image -> pad to F frames w/ zeros -> vae.encode -> image_latent[16,F,H,W]
#     y = concat([msk, image_latent], dim=channel) = [20,F,H,W]
# For a SINGLE-frame (F=1) smoke the mask is ALL-ONES over the 4 mask channels
# (msk[:,1:]=0 is a no-op at F=1; repeat_interleave of frame0 x4 -> [1,4,1,H,W]
# all ones), and image_latent == the same VAE encoding used as the 16-ch target
# (the conditioning image IS the target image). This is intentionally degenerate
# (target ~= conditioning) but exercises the full 36-ch input mechanism.
# ── I2V21 MODE (--i2v21) ─────────────────────────────────────────────────────
# Wan2.1 I2V-14B is the DUAL cross-attn variant (WanI2VCrossAttention): in
# addition to everything --i2v produces (latent + text + cond_y), it needs the
# CLIP image embedding of the first frame — 257 tokens x 1280 dims (256 patch + 1
# cls, penultimate block via use_31_block=True) — which the frozen img_emb/MLPProj
# projects to `dim` and PREPENDS to the umt5 text context. --i2v21 implies --i2v
# (cond_y is still required). The extra output key is "clip" [257,1280] fp16.
# See musubi wan/modules/clip.py + wan_cache_latents.py:48-55.
I2V21 = "--i2v21" in sys.argv[1:]
I2V = ("--i2v" in sys.argv[1:]) or I2V21

DATASET = os.environ.get("WAN_CACHE_DATASET", "/mnt/disk2/datasets/40_woman")
VAE_PATH = "/home/alex/.serenity/models/checkpoints/SCAIL-2/Wan2.1_VAE.pth"
UMT5_DIR = "/home/alex/.serenity/models/checkpoints/SCAIL-2/umt5-xxl"
UMT5_WEIGHTS = os.path.join(UMT5_DIR, "models_t5_umt5-xxl-enc-bf16.pth")
# xlm-roberta-vit-h-14 "onlyvisual" CLIP checkpoint (fp16).
CLIP_PATH = ("/home/alex/.serenity/models/checkpoints/SCAIL-2/"
             "models_clip_open-clip-xlm-roberta-large-vit-huge-14-onlyvisual.pth")
if I2V21:
    OUT_DIR = "/home/alex/.serenity/wan22_cache/40_woman_i2v21"
elif I2V:
    OUT_DIR = "/home/alex/.serenity/wan22_cache/40_woman_i2v"
else:
    OUT_DIR = "/home/alex/.serenity/wan22_cache/40_woman"
# env overrides (default-preserving: absent => byte-identical 40_woman path)
OUT_DIR = os.environ.get("WAN_CACHE_OUT", OUT_DIR)

N_SAMPLES = int(os.environ.get("WAN_CACHE_N", "8"))
RES = int(os.environ.get("WAN_CACHE_RES", "256"))
TEXT_LEN = 512
TEXT_DIM = 4096
Z_DIM = 16

DEVICE = torch.device("cuda")


def load_image_video(path):
    """Load image -> center-crop square -> resize RES -> [-1,1] -> [1,3,1,RES,RES]."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    img = img.crop((left, top, left + s, top + s)).resize((RES, RES), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32)          # [H, W, C], 0..255
    t = torch.from_numpy(arr)                        # [H, W, C]
    t = t.permute(2, 0, 1).contiguous()              # [C, H, W]
    t = t / 127.5 - 1.0                              # [-1, 1]
    t = t.unsqueeze(0).unsqueeze(2)                  # [1, C, 1, H, W]  (B,C,F,H,W)
    return t


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # collect first N pairs (numeric filename order)
    stems = []
    for name in os.listdir(DATASET):
        if name.endswith(".jpg"):
            stem = name[:-4]
            if os.path.exists(os.path.join(DATASET, stem + ".txt")):
                stems.append(stem)
    stems = sorted(stems, key=lambda s: (0, int(s)) if s.isdigit() else (1, s))[:N_SAMPLES]
    print(f"[cache] {len(stems)} pairs: {stems}")

    captions = []
    for stem in stems:
        with open(os.path.join(DATASET, stem + ".txt"), "r") as f:
            captions.append(f.read().strip())

    # ── PHASE 1: VAE encode all latents ───────────────────────────────────────
    from musubi_tuner.wan.modules.vae import WanVAE
    print(f"[cache] loading WanVAE z_dim={Z_DIM} <- {VAE_PATH}")
    vae = WanVAE(z_dim=Z_DIM, vae_path=VAE_PATH, dtype=torch.bfloat16, device="cuda")

    latents = []
    cond_ys = []   # I2V only: y = concat([mask(4), image_latent(16)]) = [20,1,32,32]
    first_frames = []  # I2V21 only: [-1,1] first frame [1,3,1,256,256] for CLIP
    with torch.no_grad():
        for i, stem in enumerate(stems):
            video = load_image_video(os.path.join(DATASET, stem + ".jpg"))
            if I2V21:
                # keep the [-1,1] first frame on CPU for the CLIP phase (CLIP is
                # loaded AFTER the VAE is freed — one model at a time on 16 GB).
                first_frames.append(video[:, :, 0:1, :, :].clone().cpu().float())
            video = video.to(DEVICE, dtype=torch.bfloat16)        # [1,3,1,256,256]
            # WanVAE.encode iterates the first dim as a list of [C,F,H,W] videos;
            # its .model.encode applies Wan2.1 per-channel (mean, 1/std) inside.
            lat = vae.encode(video)[0].float()                    # [16, 1, 32, 32]
            latents.append(lat.cpu())

            if I2V:
                # musubi wan_cache_latents.py:63-80 at F=1: mask is all-ones over
                # the 4 mask channels; image_latent == this same VAE encoding (the
                # conditioning image is the target image). y = concat([mask, lat]).
                _, Fl, lat_h, lat_w = lat.shape                    # 16,1,32,32
                msk = torch.ones(4, Fl, lat_h, lat_w, dtype=torch.float32)  # [4,1,32,32]
                cond_y = torch.concat([msk, lat.cpu()], dim=0)     # [20,1,32,32]
                cond_ys.append(cond_y.contiguous())
                cy_std = cond_y.std().item()
                cy_mean = cond_y.mean().item()
                # mask channels are all 1.0 (std 0, mean 1) and image_latent std is
                # ~0.35-0.60; the combined 20-ch std/mean is a blend of the two.
                cy_flag = "" if (0.30 <= cy_std <= 1.30) else \
                    "  <-- COND_Y STD OUT OF RANGE (possible scramble!)"
                print(f"[cache] sample {i} ({stem}.jpg): cond_y shape "
                      f"{list(cond_y.shape)} std={cy_std:.4f} mean={cy_mean:.4f}"
                      f"{cy_flag}")

            std = lat.std().item()
            mean = lat.mean().item()
            flag = ""
            # A SINGLE image's normalized std is legitimately ~0.35-0.60 (Wan's
            # per-channel std 2.8/1.45/... is the whole-dataset population std,
            # which includes inter-image variance one sample lacks; encode/decode
            # round-trip corr=0.998 confirms correct layout). A near-ZERO std or a
            # value far above 1 would signal a real scramble/HWC<->CHW bug.
            if not (0.20 <= std <= 1.30):
                flag = "  <-- STD OUT OF RANGE (possible HWC<->CHW / scramble bug!)"
            print(f"[cache] sample {i} ({stem}.jpg): latent shape {list(lat.shape)} "
                  f"std={std:.4f} mean={mean:.4f}{flag}")

    del vae
    torch.cuda.empty_cache()
    print("[cache] VAE freed")

    # ── PHASE 1b: CLIP visual encode (I2V21 only) ─────────────────────────────
    # First frame -> resize 224 -> penultimate-block visual tokens [257,1280].
    # musubi CLIPModel.visual iterates a list of [C,F,H,W] videos internally.
    clip_ctxs = []
    if I2V21:
        if not os.path.exists(CLIP_PATH):
            print(f"[cache] CLIP weights ABSENT at {CLIP_PATH}")
            print("[cache]   -> skipping the 'clip' key (I2V21 clip encode DEFERRED "
                  "until the onlyvisual checkpoint is downloaded).")
        else:
            from musubi_tuner.wan.modules.clip import CLIPModel
            print(f"[cache] loading CLIP xlm-roberta-vit-h-14 <- {CLIP_PATH}")
            # The on-disk checkpoint is the "-onlyvisual" variant (visual tower only).
            # Musubi builds the model on `meta` then load_state_dict(strict=True,
            # assign=True), so the missing textual.* keys must be present as REAL
            # tensors or they stay `meta` and the later .to(device) fails. We only
            # ever call clip.visual(), so fill the unused textual keys with zeros.
            _orig_lsd = torch.nn.Module.load_state_dict
            def _fill_lsd(self, state_dict, strict=True, assign=False):
                sd = dict(state_dict)
                for k, v in self.state_dict().items():
                    if k not in sd:
                        sd[k] = torch.zeros(v.shape, dtype=v.dtype, device=DEVICE)
                return _orig_lsd(self, sd, strict=True, assign=True)
            torch.nn.Module.load_state_dict = _fill_lsd
            try:
                clip = CLIPModel(dtype=torch.float16, device=DEVICE, weight_path=CLIP_PATH)
            finally:
                torch.nn.Module.load_state_dict = _orig_lsd
            with torch.no_grad():
                for i, ff in enumerate(first_frames):
                    vid = ff.to(DEVICE, dtype=torch.float16)      # [1,3,1,256,256] in [-1,1]
                    with torch.amp.autocast(device_type=DEVICE.type, dtype=torch.float16):
                        cc = clip.visual(vid)                     # [1,257,1280]
                    cc = cc.to(torch.float16)[0].cpu()            # [257,1280]
                    clip_ctxs.append(cc.contiguous())
                    print(f"[cache] sample {i}: clip shape {list(cc.shape)} "
                          f"std={cc.float().std().item():.4f}")
            del clip
            torch.cuda.empty_cache()
            print("[cache] CLIP freed")

    # ── PHASE 2: umt5 encode all texts ────────────────────────────────────────
    # Musubi's T5EncoderModel passes subfolder=None to HuggingfaceTokenizer when a
    # tokenizer_path is given; this transformers version chokes on subfolder=None
    # (os.path.join(None, ...)). Strip None-valued kwargs so AutoTokenizer loads
    # the LOCAL umt5-xxl dir offline.
    import musubi_tuner.wan.modules.tokenizers as _tokmod
    _orig_tok_init = _tokmod.HuggingfaceTokenizer.__init__
    def _patched_tok_init(self, name, seq_len=None, clean=None, **kwargs):
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        _orig_tok_init(self, name, seq_len=seq_len, clean=clean, **kwargs)
    _tokmod.HuggingfaceTokenizer.__init__ = _patched_tok_init

    from musubi_tuner.wan.modules.t5 import T5EncoderModel
    print(f"[cache] loading umt5-xxl (text_len={TEXT_LEN}) <- {UMT5_WEIGHTS}")
    text_encoder = T5EncoderModel(
        text_len=TEXT_LEN,
        dtype=torch.bfloat16,
        device=DEVICE,
        weight_path=UMT5_WEIGHTS,
        tokenizer_path=UMT5_DIR,   # local dir -> AutoTokenizer offline
        fp8=False,
    )

    texts = []
    text_lens = []
    with torch.no_grad():
        for i, cap in enumerate(captions):
            ctx = text_encoder([cap], DEVICE)[0]   # [true_len, 4096] (varlen, trimmed)
            true_len = int(ctx.shape[0])
            padded = torch.zeros(TEXT_LEN, TEXT_DIM, dtype=torch.bfloat16)
            n = min(true_len, TEXT_LEN)
            padded[:n] = ctx[:n].to(torch.bfloat16).cpu()
            texts.append(padded)
            text_lens.append(min(true_len, TEXT_LEN))
            print(f"[cache] sample {i}: text shape {list(padded.shape)} "
                  f"true_len={true_len} (cap[:60]={cap[:60]!r})")

    del text_encoder
    torch.cuda.empty_cache()

    # ── WRITE ─────────────────────────────────────────────────────────────────
    for i in range(len(stems)):
        out = os.path.join(OUT_DIR, f"sample_{i}.safetensors")
        tensors = {
            "latent": latents[i].contiguous().to(torch.float32),
            "text": texts[i].contiguous().to(torch.bfloat16),
            "text_len": torch.tensor([text_lens[i]], dtype=torch.int32),
        }
        if I2V:
            # I2V conditioning: y = [mask(4) + image_latent(16)] = [20,1,32,32] f32.
            # The Mojo trainer patchifies this (1,2,2) -> [S,80] and concatenates it
            # per-token AFTER the noisy-latent patch [S,64] -> model input [S,144].
            tensors["cond_y"] = cond_ys[i].contiguous().to(torch.float32)
        if I2V21 and clip_ctxs:
            # CLIP image embedding [257,1280] fp16 — the frozen MLPProj projects it
            # to [257,dim] and prepends it to the umt5 text context (WanI2VCrossAttn).
            tensors["clip"] = clip_ctxs[i].contiguous().to(torch.float16)
        save_file(tensors, out)
        print(f"[cache] wrote {out}")

    if I2V21:
        mode = "I2V21 (latent+text+cond_y+clip)" if clip_ctxs \
            else "I2V21 (latent+text+cond_y; clip DEFERRED - weights absent)"
    elif I2V:
        mode = "I2V (latent+text+cond_y)"
    else:
        mode = "T2V (latent+text)"
    print(f"[cache] DONE [{mode}]: {len(stems)} samples -> {OUT_DIR}")


if __name__ == "__main__":
    main()

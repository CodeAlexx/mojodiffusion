#!/usr/bin/env python
# mageflow_edit_oracle.py — E2E parity oracle for the MageFlow *edit* turbo pipeline.
#
# Faithfully replicates mage_flow.pipeline.generate_edits for the single-sample,
# cfg=1.0 (turbo, no negative branch) case, at a small fixed resolution, using the
# SAME helper functions the real pipeline calls (_encode_edits_packed via the same
# processor path, compute_vae_encodings, get_noise+encode_noise, _build_pack_ctx,
# _velocity, _get_scheduler, vae.decode). The ONE deliberate deviation from the
# stock pipeline: the ref image is VAE-encoded with sample_posterior=False (the
# posterior MEAN, deterministic) so it matches the pure-Mojo mageflow_encode (which
# returns the mean). Everything else is byte-faithful.
#
# HOW THE REF IMAGE FEEDS THE DiT (the finding this gate proves):
#   The ref image is VAE-encoded into CLEAN latent tokens and sequence-CONCATENATED
#   (NOT channel-concat; in_channels stays 128) to the target NOISE tokens:
#     img = [target(Lt), ref(Lr)]  along the token axis.
#   The target starts from PURE noise (like T2I). Refs stay clean across ALL steps.
#   Each step runs ONE DiT forward over [txt, target, ref]; only the target-token
#   velocity is used to Euler-step the target. RoPE: target frame-idx 0, ref
#   frame-idx 1 (msrope axis-0), h/w restart per shape — reproduced by the Mojo
#   DiT with frame=2, height=width=SH (target & ref same resolution).
#
# Dumps (raw <f4 into mageflow_edit_dumps/):
#   input_ids.bin     [L]          exact processor token ids (int -> f32)
#   pixel_values.bin  [seq,1536]   Qwen3-VL vision-tower input (384-capped ref)
#   grid_thw.bin      [3]          (t,gh,gw) of the vision grid (int -> f32)
#   ref_pixel.bin     [3,H,W]      the [-1,1] target-res ref fed to the VAE encoder
#   ref_latent.bin    [Lr,128]     VAE mean latent of the ref (clean cond tokens)
#   init_noise.bin    [Lt,128]     target noise entering the loop (post encode_noise)
#   final_latent.bin  [Lt,128]     target latent after the 4 Euler steps (pre-VAE)
#   rgb.bin           [1,3,H,W]    vae.decode(unpack(final_target)), clamp[-1,1]
#   oracle_edit.png   the decoded edited image (visual reference)
#   input_source.png  the (unedited) source image at target res (for comparison)
#   meta.txt          shapes + hyperparameters
#
# Run (current Python environment, mage_flow on PYTHONPATH):
#   cd /home/alex/mojodiffusion
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
#     PYTHONPATH=/home/alex/Mage pixi run python \
#     serenitymojo/pipeline/parity/mageflow_edit_oracle.py
import os
import gc

import numpy as np
import torch
from einops import rearrange
from PIL import Image

MAGE_EDIT = "/home/alex/.serenity/models/checkpoints/Mage-Flow-Edit-Turbo"
IMAGE = "/home/alex/Mage/mage_flow/assets/dog.jpg"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mageflow_edit_dumps")
os.makedirs(OUT, exist_ok=True)

INSTRUCTION = "Change the background to a snowy forest."
SEED = 42
STEPS = 4       # turbo edit
CFG = 1.0       # turbo edit -> no negative branch (single forward per step)
H = 256
W = 256
VL_COND_LONG_EDGE = 384
IMAGE_TOKEN_ID = 151655
EDIT_DROP_IDX = 64
_EDIT_IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"


def dump_f32(name, arr):
    v = np.asarray(arr).ravel().astype("<f4")
    with open(os.path.join(OUT, name), "wb") as f:
        f.write(v.tobytes())
    return list(np.asarray(arr).shape)


@torch.no_grad()
def main():
    from mage_flow.pipeline import (
        load_from_repo, _encode_edits_packed, _slice_packed, _build_pack_ctx,
        _velocity, _get_scheduler, _preprocess_ref_image, _resize_long_edge,
        _edit_prompt_body, _lens_to_cu,
    )
    from mage_flow.models.utils import get_noise, PROMPT_TEMPLATE, unpack
    from mage_flow.models.modules.mage_latent import encode_noise, resolve_gs_key
    from mage_flow.models.modules._attn_backend import set_attn_backend

    dev = torch.device("cuda:0")
    print(f"[mf-edit-oracle] loading Mage-Flow-Edit-Turbo (CPU-resident) from {MAGE_EDIT}")
    # 16 GB card: load everything on CPU, shuttle one submodule to GPU per stage
    # (the same offload discipline the Mojo pipeline uses).
    model = load_from_repo(MAGE_EDIT, "cpu")
    # No flash-attn wheel (CUDA 13 / sm_120): torch-SDPA varlen fallback for BOTH
    # the packed VL text encoder and the DiT joint attn. Set AFTER load.
    set_attn_backend("sdpa")
    # Deterministic ref latent = posterior MEAN (matches Mojo mageflow_encode).
    model.vae.sample_posterior = False

    info = PROMPT_TEMPLATE["mage-flow-edit"]
    template = info["template"]
    drop_idx = int(info["start_idx"])
    assert drop_idx == EDIT_DROP_IDX

    # ---- reference PILs + target size (single sample, explicit H/W) ----
    src_pil = Image.open(IMAGE).convert("RGB")
    src_pil.resize((W, H), Image.BICUBIC).save(os.path.join(OUT, "input_source.png"))

    # ==== STAGE 1: edit text conditioning (VL encoder on GPU) ====
    edit_refs = [[_resize_long_edge(src_pil, VL_COND_LONG_EDGE)]]  # one sample, one ref
    model.txt_enc.to(dev)
    txt_flat, vec_all, lens_t = _encode_edits_packed(
        model, edit_refs, [INSTRUCTION], template, drop_idx, dev)
    txt, txt_cu, txt_mask, vec = _slice_packed(txt_flat, vec_all, lens_t, 0, 1, dev)
    txt = txt.detach().clone(); vec = vec.detach().clone()
    editcond_txt_np = txt.float().cpu().numpy()   # [1, N_TXT, 2560] DiT context

    # Re-run the processor once (outside the encoder) to grab the exact ids /
    # pixel_values / grid the Mojo gate feeds to encode_mageflow_edit.
    processor = model.txt_enc.processor
    body = _edit_prompt_body(INSTRUCTION, 1)
    formatted = template.format(body)
    vlp = processor(text=[formatted],
                    images=[_resize_long_edge(src_pil, VL_COND_LONG_EDGE)],
                    padding=True, return_tensors="pt")
    input_ids = vlp["input_ids"][0]
    pixel_values = vlp["pixel_values"].float()
    grid_thw = vlp["image_grid_thw"][0]
    L = int(input_ids.numel())
    N_TXT = L - drop_idx
    t_, gh_, gw_ = [int(x) for x in grid_thw.tolist()]
    seq = t_ * gh_ * gw_
    nvis = int((input_ids == IMAGE_TOKEN_ID).sum().item())
    print(f"[mf-edit-oracle] L={L} N_TXT={N_TXT} vision seq={seq} grid=({t_},{gh_},{gw_}) nvis={nvis}")

    model.txt_enc.to("cpu")
    torch.cuda.empty_cache()
    print("[mf-edit-oracle] STAGE1 edit-cond encoded; VL encoder offloaded")

    # ==== STAGE 2a: VAE-encode the ref at target resolution (mean) ====
    model.vae.to(dev)
    ref_tensor = _preprocess_ref_image(src_pil, H, W, dev)  # [3,H,W] in [-1,1]
    ref_tok, ref_shapes, ref_ids = model.compute_vae_encodings([ref_tensor], with_ids=True)
    ref_tok = ref_tok.to(torch.bfloat16)                    # [1, Lr, 128]
    ref_pixel_np = ref_tensor.float().cpu().numpy()         # [3,H,W]
    ref_latent_np = ref_tok.float().cpu().numpy()           # [1,Lr,128]
    Lr = ref_tok.shape[1]
    model.vae.to("cpu")
    torch.cuda.empty_cache()
    print(f"[mf-edit-oracle] STAGE2a ref VAE-encoded Lr={Lr}; VAE offloaded")

    # ---- target noise (get_noise + Gaussian-Shading encode_noise) ----
    ch = model.vae.latent_channels
    gs_key = resolve_gs_key(None)
    torch.manual_seed(SEED)
    x = get_noise(num_samples=1, channel=ch, height=H, width=W,
                  device=dev, dtype=torch.bfloat16, seed=SEED)
    x = encode_noise(tuple(x.shape[1:]), key=gs_key, seed=SEED, device=dev,
                     dtype=torch.bfloat16)
    _, _, gh, gw = x.shape
    Lt = gh * gw
    tgt = rearrange(x, "b c h w -> b (h w) c")              # [1, Lt, 128]
    init_noise_np = tgt.clone().float().cpu().numpy()

    # combined [target, ref] position ids + shapes (ids vestigial; RoPE uses shapes)
    tgt_ids = torch.zeros(gh, gw, 3, device=dev)
    tgt_ids[..., 1] = tgt_ids[..., 1] + torch.arange(gh, device=dev)[:, None]
    tgt_ids[..., 2] = tgt_ids[..., 2] + torch.arange(gw, device=dev)[None, :]
    tgt_ids = rearrange(tgt_ids, "h w c -> (h w) c").unsqueeze(0)
    img_ids = torch.cat([tgt_ids, ref_ids.to(dev)], dim=1)   # [1, Lt+Lr, 3]
    samp_lens = [Lt + Lr]
    img_cu = _lens_to_cu(samp_lens, dev)
    shape_seq = [(1, gh, gw)] + [s[0] for s in ref_shapes]   # target frame0, ref frame1
    img_shapes = [shape_seq]
    target_idx = torch.arange(0, Lt, device=dev)
    print(f"[mf-edit-oracle] Lt={Lt} Lr={Lr} SH={gh} shape_seq={shape_seq}")

    ctx = _build_pack_ctx(img_ids, img_cu, img_shapes, samp_lens, txt, txt_cu, txt_mask, vec,
                          None, None, None, None, CFG, False, True, dev)

    # ==== STAGE 2b: 4-step Euler denoise (target only; ref stays clean) ====
    model.transformer.to(dev)
    scheduler = _get_scheduler(model, STEPS, "cuda:0", None)
    sig = [float(s) for s in scheduler.sigmas.tolist()]
    print(f"[mf-edit-oracle] sigmas = {sig}")
    tgt = tgt.to(torch.bfloat16)
    ref_tok = ref_tok.to(dev)
    for si, t in enumerate(scheduler.timesteps):
        img = torch.cat([tgt, ref_tok], dim=1)               # [1, Lt+Lr, 128], ref clean
        vel = _velocity(model.transformer, img, ctx, scheduler.sigmas[si].item())
        pred_t = vel[:, target_idx, :]                       # target tokens only
        tgt = scheduler.step(pred_t, t, tgt, return_dict=False)[0]
    final_latent_np = tgt.float().cpu().numpy()              # [1, Lt, 128]
    model.transformer.to("cpu")
    torch.cuda.empty_cache()
    print("[mf-edit-oracle] STAGE2b denoise done; DiT offloaded")

    # ==== STAGE 3: VAE decode the target ====
    model.vae.to(dev)
    with torch.autocast(device_type=dev.type, dtype=torch.bfloat16):
        out = model.vae.decode(unpack(tgt.float(), H, W))    # [1,3,H,W]
    out = out.clamp(-1, 1)
    rgb = out.float().cpu().numpy()
    img_u8 = (127.5 * (rearrange(out, "b c h w -> b h w c") + 1.0)).cpu().byte().numpy()
    Image.fromarray(img_u8[0]).save(os.path.join(OUT, "oracle_edit.png"))
    model.vae.to("cpu")
    torch.cuda.empty_cache()

    shapes = {}
    shapes["input_ids.bin"] = dump_f32("input_ids.bin", input_ids.cpu().numpy())
    shapes["pixel_values.bin"] = dump_f32("pixel_values.bin", pixel_values.cpu().numpy())
    shapes["grid_thw.bin"] = dump_f32("grid_thw.bin", grid_thw.cpu().numpy())
    shapes["editcond_txt.bin"] = dump_f32("editcond_txt.bin", editcond_txt_np)
    shapes["ref_pixel.bin"] = dump_f32("ref_pixel.bin", ref_pixel_np)
    shapes["ref_latent.bin"] = dump_f32("ref_latent.bin", ref_latent_np)
    shapes["init_noise.bin"] = dump_f32("init_noise.bin", init_noise_np)
    shapes["final_latent.bin"] = dump_f32("final_latent.bin", final_latent_np)
    shapes["rgb.bin"] = dump_f32("rgb.bin", rgb)

    with open(os.path.join(OUT, "meta.txt"), "w") as f:
        f.write(f"instruction={INSTRUCTION!r}\n")
        f.write(f"image={IMAGE!r}\n")
        f.write(f"seed={SEED} steps={STEPS} cfg={CFG} H={H} W={W}\n")
        f.write(f"L={L} drop={drop_idx} N_TXT={N_TXT}\n")
        f.write(f"vision_seq={seq} grid_thw=({t_},{gh_},{gw_}) nvis={nvis}\n")
        f.write(f"Lt={Lt} Lr={Lr} SH={gh}\n")
        f.write(f"sigmas={sig}\n")
        f.write(f"input_ids={input_ids.cpu().tolist()}\n")
        for k, v in shapes.items():
            f.write(f"{k} shape={v}\n")
        f.write(f"ref_latent.std={float(np.std(ref_latent_np)):.6f}\n")
        f.write(f"init_noise.std={float(np.std(init_noise_np)):.6f}\n")
        f.write(f"final_latent.std={float(np.std(final_latent_np)):.6f}\n")
        f.write(f"rgb.min={float(rgb.min()):.4f} rgb.max={float(rgb.max()):.4f}\n")
    print(f"[mf-edit-oracle] dumped: {shapes}")
    print(f"[mf-edit-oracle] SET GATE COMPTIME: N_TXT={N_TXT} S_VIS={seq} "
          f"GRID_T={t_} GRID_H={gh_} GRID_W={gw_} Lt={Lt} Lr={Lr} SH={gh}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("[mf-edit-oracle] done")


if __name__ == "__main__":
    main()

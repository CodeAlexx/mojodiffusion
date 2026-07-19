#!/usr/bin/env python
# Oracle-E: the FULL LingBot T2I end-to-end pipeline reference (memory-lean).
#
#   prompt + negative -> Qwen3-VL encode  (then FREE the 8.9GB encoder)
#   randn init latent (1,16,1,lh,lw)      (H=W=256 -> lh=lw=32, n_video=256)
#   12-step FlowUniPC CFG denoise loop (gs=5) with the MEMORY-LEAN per-block
#     STREAMED transformer forward (48 blocks, one ~1.2GB block at a time from
#     the 13-shard 60GB model — same streaming as oracle_b2_backbone.py)
#   _dit_latent_to_vae -> AutoencoderKLWan.decode -> clamp -> (x+1)/2 -> pixels
#
# Saves oracle_e.safetensors (embeds, masks, init + per-step latents, final
# latent, pixels) + oracle_e_reference.png. This is the PARITY ORACLE the Mojo
# pipeline (Stage 2) is gated against; Python is oracle-only.
#
# Run from the creator source tree:
#   cd /mnt/disk1/lingbot-src/lingbot-video && \
#     /home/alex/SerenityTrainer/venv/bin/python \
#       /home/alex/mojodiffusion/serenitymojo/models/lingbotvideo/parity/oracle_e_t2i.py
import os, sys, json, math, time, gc
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch, torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file
from PIL import Image

sys.path.insert(0, "/mnt/disk1/lingbot-src/lingbot-video")
from lingbot_video.transformer_lingbot_video import (
    LingBotVideoBlock, LingBotVideoTextEmbedder, LingBotVideoRotaryEmbedding,
    make_joint_position_ids,
)
from lingbot_video.scheduling_flow_unipc import FlowUniPCMultistepScheduler
from lingbot_video.pipeline_lingbot_video import (
    PROMPT_TEMPLATE, DEFAULT_NEGATIVE_PROMPT_IMAGE, TOKEN_LENGTH,
)
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers import AutoencoderKLWan
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

ROOT = "/mnt/disk1/models/lingbot-video-moe"
MDIR = os.path.join(ROOT, "transformer")
TE_DIR = os.path.join(ROOT, "text_encoder")
PROC_DIR = os.path.join(ROOT, "processor")
VAE_DIR = os.path.join(ROOT, "vae")
OUT = "/home/alex/mojodiffusion/serenitymojo/models/lingbotvideo/parity"
DEV = "cuda"

# ── Real T2I settings (model's target res) ──────────────────────────────────
H = 480
W = 832
NUM_STEPS = 40
GUIDANCE = 5.0
SHIFT = 3.0
SEED = 12345
HIDDEN_STATE_SKIP_LAYER = 0

# The DiT consumes STRUCTURED-JSON captions (normally emitted by the rewriter's
# IMAGE_STEP2_MAP step; here we feed the JSON directly per the locked scope).
# Positive caption for a simple photoreal still-life, following the still-image
# schema in rewriter/system_prompts.py (comprehensive_description + camera_info
# + world_knowledge + prominent_elements). json.dumps -> compact valid JSON,
# same shape as DEFAULT_NEGATIVE_PROMPT_IMAGE (the negative, kept unchanged).
PROMPT = json.dumps({
    "comprehensive_description": (
        "A single glossy red apple rests at the center of a warm-toned wooden "
        "table, photographed as a crisp photorealistic studio still life. The "
        "apple's smooth waxy skin shows deep crimson and scarlet tones with "
        "subtle yellow-green mottling near its short brown stem, catching a "
        "bright specular highlight from a soft key light above. It sits on the "
        "horizontal planks of a rustic oak table whose visible grain runs left "
        "to right, lit by clean studio lighting that casts a soft contact "
        "shadow beneath the fruit against a smoothly graded neutral background."
    ),
    "camera_info": {
        "color": "Warm",
        "frame_size": "Close Up",
        "shot_type_angle": "Eye level",
        "lens_size": "Medium",
        "composition": "Center",
        "lighting": "Soft light",
        "lighting_type": "Artificial light",
    },
    "world_knowledge": [],
    "prominent_elements": [
        {
            "name": "red apple",
            "description": "A single ripe red apple, the dominant subject, with a rounded body and a short woody stem at the top.",
            "location": "center of the frame, resting on the table surface",
            "relative_size": "dominant",
            "shape_and_color": "rounded spherical form in deep red and crimson with faint yellow-green flecks",
            "texture": "smooth, waxy, glossy",
            "appearance_details": "bright specular highlight on the upper left, a small brown stem, faint natural mottling on the skin",
            "relationship": "sits on top of the wooden table, casting a soft shadow onto its surface",
            "orientation": "upright with the stem pointing up",
        },
        {
            "name": "wooden table",
            "description": "A rustic oak tabletop that fills the lower portion of the frame and supports the apple.",
            "location": "lower half of the frame, extending horizontally",
            "relative_size": "large",
            "shape_and_color": "flat horizontal surface in warm honey-brown tones",
            "texture": "matte wood with visible grain",
            "appearance_details": "natural wood grain lines running left to right, subtle knots and color variation",
            "relationship": "supports the red apple and receives its soft contact shadow",
            "orientation": "horizontal",
        },
    ],
})


# ═════════════════════════════════════════════════════════════════════════════
# 1. prompt encode (Qwen3-VL) — mirrors pipeline.encode_prompt for T2I / B=1.
# ═════════════════════════════════════════════════════════════════════════════
def _apply_template(text):
    return PROMPT_TEMPLATE.format(text)


def _compute_crop_start(processor):
    marker = "<|USER_INPUT_MARKER|>"
    marked = PROMPT_TEMPLATE.format(marker)
    pos = marked.find(marker)
    if pos < 0:
        return 0
    prefix = processor(text=marked[:pos], images=None, videos=None, return_tensors="pt")
    return int(prefix["input_ids"].shape[1])


@torch.no_grad()
def encode_one(text_encoder, processor, crop_start, prompt):
    text = _apply_template(prompt)
    inputs = processor(
        text=[text], images=None, videos=None, do_resize=False, truncation=True,
        max_length=TOKEN_LENGTH, padding="longest", return_tensors="pt",
    ).to(DEV)
    outputs = text_encoder(**inputs, output_hidden_states=True)
    embeds = outputs.hidden_states[-(HIDDEN_STATE_SKIP_LAYER + 1)]
    mask = inputs["attention_mask"]
    if crop_start > 0:
        embeds = embeds[:, crop_start:]
        mask = mask[:, crop_start:]
    # B=1: drop right padding.
    true_len = int(mask[0].sum().item())
    embeds = embeds[:, :true_len]
    mask = mask[:, :true_len]
    return embeds, mask


def encode_prompts():
    print("[E] loading Qwen3-VL text encoder (8.9GB) + processor ...", flush=True)
    processor = AutoProcessor.from_pretrained(PROC_DIR, trust_remote_code=True)
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        TE_DIR, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(DEV).eval()
    crop_start = _compute_crop_start(processor)
    print(f"[E] crop_start = {crop_start}", flush=True)
    p_embeds, p_mask = encode_one(text_encoder, processor, crop_start, PROMPT)
    n_embeds, n_mask = encode_one(text_encoder, processor, crop_start, DEFAULT_NEGATIVE_PROMPT_IMAGE)
    print(f"[E] prompt embeds {list(p_embeds.shape)} | negative embeds {list(n_embeds.shape)}", flush=True)
    # capture on CPU (float) then FREE the encoder — must not coexist with the
    # streamed transformer.
    out = (
        p_embeds.float().cpu(), p_mask.cpu(),
        n_embeds.float().cpu(), n_mask.cpu(),
    )
    del text_encoder, processor
    gc.collect()
    torch.cuda.empty_cache()
    print("[E] text encoder freed", flush=True)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 2. memory-lean streamed transformer forward (48 blocks, one at a time).
#    Refactor of oracle_b2_backbone.py into streamed_forward(latent, tb, embeds).
# ═════════════════════════════════════════════════════════════════════════════
class StreamedTransformer:
    def __init__(self):
        self.cfg = json.load(open(os.path.join(MDIR, "config.json")))
        self.wm = json.load(open(os.path.join(
            MDIR, "diffusion_pytorch_model.safetensors.index.json")))["weight_map"]
        self.handles = {}
        c = self.cfg
        self.H = c["hidden_size"]; self.DEPTH = c["depth"]; self.PATCH = tuple(c["patch_size"])
        self.IN_CH = c["in_channels"]; self.TEXT_DIM = c["text_dim"]; self.FREQ = c["freq_dim"]
        self.EPS = c["norm_eps"]; self.OUT_CH = c["out_channels"]
        H = self.H
        # embedding + post modules (real weights, small; loaded once).
        self.patch_embedder = nn.Linear(self.IN_CH * math.prod(self.PATCH), H, bias=True)
        self.patch_embedder.weight.data = self.W("patch_embedder.weight")
        self.patch_embedder.bias.data = self.W("patch_embedder.bias")
        self.text_embedder = LingBotVideoTextEmbedder(self.TEXT_DIM, H)
        self.text_embedder.norm.weight.data = self.W("text_embedder.norm.weight")
        self.text_embedder.linear_1.weight.data = self.W("text_embedder.linear_1.weight")
        self.text_embedder.linear_1.bias.data = self.W("text_embedder.linear_1.bias")
        self.text_embedder.linear_2.weight.data = self.W("text_embedder.linear_2.weight")
        self.text_embedder.linear_2.bias.data = self.W("text_embedder.linear_2.bias")
        self.time_proj = Timesteps(self.FREQ, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(self.FREQ, H, act_fn="silu", sample_proj_bias=True)
        self.time_embedder.linear_1.weight.data = self.W("time_embedder.linear_1.weight")
        self.time_embedder.linear_1.bias.data = self.W("time_embedder.linear_1.bias")
        self.time_embedder.linear_2.weight.data = self.W("time_embedder.linear_2.weight")
        self.time_embedder.linear_2.bias.data = self.W("time_embedder.linear_2.bias")
        self.time_modulation = nn.Sequential(nn.SiLU(), nn.Linear(H, 6 * H))
        self.time_modulation[1].weight.data = self.W("time_modulation.1.weight")
        self.time_modulation[1].bias.data = self.W("time_modulation.1.bias")
        self.rope = LingBotVideoRotaryEmbedding(tuple(c["axes_dims"]), tuple(c["axes_lens"]), c["rope_theta"])
        self.norm_out = nn.LayerNorm(H, elementwise_affine=False, eps=self.EPS)
        self.norm_out_modulation = nn.Sequential(nn.SiLU(), nn.Linear(H, 2 * H))
        self.norm_out_modulation[1].weight.data = self.W("norm_out_modulation.1.weight")
        self.norm_out_modulation[1].bias.data = self.W("norm_out_modulation.1.bias")
        self.proj_out = nn.Linear(H, math.prod(self.PATCH) * self.OUT_CH)
        self.proj_out.weight.data = self.W("proj_out.weight")
        self.proj_out.bias.data = self.W("proj_out.bias")
        for m in (self.patch_embedder, self.text_embedder, self.time_embedder,
                  self.time_modulation, self.norm_out, self.norm_out_modulation, self.proj_out):
            m.to(DEV).eval()

    def W(self, name):
        shard = self.wm[name]
        if shard not in self.handles:
            self.handles[shard] = safe_open(os.path.join(MDIR, shard), framework="pt", device="cpu")
        return self.handles[shard].get_tensor(name)

    @staticmethod
    def transformer_timestep(t_int, transformer_dtype=torch.bfloat16):
        # pipeline._transformer_timestep: sigma=t/1000; to(bf16); (sigma*1000).float()
        sigma = torch.tensor([float(t_int)], device=DEV).float() / 1000.0
        sigma = sigma.to(transformer_dtype)
        return (sigma * 1000.0).float()

    @torch.no_grad()
    def forward(self, latent, timestep_batch, text_embeds_bf16):
        c = self.cfg
        pF, pH, pW = self.PATCH
        B, C, T, Hh, Ww = latent.shape
        gt, gh, gw = T // pF, Hh // pH, Ww // pW
        n_video = gt * gh * gw
        text_len = text_embeds_bf16.shape[1]
        S = n_video + text_len
        # patchify (channel-fastest within patch): (0,2,4,6,3,5,7,1)
        pt = latent.reshape(B, C, gt, pF, gh, pH, gw, pW)
        pt = pt.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(B, n_video, pF * pH * pW * C)
        x = self.patch_embedder(pt.to(self.patch_embedder.weight.dtype))
        text = self.text_embedder(text_embeds_bf16.to(self.text_embedder.linear_1.weight.dtype))
        joint = torch.cat([x, text], dim=1)
        pos_ids = make_joint_position_ids(text_len, gt, gh, gw, torch.device(DEV))
        freqs = self.rope(pos_ids)
        rotary = freqs.unsqueeze(0)
        t_emb = self.time_embedder(self.time_proj(timestep_batch.float()))
        temb_input = t_emb.unsqueeze(1).expand(B, S, -1)
        temb6 = self.time_modulation(temb_input.reshape(B * S, -1))

        # stream the 48 blocks (one ~1.2GB block at a time)
        for i in range(self.DEPTH):
            blk = LingBotVideoBlock(
                hidden_size=self.H, num_attention_heads=c["num_attention_heads"],
                intermediate_size=c["intermediate_size"], norm_eps=self.EPS,
                qkv_bias=c["qkv_bias"], out_bias=c["out_bias"], num_experts=c["num_experts"],
                num_experts_per_tok=c["num_experts_per_tok"], moe_intermediate_size=c["moe_intermediate_size"],
                decoder_sparse_step=c["decoder_sparse_step"], mlp_only_layers=tuple(c["mlp_only_layers"]),
                n_shared_experts=c["n_shared_experts"], score_func=c["score_func"],
                norm_topk_prob=c["norm_topk_prob"], n_group=c["n_group"], topk_group=c["topk_group"],
                routed_scaling_factor=c["routed_scaling_factor"], layer_idx=i)
            pref = f"blocks.{i}."
            sd = {k[len(pref):]: self.W(k) for k in self.wm if k.startswith(pref)}
            blk.load_state_dict(sd, assign=True)
            blk.to(DEV).eval()
            blk.ffn._run_grouped_experts = blk.ffn._run_experts_for_loop.__get__(blk.ffn, type(blk.ffn))
            joint = blk(joint, temb6, rotary)
            blk.ffn._run_grouped_experts = None
            del blk, sd
            gc.collect()
            torch.cuda.empty_cache()

        final_mod = self.norm_out_modulation(temb_input.reshape(B * S, -1))
        shift, scale = final_mod.reshape(B, S, -1).chunk(2, dim=-1)
        final_hidden = self.norm_out(joint) * (1.0 + scale) + shift
        projected = self.proj_out(final_hidden.to(self.proj_out.weight.dtype))
        xv = projected[:, :n_video]
        Cout = self.OUT_CH
        xv = xv.reshape(B, gt, gh, gw, pF, pH, pW, Cout)
        xv = xv.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(B, Cout, T, Hh, Ww)
        return xv.float()


# ═════════════════════════════════════════════════════════════════════════════
# 3. VAE decode helpers
# ═════════════════════════════════════════════════════════════════════════════
def dit_latent_to_vae(latents, mean, std):
    m = torch.tensor(mean, device=latents.device, dtype=torch.float32).view(1, -1, 1, 1, 1)
    s = torch.tensor(std, device=latents.device, dtype=torch.float32).view(1, -1, 1, 1, 1)
    return latents.float() * s + m


def main():
    t_start = time.time()
    caps = {}

    # ---- 1. encode prompts (then free encoder) ----
    p_embeds, p_mask, n_embeds, n_mask = encode_prompts()
    caps["prompt_embeds"] = p_embeds
    caps["prompt_mask"] = p_mask.float()
    caps["neg_embeds"] = n_embeds
    caps["neg_mask"] = n_mask.float()
    L_cond = p_embeds.shape[1]
    L_uncond = n_embeds.shape[1]

    # ---- 2. init latent (T2I, num_frames=1) ----
    torch.manual_seed(SEED)
    gen = torch.Generator(device=DEV).manual_seed(SEED)
    lh, lw = H // 8, W // 8
    latent = torch.randn((1, 16, 1, lh, lw), generator=gen, device=DEV, dtype=torch.float32)
    caps["init_latent"] = latent.float().cpu()
    print(f"[E] init latent {list(latent.shape)}  mean {latent.mean():.4f} std {latent.std():.4f}", flush=True)

    # ---- 3. scheduler ----
    sch = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
    sch.set_timesteps(NUM_STEPS, device=DEV, shift=SHIFT)
    print(f"[E] timesteps: {[int(t) for t in sch.timesteps.tolist()]}", flush=True)

    # ---- 4. streamed transformer + CFG denoise loop ----
    net = StreamedTransformer()
    p_bf16 = p_embeds.to(DEV, dtype=torch.bfloat16)
    n_bf16 = n_embeds.to(DEV, dtype=torch.bfloat16)

    for i, t in enumerate(sch.timesteps):
        t_int = int(t.item())
        tb = net.transformer_timestep(t_int)
        f0 = time.time()
        noise_cond = net.forward(latent, tb, p_bf16)
        noise_uncond = net.forward(latent, tb, n_bf16)
        noise = noise_uncond + GUIDANCE * (noise_cond - noise_uncond)
        caps[f"latent_step_{i}"] = latent.float().cpu()
        latent = sch.step(noise, t, latent, return_dict=False)[0]
        print(f"[E] step {i:02d}/{NUM_STEPS} t={t_int:4d} "
              f"lat(std={latent.std().item():.4f} absmax={latent.abs().max().item():.3f}) "
              f"noise(std={noise.std().item():.4f}) [{time.time()-f0:.0f}s]", flush=True)

    caps["final_latent"] = latent.float().cpu()

    # ---- 5. VAE decode ----
    print("[E] loading AutoencoderKLWan (fp32) ...", flush=True)
    vae = AutoencoderKLWan.from_pretrained(VAE_DIR, torch_dtype=torch.float32).to(DEV).eval()
    with torch.no_grad():
        vae_latent = dit_latent_to_vae(latent, vae.config.latents_mean, vae.config.latents_std)
        vae_latent = vae_latent.to(device=DEV, dtype=torch.float32).contiguous(memory_format=torch.channels_last_3d)
        decoded = vae.decode(vae_latent)
        frames = decoded[0] if isinstance(decoded, tuple) else decoded.sample
        frames = frames.float().clamp_(-1, 1)
        frames = (frames + 1.0) / 2.0     # [1,3,1,H,W] in [0,1]
    pixels = frames.float().cpu()
    caps["pixels"] = pixels
    print(f"[E] pixels {list(pixels.shape)} mean {pixels.mean():.4f} std {pixels.std():.4f}", flush=True)

    # ---- 6. save safetensors + PNG ----
    save_file(caps, os.path.join(OUT, "oracle_e.safetensors"))
    meta = {"H": H, "W": W, "num_steps": NUM_STEPS, "guidance": GUIDANCE, "shift": SHIFT,
            "seed": SEED, "prompt": PROMPT, "L_cond": L_cond, "L_uncond": L_uncond,
            "n_video": (lh // 2) * (lw // 2), "lh": lh, "lw": lw}
    json.dump(meta, open(os.path.join(OUT, "oracle_e_meta.json"), "w"), indent=2)

    img = pixels[0, :, 0].permute(1, 2, 0).numpy()   # H,W,3
    img = np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(os.path.join(OUT, "oracle_e_reference.png"))

    print(f"[E] SAVED oracle_e.safetensors + oracle_e_reference.png "
          f"(L_cond={L_cond} L_uncond={L_uncond})  total {time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()

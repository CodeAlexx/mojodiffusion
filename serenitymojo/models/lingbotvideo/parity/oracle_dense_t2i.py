#!/usr/bin/env python
# Oracle-DENSE: the FULL LingBot-Video-Dense-1.3B T2I end-to-end reference,
# RESIDENT (no streaming — the 2.6GB bf16 transformer fits 16GB VRAM). This is
# the TRUSTWORTHY dense ground truth (unlike the MoE oracle, the WHOLE model is
# resident, so it is exactly reproducible).
#
# Dense-1.3B == the MoE architecture EXCEPT: FFN is a plain LingBotVideoMLP
# (num_experts=0 in the config -> LingBotVideoBlock builds LingBotVideoMLP), and
# depth=24 (not 48), intermediate_size=6144. Same hidden 2048 / 16 heads /
# head_dim 128 / patch [1,2,2] / 16-ch latent / text_dim 2560 / RoPE axes
# [32,48,48] / same Qwen3-VL text encoder (REUSED from the MoE dir) / same Wan
# VAE / same FlowUniPC scheduler.
#
#   prompt + negative -> Qwen3-VL encode  (then FREE the 8.9GB encoder)
#   [tap forward] one small S~72 forward with per-block taps + velocity capture
#   randn init latent (1,16,1,60,104)  (H=480,W=832)
#   40-step FlowUniPC CFG denoise loop (gs=5) with the RESIDENT transformer
#   _dit_latent_to_vae -> AutoencoderKLWan.decode -> clamp -> (x+1)/2 -> pixels
#
# Saves oracle_dense.safetensors (embeds, init + per-step + final latents, pixels,
# plus the sf_* tap-forward captures) + oracle_dense_reference.png.
#
# Run from the creator source tree:
#   cd /mnt/disk1/lingbot-src/lingbot-video && \
#     /home/alex/SerenityTrainer/venv/bin/python \
#       /home/alex/mojodiffusion/serenitymojo/models/lingbotvideo/parity/oracle_dense_t2i.py
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

DENSE_ROOT = "/mnt/disk1/models/lingbot-video-dense"
MDIR = os.path.join(DENSE_ROOT, "transformer")       # dense transformer (single file)
VAE_DIR = os.path.join(DENSE_ROOT, "vae")            # dense vae (== MoE vae, identical)
MOE_ROOT = "/mnt/disk1/models/lingbot-video-moe"
TE_DIR = os.path.join(MOE_ROOT, "text_encoder")      # REUSE MoE Qwen3-VL (identical)
PROC_DIR = os.path.join(MOE_ROOT, "processor")
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

# Structured-JSON caption (same schema/scene as oracle_e_t2i.py — a photoreal
# red-apple still-life). Fed directly per the locked scope.
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
    true_len = int(mask[0].sum().item())
    embeds = embeds[:, :true_len]
    mask = mask[:, :true_len]
    return embeds, mask


def encode_prompts():
    print("[D] loading Qwen3-VL text encoder (8.9GB) + processor ...", flush=True)
    processor = AutoProcessor.from_pretrained(PROC_DIR, trust_remote_code=True)
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        TE_DIR, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(DEV).eval()
    crop_start = _compute_crop_start(processor)
    print(f"[D] crop_start = {crop_start}", flush=True)
    p_embeds, p_mask = encode_one(text_encoder, processor, crop_start, PROMPT)
    n_embeds, n_mask = encode_one(text_encoder, processor, crop_start, DEFAULT_NEGATIVE_PROMPT_IMAGE)
    print(f"[D] prompt embeds {list(p_embeds.shape)} | negative embeds {list(n_embeds.shape)}", flush=True)
    out = (
        p_embeds.float().cpu(), p_mask.cpu(),
        n_embeds.float().cpu(), n_mask.cpu(),
    )
    del text_encoder, processor
    gc.collect()
    torch.cuda.empty_cache()
    print("[D] text encoder freed", flush=True)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 2. RESIDENT dense transformer (all 24 blocks live on GPU, built once).
# ═════════════════════════════════════════════════════════════════════════════
class ResidentDenseTransformer:
    def __init__(self):
        self.cfg = json.load(open(os.path.join(MDIR, "config.json")))
        c = self.cfg
        assert c["num_experts"] == 0, "expected DENSE config (num_experts==0)"
        # single-file weights (no index.json for the dense model).
        self.handle = safe_open(
            os.path.join(MDIR, "diffusion_pytorch_model.safetensors"),
            framework="pt", device="cpu")
        self.keys = set(self.handle.keys())
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

        # ── build ALL 24 dense blocks RESIDENT (num_experts=0 -> LingBotVideoMLP) ──
        self.blocks = []
        t0 = time.time()
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
            from lingbot_video.transformer_lingbot_video import LingBotVideoMLP
            assert isinstance(blk.ffn, LingBotVideoMLP), f"block {i} ffn is not dense MLP"
            pref = f"blocks.{i}."
            sd = {k[len(pref):]: self.W(k) for k in self.keys if k.startswith(pref)}
            blk.load_state_dict(sd, assign=True)  # keep stored dtypes (bf16 bulk / f32 norms)
            blk.to(DEV).eval()
            self.blocks.append(blk)
        torch.cuda.synchronize()
        mem = torch.cuda.memory_allocated() / 1e9
        print(f"[D] resident dense transformer: {self.DEPTH} blocks live on GPU "
              f"({time.time()-t0:.1f}s, {mem:.2f} GB allocated)", flush=True)

    def W(self, name):
        return self.handle.get_tensor(name)

    @staticmethod
    def transformer_timestep(t_int, transformer_dtype=torch.bfloat16):
        sigma = torch.tensor([float(t_int)], device=DEV).float() / 1000.0
        sigma = sigma.to(transformer_dtype)
        return (sigma * 1000.0).float()

    @torch.no_grad()
    def forward(self, latent, timestep_batch, text_embeds_bf16, capture=None):
        c = self.cfg
        pF, pH, pW = self.PATCH
        B, C, T, Hh, Ww = latent.shape
        gt, gh, gw = T // pF, Hh // pH, Ww // pW
        n_video = gt * gh * gw
        text_len = text_embeds_bf16.shape[1]
        S = n_video + text_len
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

        if capture is not None:
            capture["sf_joint_preblock"] = joint.float().cpu()
            capture["sf_temb6"] = temb6.float().cpu()
            capture["sf_freqs_cos"] = freqs.real.float().cpu()
            capture["sf_freqs_sin"] = freqs.imag.float().cpu()

        for i in range(self.DEPTH):
            joint = self.blocks[i](joint, temb6, rotary)
            if capture is not None:
                capture[f"sf_block_{i}"] = joint.float().cpu()

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
# 3. VAE decode helper
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

    # ---- 2. build RESIDENT dense transformer ----
    net = ResidentDenseTransformer()

    # ---- 2b. tap forward (small S~72) for per-block + velocity parity ----
    torch.manual_seed(31415)
    GT, GH, GW, TEXT_LEN = 1, 8, 8, 8       # latent 16x16 -> 8x8 video toks; S=72
    sf_latent = torch.randn(1, net.IN_CH, GT, GH * net.PATCH[1], GW * net.PATCH[2], device=DEV, dtype=torch.float32)
    sf_timestep = torch.tensor([500.0], device=DEV)
    sf_text = torch.randn(1, TEXT_LEN, net.TEXT_DIM, device=DEV, dtype=torch.bfloat16)
    caps["sf_latent"] = sf_latent.float().cpu()
    caps["sf_timestep"] = sf_timestep.float().cpu()
    caps["sf_text_embeds"] = sf_text.float().cpu()
    sf_tb = net.transformer_timestep(int(sf_timestep.item()))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        sf_vel = net.forward(sf_latent, sf_tb, sf_text, capture=caps)
    caps["sf_velocity"] = sf_vel.float().cpu()
    print(f"[D] tap forward S={GT*GH*GW+TEXT_LEN}: velocity mean {sf_vel.mean():.5f} "
          f"std {sf_vel.std():.5f} absmax {sf_vel.abs().max():.4f}", flush=True)

    # ---- 3. init latent (T2I, num_frames=1) ----
    torch.manual_seed(SEED)
    gen = torch.Generator(device=DEV).manual_seed(SEED)
    lh, lw = H // 8, W // 8
    latent = torch.randn((1, 16, 1, lh, lw), generator=gen, device=DEV, dtype=torch.float32)
    caps["init_latent"] = latent.float().cpu()
    print(f"[D] init latent {list(latent.shape)}  mean {latent.mean():.4f} std {latent.std():.4f}", flush=True)

    # ---- 4. scheduler ----
    sch = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
    sch.set_timesteps(NUM_STEPS, device=DEV, shift=SHIFT)
    print(f"[D] timesteps: {[int(t) for t in sch.timesteps.tolist()]}", flush=True)

    # ---- 5. RESIDENT CFG denoise loop ----
    p_bf16 = p_embeds.to(DEV, dtype=torch.bfloat16)
    n_bf16 = n_embeds.to(DEV, dtype=torch.bfloat16)

    for i, t in enumerate(sch.timesteps):
        t_int = int(t.item())
        tb = net.transformer_timestep(t_int)
        f0 = time.time()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            noise_cond = net.forward(latent, tb, p_bf16)
            noise_uncond = net.forward(latent, tb, n_bf16)
        noise = noise_uncond + GUIDANCE * (noise_cond - noise_uncond)
        caps[f"latent_step_{i}"] = latent.float().cpu()
        latent = sch.step(noise, t, latent, return_dict=False)[0]
        print(f"[D] step {i:02d}/{NUM_STEPS} t={t_int:4d} "
              f"lat(std={latent.std().item():.4f} absmax={latent.abs().max().item():.3f}) "
              f"noise(std={noise.std().item():.4f}) [{time.time()-f0:.1f}s]", flush=True)

    caps["final_latent"] = latent.float().cpu()

    # ---- 6. VAE decode ----
    print("[D] loading AutoencoderKLWan (fp32) ...", flush=True)
    del net
    gc.collect(); torch.cuda.empty_cache()
    vae = AutoencoderKLWan.from_pretrained(VAE_DIR, torch_dtype=torch.float32).to(DEV).eval()
    with torch.no_grad():
        vae_latent = dit_latent_to_vae(latent, vae.config.latents_mean, vae.config.latents_std)
        vae_latent = vae_latent.to(device=DEV, dtype=torch.float32).contiguous(memory_format=torch.channels_last_3d)
        decoded = vae.decode(vae_latent)
        frames = decoded[0] if isinstance(decoded, tuple) else decoded.sample
        frames = frames.float().clamp_(-1, 1)
        frames = (frames + 1.0) / 2.0
    pixels = frames.float().cpu()
    caps["pixels"] = pixels
    print(f"[D] pixels {list(pixels.shape)} mean {pixels.mean():.4f} std {pixels.std():.4f}", flush=True)

    # ---- 7. save safetensors + PNG ----
    save_file(caps, os.path.join(OUT, "oracle_dense.safetensors"))
    meta = {"H": H, "W": W, "num_steps": NUM_STEPS, "guidance": GUIDANCE, "shift": SHIFT,
            "seed": SEED, "prompt": PROMPT, "L_cond": L_cond, "L_uncond": L_uncond,
            "depth": 24,
            "n_video": (lh // 2) * (lw // 2), "lh": lh, "lw": lw,
            "sf_S": GT*GH*GW+TEXT_LEN, "sf_GT": GT, "sf_GH": GH, "sf_GW": GW, "sf_TEXT_LEN": TEXT_LEN}
    json.dump(meta, open(os.path.join(OUT, "oracle_dense_meta.json"), "w"), indent=2)

    img = pixels[0, :, 0].permute(1, 2, 0).numpy()
    img = np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(os.path.join(OUT, "oracle_dense_reference.png"))

    print(f"[D] SAVED oracle_dense.safetensors + oracle_dense_reference.png "
          f"(L_cond={L_cond} L_uncond={L_uncond})  total {time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()

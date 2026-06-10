# DEV-ONLY oracle for the Ideogram-4 TRAINING predict path on a REAL giger3
# sample (real image -> VAE latent, real .json caption -> Qwen3-VL features),
# 1:1 from ai-toolkit ideogram4: encode_images + get_prompt_embeds + the flow
# add_noise + predict_velocity (pipeline.py) + get_loss_target.
# Loads ONE model at a time (VAE -> Qwen -> transformer) for 24GB.
# Dumps the latent/noise/t/features + every packed intermediate + velocity+target
# so the Mojo predict path can be gated element-wise on real data.
import sys, json, gc, torch
sys.path.insert(0, "/home/alex/ideogram4-ref/src")
from PIL import Image
from tokenizers import Tokenizer
from safetensors.torch import load_file, save_file
from ideogram4.autoencoder import AutoEncoder, AutoEncoderParams, convert_diffusers_state_dict
from ideogram4.latent_norm import get_latent_norm
from ideogram4.constants import (
    IMAGE_POSITION_OFFSET, LLM_TOKEN_INDICATOR, OUTPUT_IMAGE_INDICATOR,
    SEQUENCE_PADDING_INDICATOR, QWEN3_VL_ACTIVATION_LAYERS,
)

ROOT = "/home/alex/.serenity/models/ideogram-4-fp8"
OUT = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity"
IMG = "/home/alex/1/datasets/gigerver3_json/10.jpg"
CAP = "/home/alex/1/datasets/gigerver3_json/10.json"
dev = torch.device("cuda"); dt = torch.bfloat16
RES = 256; PATCH = 2  # real giger image at 256 (seq = 256 img + ~651 text); fits 24GB
def free():
    gc.collect(); torch.cuda.empty_cache()


def patchify_latents(z, patch_size=2):
    b, ae_ch, h8, w8 = z.shape
    ph = pw = patch_size
    gh, gw = h8 // ph, w8 // pw
    z = z.view(b, ae_ch, gh, ph, gw, pw).permute(0, 3, 5, 1, 2, 4).reshape(b, ph * pw * ae_ch, gh, gw)
    return z


def main():
    torch.manual_seed(0)
    # ── real giger image -> [1,3,512,512] in [-1,1] (dump the exact preprocessed tensor) ──
    im = Image.open(IMG).convert("RGB").resize((RES, RES), Image.BICUBIC)
    img = (torch.from_numpy(__import__("numpy").asarray(im)).float() / 255.0).permute(2, 0, 1).unsqueeze(0)
    img = (img * 2.0 - 1.0).to(dev)                                        # [1,3,512,512]

    # ── VAE encode (ai-toolkit encode_images) -> latents [1,128,32,32] ──
    ae = AutoEncoder(AutoEncoderParams())
    ae.load_state_dict(convert_diffusers_state_dict(load_file(f"{ROOT}/vae/diffusion_pytorch_model.safetensors")))
    ae.to(device=dev, dtype=dt); ae.eval()
    with torch.no_grad():
        moments = ae.encoder(img.to(dt))
    mean = moments[:, :ae.params.z_channels]
    patched = patchify_latents(mean.float(), PATCH)
    shift, scale = get_latent_norm()
    shift = shift.view(1, -1, 1, 1).to(dev, torch.float32); scale = scale.view(1, -1, 1, 1).to(dev, torch.float32)
    latents = (patched - shift) / scale                                   # [1,128,32,32] = batch.latents
    del ae; free()
    GH = GW = latents.shape[-1]                                           # 32
    print(f"[predict] real latent {tuple(latents.shape)} std={latents.std():.4f}")

    # ── real .json caption -> Qwen3-VL 13-tap features [1,Lt,53248] ──
    raw = open(CAP, "r", encoding="utf-8").read()
    tok = Tokenizer.from_file(f"{ROOT}/tokenizer/tokenizer.json")
    rendered = "<|im_start|>user\n" + raw + "<|im_end|>\n<|im_start|>assistant\n"
    ids = tok.encode(rendered, add_special_tokens=False).ids
    token_ids = torch.tensor([ids], dtype=torch.long, device=dev); LT = token_ids.shape[1]
    from transformers import AutoConfig, AutoModel
    from ideogram4.quantized_loading import swap_linears_to_fp8, load_fp8_state_dict
    qcfg = AutoConfig.from_pretrained(f"{ROOT}/text_encoder")
    # transformers 4.57 Qwen3-VL wants config.rope_scaling; the checkpoint stores it
    # under the newer rope_parameters key -> bridge it (mrope_section [24,20,20], theta 5e6).
    _rp = getattr(qcfg.text_config, "rope_parameters", None) or {"rope_type": "default", "mrope_section": [24, 20, 20], "rope_theta": 5000000}
    qcfg.text_config.rope_scaling = dict(_rp)
    qcfg.text_config.rope_theta = _rp.get("rope_theta", 5000000)
    qm = AutoModel.from_config(qcfg)
    qsd = load_file(f"{ROOT}/text_encoder/model.safetensors"); swap_linears_to_fp8(qm, qsd, compute_dtype=dt)
    load_fp8_state_dict(qm, qsd, device=dev, dtype=dt, assign=True, strict=False); qm.eval()
    lm = qm.language_model
    ie = lm.embed_tokens(token_ids)
    p2 = (torch.ones_like(token_ids).cumsum(-1) - 1).clamp(min=0)
    p4 = p2[None].expand(4, 1, -1); tpos = p4[0]; mpos = p4[1:]
    from transformers.masking_utils import create_causal_mask
    cm = create_causal_mask(config=lm.config, input_embeds=ie, attention_mask=torch.ones_like(token_ids), cache_position=torch.arange(LT, device=dev), past_key_values=None, position_ids=tpos)
    pe = lm.rotary_emb(ie, mpos); taps = set(QWEN3_VL_ACTIVATION_LAYERS); cap = {}; hs = ie
    with torch.no_grad():
        for i, ly in enumerate(lm.layers):
            hs = ly(hs, attention_mask=cm, position_ids=tpos, past_key_values=None, position_embeddings=pe)
            if isinstance(hs, tuple): hs = hs[0]
            if i in taps: cap[i] = hs
    sel = [cap[i] for i in QWEN3_VL_ACTIVATION_LAYERS]
    llm_features = torch.permute(torch.stack(sel, 0), (1, 2, 3, 0)).reshape(1, LT, -1).to(dt)  # [1,LT,53248] GPU-resident (~65MB)
    # free EVERY Qwen reference (lm/ie/hs/cap/sel/pe/cm hold its 11GB) so it releases before the transformer loads.
    del qm, qsd, lm, ie, hs, cap, sel, pe, cm, p2, p4, tpos, mpos
    free()
    print(f"[predict] real caption LT={LT} features {tuple(llm_features.shape)} (VRAM {torch.cuda.memory_allocated()/1e9:.2f}GB)")

    # ── flow add_noise + get_loss_target ──
    noise = torch.randn(1, 128, GH, GW, device=dev, dtype=torch.float32)
    t = torch.tensor([0.7], device=dev, dtype=torch.float32)
    t01 = t.view(-1, 1, 1, 1)
    noisy = (1.0 - t01) * latents + t01 * noise                          # ai-toolkit flow add_noise
    target = (noise - latents)                                           # get_loss_target

    # ── predict_velocity (COPY of ai-toolkit pipeline.py:152-250, dump intermediates) ──
    from ideogram4.modeling_ideogram4 import Ideogram4Config, Ideogram4Transformer
    sd = load_file(f"{ROOT}/transformer/diffusion_pytorch_model.safetensors")
    m = Ideogram4Transformer(Ideogram4Config()); m.to(dt)
    swap_linears_to_fp8(m, sd, compute_dtype=dt); load_fp8_state_dict(m, sd, device=dev, dtype=dt); m.eval()
    del sd; free()
    text_mask = torch.ones(1, LT, dtype=torch.long, device=dev)
    b, c, gh, gw = noisy.shape
    num_image = gh * gw; num_text = LT; seq_len = num_text + num_image
    image_tokens = noisy.permute(0, 2, 3, 1).reshape(b, num_image, c)
    tm_bool = text_mask > 0; tm_long = tm_bool.long()
    x = torch.cat([torch.zeros(b, num_text, c, device=dev, dtype=image_tokens.dtype), image_tokens], dim=1)
    llm_full = torch.cat([llm_features, torch.zeros(b, num_image, llm_features.shape[-1], device=dev, dtype=dt)], dim=1)
    indicator = torch.zeros(b, seq_len, dtype=torch.long, device=dev)
    indicator[:, :num_text] = tm_long * LLM_TOKEN_INDICATOR
    indicator[:, num_text:] = OUTPUT_IMAGE_INDICATOR
    segment_ids = torch.ones(b, seq_len, dtype=torch.long, device=dev)
    segment_ids[:, :num_text] = torch.where(tm_bool, torch.ones_like(tm_long), torch.full_like(tm_long, SEQUENCE_PADDING_INDICATOR))
    text_pos = (tm_long.cumsum(dim=-1) - 1).clamp(min=0)
    text_pos_3d = text_pos.unsqueeze(-1).expand(-1, -1, 3)
    h_idx = torch.arange(gh, device=dev).view(-1, 1).expand(gh, gw).reshape(-1)
    w_idx = torch.arange(gw, device=dev).view(1, -1).expand(gh, gw).reshape(-1)
    image_pos = torch.stack([torch.zeros_like(h_idx), h_idx, w_idx], dim=1) + IMAGE_POSITION_OFFSET
    position_ids = torch.cat([text_pos_3d, image_pos.unsqueeze(0).expand(b, -1, -1)], dim=1)
    model_t = 1.0 - t
    with torch.no_grad():
        out = m(llm_features=llm_full, x=x.to(dt), t=model_t, position_ids=position_ids, segment_ids=segment_ids, indicator=indicator)
    image_velocity = out[:, num_text:].reshape(b, gh, gw, c).permute(0, 3, 1, 2)
    velocity = -image_velocity                                           # toolkit velocity

    fx = {
        "image_512": img.float().cpu(), "clean_latent": latents.cpu(), "noise": noise.cpu(),
        "t": t.cpu(), "model_t": model_t.cpu(), "noisy": noisy.float().cpu(), "target": target.float().cpu(),
        "llm_features": llm_features.float().cpu(), "text_mask": text_mask.to(torch.int32).cpu(),
        "token_ids": token_ids.to(torch.int32).cpu(),
        "x": x.float().cpu(),
        "position_ids": position_ids.to(torch.int32).cpu(), "position_ids_f32": position_ids.to(torch.float32).cpu(),
        "indicator": indicator.to(torch.int32).cpu(), "indicator_f32": indicator.to(torch.float32).cpu(),
        "segment_ids": segment_ids.to(torch.int32).cpu(), "segment_ids_f32": segment_ids.to(torch.float32).cpu(),
        "velocity": velocity.float().cpu(),
    }
    fx = {k: v.contiguous() for k, v in fx.items()}
    save_file(fx, f"{OUT}/ideogram4_fx_predict.safetensors")
    json.dump({"res": RES, "gh": GH, "gw": GW, "Lt": int(LT), "seq": int(seq_len), "t": 0.7,
               "image": IMG, "caption": CAP, "velocity_std": float(velocity.std()), "target_std": float(target.std())},
              open(f"{OUT}/ideogram4_fx_predict_meta.json", "w"), indent=2)
    print(f"[predict] saved real-giger fixture: seq={seq_len} (Lt={LT}+img={num_image}) "
          f"vel{tuple(velocity.shape)} vel_std={velocity.std():.4f} target_std={target.std():.4f}")


if __name__ == "__main__":
    main()

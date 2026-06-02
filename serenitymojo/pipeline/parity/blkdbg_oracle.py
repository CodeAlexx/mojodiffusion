#!/usr/bin/env python3
# blkdbg_oracle.py — block-by-block parity oracle for the Z-Image NextDiT,
# GPU bf16, REAL inputs (lat_step_27, cond, t=0.82353). Adapts the fp32-CPU
# random gen_oracle.py to the actual divergence test point. Dumps each named
# intermediate to parity/blkdbg/<name>.bin (+ .shape), mirroring the Mojo
# debug_stage() boundaries so a Mojo driver can cos-compare per stage.
#
# Run with the CUDA venv:
#   /home/alex/serenityflow-v2/.venv/bin/python blkdbg_oracle.py
#
# bf16 on cuda (matches the Mojo BF16-storage / F32-accum path). Loads the
# transformer ALONE (12.3 GB) on the 24 GB 3090Ti, wrapped in no_grad.
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch

from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "blkdbg")
os.makedirs(OUT_DIR, exist_ok=True)

XFMR_DIR = (
    "/home/alex/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image/"
    "snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021/transformer"
)

# Real divergence test point.
T_VAL = 0.82353
HL = WL = 32
CAPLEN = 173
DEV = "cuda"
DT = torch.bfloat16


def load_bin(name, shape):
    arr = np.fromfile(os.path.join(HERE, name + ".bin"), dtype="<f4").reshape(shape)
    return torch.from_numpy(arr)


def dump(name, t):
    if isinstance(t, (list, tuple)):
        t = t[0]
    arr = t.detach().to(torch.float32).contiguous().cpu().numpy().ravel()
    arr.astype("<f4").tofile(os.path.join(OUT_DIR, name + ".bin"))
    with open(os.path.join(OUT_DIR, name + ".shape"), "w") as f:
        f.write(",".join(str(d) for d in t.shape))
    print(f"  dumped {name:34s} shape={tuple(t.shape)} "
          f"mean={float(t.float().mean()):+.5f} std={float(t.float().std()):+.5f}")


def main():
    model = ZImageTransformer2DModel.from_pretrained(XFMR_DIR, torch_dtype=DT).to(DEV)
    model.eval()
    cfg = model.config
    print(f"dim={cfg.dim} n_heads={cfg.n_heads} n_layers={cfg.n_layers} "
          f"n_refiner={cfg.n_refiner_layers} cap_feat_dim={cfg.cap_feat_dim} "
          f"rope_theta={cfg.rope_theta} t_scale={cfg.t_scale} "
          f"axes_dims={cfg.axes_dims} axes_lens={cfg.axes_lens} eps={cfg.norm_eps}")

    # Real inputs.
    img = load_bin("lat_step_27", (1, 16, HL, WL))[0].unsqueeze(1).to(DEV, DT)  # (C=16,F=1,H,W)
    cap = load_bin("cond", (CAPLEN, 2560)).to(DEV, DT)
    t = torch.tensor([T_VAL], dtype=DT, device=DEV)

    patch_size, f_patch_size = 2, 1
    captures = {}

    with torch.no_grad():
        # adaln_input = t_embedder(t * t_scale)
        adaln_input = model.t_embedder(t * cfg.t_scale).type_as(img)
        captures["t_emb"] = adaln_input.clone()

        # Patchify + embed (single image).
        (x_p, cap_p, x_size, x_pos_ids, cap_pos_ids, x_pad_mask, cap_pad_mask) = \
            model.patchify_and_embed([img], [cap], patch_size, f_patch_size)
        x_seqlens = [len(xi) for xi in x_p]
        x = model.all_x_embedder[f"{patch_size}-{f_patch_size}"](torch.cat(x_p, dim=0))
        captures["x_after_embedder"] = x.clone()  # [img_padded_len, dim]

        x, x_freqs, x_mask, _, x_noise_tensor = model._prepare_sequence(
            list(x.split(x_seqlens, dim=0)), x_pos_ids, x_pad_mask,
            model.x_pad_token, None, img.device,
        )
        captures["x_after_prepare"] = x.clone()   # [1, img_padded_len, dim]

        # Noise refiner.0 sub-step localization (modulated attention branch).
        nr0 = model.noise_refiner[0]
        mod0 = nr0.adaLN_modulation(adaln_input)
        sm, gm, smlp, gmlp = mod0.unsqueeze(1).chunk(4, dim=2)
        captures["nr0_scale_msa"] = (1.0 + sm).clone()
        captures["nr0_gate_msa"] = gm.tanh().clone()
        captures["nr0_scale_mlp"] = (1.0 + smlp).clone()
        captures["nr0_gate_mlp"] = gmlp.tanh().clone()
        scale_msa = 1.0 + sm
        nr0_n1 = nr0.attention_norm1(x)
        captures["nr0_norm1"] = nr0_n1.clone()
        nr0_n1s = nr0_n1 * scale_msa
        captures["nr0_norm1_scaled"] = nr0_n1s.clone()
        nr0_attn = nr0.attention(nr0_n1s, attention_mask=x_mask, freqs_cis=x_freqs)
        captures["nr0_attn_out"] = nr0_attn.clone()

        img_tokens = (HL // 2) * (WL // 2)
        for li, layer in enumerate(model.noise_refiner):
            x = layer(x, x_mask, x_freqs, adaln_input, None, None, None)
            captures[f"x_after_noise_refiner_{li}"] = x.clone()
            if li == 0:
                captures["x_after_noise_refiner_0_real"] = x[:, :img_tokens, :].clone()

        # Cap embed + refine.
        cap_seqlens = [len(ci) for ci in cap_p]
        capf = model.cap_embedder(torch.cat(cap_p, dim=0))
        captures["cap_after_embedder"] = capf.clone()
        capf, cap_freqs, cap_mask, _, _ = model._prepare_sequence(
            list(capf.split(cap_seqlens, dim=0)), cap_pos_ids, cap_pad_mask,
            model.cap_pad_token, None, img.device,
        )
        captures["cap_after_prepare"] = capf.clone()
        for li, layer in enumerate(model.context_refiner):
            capf = layer(capf, cap_mask, cap_freqs)
            captures[f"cap_after_context_refiner_{li}"] = capf.clone()

        # Unified [x, cap] (basic mode order).
        unified, unified_freqs, unified_mask, unified_noise_tensor = \
            model._build_unified_sequence(
                x, x_freqs, x_seqlens, None,
                capf, cap_freqs, cap_seqlens, None,
                None, None, None, None,
                False, img.device,
            )
        captures["unified_initial"] = unified.clone()

        # Main layers: capture after each layer for fine-grained localization.
        for li, layer in enumerate(model.layers):
            unified = layer(unified, unified_mask, unified_freqs, adaln_input,
                            unified_noise_tensor, None, None)
            captures[f"unified_after_layer_{li}"] = unified.clone()
        captures["unified_after_main"] = unified.clone()

        # Final layer + sub-step localization.
        fl = model.all_final_layer[f"{patch_size}-{f_patch_size}"]
        fl_scale = 1.0 + fl.adaLN_modulation(adaln_input)   # [1, dim]
        captures["fl_scale"] = fl_scale.unsqueeze(1).clone()  # [1,1,dim]
        fl_norm = fl.norm_final(unified)                    # LayerNorm no-affine
        captures["fl_norm"] = fl_norm.clone()               # [1, S, dim]
        fl_scaled = fl_norm * fl_scale.unsqueeze(1)
        captures["fl_scaled"] = fl_scaled.clone()           # [1, S, dim]
        final = model.all_final_layer[f"{patch_size}-{f_patch_size}"](unified, c=adaln_input)
        captures["after_final_layer"] = final.clone()

        # Unpatchify -> (C, F, H, W); take [0].
        out = model.unpatchify(list(final.unbind(dim=0)), x_size, patch_size, f_patch_size, None)
        out0 = out[0]
        captures["out"] = out0.clone()

        # Cross-check vs the public forward (the same call vf_4.bin used).
        full = model(x=[img], t=t, cap_feats=[cap], return_dict=True).sample[0]
        d = float((full.float() - out0.float()).abs().max())
        print(f"  manual-walk vs forward() max_abs_diff = {d:.3e}")

        # Sanity vs vf_4.bin (the established final reference).
        vf4 = load_bin("vf_4", (16, 1, 32, 32)).to(DEV, torch.float32).ravel()
        of = out0.float().ravel()
        cos = float(torch.dot(of, vf4) / (of.norm() * vf4.norm()))
        print(f"  oracle out vs vf_4.bin cos = {cos:.6f}")

    for k, v in captures.items():
        dump(k, v)

    img_tokens = (HL // 2) * (WL // 2)
    cap_pad = ((-CAPLEN) % 32)
    cap_padded = CAPLEN + cap_pad
    with open(os.path.join(OUT_DIR, "dims.txt"), "w") as f:
        f.write(f"HL={HL}\nWL={WL}\nCAPLEN={CAPLEN}\n")
        f.write(f"img_tokens={img_tokens}\nimg_padded={img_tokens}\n")
        f.write(f"cap_padded={cap_padded}\nunified={img_tokens + cap_padded}\n")
    print(f"  dims: img_tokens={img_tokens} cap_padded={cap_padded} "
          f"unified={img_tokens + cap_padded}")
    print("blkdbg oracle dump complete ->", OUT_DIR)


if __name__ == "__main__":
    main()

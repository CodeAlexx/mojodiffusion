#!/usr/bin/env python3
# serenitymojo/models/anima/parity/anima_ot_step_oracle.py
#
# OneTrainer-FAITHFUL recipe oracle for the ANIMA LoRA training STEP (Chunk A gate).
#
# WHAT THIS PROVES (the Chunk A recipe delta — NOT the transformer, already green):
#   predicted_flow + MSE-loss computed by the OT recipe
#     1. scale_latents:  scaled = (latent - mean) * (1/std)   (per-channel, 16 ch)
#                        AnimaModel.py:233-236; consts from the Qwen-Image VAE
#                        config.json latents_mean/latents_std.
#     2. discrete timestep -> sigma = (timestep_index + 1) / num_train_timesteps
#                        ModelSetupFlowMatchingMixin._add_noise_discrete:23-29
#                        (all_timesteps = arange(1,N+1); sigma = all/N; sigma[ts]).
#     3. noisy  = noise * sigma + scaled * (1 - sigma)
#        target = noise - scaled
#                        ModelSetupFlowMatchingMixin.py:36-37; BaseAnimaSetup.py:143.
#     4. predicted_flow = transformer(hidden=noisy, timestep=timestep/1000,
#                                     encoder=context[1,512,1024], padding_mask)
#                        BaseAnimaSetup.py:135-141 — NOTE timestep/1000 into the
#                        sinusoidal t-embedder (the old trainer passed raw sigma).
#     5. loss = mean((predicted_flow - target)^2)   (unmasked MSE)
#                        BaseAnimaSetup.calculate_loss -> _flow_matching_losses ->
#                        __unmasked_losses F.mse_loss reduction='none'.mean.
#
# ORACLE CHOICE (gate-zero feasibility, see report):
#   We do NOT use the installed diffusers CosmosTransformer3DModel. Reasons:
#     * No diffusers config.json exists on disk for Anima (only the original net.*
#       split_files checkpoint); the real model needs the non-default Anima config
#       (16 heads not 32, adaln_lora_dim, crossattn projection, img-context dims).
#     * The pure-Mojo Anima step uses the SAME triangulated transformer math that is
#       already parity-gated cos>=0.99999999 vs stack_oracle.py (simplified single-
#       axis RoPE table, GELU tanh-approx, LayerNorm-no-affine AdaLN, half-split RoPE).
#       The real CosmosTransformer3DModel uses TRUE 3D RoPE + learnable extra-pos-embed
#       + padding-mask-concat patchify — feeding its predicted_flow would diff against
#       the Mojo's (already-green) simplified transformer, conflating the proven
#       transformer with the recipe-under-test.
#   So this oracle replicates the Mojo transformer math EXACTLY (the stack_oracle.py
#   lineage) but loads the REAL Anima checkpoint weights (block 0..L-1 + base + the
#   t_embedder), and drives the OT recipe end-to-end. Parity then isolates the recipe.
#   This is the spec's explicit "FALL BACK to stack_oracle" branch.
#
# Run (SEPARATE command; system python3 has torch + safetensors):
#   cd /home/alex/mojodiffusion
#   python3 serenitymojo/models/anima/parity/anima_ot_step_oracle.py
#
# Emits FIXED .bin inputs the Mojo gate reads + reference predicted_flow + loss.

import math
import struct
import os
import torch
from safetensors import safe_open

DT = torch.float64

# ── real Anima dims (anima_contract.mojo) ──
B = 1
H = 16
Dh = 128
D = H * Dh            # 2048
JOINT = 1024
Fmlp = 8192           # real MLP hidden
ADALN = 256
C = 16                # latent channels
PS = 2
IN_PATCH = (C + 1) * PS * PS   # 68
OUT_PATCH = C * PS * PS        # 64
EPS = 1e-6
SCALE = 1.0 / math.sqrt(Dh)
THETA = 10000.0

# Chunk-A recipe constants
S_TXT = 512                    # AnimaModel.PROMPT_MAX_LENGTH=512 (delta 1)
LATENT_HW = 16                 # crop latent grid -> S_IMG = (16/2)^2 = 64 (thermal-safe)
S_IMG = (LATENT_HW // PS) * (LATENT_HW // PS)   # 64
NUM_TRAIN_TIMESTEPS = 1000     # FlowMatchEulerDiscreteScheduler default
FIXED_TIMESTEP = 500           # fixed discrete index for the gate (non-degenerate)
L = 2                          # gate depth: real blocks 0..L-1 (recipe gate, not depth)

CKPT = "/home/alex/.serenity/models/anima/split_files/diffusion_models/anima-base-v1.0.safetensors"
VAE_CFG = "/home/alex/.serenity/models/checkpoints/qwen-image-2512/vae/config.json"

REF_DIR = os.path.dirname(os.path.abspath(__file__))


def W(name, tensor):
    flat = tensor.detach().reshape(-1).to(torch.float32).numpy()
    with open(os.path.join(REF_DIR, name + ".bin"), "wb") as f:
        f.write(struct.pack("<%df" % flat.size, *flat.tolist()))
    print("wrote", name, tuple(tensor.shape))


# ── VAE scale_latents constants (delta 2) — read from the Qwen-Image VAE config ──
def read_vae_consts():
    import json
    with open(VAE_CFG) as f:
        cfg = json.load(f)
    mean = torch.tensor(cfg["latents_mean"], dtype=DT)
    std = torch.tensor(cfg["latents_std"], dtype=DT)
    assert mean.numel() == C and std.numel() == C, "VAE consts must be 16-ch"
    return mean, std


# ── ops (match anima.rs / stack_oracle.py — the proven Mojo transformer math) ──
def layer_norm_noaffine(x):
    mean = x.mean(-1, keepdim=True)
    var = x.var(-1, unbiased=False, keepdim=True)
    return (x - mean) / torch.sqrt(var + EPS)


def rms_norm_lastdim(x, weight):
    ms = x.pow(2).mean(-1, keepdim=True)
    return x / torch.sqrt(ms + EPS) * weight


def rms_norm_vec(x, weight):
    # RMSNorm over last dim for a [.., D] vector (t_embedding_norm).
    ms = x.pow(2).mean(-1, keepdim=True)
    return x / torch.sqrt(ms + EPS) * weight


def rope_halfsplit(x, cos, sin):
    half = Dh // 2
    c = cos.reshape(1, x.shape[1], 1, half)
    s = sin.reshape(1, x.shape[1], 1, half)
    x1 = x[..., :half]
    x2 = x[..., half:]
    o1 = x1 * c - x2 * s
    o2 = x2 * c + x1 * s
    return torch.cat([o1, o2], dim=-1)


def build_3axis_rope(s_img):
    """REAL Anima 3-axis (T,H,W) NTK rope table -> (cos, sin) each [S_IMG, Dh/2].

    Mirrors anima_dit.build_anima_3d_rope (and diffusers CosmosRotaryPosEmbed):
    head_dim split into [dim_t | dim_h | dim_w] = [44 | 42 | 42] bands, per-axis
    NTK-scaled theta from rope_scale (t=1.0, h=4.0, w=4.0). Each token's angle is
    indexed by its (T, ih, iw) grid coordinate, NOT a flat linear position.
    Column order [t-bins | h-bins | w-bins] matches the Mojo trainer table and the
    half-split consumer (column-i = angle for pair i).
    """
    half = Dh // 2            # 64
    full_d = Dh              # 128
    nh = LATENT_HW // PS     # 8
    nw = LATENT_HW // PS     # 8
    t_frames = 1
    assert nh * nw == s_img, f"rope grid mismatch nh*nw={nh*nw} != S_IMG={s_img}"

    dim_h = full_d // 6 * 2   # 42
    dim_w = dim_h             # 42
    dim_t = full_d - 2 * dim_h  # 44
    bins_t = dim_t // 2       # 22
    bins_h = dim_h // 2       # 21
    bins_w = dim_w // 2       # 21
    assert bins_t + bins_h + bins_w == half

    base_theta = 10000.0
    h_ntk = 4.0 ** (dim_h / (dim_h - 2.0))
    w_ntk = 4.0 ** (dim_w / (dim_w - 2.0))
    t_ntk = 1.0
    theta_t = base_theta * t_ntk
    theta_h = base_theta * h_ntk
    theta_w = base_theta * w_ntk
    freqs_t = [1.0 / (theta_t ** (2.0 * i / dim_t)) for i in range(bins_t)]
    freqs_h = [1.0 / (theta_h ** (2.0 * i / dim_h)) for i in range(bins_h)]
    freqs_w = [1.0 / (theta_w ** (2.0 * i / dim_w)) for i in range(bins_w)]

    cos = torch.zeros(s_img, half, dtype=DT)
    sin = torch.zeros(s_img, half, dtype=DT)
    for tf in range(t_frames):
        for ih in range(nh):
            for iw in range(nw):
                s = (tf * nh + ih) * nw + iw
                col = 0
                for fi in range(bins_t):
                    a = tf * freqs_t[fi]
                    cos[s, col] = math.cos(a); sin[s, col] = math.sin(a); col += 1
                for fi in range(bins_h):
                    a = ih * freqs_h[fi]
                    cos[s, col] = math.cos(a); sin[s, col] = math.sin(a); col += 1
                for fi in range(bins_w):
                    a = iw * freqs_w[fi]
                    cos[s, col] = math.cos(a); sin[s, col] = math.sin(a); col += 1

    # ── NON-DEGENERACY guard: the table must NOT collapse to the old single-axis
    # (or any aliased) form. (a) distinct cos rows across positions — a degenerate
    # table would have many identical/aliased rows; (b) sin reaches a real rotation
    # (the old token-9 ih=iw=1 case had real spatial rotation; a degenerate t-only
    # table would leave the h/w bands at sin=0).
    distinct = len({tuple(round(float(v), 6) for v in cos[s]) for s in range(s_img)})
    assert distinct >= s_img - 1, (
        f"rope table degenerate: only {distinct} distinct cos rows for {s_img} positions"
    )
    # h/w spatial bands must actually rotate (token (ih=1,iw=1) -> nonzero sin in h&w):
    s11 = (0 * nh + 1) * nw + 1
    sin_h_band = sin[s11, bins_t:bins_t + bins_h].abs().max().item()
    sin_w_band = sin[s11, bins_t + bins_h:].abs().max().item()
    assert sin_h_band > 1e-3 and sin_w_band > 1e-3, (
        f"rope spatial bands not rotating: |sin_h|max={sin_h_band:.3e} "
        f"|sin_w|max={sin_w_band:.3e} (degenerate/aliased table)"
    )
    return cos, sin


def sdpa(q, k, v):
    qh = q.permute(0, 2, 1, 3)
    kh = k.permute(0, 2, 1, 3)
    vh = v.permute(0, 2, 1, 3)
    scores = (qh @ kh.transpose(-1, -2)) * SCALE
    attn = torch.softmax(scores, dim=-1)
    out = attn @ vh
    return out.permute(0, 2, 1, 3)


def gelu_tanh(x):
    return torch.nn.functional.gelu(x, approximate="tanh")


def adaln_mod(w, sub, t_silu, base_adaln):
    h = t_silu @ w[f"{sub}_mod1"].T
    mod_out = h @ w[f"{sub}_mod2"].T + base_adaln
    shift = mod_out[:, 0:D]
    scale = mod_out[:, D:2 * D]
    gate = mod_out[:, 2 * D:3 * D]
    return shift, scale, gate


def adaln_pre(xx, shift, scale):
    ln = layer_norm_noaffine(xx)
    return (1.0 + scale).unsqueeze(1) * ln + shift.unsqueeze(1)


def block_forward(x, w, t_silu, base_adaln, context, cos, sin):
    # self-attn
    sh, sc, ga = adaln_mod(w, "sa", t_silu, base_adaln)
    xmod = adaln_pre(x, sh, sc).reshape(B * S_IMG, D)
    q = (xmod @ w["sa_q"].T).reshape(B, S_IMG, H, Dh)
    k = (xmod @ w["sa_k"].T).reshape(B, S_IMG, H, Dh)
    v = (xmod @ w["sa_v"].T).reshape(B, S_IMG, H, Dh)
    q = rms_norm_lastdim(q, w["sa_qn"])
    k = rms_norm_lastdim(k, w["sa_kn"])
    q = rope_halfsplit(q, cos, sin)
    k = rope_halfsplit(k, cos, sin)
    att = sdpa(q, k, v).reshape(B * S_IMG, D)
    sa_out = (att @ w["sa_out"].T).reshape(B, S_IMG, D)
    x = x + ga.unsqueeze(1) * sa_out
    # cross-attn (k/v from frozen context, no RoPE)
    sh, sc, ga = adaln_mod(w, "ca", t_silu, base_adaln)
    xmod = adaln_pre(x, sh, sc).reshape(B * S_IMG, D)
    ctx2d = context.reshape(B * S_TXT, JOINT)
    q = (xmod @ w["ca_q"].T).reshape(B, S_IMG, H, Dh)
    k = (ctx2d @ w["ca_k"].T).reshape(B, S_TXT, H, Dh)
    v = (ctx2d @ w["ca_v"].T).reshape(B, S_TXT, H, Dh)
    q = rms_norm_lastdim(q, w["ca_qn"])
    k = rms_norm_lastdim(k, w["ca_kn"])
    att = sdpa(q, k, v).reshape(B * S_IMG, D)
    ca_out = (att @ w["ca_out"].T).reshape(B, S_IMG, D)
    x = x + ga.unsqueeze(1) * ca_out
    # mlp
    sh, sc, ga = adaln_mod(w, "mlp", t_silu, base_adaln)
    xmod = adaln_pre(x, sh, sc).reshape(B * S_IMG, D)
    h = gelu_tanh(xmod @ w["mlp1"].T)
    mlp_out = (h @ w["mlp2"].T).reshape(B, S_IMG, D)
    x = x + ga.unsqueeze(1) * mlp_out
    return x


# ── real-weight loader (net.* layout) ──
def load_real():
    t = {}
    base = {}
    with safe_open(CKPT, "pt") as f:
        def g(name):
            return f.get_tensor(name).to(DT)
        base["x_embed"] = g("net.x_embedder.proj.1.weight")          # [2048,68]
        base["te_lin1"] = g("net.t_embedder.1.linear_1.weight")      # [2048,2048]
        base["te_lin2"] = g("net.t_embedder.1.linear_2.weight")      # [6144,2048]
        base["t_norm"] = g("net.t_embedding_norm.weight")            # [2048]
        base["fl_mod1"] = g("net.final_layer.adaln_modulation.1.weight")  # [256,2048]
        base["fl_mod2"] = g("net.final_layer.adaln_modulation.2.weight")  # [4096,256]
        base["fl_lin"] = g("net.final_layer.linear.weight")          # [64,2048]
        blk = []
        for bi in range(L):
            bp = f"net.blocks.{bi}."
            w = {}
            w["sa_mod1"] = g(bp + "adaln_modulation_self_attn.1.weight")
            w["sa_mod2"] = g(bp + "adaln_modulation_self_attn.2.weight")
            w["ca_mod1"] = g(bp + "adaln_modulation_cross_attn.1.weight")
            w["ca_mod2"] = g(bp + "adaln_modulation_cross_attn.2.weight")
            w["mlp_mod1"] = g(bp + "adaln_modulation_mlp.1.weight")
            w["mlp_mod2"] = g(bp + "adaln_modulation_mlp.2.weight")
            w["sa_q"] = g(bp + "self_attn.q_proj.weight")
            w["sa_k"] = g(bp + "self_attn.k_proj.weight")
            w["sa_v"] = g(bp + "self_attn.v_proj.weight")
            w["sa_out"] = g(bp + "self_attn.output_proj.weight")
            w["sa_qn"] = g(bp + "self_attn.q_norm.weight")
            w["sa_kn"] = g(bp + "self_attn.k_norm.weight")
            w["ca_q"] = g(bp + "cross_attn.q_proj.weight")
            w["ca_k"] = g(bp + "cross_attn.k_proj.weight")
            w["ca_v"] = g(bp + "cross_attn.v_proj.weight")
            w["ca_out"] = g(bp + "cross_attn.output_proj.weight")
            w["ca_qn"] = g(bp + "cross_attn.q_norm.weight")
            w["ca_kn"] = g(bp + "cross_attn.k_norm.weight")
            w["mlp1"] = g(bp + "mlp.layer1.weight")
            w["mlp2"] = g(bp + "mlp.layer2.weight")
            blk.append(w)
    return base, blk


# ── sinusoidal t-embedder (anima_dit _anima_sinusoidal, cos-first) ──
def sinusoidal(val, dim):
    half = dim // 2
    neg_ln = -math.log(10000.0)
    out = torch.zeros(dim, dtype=DT)
    for i in range(half):
        freq = math.exp(neg_ln * (i / half))
        angle = val * freq
        out[i] = math.cos(angle)
        out[half + i] = math.sin(angle)
    return out


# ── deterministic host gaussian noise (MUST match train_anima_real._host_noise) ──
def host_noise(n, seed):
    out = []
    state = seed & ((1 << 64) - 1)
    i = 0
    M = (1 << 64) - 1
    while i < n:
        state = (state * 6364136223846793005 + 1442695040888963407) & M
        u1 = float((state >> 11) & 0xFFFFFFFFFFFFF) * (1.0 / 4503599627370496.0)
        state = (state * 6364136223846793005 + 1442695040888963407) & M
        u2 = float((state >> 11) & 0xFFFFFFFFFFFFF) * (1.0 / 4503599627370496.0)
        if u1 < 1.0e-12:
            u1 = 1.0e-12
        r = math.sqrt(-2.0 * math.log(u1))
        theta = 6.283185307179586 * u2
        out.append(r * math.cos(theta))
        if i + 1 < n:
            out.append(r * math.sin(theta))
        i += 2
    return torch.tensor(out[:n], dtype=DT)


# ── patchify (match train_anima_real _patchify_in / _patchify_out) ──
def patchify_in(x_bthwc):  # [B,1,Hd,Wd,C] -> [B*N, 68], mask channel = 0
    Hd = Wd = LATENT_HW
    nH = Hd // PS
    nW = Wd // PS
    Cp = C + 1
    N = nH * nW
    out = torch.zeros(B * N * (Cp * PS * PS), dtype=DT)
    for b in range(B):
        for ih in range(nH):
            for iw in range(nW):
                pn = ih * nW + iw
                for c in range(Cp):
                    for ph in range(PS):
                        for pw in range(PS):
                            od = (b * N + pn) * (Cp * PS * PS) + (c * PS * PS + ph * PS + pw)
                            if c < C:
                                out[od] = x_bthwc[b, 0, ih * PS + ph, iw * PS + pw, c]
    return out.reshape(B * N, Cp * PS * PS)


def patchify_out(x_bthwc):  # [B,1,Hd,Wd,C] -> [B*N, 64], C fastest
    Hd = Wd = LATENT_HW
    nH = Hd // PS
    nW = Wd // PS
    N = nH * nW
    out = torch.zeros(B * N * (C * PS * PS), dtype=DT)
    for b in range(B):
        for ih in range(nH):
            for iw in range(nW):
                pn = ih * nW + iw
                for ph in range(PS):
                    for pw in range(PS):
                        for c in range(C):
                            od = (b * N + pn) * (C * PS * PS) + (ph * PS * C + pw * C + c)
                            out[od] = x_bthwc[b, 0, ih * PS + ph, iw * PS + pw, c]
    return out.reshape(B * N, C * PS * PS)


def main():
    torch.manual_seed(0)
    mean, std = read_vae_consts()
    base, blk = load_real()

    # ── FIXED non-degenerate raw latent [B,1,Hd,Wd,C] (sinusoidal so it's structured) ──
    Hd = Wd = LATENT_HW
    raw = torch.zeros(B, 1, Hd, Wd, C, dtype=DT)
    for h in range(Hd):
        for w in range(Wd):
            for c in range(C):
                raw[0, 0, h, w, c] = math.sin(0.13 * h + 0.21 * w + 0.07 * c) * 1.5 + 0.3 * c - 2.0

    # delta 2: scale_latents per-channel (raw - mean)*(1/std)
    scaled = (raw - mean.reshape(1, 1, 1, 1, C)) * (1.0 / std.reshape(1, 1, 1, 1, C))

    # delta 3: discrete sigma = (timestep_index + 1)/N ; noisy / target
    sigma = float((FIXED_TIMESTEP + 1)) / float(NUM_TRAIN_TIMESTEPS)
    n_lat = B * 1 * Hd * Wd * C
    noise_flat = host_noise(n_lat, 104729 * 42 & ((1 << 64) - 1))   # SEED*104729 (smoke seed=42)
    noise = noise_flat.reshape(B, 1, Hd, Wd, C)
    noisy = noise * sigma + scaled * (1.0 - sigma)
    target = noise - scaled                                         # BaseAnimaSetup.py:143

    # delta 4: timestep/1000 into the sinusoidal embedder (NOT raw sigma)
    t_in = float(FIXED_TIMESTEP) / 1000.0
    emb = sinusoidal(t_in, D).reshape(1, D)
    hidden = torch.nn.functional.silu(emb @ base["te_lin1"].T)
    base_adaln = hidden @ base["te_lin2"].T                          # [1,6144]
    t_cond = rms_norm_vec(emb, base["t_norm"])                       # RAW (un-silu'd)
    t_silu = torch.nn.functional.silu(t_cond)                        # block/final silu internally

    # frozen context [1,512,1024] (fixed structured)
    context = torch.zeros(B, S_TXT, JOINT, dtype=DT)
    for s in range(S_TXT):
        for j in range(JOINT):
            context[0, s, j] = math.sin(0.011 * s + 0.017 * j) * 0.5

    # rope tables: REAL 3-axis (T,H,W) NTK rope — matches Mojo _rope_tables AND
    # anima_dit.build_anima_3d_rope / diffusers CosmosRotaryPosEmbed. The previous
    # single-axis table (theta=10000 over the flat token index) was OT-UNFAITHFUL
    # (cos≈0.71 vs the real 3-axis table) and made the recipe gate a shared-error
    # tautology on the positional axis. Each token's rotation depends on its
    # (T, ih, iw) grid coordinate; rope_scale (t=1.0, h=4.0, w=4.0).
    cos, sin = build_3axis_rope(S_IMG)

    # patchify noisy -> input patches; build x via patch-embed
    patches = patchify_in(noisy)                                    # [B*N, 68]
    target_patches = patchify_out(target)                           # [B*N, 64]
    x = (patches @ base["x_embed"].T).reshape(B, S_IMG, D)

    for l in range(L):
        x = block_forward(x, blk[l], t_silu, base_adaln, context, cos, sin)

    # final layer
    fl_h = t_silu @ base["fl_mod1"].T
    fl_modout = fl_h @ base["fl_mod2"].T + base_adaln[:, :2 * D]
    fl_shift = fl_modout[:, 0:D]
    fl_scale = fl_modout[:, D:2 * D]
    x_mod = adaln_pre(x, fl_shift, fl_scale)
    pred = (x_mod.reshape(B * S_IMG, D) @ base["fl_lin"].T)         # [B*N, 64]

    # delta 5: unmasked MSE over the output-patch layout target
    loss = ((pred - target_patches) ** 2).mean()

    # ── emit FIXED inputs the Mojo gate reads (so both consume IDENTICAL data) ──
    W("ot_scaled_bthwc", scaled.reshape(-1))      # [B,1,Hd,Wd,C] flat
    W("ot_noise_bthwc", noise.reshape(-1))
    W("ot_context", context.reshape(-1))          # [B,512,1024]
    # references
    W("ot_pred", pred)                            # [B*N,64] predicted_flow (patch layout)
    W("ot_target_patches", target_patches)        # [B*N,64]
    with open(os.path.join(REF_DIR, "ot_loss.bin"), "wb") as f:
        f.write(struct.pack("<f", float(loss)))
    with open(os.path.join(REF_DIR, "ot_meta.txt"), "w") as f:
        f.write(f"sigma={sigma}\nt_in={t_in}\nL={L}\nS_IMG={S_IMG}\nS_TXT={S_TXT}\n")
        f.write(f"FIXED_TIMESTEP={FIXED_TIMESTEP}\nNUM_TRAIN_TIMESTEPS={NUM_TRAIN_TIMESTEPS}\n")

    print("=== Anima OT-recipe step oracle ===")
    print(f"sigma=(ts+1)/N = ({FIXED_TIMESTEP}+1)/{NUM_TRAIN_TIMESTEPS} = {sigma}")
    print(f"timestep/1000 into embedder = {t_in}")
    print(f"L={L}  S_IMG={S_IMG}  S_TXT={S_TXT}")
    print(f"scaled latent: mean={float(scaled.mean()):.5f} std={float(scaled.std()):.5f}")
    print(f"pred norm={float(pred.norm()):.6f}  loss={float(loss):.8f}")
    print("DONE")


if __name__ == "__main__":
    main()

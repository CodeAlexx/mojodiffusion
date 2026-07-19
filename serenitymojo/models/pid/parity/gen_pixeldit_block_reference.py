#!/usr/bin/env python3
# gen_pixeldit_block_reference.py — DEV-ONLY PyTorch oracle for the PiD
# PixelDiT MMDiT joint-attention block parity smoke
# (serenitymojo/models/pid/pixeldit_block_smoke.mojo).
#
# NOT in the runtime path. Run with SYSTEM python3 (the pixi env has no torch):
#   python3 serenitymojo/models/pid/parity/gen_pixeldit_block_reference.py > \
#       serenitymojo/models/pid/parity/pixeldit_block_ref_data.mojo
#
# Mirrors the EXACT classes from the PiD repo (/tmp/PiD_repo), verbatim:
#   pixeldit_official.py:
#     - RMSNorm                                 (lines 111-122)
#     - FeedForward (SwiGLU)                    (lines 125-135)
#     - precompute_freqs_cis_2d_ntk             (lines 154-193)
#     - apply_rotary_emb                        (lines 196-206)
#     - MMDiTJointAttention                     (lines 517-624)
#     - MMDiTBlockT2I                           (lines 627-682)
#
# Random SEEDED weights, tiny inputs. F32 throughout so the Mojo gate is
# bit-close (cos >= 0.999). No Python at runtime — the Mojo smoke inlines the
# numbers printed here.

import math
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention

SEED = 5151
torch.manual_seed(SEED)
np.random.seed(SEED)


def emit_scalar_i(name, v):
    print(f"comptime {name} = {int(v)}")


def emit_scalar_f(name, v):
    print(f"comptime {name} = Float32({float(v):.8f})")


def emit_list(name, arr):
    a = np.asarray(arr, dtype=np.float64).reshape(-1)
    print(f"def {name}() -> List[Float32]:")
    print("    var v = List[Float32]()")
    for x in a:
        print(f"    v.append(Float32({float(x):.8f}))")
    print("    return v^")
    print()


# ============================================================================
# Verbatim PiD repo classes (pixeldit_official.py).
# ============================================================================
def apply_adaln(x, shift, scale):
    return x * (1 + scale) + shift


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        x = self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))
        return x


def precompute_freqs_cis_2d_ntk(dim, height, width, ref_grid_h, ref_grid_w,
                                theta=10000.0, scale=16.0):
    dim_axis = dim // 2
    h_scale = height / ref_grid_h
    w_scale = width / ref_grid_w
    h_ntk = h_scale ** (dim_axis / (dim_axis - 2)) if dim_axis > 2 else 1.0
    w_ntk = w_scale ** (dim_axis / (dim_axis - 2)) if dim_axis > 2 else 1.0
    h_theta = theta * h_ntk
    w_theta = theta * w_ntk
    x_pos = torch.linspace(0, scale, width)
    y_pos = torch.linspace(0, scale, height)
    y_pos, x_pos = torch.meshgrid(y_pos, x_pos, indexing="ij")
    y_pos = y_pos.reshape(-1)
    x_pos = x_pos.reshape(-1)
    freqs_w = 1.0 / (w_theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_h = 1.0 / (h_theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    x_freqs = torch.outer(x_pos, freqs_w).float()
    y_freqs = torch.outer(y_pos, freqs_h).float()
    x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
    y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
    freqs_cis = torch.cat([x_cis.unsqueeze(-1), y_cis.unsqueeze(-1)], dim=-1)
    freqs_cis = freqs_cis.reshape(height * width, -1)
    return freqs_cis


def apply_rotary_emb(xq, xk, freqs_cis):
    freqs_cis = freqs_cis[None, :, None, :]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class MMDiTJointAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv_x = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_y = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm_x = RMSNorm(self.head_dim)
        self.k_norm_x = RMSNorm(self.head_dim)
        self.q_norm_y = RMSNorm(self.head_dim)
        self.k_norm_y = RMSNorm(self.head_dim)
        self.proj_x = nn.Linear(dim, dim)
        self.proj_y = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop_x = nn.Dropout(proj_drop)
        self.proj_drop_y = nn.Dropout(proj_drop)
        self._cp_group = None

    def forward(self, x, y, pos_img, pos_txt=None, attn_mask=None):
        B, Nx, C = x.shape
        By, Ny, Cy = y.shape
        qkv_x = self.qkv_x(x).reshape(B, Nx, 3, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)
        qx, kx, vx = qkv_x[0], qkv_x[1], qkv_x[2]
        qx = self.q_norm_x(qx)
        kx = self.k_norm_x(kx)
        qkv_y = self.qkv_y(y).reshape(B, Ny, 3, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)
        qy, ky, vy = qkv_y[0], qkv_y[1], qkv_y[2]
        qy = self.q_norm_y(qy)
        ky = self.k_norm_y(ky)
        qx, kx = apply_rotary_emb(qx, kx, freqs_cis=pos_img)
        if pos_txt is not None:
            qy, ky = apply_rotary_emb(qy, ky, freqs_cis=pos_txt)
        qx = qx.transpose(1, 2); kx = kx.transpose(1, 2); vx = vx.transpose(1, 2)
        qy = qy.transpose(1, 2); ky = ky.transpose(1, 2); vy = vy.transpose(1, 2)
        q_joint = torch.cat([qy, qx], dim=2)
        k_joint = torch.cat([ky, kx], dim=2)
        v_joint = torch.cat([vy, vx], dim=2)
        out_joint = F.scaled_dot_product_attention(q_joint, k_joint, v_joint, dropout_p=0.0, attn_mask=attn_mask)
        out_y = out_joint[:, :, :Ny, :]
        out_x = out_joint[:, :, Ny:, :]
        out_y = out_y.transpose(1, 2).reshape(B, Ny, C)
        out_x = out_x.transpose(1, 2).reshape(B, Nx, C)
        out_x = self.proj_drop_x(self.proj_x(out_x))
        out_y = self.proj_drop_y(self.proj_y(out_y))
        return out_x, out_y


class MMDiTBlockT2I(nn.Module):
    def __init__(self, hidden_size, groups, mlp_ratio=4.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.groups = groups
        self.head_dim = hidden_size // groups
        self.norm_x1 = RMSNorm(hidden_size, eps=1e-6)
        self.norm_y1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = MMDiTJointAttention(hidden_size, num_heads=groups, qkv_bias=False)
        self.norm_x2 = RMSNorm(hidden_size, eps=1e-6)
        self.norm_y2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp_x = FeedForward(hidden_size, mlp_hidden_dim)
        self.mlp_y = FeedForward(hidden_size, mlp_hidden_dim)
        self.adaLN_modulation_img = nn.Sequential(nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        self.adaLN_modulation_txt = nn.Sequential(nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x, y, c, pos_img, pos_txt=None, attn_mask=None):
        shift_msa_x, scale_msa_x, gate_msa_x, shift_mlp_x, scale_mlp_x, gate_mlp_x = self.adaLN_modulation_img(c).chunk(6, dim=-1)
        shift_msa_y, scale_msa_y, gate_msa_y, shift_mlp_y, scale_mlp_y, gate_mlp_y = self.adaLN_modulation_txt(c).chunk(6, dim=-1)
        x_norm = apply_adaln(self.norm_x1(x), shift_msa_x, scale_msa_x)
        y_norm = apply_adaln(self.norm_y1(y), shift_msa_y, scale_msa_y)
        attn_x, attn_y = self.attn(x_norm, y_norm, pos_img, pos_txt, attn_mask)
        x = x + gate_msa_x * attn_x
        y = y + gate_msa_y * attn_y
        x = x + gate_mlp_x * self.mlp_x(apply_adaln(self.norm_x2(x), shift_mlp_x, scale_mlp_x))
        y = y + gate_mlp_y * self.mlp_y(apply_adaln(self.norm_y2(y), shift_mlp_y, scale_mlp_y))
        return x, y


# ============================================================================
# Tiny config (exercises full block: joint attn, image NTK RoPE, QK-norm,
# per-stream AdaLN 6-chunk, SwiGLU FFN). B=1 (per-channel modulation).
# ============================================================================
HIDDEN = 48
GROUPS = 4
HEAD_DIM = HIDDEN // GROUPS          # 12  (even; dim//4 = 3 RoPE pairs)
B = 1
NX_H, NX_W = 2, 3                    # image patch grid -> Nx = 6
NX = NX_H * NX_W
NY = 4                               # text tokens
ROPE_REF = 1024
MLP_RATIO = 4.0

blk = MMDiTBlockT2I(HIDDEN, GROUPS, mlp_ratio=MLP_RATIO).eval()
# scale projections small so F32 round-trips cleanly
with torch.no_grad():
    blk.attn.qkv_x.weight.mul_(0.3)
    blk.attn.qkv_y.weight.mul_(0.3)
    blk.attn.proj_x.weight.mul_(0.3)
    blk.attn.proj_y.weight.mul_(0.3)
    blk.mlp_x.w1.weight.mul_(0.3); blk.mlp_x.w2.weight.mul_(0.3); blk.mlp_x.w3.weight.mul_(0.3)
    blk.mlp_y.w1.weight.mul_(0.3); blk.mlp_y.w2.weight.mul_(0.3); blk.mlp_y.w3.weight.mul_(0.3)
    blk.adaLN_modulation_img[0].weight.mul_(0.2)
    blk.adaLN_modulation_txt[0].weight.mul_(0.2)
    # give the qk_norm weights some structure (not all-ones) to exercise it
    blk.attn.q_norm_x.weight.copy_(0.5 + 0.1 * torch.randn(HEAD_DIM))
    blk.attn.k_norm_x.weight.copy_(0.5 + 0.1 * torch.randn(HEAD_DIM))
    blk.attn.q_norm_y.weight.copy_(0.5 + 0.1 * torch.randn(HEAD_DIM))
    blk.attn.k_norm_y.weight.copy_(0.5 + 0.1 * torch.randn(HEAD_DIM))
    # block norms (norm_x1 etc) also non-unit
    blk.norm_x1.weight.copy_(0.5 + 0.1 * torch.randn(HIDDEN))
    blk.norm_y1.weight.copy_(0.5 + 0.1 * torch.randn(HIDDEN))
    blk.norm_x2.weight.copy_(0.5 + 0.1 * torch.randn(HIDDEN))
    blk.norm_y2.weight.copy_(0.5 + 0.1 * torch.randn(HIDDEN))

x = torch.randn(B, NX, HIDDEN, dtype=torch.float32)
y = torch.randn(B, NY, HIDDEN, dtype=torch.float32)
c = torch.randn(B, 1, HIDDEN, dtype=torch.float32)
pos_img = precompute_freqs_cis_2d_ntk(HEAD_DIM, NX_H, NX_W, ROPE_REF, ROPE_REF)  # complex [Nx, head_dim//2]

with torch.no_grad():
    x_out, y_out = blk(x, y, c, pos_img, pos_txt=None, attn_mask=None)

# RoPE table split (cos = real, sin = imag), [Nx, head_dim//2]
rope_cos = pos_img.real.contiguous().numpy()
rope_sin = pos_img.imag.contiguous().numpy()
ROPE_HALF = HEAD_DIM // 2

# FeedForward inner hidden dim (after the int(2*h/3) reduction)
FF_HIDDEN = int(2 * int(HIDDEN * MLP_RATIO) / 3)


def w(p):
    return p.detach().numpy()


# ============================================================================
# Emit Mojo fixture.
# ============================================================================
print("# pixeldit_block_ref_data.mojo — GENERATED by gen_pixeldit_block_reference.py. DO NOT EDIT.")
print("#")
print("# DEV-ONLY parity fixture for serenitymojo/models/pid/pixeldit_block_smoke.mojo.")
print("# PyTorch oracle (system python3, torch %s), seed=%d. F32." % (torch.__version__, SEED))
print("# One PiD PixelDiT MMDiTBlockT2I forward (joint attn + image NTK RoPE +")
print("# per-stream QK-RMSNorm + per-stream AdaLN 6-chunk + SwiGLU FFN), B=1, pos_txt=None.")
print()

emit_scalar_i("HIDDEN", HIDDEN)
emit_scalar_i("GROUPS", GROUPS)
emit_scalar_i("HEAD_DIM", HEAD_DIM)
emit_scalar_i("B", B)
emit_scalar_i("NX", NX)
emit_scalar_i("NY", NY)
emit_scalar_i("NX_H", NX_H)
emit_scalar_i("NX_W", NX_W)
emit_scalar_i("ROPE_REF", ROPE_REF)
emit_scalar_i("ROPE_HALF", ROPE_HALF)
emit_scalar_i("FF_HIDDEN", FF_HIDDEN)
print()

# Inputs
emit_list("inp_x", x.numpy())
emit_list("inp_y", y.numpy())
emit_list("inp_c", c.numpy())
emit_list("rope_cos", rope_cos)
emit_list("rope_sin", rope_sin)

# AdaLN modulation Linears
emit_list("adaln_img_w", w(blk.adaLN_modulation_img[0].weight))  # [6H, H]
emit_list("adaln_img_b", w(blk.adaLN_modulation_img[0].bias))    # [6H]
emit_list("adaln_txt_w", w(blk.adaLN_modulation_txt[0].weight))
emit_list("adaln_txt_b", w(blk.adaLN_modulation_txt[0].bias))

# Block norms
emit_list("norm_x1_w", w(blk.norm_x1.weight))
emit_list("norm_y1_w", w(blk.norm_y1.weight))
emit_list("norm_x2_w", w(blk.norm_x2.weight))
emit_list("norm_y2_w", w(blk.norm_y2.weight))

# Attention: separate qkv_x / qkv_y (no bias), qk-norms, output proj (with bias)
emit_list("qkv_x_w", w(blk.attn.qkv_x.weight))  # [3H, H]
emit_list("qkv_y_w", w(blk.attn.qkv_y.weight))
emit_list("q_norm_x_w", w(blk.attn.q_norm_x.weight))  # [head_dim]
emit_list("k_norm_x_w", w(blk.attn.k_norm_x.weight))
emit_list("q_norm_y_w", w(blk.attn.q_norm_y.weight))
emit_list("k_norm_y_w", w(blk.attn.k_norm_y.weight))
emit_list("proj_x_w", w(blk.attn.proj_x.weight))  # [H, H]
emit_list("proj_x_b", w(blk.attn.proj_x.bias))    # [H]
emit_list("proj_y_w", w(blk.attn.proj_y.weight))
emit_list("proj_y_b", w(blk.attn.proj_y.bias))

# SwiGLU FFNs (no bias)
emit_list("mlp_x_w1", w(blk.mlp_x.w1.weight))  # [FF, H]
emit_list("mlp_x_w3", w(blk.mlp_x.w3.weight))  # [FF, H]
emit_list("mlp_x_w2", w(blk.mlp_x.w2.weight))  # [H, FF]
emit_list("mlp_y_w1", w(blk.mlp_y.w1.weight))
emit_list("mlp_y_w3", w(blk.mlp_y.w3.weight))
emit_list("mlp_y_w2", w(blk.mlp_y.w2.weight))

# Reference outputs
emit_list("out_x_ref", x_out.numpy())
emit_list("out_y_ref", y_out.numpy())

print("# end of generated fixture", file=sys.stderr)

#!/usr/bin/env python3
# gen_pit_block_reference.py — DEV-ONLY PyTorch oracle for the PiTBlock parity
# smoke (serenitymojo/models/pid/pit_block_smoke.mojo).
#
# NOT in the runtime path. Run with SYSTEM python3 (the pixi env has no torch):
#   python3 serenitymojo/models/pid/parity/gen_pit_block_reference.py > \
#       serenitymojo/models/pid/parity/pit_block_ref_data.mojo
#
# Mirrors the EXACT PiTBlock from the PiD repo (pixeldit_official.py:416-509)
# in the no-context-parallel (cp_size=1), mask=None path. Random SEEDED weights,
# tiny inputs. F32 throughout so the Mojo gate is bit-close (cos >= 0.999).
#
# Sub-modules reproduced verbatim from pixeldit_official.py:
#   RMSNorm (eps=1e-6), RotaryAttention (qkv_bias=False, qk_norm=True),
#   precompute_freqs_cis_2d_ntk, apply_rotary_emb, MLP (nn.GELU exact erf),
#   adaLN_modulation = Linear(context_dim, 6*pixel_dim*P2).

import math
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention

SEED = 4242
torch.manual_seed(SEED)
np.random.seed(SEED)


def emit_scalar_i(name, v):
    print(f"comptime {name} = {int(v)}")


def emit_list(name, arr):
    a = np.asarray(arr, dtype=np.float64).reshape(-1)
    print(f"def {name}() -> List[Float32]:")
    print("    var v = List[Float32]()")
    for x in a:
        print(f"    v.append(Float32({float(x):.8f}))")
    print("    return v^")
    print()


# ── verbatim PiD sub-modules ────────────────────────────────────────────────
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


class RotaryAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_norm=True,
                 norm_layer=RMSNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, pos, mask):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = apply_rotary_emb(q, k, freqs_cis=pos)
        q = q.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2).contiguous()
        v = v.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2).contiguous()
        x = scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PiTBlock(nn.Module):
    def __init__(self, pixel_hidden_size, patch_hidden_size, patch_size,
                 num_heads, mlp_ratio=4.0, attn_hidden_size=None,
                 attn_num_heads=None, rope_mode="ntk_aware",
                 rope_ref_grid_h=32, rope_ref_grid_w=32):
        super().__init__()
        self.pixel_dim = int(pixel_hidden_size)
        self.context_dim = int(patch_hidden_size)
        self.patch_size = int(patch_size)
        self.attn_dim = int(attn_hidden_size) if attn_hidden_size is not None else self.context_dim
        self.num_heads = int(attn_num_heads) if attn_num_heads is not None else int(num_heads)
        self.rope_mode = rope_mode
        self.rope_ref_grid_h = rope_ref_grid_h
        self.rope_ref_grid_w = rope_ref_grid_w
        p2 = self.patch_size * self.patch_size
        self.compress_to_attn = nn.Linear(p2 * self.pixel_dim, self.attn_dim, bias=True)
        self.expand_from_attn = nn.Linear(self.attn_dim, p2 * self.pixel_dim, bias=True)
        self.norm1 = RMSNorm(self.pixel_dim, eps=1e-6)
        self.attn = RotaryAttention(self.attn_dim, num_heads=self.num_heads, qkv_bias=False)
        self.norm2 = RMSNorm(self.pixel_dim, eps=1e-6)
        self.mlp = MLP(self.pixel_dim, mlp_ratio=mlp_ratio, drop=0.0)
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(self.context_dim, 6 * self.pixel_dim * p2, bias=True)
        )

    def _fetch_pos(self, height, width):
        head_dim = self.attn_dim // self.num_heads
        return precompute_freqs_cis_2d_ntk(
            head_dim, height, width, self.rope_ref_grid_h, self.rope_ref_grid_w
        )

    def forward(self, x, s_cond, image_height, image_width, patch_size, mask=None):
        BL, P2, C = x.shape
        Hs, Ws = image_height // patch_size, image_width // patch_size
        L = Hs * Ws
        L_local = L
        B = BL // L_local
        cond_params = self.adaLN_modulation(s_cond)
        cond_params = cond_params.view(BL, P2, 6 * self.pixel_dim)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(
            cond_params, 6, dim=-1
        )
        x_norm = apply_adaln(self.norm1(x), shift_msa, scale_msa)
        x_flat = x_norm.view(BL, P2 * self.pixel_dim)
        x_comp = self.compress_to_attn(x_flat).view(B, L_local, self.attn_dim)
        pos_comp = self._fetch_pos(Hs, Ws)
        attn_out = self.attn(x_comp, pos_comp, mask)
        attn_flat = self.expand_from_attn(attn_out.view(B * L_local, self.attn_dim))
        attn_exp = attn_flat.view(BL, P2, self.pixel_dim)
        x = x + gate_msa * attn_exp
        mlp_out = self.mlp(apply_adaln(self.norm2(x), shift_mlp, scale_mlp))
        x = x + gate_mlp * mlp_out
        return x


# ── tiny config — exercises full block, multi-patch grid, multi-head ────────
PIXEL_DIM = 8       # pixel hidden size (C)
CONTEXT_DIM = 12    # patch hidden size (s_cond dim)
ATTN_DIM = 16       # attention hidden size
ATTN_HEADS = 2      # head_dim = 8
PATCH_SIZE = 2      # P2 = 4
MLP_RATIO = 4.0
ROPE_REF = 1024
# image: H=4, W=6 -> Hs=2, Ws=3 -> L=6 patches (NTK exercised: 2,3 vs 1024)
IMG_H, IMG_W = 4, 6
Hs, Ws = IMG_H // PATCH_SIZE, IMG_W // PATCH_SIZE   # 2, 3
L = Hs * Ws                                          # 6
B = 1
P2 = PATCH_SIZE * PATCH_SIZE                          # 4
BL = B * L                                            # 6

blk = PiTBlock(
    pixel_hidden_size=PIXEL_DIM,
    patch_hidden_size=CONTEXT_DIM,
    patch_size=PATCH_SIZE,
    num_heads=ATTN_HEADS,
    mlp_ratio=MLP_RATIO,
    attn_hidden_size=ATTN_DIM,
    attn_num_heads=ATTN_HEADS,
    rope_mode="ntk_aware",
    rope_ref_grid_h=ROPE_REF,
    rope_ref_grid_w=ROPE_REF,
).eval()

# scale all linear weights small so F32 stays well-conditioned + gate near 0.5
with torch.no_grad():
    blk.compress_to_attn.weight.mul_(0.2)
    blk.expand_from_attn.weight.mul_(0.2)
    blk.attn.qkv.weight.mul_(0.2)
    blk.attn.proj.weight.mul_(0.2)
    blk.mlp.fc1.weight.mul_(0.2)
    blk.mlp.fc2.weight.mul_(0.2)
    blk.adaLN_modulation[0].weight.mul_(0.1)

x_in = torch.randn(BL, P2, PIXEL_DIM, dtype=torch.float32)
s_cond = torch.randn(BL, CONTEXT_DIM, dtype=torch.float32)

with torch.no_grad():
    y = blk(x_in, s_cond, IMG_H, IMG_W, PATCH_SIZE, mask=None)

HEAD_DIM = ATTN_DIM // ATTN_HEADS   # 8
P2D = P2 * PIXEL_DIM                  # 32 (compress in-dim)
MLP_HIDDEN = int(PIXEL_DIM * MLP_RATIO)  # 32

# weights (PyTorch row-major [out,in])
compress_w = blk.compress_to_attn.weight.detach().numpy()  # [ATTN_DIM, P2D]
compress_b = blk.compress_to_attn.bias.detach().numpy()    # [ATTN_DIM]
expand_w = blk.expand_from_attn.weight.detach().numpy()     # [P2D, ATTN_DIM]
expand_b = blk.expand_from_attn.bias.detach().numpy()       # [P2D]
qkv_w = blk.attn.qkv.weight.detach().numpy()                # [3*ATTN_DIM, ATTN_DIM]
proj_w = blk.attn.proj.weight.detach().numpy()              # [ATTN_DIM, ATTN_DIM]
proj_b = blk.attn.proj.bias.detach().numpy()                # [ATTN_DIM]
qnorm_w = blk.attn.q_norm.weight.detach().numpy()           # [HEAD_DIM]
knorm_w = blk.attn.k_norm.weight.detach().numpy()           # [HEAD_DIM]
norm1_w = blk.norm1.weight.detach().numpy()                 # [PIXEL_DIM]
norm2_w = blk.norm2.weight.detach().numpy()                 # [PIXEL_DIM]
fc1_w = blk.mlp.fc1.weight.detach().numpy()                 # [MLP_HIDDEN, PIXEL_DIM]
fc1_b = blk.mlp.fc1.bias.detach().numpy()                   # [MLP_HIDDEN]
fc2_w = blk.mlp.fc2.weight.detach().numpy()                 # [PIXEL_DIM, MLP_HIDDEN]
fc2_b = blk.mlp.fc2.bias.detach().numpy()                   # [PIXEL_DIM]
adaln_w = blk.adaLN_modulation[0].weight.detach().numpy()   # [6*PIXEL_DIM*P2, CONTEXT_DIM]
adaln_b = blk.adaLN_modulation[0].bias.detach().numpy()     # [6*PIXEL_DIM*P2]


print("# pit_block_ref_data.mojo — GENERATED by gen_pit_block_reference.py. DO NOT EDIT.")
print("#")
print("# DEV-ONLY parity fixture for serenitymojo/models/pid/pit_block_smoke.mojo.")
print("# PyTorch oracle (system python3, torch %s), seed=%d. F32." % (torch.__version__, SEED))
print("# Mirrors PiD repo PiTBlock (pixeldit_official.py:416) no-CP, mask=None path.")
print()

emit_scalar_i("PIXEL_DIM", PIXEL_DIM)
emit_scalar_i("CONTEXT_DIM", CONTEXT_DIM)
emit_scalar_i("ATTN_DIM", ATTN_DIM)
emit_scalar_i("ATTN_HEADS", ATTN_HEADS)
emit_scalar_i("HEAD_DIM", HEAD_DIM)
emit_scalar_i("PATCH_SIZE", PATCH_SIZE)
emit_scalar_i("P2", P2)
emit_scalar_i("P2D", P2D)
emit_scalar_i("MLP_HIDDEN", MLP_HIDDEN)
emit_scalar_i("IMG_H", IMG_H)
emit_scalar_i("IMG_W", IMG_W)
emit_scalar_i("HS", Hs)
emit_scalar_i("WS", Ws)
emit_scalar_i("L", L)
emit_scalar_i("B", B)
emit_scalar_i("BL", BL)
emit_scalar_i("ROPE_REF", ROPE_REF)
print()

emit_list("pit_x", x_in.numpy())
emit_list("pit_scond", s_cond.numpy())
emit_list("pit_compress_w", compress_w)
emit_list("pit_compress_b", compress_b)
emit_list("pit_expand_w", expand_w)
emit_list("pit_expand_b", expand_b)
emit_list("pit_qkv_w", qkv_w)
emit_list("pit_proj_w", proj_w)
emit_list("pit_proj_b", proj_b)
emit_list("pit_qnorm_w", qnorm_w)
emit_list("pit_knorm_w", knorm_w)
emit_list("pit_norm1_w", norm1_w)
emit_list("pit_norm2_w", norm2_w)
emit_list("pit_fc1_w", fc1_w)
emit_list("pit_fc1_b", fc1_b)
emit_list("pit_fc2_w", fc2_w)
emit_list("pit_fc2_b", fc2_b)
emit_list("pit_adaln_w", adaln_w)
emit_list("pit_adaln_b", adaln_b)
emit_list("pit_y_ref", y.numpy())

print("# end of generated fixture", file=sys.stderr)

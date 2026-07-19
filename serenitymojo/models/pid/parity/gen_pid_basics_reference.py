#!/usr/bin/env python3
# gen_pid_basics_reference.py — DEV-ONLY PyTorch oracle for the PiD scalar
# primitives parity smoke (serenitymojo/models/pid/pid_ops_smoke.mojo).
#
# NOT in the runtime path. Run with SYSTEM python3 (the pixi env has no torch):
#   python3 serenitymojo/models/pid/parity/gen_pid_basics_reference.py > \
#       serenitymojo/models/pid/parity/pid_basics_ref_data.mojo
#
# Mirrors the EXACT classes/functions from the PiD repo (/tmp/PiD_repo):
#   - unfold/fold ps=16 patchify/unpatchify    pid_net.py:303,464
#   - precompute_freqs_cis_2d_ntk              pixeldit_official.py:154-193
#   - TimestepConditioner (max_period=10)      pixeldit_official.py:80-108
#   - SigmaAwareGatePerTokenPerDim             lq_projection_2d.py:28-56
#
# Random SEEDED weights, tiny inputs. F32 throughout so the Mojo gate is
# bit-close (cos >= 0.999 / small max_abs). No Python at runtime — the Mojo
# smoke inlines the numbers printed here.

import math
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 7777
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
# (a) unfold / fold at patch_size=16
#   x [B,3,H,W] -> unfold(ks=ps,stride=ps).transpose(1,2) -> [B,L,3*ps*ps]
#   token[c*ps*ps + kh*ps + kw] = x[b,c, ph*ps+kh, pw*ps+kw]
#   fold = exact inverse (non-overlapping).
# ============================================================================
PS = 16
B_uf, C_uf = 1, 3
H_uf, W_uf = 32, 48  # pH=2, pW=3, L=6 (tiny but exercises multi-patch grid)
x_uf = (torch.randn(B_uf, C_uf, H_uf, W_uf, dtype=torch.float32))
tokens_uf = F.unfold(x_uf, kernel_size=PS, stride=PS).transpose(1, 2)  # [B,L,3*256]
fold_uf = F.fold(
    tokens_uf.transpose(1, 2), (H_uf, W_uf), kernel_size=PS, stride=PS
)  # [B,3,H,W]  (exact inverse)
assert torch.equal(fold_uf, x_uf), "fold(unfold(x)) != x"


# ============================================================================
# (b) NTK-aware 2D RoPE  (precompute_freqs_cis_2d_ntk)
#   freqs_cis: complex [L, dim//2] packed as [.., (x_cis, y_cis)] interleaved
#   reshaped to [H*W, dim//2 complex]. We dump cos (real) and sin (imag) split.
#   ref_grid = 1024 (rope_ref_h = rope_ref_w). head_dim = 64 (1536/24 heads).
# ============================================================================
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
    return freqs_cis  # complex [L, dim//2]


ROPE_DIM = 64       # head_dim (hidden 1536 / 24 heads)
ROPE_REF = 1024     # rope_ref_h = rope_ref_w
RH, RW = 4, 6       # tiny non-ref grid so NTK scaling is exercised
freqs_cis = precompute_freqs_cis_2d_ntk(ROPE_DIM, RH, RW, ROPE_REF, ROPE_REF)
rope_cos = freqs_cis.real.contiguous().numpy()  # [L, dim//2]
rope_sin = freqs_cis.imag.contiguous().numpy()  # [L, dim//2]
ROPE_L = RH * RW
ROPE_HALF = ROPE_DIM // 2

# Also verify the ref-grid identity: at height==ref and width==ref the NTK
# factor is exactly 1, so the table equals the plain (non-NTK) one. We emit a
# second tiny table at the ref grid scaled down — instead we just sanity-check
# the h_ntk/w_ntk values here in Python (informational).
_dim_axis = ROPE_DIM // 2
_h_ntk = (RH / ROPE_REF) ** (_dim_axis / (_dim_axis - 2))
_w_ntk = (RW / ROPE_REF) ** (_dim_axis / (_dim_axis - 2))


# ============================================================================
# (c) TimestepConditioner   (max_period=10)
#   timestep_embedding: half=dim//2, freqs=exp(-ln(mp)*arange(half)/half)
#     args = t[...,None]*freqs ; emb = cat([cos(args), sin(args)], -1)
#   forward: emb -> Linear(freq->hidden) -> SiLU -> Linear(hidden->hidden)
# ============================================================================
class TimestepConditioner(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[..., None].float() * freqs[None, ...]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


TS_FREQ = 256       # frequency_embedding_size
TS_HIDDEN = 48      # hidden_size (tiny stand-in for 1536; exercises full MLP)
TS_MAXP = 10
tc = TimestepConditioner(TS_HIDDEN, TS_FREQ).eval()
# scale weights small so F32/bf16 stay well-conditioned
with torch.no_grad():
    tc.mlp[0].weight.mul_(0.2)
    tc.mlp[2].weight.mul_(0.2)
TS_T = torch.tensor([0.999, 0.634, 0.0], dtype=torch.float32)  # PiD student t's
with torch.no_grad():
    ts_freq = TimestepConditioner.timestep_embedding(TS_T, TS_FREQ, TS_MAXP)  # [N,256]
    ts_out = tc(TS_T)  # [N, hidden]
TS_N = TS_T.shape[0]
tc_mlp0_w = tc.mlp[0].weight.detach().numpy()  # [hidden, freq]
tc_mlp0_b = tc.mlp[0].bias.detach().numpy()    # [hidden]
tc_mlp2_w = tc.mlp[2].weight.detach().numpy()  # [hidden, hidden]
tc_mlp2_b = tc.mlp[2].bias.detach().numpy()    # [hidden]


# ============================================================================
# (d) SigmaAwareGatePerTokenPerDim
#   content_logit = content_proj(cat([x, lq], -1))             # Linear 2D->D
#   sigma_offset  = -exp(log_alpha) * sigma.view(-1,1,1)       # (B,1,1)
#   gate          = sigmoid(content_logit + sigma_offset)      # (B,N,D)
#   out           = x + gate * lq
# ============================================================================
class SigmaAwareGatePerTokenPerDim(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.content_proj = nn.Linear(dim * 2, dim)
        nn.init.trunc_normal_(self.content_proj.weight, std=0.01)
        nn.init.constant_(self.content_proj.bias, 2.0)
        self.log_alpha = nn.Parameter(torch.tensor(math.log(5.0)))

    def forward(self, x, lq, sigma):
        content_logit = self.content_proj(torch.cat([x, lq], dim=-1))
        sigma_offset = -self.log_alpha.exp() * sigma.float().view(-1, 1, 1)
        gate = torch.sigmoid(content_logit + sigma_offset)
        return x + gate * lq


GATE_D = 32
GATE_B, GATE_N = 2, 4
gate = SigmaAwareGatePerTokenPerDim(GATE_D).eval()
gate_x = torch.randn(GATE_B, GATE_N, GATE_D, dtype=torch.float32)
gate_lq = torch.randn(GATE_B, GATE_N, GATE_D, dtype=torch.float32)
gate_sigma = torch.tensor([0.0, 0.4], dtype=torch.float32)  # per-sample sigma
with torch.no_grad():
    gate_out = gate(gate_x, gate_lq, gate_sigma)  # [B,N,D]
gate_w = gate.content_proj.weight.detach().numpy()   # [D, 2D]
gate_b = gate.content_proj.bias.detach().numpy()      # [D]
gate_log_alpha = float(gate.log_alpha.detach().item())


# ============================================================================
# Emit the Mojo reference file
# ============================================================================
print("# pid_basics_ref_data.mojo — GENERATED by gen_pid_basics_reference.py. DO NOT EDIT.")
print("#")
print("# DEV-ONLY parity fixture for serenitymojo/models/pid/pid_ops_smoke.mojo.")
print("# PyTorch oracle (system python3, torch %s), seed=%d. F32." % (torch.__version__, SEED))
print("# Mirrors PiD repo: unfold/fold ps=16, precompute_freqs_cis_2d_ntk,")
print("# TimestepConditioner(max_period=10), SigmaAwareGatePerTokenPerDim.")
print(f"# rope NTK factors: h_ntk={_h_ntk:.6e} w_ntk={_w_ntk:.6e}")
print()

# (a) unfold/fold
emit_scalar_i("UF_B", B_uf)
emit_scalar_i("UF_C", C_uf)
emit_scalar_i("UF_H", H_uf)
emit_scalar_i("UF_W", W_uf)
emit_scalar_i("UF_PS", PS)
emit_scalar_i("UF_L", tokens_uf.shape[1])
emit_scalar_i("UF_TOK", tokens_uf.shape[2])  # 3*ps*ps
print()
emit_list("uf_x", x_uf.numpy())
emit_list("uf_tokens_ref", tokens_uf.numpy())
emit_list("uf_fold_ref", fold_uf.numpy())

# (b) ntk rope
emit_scalar_i("ROPE_DIM", ROPE_DIM)
emit_scalar_i("ROPE_REF", ROPE_REF)
emit_scalar_i("ROPE_RH", RH)
emit_scalar_i("ROPE_RW", RW)
emit_scalar_i("ROPE_L", ROPE_L)
emit_scalar_i("ROPE_HALF", ROPE_HALF)
print()
emit_list("rope_cos_ref", rope_cos)
emit_list("rope_sin_ref", rope_sin)

# (c) timestep conditioner
emit_scalar_i("TS_N", TS_N)
emit_scalar_i("TS_FREQ", TS_FREQ)
emit_scalar_i("TS_HIDDEN", TS_HIDDEN)
emit_scalar_f("TS_MAXP", TS_MAXP)
print()
emit_list("ts_t", TS_T.numpy())
emit_list("ts_freq_ref", ts_freq.numpy())
emit_list("ts_mlp0_w", tc_mlp0_w)
emit_list("ts_mlp0_b", tc_mlp0_b)
emit_list("ts_mlp2_w", tc_mlp2_w)
emit_list("ts_mlp2_b", tc_mlp2_b)
emit_list("ts_out_ref", ts_out.numpy())

# (d) sigma-aware gate
emit_scalar_i("GATE_D", GATE_D)
emit_scalar_i("GATE_B", GATE_B)
emit_scalar_i("GATE_N", GATE_N)
emit_scalar_f("GATE_LOG_ALPHA", gate_log_alpha)
print()
emit_list("gate_x", gate_x.numpy())
emit_list("gate_lq", gate_lq.numpy())
emit_list("gate_sigma", gate_sigma.numpy())
emit_list("gate_w", gate_w)
emit_list("gate_b", gate_b)
emit_list("gate_out_ref", gate_out.numpy())

print("# end of generated fixture", file=sys.stderr)

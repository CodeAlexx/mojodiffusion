# Generate the FULL PidNet forward reference (F32) for the Mojo parity smoke.
#
# Builds PidNet with the released SD3 res2k config, loads the converted
# safetensors weights (cast to F32), runs ONE forward at a small grid
# (H=W=64 -> pH=pW=4 -> L=16 patch tokens), and dumps:
#   inputs:  x [B,3,H,W], t_scaled [B], y [B,Ltxt,2304], lq_latent [B,16,zH,zW],
#            degrade_sigma [B]
#   per-block ladder: s_main after each of the 14 patch blocks (post-block),
#            and after each LQ-gate injection (pre-block), x_pixels after each
#            of the 2 pixel blocks
#   final:   net output [B,3,H,W] AND velocity->x0 at t (for the sampler gate)
#
# Everything F32 on CUDA so the Mojo F32 chain compares against an F32 oracle
# (isolates op correctness from bf16 drift; the verdict still classifies bf16
# behaviour). Output: pid_net_ref.safetensors next to this file.
import sys
import math
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/tmp/PiD_repo")

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from pid._src.networks.pid_net import PidNet

CKPT = "/home/alex/.serenity/models/pid/checkpoints/PiD_res2k_sr4x_official_sd3_distill_4step/model_ema_bf16.safetensors"
OUT = "/home/alex/mojodiffusion/serenitymojo/models/pid/parity/pid_net_ref.safetensors"

torch.manual_seed(7777)
dev = "cuda"

# --- Build the net with the released SD3 config (model_pid PID_SR4X + experiment override) ---
net = PidNet(
    in_channels=3,
    num_groups=24,
    hidden_size=1536,
    pixel_hidden_size=16,
    pixel_attn_hidden_size=1152,
    pixel_num_groups=16,
    patch_depth=14,
    pixel_depth=2,
    patch_size=16,
    txt_embed_dim=2304,
    txt_max_length=300,
    use_text_rope=True,
    text_rope_theta=10000.0,
    rope_mode="ntk_aware",
    rope_ref_h=1024,
    rope_ref_w=1024,
    repa_encoder_index=6,
    enable_ed=False,
    lq_inject_mode="controlnet",
    lq_in_channels=0,
    lq_latent_channels=16,
    lq_hidden_dim=512,
    lq_num_res_blocks=4,
    lq_gate_type="sigma_aware_per_token_per_dim",
    lq_interval=2,
    zero_init_lq=True,
    sr_scale=4,
    latent_spatial_down_factor=8,
    pit_lq_inject=False,
)

# --- Load weights (F32) ---
sd = {}
with safe_open(CKPT, "pt") as f:
    for k in f.keys():
        sd[k] = f.get_tensor(k).float()
missing, unexpected = net.load_state_dict(sd, strict=False)
print("MISSING", len(missing), "UNEXPECTED", len(unexpected))
if unexpected:
    print("  unexpected sample:", unexpected[:5])
net = net.float().to(dev).eval()

# --- Inputs (small grid) ---
B = 1
H = W = 64
ps = 16
pH = pW = H // ps  # 4
L = pH * pW        # 16
Ltxt = 8
zdown = 8
sr = 4
# z_to_patch_ratio = (sr*zdown)/ps = 2 -> latent res = patch_grid/2 -> zH = pH/2 = 2
zH = zW = pH // 2  # 2

x = torch.randn(B, 3, H, W, device=dev)
t01 = torch.tensor([0.999], device=dev)          # student first timestep (unscaled)
timescale = 1000.0
t_scaled = t01 * timescale                        # what the net consumes
y = torch.randn(B, Ltxt, 2304, device=dev) * 0.5
lq_latent = torch.randn(B, 16, zH, zW, device=dev)
degrade_sigma = torch.zeros(B, device=dev)

# --- Capture per-block states via hooks ---
captured = {}

patch_inputs = {}   # s_main BEFORE block i (post any LQ gate)
def make_patch_pre(i):
    def hook(module, args):
        # args = (s_main, y_emb, condition, pos, pos_txt, attn_mask)
        captured[f"patch_pre_{i}"] = args[0].detach().float().cpu().contiguous()
    return hook
def make_patch_post(i):
    def hook(module, args, output):
        captured[f"patch_post_{i}"] = output[0].detach().float().cpu().contiguous()
    return hook
for i, blk in enumerate(net.patch_blocks):
    blk.register_forward_pre_hook(make_patch_pre(i))
    blk.register_forward_hook(make_patch_post(i))

def make_pixel_post(i):
    def hook(module, args, output):
        captured[f"pixel_post_{i}"] = output.detach().float().cpu().contiguous()
    return hook
for i, blk in enumerate(net.pixel_blocks):
    blk.register_forward_hook(make_pixel_post(i))

# Capture lq_features and s_cond and x_pixels-init by wrapping methods
orig_compute = net._compute_lq_features
def wrapped_compute(*a, **k):
    feats = orig_compute(*a, **k)
    for j, ft in enumerate(feats):
        captured[f"lq_feat_{j}"] = ft.detach().float().cpu().contiguous()
    return feats
net._compute_lq_features = wrapped_compute

orig_pix = net.pixel_embedder.forward
def wrapped_pix(*a, **k):
    out = orig_pix(*a, **k)
    captured["x_pixels_init"] = out.detach().float().cpu().contiguous()
    return out
net.pixel_embedder.forward = wrapped_pix

# Also dump the auxiliary tables the Mojo side would otherwise recompute, so
# parity on the assembled net isolates block correctness (sincos pos embed +
# image NTK RoPE + text 1D RoPE are dumped as cos/sin real tensors).
from pid._src.networks.pixeldit_official import (
    get_2d_sincos_pos_embed, precompute_freqs_cis_2d_ntk,
)
# pixel-embedder full-image sincos pos [H*W, 16]
pix_pos = torch.from_numpy(get_2d_sincos_pos_embed(16, H)).float()  # square H==W
# image NTK RoPE on patch grid (head_dim=64), [L, 32] complex -> cos/sin
img_cis = precompute_freqs_cis_2d_ntk(64, pH, pW, 1024 // 16, 1024 // 16)  # [L,32] complex
img_cos = img_cis.real.float().contiguous()
img_sin = img_cis.imag.float().contiguous()
# text 1D RoPE [Ltxt, 32] from fetch_pos_text(theta=10000, head_dim=64)
hd = 64
freqs = 1.0 / (10000.0 ** (torch.arange(0, hd, 2).float() / hd))  # [32]
posn = torch.arange(0, Ltxt).float().unsqueeze(1)                  # [Ltxt,1]
ang = posn * freqs.unsqueeze(0)                                    # [Ltxt,32]
txt_cos = torch.cos(ang).float().contiguous()
txt_sin = torch.sin(ang).float().contiguous()
# pixel-block NTK RoPE: head_dim = pixel_attn_hidden/pixel_groups = 1152/16 = 72
pix_cis = precompute_freqs_cis_2d_ntk(72, pH, pW, 1024 // 16, 1024 // 16)  # [L,36]
pix_cos = pix_cis.real.float().contiguous()
pix_sin = pix_cis.imag.float().contiguous()

with torch.no_grad():
    out = net(x, t_scaled, y, lq_latent=lq_latent, degrade_sigma=degrade_sigma)

# velocity -> x0 at unscaled t (matches _velocity_to_x0, double precision)
t_shaped = t01.double().view(B, 1, 1, 1)
x0 = (x.double() - t_shaped * out.double()).float()

print("OUT", tuple(out.shape), "mean", out.mean().item(), "std", out.std().item())
print("X0 ", tuple(x0.shape), "mean", x0.mean().item(), "std", x0.std().item())

tensors = {
    "x": x.float().cpu().contiguous(),
    "t01": t01.float().cpu().contiguous(),
    "t_scaled": t_scaled.float().cpu().contiguous(),
    "y": y.float().cpu().contiguous(),
    "lq_latent": lq_latent.float().cpu().contiguous(),
    "degrade_sigma": degrade_sigma.float().cpu().contiguous(),
    "net_out": out.float().cpu().contiguous(),
    "x0": x0.cpu().contiguous(),
}
tensors.update(captured)
tensors["pix_pos"] = pix_pos.cpu().contiguous()
tensors["img_cos"] = img_cos.cpu().contiguous()
tensors["img_sin"] = img_sin.cpu().contiguous()
tensors["txt_cos"] = txt_cos.cpu().contiguous()
tensors["txt_sin"] = txt_sin.cpu().contiguous()
tensors["pix_cos"] = pix_cos.cpu().contiguous()
tensors["pix_sin"] = pix_sin.cpu().contiguous()
# meta scalars as 1-elem tensors
tensors["_meta"] = torch.tensor([B, H, W, ps, pH, pW, L, Ltxt, zH, zW], dtype=torch.int32)
save_file(tensors, OUT)
print("SAVED", OUT, "n_tensors", len(tensors))
print("captured keys:", sorted([k for k in captured]))

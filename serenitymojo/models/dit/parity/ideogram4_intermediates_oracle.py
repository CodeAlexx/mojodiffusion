# DEV-ONLY: dump Ideogram-4 transformer INTERMEDIATES on the predict fixture's
# inputs, via forward hooks, so the Rust DiT can be gated layer-by-layer to
# localize where it diverges from torch (the full velocity is cos~0).
import sys, torch
sys.path.insert(0, "/home/alex/ideogram4-ref/src")
from safetensors.torch import load_file, save_file
from ideogram4.modeling_ideogram4 import Ideogram4Config, Ideogram4Transformer
from ideogram4.quantized_loading import swap_linears_to_fp8, load_fp8_state_dict

ROOT = "/home/alex/.serenity/models/ideogram-4-fp8"
P = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity"
dev = torch.device("cuda"); dt = torch.bfloat16

fx = load_file(f"{P}/ideogram4_fx_predict.safetensors")
x = fx["x"].to(dev, dt)                         # [1,907,128] packed
llmf = fx["llm_features"].to(dev, dt)           # [1,651,53248]
nt = 651; nimg = 256
llm_full = torch.cat([llmf, torch.zeros(1, nimg, llmf.shape[-1], device=dev, dtype=dt)], 1)
model_t = fx["model_t"].to(dev)                 # [1]
indicator = fx["indicator"].to(dev).long()      # [1,907]
position_ids = fx["position_ids"].to(dev)
segment_ids = fx["segment_ids"].to(dev)

sd = load_file(f"{ROOT}/transformer/diffusion_pytorch_model.safetensors")
m = Ideogram4Transformer(Ideogram4Config()); m.to(dt)
swap_linears_to_fp8(m, sd, compute_dtype=dt); load_fp8_state_dict(m, sd, device=dev, dtype=dt); m.eval()

caps = {}
def save_out(name):
    def h(mod, inp, out):
        caps[name] = (out[0] if isinstance(out, tuple) else out).detach().float().cpu()
    return h
def save_in(name):
    def h(mod, inp):
        caps[name] = inp[0].detach().float().cpu()
    return h
m.input_proj.register_forward_hook(save_out("input_proj_out"))
m.layers[0].register_forward_pre_hook(save_in("h_pre"))
m.layers[0].register_forward_hook(save_out("block0_out"))
m.layers[1].register_forward_hook(save_out("block1_out"))
m.layers[8].register_forward_hook(save_out("block8_out"))
m.layers[16].register_forward_hook(save_out("block16_out"))
m.layers[33].register_forward_hook(save_out("block33_out"))
m.register_forward_hook(save_out("transformer_out"))

with torch.no_grad():
    m(llm_features=llm_full, x=x, t=model_t, position_ids=position_ids, segment_ids=segment_ids, indicator=indicator)

save_file({k: v.contiguous() for k, v in caps.items()}, f"{P}/ideogram4_fx_intermediates.safetensors")
print("saved", {k: tuple(v.shape) for k, v in caps.items()})

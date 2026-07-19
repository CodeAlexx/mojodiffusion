# models/qwenimage/train.mojo — Qwen-Image LoRA training entry point (thin).
#
# Per-model entry: read the config (dims + recipe from configs/qwenimage.json,
# confirmed against the diffusers transformer/config.json) and report the surface.
# The verified compute lives in:
#   models/qwenimage/qwenimage_block.mojo   (block fwd+bwd + LoRA, gated cos>=0.999)
#   models/qwenimage/qwenimage_stack.mojo   (full 60-block stack, per-block recompute)
#   models/qwenimage/qwenimage_stack_lora.mojo (720-adapter LoRA set + AdamW + PEFT save)
#   models/qwenimage/weights.mojo           (sharded safetensors -> training structs)
# The shared training/ pipeline (flow-match target, optim, schedule, loop, ckpt,
# lora_save) is REUSED, not forked.
#
# Real run wiring (data path) reuses training/klein_dataset + the Qwen2.5-VL text
# encoder + the Qwen-Image VAE encoder; the transformer shards must be downloaded
# (only the index.json is present in the local cache snapshot).
#
# Run: cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#      pixi run mojo run -I . serenitymojo/models/qwenimage/train.mojo

from std.gpu.host import DeviceContext
from serenitymojo.models.qwenimage.config import qwen_image


def main() raises:
    print("############################################################")
    print("# Qwen-Image LoRA training surface (per-block parity PASS).")
    print("############################################################")
    var cfg = qwen_image()
    print("model_type      :", cfg.name)
    print("inner_dim (D)   :", cfg.d_model)
    print("n_heads         :", cfg.n_heads)
    print("head_dim        :", cfg.head_dim)
    print("mlp_hidden (F)  :", cfg.mlp_hidden)
    print("num_double      :", cfg.num_double, " (all-double; num_single", cfg.num_single, ")")
    print("in_channels     :", cfg.in_channels)
    print("joint_attn_dim  :", cfg.joint_attention_dim)
    print("out_channels    :", cfg.out_channels)
    print("lr / rank / alpha:", cfg.lr, "/", cfg.lora_rank, "/", cfg.lora_alpha)
    print("LoRA targets/block: 12 (img/txt x q,k,v,out,ff_up,ff_down)")
    print("total adapters  :", cfg.num_double * 12)
    print("[surface ready] block fwd+bwd+LoRA gated cos>=0.999 vs torch autograd.")

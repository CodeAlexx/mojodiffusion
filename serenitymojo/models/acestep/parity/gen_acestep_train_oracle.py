#!/usr/bin/env python3
"""ACE-Step-1.5 turbo — reference training-step oracle for the Mojo trainer gate.

Runs ONE upstream fixed-recipe training step on a FIXED synthetic batch with
FIXED x1/t and CFG dropout OFF (fully deterministic), in bf16 (matching the
in-tree Mojo acestep_dit dtype), and dumps every intermediate + each LoRA grad.

The Mojo trainer step differentiates against these dumps:
  forward:  acestep_forward(+LoRA overlay) vs  out0            (cos >= bar)
  loss:     mse(out0, flow)                 vs  loss_scalar     (~exact bf16)
  backward: LoRA A/B grads                  vs  grad_<name>     (cos, measured band)

Run with the torchref venv (torch 2.12 + peft):
  /home/alex/torchref-image/venv/bin/python scratchpad/acestep_train_ref_dump.py

Reference runtime caveat: torch 2.12+cu130 vs ACE-Step's pinned 2.10+cu128 —
same weights/modeling/recipe; minor-version matmul/layernorm diffs are within
the bf16 noise floor. Documented per numeric-parity discipline.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ACE_ROOT = "/home/alex/ACE-Step-1.5"
CKPT = f"{ACE_ROOT}/checkpoints/acestep-v15-xl-base"  # 4B DiT — Alex-chosen target
OUT_DIR = Path("/home/alex/mojodiffusion/serenitymojo/models/acestep/parity/acestep_train_ref")

sys.path.insert(0, ACE_ROOT)
sys.path.insert(0, CKPT)  # so modeling file finds sibling configuration_acestep_v15

# --- fixed-shape synthetic batch (make_test_fixtures dims; bs=1) -------------
BSZ = 1
T = 128          # latent length (→ T/2=64 after conv1d patch; <= sliding_window 128 → sliding==full)
L = 64           # encoder (condition) length
TARGET_DIM = 64  # audio_acoustic_hidden_dim
CTX_DIM = 128    # 64 src_latents + 64 chunk_masks
ENC_DIM = 2048   # hidden_size
SEED = 42
DTYPE = torch.bfloat16


def log(*a):
    print("[oracle]", *a, flush=True)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    torch.manual_seed(SEED)

    # -- deps present? ------------------------------------------------------
    import importlib.util
    from safetensors.torch import load_file, save_file
    from peft import LoraConfig, get_peft_model

    def _load_mod(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    # -- build ONLY the DiT decoder (skip tokenizer/VAE/FSQ full-model path) --
    import glob
    cfgmod = _load_mod("configuration_acestep_v15", f"{CKPT}/configuration_acestep_v15.py")
    mf_path = glob.glob(f"{CKPT}/modeling_acestep_v15_*.py")[0]  # turbo or xl_base
    Mmod = _load_mod(Path(mf_path).stem, mf_path)
    log("modeling file:", Path(mf_path).name)
    AceStepConfig = cfgmod.AceStepConfig
    AceStepDiTModel = Mmod.AceStepDiTModel

    config = AceStepConfig.from_pretrained(CKPT)
    log("building DiT decoder (AceStepDiTModel)…")
    dit = AceStepDiTModel(config).to(DTYPE)

    # load ONLY decoder.* weights (single file or sharded via index)
    from safetensors import safe_open as _safe_open
    idx_path = f"{CKPT}/model.safetensors.index.json"
    dec_sd = {}
    if os.path.exists(idx_path):
        wm = json.load(open(idx_path))["weight_map"]
        from collections import defaultdict
        by_shard = defaultdict(list)
        for k, shard in wm.items():
            if k.startswith("decoder."):
                by_shard[shard].append(k)
        for shard, keys in by_shard.items():
            with _safe_open(f"{CKPT}/{shard}", framework="pt") as f:
                for k in keys:
                    dec_sd[k[len("decoder."):]] = f.get_tensor(k)
        log(f"sharded load: {len(dec_sd)} decoder tensors from {len(by_shard)} shards")
    else:
        sd = load_file(f"{CKPT}/model.safetensors")
        dec_sd = {k[len("decoder."):]: v for k, v in sd.items() if k.startswith("decoder.")}
    missing, unexpected = dit.load_state_dict(dec_sd, strict=False)
    log(f"decoder weights: {len(dec_sd)} loaded, missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        log("  missing[:6]:", list(missing)[:6])
    if unexpected:
        log("  unexpected[:6]:", list(unexpected)[:6])
    dit = dit.to(dev)

    # -- LoRA via PEFT (r8/a16, q/k/v/o, dropout=0 for determinism) -----------
    peft_cfg = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    dit = get_peft_model(dit, peft_cfg)
    dit.train()
    n_tr = sum(p.numel() for p in dit.parameters() if p.requires_grad)
    log("LoRA injected via peft:", f"{n_tr:,}", "trainable params")

    # Perturb LoRA A/B to nonzero (mid-training-like) so the gate exercises the
    # full LoRA path.  PEFT zero-inits lora_B → degenerate (out==base, grad_A==0).
    # These exact weights are dumped so the Mojo side loads identical LoRA.
    gA = torch.Generator(device=dev).manual_seed(SEED + 1)
    with torch.no_grad():
        for name, p in dit.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                p.copy_(torch.randn(p.shape, generator=gA, device=dev, dtype=p.dtype) * 0.02)

    # -- FIXED synthetic batch ---------------------------------------------
    g = torch.Generator(device=dev).manual_seed(SEED)
    target_latents = torch.randn(BSZ, T, TARGET_DIM, generator=g, device=dev, dtype=DTYPE)
    attention_mask = torch.ones(BSZ, T, device=dev, dtype=DTYPE)
    encoder_hidden_states = torch.randn(BSZ, L, ENC_DIM, generator=g, device=dev, dtype=DTYPE)
    encoder_attention_mask = torch.ones(BSZ, L, device=dev, dtype=DTYPE)
    context_latents = torch.randn(BSZ, T, CTX_DIM, generator=g, device=dev, dtype=DTYPE)

    # -- FIXED flow-matching noise + timestep (replicates sample_timesteps) --
    x1 = torch.randn(BSZ, T, TARGET_DIM, generator=g, device=dev, dtype=DTYPE)  # noise
    x0 = target_latents                                                         # data
    a = torch.randn(BSZ, generator=g, device=dev, dtype=DTYPE)
    b = torch.randn(BSZ, generator=g, device=dev, dtype=DTYPE)
    t = torch.sigmoid(torch.maximum(a, b) * 1.0 + (-0.4))  # t=max(sigmoid…); r=t
    log("t =", t.float().tolist())

    t_ = t.unsqueeze(-1).unsqueeze(-1)
    xt = t_ * x1 + (1.0 - t_) * x0

    # -- block-0 capture hook (per-block backward oracle) -------------------
    cap = {}

    def _blk0_hook(module, args, kwargs, output):
        # positional call: (hidden, position_embeddings, timestep_proj, attn_mask,
        #   position_ids, past_kv, output_attn, use_cache, cache_pos, enc, enc_mask)
        cap["x_in"] = args[0].detach().float().cpu()
        pe = args[1]
        cap["cos"] = pe[0].detach().float().cpu()
        cap["sin"] = pe[1].detach().float().cpu()
        cap["temb"] = args[2].detach().float().cpu()
        if len(args) > 9 and args[9] is not None:
            cap["enc"] = args[9].detach().float().cpu()
        out_t = output[0] if isinstance(output, (tuple, list)) else output
        out_t.register_hook(lambda g: cap.__setitem__("d_out", g.detach().float().cpu()))

    layer0 = dit.base_model.model.layers[0]
    _h = layer0.register_forward_hook(_blk0_hook, with_kwargs=True)

    # -- decoder forward (exact training_step kwargs) -----------------------
    log("forward…")
    out = dit(
        hidden_states=xt,
        timestep=t,
        timestep_r=t,
        attention_mask=attention_mask,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        context_latents=context_latents,
    )
    out0 = out[0] if isinstance(out, (tuple, list)) else getattr(out, "sample", out)
    log("out0:", tuple(out0.shape), out0.dtype)

    # -- flow-matching loss + backward --------------------------------------
    flow = x1 - x0
    loss = F.mse_loss(out0, flow).float()
    log("loss =", loss.item())
    loss.backward()

    # -- collect LoRA grads --------------------------------------------------
    grads = {}
    weights = {}
    n_lora = 0
    for name, p in dit.named_parameters():
        if p.requires_grad and "lora_" in name:
            n_lora += 1
            weights["w_" + name.replace(".", "__")] = p.detach().float().cpu().contiguous()
            if p.grad is None:
                log("  WARN no grad:", name)
                continue
            grads["grad_" + name.replace(".", "__")] = p.grad.detach().float().cpu().contiguous()
    log(f"collected {len(grads)}/{n_lora} LoRA grads + {len(weights)} LoRA weights")

    # -- dump ---------------------------------------------------------------
    tensors = {
        "target_latents": x0.detach().float().cpu().contiguous(),
        "attention_mask": attention_mask.detach().float().cpu().contiguous(),
        "encoder_hidden_states": encoder_hidden_states.detach().float().cpu().contiguous(),
        "encoder_attention_mask": encoder_attention_mask.detach().float().cpu().contiguous(),
        "context_latents": context_latents.detach().float().cpu().contiguous(),
        "x1": x1.detach().float().cpu().contiguous(),
        "xt": xt.detach().float().cpu().contiguous(),
        "t": t.detach().float().cpu().contiguous(),
        "out0": out0.detach().float().cpu().contiguous(),
        "flow": flow.detach().float().cpu().contiguous(),
        "loss": loss.detach().float().cpu().contiguous(),
    }
    tensors.update(grads)
    tensors.update(weights)
    # block-0 backward oracle (per-block gate)
    _h.remove()
    for k in ("x_in", "temb", "cos", "sin", "enc", "d_out"):
        if k in cap:
            tensors["block0_" + k] = cap[k].contiguous()
    log("block0 captured:", sorted(cap.keys()))
    save_file(tensors, str(OUT_DIR / "acestep_train_ref.safetensors"))

    manifest = {
        "variant": "acestep-v15-xl-base (4B DiT: 32L/2560/32h/8kv/9728)",
        "recipe": "acestep-v15 fixed (continuous t + CFG dropout); cfg OFF for gate",
        "dtype": "bfloat16 fwd/bwd, fp32 loss+dump",
        "torch": torch.__version__,
        "seed": SEED,
        "shapes": {"BSZ": BSZ, "T": T, "L": L, "TARGET_DIM": TARGET_DIM,
                   "CTX_DIM": CTX_DIM, "ENC_DIM": ENC_DIM},
        "lora": {"r": 8, "alpha": 16, "dropout": 0.0,
                 "targets": ["q_proj", "k_proj", "v_proj", "o_proj"]},
        "t": t.float().tolist(),
        "loss": loss.item(),
        "n_lora_grads": len(grads),
        "lora_grad_keys": sorted(grads.keys())[:8],
        "out0_shape": list(out0.shape),
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log("DUMPED →", OUT_DIR)
    log("loss=%.8f  n_grads=%d  out0=%s" % (loss.item(), len(grads), tuple(out0.shape)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

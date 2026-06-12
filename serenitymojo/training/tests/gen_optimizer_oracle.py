#!/usr/bin/env python
# training/tests/gen_optimizer_oracle.py — T1.C optimizer-parity oracle.
#
# Drives the EXACT reference implementations SimpleTuner uses:
#
#  * adafactor  -> SimpleTuner key "torch-adafactor"
#      class: torch.optim.Adafactor
#        (/home/alex/SimpleTuner/simpletuner/helpers/training/optimizer_param.py:153-162,
#         math in torch/optim/_adafactor.py:330-416, torch 2.10.0)
#      defaults SimpleTuner passes: beta2_decay=-0.8, eps=(None, 1e-3), d=1.0,
#      weight_decay=0.0; lr comes from args (explicit, no relative-step mode —
#      torch's Adafactor has none; rho_t = min(lr, 1/sqrt(t)) internally).
#      NOTE: SimpleTuner's old transformers-Adafactor key "adafactor" was
#      REMOVED (optimizer_param.py:879 deprecation map).
#
#  * adamw_schedulefree -> SimpleTuner key "adamw_schedulefree"
#      class: AdamWScheduleFreeKahan — SimpleTuner IN-HOUSE, NOT facebookresearch
#        (/home/alex/SimpleTuner/simpletuner/helpers/training/optimizers/
#         adamw_schedulefree/__init__.py:10-145, imported at optimizer_param.py:20,
#         registered with override_lr_scheduler/is_schedulefree/can_warmup at
#         optimizer_param.py:249-259)
#      defaults: betas=(0.9,0.999), weight_decay=1e-2, eps=1e-8;
#      warmup_steps := args.lr_warmup_steps (optimizer_param.py:1114-1116).
#
# Output: /tmp/optimizer_oracle.safetensors consumed by
# serenitymojo/training/tests/optimizer_parity.mojo (20 steps, F32, CPU).
#
# Run: /home/alex/SimpleTuner/.venv/bin/python gen_optimizer_oracle.py

import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, "/home/alex/SimpleTuner")
from simpletuner.helpers.training.optimizers.adamw_schedulefree import (  # noqa: E402
    AdamWScheduleFreeKahan,
)

STEPS = 20
R_A, C_A = 16, 64   # LoRA-A-like
R_B, C_B = 64, 16   # LoRA-B-like

torch.manual_seed(1234)
p0_a = torch.randn(R_A, C_A, dtype=torch.float32) * 0.05
p0_b = torch.randn(R_B, C_B, dtype=torch.float32) * 0.05
grads_a = torch.randn(STEPS, R_A, C_A, dtype=torch.float32) * 0.02
grads_b = torch.randn(STEPS, R_B, C_B, dtype=torch.float32) * 0.02

out = {
    "p0_a": p0_a.clone(),
    "p0_b": p0_b.clone(),
    "grads_a": grads_a.clone(),
    "grads_b": grads_b.clone(),
}


def run(opt_name, make_opt, post=None):
    pa = p0_a.clone().requires_grad_(False)
    pb = p0_b.clone().requires_grad_(False)
    pa = torch.nn.Parameter(pa)
    pb = torch.nn.Parameter(pb)
    opt = make_opt([pa, pb])
    if hasattr(opt, "train"):
        try:
            opt.train()
        except Exception:
            pass
    for t in range(STEPS):
        # grads are CLONED per step: AdamWScheduleFreeKahan mutates p.grad
        # in place (grad.add_(kahan_comp), __init__.py:117).
        pa.grad = grads_a[t].clone()
        pb.grad = grads_b[t].clone()
        opt.step()
    out[f"{opt_name}_pa"] = pa.detach().clone()
    out[f"{opt_name}_pb"] = pb.detach().clone()
    if post is not None:
        post(opt, pa, pb)
    return opt


# ── 1. torch-adafactor, SimpleTuner default settings, lr=1e-2 ───────────────
run(
    "ada",
    lambda ps: torch.optim.Adafactor(
        ps, lr=1e-2, beta2_decay=-0.8, eps=(None, 1e-3), d=1.0, weight_decay=0.0
    ),
)

# ── 2. torch-adafactor with weight_decay=0.01 (stepweight-decay branch) ─────
run(
    "ada_wd",
    lambda ps: torch.optim.Adafactor(
        ps, lr=1e-2, beta2_decay=-0.8, eps=(None, 1e-3), d=1.0, weight_decay=0.01
    ),
)


# ── 3. adamw_schedulefree (AdamWScheduleFreeKahan), warmup_steps=5 ──────────
def sf_post(opt, pa, pb):
    # eval() is what SimpleTuner runs before save; verify (and record) that in
    # THIS reference it is a no-op: state["z"] is never created
    # (adamw_schedulefree/__init__.py:47-53 init has no "z"; eval() guards on
    # 'if "z" in state', :62), so train/eval weights are identical.
    pa_train = pa.detach().clone()
    pb_train = pb.detach().clone()
    opt.eval()
    assert torch.equal(pa_train, pa.detach()), "eval() changed pa — z exists?!"
    assert torch.equal(pb_train, pb.detach()), "eval() changed pb — z exists?!"
    print("schedulefree eval() confirmed no-op (no z state in reference)")
    out["sf_eval_pa"] = pa.detach().clone()
    out["sf_eval_pb"] = pb.detach().clone()
    # moments for debugging
    out["sf_m_a"] = opt.state[pa]["exp_avg"].clone()
    out["sf_v_a"] = opt.state[pa]["exp_avg_sq"].clone()
    out["sf_kahan_a"] = opt.state[pa]["kahan_comp"].clone()


run(
    "sf",
    lambda ps: AdamWScheduleFreeKahan(
        ps,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        warmup_steps=5,
        kahan_sum=True,
    ),
    post=sf_post,
)

print("kahan_comp max |.| after 20 steps:", out["sf_kahan_a"].abs().max().item())

save_file(out, "/tmp/optimizer_oracle.safetensors")
print("wrote /tmp/optimizer_oracle.safetensors")
for k, v in out.items():
    print(f"  {k}: {tuple(v.shape)} {v.dtype}")

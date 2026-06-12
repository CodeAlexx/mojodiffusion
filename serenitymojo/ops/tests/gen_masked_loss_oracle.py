# gen_masked_loss_oracle.py — torch oracle for the T1.E masked-loss fns in
# ops/loss_fns.mojo (TIER1_PARITY_CAMPAIGN_2026-06-11.md phase T1.E).
#
# CITED REFERENCE MATH (ported semantics):
# * SimpleTuner masked loss: per-element loss tensor (reduction='none',
#   simpletuner/helpers/models/common.py:4564 / :4629-4635), then
#   `loss = loss * mask_image` BEFORE reduction (common.py:4694), then
#   per-sample mean + batch mean (common.py:4704-4707). SimpleTuner has NO
#   unmasked_weight / normalize lever — that path is weights == mask.
# * unmasked_weight + normalize_masked_area_loss semantics (the UI keys):
#   OneTrainer modules/util/loss/masked_loss.py:11-18 —
#       clamped_mask = torch.clamp(mask, unmasked_weight, 1)
#       losses *= clamped_mask
#       if normalize: losses /= clamped_mask.mean(dim=(1,2,3), keepdim=True)
#   (already 1:1-ported at serenity-trainer src/.../util/loss/masked_loss.mojo).
#   For a single flattened sample the (1,2,3) keepdim mean == the global mask
#   mean, which is what the host per-sample list math uses.
# NOTE: for BINARY masks clamp(m, uw, 1) == m*1 + (1-m)*uw (the lerp form);
# they differ only on soft mask interiors — we port the clamp (OneTrainer/
# serenity-trainer) form since those are the key names being wired.
#
# Dumps pred/target/masks + expected loss+grad per case to
# /tmp/masked_loss_oracle.safetensors. Run:
#   /home/alex/EriDiffusion/.venv_cache/bin/python \
#       serenitymojo/ops/tests/gen_masked_loss_oracle.py
#
# Gate: serenitymojo/ops/tests/masked_loss_parity.mojo
# (PASS iff |loss diff| <= 1e-6 rel and grad cos >= 0.999999 per case).
import torch
from safetensors.torch import save_file

torch.manual_seed(20260611)
N = 4096

pred = torch.randn(N, dtype=torch.float32)
target = torch.randn(N, dtype=torch.float32)
mask_soft = torch.rand(N, dtype=torch.float32)            # soft mask in [0,1]
mask_bin = (torch.rand(N) > 0.6).to(torch.float32)        # binary 0/1 mask

out = {
    "pred": pred,
    "target": target,
    "mask_soft": mask_soft,
    "mask_bin": mask_bin,
}


def masked_case(elem_fn, mask, uw, norm):
    """OneTrainer masked_loss.py:11-18 (single flattened sample) followed by
    SimpleTuner's mean reduction (common.py:4704-4707); grads via autograd.
    uw=0, norm=False degenerates to SimpleTuner common.py:4694 loss*mask."""
    p = pred.clone().requires_grad_(True)
    le = elem_fn(p)                                  # per-element, no reduction
    cm = torch.clamp(mask, uw, 1)                    # masked_loss.py:11
    le = le * cm                                     # masked_loss.py:13 / ST :4694
    if norm:
        le = le / cm.mean()                          # masked_loss.py:15-16
    loss = le.mean()                                 # common.py:4704-4707
    loss.backward()
    return loss.detach().reshape(1), p.grad.detach().clone()


def mse_elem(p):
    return (p - target) ** 2                         # common.py:4564 (FLOW L2)


def huber_elem(delta):
    return lambda p: torch.nn.functional.huber_loss(
        p, target, reduction="none", delta=delta
    )


def sl1_elem(beta):
    return lambda p: torch.nn.functional.smooth_l1_loss(
        p, target, reduction="none", beta=beta
    )


CASES = {
    # pure SimpleTuner semantics: weights == mask (uw=0), no normalize
    "st_mse_soft": (mse_elem, mask_soft, 0.0, False),
    # OneTrainer lever cases (the UI keys)
    "ot_mse_bin": (mse_elem, mask_bin, 0.1, False),
    "ot_mse_bin_norm": (mse_elem, mask_bin, 0.1, True),
    "ot_huber_d01_bin_norm": (huber_elem(0.1), mask_bin, 0.1, True),
    "ot_huber_d1_soft": (huber_elem(1.0), mask_soft, 0.2, False),
    "ot_sl1_b05_bin_norm": (sl1_elem(0.5), mask_bin, 0.1, True),
}
for tag, (fn, mask, uw, norm) in CASES.items():
    out[f"{tag}_loss"], out[f"{tag}_grad"] = masked_case(fn, mask, uw, norm)

# mask_weights parity: torch.clamp(mask, uw, 1) on the soft mask, uw=0.2
out["clamp_soft_uw02"] = torch.clamp(mask_soft, 0.2, 1)

# levers-level composed case: ot_huber_d01_bin_norm additionally scaled by the
# T1.A flow min-SNR weight w = min(SNR,γ)/SNR, SNR = ((1-σ)/σ)^2 (ε-style,
# common.py:4660-4672 non-v-pred divisor), σ=0.2 γ=5 → SNR=16, w=0.3125.
SIGMA, GAMMA = 0.2, 5.0
snr = torch.tensor(((1.0 - SIGMA) / SIGMA) ** 2, dtype=torch.float32)
w_snr = torch.minimum(snr, torch.tensor(GAMMA)) / snr
out["levers_snr_sigma"] = torch.tensor([SIGMA], dtype=torch.float32)
out["levers_snr_gamma"] = torch.tensor([GAMMA], dtype=torch.float32)
out["levers_huber_snr_loss"] = out["ot_huber_d01_bin_norm_loss"] * w_snr
out["levers_huber_snr_grad"] = out["ot_huber_d01_bin_norm_grad"] * w_snr

save_file(out, "/tmp/masked_loss_oracle.safetensors")
for k, v in out.items():
    if v.numel() == 1:
        print(f"{k} = {v.item():.9g}")
print("wrote /tmp/masked_loss_oracle.safetensors")

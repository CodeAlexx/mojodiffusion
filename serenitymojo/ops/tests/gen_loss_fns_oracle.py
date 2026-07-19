# gen_loss_fns_oracle.py — torch oracle for ops/loss_fns.mojo (phase T1.A).
#
# Dumps pred/target/expected loss+grad for mse / huber(δ) / smooth_l1(β) at
# n=4096 random values, plus min-SNR-γ expected weights over 4096 sigmas, to
# /tmp/loss_fns_oracle.safetensors. Run:
#   /home/alex/EriDiffusion/.venv_cache/bin/python \
#       serenitymojo/ops/tests/gen_loss_fns_oracle.py
#
# Gate: serenitymojo/ops/tests/loss_fns_parity.mojo
# (PASS iff |loss diff| <= 1e-6 rel and grad cos >= 0.999999).
import torch
from safetensors.torch import save_file

torch.manual_seed(20260611)
N = 4096

pred = torch.randn(N, dtype=torch.float32)
target = torch.randn(N, dtype=torch.float32)

out = {"pred": pred, "target": target}


def loss_and_grad(fn):
    p = pred.clone().requires_grad_(True)
    loss = fn(p)
    loss.backward()
    return loss.detach().reshape(1), p.grad.detach().clone()


# MSE (the existing trainers' math — oracle baseline)
out["mse_loss"], out["mse_grad"] = loss_and_grad(
    lambda p: torch.nn.functional.mse_loss(p, target, reduction="mean")
)

# Huber, torch semantics, two deltas (1.0 = torch default; 0.1 = SimpleTuner
# huber_c default scale) so both branches of the piecewise form are exercised.
for tag, delta in (("huber_d1", 1.0), ("huber_d01", 0.1)):
    out[f"{tag}_loss"], out[f"{tag}_grad"] = loss_and_grad(
        lambda p, d=delta: torch.nn.functional.huber_loss(
            p, target, reduction="mean", delta=d
        )
    )

# Smooth-L1, torch semantics, two betas.
for tag, beta in (("sl1_b1", 1.0), ("sl1_b05", 0.5)):
    out[f"{tag}_loss"], out[f"{tag}_grad"] = loss_and_grad(
        lambda p, b=beta: torch.nn.functional.smooth_l1_loss(
            p, target, reduction="mean", beta=b
        )
    )

# min-SNR-γ weights for flow-match sigma (SimpleTuner formula, see
# ops/loss_fns.mojo header: min_snr_gamma.py:40 snr=(alpha/sigma)^2 with
# flow-match alpha=1-σ, sigma=σ; common.py:4660-4672 w=min(snr,γ)/snr).
GAMMA = 5.0
sigmas = torch.rand(N, dtype=torch.float32) * 0.998 + 0.001  # (0.001, 0.999)
snr = ((1.0 - sigmas) / sigmas) ** 2
w = torch.minimum(snr, torch.full_like(snr, GAMMA)) / snr
out["minsnr_sigmas"] = sigmas
out["minsnr_gamma"] = torch.tensor([GAMMA], dtype=torch.float32)
out["minsnr_weights"] = w

save_file(out, "/tmp/loss_fns_oracle.safetensors")
for k, v in out.items():
    if v.numel() == 1:
        print(f"{k} = {v.item():.9g}")
print("wrote /tmp/loss_fns_oracle.safetensors")

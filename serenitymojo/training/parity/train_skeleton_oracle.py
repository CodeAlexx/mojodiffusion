# train_skeleton_oracle.py — PyTorch reference for the Mojo walking-skeleton.
#
# Mirrors serenitymojo/training/parity/train_skeleton.mojo EXACTLY:
#   * same tiny 2-layer MLP  y = linear2(silu(linear1(X)))
#   * same deterministic init (a closed-form fill both sides reproduce)
#   * same fixed input X and target T
#   * same AdamW hyperparameters (decoupled WD)
#   * same number of steps
# Emits the loss trajectory (every 10 steps) so the Mojo file's curve can be
# compared (cosine of the loss-vs-step vectors must be >= 0.99).
#
# Init is a deterministic sinusoidal fill (NOT torch.randn) so the Mojo side
# can produce byte-for-byte the same starting weights without porting an RNG:
#   fill(i) = scale * sin(0.1*i + phase)
# Run:
#   /home/alex/serenityflow-v2/.venv/bin/python \
#     serenitymojo/training/parity/train_skeleton_oracle.py

import math
import torch

torch.set_printoptions(precision=8)

B, IN, HID, OUT = 8, 4, 8, 3
STEPS = 200
LR = 1e-2
BETA1, BETA2, EPS, WD = 0.9, 0.999, 1e-8, 0.0  # WD=0 (matches Mojo run)


def fill(n, scale, phase):
    # Deterministic fill mirrored exactly on the Mojo side.
    return [scale * math.sin(0.1 * i + phase) for i in range(n)]


def make(rows, cols, scale, phase):
    flat = fill(rows * cols, scale, phase)
    return torch.tensor(flat, dtype=torch.float32).reshape(rows, cols)


def make_vec(n, scale, phase):
    return torch.tensor(fill(n, scale, phase), dtype=torch.float32)


# Fixed data (same fills the Mojo side uses).
X = make(B, IN, 1.0, 0.0)          # [B, IN]
T = make(B, OUT, 0.7, 1.3)         # [B, OUT]

# Params: weight [out, in] (torch nn.Linear convention, matches Mojo linear).
W1 = make(HID, IN, 0.5, 0.5).clone().requires_grad_(True)   # [HID, IN]
b1 = make_vec(HID, 0.1, 0.2).clone().requires_grad_(True)   # [HID]
W2 = make(OUT, HID, 0.5, 0.9).clone().requires_grad_(True)  # [OUT, HID]
b2 = make_vec(OUT, 0.1, 0.4).clone().requires_grad_(True)   # [OUT]

opt = torch.optim.AdamW(
    [W1, b1, W2, b2], lr=LR, betas=(BETA1, BETA2), eps=EPS, weight_decay=WD
)

losses = []
for step in range(STEPS):
    opt.zero_grad()
    h1 = X @ W1.t() + b1                 # [B, HID]
    a = torch.nn.functional.silu(h1)    # [B, HID]
    y = a @ W2.t() + b2                 # [B, OUT]
    loss = ((y - T) ** 2).mean()
    loss.backward()
    opt.step()
    losses.append(loss.item())
    if step % 10 == 0 or step == STEPS - 1:
        print(f"step {step:4d}  loss {loss.item():.8f}")

print("INITIAL", losses[0])
print("FINAL", losses[-1])
print("RATIO", losses[-1] / losses[0])
# Print the every-10 trajectory as a flat list for cos comparison.
traj = [losses[s] for s in range(0, STEPS, 10)]
print("TRAJ10", " ".join(f"{v:.8f}" for v in traj))

# serenitymojo/training/parity/stack_train_oracle.py
#
# Torch oracle for stack_train_parity.mojo. Builds the IDENTICAL stacked-DiT-block
# training run (N blocks, deterministic sinusoidal init, AdamW, MSE) and emits:
#   * the loss trajectory (every 10 steps + final)  -> cross-check descent shape
#   * the FIRST-STEP gradient of the DEEPEST block's params (block 0, the one
#     furthest from the loss — its grad rides through the ENTIRE inter-block
#     grad chain) -> the bonus gate: proves the d_x->d_y handoff is correct,
#     not just that the loss happens to fall.
#
# Block (matches the Mojo forward exactly):
#   h1   = rms_norm(x, g1)
#   q,k,v= linear(h1, Wq/Wk/Wv)              [M,D]
#   attn = sdpa([1,M,H,Dh], scale)           [M,D]
#   ao   = linear(attn, Wo)                   [M,D]
#   r1   = x + ao                            (residual #1)
#   h2   = rms_norm(r1, g2)
#   gate = linear(h2, Wg) ; up = linear(h2, Wu)   [M,F]
#   act  = silu(gate)*up                      [M,F]
#   mlp  = linear(act, Wd)                    [M,D]
#   y    = r1 + mlp                          (residual #2)
# Stack: x -> block0 -> block1 -> ... -> blockN-1 -> mse(out, target)
#
# Deterministic fill v(i) = scale*sin(0.1*i + phase) — byte-identical to the Mojo
# _fill so the two runs start from the same numbers.
#
# Run: /home/alex/serenityflow-v2/.venv/bin/python \
#        serenitymojo/training/parity/stack_train_oracle.py

import math
import torch

torch.set_default_dtype(torch.float64)  # match: do the math in high precision,
# the Mojo side is F32 but cos-of-trajectory + cos-of-grad are robust to that.

M = 4
D = 8
H = 2
Dh = 4
FF = 16
NBLOCKS = 3
STEPS = 80
EPS = 1e-6
SCALE = 0.5  # sdpa scale (matches Mojo SCALE)

LR = 1e-2
BETA1 = 0.9
BETA2 = 0.999
ADAM_EPS = 1e-8
WD = 0.0


def fill(n, scale, phase):
    return torch.tensor([scale * math.sin(0.1 * i + phase) for i in range(n)])


def rms_norm(x, g, eps):
    # x: [M, D], g: [D]
    ms = (x * x).mean(dim=-1, keepdim=True)
    return x / torch.sqrt(ms + eps) * g


def sdpa(q, k, v, scale):
    # q,k,v as [M, D] viewed as [H, M, Dh] (row-major [M,D] == [M,H,Dh]; per-head)
    qh = q.view(M, H, Dh).transpose(0, 1)  # [H, M, Dh]
    kh = k.view(M, H, Dh).transpose(0, 1)
    vh = v.view(M, H, Dh).transpose(0, 1)
    att = torch.matmul(qh, kh.transpose(-1, -2)) * scale  # [H, M, M]
    att = torch.softmax(att, dim=-1)
    out = torch.matmul(att, vh)  # [H, M, Dh]
    return out.transpose(0, 1).reshape(M, D)  # [M, D]


def block_forward(x, p):
    g1, g2, Wq, Wk, Wv, Wo, Wg, Wu, Wd = p
    h1 = rms_norm(x, g1, EPS)
    q = h1 @ Wq.t()
    k = h1 @ Wk.t()
    v = h1 @ Wv.t()
    attn = sdpa(q, k, v, SCALE)
    ao = attn @ Wo.t()
    r1 = x + ao
    h2 = rms_norm(r1, g2, EPS)
    gate = h2 @ Wg.t()
    up = h2 @ Wu.t()
    act = torch.nn.functional.silu(gate) * up
    mlp = act @ Wd.t()
    y = r1 + mlp
    return y


def make_block_params(bi):
    # Per-block deterministic init; block index shifts the phase so blocks differ.
    ph = 0.3 * bi
    g1 = fill(D, 1.0, 0.5 + ph).clone().requires_grad_(True)
    g2 = fill(D, 1.0, 0.9 + ph).clone().requires_grad_(True)
    Wq = fill(D * D, 0.3, 0.1 + ph).view(D, D).clone().requires_grad_(True)
    Wk = fill(D * D, 0.3, 0.4 + ph).view(D, D).clone().requires_grad_(True)
    Wv = fill(D * D, 0.3, 0.7 + ph).view(D, D).clone().requires_grad_(True)
    Wo = fill(D * D, 0.3, 1.0 + ph).view(D, D).clone().requires_grad_(True)
    Wg = fill(FF * D, 0.3, 1.3 + ph).view(FF, D).clone().requires_grad_(True)
    Wu = fill(FF * D, 0.3, 1.6 + ph).view(FF, D).clone().requires_grad_(True)
    Wd = fill(D * FF, 0.3, 1.9 + ph).view(D, FF).clone().requires_grad_(True)
    return [g1, g2, Wq, Wk, Wv, Wo, Wg, Wu, Wd]


def main():
    X = fill(M * D, 1.0, 0.0).view(M, D)
    T = fill(M * D, 0.7, 1.3).view(M, D)

    blocks = [make_block_params(bi) for bi in range(NBLOCKS)]
    all_params = [pp for blk in blocks for pp in blk]
    opt = torch.optim.AdamW(
        all_params, lr=LR, betas=(BETA1, BETA2), eps=ADAM_EPS, weight_decay=WD
    )

    losses = []
    first_step_grads = None
    pnames = ["g1", "g2", "Wq", "Wk", "Wv", "Wo", "Wg", "Wu", "Wd"]

    for step in range(STEPS):
        opt.zero_grad()
        h = X
        for blk in blocks:
            h = block_forward(h, blk)
        loss = ((h - T) ** 2).mean()
        loss.backward()

        if step == 0:
            # Deepest block = block 0 (furthest from loss; its grad rode the
            # whole inter-block d_x->d_y chain). Capture its param grads.
            first_step_grads = {
                pnames[i]: blocks[0][i].grad.detach().flatten().tolist()
                for i in range(len(pnames))
            }

        opt.step()
        losses.append(float(loss.detach()))

    # ---- emit ----
    print("=== STACK TRAIN ORACLE (torch) ===")
    print("NBLOCKS", NBLOCKS, "STEPS", STEPS, "M", M, "D", D, "H", H, "Dh", Dh, "FF", FF)
    print("INITIAL", losses[0])
    print("FINAL", losses[-1])
    print("RATIO", losses[-1] / losses[0])
    print("--- trajectory every 10 ---")
    for s in range(0, STEPS, 10):
        print("step", s, "loss", losses[s])

    # Write a ref file (space-separated, archival) AND emit paste-ready comma
    # forms for the Mojo REF_* literals (stack_train_parity.mojo embeds these —
    # no file I/O / string parsing on the Mojo side, matching the proven
    # block_composed_parity.mojo pattern).
    with open(
        "/home/alex/mojodiffusion/serenitymojo/training/parity/stack_train_ref.txt",
        "w",
    ) as f:
        f.write("# stack_train oracle ref (torch). deepest=block0 first-step grads.\n")
        f.write("INITIAL %.10g\n" % losses[0])
        f.write("FINAL %.10g\n" % losses[-1])
        f.write("RATIO %.10g\n" % (losses[-1] / losses[0]))
        f.write("TRAJ " + " ".join("%.10g" % losses[s] for s in range(0, STEPS, 10)) + "\n")
        for name in pnames:
            vals = first_step_grads[name]
            f.write("GRAD_%s " % name + " ".join("%.10g" % x for x in vals) + "\n")
    print("wrote stack_train_ref.txt")

    print("")
    print("=== PASTE-READY Mojo REF_* literals (block0, first step) ===")
    for name, refname in [
        ("Wq", "REF_DWQ"), ("Wo", "REF_DWO"), ("Wd", "REF_DWD"),
        ("g1", "REF_DG1"), ("g2", "REF_DG2"),
    ]:
        vals = first_step_grads[name]
        body = ", ".join("%.10g" % x for x in vals)
        print("    var %s = List[Float32](%s)" % (refname, body))


if __name__ == "__main__":
    main()

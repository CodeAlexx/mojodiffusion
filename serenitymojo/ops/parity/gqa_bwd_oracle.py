#!/usr/bin/env python3
# Oracle for ops/gqa_backward.mojo (GQA repeat_kv fwd + bwd).
# repeat_kv: BSHD [1,S,Hkv,Dh] -> [1,S,H,Dh], dst head h reads src h//n_rep.
# Dumps non-degenerate sinusoidal src + upstream d_dst + torch-autograd d_src.
import os, struct
import torch

REF = os.path.dirname(os.path.abspath(__file__)) + "/"
S, HKV, NREP, DH = 5, 8, 2, 6
H = HKV * NREP


def w(name, t):
    a = t.detach().contiguous().float().cpu().numpy().ravel()
    with open(REF + name + ".bin", "wb") as f:
        f.write(struct.pack("<%df" % a.size, *a.tolist()))


def sinusoidal(n, seed):
    i = torch.arange(n, dtype=torch.float64)
    return (torch.sin(0.7 * i + seed) * 0.5 + 0.3 * torch.cos(0.31 * i)).float()


def repeat_kv(x, n_rep):
    # x [1,S,Hkv,Dh] -> [1,S,H,Dh], pytorch style (head h reads h//n_rep)
    b, s, hkv, dh = x.shape
    return (
        x[:, :, :, None, :]
        .expand(b, s, hkv, n_rep, dh)
        .reshape(b, s, hkv * n_rep, dh)
    )


def main():
    torch.manual_seed(0)
    src = sinusoidal(S * HKV * DH, 1.0).reshape(1, S, HKV, DH).clone()
    src.requires_grad_(True)
    dst = repeat_kv(src, NREP)  # [1,S,H,DH]
    d_dst = sinusoidal(S * H * DH, 2.0).reshape(1, S, H, DH)
    dst.backward(d_dst)
    w("in_src", src)
    w("in_d_dst", d_dst)
    w("ref_dst", dst)
    w("ref_d_src", src.grad)
    print("gqa oracle done S=%d HKV=%d NREP=%d DH=%d" % (S, HKV, NREP, DH))


if __name__ == "__main__":
    main()

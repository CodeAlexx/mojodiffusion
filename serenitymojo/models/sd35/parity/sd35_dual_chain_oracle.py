#!/usr/bin/env python3
# serenitymojo/models/sd35/parity/sd35_dual_chain_oracle.py
#
# Torch oracle for a 2-BLOCK CHAIN: block A = DUAL-attention joint block, block B
# = STANDARD joint block, chained (A's (ctx_out,x_out) feed B). This is the
# minimal proof that the stack wiring threads a dual block into a following
# standard block correctly in BOTH forward and backward — the dispatch the
# offload stack loop performs for blocks 0-12 (dual) then 13+ (standard).
# Both block maths are already gate-verified individually; this checks the plumbing.
#
# Dumps: final ctx_out/x_out, input grads d_ctx0/d_x0, and one weight grad from
# EACH block (A.attn2.wqkv, B.wqkv) to confirm grads reach both blocks. cos>=0.999.
#
# Run:
#   /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/models/sd35/parity/sd35_dual_chain_oracle.py

import math, struct, os, torch
torch.manual_seed(0)
DT = torch.float64
H, Dh = 24, 8
D = H * Dh           # 192
N_CTX, N_IMG = 3, 5
MLP = 32
EPS = 1e-6
SCALE = 1.0 / math.sqrt(Dh)
REF_DIR = os.path.dirname(os.path.abspath(__file__))


def fill(n, a, b, c): return torch.tensor([math.sin(a*i+b)*c for i in range(n)], dtype=DT)
def fillc(n, a, b, c): return torch.tensor([math.cos(a*i+b)*c for i in range(n)], dtype=DT)
def t2(n, m, a, b, c): return fill(n*m, a, b, c).reshape(n, m)
def ln(x):
    return (x - x.mean(-1, keepdim=True)) / torch.sqrt(x.var(-1, unbiased=False, keepdim=True) + EPS)
def mod(x, s, sh): return (1.0+s)*x + sh
def gr(s, g, y): return s + g*y
def rms(x, w):
    return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w
def gelu(x): return 0.5*x*(1.0+torch.tanh(math.sqrt(2.0/math.pi)*(x+0.044715*x.pow(3))))
def sdpa(q, k, v):
    qh, kh, vh = q.permute(0,2,1,3), k.permute(0,2,1,3), v.permute(0,2,1,3)
    return (torch.softmax((qh@kh.transpose(-1,-2))*SCALE, -1)@vh).permute(0,2,1,3)
def hd(t, N): return t.reshape(1, N, H, Dh)


def mkstream(seed):
    g = torch.Generator().manual_seed(seed)
    def r(*s): return torch.randn(*s, generator=g, dtype=torch.float32).to(DT)
    w = dict(wqkv=r(3*D,D)*0.05, bqkv=r(3*D)*0.05, wproj=r(D,D)*0.05, bproj=r(D)*0.05,
             wfc1=r(MLP,D)*0.05, bfc1=r(MLP)*0.05, wfc2=r(D,MLP)*0.05, bfc2=r(D)*0.05,
             q_norm=r(Dh)*0.1+1.0, k_norm=r(Dh)*0.1+1.0)
    for v in w.values(): v.requires_grad_(True)
    return w
def mkattn2(seed):
    g = torch.Generator().manual_seed(seed)
    def r(*s): return torch.randn(*s, generator=g, dtype=torch.float32).to(DT)
    w = dict(wqkv=r(3*D,D)*0.05, bqkv=r(3*D)*0.05, wproj=r(D,D)*0.05, bproj=r(D)*0.05,
             q_norm=r(Dh)*0.1+1.0, k_norm=r(Dh)*0.1+1.0)
    for v in w.values(): v.requires_grad_(True)
    return w
def mkmod(off):
    return dict(shift_msa=fill(D,.013,.1+off,.3).requires_grad_(True), scale_msa=fillc(D,.017,.2+off,.2).requires_grad_(True),
                gate_msa=fill(D,.011,.3+off,.4).requires_grad_(True), shift_mlp=fillc(D,.019,.4+off,.3).requires_grad_(True),
                scale_mlp=fill(D,.015,.5+off,.2).requires_grad_(True), gate_mlp=fillc(D,.012,.6+off,.4).requires_grad_(True))
def mkmod2(off):
    return dict(shift_msa2=fill(D,.014,.15+off,.3).requires_grad_(True), scale_msa2=fillc(D,.016,.25+off,.2).requires_grad_(True),
                gate_msa2=fill(D,.012,.35+off,.4).requires_grad_(True))
def qkv3(q, N):
    return hd(q[:,0:D],N), hd(q[:,D:2*D],N), hd(q[:,2*D:3*D],N)


def std_block(context, x, cw, xw, cm, xm):
    cn = mod(ln(context), cm["scale_msa"], cm["shift_msa"])
    cq,ck,cv = qkv3(cn@cw["wqkv"].T+cw["bqkv"], N_CTX); cq=rms(cq,cw["q_norm"]); ck=rms(ck,cw["k_norm"])
    xn = mod(ln(x), xm["scale_msa"], xm["shift_msa"])
    xq,xk,xv = qkv3(xn@xw["wqkv"].T+xw["bqkv"], N_IMG); xq=rms(xq,xw["q_norm"]); xk=rms(xk,xw["k_norm"])
    att = sdpa(torch.cat([cq,xq],1), torch.cat([ck,xk],1), torch.cat([cv,xv],1))
    catt = att[:,0:N_CTX].reshape(N_CTX,D); xatt = att[:,N_CTX:].reshape(N_IMG,D)
    def post(s, a, w, m, N):
        ar = gr(s, m["gate_msa"], a@w["wproj"].T+w["bproj"])
        mi = mod(ln(ar), m["scale_mlp"], m["shift_mlp"])
        return gr(ar, m["gate_mlp"], gelu(mi@w["wfc1"].T+w["bfc1"])@w["wfc2"].T+w["bfc2"])
    return post(context, catt, cw, cm, N_CTX), post(x, xatt, xw, xm, N_IMG)


def dual_block(context, x, cw, xw, a2, cm, xm, xm2):
    cn = mod(ln(context), cm["scale_msa"], cm["shift_msa"])
    cq,ck,cv = qkv3(cn@cw["wqkv"].T+cw["bqkv"], N_CTX); cq=rms(cq,cw["q_norm"]); ck=rms(ck,cw["k_norm"])
    lnx = ln(x)
    xn = mod(lnx, xm["scale_msa"], xm["shift_msa"])
    xn2 = mod(lnx, xm2["scale_msa2"], xm2["shift_msa2"])
    xq,xk,xv = qkv3(xn@xw["wqkv"].T+xw["bqkv"], N_IMG); xq=rms(xq,xw["q_norm"]); xk=rms(xk,xw["k_norm"])
    att = sdpa(torch.cat([cq,xq],1), torch.cat([ck,xk],1), torch.cat([cv,xv],1))
    catt = att[:,0:N_CTX].reshape(N_CTX,D); xatt = att[:,N_CTX:].reshape(N_IMG,D)
    xhid = gr(x, xm["gate_msa"], xatt@xw["wproj"].T+xw["bproj"])
    a2q,a2k,a2v = qkv3(xn2@a2["wqkv"].T+a2["bqkv"], N_IMG); a2q=rms(a2q,a2["q_norm"]); a2k=rms(a2k,a2["k_norm"])
    a2att = sdpa(a2q,a2k,a2v).reshape(N_IMG,D)
    xhid = gr(xhid, xm2["gate_msa2"], a2att@a2["wproj"].T+a2["bproj"])
    xmi = mod(ln(xhid), xm["scale_mlp"], xm["shift_mlp"])
    xout = gr(xhid, xm["gate_mlp"], gelu(xmi@xw["wfc1"].T+xw["bfc1"])@xw["wfc2"].T+xw["bfc2"])
    cr = gr(context, cm["gate_msa"], catt@cw["wproj"].T+cw["bproj"])
    cmi = mod(ln(cr), cm["scale_mlp"], cm["shift_mlp"])
    cout = gr(cr, cm["gate_mlp"], gelu(cmi@cw["wfc1"].T+cw["bfc1"])@cw["wfc2"].T+cw["bfc2"])
    return cout, xout


def W(name, t):
    flat = t.detach().reshape(-1).to(torch.float32).numpy()
    with open(os.path.join(REF_DIR, name+".bin"), "wb") as f:
        f.write(struct.pack("<%df" % flat.size, *flat.tolist()))
    print("wrote", name, tuple(t.shape))


def main():
    ctx0 = t2(N_CTX, D, 0.021, 0.05, 0.5).requires_grad_(True)
    x0 = t2(N_IMG, D, 0.023, 0.07, 0.5).requires_grad_(True)
    # block A = DUAL
    Acw, Axw, Aa2, Acm, Axm, Axm2 = mkstream(1), mkstream(2), mkattn2(3), mkmod(0.0), mkmod(1.0), mkmod2(1.0)
    # block B = STANDARD
    Bcw, Bxw, Bcm, Bxm = mkstream(11), mkstream(12), mkmod(0.5), mkmod(1.5)

    ctx1, x1 = dual_block(ctx0, x0, Acw, Axw, Aa2, Acm, Axm, Axm2)
    ctx2, x2 = std_block(ctx1, x1, Bcw, Bxw, Bcm, Bxm)

    d_ctx = t2(N_CTX, D, 0.027, 0.11, 0.05)
    d_x = t2(N_IMG, D, 0.029, 0.13, 0.05)
    loss = (ctx2*d_ctx).sum() + (x2*d_x).sum()
    loss.backward()

    # references the Mojo chain gate compares
    W("chain_ctx_out", ctx2); W("chain_x_out", x2)
    W("chain_d_ctx0", ctx0.grad); W("chain_d_x0", x0.grad)
    W("chain_A_a2_d_wqkv", Aa2["wqkv"].grad)   # dual block's attn2 qkv grad
    W("chain_B_x_d_wqkv", Bxw["wqkv"].grad)    # standard block's x qkv grad
    # inputs the Mojo gate reconstructs
    W("chain_ctx0", ctx0); W("chain_x0", x0); W("chain_d_ctx_up", d_ctx); W("chain_d_x_up", d_x)
    for nm, w in [("Acw", Acw), ("Axw", Axw)]:
        for kk in ["wqkv","bqkv","wproj","bproj","wfc1","bfc1","wfc2","bfc2","q_norm","k_norm"]:
            W("chain_%s_%s" % (nm, kk), w[kk])
    for kk in ["wqkv","bqkv","wproj","bproj","q_norm","k_norm"]:
        W("chain_Aa2_%s" % kk, Aa2[kk])
    for nm, m in [("Acm", Acm), ("Axm", Axm)]:
        for kk in ["shift_msa","scale_msa","gate_msa","shift_mlp","scale_mlp","gate_mlp"]:
            W("chain_%s_%s" % (nm, kk), m[kk])
    for kk in ["shift_msa2","scale_msa2","gate_msa2"]:
        W("chain_Axm2_%s" % kk, Axm2[kk])
    for nm, w in [("Bcw", Bcw), ("Bxw", Bxw)]:
        for kk in ["wqkv","bqkv","wproj","bproj","wfc1","bfc1","wfc2","bfc2","q_norm","k_norm"]:
            W("chain_%s_%s" % (nm, kk), w[kk])
    for nm, m in [("Bcm", Bcm), ("Bxm", Bxm)]:
        for kk in ["shift_msa","scale_msa","gate_msa","shift_mlp","scale_mlp","gate_mlp"]:
            W("chain_%s_%s" % (nm, kk), m[kk])
    print("chain loss =", float(loss))
    print("DONE")


if __name__ == "__main__":
    main()

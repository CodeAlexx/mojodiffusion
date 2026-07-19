#!/usr/bin/env python3
# Torch oracle for the SD3.5 context_pre_only FINAL block.
# x stream = standard joint block; context stream = qkv-only (AdaLayerNormContinuous
# norm: (1+scale)*LN(ctx)+shift), contributes K/V/Q to the joint attention with NO
# output. Dumps refs for sd35_ctxpre_parity.mojo at cos>=0.999.
import math, struct, os, torch
torch.manual_seed(0); DT = torch.float64
H, Dh = 24, 8; D = H*Dh; N_CTX, N_IMG = 3, 5; MLP = 32
EPS = 1e-6; SCALE = 1.0/math.sqrt(Dh)
REF = os.path.dirname(os.path.abspath(__file__))
def fill(n,a,b,c): return torch.tensor([math.sin(a*i+b)*c for i in range(n)],dtype=DT)
def fillc(n,a,b,c): return torch.tensor([math.cos(a*i+b)*c for i in range(n)],dtype=DT)
def t2(n,m,a,b,c): return fill(n*m,a,b,c).reshape(n,m)
def ln(x): return (x-x.mean(-1,keepdim=True))/torch.sqrt(x.var(-1,unbiased=False,keepdim=True)+EPS)
def mod(x,s,sh): return (1.0+s)*x+sh
def gr(s,g,y): return s+g*y
def rms(x,w): return x/torch.sqrt(x.pow(2).mean(-1,keepdim=True)+EPS)*w
def gelu(x): return 0.5*x*(1.0+torch.tanh(math.sqrt(2.0/math.pi)*(x+0.044715*x.pow(3))))
def sdpa(q,k,v):
    qh,kh,vh=q.permute(0,2,1,3),k.permute(0,2,1,3),v.permute(0,2,1,3)
    return (torch.softmax((qh@kh.transpose(-1,-2))*SCALE,-1)@vh).permute(0,2,1,3)
def hd(t,N): return t.reshape(1,N,H,Dh)
def qkv3(q,N): return hd(q[:,0:D],N),hd(q[:,D:2*D],N),hd(q[:,2*D:3*D],N)
def mkstream(seed):
    g=torch.Generator().manual_seed(seed)
    def r(*s): return torch.randn(*s,generator=g,dtype=torch.float32).to(DT)
    w=dict(wqkv=r(3*D,D)*.05,bqkv=r(3*D)*.05,wproj=r(D,D)*.05,bproj=r(D)*.05,
           wfc1=r(MLP,D)*.05,bfc1=r(MLP)*.05,wfc2=r(D,MLP)*.05,bfc2=r(D)*.05,
           q_norm=r(Dh)*.1+1.0,k_norm=r(Dh)*.1+1.0)
    for v in w.values(): v.requires_grad_(True)
    return w
def mkmod(off):
    return dict(shift_msa=fill(D,.013,.1+off,.3).requires_grad_(True),scale_msa=fillc(D,.017,.2+off,.2).requires_grad_(True),
                gate_msa=fill(D,.011,.3+off,.4).requires_grad_(True),shift_mlp=fillc(D,.019,.4+off,.3).requires_grad_(True),
                scale_mlp=fill(D,.015,.5+off,.2).requires_grad_(True),gate_mlp=fillc(D,.012,.6+off,.4).requires_grad_(True))
def W(name,t):
    flat=t.detach().reshape(-1).to(torch.float32).numpy()
    open(os.path.join(REF,name+".bin"),"wb").write(struct.pack("<%df"%flat.size,*flat.tolist()))
    print("wrote",name,tuple(t.shape))
def main():
    context=t2(N_CTX,D,.021,.05,.5).requires_grad_(True)
    x=t2(N_IMG,D,.023,.07,.5).requires_grad_(True)
    cw=dict(wqkv=mkstream(1)["wqkv"],bqkv=mkstream(1)["bqkv"],q_norm=mkstream(1)["q_norm"],k_norm=mkstream(1)["k_norm"])
    # ctx qkv-only weights (fresh leaf tensors)
    g=torch.Generator().manual_seed(5)
    def r(*s): return torch.randn(*s,generator=g,dtype=torch.float32).to(DT)
    cqkv_w=(r(3*D,D)*.05).requires_grad_(True); cqkv_b=(r(3*D)*.05).requires_grad_(True)
    cqn=(r(Dh)*.1+1.0).requires_grad_(True); ckn=(r(Dh)*.1+1.0).requires_grad_(True)
    ctx_scale=fillc(D,.02,.3,.2).requires_grad_(True); ctx_shift=fill(D,.018,.2,.3).requires_grad_(True)
    xw=mkstream(2); xm=mkmod(1.0)
    # ── ctx pre (qkv only) ──
    ctx_norm=mod(ln(context),ctx_scale,ctx_shift)
    cq,ck,cv=qkv3(ctx_norm@cqkv_w.T+cqkv_b,N_CTX); cq=rms(cq,cqn); ck=rms(ck,ckn)
    # ── x pre (standard) ──
    xn=mod(ln(x),xm["scale_msa"],xm["shift_msa"])
    xq,xk,xv=qkv3(xn@xw["wqkv"].T+xw["bqkv"],N_IMG); xq=rms(xq,xw["q_norm"]); xk=rms(xk,xw["k_norm"])
    # ── joint attn (ctx first) ; ctx output discarded ──
    att=sdpa(torch.cat([cq,xq],1),torch.cat([ck,xk],1),torch.cat([cv,xv],1))
    xatt=att[:,N_CTX:].reshape(N_IMG,D)
    # ── x post (standard) ──
    ar=gr(x,xm["gate_msa"],xatt@xw["wproj"].T+xw["bproj"])
    mi=mod(ln(ar),xm["scale_mlp"],xm["shift_mlp"])
    xout=gr(ar,xm["gate_mlp"],gelu(mi@xw["wfc1"].T+xw["bfc1"])@xw["wfc2"].T+xw["bfc2"])
    d_x=t2(N_IMG,D,.029,.13,.05)
    (xout*d_x).sum().backward()
    W("cp_x_out",xout); W("cp_d_x",x.grad); W("cp_d_ctx",context.grad)
    W("cp_x_d_wqkv",xw["wqkv"].grad); W("cp_x_d_wfc1",xw["wfc1"].grad)
    # inputs
    W("cp_context",context); W("cp_x",x)
    W("cp_cqkv_w",cqkv_w); W("cp_cqkv_b",cqkv_b); W("cp_cqn",cqn); W("cp_ckn",ckn)
    W("cp_ctx_scale",ctx_scale); W("cp_ctx_shift",ctx_shift)
    for kk in ["wqkv","bqkv","wproj","bproj","wfc1","bfc1","wfc2","bfc2","q_norm","k_norm"]:
        W("cp_xw_%s"%kk,xw[kk])
    for kk in ["shift_msa","scale_msa","gate_msa","shift_mlp","scale_mlp","gate_mlp"]:
        W("cp_xm_%s"%kk,xm[kk])
    W("cp_d_x_up",d_x)
    print("ctxpre loss =",float((xout*d_x).sum())); print("DONE")
if __name__=="__main__": main()

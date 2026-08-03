# Oracle for serenitymojo/pipeline/minimax_h3_torch_cpu_rng.mojo.
#
# CPU ONLY — never touches CUDA (the overnight chain owns the GPU). The
# keyframe posterior sample is drawn off `torch.Generator().manual_seed(42)`,
# which is a CPU MT19937 generator (encoders.py:297), so this is exactly the
# stream the Mojo port has to reproduce.
#
# Dumps, as a safetensors file the Mojo probe reads back:
#   uniform_<n>   torch.rand(n, generator=Generator().manual_seed(42))
#   normal_<n>    torch.rand->randn(n, generator=Generator().manual_seed(42))
#   normal_s<seed>_<n>  same at another seed, to prove seeding is right
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch
from safetensors.torch import save_file

OUT_DIR = "/home/alex/mojodiffusion/output/minimax_h3_keyframe"
os.makedirs(OUT_DIR, exist_ok=True)
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUT_DIR, "torch_cpu_rng_ref.safetensors")

tensors = {}
for n in (16, 17, 31, 1024, 37440):          # 37440 = 24*30*52, a real keyframe latent
    g = torch.Generator()
    g.manual_seed(42)
    tensors[f"uniform_{n}"] = torch.rand(n, generator=g, dtype=torch.float32)
    g = torch.Generator()
    g.manual_seed(42)
    tensors[f"normal_{n}"] = torch.randn(n, generator=g, dtype=torch.float32)

for seed in (0, 1, 12345):
    g = torch.Generator()
    g.manual_seed(seed)
    tensors[f"normal_s{seed}_1024"] = torch.randn(1024, generator=g, dtype=torch.float32)

save_file(tensors, out)
print("wrote", out)
for k in sorted(tensors):
    v = tensors[k]
    print(f"  {k:22s} n={v.numel():7d} first={v[0].item():+.9f} last={v[-1].item():+.9f}")
print("torch", torch.__version__, "cuda_initialized", torch.cuda.is_initialized())

# ── SEQUENTIAL draws off ONE generator ──────────────────────────────────────
# The shape a real FL2VA request has: two keyframe conditioning draws in packed
# order, then the target video noise, then the target audio noise — all off the
# request's own generator, in that order (packing.py:511-515,
# before_denoise.py:309-327). A tail-recompute draw (numel % 16 != 0) advances
# the stream by 16 extra words, so this also pins that bookkeeping.
seq = {}
g = torch.Generator(); g.manual_seed(7)
seq["seq_a"] = torch.randn(24 * 1 * 30 * 52, generator=g, dtype=torch.float32)
seq["seq_b"] = torch.randn(24 * 1 * 30 * 52, generator=g, dtype=torch.float32)
seq["seq_c"] = torch.randn(24 * 4 * 30 * 52, generator=g, dtype=torch.float32)
seq["seq_d"] = torch.randn(74 * 32, generator=g, dtype=torch.float32)
g = torch.Generator(); g.manual_seed(7)
seq["seqodd_a"] = torch.randn(31, generator=g, dtype=torch.float32)   # 31 % 16 != 0
seq["seqodd_b"] = torch.randn(1024, generator=g, dtype=torch.float32)
tensors.update(seq)
save_file(tensors, out)
print("re-wrote", out, "with", len(seq), "sequential-draw tensors")
for k in sorted(seq):
    print(f"  {k:10s} n={seq[k].numel():7d} first={seq[k][0].item():+.9f}")

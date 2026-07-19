# gen_lora_ema_oracle.py — torch oracle for training/lora_ema.mojo (phase T1.B).
#
# Replicates SimpleTuner EMAModel
# (/home/alex/SimpleTuner/simpletuner/helpers/training/ema.py) update semantics
# on random LoRA-shaped tensors for N=30 steps, F32 shadows over bf16 live
# params (the production config: bf16 host mirrors, F32 master shadows):
#   shadow init  = clone of initial params              (ema.py:123)
#   interval     = step % ema_update_interval == 0      (ema.py:29-37)
#   step counter = optimization_step := global_step     (ema.py:360-362)
#   decay        = get_decay non-warmup branch          (ema.py:311-331):
#                  s = max(0, step-update_after_step-1) (:318), <=0 -> 0.0
#                  (:320-321), (1+s)/(10+s) (:326), min(.,decay) (:328),
#                  max(.,min_decay) (:330)
#   update       target = p.to(shadow.dtype)            (ema.py:401-402)
#                shadow.sub_((1-decay)*(shadow-target)) (ema.py:369, :405)
#   export       shadow.to(param dtype)  [copy_to cast] (ema.py:454)
#
# Two states over the same param trajectory:
#   state1: decay=0.65 cap, min_decay=0.0, update_after_step=2, interval=1
#           (exercises the <=0 copy region, the (1+s)/(10+s) ramp AND the cap)
#   state2: decay=0.9, min_decay=0.2, update_after_step=0, interval=2
#           (exercises the min_decay floor AND the interval skip gate)
# decay/min_decay are wrapped through f32 to match Mojo's Float32 TrainConfig
# fields (Float64(Float32(x)) on the Mojo side).
#
# Output: /tmp/lora_ema_oracle.safetensors. Run:
#   /home/alex/EriDiffusion/.venv_cache/bin/python \
#       serenitymojo/training/tests/gen_lora_ema_oracle.py
#
# Gate: serenitymojo/training/tests/lora_ema_parity.mojo
# (PASS iff final shadows max rel diff <= 1e-6 per tensor; bf16 export exact).
import torch
from safetensors.torch import save_file

torch.manual_seed(20260611)
N = 30
# Two LoRA adapters: (a [rank,in], b [out,rank]) = (8x64, 96x8) and (4x32, 48x4)
SHAPES = [(8, 64), (96, 8), (4, 32), (48, 4)]


def f32(x):
    return float(torch.tensor(x, dtype=torch.float32))


CFG1 = dict(decay=f32(0.65), min_decay=f32(0.0), update_after_step=2, interval=1)
CFG2 = dict(decay=f32(0.9), min_decay=f32(0.2), update_after_step=0, interval=2)


def get_decay(optimization_step, cfg):
    # ema.py:311-331 (use_ema_warmup=False branch)
    step = max(0, optimization_step - cfg["update_after_step"] - 1)  # :318
    if step <= 0:                                                    # :320
        return 0.0                                                   # :321
    cur = (1 + step) / (10 + step)                                   # :326
    cur = min(cur, cfg["decay"])                                     # :328
    cur = max(cur, cfg["min_decay"])                                 # :330
    return cur


params = [torch.randn(s, dtype=torch.float32).to(torch.bfloat16) for s in SHAPES]
out = {}
for i, t in enumerate(params):
    out[f"p_init_{i}"] = t.clone()

# shadow init = p.clone() (ema.py:123); F32 master shadows over bf16 live.
sh1 = [t.to(torch.float32) for t in params]
sh2 = [t.to(torch.float32) for t in params]
decays1, decays2 = [], []

for gstep in range(1, N + 1):
    # Param drift stands in for the optimizer step.
    params = [
        (t.to(torch.float32) + 0.05 * torch.randn(t.shape)).to(torch.bfloat16)
        for t in params
    ]
    for i, t in enumerate(params):
        out[f"p_s{gstep}_{i}"] = t.clone()
    for cfg, sh, dl in ((CFG1, sh1, decays1), (CFG2, sh2, decays2)):
        if cfg["interval"] > 1 and gstep % cfg["interval"] != 0:  # ema.py:29-37
            dl.append(-1.0)  # skip marker (SimpleTuner returns early, :335-336)
            continue
        decay = get_decay(gstep, cfg)  # optimization_step := global_step (:360-362)
        one_minus = 1 - decay          # ema.py:369
        for s, p in zip(sh, params):
            target = p.to(s.dtype)               # ema.py:401-402
            s.sub_(one_minus * (s - target))     # ema.py:405
        dl.append(decay)

for i, s in enumerate(sh1):
    out[f"shadow1_{i}"] = s
    out[f"shadow1_bf16_{i}"] = s.to(torch.bfloat16)  # copy_to cast (ema.py:454)
for i, s in enumerate(sh2):
    out[f"shadow2_{i}"] = s
out["decays1"] = torch.tensor(decays1, dtype=torch.float32)
out["decays2"] = torch.tensor(decays2, dtype=torch.float32)

save_file(out, "/tmp/lora_ema_oracle.safetensors")
print(f"wrote /tmp/lora_ema_oracle.safetensors ({len(out)} tensors, N={N})")
print("decays1:", [round(d, 6) for d in decays1])
print("decays2:", [round(d, 6) for d in decays2])

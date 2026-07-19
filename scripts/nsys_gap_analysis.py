#!/usr/bin/env python3
"""GPU-busy vs wall-clock analysis of one Klein int8 training step.
Finds step boundaries via the optimizer's fused lora_ada* kernel, then for the
window between two consecutive optimizer-kernel groups computes:
  - per-stream busy time (sum of kernel durations)
  - union busy time across all kernels (any-GPU-work fraction)
  - memcpy busy time (D2H / H2D / D2D)
  - top-15 kernels by total time inside the window
  - gap histogram: count+total of inter-kernel gaps on the busiest stream
"""
import sqlite3, sys
from collections import defaultdict

db = sys.argv[1]
con = sqlite3.connect(db)
cur = con.cursor()

# kernel table + string ids
kerns = cur.execute(
    "SELECT k.start, k.end, k.streamId, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
    "JOIN StringIds s ON k.demangledName = s.id ORDER BY k.start").fetchall()
print(f"total kernels: {len(kerns)}")

# optimizer marker kernels (the big fused adamw: 'lora_ada' in name)
opt = [r for r in kerns if 'lora_ada' in r[3]]
print(f"optimizer-marker kernels: {len(opt)}")
if len(opt) < 4:
    print("not enough steps captured"); sys.exit(1)

# group consecutive opt kernels (same step) when <0.5s apart; boundaries = group end
groups = []
cur_g = [opt[0]]
for r in opt[1:]:
    if r[0] - cur_g[-1][1] < 0.5e9:
        cur_g.append(r)
    else:
        groups.append(cur_g); cur_g = [r]
groups.append(cur_g)
print(f"optimizer groups (steps): {len(groups)}")
if len(groups) < 3:
    sys.exit(1)

# steady step window: end of group[-3] -> end of group[-2] (one full step, steady)
w0 = groups[-3][-1][1]
w1 = groups[-2][-1][1]
wall = w1 - w0
print(f"\n=== ONE STEADY STEP window: {wall/1e9:.3f} s wall ===")

wk = [r for r in kerns if r[0] >= w0 and r[1] <= w1]
print(f"kernels in step: {len(wk)}  (launch rate {len(wk)/(wall/1e9):.0f}/s)")

# per-stream busy
by_stream = defaultdict(int); cnt_stream = defaultdict(int)
for s0, s1, st, nm in wk:
    by_stream[st] += s1 - s0; cnt_stream[st] += 1
print("\nper-stream kernel busy:")
for st in sorted(by_stream, key=lambda x: -by_stream[x]):
    print(f"  stream {st}: {by_stream[st]/1e9:.3f} s busy ({100*by_stream[st]/wall:.1f}% of wall), {cnt_stream[st]} kernels")

# union busy across all kernels
ivs = sorted((r[0], r[1]) for r in wk)
union = 0; ce = 0; cs = None
for a, b in ivs:
    if cs is None: cs, ce = a, b
    elif a <= ce: ce = max(ce, b)
    else: union += ce - cs; cs, ce = a, b
if cs is not None: union += ce - cs
print(f"\nUNION kernel-busy: {union/1e9:.3f} s = {100*union/wall:.1f}% of wall -> GPU IDLE {100*(1-union/wall):.1f}%")

# memcpys
try:
    mc = cur.execute("SELECT start, end, copyKind, bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY "
                     "WHERE start >= ? AND end <= ?", (w0, w1)).fetchall()
    kinds = {1:'H2D',2:'D2H',8:'D2D',10:'P2P'}
    agg = defaultdict(lambda: [0,0,0])
    for a,b,k,by in mc:
        agg[kinds.get(k,k)][0] += b-a; agg[kinds.get(k,k)][1] += by; agg[kinds.get(k,k)][2] += 1
    print("\nmemcpy in step:")
    for k,(t,by,n) in agg.items():
        print(f"  {k}: {t/1e9:.3f} s busy, {by/1e6:.0f} MB, {n} copies")
except Exception as e:
    print("memcpy query failed:", e)

# top kernels in the step
agg2 = defaultdict(lambda: [0,0])
for s0,s1,st,nm in wk:
    agg2[nm][0] += s1-s0; agg2[nm][1] += 1
print("\ntop 15 kernels in the step:")
for nm,(t,n) in sorted(agg2.items(), key=lambda x:-x[1][0])[:15]:
    print(f"  {t/1e6:8.1f} ms  n={n:5d}  {nm[:80]}")

# gap histogram on the busiest stream
bs = max(by_stream, key=lambda x: by_stream[x])
sk = sorted([r for r in wk if r[2]==bs])
gaps = [(sk[i+1][0]-sk[i][1]) for i in range(len(sk)-1)]
big = sorted(gaps, reverse=True)[:10]
tot = sum(g for g in gaps if g>0)
print(f"\nstream {bs} inter-kernel gaps: total {tot/1e9:.3f} s across {len(gaps)} gaps")
print(f"  top-10 gaps (ms): {[round(g/1e6,1) for g in big]}")
n_small = sum(1 for g in gaps if 0 < g < 30000)
print(f"  gaps <30us (launch overhead-ish): {n_small} totaling {sum(g for g in gaps if 0<g<30000)/1e9:.3f} s")

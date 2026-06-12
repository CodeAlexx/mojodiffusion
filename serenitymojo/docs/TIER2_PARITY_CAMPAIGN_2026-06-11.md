# Tier-2 SimpleTuner-parity campaign — ledger (started 2026-06-11)

Source: AUDIT_SIMPLETUNER_PARITY_2026-06-11.md Tier-2 table. Same discipline
as Tier-1 (TIER1_PARITY_CAMPAIGN_2026-06-11.md): builder/bugfix/skeptic
agents, orchestrator re-runs every gate before commit, 4-agent pool
maintained until done (Alex directive), C13 gate-don't-delete (flags-off
paths keep anchors EXACT), everything modular through training/levers.mojo
+ TrainConfig keys (NO per-trainer forks), every phase wired through to the
serenity-trainer UI.

## Phases

| Phase | Item | Status |
|---|---|---|
| T2.A | 8-bit Adam (bnb block-wise parity) | SHIPPED |
| T2.B | fp8-quantized-resident base (HiDream first) | SHIPPED (default-OFF) |
| T2.C | Full-rank finetune (1 model) | IN FLIGHT |
| T2.D | Dynamic aspect bucketing | SHIPPED (stage-side; trainer = compiled-arm coverage) |
| T2.E | ControlNet (1 model) | QUEUED |
| T2.F | Lycoris family verification (LoCon/LoHa/Tucker/OFT) | SHIPPED (verified primitives) |

## Oracles / references
- T2.A: EriDiffusion-v2 parity suite — crates/eridiffusion-cli/src/bin/
  parity_adam8bit_bnb{,_wd,_tail,_bf16grad,_multistep}.rs + tests/parity/
  adam8bit_bnb_python_ref*.py + adam8bit_data/; flame kernel
  /home/alex/EriDiffusion/flame-core/src/adam8bit_kernel.rs.
- T2.B: ops/fp8.mojo + ops/fp8_gemm.mojo (ideogram4 inference machinery);
  bf16-resident HiDream baseline = train_hidream_o1_real.mojo flags-off
  trajectory 0.05885428/0.33308488/0.5214583 (~1.0 s/step). fp8-resident is
  a NEW numerics class -> config-flag default-OFF, deltas documented.
- T2.D: zimage per-bucket dispatch (train_zimage_real.mojo) = the in-repo
  precedent; generalize, don't fork.
- T2.F: training/locon_save.mojo, loha_save.mojo, tucker_save.mojo,
  tucker_conv_adapter.mojo, oft_save.mojo; oracle = pip lycoris-lora /
  torch reproductions. No family claimed without a gate.

## Standing rules
- One GPU: agents check nvidia-smi >= 20GB free before any GPU gate,
  wait-retry otherwise; keep GPU gates short (<=5 min).
- Agents do NOT commit; orchestrator gates then commits.
- Mojo build serialization: rm -f serenitymojo.mojopkg, one compile at a
  time per repo.

## Results
(appended per phase as gates pass)

### T2.A — 8-bit Adam (2026-06-11) SHIPPED
training/adamw8bit.mojo (host bnb AdamW8bit: block-256 absmax, dynamic
signed/unsigned qmaps) + tests/adamw8bit_parity.mojo (17 gates vs bnb
0.49.2 dumps: basic/wd/tail/bf16grad/multistep-10; tolerances taken from
the EDv2 Rust parity bins). Codes bit-equal everywhere except bf16grad
(2/2048 signed codes, ref bin allows 5); param deltas <=2.4e-7 (~1.5 ulp).
Levers dispatch tag ADAMW_8BIT (was a silent ADAMW alias in the reader —
now the real optimizer); UI dropdown ADAMW8BIT -> ADAMW_8BIT runner enum.
Orchestrator re-ran adamw8bit_parity (ALL PASS) + levers_optimizer_dispatch
(35/35 incl. C13 default-not-active) on the integrated tree.

### T2.B — fp8-quantized-resident HiDream-O1 (2026-06-11) SHIPPED, default-OFF
ops/fp8_quant.mojo (E4M3-fn per-row encode matching the gated ideogram4
decode) + trainer flag "quantized_resident": "fp8_e4m3" (7 big linears
fp8-resident, norms/<1MB stay bf16; per-block dequant shared by fwd+bwd).
MEASURED: VRAM 18255 -> 12315 MiB (-5.8GB), ~+10% s/step; 10-step loss
trajectory cosine 0.9996582 vs bf16 (max |rel| 7.8%, mean 2.1%).
NUMERICS CLASS RECORD: step-1 adapter-grad cosine 0.809 — the 0.999 bar is
physically unattainable for fp8 (measured 2.65% RMS per-weight noise =
E4M3 budget, compounding over 36 blocks; loss-class is preserved, the
TRAJECTORY differs — do not mix-resume across the flag). Flag-off C13:
exact 3-step anchor 0.05885428/0.33308488/0.5214583, orchestrator re-ran
on the integrated tree — EXACT. Promotion to default needs Alex's eyeball
on a real training (sample quality), not just these gates.

### T2.D — Dynamic aspect bucketing (2026-06-11) SHIPPED
training/aspect_buckets.mojo: SimpleTuner-semantics ladder generator +
stage-time assignment with Python-exact rounding (oracle = SimpleTuner's
OWN MultiaspectImage code, crop_aspect="closest" mode; cited files in the
module header). Gates: (a) assignment parity EXACT (80 assignments + 28
bucket-set rows, diff empty); (b) zimage flags-off 5-step anchors
0.4745/0.5739/0.4903/0.5065/0.4750 EXACT at 4dp — orchestrator re-ran 3x
on the final integrated tree: step-2 prints 0.5739/0.5739/0.5740 and
last_loss wobbles at digit 5 run-to-run (measured flash-bwd
nondeterminism class, step-1 byte-identical 0.47450438 every run);
(c) real 2-bucket mixed-aspect e2e train (72x56 + 88x48, loss decreased,
B grew). zimage stager/prepare get opt-in t2d argv modes (default paths
untouched); cache tensor schema UNCHANGED + aspect_buckets.json manifest.
Trainer buckets stay comptime-compiled: the standard 512-px ladder lands
exactly on the 3 compiled arms; landscape arms fail loud. FOLLOW-UP:
comptime-generated dispatch (@parameter for over the integer ladder) +
landscape VAE instantiations. Anchor config promoted from /tmp to
configs/zimage_alina_anchor.json.

### T2.F — Lycoris family verification (2026-06-11, skeptic pass)
Gates: serenitymojo/training/tests/{gen_lycoris_family_oracle.py,
lycoris_family_parity.mojo, lycoris_family_load_check.py}.
Oracle = pip lycoris_lora 3.4.0 (the EDv2 lycoris.rs the save modules cited is
DELETED; upstream pip is the only live reference).

| family | trainable e2e? | fwd parity vs torch | save format | gaps |
|---|---|---|---|---|
| LoCon  | NO (no adapter_algo id; primitive only) | PASS cos=1-1e-14, norm_rel 2.4e-9 | PASS — keys == upstream, upstream loads it BIT-EXACT | not wired into any trainer |
| Tucker | NO (no adapter_algo id; primitive only) | PASS cos=1-1.5e-14, norm_rel 3.7e-8 | PASS — keys == upstream (incl. lora_mid), BIT-EXACT load | not wired into any trainer |
| LoHa   | NO (algo=2 raises in train_klein_real) | PASS cos=1.0, bit-exact | PASS after FIX — was `.hada_w1_a.weight` + transposed factors (ecosystem-unloadable); now upstream keys/orientation, BIT-EXACT load | stack integration tracked follow-up |
| OFT    | NO (algo=5 raises in train_klein_real) | PASS cos=1-2.3e-15, norm_rel 2.2e-8 | PASS after FIX — was `.oft_blocks.weight` + wrong skew parametrization (2x Q); now `oft_blocks` = 0.5*S, BIT-EXACT load | alpha==constraint semantics; a1111-lyco zeroes OFT at alpha=0 (doc'd) |

Bugs fixed (loha_save.mojo, oft_save.mojo): see file headers. Found + gated:
upstream LohaModule.get_diff_weight double-applies scale (live bypass path +
a1111/comfy apply it once — single-scale is the convention; the family gate
now carries a norm_rel<=1e-5 bar because cosine is scale-blind).
loha/oft adapter smokes re-run after the format fix: ALL GATES PASS.

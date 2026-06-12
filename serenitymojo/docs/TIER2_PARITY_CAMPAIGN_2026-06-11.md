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
| T2.C | Full-rank finetune (1 model) | WIP — scaffolding committed UNGATED (agent died on spend limit; gates owed) |
| T2.D | Dynamic aspect bucketing | SHIPPED + follow-up SHIPPED (comptime-generated 14-arm dispatch, landscape buckets live) |
| T2.E | ControlNet (1 model) | GATED BLOCK (zimage; trainer data path = follow-up) |
| T2.F | Lycoris LoCon/LoHa/Tucker/OFT | SHIPPED (verified primitives) |
| T2.F-2 | Lycoris LoKr/BOFT/DoRA | WIP — lokr/boft save fixes committed, smokes pass, UPSTREAM GATES OWED (agent died on spend limit); dora untouched |
| UI | Lever delivery (ideogram4 argv bridge, hidream runner) | SHIPPED (serenity-trainer eaa88f1) |

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

### T2.D follow-up — comptime-generated bucket dispatch (2026-06-11) SHIPPED
aspect_buckets.mojo comptime integer ladder (exact integer nearest-sqrt,
ties-to-even) provably equal to generate_aspect_buckets (gate:
tests/zimage_comptime_ladder_gate.mojo — 7 buckets exact, orchestrator
re-ran). train_zimage_real dispatch = comptime-for over ladder x cap lens
= 14 generated arms (landscape 56x72/48x88/56x80/80x56 + 64x64 square now
real); zimage_prepare instantiates all 7 VAE buckets. MEASURED: trainer
build 1m28s (feared 2x blowup absent). Landscape e2e smoke: rotated
1350x1080 -> bucket 576x448 -> latent 56x72 -> 2 train steps, B grew.
C13 anchors EXACT (orchestrator re-ran on the wave-2 intermingled tree:
0.4745/0.5739/0.4903/0.5065/0.4750, step-1 byte-anchor 0.47450438).

### UI lever delivery (2026-06-11) SHIPPED — serenity-trainer eaa88f1
Bridge appends ideogram4 argv 10 (dropout) / 11 (levers JSON, "-"
sentinel); NEW hidream runner target + binary + config.json delivery
(levers + quantized_resident "OFF"). GATES (orchestrator): runner gate
128 PASS; bridge parser PASS; ideogram4 argv old-vs-new equivalence
EXACT (1.12493/1.1416154 both forms).

### T2.C — full-rank finetune zimage (2026-06-11) WIP, UNGATED
Agent died on spend limit mid-build. On disk (committed so the tree is
clean, UNGATED): training/full_finetune_zimage.mojo (8-bit-AdamW-based
full-FT step machinery; orchestrator fixed an UnsafePointer origin
syntax error at :148 to unblock the tree), configs/zimage_full_ft_smoke
.json, train_config/reader keys, train_zimage_real hooks. Flag-off C13
PROVEN (anchor gate EXACT on this tree); the full-FT path itself has NO
gate yet — VRAM math, smoke, save-format check all owed before any
"runnable full-FT" claim.

### T2.F-2 — LoKr/BOFT/DoRA (2026-06-11) WIP, upstream gates OWED
Agent died on spend limit. Committed: lokr_adapter.mojo both-full scale
quirk mirror (upstream lokr.py:209-211 forces scale=1 when W1+W2 both
full — header cites lines), lokr_save.mojo + boft_save.mojo format
rewrites (LoHa/OFT bug-class precedent). In-repo smokes PASS
(orchestrator re-ran: lokr 3 variants + boft b2x4/b2x2, round-trips
byte-exact) — but parity vs pip lycoris_lora + upstream load checks DO
NOT EXIST yet for these three; dora untouched. No family claim until
those gates run (T2.F bar).

### T2.E — ControlNet, Z-Image (2026-06-11) GATED BLOCK; trainer data path = follow-up
SURVEY: model = zimage (most mature vertical: anchors in configs/, T2.D stager
argv precedent, v2 engine). Reference = diffusers 0.38.0.dev0
ZImageControlNetModel (controlnet_z_image.py — the OFFICIAL Alibaba Z-Image
ControlNet; the only exact-architecture DiT ControlNet reference on this box
for our verticals — SimpleTuner's chroma/hidream controlnets exist but chroma
is the 54GB trainer and HiDream-O1-vs-I1 arch match is unverified). DiT
pattern confirmed: control block = copy of ZImageTransformerBlock +
zero-init before_proj (block 0) / after_proj (every block); hints added to
`unified` AFTER each main layer at control_layers_places
(transformer_z_image.py:1032).

SHIPPED:
- models/zimage/controlnet_block.mojo — control block + N-block control-stack
  fwd/bwd COMPOSING the parity-gated zimage block; diffusers checkpoint key
  mapping documented + loader (reuses weights.mojo prefixed loader for
  control_layers.{i}); zero-init constructors.
- GATE (a) block parity: zimage_controlnet_block_oracle.py (F64 hand math
  GROUNDED vs the real diffusers ZImageControlTransformerBlock inside the
  oracle — cross-check max|diff| 2.2e-7/2.2e-7/2.3e-7, its internal F32 rope
  floor) + zimage_controlnet_block_parity.mojo: 2-block chained stack, one
  fwd+bwd, 46/46 comparisons PASS at cos >= 0.99999 (measured ~1-1e-11),
  covering hints/c_final/d_c0/d_x + all 17 trainable grads per block + the
  before/after projection grads.
- GATE (c) e2e training smoke: zimage_controlnet_step_smoke.mojo — 8 SGD steps
  at parity dims (D=3840, S=8): frozen base (bit-identical after run),
  control-only updates, post-layer hint injection fwd+bwd; loss 0.3725 ->
  0.3123 (5/7 steps down); zero-init cascade verified (after_proj off zero at
  step 1, before_proj off zero at step 2). PASS, reproduced byte-identically.
- TrainConfig keys (default-off): controlnet_layers (Int 0) /
  controlnet_scale (Float64 1.0) / controlnet_checkpoint (String "") +
  reader plumbing (negative count fails loud at load).
- GATE (b) C13: zimage flags-off 5-step anchors on the integrated tree
  0.4745/0.5740/0.4903/0.5065/0.4750 (step-2 last-digit wobble is the
  documented flash-bwd nondeterminism class; step-1 byte-anchor 0.47450438
  EXACT). controlnet_layers>0 FAILS LOUD in train_zimage_real (T2.F
  adapter-algo precedent) with a pointer to the module contract.

FOLLOW-UP (the remaining trainer data path, contract in the module header):
control-image channel in zimage_stage_alina/zimage_prepare (T2.D argv-mode
pattern), control x_embedder + control_noise_refiner (2 plain modulated
blocks — zimage_block_forward as-is), hint injection in the bf16-resident
main loop fwd+bwd (+ v2 graph arm), control-param fused-AdamW group, diffusers
ZImageControlNetModel-format save.

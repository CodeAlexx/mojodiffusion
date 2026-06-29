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
| T2.C | Full-rank finetune (zimage) — RE-SCOPED per Alex 2026-06-12: OneTrainer-contract item, NOT SimpleTuner parity. Oracle = full_finetune_contract.mojo / OneTrainer *FineTuneSetup | SHIPPED v1 gated (c789b6d): 5.31B params, 67-81 s/step, 15.8GB device peak; follow-ups in results |
| T2.D | Dynamic aspect bucketing | SHIPPED + follow-up SHIPPED (comptime-generated 14-arm dispatch, landscape buckets live) |
| T2.E | ControlNet (1 model) | GATED BLOCK (zimage; trainer data path = follow-up) |
| T2.F | Lycoris LoCon/LoHa/Tucker/OFT | SHIPPED (verified primitives; LoCon selector/linear route wired 2026-06-27; LoHa/OFT trainer integration expanded 2026-06-29) |
| T2.F-2 | Lycoris LoKr/BOFT/DoRA | SHIPPED primitives; trainer wiring expanded 2026-06-29 for LoKr/LoHa/DoRA/OFT across the non-LTX2, non-Wan scope; BOFT remains intentionally excluded |
| UI | Lever delivery (ideogram4 argv bridge, hidream runner) | SHIPPED (serenity-trainer eaa88f1) |
| T2.G | SimpleTuner-LoKr full parity (Alex directive) | SHIPPED — LoKr e2e TRAINABLE on klein (adapter_algo=4), init_lokr_norm ported exactly, parity matrix in results |

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
| LoCon  | LIMITED (adapter_algo=7; linear LoRA-compatible targets route through existing down/up carrier) | PASS cos=1-1e-14, norm_rel 2.4e-9 | PASS — keys == upstream, upstream loads it BIT-EXACT | conv LoCon primitive is gated but not threaded into conv-bearing model stacks |
| Tucker | NO (no adapter_algo id; primitive only) | PASS cos=1-1.5e-14, norm_rel 3.7e-8 | PASS — keys == upstream (incl. lora_mid), BIT-EXACT load | not wired into any trainer |
| LoHa   | YES for current non-LTX2/non-Wan trainer scope where model-specific carrier stacks exist; older linear-only fallback still fails loud | PASS cos=1.0, bit-exact | PASS after FIX — was `.hada_w1_a.weight` + transposed factors (ecosystem-unloadable); now upstream keys/orientation, BIT-EXACT load | 24 GB proof varies by model/preset; keep carrier-byte preflights strict |
| OFT    | YES for current non-LTX2/non-Wan direct/full-delta trainer scope; BOFT remains excluded | PASS cos=1-2.3e-15, norm_rel 2.2e-8 | PASS after FIX — was `.oft_blocks.weight` + wrong skew parametrization (2x Q); now `oft_blocks` = 0.5*S, BIT-EXACT load for legacy LyCORIS primitive; OneTrainer trainer save uses `<prefix>.oft_R.weight` | Some gates are smoke/update/save only; full-resolution SDXL and all-target Klein DoRA remain separate claims |

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

### T2.C — full-rank finetune zimage (2026-06-12) SHIPPED v1, gated (c789b6d)
RUNNABLE on the OneTrainer contract: F32 masters (21.2GB host) + bnb 8-bit
moments (10.6GB host) + bf16-resident device weights = RNE master image.
v1 surface = 30 main blocks x 7 slots, 5.31B/6.15B params (86%).
MEASURED: device peak 15831 MiB <= 24GB; host ~43/62GB; 67-81 s/step
(was 407.8 serial — FP lesson: closure re-body picked up FMA contraction
and FAILED the bit gate; fix = byte-identical @no_inline body, parallelism
at the 210-slot level, slot-parallel bit-gate mismatches=0). GATES: run-
start fast-requant equivalence bit-equal; 5-step smoke (upd_l1 nonzero
every step, trained delta 170M elems); checkpoint 521 keys SOURCE-schema
diff PASS; C13 LoRA anchors EXACT (byte-anchor 0.47450438).
FOLLOW-UPS: surface extension to 100% (measured per-group gap list in the
T2.C report: final linear cheapest -> adaLN/norms -> embedders ->
refiners; host fits, predicted 49.2GB); resume sidecar (8-bit state vs
contract's F32 moments = open design); GPU-resident streamed 8-bit
optimizer kernel (step is ~90% host optimizer).

### T2.F-2 — LoKr/BOFT/DoRA (2026-06-11) SHIPPED, upstream gates RUN
Predecessor committed: lokr_adapter.mojo both-full scale quirk mirror
(upstream lokr.py:209-211 forces scale=1 when W1+W2 both full),
lokr_save.mojo + boft_save.mojo format rewrites. THIS PASS extended the
T2.F gate trio (gen_lycoris_family_oracle.py / lycoris_family_parity.mojo
/ lycoris_family_load_check.py) with lokr x3 / boft / dora — same oracle
(pip lycoris_lora 3.4.0), same bars (fwd cos>=0.99999 + norm_rel<=1e-5;
upstream loads the Mojo file and reproduces the forward BIT-EXACT,
max|d|=0.0). The original four families re-ran on the extended gate: ALL
STILL PASS.

| family | trainable e2e? | fwd parity vs torch | save format | gaps |
|---|---|---|---|---|
| LoKr | YES for current non-LTX2/non-Wan trainer scope: Flux, Chroma, Qwen-Image, SD3.5, SDXL, Anima, Z-Image, L2P, ERNIE, HiDream O1, Klein, and Krea2 have LoKr branches/build/update/save wiring in the checkout | PASS x3 variants (W1full+W2factored cos=1-1.7e-15 nr 3.7e-9; both-full cos=1.0 nr 1.2e-9 incl. forced-scale=1 quirk; both-factored cos=1-5.3e-15 nr 4.7e-9) | PASS — bare lokr_w1[_a/_b]/lokr_w2[_a/_b]+alpha keys == upstream, LokrModule loads all 3 BIT-EXACT | 24 GB proof is config-dependent because LoKr uses carrier dispatch; large/all-target/full-matrix settings must fail loud on carrier bytes |
| BOFT | NO (algo=6 raises in train_klein_real) | PASS (b=2,nb=4,boft_m=3, all stages nontrivially permuted: cos=1-4.8e-15, nr 7.6e-8) | PASS — oft_blocks 4D [m,nb,b,b] (the a1111/comfy BOFT-vs-OFT rank discriminator, upstream algo_check verified) + blocks=-0.5*S orientation fold, ButterflyOFTModule loads BIT-EXACT | alpha==constraint semantics (0 written), like OFT |
| DoRA | YES for current non-LTX2/non-Wan direct/full-delta trainer scope, with model-specific runtime gates recorded in `LYCORIS_CARRIER_DISPATCH_2026-06-27.md` | PASS (cos=1-2e-15, nr 4.7e-9; FULL-forward effective-weight replacement, eps=finfo(f32).eps) | PASS after 2 FIXES — keys were PEFT lora_A/lora_B hybrid (ecosystem-unloadable); now upstream-lycoris lora_down/lora_up + dora_scale + alpha, LoConModule(wd=True) loads BIT-EXACT; OneTrainer input-axis save is also wired for direct trainer paths | Severe throughput caveats remain on SD3.5/L2P/Z-Image DoRA; all-target Klein DoRA was interrupted before step completion |

Bugs fixed this pass:
1. dora_save.mojo KEYS: `.lora_A.weight`/`.lora_B.weight` (PEFT) →
   `.lora_down.weight`/`.lora_up.weight` (upstream lycoris LoCon(wd=True)
   schema; same bug class as the T2.F LoHa/OFT key fixes).
2. dora_adapter.mojo MAGNITUDE DTYPE: m was BF16 like the low-rank legs,
   which broke the documented identity-at-init contract by ~0.3%
   (measured: smoke a-init max|Δ|=3.05e-3 vs the 1e-5 bar — PRE-EXISTING
   on the committed tree, the dora smoke had never been re-run). Upstream
   explicitly keeps dora_scale float32 even in bf16 models
   (locon.py/lokr.py `nn.Parameter(...).float()`); m is now F32 storage
   + F32 dora_scale on disk. dora_adapter_smoke re-run: ALL GATES PASS
   (a-init max|Δ|=1.19e-7 = the eps).
Also gated: LoKr is a SECOND instance of the upstream double-scale quirk
(get_weight folds scale via make_kron; get_diff_weight multiplies again
— oracle uses the single-scale live-forward convention, like LoHa).
Load-check hygiene: mmap'd safetensors x inputs are .clone()d — mmap
alignment flips torch's GEMM path (±1 ulp) and broke the bit-exact bar.
lokr/boft/dora in-repo smokes re-run on this tree: ALL PASS.

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

### T2.E follow-up — trainer data path (2026-06-11) SHIPPED
- DATA: zimage_stage_alina `cn` argv mode (image + SAME-bucket control image;
  control source = optional second folder, identity control when omitted —
  simplest faithful conditioning, documented) -> zimage_prepare `cn` mode
  (both through the SAME VAE encode, diffusers pipeline_z_image_controlnet.py
  :550-551 semantics) -> ADDITIVE cache key `control_latent`
  (klein_dataset.write_sample_control / load_control / has_control; old
  3-key caches load unchanged — C13).
- TRAINER (controlnet_layers>0, its own runtime driver — LoRA path
  untouched): training/controlnet_zimage.mojo (named F32 master store,
  copy-from-base init + zero projections, controlnet_checkpoint load,
  control x_embedder + control_noise_refiner fwd/bwd, adaLN grads from the
  RAW mod-vec grads, GLOBAL-clip + host-AdamW control group, diffusers
  folder save) + zimage_stack_lora.mojo *_cn arms (post-layer hint injection
  fwd; v2 GRAPH backward emitting per-place d_hints; frozen base via a
  rank-16 B=0 LoRA set — the graph engine records LoRA ops unconditionally,
  so the full-FT [1,1] zero-placeholder set does NOT work there).
  places = evenly spaced [i*30//N] (0 included, the reference assert).
- GATES (all on the integrated tree): (a) zimage_controlnet_block_parity
  46/46 + step smoke re-run PASS; (b) C13 flags-off anchors EXACT
  0.4745/0.5739/0.4903/0.5065/0.4750, step-1 byte 0.47450438, SLAB
  5640/5640/5640; (c) e2e 5 real steps on the cn cache (zimage_cn_smoke.json,
  N=2 places 0/15): loss finite 0.4726->0.5159, |after_w0|_1 1465->3638,
  before_proj 0 at step1 / 1042 at step2 (zero-init cascade), frozen base
  BIT-IDENTICAL (sampled layers.0.to_q + layers.29.w2); (d) saved checkpoint
  key/shape diff vs diffusers ZImageControlNetModel: EMPTY (68/68,
  parity/zimage_controlnet_save_keydiff.py).
- v1 limits (documented): batch-1; 72x56 + 64x64 buckets; control group
  host-AdamW + per-step F32 re-upload (~18 s of the ~21 s step is
  ctl_bwd+host-opt — device-resident control AdamW is the perf follow-up);
  control adaLN IS trained (full ZImageControlNetModel surface).

### T2.G — SimpleTuner-LoKr full parity (2026-06-12) SHIPPED
Alex: "simpletuner has best lokr anywhere, make sure we have full parity
there" + init_lokr_norm must-have. Parity matrix (knob-by-knob, ST cites)
in the agent report; HAVE: algo/dim/alpha/factor(-1 auto)/apply_preset
(3 preset classes)/module_algo_map factors/full_matrix/decompose_both/
default zero-init/init_lokr_norm (exact op-order port of peft_init.py,
Box-Muller, F64 reductions, both-full-only applicability)/upstream save
keys/bypass-equivalent/grad clip. MISSING (fail-loud): init_lora
warm-start, resume, EMA-with-lokr, validation sampling, exotic kwargs
(rank_dropout/use_scalar/...), ST header metadata (cosmetic).

ARCHITECTURE: Kronecker mixed-product identity folds every LoKr variant
into one plain-LoRA carrier pair the existing klein stack consumes — zero
stack/kernel changes; masters -> carriers -> stack grads chain exactly.

GATES (orchestrator re-ran the suite + C13 + load check): factorization
table EXACT 19 cases (odd-max //2 bug FIXED to upstream float semantics);
leg/shape table EXACT 11 cases vs real LokrModule; reduced-dim 3-step
torch+lycoris training repro delta cos 0.9995/0.99999; perturbed-init
stats vs ST helper 0.05%; klein 10-step smokes BOTH PASS (factored
~4.5 s/step 144 modules; full_matrix+init_lokr_norm targets=attn
~50 s/step, w1 trains); upstream LokrModule loads trained checkpoints
with BIT-EXACT reconstruction; klein flags-off anchors in-class
(0.5414 exact / 0.2155 / 0.7810).

BUG FOUND BY GATE: bf16 RNE writeback bit-froze w1=1.0 under the
perturbed init (lr*update < half-ULP(1.0)) — fixed with the repo's
canonical stochastic-rounding writeback; w1 measurably trains.

2026-06-27 UI/config follow-up: `TrainConfig.adapter_algo` now has named
constants for LoRA/Full/LoHa/DoRA/LoKr/OFT/BOFT/LoCon; the reader accepts
`network_algorithm`, `adapter_algo`, or `algo`. The serenity-trainer UI emits
both `network_algorithm` and `adapter_algo`. LTX2 remains LoRA-only.

2026-06-29 trainer-integration follow-up: the current non-LTX2 and non-Wan
trainer scope has LoKr/LoHa/LoCon/DoRA/OFT wiring. LoKr support is present in
Flux, Chroma, Qwen-Image, SD3.5, SDXL, Anima, Z-Image, L2P, ERNIE, HiDream O1,
Klein, and Krea2 trainer paths. DoRA/OFT direct or bounded full-delta paths now
have recorded 24 GB smoke/update/save evidence across the same live scope except
that all-target Klein DoRA still needs a completed step rerun. Wan code exists
historically in this checkout but is excluded from the active user scope. BOFT
continues to fail loud and must not be silently mapped to OFT.

FOLLOW-UPS: structured-kron GPU kernels (full_matrix at 9B full target
set needs ~17GB dense carriers — preflight fails loud with bytes);
LoKr resume/init_lora/EMA; step-cost optimization (4.5s vs 2.2 LoRA).

# Tier-3 parity campaign — ledger (started 2026-06-12)

Source: AUDIT_SIMPLETUNER_PARITY_2026-06-11.md Tier-3 table + the Tier-3
plan in HANDOFF_2026-06-12.md (local). Scope per Alex: S3/cloud-data/
webhooks/trackers SKIPPED ("infra conveniences"); multi-GPU excluded
(standing). Same discipline as Tier-1/2: orchestrator re-runs gates
before commit, 4-agent pool, C13, everything modular through levers/
TrainConfig, every phase reaches the serenity-trainer UI.

## Phases — PAUSED per Alex 2026-06-12 ~01:00 ("stop them, do next session")
| Phase | Item | Status |
|---|---|---|
| T3.A | TREAD/CREPA regularizers as levers | PAUSED mid-survey (ST side largely surveyed; zimage stack study next) |
| T3.B | Edit/Kontext-class conditioning (survey -> build) | PAUSED mid-survey (was reading OneTrainer BaseFlux2Setup conditioning) |
| T3.C | Audio training: ACE-Step trainer vertical | PAUSED — ACE-Step RECIPE FULLY EXTRACTED (see below), trainer build not started |
| T3.D | Model-family breadth (gap-scan -> top pick e2e) | PAUSED mid-scan (checkpoint inventory underway) |
| T3.E | Concept sliders | QUEUED |

NEXT SESSION: relaunch the 4 agents with their original briefs (in this
ledger's git history / HANDOFF_2026-06-12.md section 3) + the salvage
below. Trees were verified clean at pause (no agent edits to revert).

## SALVAGE — ACE-Step training recipe (extracted 2026-06-12, T3.C sub-survey;
## every line cited from /home/alex/SimpleTuner/simpletuner/helpers/models/ace_step/)
- Timesteps: logit-normal (mu=0, sigma=1) -> sigmoid -> index into
  FlowMatchEulerDiscreteScheduler(1000, shift=3.0) sigmas (model.py:1015-1025,
  :270-282; sigma shift formula scheduling_flow_match_euler_discrete.py:78-86).
- Noisy latent: sigma*noise + (1-sigma)*latent (model.py:1030-1035).
- Target: DATA prediction x0_hat = noisy - sigma*v_theta (model.py:1124-25);
  loss = MSE on attention-MASKED regions, per-sample mean weighted by valid-
  token fraction (model.py:1137-1171; flow-matching always L2 common.py:4562).
- Auxiliary SSL projection losses, weight 0.1 default (model.py:1173-1201;
  ace_step_ssl_loss_weight).
- LoRA targets default 7: linear_q/k/v + to_q/k/v + to_out.0
  (model.py:93-101); 3 selectable sets via acestep_lora_target (:233-268).
- Conditioning into transformer.forward: noisy latents + audio attention
  mask + UMT5-base text (768d) + text mask + speaker_embeds (512d, zeros if
  absent) + lyric BPE tokens (VoiceBpeTokenizer vocab 6681, start=261,
  sep=2) + lyric mask + timestep + cached SSL features (MERT-330M 1024d @
  24kHz + mHuBERT-147 768d @ 16kHz) (model.py:1107-1122 + sources cited).
- VAE: music DCAE (AutoencoderDC f8c8, NOT Oobleck), 48kHz in -> resample
  44.1k -> mel (hop 512) -> normalize ((mel+11)/14 then mean/std 0.5) ->
  encode 8ch -> (lat - (-1.9091)) * 0.1786
  (music_dcae_pipeline.py:34-125); latent_len =
  samples/48000*44100/512/8; defaults: tokenizer_max_length 256,
  max_grad_norm 1.0 (model.py:454-479). ACE-Step v1 3.5B.

## Standing rules
- One GPU: nvidia-smi >=20GB free before GPU gates, wait-retry; <=10 min runs.
- Agents do NOT commit; orchestrator gates then commits.
- One mojo compile at a time per repo (rm -f serenitymojo.mojopkg, retry on contention).
- Long-running smokes are launched by the ORCHESTRATOR (agent-session children die on session end).
- Gated FP loops: ONE @no_inline body; flash-class trainers gate as 4dp value-classes.

## Results
(appended per phase as gates pass)

## SALVAGE 2 — in-tree ACE-Step state (T3.C sub-survey, 2026-06-12)
- In-tree = ACE-Step **1.5** DiT inference (models/dit/acestep_dit.mojo:
  24 layers, GQA, sliding-window 128, AdaLN 6-way, patch conv1d) with
  REAL-checkpoint parity gates: block0 (cos>=0.999), full fwd + longseq
  (cos>=0.99, sliding mask proven load-bearing), flow-match sampler gate.
  Checkpoint: /home/alex/ACE-Step-1.5/checkpoints/acestep-v15-turbo/.
- NOT ported: condition encoder (Qwen3-Embedding-0.6B text/lyric/timbre
  -> [1,L,2048]) and the acoustic VAE — gates feed random conditioning.
- MISMATCH TO RESOLVE next session: SimpleTuner's recipe (Salvage 1) is
  ACE-Step **v1 3.5B** (UMT5 text, music DCAE mel VAE, MERT/mHuBERT SSL)
  — a DIFFERENT generation than the in-tree v1.5 (Qwen3-Embedding cond,
  acoustic VAE). The trainer build must pick ONE: v1.5 needs a recipe
  source (upstream ACE-Step-1.5 repo training code?) OR train v1 needs
  the v1 model ported. Resolve before building.

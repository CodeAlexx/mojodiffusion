#!/usr/bin/env python3
# scripts/krea2_omini_torch_oracle.py
#
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CUDA-ONLY torch parity ORACLE for the krea2 OminiControl EDIT vertical.  ║
# ║  CPU torch is NEVER a parity oracle for this stack. Every number this     ║
# ║  script emits is produced by torch on the RTX 5080 (sm_120) with the REAL ║
# ║  Krea-2-Raw checkpoint weights in bf16. The script HARD-FAILS if CUDA is  ║
# ║  unavailable — there is no CPU fallback, by design. A CPU run may only be ║
# ║  used to eyeball structure; it must never produce a gate number.          ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# WHAT THIS GATES
# ---------------
# ONE krea2 SingleStreamBlock running the OminiControl EDIT row layout owned by
# serenitymojo/training/krea2_omini_layout.mojo:
#
#     [ TXT_real(0:lt) | IMG(lt:lt+S_IMG) | COND(..+S_COND) | TXT_pad(tail) ]
#     LFULL_EDIT = LTMAX + S_IMG + S_COND        real_len = lt + S_IMG + S_COND
#
# with the three mechanics the Mojo C2/C3/C4 chunks must reproduce:
#   (1) PER-SEGMENT MODULATION — rows [0, cond_off) use mods(t); rows
#       [cond_off, LFULL) use mods(t=0). Both modulation vectors come from the
#       REAL krea2 temb -> tmlp -> tproj chain (checkpoint tmlp.*/tproj.1.*).
#   (2) ROPE positions with the EDIT condition delta [0,0] / scale 1.0, i.e. the
#       cond grid EQUALS the img grid (flux_omini.py:128-141 with dh=dw=0).
#   (3) LoRA applied ONLY to the CONDITION ROWS. Text and image rows run the
#       frozen base weights (OminiControl trainer.py:57 adapter_names
#       [None, None, "default"], confirmed by their own comment trainer.py:370).
#       Concretely: base matmul over the full sequence, low-rank delta computed
#       on x[:, cond_off:cond_off+S_COND] and scatter-added into those rows only.
#
# Attention runs over the real-length prefix; the pad tail is masked as KEY
# columns AND as query rows (the reference's outer-product bool mask). The Mojo
# pad-mask kernel (krea2_cache_reader.mojo:117-124) masks only key columns —
# equivalent, because pad ROWS are never read downstream.
#
# FIDELITY / SELF-CHECK (printed every run, non-negotiable)
# ---------------------------------------------------------
# The hand-written forward in this file is cross-checked, on the same device,
# same weights, same inputs, against the REAL reference module
# mmdit.SingleStreamBlock imported from
#   /home/alex/ai-toolkit/extensions_built_in/diffusion_models/krea2/src/mmdit.py
# with LoRA disabled, in BOTH modulation modes:
#   * single-vec  (vec)                    -> uniform mods(t)
#   * tuple-vec   (vec, refvec, split)     -> per-segment mods(t) | mods(0)
# NOTE (finding, see the handoff): the ai-toolkit krea2 reference ALREADY
# implements the per-segment modulation the intake brief listed as the one
# structural change to build (mmdit.py SingleStreamBlock.forward tuple branch,
# and SingleStreamDiT.forward `reflen` / `blockvec = (tvec, tproj(t0), split)`).
# It also already ships the OminiControl2 "independent_condition" asymmetric
# mask (`isolate_refs`) and the ref-K/V cache (`ref_span`/`kv_capture`). The
# reference's ROW ORDER differs from ours — it is [TXT | IMG | REF | PAD256]
# with a full bool key-padding mask, while our cuDNN flash tail-padmask forces
# [TXT_real | IMG | COND | TXT_pad]. The mechanics are identical; only the
# permutation and the mask construction differ.
#
# EXACT WEIGHTS USED
# ------------------
#   path : /home/alex/.serenity/models/checkpoints/krea2-raw.safetensors
#          -> symlink to ~/.cache/huggingface/hub/models--krea--Krea-2-Raw/
#             snapshots/4ad9f4b627a647fad78b3dfeebb09f2654aeb494/raw.safetensors
#          (26.28 GB, 28 single-stream blocks)
#   blocks : block 0 (first) and block 27 (last) — two separate fixture files.
#            Only those two blocks' tensors are ever resident (~0.87 GB each in
#            bf16), never the whole model.
#   per-block keys read (all REAL, none random):
#       blocks.N.attn.wq.weight        BF16 [6144, 6144]
#       blocks.N.attn.wk.weight        BF16 [1536, 6144]
#       blocks.N.attn.wv.weight        BF16 [1536, 6144]
#       blocks.N.attn.gate.weight      BF16 [6144, 6144]
#       blocks.N.attn.wo.weight        BF16 [6144, 6144]
#       blocks.N.attn.qknorm.qnorm.scale F32 [128]
#       blocks.N.attn.qknorm.knorm.scale F32 [128]
#       blocks.N.mlp.gate.weight       BF16 [16384, 6144]
#       blocks.N.mlp.up.weight         BF16 [16384, 6144]
#       blocks.N.mlp.down.weight       BF16 [6144, 16384]
#       blocks.N.mod.lin               F32  [36864]
#       blocks.N.prenorm.scale         F32  [6144]
#       blocks.N.postnorm.scale        F32  [6144]
#   modulation chain keys (REAL): tmlp.0.{weight,bias} [6144,256]/[6144],
#       tmlp.2.{weight,bias} [6144,6144]/[6144], tproj.1.{weight,bias}
#       [36864,6144]/[36864].
#   The eight big [out,in] base matrices are NOT re-dumped into the fixture
#   (0.87 GB/block of redundant bytes). The fixture records meta_block_index and
#   the checkpoint path in the safetensors __metadata__; the Mojo gate loads the
#   SAME keys from the SAME file. Only the small per-block params (norm scales,
#   mod.lin) are copied in, for convenience.
#
# DTYPE POLICY (matched to the Mojo compute path)
# -----------------------------------------------
#   storage/activations : BF16   (train_krea2.mojo builds `combined` from BF16
#                                 pieces: img_bf cast BF16 at :803, krea2_temb
#                                 emitted BF16 at :809)
#   base weights        : BF16   (as stored in the checkpoint)
#   matmul accumulate   : F32    (ops/linear.mojo header: "F32 accumulation
#                                 throughout. BF16/F16 only at the storage
#                                 boundary" — torch bf16 CUDA GEMM does the same)
#   RMSNorm             : F32 internal, weight = raw_scale + 1.0, cast back to
#                         the input dtype at store (krea2_rmsnorm docstring,
#                         mmdit.py RMSNorm.forward)
#   attention           : Q@K^T, softmax and A@V in F32 (q/k/v upcast from their
#                         BF16 storage), result cast back to BF16. This is the
#                         deterministic MATH path == Mojo `sdpa_nomask`, which
#                         krea2_block.mojo documents as the BIT-IDENTICAL gate
#                         path. (The Mojo trainer's cuDNN bf16 flash-padmask
#                         kernel is the same math with a different accumulation
#                         order; it is a value-tolerance path, not a bit path.)
#   RoPE tables         : built with an F64 exponent + F64 angle + mod-2pi range
#                         reduction, then F32 trig — the exact recipe of
#                         _krea2_rope_kernel (krea2_dit.mojo:487-518). Stored F32.
#   grads               : autograd over the BF16 graph, dumped as F32 copies.
#   ENVELOPE            : for block 0 the whole forward+backward is ALSO run with
#                         F32 storage (same CUDA device, same real weights) and
#                         dumped under *_f32env keys, so the Mojo gate can tell
#                         bf16 rounding noise apart from a real bug.
#
# INPUT ACTIVATIONS — REAL, NOT RANDOM (this is load-bearing, not cosmetic)
# --------------------------------------------------------------------------
#   Source: /home/alex/trainings/krea2_eri2_cache_1024/cache.safetensors, the
#   real krea2 training cache (117 samples; clean.<i> [1,16,128,128] VAE latents,
#   context.<i> [1,lt,12,2560] Qwen3-VL features, text_len.<i>).
#     TXT  = txtmlp(txtfusion(zero-pad(context.0 -> LTMAX)))   REAL weights
#     IMG  = first(patchify((1-t)*clean.0_crop + t*noise))     REAL weights
#     COND = first(patchify(clean.1_crop))                     REAL weights, CLEAN
#   Both latents are centre-cropped 128x128 -> 64x64 (= 512px, 32x32 tokens).
#   A DIFFERENT real image supplies the condition so the COND segment is clearly
#   distinguishable from IMG.
#
#   WHY: a krea2 SingleStreamBlock is RMSNorm-gated, so its output magnitude is
#   independent of input SCALE but strongly dependent on input DIRECTION. Fed
#   i.i.d. Gaussian tokens it runs far off-distribution and goes ill-conditioned.
#   MEASURED on this exact fixture (same weights, same seed, same code):
#       random tokens : bf16 vs f32   cos(out)=0.9984   cos(d_x)=0.62
#       real cache    : bf16 vs f32   cos(out)=0.99995  cos(d_x)=0.99999
#   With random tokens a cos>=0.999 gate on the backward is mathematically
#   impossible to pass and a real Mojo bug would be indistinguishable from bf16
#   rounding. Random-token fixtures are therefore NOT acceptable for this block
#   in bf16. (The older CPU oracle
#   serenitymojo/models/krea2/parity/krea2_block_oracle.py gets away with random
#   fills only because it runs in float64.)
#
# SEED
# ----
#   torch.manual_seed(20260730); every draw (flow noise, LoRA A/B, the fixed
#   upstream gradient) uses an explicit torch.Generator(device="cuda") seeded
#   from 20260730. VERIFIED reproducible: two independent runs produced
#   bit-identical values for all 85 multi-element tensors and all scalars. (The
#   file md5 still differs run to run — safetensors serialises the __metadata__
#   dict in Rust HashMap order — but the tensor payload is deterministic.)
#
# SHAPES (512px case, from krea2_omini_layout.mojo main())
# --------------------------------------------------------
#   LTMAX=384  S_IMG=1024  S_COND=1024  LFULL=2432
#   lt is the REAL caption length of the cache sample (sample 0 -> lt=190), so
#   real_len/cond_off/pad_off follow it; all are recorded in the meta_* keys.
#   t = 0.62 (the timestep the target latent is noised at and the modulation
#   vector is built from); the condition segment rides t = 0.
#   FEATURES=6144 HEADS=48 KVHEADS=12 HEADDIM=128 MLPDIM=16384 theta=1e3
#   rope axes [32,48,48]  img/cond grid 32x32 (64x64 latent, patch 2 -> 512px)
#   LoRA rank=16 alpha=16 scale=1.0 (serenitymojo/configs/krea2.json). NOTE the
#   OminiControl configs use r=4/alpha=4 for spatial tasks; the krea2 preset is
#   what our trainer ships, so it is what the oracle gates. Override with
#   KREA2_OMINI_RANK / KREA2_OMINI_ALPHA.
#
# FIXTURE KEYS (see also `python scripts/krea2_omini_torch_oracle.py --manifest`)
# ------------------------------------------------------------------------------
#   INPUTS — everything the Mojo gate must reconstruct byte-for-byte
#     kin_x               [1,LFULL,F] bf16  assembled block input, COMBINED order
#     kin_txt             [1,LTMAX,F] bf16  source-order text tokens (pre-reorder)
#     kin_img             [1,S_IMG,F] bf16  image tokens
#     kin_cond            [1,S_COND,F] bf16 condition tokens
#     kin_t               [1] f32           the timestep fed to the t-branch
#     kin_pos_src         [1,LTMAX+S_IMG+S_COND,3] f32  SOURCE-order positions
#                                           (txt zeros | img grid | cond grid)
#     kin_pos             [1,LFULL,3] f32   COMBINED-order positions (the reorder
#                                           krea2_omini_layout.combined_src_row)
#     kin_cos / kin_sin   [LFULL,64] f32    RoPE tables for kin_pos
#     kin_blk_vec_t       [1,6F] bf16       tproj(tmlp(temb(t)))   — mods(t) input
#     kin_blk_vec_cond    [1,6F] bf16       tproj(tmlp(temb(0)))   — mods(0) input
#     kin_blk_vec_t_f32   [1,6F] f32        same chain run in F32 (seam envelope)
#     kin_blk_vec_cond_f32[1,6F] f32
#     kin_keymask         [LFULL] bool      True on the real prefix, False on pad
#     kin_qnorm/kin_knorm [128] f32         REAL block params (copied for ease)
#     kin_prenorm/kin_postnorm [6144] f32
#     kin_mod_lin         [36864] f32
#     kin_lo_<slot>_A     [rank,in]  bf16   8 slots: wq wk wv gate wo
#     kin_lo_<slot>_B     [out,rank] bf16   mlp_gate mlp_up mlp_down
#     kin_d_out           [1,LFULL,F] bf16  the FIXED upstream gradient
#   FORWARD REFERENCES — the seams the Mojo code checks in order
#
#   ⚠ LoRA-ON vs LoRA-OFF — READ THIS BEFORE GATING ANYTHING ⚠
#   The fixture contains TWO forwards of the same block on the same inputs:
#     * the ADAPTER-ENABLED run (`out, seams = omini_block_forward(..., lora, ...)`)
#       — cond-row LoRA active on all 8 Linears. EVERY `kref_*` key below whose
#       name does NOT end in `_nolora` comes from THIS run. That includes
#       kref_attn_raw: it is the LoRA-**ON** attention output, because q/k/v/gate
#       already carry the cond-row delta by the time SDPA runs. (kref_xm is the
#       one key that is numerically identical either way — it is computed BEFORE
#       any projection — but it is still dumped from the LoRA-on run.)
#     * the ADAPTER-DISABLED run (`out_off, seams_off`, adapters replaced by
#       (None, None)) — the source of every key ending in `_nolora`.
#   Getting this backwards cost a debugging cycle in C2: a LoRA-off Mojo forward
#   cannot match kref_attn_raw, and no threshold change can make it. Use
#   kref_attn_raw_nolora for a LoRA-off attention gate and kref_attn_raw for a
#   LoRA-on one.
#
#     ── from the LoRA-ON run ────────────────────────────────────────────────
#     kref_xm             [1,LFULL,F]  post pre-modulation ((1+prescale)*prenorm(x)
#                                      +preshift), PER-SEGMENT. Pre-projection, so
#                                      LoRA-independent in value.
#     kref_attn_raw       [1,LFULL,F]  LoRA-ON SDPA output, flattened, before the
#                                      sigmoid gate and before wo
#     kref_gated          [1,LFULL,F]  LoRA-ON sdpa * sigmoid(gate) — wo's LoRA input
#     kref_attn           [1,LFULL,F]  LoRA-ON post-attention == wo(sdpa*sigmoid(gate))
#     kref_x1             [1,LFULL,F]  x + pregate*attn (per-segment pregate)
#     kref_xm2            [1,LFULL,F]  post post-modulation, PER-SEGMENT
#     kref_swiglu         [1,LFULL,MLPDIM] silu(mlp_gate)*mlp_up — mlp_down's input
#     kref_ff             [1,LFULL,F]  post-FF == mlp_down(silu(gate)*up)
#     kref_out            [1,LFULL,F]  BLOCK OUTPUT (x1 + postgate*ff)
#     kref_out_img        [1,S_IMG,F]  the IMG row slice of the output (the only
#                                      rows the loss reads — Omini discards cond)
#     ── from the LoRA-OFF run (adapters replaced by (None, None)) ───────────
#     kref_attn_raw_nolora [1,LFULL,F] LoRA-OFF SDPA output — the LoRA-off twin of
#                                      kref_attn_raw (added for C3; C2 had to
#                                      reconstruct it inside the gate)
#     kref_out_nolora     [1,LFULL,F]  LoRA-OFF block output — the C2 gate target
#     kref_out_nolora_img [1,S_IMG,F]  its IMG row slice
#   BACKWARD REFERENCES — torch autograd from loss = (out * kin_d_out).sum()
#     kref_d_x            [1,LFULL,F]  dX for the FULL sequence
#     kref_<slot>_dA      [rank,in]    cond-row LoRA dA, 8 slots
#     kref_<slot>_dB      [out,rank]   cond-row LoRA dB, 8 slots
#     kref_d_blk_vec_t    [1,6F]       grad into mods(t)   — gates the per-segment
#     kref_d_blk_vec_cond [1,6F]       grad into mods(0)     d_mod accumulation
#   ENVELOPE — the same forward+backward re-run in F32 storage on the SAME CUDA
#   device with the SAME real weights, plus the measured bf16-vs-f32 agreement
#     kref_out_f32env kref_d_x_f32env kref_d_blk_vec_{t,cond}_f32env
#     kref_<slot>_dA_f32env  kref_<slot>_dB_f32env
#     env_cos_<key>       [1] f32  measured cos(bf16, f32) for every gateable key
#     env_cos_d_x_{txt,img,cond,pad,prefix}   per-segment breakdown of d_x
#     env_cos_d_x_uniformmod  the isolation probe (see NUMERICS below)
#     PER-SEAM FORWARD ENVELOPES (each key gets its OWN, so no gate ever has to
#     borrow another key's threshold — C2 had to borrow env_cos_out for xm and
#     attn_raw):
#       LoRA-ON  : env_cos_xm env_cos_attn_raw env_cos_gated env_cos_attn
#                  env_cos_x1 env_cos_xm2 env_cos_swiglu env_cos_ff
#                  env_cos_out env_cos_out_img
#       LoRA-OFF : env_cos_attn_raw_nolora env_cos_out_nolora
#                  env_cos_out_nolora_img
#     Every one is cos(bf16 seam, f32 seam) of the ORACLE'S OWN two runs on the
#     same CUDA device with the same real weights — the ceiling for any bf16
#     implementation of that seam. NEVER gate above it; NEVER loosen it.
#     env_cos_<key>_modround  [1] f32  ROUNDING-POINT FREEDOM (informational)
#       cos(this same bf16 forward with the modulate / residual-gate arithmetic
#       done in F32 and rounded ONCE, vs the bf16-chain baseline every kref_*
#       tensor is produced with). Same device, same real weights, same inputs,
#       same seed — the ONLY difference is where bf16 rounding happens, which is
#       a free and equally correct implementation choice (serenitymojo's
#       ops/elementwise.modulate makes the F32-math one). This number is the
#       scale of residual a CORRECT independent bf16 implementation can carry
#       for that seam. It is NOT a threshold and no gate derives one from it —
#       it exists so a 1e-7 miss against env_cos_<key> can be told apart from a
#       bug. MEASURED: at block 27 modround moves `out` by 1-cos = 1.02e-5
#       while env_cos_out_nolora is 1-cos = 1.03e-5, i.e. the rounding point
#       ALONE is enough to sit on top of that envelope.
#
#   ╔════════════════════════════════════════════════════════════════════════╗
#   ║  TWO REFERENCE SCHEDULES — READ THIS BEFORE GATING ANYTHING            ║
#   ╚════════════════════════════════════════════════════════════════════════╝
#   The fixture ships the SAME nine gated seams computed under TWO different,
#   both-correct, bf16 ROUNDING SCHEDULES. Everything else (seed, real krea2
#   weights, real cached activations, layout, mask, RoPE, attention math, LoRA
#   routing, F32 matmul accumulation, CUDA device) is byte-for-byte identical
#   between them. The ONLY difference is WHERE bf16 rounding happens inside
#   `modulate` and `residual_gate`:
#
#     SCHEDULE A — "bf16-chain" (the plain-torch one; keys kref_<key>)
#       (1+scale)*h + shift evaluated in bf16 => THREE roundings per modulate,
#       TWO per residual gate. This is what torch does when you write the
#       expression on bf16 tensors, and it is what the mmdit fidelity check
#       compares against, so these keys stay and prove the hand forward equals
#       the real reference module.
#       Envelope: env_cos_<key>          = cos(A_bf16, F32-storage run)
#
#     SCHEDULE B — "F32-math modulate" (keys kref_<key>_f32mod)   <-- GATED
#       the same algebra evaluated in F32 and rounded ONCE at store.
#       Envelope: env_cos_<key>_f32mod   = cos(B_bf16, the SAME F32-storage run)
#
#   THE MOJO GATES TARGET SCHEDULE B, and the reason is a concrete bug that
#   this section exists to prevent recurring:
#     serenitymojo/ops/elementwise.mojo `_modulate_kernel_bf16` /
#     `_resgate_kernel_bf16` upcast x/scale/shift to F32, do the arithmetic in
#     F32 and cast to bf16 exactly ONCE, at the store ("returns [..., D] (x's
#     dtype; F32 math)"). That is SCHEDULE B. Gating the Mojo forward against
#     SCHEDULE A keys with SCHEDULE A envelopes is a MIS-SPECIFIED CRITERION:
#     env_cos_<key> measures bf16-storage-vs-f32-storage *under schedule A's own
#     rounding*, so it does not bound an implementation that legitimately uses
#     schedule B. The proof is in this fixture: at block 27
#         env_cos_out_img          1-cos = 1.043e-05   (the old threshold)
#         env_cos_out_img_modround 1-cos = 1.097e-05   (A-vs-B distance alone)
#     i.e. merely choosing the F32-math modulate puts a PROVABLY CORRECT
#     implementation outside the key's envelope. Four keys failed at block 27 by
#     1e-7..9e-7 for exactly that reason (kref_out_img, kref_attn_raw_nolora,
#     kref_out_nolora, kref_out_nolora_img). This is a criterion bug, not a
#     tolerance problem, and it is NOT fixed by widening anything.
#
#   NOTE the F32-STORAGE run is shared by both envelopes and that is exact, not
#   an approximation: with F32 storage the two schedules are literally the same
#   sequence of F32 ops (`.float()` and `.to(float32)` are identities there), so
#   there is one and only one F32 reference and both envelopes are measured
#   against it. Same device, same weights, same inputs.
#
#     kref_<key>_f32mod    [same shape/dtype as kref_<key>]
#       FORWARD (13): xm attn_raw gated attn x1 xm2 swiglu ff out out_img
#                     (LoRA-ON) + attn_raw_nolora out_nolora out_nolora_img
#       BACKWARD (19, added by C4): d_x, d_blk_vec_t, d_blk_vec_cond and the 8
#                     <slot>_dA / <slot>_dB pairs.
#     env_cos_<key>_f32mod [1] f32   one per key above — no key is
#       schedule-A-only any more, and no gate has to borrow another key's number.
#
#   ── C4 BACKWARD SPECIFICS (read before gating a gradient) ────────────────
#   (a) UPSTREAM GRADIENT. Every schedule-B backward key is produced with
#       kin_d_out_pz = kin_d_out with rows [real_len:] ZEROED, not kin_d_out.
#       WHY: under per-segment modulation the TXT_pad tail rides the t=0 span
#       (mmdit.py:509-517), so a nonzero pad-row d_out injects pad-row gradient
#       straight into d_blk_vec_cond — the condition temb chain. That reference
#       would be un-matchable AND undesirable: the Mojo production SDPA is the
#       cuDNN flash TAIL-padmask kernel, which masks only KEY columns, so its
#       pad QUERY rows carry masked-out garbage BY DESIGN (krea2_block.mojo:
#       844-850); and the trainer's loss reads the IMG rows only, so the real
#       pad-row d_out is zero anyway. The SCHEDULE-A backward keys are UNCHANGED
#       (still the full d_out) and the difference is shipped MEASURED:
#         env_cos_<key>_padcontrib      cos(schedule-B full-d_out grad,
#                                           schedule-B pad-zeroed grad)
#         env_bit_<key>_padzero_equal   1.0 iff the two are BIT-equal
#       On this fixture d_x[prefix] and all 16 dA/dB are BIT-EQUAL between the
#       two (pad rows never reach the real rows: pad KEY columns get exp(-inf)=0
#       exactly, so dK/dV on pad rows are exactly zero) while d_blk_vec_cond is
#       NOT — i.e. the pad tail's ONLY effect is contaminating the t=0 chunk
#       grads. See the run log for this block's numbers.
#   (b) THE F32 SIDE. kref_<key>_f32env_pz is the F32-STORAGE backward driven by
#       the SAME pad-zeroed gradient — the F32 reference for the early blocks,
#       where the t=0 modulation makes the bf16 backward fragile (see NUMERICS).
#   (c) THE UNPADDED FORMULATION. env_cos_<key>_f32unpad = cos(F32 backward on
#       the REAL PREFIX ONLY — rows [0,real_len), no pad tail, no mask — vs the
#       padded+masked F32 backward). That is the shape the Mojo F32 gate path
#       runs (krea2_block.mojo: real_len == L -> sdpa_nomask, "the F32 parity
#       gate guard"), so this number is the FORMULATION-ONLY component of any
#       residual such a gate reports. It is INFORMATIONAL: it does not bound
#       cross-library F32 GEMM/reduction differences, so it is not a threshold.
#   META (int32/float32 scalars; strings live in safetensors __metadata__)
#     meta_ltmax meta_s_img meta_s_cond meta_lt meta_lfull meta_real_len
#     meta_cond_off meta_pad_off meta_block_index meta_rank meta_features
#     meta_heads meta_kvheads meta_headdim meta_mlpdim meta_seed
#     meta_alpha meta_lora_scale meta_t meta_theta meta_eps
#     meta_cond_delta_h meta_cond_delta_w meta_cond_pos_scale
#
# NUMERICS — WHICH KEYS ARE bf16-GATEABLE (measured, block-dependent)
# --------------------------------------------------------------------
#   The FORWARD is bf16-solid everywhere: env_cos_out = 0.99995 (block 0) and
#   0.99999 (block 27). Gate the forward keys at cos >= 0.999 in bf16.
#
#   The BACKWARD is NOT uniformly bf16-solid, and the reason is the OminiControl
#   mechanism itself. Isolated by env_cos_d_x_uniformmod (same block, same
#   inputs, same everything, only the modulation policy changed):
#       block  0   uniform mods(t) : cos(d_x) = 0.99849
#       block  0   per-segment     : cos(d_x) = 0.49377   <-- collapses
#       block 27   uniform mods(t) : cos(d_x) = 0.99993
#       block 27   per-segment     : cos(d_x) = 0.99993   <-- unaffected
#   Running the condition rows at t=0 roughly doubles the block's output dynamic
#   range at block 0 (|out|max 2673 -> 6370) because krea2's t=0 modulation
#   chunks are the largest in the schedule, and the resulting cancellation in
#   d_x falls below bf16's 8-bit mantissa. This is a property of the METHOD on
#   the REAL weights, not of any implementation.
#   CONSEQUENCE for the Mojo gates:
#     * dA/dB at the late blocks: bf16-gateable at cos >= 0.999.
#     * d_x (and the weaker dA/dB) at the EARLY blocks: gate against the
#       *_f32env keys using the Mojo F32 block path — krea2_block.mojo already
#       documents `real_len == L -> sdpa_nomask` as "BIT-IDENTICAL ... the F32
#       parity gate guard". Use kref_d_x only as a loose value check, with
#       env_cos_d_x as the ceiling.
#     * Never set a threshold above the env_cos_<key> shipped in the fixture —
#       it is the best any bf16 implementation can do.
#   It is also worth flagging to the training work: bf16 gradient noise at the
#   early blocks is a real risk for this vertical, not just a gating nuisance.
#
# WHAT THE MOJO C2 CHUNK SHOULD GATE FIRST
# ----------------------------------------
#   (every kref_* below means the *_f32mod variant against env_cos_*_f32mod —
#    see "TWO REFERENCE SCHEDULES". The schedule-A key is printed as INFO.)
#   1. kin_pos / kin_cos / kin_sin   (pure layout+table math, no weights, exact)
#   2. kref_xm                       (per-segment modulation, LoRA irrelevant —
#                                     this is THE new structural code, and it is
#                                     one elementwise op away from the inputs, so
#                                     a mismatch here is unambiguous)
#   3. kref_attn_raw_nolora          (mask + RoPE + GQA over the new layout, LoRA
#                                     OFF — NOT kref_attn_raw, which is LoRA-ON)
#   4. kref_out_nolora / _img        (whole block, adapter disabled)
#   C3 (cond-row LoRA routing in the forward) then gates, in order:
#     kref_gated -> kref_attn -> kref_out -> kref_out_img, each against its OWN
#     env_cos_<key>, and re-runs the C2 keys plus the CONDLEN=0 bit-equality.
#   C4 gates, in order: kref_d_x -> the 8 dA/dB pairs -> kref_d_blk_vec_t and
#   kref_d_blk_vec_cond (the pair that gates per-segment d_mod), each against
#   its OWN env_cos_<key>_f32mod, with kin_d_out_pz as the upstream gradient.
#
# RUN (never chain after a mojo build with &&)
#   cd /home/alex/mojodiffusion
#   /home/alex/OneTrainer/venv/bin/python scripts/krea2_omini_torch_oracle.py
#
# Env overrides: KREA2_OMINI_BLOCKS="0,27"  KREA2_OMINI_RANK  KREA2_OMINI_ALPHA
#                KREA2_OMINI_OUT_DIR  KREA2_OMINI_SKIP_F32ENV=1

import argparse
import json
import math
import os
import struct
import sys

import torch
import torch.nn.functional as F
from safetensors.torch import safe_open, save_file

# ── constants ────────────────────────────────────────────────────────────────
CKPT = "/home/alex/.serenity/models/checkpoints/krea2-raw.safetensors"
CACHE = os.environ.get(
    "KREA2_OMINI_CACHE",
    "/home/alex/trainings/krea2_eri2_cache_1024/cache.safetensors",
)
TGT_SAMPLE = int(os.environ.get("KREA2_OMINI_TGT_SAMPLE", "0"))
COND_SAMPLE = int(os.environ.get("KREA2_OMINI_COND_SAMPLE", "1"))
MMDIT_SRC = "/home/alex/ai-toolkit/extensions_built_in/diffusion_models/krea2/src"
OUT_DIR = os.environ.get(
    "KREA2_OMINI_OUT_DIR",
    "/home/alex/mojodiffusion/serenitymojo/models/krea2/parity",
)

SEED = 20260730

# ── ROUNDING-POINT SWITCH (see "TWO REFERENCE SCHEDULES" in the header) ──────
# False = SCHEDULE A ("bf16-chain", what every kref_<key> tensor is produced
#   with): seg()/gate_seg() do their arithmetic in the storage dtype, so a bf16
#   run rounds THREE times per modulate and TWICE per residual gate — plain
#   torch semantics, and the schedule the mmdit fidelity check compares against.
# True  = SCHEDULE B ("F32-math modulate", what every kref_<key>_f32mod tensor
#   is produced with): the same algebra evaluated in F32 and rounded ONCE at
#   store. This is what serenitymojo/ops/elementwise.mojo actually computes —
#   _modulate_kernel_bf16 / _resgate_kernel_bf16 upcast to F32 and
#   .cast[bfloat16]() exactly once, at the store — so SCHEDULE B is the one the
#   Mojo gates target.
# BOTH are correct bf16 implementations. env_cos_<key>_modround measures how far
# apart they are; the *_f32mod key/envelope pair is what makes the Mojo
# comparison apples-to-apples instead of cross-schedule.
F32_MOD = False

FEATURES = 6144
HEADS = 48
KVHEADS = 12
HEADDIM = 128
N_REP = HEADS // KVHEADS
HALF = HEADDIM // 2
MLPDIM = 16384
TDIM = 256
THETA = 1e3
EPS = 1e-5
AXES = [HEADDIM - 12 * (HEADDIM // 16), 6 * (HEADDIM // 16), 6 * (HEADDIM // 16)]

# 512px EDIT case (krea2_omini_layout.mojo main()).
LTMAX = 384
GRID = 32                    # 64x64 latent, patch 2 -> 32x32 tokens -> 512px
S_IMG = GRID * GRID          # 1024
S_COND = GRID * GRID         # 1024
LFULL = LTMAX + S_IMG + S_COND      # 2432
# lt is the REAL caption length of the cache sample; the rest follow from it and
# are installed by set_layout() before anything else runs.
LT = 0
COND_OFF = 0
PAD_OFF = 0
REAL_LEN = 0
T_VALUE = 0.62                      # the timestep the target latent is noised at


def set_layout(lt):
    """Install the runtime row layout for this sample (krea2_omini_layout.mojo:
    Krea2OminiLayout(LTMAX, S_IMG, S_COND, lt))."""
    global LT, COND_OFF, PAD_OFF, REAL_LEN
    if not (0 <= lt <= LTMAX):
        raise SystemExit(f"FATAL: cache text_len {lt} outside [0, LTMAX={LTMAX}]")
    LT = lt
    COND_OFF = LT + S_IMG
    PAD_OFF = COND_OFF + S_COND
    REAL_LEN = PAD_OFF

# EDIT condition position policy (intake §1.2, flux_omini.py:128-141 with dh=dw=0)
COND_DELTA_H = 0
COND_DELTA_W = 0
COND_POS_SCALE = 1.0

RANK = int(os.environ.get("KREA2_OMINI_RANK", "16"))
ALPHA = float(os.environ.get("KREA2_OMINI_ALPHA", "16"))
LSCALE = ALPHA / RANK

SLOTS = ["wq", "wk", "wv", "gate", "wo", "mlp_gate", "mlp_up", "mlp_down"]
SLOT_IO = {
    "wq": (FEATURES, HEADS * HEADDIM),
    "wk": (FEATURES, KVHEADS * HEADDIM),
    "wv": (FEATURES, KVHEADS * HEADDIM),
    "gate": (FEATURES, FEATURES),
    "wo": (FEATURES, FEATURES),
    "mlp_gate": (FEATURES, MLPDIM),
    "mlp_up": (FEATURES, MLPDIM),
    "mlp_down": (MLPDIM, FEATURES),
}
# checkpoint key suffix per slot
SLOT_KEY = {
    "wq": "attn.wq.weight",
    "wk": "attn.wk.weight",
    "wv": "attn.wv.weight",
    "gate": "attn.gate.weight",
    "wo": "attn.wo.weight",
    "mlp_gate": "mlp.gate.weight",
    "mlp_up": "mlp.up.weight",
    "mlp_down": "mlp.down.weight",
}


# ── CUDA-ONLY guard (the ground rule) ────────────────────────────────────────
def require_cuda():
    if not torch.cuda.is_available():
        raise SystemExit(
            "FATAL: CUDA is not available. This oracle is CUDA-ONLY by design — "
            "CPU torch is never a parity oracle for the krea2 stack. Refusing to "
            "emit fixtures."
        )
    dev = torch.device("cuda")
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"[cuda] device={name} sm_{cap[0]}{cap[1]} torch={torch.__version__} "
          f"cuda={torch.version.cuda}", flush=True)
    return dev


# ── deterministic attention: F32 scores/softmax/AV, bf16 in/out ──────────────
def attn_f32(q, k, v, keymask, rowmask):
    """q [1,H,L,Dh], k/v [1,KV,L,Dh] (GQA expanded here), bool masks [L].

    Scores/softmax/AV in F32 (== Mojo sdpa_nomask, the bit-gate path). Pad KEY
    columns get -inf; pad QUERY rows are zeroed after the softmax so no NaN can
    leak from an all-masked row (they are don't-care downstream either way).
    Returns [1, L, H*Dh] in q's storage dtype.
    """
    L = q.shape[2]
    kf = k.repeat_interleave(N_REP, dim=1)
    vf = v.repeat_interleave(N_REP, dim=1)
    scale = 1.0 / math.sqrt(HEADDIM)
    scores = torch.matmul(q.float(), kf.float().transpose(-1, -2)) * scale
    bias = torch.zeros(L, dtype=torch.float32, device=q.device)
    bias = bias.masked_fill(~keymask, float("-inf"))
    scores = scores + bias.view(1, 1, 1, L)
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, vf.float())                     # [1,H,L,Dh]
    out = out * rowmask.view(1, 1, L, 1).to(out.dtype)
    out = out.permute(0, 2, 1, 3).reshape(1, L, HEADS * HEADDIM)
    return out.to(q.dtype)


# ── krea2 RMSNorm (F32 internal, weight = raw + 1.0) ─────────────────────────
def rmsnorm(x, raw_scale):
    t = x.float()
    w = raw_scale.float() + 1.0
    t = F.rms_norm(t, (x.shape[-1],), eps=EPS, weight=w)
    return t.to(x.dtype)


# ── RoPE tables mirroring _krea2_rope_kernel exactly ─────────────────────────
def build_krea2_rope(pos, axes, theta, device):
    """pos [rows, 3] f32 -> (cos, sin) each [rows, sum(axes)/2] f32.

    F64 exponent, F64 angle, mod-2pi range reduction, F32 trig at the end —
    the exact sequence of serenitymojo/models/dit/krea2_dit.mojo:487-518.
    """
    rows = pos.shape[0]
    halves = [a // 2 for a in axes]
    cols = []
    sols = []
    log_theta = math.log(theta)
    two_pi = 6.283185307179586476925286766559
    for a, ha in enumerate(halves):
        li = torch.arange(ha, dtype=torch.float64, device=device)
        inv = torch.exp((-li / float(ha)) * log_theta)                # [ha] f64
        angle = pos[:, a].to(torch.float64).unsqueeze(1) * inv.unsqueeze(0)
        k = torch.floor(angle / two_pi + 0.5)
        reduced = (angle - k * two_pi).to(torch.float32)
        cols.append(torch.cos(reduced))
        sols.append(torch.sin(reduced))
    return torch.cat(cols, dim=1), torch.cat(sols, dim=1)


def rope_interleaved(x, cos, sin):
    """x [1,H,L,Dh]; cos/sin [L,Dh/2] -> interleaved rotation, F32 math."""
    xf = x.float()
    c = cos.view(1, 1, cos.shape[0], HALF)
    s = sin.view(1, 1, sin.shape[0], HALF)
    x0 = xf[..., 0::2]
    x1 = xf[..., 1::2]
    o = torch.empty_like(xf)
    o[..., 0::2] = x0 * c - x1 * s
    o[..., 1::2] = x0 * s + x1 * c
    return o.to(x.dtype)


# ── LoRA restricted to the condition rows ────────────────────────────────────
def lin_cond_lora(x, W, A, B, cond_off, s_cond):
    """Frozen base over the WHOLE sequence + low-rank delta on COND ROWS ONLY.

    This is the row-slice form of OminiControl's per-branch matmul list
    (flux_omini.py:254-257 passing adapters[i], with adapter_names
    [None, None, "default"] -> only the condition branch carries the adapter).
    """
    y = x @ W.transpose(0, 1)
    if A is None:
        return y
    xc = x[:, cond_off:cond_off + s_cond]
    d = LSCALE * ((xc @ A.transpose(0, 1)) @ B.transpose(0, 1))
    return torch.cat([y[:, :cond_off], y[:, cond_off:cond_off + s_cond] + d,
                      y[:, cond_off + s_cond:]], dim=1)


# ── the block forward: krea2 SingleStreamBlock + Omini edit layout ───────────
def omini_block_forward(x, blk_vec_t, blk_vec_cond, W, P, lora, cos, sin,
                        keymask, rowmask, cond_off, s_cond):
    """Returns (out, seams dict). Mirrors mmdit.py SingleStreamBlock.forward's
    tuple-vec branch (per-segment mod, split == cond_off) with the LoRA delta
    routed to the cond rows only."""
    L = x.shape[1]

    def mods(vec):
        o = vec + P["mod_lin"].to(vec.dtype)
        return o.reshape(6, FEATURES)

    mt = mods(blk_vec_t)
    mc = mods(blk_vec_cond)

    def seg(h, i):
        """Per-segment (1+scale)*h + shift with the mmdit 2-span split at
        cond_off. Rows [split:] (COND + TXT_pad) take mods(0); pad rows are
        don't-care and mmdit puts them in the same span (mmdit.py:509-517
        comment: 'Padding tokens fall in the t=0 span ... never matter')."""
        if F32_MOD:                       # F32 math, ONE rounding (Mojo's op)
            return torch.cat([
                (1 + mt[i].float()) * h[:, :cond_off].float() + mt[i + 1].float(),
                (1 + mc[i].float()) * h[:, cond_off:].float() + mc[i + 1].float(),
            ], dim=1).to(h.dtype)
        return torch.cat([
            (1 + mt[i]) * h[:, :cond_off] + mt[i + 1],
            (1 + mc[i]) * h[:, cond_off:] + mc[i + 1],
        ], dim=1)

    def gate_seg(h, i):
        return torch.cat([mt[i] * h[:, :cond_off], mc[i] * h[:, cond_off:]], dim=1)

    def res_gate(r, h, i):
        """r + gate*h, per segment (== Mojo ops/elementwise.residual_gate)."""
        if F32_MOD:                       # F32 math, ONE rounding (Mojo's op)
            return torch.cat([
                r[:, :cond_off].float() + mt[i].float() * h[:, :cond_off].float(),
                r[:, cond_off:].float() + mc[i].float() * h[:, cond_off:].float(),
            ], dim=1).to(r.dtype)
        return r + gate_seg(h, i)

    # ── attention branch ─────────────────────────────────────────────────────
    xn = rmsnorm(x, P["prenorm"])
    xm = seg(xn, 0)                                            # prescale/preshift

    q = lin_cond_lora(xm, W["wq"], *lora["wq"], cond_off, s_cond)
    k = lin_cond_lora(xm, W["wk"], *lora["wk"], cond_off, s_cond)
    v = lin_cond_lora(xm, W["wv"], *lora["wv"], cond_off, s_cond)
    g = lin_cond_lora(xm, W["gate"], *lora["gate"], cond_off, s_cond)

    q = q.reshape(1, L, HEADS, HEADDIM).permute(0, 2, 1, 3)
    k = k.reshape(1, L, KVHEADS, HEADDIM).permute(0, 2, 1, 3)
    v = v.reshape(1, L, KVHEADS, HEADDIM).permute(0, 2, 1, 3)

    q = rmsnorm(q, P["qnorm"])
    k = rmsnorm(k, P["knorm"])
    q = rope_interleaved(q, cos, sin)
    k = rope_interleaved(k, cos, sin)

    attn_raw = attn_f32(q, k, v, keymask, rowmask)             # [1,L,F]
    gated = attn_raw * torch.sigmoid(g)
    a = lin_cond_lora(gated, W["wo"], *lora["wo"], cond_off, s_cond)
    x1 = res_gate(x, a, 2)                                     # pregate

    # ── MLP branch ───────────────────────────────────────────────────────────
    xn2 = rmsnorm(x1, P["postnorm"])
    xm2 = seg(xn2, 3)                                          # postscale/postshift
    mg = lin_cond_lora(xm2, W["mlp_gate"], *lora["mlp_gate"], cond_off, s_cond)
    mu = lin_cond_lora(xm2, W["mlp_up"], *lora["mlp_up"], cond_off, s_cond)
    sw = F.silu(mg) * mu
    ff = lin_cond_lora(sw, W["mlp_down"], *lora["mlp_down"], cond_off, s_cond)
    out = res_gate(x1, ff, 5)                                  # postgate

    seams = {
        "xm": xm, "attn_raw": attn_raw, "attn": a, "x1": x1,
        "xm2": xm2, "ff": ff, "out": out,
        # per-slot LoRA INPUTS, kept so the routing check can verify the
        # row-restriction at every one of the 8 seams (not just the block output).
        "lora_in": {"wq": xm, "wk": xm, "wv": xm, "gate": xm, "wo": gated,
                    "mlp_gate": xm2, "mlp_up": xm2, "mlp_down": sw},
    }
    return out, seams


# ── helpers ──────────────────────────────────────────────────────────────────
def cos_sim(a, b):
    af = a.detach().float().reshape(-1)
    bf = b.detach().float().reshape(-1)
    return float(F.cosine_similarity(af, bf, dim=0))


def rel_err(a, b):
    af = a.detach().float()
    bf = b.detach().float()
    d = (af - bf).norm()
    n = bf.norm()
    return float(d / n) if float(n) > 0 else float(d)


def load_block(f, idx, device, dtype):
    W, P = {}, {}
    for s in SLOTS:
        W[s] = f.get_tensor(f"blocks.{idx}.{SLOT_KEY[s]}").to(device=device, dtype=dtype)
    P["qnorm"] = f.get_tensor(f"blocks.{idx}.attn.qknorm.qnorm.scale").to(device).float()
    P["knorm"] = f.get_tensor(f"blocks.{idx}.attn.qknorm.knorm.scale").to(device).float()
    P["prenorm"] = f.get_tensor(f"blocks.{idx}.prenorm.scale").to(device).float()
    P["postnorm"] = f.get_tensor(f"blocks.{idx}.postnorm.scale").to(device).float()
    P["mod_lin"] = f.get_tensor(f"blocks.{idx}.mod.lin").to(device).float()
    return W, P


def krea2_temb(t, dim, device, dtype):
    """mmdit.py temb(): cos-first sinusoidal embedding with tfactor=1e3 prescale,
    period=1e4. Returns [B, 1, dim]."""
    half = dim // 2
    freqs = torch.exp(-math.log(1e4)
                      * torch.arange(half, dtype=torch.float32, device=device) / half)
    args = (t.float() * 1e3)[:, None, None] * freqs
    return torch.cat((torch.cos(args), torch.sin(args)), dim=-1).to(dtype)


def blk_vec_chain(t, C, device, dtype):
    """REAL krea2 modulation chain: temb -> tmlp(Linear,GELU-tanh,Linear) ->
    tproj(GELU-tanh, Linear). Returns [1, 6*FEATURES]."""
    h = krea2_temb(t, TDIM, device, dtype)
    h = F.linear(h, C["tmlp0_w"].to(dtype), C["tmlp0_b"].to(dtype))
    h = F.gelu(h, approximate="tanh")
    h = F.linear(h, C["tmlp2_w"].to(dtype), C["tmlp2_b"].to(dtype))
    h = F.gelu(h, approximate="tanh")
    h = F.linear(h, C["tproj_w"].to(dtype), C["tproj_b"].to(dtype))
    return h.reshape(1, 6 * FEATURES)


def krea2_patchify(lat):
    """[1,16,LH,LW] -> [1, (LH/2)*(LW/2), 64]  (krea2_cache_reader.mojo:70-78)."""
    b, ch, h, w = lat.shape
    x = lat.reshape(b, ch, h // 2, 2, w // 2, 2)
    x = x.permute(0, 2, 4, 1, 3, 5)
    return x.reshape(b, (h // 2) * (w // 2), ch * 4)


def build_real_inputs(ck, device, dt):
    """Build the block-0 input from the REAL krea2 cache through the REAL krea2
    preamble (first / txtfusion / txtmlp). Returns (txt, img, cond, lt, info).

    WHY REAL ACTIVATIONS ARE MANDATORY HERE (measured, not assumed):
      A krea2 SingleStreamBlock is RMSNorm-gated, so its output magnitude is
      independent of the input SCALE but very much depends on the input
      DIRECTION. Fed i.i.d. Gaussian tokens it runs far off-distribution and
      becomes ill-conditioned: on this exact fixture, bf16-vs-f32 agreement was
      cos(out)=0.9984 and cos(d_x)=0.62 — i.e. a cos>=0.999 gate on the backward
      would have been mathematically impossible to pass, and any Mojo mismatch
      would have been indistinguishable from rounding. With the real cached
      activations below the same comparison is cos(out)=0.99995 and
      cos(d_x)=0.99999. Random-token fixtures are therefore NOT acceptable for
      this block at bf16, no matter how well seeded.
      (The older CPU oracle serenitymojo/models/krea2/parity/krea2_block_oracle.py
      gets away with random fills only because it runs in float64.)

    Composition:
      TXT  : cache context.<TGT_SAMPLE> [1,lt,12,2560] real Qwen3-VL features,
             ZERO-PADDED to LTMAX first (train_krea2.mojo:2868-2872 does exactly
             this), then txtfusion -> txtmlp with the real checkpoint weights.
      IMG  : cache clean.<TGT_SAMPLE> [1,16,128,128], centre-cropped to
             [1,16,64,64] (= 512px), flow-noised at t: x_t = (1-t)*clean + t*noise
             (krea2 velocity convention, target = noise - clean), patchified,
             through the real `first`.
      COND : cache clean.<COND_SAMPLE> cropped the same way and left CLEAN (the
             OminiControl condition is never noised, and rides t=0 modulation),
             patchified, through the same real `first`. A DIFFERENT real image is
             used so the COND segment is clearly distinguishable from IMG in the
             fixture (a same-image condition would make the two segments nearly
             identical and weaken the gate).
    """
    sys.path.insert(0, MMDIT_SRC)
    import mmdit  # noqa: E402

    if not os.path.exists(CACHE):
        raise SystemExit(
            f"FATAL: krea2 activation cache not found at {CACHE}. This oracle "
            "refuses to substitute random tokens — see build_real_inputs()."
        )
    c = safe_open(CACHE, framework="pt", device="cpu")
    need = [f"clean.{TGT_SAMPLE}", f"context.{TGT_SAMPLE}",
            f"text_len.{TGT_SAMPLE}", f"clean.{COND_SAMPLE}"]
    missing = [k for k in need if k not in c.keys()]
    if missing:
        raise SystemExit(f"FATAL: cache {CACHE} is missing {missing}")

    lat_t = c.get_tensor(f"clean.{TGT_SAMPLE}").to(device=device, dtype=dt)
    lat_c = c.get_tensor(f"clean.{COND_SAMPLE}").to(device=device, dtype=dt)
    ctx = c.get_tensor(f"context.{TGT_SAMPLE}").to(device=device, dtype=dt)
    lt = int(c.get_tensor(f"text_len.{TGT_SAMPLE}").item())
    LH = LW = GRID * 2                      # 64x64 latent -> 32x32 tokens -> 512px
    if lat_t.shape[-1] < LW or lat_t.shape[-2] < LH:
        raise SystemExit(f"FATAL: cached latent {tuple(lat_t.shape)} smaller than "
                         f"the {LH}x{LW} crop the 512px case needs")
    oh = (lat_t.shape[-2] - LH) // 2
    ow = (lat_t.shape[-1] - LW) // 2
    tgt = lat_t[:, :, oh:oh + LH, ow:ow + LW]
    cnd = lat_c[:, :, oh:oh + LH, ow:ow + LW]

    # real preamble modules, real weights
    first = torch.nn.Linear(64, FEATURES).to(device=device, dtype=dt)
    tf = mmdit.TextFusionTransformer(12, 2560, 20, 4, False, 20)
    tf.load_state_dict({k[len("txtfusion."):]: ck.get_tensor(k)
                        for k in ck.keys() if k.startswith("txtfusion.")},
                       strict=True)
    tf = tf.to(device=device, dtype=dt)
    txtmlp = torch.nn.Sequential(
        mmdit.RMSNorm(2560), torch.nn.Linear(2560, FEATURES),
        torch.nn.GELU(approximate="tanh"), torch.nn.Linear(FEATURES, FEATURES),
    ).to(device=device, dtype=dt)
    with torch.no_grad():
        first.weight.copy_(ck.get_tensor("first.weight"))
        first.bias.copy_(ck.get_tensor("first.bias"))
        txtmlp[1].weight.copy_(ck.get_tensor("txtmlp.1.weight"))
        txtmlp[1].bias.copy_(ck.get_tensor("txtmlp.1.bias"))
        txtmlp[3].weight.copy_(ck.get_tensor("txtmlp.3.weight"))
        txtmlp[3].bias.copy_(ck.get_tensor("txtmlp.3.bias"))
        txtmlp[0].scale.data = ck.get_tensor("txtmlp.0.scale").to(device).float()
        # keep every RMSNorm scale F32 (krea2_stack.mojo:658)
        for mod in tf.modules():
            if isinstance(mod, mmdit.RMSNorm):
                mod.scale.data = mod.scale.data.float()

        g = torch.Generator(device=device).manual_seed(SEED)
        noise = torch.randn(tgt.shape, generator=g, device=device,
                            dtype=torch.float32)
        x_t = ((1.0 - T_VALUE) * tgt.float() + T_VALUE * noise).to(dt)

        img = first(krea2_patchify(x_t))
        cond = first(krea2_patchify(cnd))
        ctx_pad = torch.zeros(1, LTMAX, ctx.shape[2], ctx.shape[3],
                              device=device, dtype=dt)
        ctx_pad[:, :lt] = ctx[:, :lt]
        txt = txtmlp(tf(ctx_pad, mask=None))

    info = {
        "cache": CACHE, "tgt_sample": TGT_SAMPLE, "cond_sample": COND_SAMPLE,
        "lt": lt, "crop": f"{LH}x{LW} at ({oh},{ow})",
        "img_std": float(img.float().std()), "cond_std": float(cond.float().std()),
        "txt_std": float(txt.float().std()),
    }
    del first, tf, txtmlp, lat_t, lat_c, ctx, ctx_pad
    torch.cuda.empty_cache()
    return txt.detach(), img.detach(), cond.detach(), lt, info


def build_positions(device):
    """SOURCE order [TXT zeros(LTMAX) | IMG grid | COND grid] and the COMBINED
    order the trainer feeds, per krea2_omini_layout.combined_src_row()."""
    src = torch.zeros(LTMAX + S_IMG + S_COND, 3, dtype=torch.float32, device=device)
    hi = torch.arange(GRID, device=device, dtype=torch.float32).repeat_interleave(GRID)
    wi = torch.arange(GRID, device=device, dtype=torch.float32).repeat(GRID)
    src[LTMAX:LTMAX + S_IMG, 1] = hi
    src[LTMAX:LTMAX + S_IMG, 2] = wi
    # EDIT condition: delta [0,0], scale 1.0 -> cond grid == img grid.
    ch = hi + COND_DELTA_H
    cw = wi + COND_DELTA_W
    if COND_POS_SCALE != 1.0:
        b = (COND_POS_SCALE - 1.0) / 2.0
        ch = ch * COND_POS_SCALE + b
        cw = cw * COND_POS_SCALE + b
    src[LTMAX + S_IMG:, 1] = ch
    src[LTMAX + S_IMG:, 2] = cw

    idx = torch.empty(LFULL, dtype=torch.long, device=device)
    idx[:LT] = torch.arange(LT, device=device)                                # TXT_real
    idx[LT:COND_OFF] = LTMAX + torch.arange(S_IMG, device=device)             # IMG
    idx[COND_OFF:PAD_OFF] = LTMAX + S_IMG + torch.arange(S_COND, device=device)  # COND
    idx[PAD_OFF:] = LT + torch.arange(LTMAX - LT, device=device)              # TXT_pad
    return src, src[idx], idx


# ── one fixture ──────────────────────────────────────────────────────────────
def run_block(idx, f, C, device, dtype, gen_seed, do_f32env, validate,
              real_in):
    dt = dtype
    print(f"\n===== block {idx}  storage dtype={dt} =====", flush=True)
    W, P = load_block(f, idx, device, dt)

    g = torch.Generator(device=device).manual_seed(gen_seed)

    def rnd(*shape, sc=1.0):
        return (torch.randn(*shape, generator=g, device=device,
                            dtype=torch.float32) * sc)

    # ── inputs: REAL cached activations through the REAL krea2 preamble.
    #    (See build_real_inputs() for why random tokens are not acceptable.)
    txt, img, cond = real_in
    x_full = torch.cat([txt[:, :LT], img, cond, txt[:, LT:]], dim=1).to(dt)
    x_full = x_full.detach().requires_grad_(True)
    assert x_full.shape[1] == LFULL

    t = torch.tensor([T_VALUE], dtype=torch.float32, device=device)
    t0 = torch.zeros_like(t)

    blk_vec_t = blk_vec_chain(t, C, device, dt).detach().requires_grad_(True)
    blk_vec_cond = blk_vec_chain(t0, C, device, dt).detach().requires_grad_(True)
    blk_vec_t_f32 = blk_vec_chain(t, C, device, torch.float32).detach()
    blk_vec_cond_f32 = blk_vec_chain(t0, C, device, torch.float32).detach()

    pos_src, pos_comb, gather = build_positions(device)
    cos, sin = build_krea2_rope(pos_comb, AXES, THETA, device)

    keymask = torch.zeros(LFULL, dtype=torch.bool, device=device)
    keymask[:REAL_LEN] = True
    rowmask = keymask

    # LoRA: A gaussian-small (OminiControl init_lora_weights "gaussian"),
    # B NONZERO so dA is non-degenerate (production init is zero -> dA == 0).
    lora, lora_params = {}, {}
    for s in SLOTS:
        in_f, out_f = SLOT_IO[s]
        A = rnd(RANK, in_f, sc=0.02).to(dt).detach().requires_grad_(True)
        B = rnd(out_f, RANK, sc=0.02).to(dt).detach().requires_grad_(True)
        lora[s] = (A, B)
        lora_params[s] = (A, B)

    d_out = rnd(1, LFULL, FEATURES, sc=0.05).to(dt)

    # ── FIDELITY CHECK against the REAL mmdit.SingleStreamBlock ──────────────
    if validate:
        validate_against_mmdit(x_full, blk_vec_t, blk_vec_cond, W, P, cos, sin,
                               keymask, rowmask, device, dt)

    # ── forward ──────────────────────────────────────────────────────────────
    out, seams = omini_block_forward(
        x_full, blk_vec_t, blk_vec_cond, W, P, lora, cos, sin,
        keymask, rowmask, COND_OFF, S_COND)

    loss = (out.float() * d_out.float()).sum()
    loss.backward()

    print(f"  forward loss           = {float(loss.detach()):.6f}")
    oi = out[:, LT:COND_OFF].detach().float()
    print(f"  out[img rows] mean/std = {float(oi.mean()):.6f} / {float(oi.std()):.6f}")
    print(f"  |d_x| (fro)            = {float(x_full.grad.float().norm()):.6f}")
    nz = sum(int(lora_params[s][0].grad.float().norm() > 0) for s in SLOTS)
    print(f"  nonzero dA slots       = {nz}/8   nonzero dB slots = "
          f"{sum(int(lora_params[s][1].grad.float().norm() > 0) for s in SLOTS)}/8")

    # ── cond-row-only routing evidence ──────────────────────────────────────
    # (a) PER-SEAM: at each of the 8 Linears the LoRA delta must land ONLY on
    #     rows [cond_off, pad_off). Checked on the REAL activations that flow
    #     through the block, bit-exactly.
    with torch.no_grad():
        for s in SLOTS:
            xin = seams["lora_in"][s].detach()
            A, B = lora_params[s][0].detach(), lora_params[s][1].detach()
            y_off = xin @ W[s].transpose(0, 1)
            y_on = lin_cond_lora(xin, W[s], A, B, COND_OFF, S_COND)
            pre_eq = torch.equal(y_on[:, :COND_OFF], y_off[:, :COND_OFF])
            pad_eq = torch.equal(y_on[:, PAD_OFF:], y_off[:, PAD_OFF:])
            cd = float((y_on[:, COND_OFF:PAD_OFF]
                        - y_off[:, COND_OFF:PAD_OFF]).float().abs().max())
            if not (pre_eq and pad_eq):
                raise SystemExit(f"FATAL: LoRA leaked outside cond rows at slot {s}")
            if cd == 0.0:
                raise SystemExit(f"FATAL: LoRA had no effect on cond rows at slot {s}")
        print("  [routing] all 8 seams: txt+img+pad rows bit-equal to base, "
              "cond rows changed  OK")

        # (b) BLOCK OUTPUT: the txt+img rows DO change — not a leak, this is the
        #     method working. LoRA edits the condition rows' K/V, and v1-mode
        #     attention is fully bidirectional (intake §1.4), so the influence
        #     reaches the image rows through attention only. Reported as a number
        #     so the Mojo gate can confirm the same coupling exists.
        off = {s: (None, None) for s in SLOTS}
        # seams_off is the LoRA-OFF twin of `seams`; kref_attn_raw_nolora comes
        # from here so a LoRA-off Mojo forward has a reference it CAN match.
        out_off, seams_off = omini_block_forward(
            x_full.detach(), blk_vec_t.detach(), blk_vec_cond.detach(), W, P,
            off, cos, sin, keymask, rowmask, COND_OFF, S_COND)
        img_delta = float((out_off[:, LT:COND_OFF]
                           - out[:, LT:COND_OFF].detach()).float().abs().max())
        cond_delta = float((out_off[:, COND_OFF:PAD_OFF]
                            - out[:, COND_OFF:PAD_OFF].detach()).float().abs().max())
    print(f"  [coupling] block-out max|delta| img rows  = {img_delta:.6f} "
          f"(via attention only — expected nonzero)")
    print(f"  [coupling] block-out max|delta| cond rows = {cond_delta:.6f}")
    if img_delta == 0.0:
        raise SystemExit("FATAL: condition LoRA never reached the image rows — "
                         "attention coupling is broken")

    # ── SCHEDULE B: the F32-MATH-MODULATE REFERENCE SEAMS ───────────────────
    # Re-run BOTH forwards (adapters on and off) with F32_MOD=True: identical
    # algebra, identical real weights, identical inputs, identical seed,
    # identical device — the ONLY change is that the modulate / residual-gate
    # arithmetic happens in F32 and rounds ONCE at store instead of rounding at
    # every bf16 op. That is a free and equally correct implementation choice,
    # and it is the one serenitymojo makes: ops/elementwise.mojo's
    # _modulate_kernel_bf16 upcasts x/scale/shift to F32, computes
    # (1+sv)*xv+shv in F32 and .cast[bfloat16]()s exactly once, and
    # _resgate_kernel_bf16 does the same for xv+gv*yv.
    #
    # TWO things come out of this run and they must not be confused:
    #   (a) env_cos_<key>_modround — cos(schedule B, schedule A). INFORMATIONAL
    #       only; the measured distance between the two rounding schedules.
    #   (b) kref_<key>_f32mod — the schedule-B REFERENCE SEAMS themselves, which
    #       (with env_cos_<key>_f32mod, computed against the shared F32-storage
    #       run further down) are what the Mojo gates actually compare against.
    #       Before these existed the gates compared a schedule-B implementation
    #       against schedule-A references under schedule-A envelopes, and four
    #       keys "failed" at block 27 by 1e-7..9e-7 purely from that mismatch.
    #       See "TWO REFERENCE SCHEDULES" in the header.
    global F32_MOD

    # ── the PAD-ZEROED upstream gradient (C4) ───────────────────────────────
    # kin_d_out is random on EVERY row, pad tail included. That is harmless for
    # the forward (nothing reads it) and it was harmless for the schedule-A
    # backward keys, which no gate targets. It is NOT harmless for the C4
    # per-segment d_mod: the pad tail sits in the t=0 span (mmdit.py:509-517),
    # so a nonzero pad d_out injects pad-row gradient DIRECTLY into
    # d_blk_vec_cond — the condition temb chain. Two facts make that reference
    # ungateable and unwanted:
    #   * the Mojo production SDPA is the cuDNN flash TAIL-padmask kernel, which
    #     masks only KEY columns; its pad QUERY rows carry masked-out garbage by
    #     design (krea2_block.mojo:844-850). Their d_mod contribution is
    #     meaningless, not merely different.
    #   * the trainer's loss reads the IMG rows only, so the real d_out is zero
    #     on the pad tail anyway.
    # Every SCHEDULE-B backward reference below is therefore produced with
    # d_out_pz (= kin_d_out with rows [real_len:] zeroed, shipped as
    # kin_d_out_pz). The pad-row contribution is not hidden: run 1 below repeats
    # the SAME schedule-B backward with the FULL d_out and the fixture ships the
    # measured difference as env_cos_*_padcontrib / env_bit_*_padzero_equal.
    d_out_pz = d_out.clone()
    d_out_pz[:, REAL_LEN:] = 0

    def _grad_run(Wd, dtype_, dvec, unpad=False, keep_seams=False):
        """One full fwd+bwd of THIS block under the CURRENT F32_MOD schedule.

        Wd/dtype_  : the weight dict and the storage dtype (bf16 or F32).
        dvec       : the upstream gradient, [1, LFULL, F] (sliced when unpad).
        unpad=True : run on the REAL PREFIX only — rows [0, real_len), no pad
                     tail and no mask. Mathematically identical on the real rows
                     to the padded+masked run (masked keys contribute exp(-inf)=0
                     exactly), and it is the shape the Mojo F32 gate path uses
                     (krea2_block.mojo: real_len == L -> sdpa_nomask, the F32
                     parity gate guard). Its distance from the padded F32 run is
                     shipped as env_cos_*_f32unpad so the Mojo F32 gate can tell
                     the FORMULATION component apart from its own residual.
        Returns (grads dict of F32 CPU-ready device tensors, seams or None).
        """
        if unpad:
            xin = x_full.detach()[:, :REAL_LEN].contiguous().to(dtype_)
            cos_u, sin_u = cos[:REAL_LEN], sin[:REAL_LEN]
            km = torch.ones(REAL_LEN, dtype=torch.bool, device=device)
            go = dvec[:, :REAL_LEN].contiguous()
        else:
            xin = x_full.detach().to(dtype_)
            cos_u, sin_u, km = cos, sin, keymask
            go = dvec
        xin = xin.requires_grad_(True)
        bt = blk_vec_t.detach().to(dtype_).requires_grad_(True)
        bc = blk_vec_cond.detach().to(dtype_).requires_grad_(True)
        ll = {}
        for s in SLOTS:
            ll[s] = (lora_params[s][0].detach().to(dtype_).requires_grad_(True),
                     lora_params[s][1].detach().to(dtype_).requires_grad_(True))
        o, sm = omini_block_forward(xin, bt, bc, Wd, P, ll, cos_u, sin_u,
                                    km, km, COND_OFF, S_COND)
        (o.float() * go.float()).sum().backward()
        g = {
            "d_x": xin.grad.detach().float().clone(),
            "d_blk_vec_t": bt.grad.detach().float().clone(),
            "d_blk_vec_cond": bc.grad.detach().float().clone(),
        }
        for s in SLOTS:
            g[s + "_dA"] = ll[s][0].grad.detach().float().clone()
            g[s + "_dB"] = ll[s][1].grad.detach().float().clone()
        sv = None
        if keep_seams:
            sv = {
                "xm": sm["xm"].detach(),
                "attn_raw": sm["attn_raw"].detach(),
                "gated": sm["lora_in"]["wo"].detach(),
                "attn": sm["attn"].detach(),
                "x1": sm["x1"].detach(),
                "xm2": sm["xm2"].detach(),
                "swiglu": sm["lora_in"]["mlp_down"].detach(),
                "ff": sm["ff"].detach(),
                "out": o.detach(),
                "out_img": o.detach()[:, LT:COND_OFF].contiguous(),
            }
        del o, sm, xin, bt, bc, ll
        torch.cuda.empty_cache()
        return g, sv

    modr = {}
    mrk = {}          # kref_<key>_f32mod, kept on device for the envelope below
    try:
        F32_MOD = True
        # run 1 — schedule B, bf16, the FULL (pad-nonzero) d_out. Kept ONLY as
        # the pad-row evidence: everything gated comes from run 2.
        gB_full, _ = _grad_run(W, dt, d_out)
        # run 2 — schedule B, bf16, the PAD-ZEROED d_out: the GATED reference
        # set (forward seams + every backward key).
        gB_pz, mrk = _grad_run(W, dt, d_out_pz, keep_seams=True)
        with torch.no_grad():
            o_mr_off, s_mr_off = omini_block_forward(
                x_full.detach(), blk_vec_t.detach(), blk_vec_cond.detach(),
                W, P, off, cos, sin, keymask, rowmask, COND_OFF, S_COND)
    finally:
        F32_MOD = False
    mrk["attn_raw_nolora"] = s_mr_off["attn_raw"]
    mrk["out_nolora"] = o_mr_off
    mrk["out_nolora_img"] = o_mr_off[:, LT:COND_OFF].contiguous()
    for nm, ka, kb in (
        ("xm", mrk["xm"], seams["xm"]),
        ("attn_raw", mrk["attn_raw"], seams["attn_raw"]),
        ("gated", mrk["gated"], seams["lora_in"]["wo"]),
        ("attn", mrk["attn"], seams["attn"]),
        ("x1", mrk["x1"], seams["x1"]),
        ("xm2", mrk["xm2"], seams["xm2"]),
        ("swiglu", mrk["swiglu"], seams["lora_in"]["mlp_down"]),
        ("ff", mrk["ff"], seams["ff"]),
        ("out", mrk["out"], out),
        ("out_img", mrk["out_img"], out[:, LT:COND_OFF]),
        ("attn_raw_nolora", mrk["attn_raw_nolora"], seams_off["attn_raw"]),
        ("out_nolora", mrk["out_nolora"], out_off),
        ("out_nolora_img", mrk["out_nolora_img"], out_off[:, LT:COND_OFF]),
    ):
        modr[nm] = cos_sim(ka, kb)
    # BACKWARD modround (schedule B vs schedule A) — both sides use the FULL
    # d_out here, so this isolates the ROUNDING SCHEDULE exactly as the forward
    # modround numbers do. Informational; never a threshold.
    BWD_KEYS = ["d_x", "d_blk_vec_t", "d_blk_vec_cond"]
    for s in SLOTS:
        BWD_KEYS += [s + "_dA", s + "_dB"]
    sched_a_g = {
        "d_x": x_full.grad, "d_blk_vec_t": blk_vec_t.grad,
        "d_blk_vec_cond": blk_vec_cond.grad,
    }
    for s in SLOTS:
        sched_a_g[s + "_dA"] = lora_params[s][0].grad
        sched_a_g[s + "_dB"] = lora_params[s][1].grad
    for k in BWD_KEYS:
        modr[k] = cos_sim(gB_full[k], sched_a_g[k])
    del o_mr_off, s_mr_off
    torch.cuda.empty_cache()
    print("  [modround] cos(bf16 with F32-math modulate, bf16 baseline) — the")
    print("             residual a correct implementation may carry purely from")
    print("             choosing a different modulate rounding point:")
    for k in sorted(modr, key=lambda z: modr[z]):
        print(f"      {k:22s} {modr[k]:.9f}   1-cos = {1.0 - modr[k]:.3e}")
    print(f"  [schedule B] retained {len(mrk)} kref_*_f32mod forward seams + "
          f"{len(BWD_KEYS)} backward keys")

    # ── PAD-ROW EVIDENCE (C4 decision, measured — not argued) ───────────────
    # Same schedule, same weights, same inputs, same seed; the ONLY difference
    # is whether the FICTITIOUS pad-tail d_out is zeroed.
    padev = {}
    for k in BWD_KEYS:
        padev[k] = cos_sim(gB_full[k], gB_pz[k])
    padev["d_x_prefix"] = cos_sim(gB_full["d_x"][:, :REAL_LEN],
                                  gB_pz["d_x"][:, :REAL_LEN])
    bit = {}
    bit["d_x_prefix"] = float(torch.equal(gB_full["d_x"][:, :REAL_LEN],
                                          gB_pz["d_x"][:, :REAL_LEN]))
    bit["d_blk_vec_t"] = float(torch.equal(gB_full["d_blk_vec_t"],
                                           gB_pz["d_blk_vec_t"]))
    for s in SLOTS:
        bit[s + "_dA"] = float(torch.equal(gB_full[s + "_dA"], gB_pz[s + "_dA"]))
        bit[s + "_dB"] = float(torch.equal(gB_full[s + "_dB"], gB_pz[s + "_dB"]))
    print("  [pad-row evidence] FULL-d_out vs PAD-ZEROED d_out, schedule B bf16:")
    print(f"      d_x [prefix 0:real_len]      cos={padev['d_x_prefix']:.9f}  "
          f"bit_equal={bool(bit['d_x_prefix'])}")
    print(f"      d_blk_vec_t                  cos={padev['d_blk_vec_t']:.9f}  "
          f"bit_equal={bool(bit['d_blk_vec_t'])}")
    print(f"      d_blk_vec_cond               cos={padev['d_blk_vec_cond']:.9f}"
          f"   <-- the t=0 chain: this is the pad-row contamination")
    nbit = sum(int(bit[s + "_dA"]) + int(bit[s + "_dB"]) for s in SLOTS)
    print(f"      LoRA dA/dB                   {nbit}/16 slots BIT-EQUAL")
    print(f"      d_x [FULL, pad rows too]     cos={padev['d_x']:.9f}")
    # The pad-INCLUDED twin of the two gated d_mod keys. It exists so a gate can
    # DISCRIMINATE without a threshold: a per-segment backward that correctly
    # stops its chunk reduction at real_len must sit closer to kref_d_blk_vec_*
    # _f32mod than to these, and one that runs the reduction to L must not. Both
    # sides are shipped, so the test is "which reference is closer", with no
    # number to invent and no sensitivity to run-to-run nondeterminism.
    padincl = {
        "d_blk_vec_t": gB_full["d_blk_vec_t"],
        "d_blk_vec_cond": gB_full["d_blk_vec_cond"],
    }
    del gB_full
    torch.cuda.empty_cache()

    # ── assemble fixture ─────────────────────────────────────────────────────
    out_t = {}
    for _k, _v in modr.items():           # shipped even with SKIP_F32ENV
        out_t["env_cos_" + _k + "_modround"] = torch.tensor(
            [_v], dtype=torch.float32)
    for _k, _v in padev.items():          # C4 pad-row evidence
        out_t["env_cos_" + _k + "_padcontrib"] = torch.tensor(
            [_v], dtype=torch.float32)
    for _k, _v in bit.items():
        out_t["env_bit_" + _k + "_padzero_equal"] = torch.tensor(
            [_v], dtype=torch.float32)

    def put(k, v):
        out_t[k] = v.detach().to("cpu").contiguous()

    put("kin_x", x_full)
    put("kin_txt", txt.to(dt))
    put("kin_img", img.to(dt))
    put("kin_cond", cond.to(dt))
    put("kin_d_out_seed", torch.tensor([gen_seed], dtype=torch.int32))
    put("kin_t", t)
    put("kin_pos_src", pos_src.unsqueeze(0))
    put("kin_pos", pos_comb.unsqueeze(0))
    put("kin_gather", gather.to(torch.int32))
    put("kin_cos", cos)
    put("kin_sin", sin)
    put("kin_blk_vec_t", blk_vec_t)
    put("kin_blk_vec_cond", blk_vec_cond)
    put("kin_blk_vec_t_f32", blk_vec_t_f32)
    put("kin_blk_vec_cond_f32", blk_vec_cond_f32)
    put("kin_keymask", keymask)
    put("kin_qnorm", P["qnorm"])
    put("kin_knorm", P["knorm"])
    put("kin_prenorm", P["prenorm"])
    put("kin_postnorm", P["postnorm"])
    put("kin_mod_lin", P["mod_lin"])
    for s in SLOTS:
        put(f"kin_lo_{s}_A", lora_params[s][0])
        put(f"kin_lo_{s}_B", lora_params[s][1])
    put("kin_d_out", d_out)
    put("kin_d_out_pz", d_out_pz)     # C4: rows [real_len:] zeroed — the d_out
                                      # EVERY schedule-B backward key uses

    put("kref_xm", seams["xm"])
    put("kref_attn_raw", seams["attn_raw"])
    put("kref_gated", seams["lora_in"]["wo"])
    put("kref_attn", seams["attn"])
    put("kref_x1", seams["x1"])
    put("kref_xm2", seams["xm2"])
    put("kref_swiglu", seams["lora_in"]["mlp_down"])
    put("kref_ff", seams["ff"])
    put("kref_out", seams["out"])
    put("kref_out_img", seams["out"][:, LT:COND_OFF])
    # LoRA-OFF references: the C2 gate targets (layout + per-segment modulation
    # + mask + RoPE, with the adapter disabled). Same inputs, B treated as absent.
    # kref_attn_raw_nolora is the LoRA-OFF attention seam — the twin of
    # kref_attn_raw, which is LoRA-ON (see the LoRA-ON/LoRA-OFF note in the
    # header). Without it C2 had to rebuild the reference inside the gate.
    put("kref_attn_raw_nolora", seams_off["attn_raw"])
    put("kref_out_nolora", out_off)
    put("kref_out_nolora_img", out_off[:, LT:COND_OFF])

    # ── SCHEDULE B references: the SAME nine seams with modulate/residual-gate
    # done in F32 and rounded ONCE at store — the schedule serenitymojo's
    # ops/elementwise kernels implement, and therefore the set the Mojo C2/C3
    # gates compare against. Same seed, same real krea2 weights, same real
    # cached inputs, same bf16 policy everywhere else, same CUDA device. Their
    # thresholds are env_cos_<key>_f32mod, measured below against the SAME
    # F32-storage run the schedule-A envelopes use.
    for _k in sorted(mrk):
        put("kref_" + _k + "_f32mod", mrk[_k])
    # ── SCHEDULE B BACKWARD references (C4) — the GATED backward set.
    # Same schedule-B kernels as the forward seams above, same real weights,
    # same real cached inputs, same seed, same CUDA device; upstream gradient =
    # kin_d_out_pz (see the PAD-ZEROED d_out note above). Thresholds are
    # env_cos_<key>_f32mod, measured below against the F32-storage run driven by
    # the SAME pad-zeroed gradient.
    for _k in BWD_KEYS:
        put("kref_" + _k + "_f32mod", gB_pz[_k])
    for _k in padincl:      # the pad-INCLUDED twins (discrimination reference)
        put("kref_" + _k + "_padincl_f32mod", padincl[_k])

    put("kref_d_x", x_full.grad.float())
    put("kref_d_blk_vec_t", blk_vec_t.grad.float())
    put("kref_d_blk_vec_cond", blk_vec_cond.grad.float())
    for s in SLOTS:
        put(f"kref_{s}_dA", lora_params[s][0].grad.float())
        put(f"kref_{s}_dB", lora_params[s][1].grad.float())

    # ── F32 envelope (same device, same real weights, F32 storage) ───────────
    if do_f32env:
        print("  [envelope] re-running the same block in F32 storage ...", flush=True)
        for s in SLOTS:
            W[s] = W[s].float()
        x32 = x_full.detach().float().requires_grad_(True)
        bt32 = blk_vec_t.detach().float().requires_grad_(True)
        bc32 = blk_vec_cond.detach().float().requires_grad_(True)
        l32 = {}
        for s in SLOTS:
            A = lora_params[s][0].detach().float().requires_grad_(True)
            B = lora_params[s][1].detach().float().requires_grad_(True)
            l32[s] = (A, B)
        o32, s32 = omini_block_forward(x32, bt32, bc32, W, P, l32, cos, sin,
                                       keymask, rowmask, COND_OFF, S_COND)
        (o32 * d_out.float()).sum().backward()
        put("kref_out_f32env", o32)
        put("kref_d_x_f32env", x32.grad)
        put("kref_d_blk_vec_t_f32env", bt32.grad)
        put("kref_d_blk_vec_cond_f32env", bc32.grad)
        for s in SLOTS:
            put(f"kref_{s}_dA_f32env", l32[s][0].grad)
            put(f"kref_{s}_dB_f32env", l32[s][1].grad)
        # ── PER-KEY ENVELOPE ────────────────────────────────────────────────
        # Every gateable key gets its MEASURED bf16-vs-f32 cosine stored in the
        # fixture as env_cos_<key>. That number is the ceiling on what ANY
        # independent bf16 implementation (including ours in Mojo) can achieve
        # against kref_<key>; a gate threshold above it is unpassable by
        # construction. Keys whose envelope is < 0.999 must be gated against the
        # *_f32env reference with the Mojo F32 block path instead
        # (krea2_block.mojo documents that path as "the F32 parity gate guard").
        # NOTE on the two envelope families: with F32 STORAGE the two rounding
        # schedules are literally the same sequence of F32 ops (`.float()` and
        # `.to(float32)` are identities), so o32/s32 IS the F32 reference for
        # both. env_cos_<key> = cos(schedule-A bf16, F32) and
        # env_cos_<key>_f32mod = cos(schedule-B bf16, F32) — same method, same
        # device, same weights, same F32 run. Neither is derived from the other.
        env = {}
        env["out"] = cos_sim(out, o32)
        env["out_img"] = cos_sim(out[:, LT:COND_OFF], o32[:, LT:COND_OFF])
        env["out_f32mod"] = cos_sim(mrk["out"], o32)
        env["out_img_f32mod"] = cos_sim(mrk["out_img"], o32[:, LT:COND_OFF])
        # PER-SEAM FORWARD ENVELOPES (LoRA-ON, both sides). Every forward key the
        # Mojo gates compare against now ships its OWN ceiling, so no gate has to
        # borrow env_cos_out (which C2 had to do for xm and attn_raw).
        for nm, kb, kf in (
            ("xm", seams["xm"], s32["xm"]),
            ("attn_raw", seams["attn_raw"], s32["attn_raw"]),
            ("gated", seams["lora_in"]["wo"], s32["lora_in"]["wo"]),
            ("attn", seams["attn"], s32["attn"]),
            ("x1", seams["x1"], s32["x1"]),
            ("xm2", seams["xm2"], s32["xm2"]),
            ("swiglu", seams["lora_in"]["mlp_down"], s32["lora_in"]["mlp_down"]),
            ("ff", seams["ff"], s32["ff"]),
        ):
            env[nm] = cos_sim(kb, kf)
            if nm in mrk:                 # schedule-B envelope, same F32 side
                env[nm + "_f32mod"] = cos_sim(mrk[nm], kf)
        del s32
        torch.cuda.empty_cache()
        env["d_x"] = cos_sim(x_full.grad, x32.grad)
        env["d_blk_vec_t"] = cos_sim(blk_vec_t.grad, bt32.grad)
        env["d_blk_vec_cond"] = cos_sim(blk_vec_cond.grad, bc32.grad)
        for s in SLOTS:
            env[f"{s}_dA"] = cos_sim(lora_params[s][0].grad, l32[s][0].grad)
            env[f"{s}_dB"] = cos_sim(lora_params[s][1].grad, l32[s][1].grad)
        # d_x, segment by segment — the interesting structure lives here.
        for nm, a0, a1 in (("txt", 0, LT), ("img", LT, COND_OFF),
                           ("cond", COND_OFF, PAD_OFF), ("pad", PAD_OFF, LFULL),
                           ("prefix", 0, PAD_OFF)):
            if a1 > a0:
                env[f"d_x_{nm}"] = cos_sim(x_full.grad[:, a0:a1], x32.grad[:, a0:a1])
        # ISOLATION PROBE: is the bf16 fragility caused by the OminiControl t=0
        # condition modulation, or is it pre-existing krea2? Re-run the SAME
        # block with UNIFORM mods(t) on every row (everything else identical)
        # and compare envelopes. Stored as env_cos_d_x_uniformmod.
        xa = x_full.detach().float().requires_grad_(True)
        oa, _ = omini_block_forward(xa, bt32.detach(), bt32.detach(), W, P,
                                    {s: (None, None) for s in SLOTS}, cos, sin,
                                    keymask, rowmask, COND_OFF, S_COND)
        (oa * d_out.float()).sum().backward()
        Wb16 = {s: W[s].bfloat16() for s in SLOTS}
        xb = x_full.detach().requires_grad_(True)
        ob, _ = omini_block_forward(xb, blk_vec_t.detach(), blk_vec_t.detach(),
                                    Wb16, P, {s: (None, None) for s in SLOTS},
                                    cos, sin, keymask, rowmask, COND_OFF, S_COND)
        (ob.float() * d_out.float()).sum().backward()
        env["d_x_uniformmod"] = cos_sim(xb.grad, xa.grad)
        del xa, xb, oa, ob, Wb16
        torch.cuda.empty_cache()

        # ── LoRA-OFF envelopes ──────────────────────────────────────────────
        # The bf16 LoRA-off run above (out_off / seams_off) re-done in F32 storage
        # on the SAME device with the SAME real weights (W is already F32 here).
        # no_grad: these keys are forward-only references, and the F32 attention
        # scores at L=2432 are ~1.1 GB each — the graph is not worth holding.
        with torch.no_grad():
            o_off32, s_off32 = omini_block_forward(
                x32.detach(), bt32.detach(), bc32.detach(), W, P,
                {s: (None, None) for s in SLOTS}, cos, sin, keymask, rowmask,
                COND_OFF, S_COND)
        env["attn_raw_nolora"] = cos_sim(seams_off["attn_raw"],
                                         s_off32["attn_raw"])
        env["out_nolora"] = cos_sim(out_off, o_off32)
        env["out_nolora_img"] = cos_sim(out_off[:, LT:COND_OFF],
                                        o_off32[:, LT:COND_OFF])
        env["attn_raw_nolora_f32mod"] = cos_sim(mrk["attn_raw_nolora"],
                                                s_off32["attn_raw"])
        env["out_nolora_f32mod"] = cos_sim(mrk["out_nolora"], o_off32)
        env["out_nolora_img_f32mod"] = cos_sim(mrk["out_nolora_img"],
                                               o_off32[:, LT:COND_OFF])
        del o_off32, s_off32
        torch.cuda.empty_cache()

        # ── C4: the F32-storage BACKWARD driven by the PAD-ZEROED d_out ─────
        # This is the F32 side of every schedule-B backward key. With F32
        # storage the two rounding schedules are the identical sequence of F32
        # ops, so this single run is the F32 reference for BOTH schedules — the
        # same argument the forward envelopes rest on. Same device, same real
        # weights (already .float() above), same real cached inputs.
        g32_pz, _ = _grad_run(W, torch.float32, d_out_pz)
        for k in BWD_KEYS:
            put("kref_" + k + "_f32env_pz", g32_pz[k])
            env[k + "_f32mod"] = cos_sim(gB_pz[k], g32_pz[k])
        env["d_x_prefix_f32mod"] = cos_sim(gB_pz["d_x"][:, :REAL_LEN],
                                           g32_pz["d_x"][:, :REAL_LEN])
        # ── C4: the F32-storage backward on the UNPADDED REAL PREFIX ────────
        # Rows [0, real_len) only, no pad tail, no mask — the shape the Mojo F32
        # gate path runs (real_len == L -> sdpa_nomask). Its distance from the
        # padded+masked F32 run above is the FORMULATION-ONLY component of any
        # residual the Mojo F32 gate reports, so the gate can separate "my
        # sequence has no pad tail" from "my implementation differs".
        g32_up, _ = _grad_run(W, torch.float32, d_out_pz, unpad=True)
        env["d_x_f32unpad"] = cos_sim(g32_up["d_x"],
                                      g32_pz["d_x"][:, :REAL_LEN])
        for k in BWD_KEYS:
            if k == "d_x":
                continue
            env[k + "_f32unpad"] = cos_sim(g32_up[k], g32_pz[k])
        del g32_up
        torch.cuda.empty_cache()

        for k, v in env.items():
            out_t["env_cos_" + k] = torch.tensor([v], dtype=torch.float32)
        print("  [envelope] bf16-vs-f32 cosine per key "
              "(this is the CEILING for any bf16 gate):")
        worst = sorted(env.items(), key=lambda kv: kv[1])
        for k, v in worst:
            flag = "  <-- NOT bf16-gateable, use *_f32env" if v < 0.999 else ""
            print(f"      {k:26s} {v:.9f}{flag}")
        print("  [schedule A vs B envelopes] the number each Mojo gate now uses")
        print("             is the _f32mod column (schedule B == the Mojo math):")
        print(f"      {'key':22s} {'1-env(A, bf16-chain)':>22s} "
              f"{'1-env(B, f32mod)':>18s} {'1-modround(A vs B)':>20s}")
        for k in sorted(mrk):
            print(f"      {k:22s} {1.0 - env[k]:22.4e} "
                  f"{1.0 - env[k + '_f32mod']:18.4e} "
                  f"{1.0 - modr[k]:20.4e}")
        print("  [C4 backward envelopes] the numbers the C4 gate uses. env(B) is")
        print("             cos(schedule-B bf16, F32) — the bf16 ceiling; f32unpad")
        print("             is cos(F32 unpadded prefix, F32 padded+masked) — the")
        print("             FORMULATION-only component for the Mojo F32 path:")
        print(f"      {'key':22s} {'1-env(B,f32mod)':>18s} "
              f"{'1-env(f32unpad)':>18s} {'1-padcontrib':>14s}")
        for k in BWD_KEYS:
            uk = k + "_f32unpad"
            print(f"      {k:22s} {1.0 - env[k + '_f32mod']:18.4e} "
                  f"{(1.0 - env[uk]) if uk in env else float('nan'):18.4e} "
                  f"{1.0 - padev[k]:14.4e}")
        print(f"      {'d_x [prefix]':22s} {1.0 - env['d_x_prefix_f32mod']:18.4e} "
              f"{1.0 - env['d_x_f32unpad']:18.4e} "
              f"{1.0 - padev['d_x_prefix']:14.4e}")
        print(f"  [isolation] cos(d_x) with UNIFORM mods(t) on every row = "
              f"{env['d_x_uniformmod']:.9f}   vs {env['d_x']:.9f} per-segment "
              f"-> the t=0 condition modulation is "
              f"{'THE' if env['d_x_uniformmod'] - env['d_x'] > 0.05 else 'not the'}"
              f" cause of any bf16 d_x fragility here")
        del o32, x32, bt32, bc32, l32
        torch.cuda.empty_cache()

    # ── meta ─────────────────────────────────────────────────────────────────
    def mi(k, v):
        out_t["meta_" + k] = torch.tensor([int(v)], dtype=torch.int32)

    def mf(k, v):
        out_t["meta_" + k] = torch.tensor([float(v)], dtype=torch.float32)

    mi("ltmax", LTMAX); mi("s_img", S_IMG); mi("s_cond", S_COND); mi("lt", LT)
    mi("lfull", LFULL); mi("real_len", REAL_LEN); mi("cond_off", COND_OFF)
    mi("pad_off", PAD_OFF); mi("block_index", idx); mi("rank", RANK)
    mi("features", FEATURES); mi("heads", HEADS); mi("kvheads", KVHEADS)
    mi("headdim", HEADDIM); mi("mlpdim", MLPDIM); mi("seed", gen_seed)
    mi("grid", GRID); mi("cond_delta_h", COND_DELTA_H); mi("cond_delta_w", COND_DELTA_W)
    mi("tgt_sample", TGT_SAMPLE); mi("cond_sample", COND_SAMPLE)
    mf("alpha", ALPHA); mf("lora_scale", LSCALE); mf("t", float(t[0]))
    mf("theta", THETA); mf("eps", EPS); mf("cond_pos_scale", COND_POS_SCALE)

    meta = {
        "oracle": "krea2_omini_torch_oracle.py",
        "device": "CUDA-ONLY (cpu torch is never a parity oracle)",
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "checkpoint": CKPT,
        "checkpoint_real": os.path.realpath(CKPT),
        "block_index": str(idx),
        "reference_module": f"{MMDIT_SRC}/mmdit.py SingleStreamBlock",
        "storage_dtype": str(dt),
        "accum": "f32 (matmul accumulate, rmsnorm internal, attention softmax)",
        "layout": "[TXT_real(lt) | IMG | COND | TXT_pad]  (krea2_omini_layout.mojo)",
        "modulation": "rows [0,cond_off) -> mods(t); rows [cond_off,LFULL) -> mods(0)",
        "rounding_schedule_A": (
            "kref_<key> + env_cos_<key>: modulate/residual_gate evaluated in the "
            "bf16 STORAGE dtype (3 roundings per modulate, 2 per residual gate) "
            "— plain torch semantics; this is the schedule the mmdit fidelity "
            "check validates the hand forward against"
        ),
        "rounding_schedule_B": (
            "kref_<key>_f32mod + env_cos_<key>_f32mod: same algebra evaluated in "
            "F32 and rounded ONCE at store — the schedule serenitymojo's "
            "ops/elementwise.mojo _modulate_kernel_bf16/_resgate_kernel_bf16 "
            "implement, hence the schedule the Mojo C2/C3 gates target"
        ),
        "gates_target": (
            "SCHEDULE B (*_f32mod). Reason: env_cos_<key> measures bf16-vs-f32 "
            "storage UNDER SCHEDULE A's OWN ROUNDING, so it does not bound a "
            "correct implementation that uses schedule B. At block 27 "
            "env_cos_out_img is 1-cos=1.043e-05 while env_cos_out_img_modround "
            "(the A-vs-B distance alone) is 1.097e-05, i.e. choosing the "
            "F32-math modulate ALONE puts a provably-correct implementation "
            "outside the key's envelope. Four keys failed at block 27 by "
            "1e-7..9e-7 for exactly that reason. Mis-specified criterion, not a "
            "tolerance problem."
        ),
        "shared_f32_reference": (
            "with F32 storage the two schedules are the identical sequence of "
            "F32 ops, so env_cos_<key> and env_cos_<key>_f32mod are both "
            "measured against the SAME F32-storage run on this device"
        ),
        "schedule_B_keys": (
            "FORWARD: xm attn_raw gated attn x1 xm2 swiglu ff out out_img "
            "attn_raw_nolora out_nolora out_nolora_img. BACKWARD (added by C4): "
            "d_x, d_blk_vec_t, d_blk_vec_cond and the 8 <slot>_dA/<slot>_dB "
            "pairs. Every one ships kref_<key>_f32mod AND env_cos_<key>_f32mod; "
            "the backward keys also ship kref_<key>_f32env_pz (the F32-storage "
            "reference) and env_cos_<key>_f32unpad (the unpadded-prefix "
            "formulation distance). NOTHING is schedule-A-only any more."
        ),
        "backward_d_out": (
            "every SCHEDULE-B backward key is produced with kin_d_out_pz = "
            "kin_d_out with rows [real_len:] zeroed. Reason: under per-segment "
            "modulation the pad tail rides the t=0 span, so a fictitious "
            "pad-row d_out injects pad-row gradient straight into "
            "d_blk_vec_cond; the Mojo production SDPA is the cuDNN flash "
            "TAIL-padmask kernel whose pad QUERY rows are masked-out garbage by "
            "design, and the trainer's loss reads the IMG rows only (real pad "
            "d_out == 0). The SCHEDULE-A backward keys keep the full d_out "
            "unchanged, and the difference is shipped measured as "
            "env_cos_<key>_padcontrib / env_bit_<key>_padzero_equal."
        ),
        "lora_routing": "condition rows only (OminiControl trainer.py:57)",
        "cond_position": f"delta [{COND_DELTA_H},{COND_DELTA_W}] scale {COND_POS_SCALE} (EDIT)",
        "seed": str(gen_seed),
        "activation_source": (
            f"REAL cache {CACHE} (clean.{TGT_SAMPLE} noised at t={T_VALUE} -> IMG, "
            f"clean.{COND_SAMPLE} clean -> COND, context.{TGT_SAMPLE} -> TXT) "
            f"through the REAL krea2 preamble first/txtfusion/txtmlp"
        ),
        "why_real_activations": (
            "iid-random tokens drive the block off-distribution: bf16-vs-f32 was "
            "cos(out)=0.9984 / cos(d_x)=0.62; with these real activations it is "
            "cos(out)=0.99995 / cos(d_x)=0.99999 (measured, see run log)"
        ),
    }
    path = os.path.join(OUT_DIR, f"krea2_omini_block{idx:02d}_oracle.safetensors")
    os.makedirs(OUT_DIR, exist_ok=True)
    save_file(out_t, path, metadata=meta)
    sz = os.path.getsize(path)
    print(f"  WROTE {path}  ({len(out_t)} tensors, {sz/1e6:.1f} MB)", flush=True)

    del W, P, x_full, out, seams, lora, lora_params, d_out, mrk
    del d_out_pz, gB_pz
    torch.cuda.empty_cache()
    return path, out_t


# ── fidelity: hand forward vs the REAL mmdit module ──────────────────────────
def validate_against_mmdit(x, blk_vec_t, blk_vec_cond, W, P, cos, sin,
                           keymask, rowmask, device, dt):
    sys.path.insert(0, MMDIT_SRC)
    import mmdit  # noqa: E402

    # mmdit.attention() pins the cuDNN SDPA backend, which has no F32 kernel for
    # these shapes and would silently differ in accumulation order from the Mojo
    # bit-gate path. Swap in THIS file's deterministic F32 attention so the
    # comparison isolates the layout / per-segment / LoRA code under test.
    def _attn(q, k, v, mask=None, scale=None, gqa=False):
        return attn_f32(q, k, v, keymask, rowmask)

    orig = mmdit.attention
    mmdit.attention = _attn
    try:
        blk = mmdit.SingleStreamBlock(FEATURES, HEADS, multiplier=1, bias=False,
                                      kvheads=KVHEADS).to(device=device, dtype=dt)
        # The module-wide .to(bf16) would also demote the four RMSNorm scales,
        # but the Mojo resident base keeps them F32 exactly as the checkpoint
        # stores them (krea2_stack.mojo:658 "qnorm/knorm/prenorm/postnorm scales
        # F32, mod.lin bf16"). Restore F32 so the reference matches our compute
        # path; mmdit's RMSNorm.forward does self.scale.float() regardless.
        for _p in (blk.prenorm, blk.postnorm, blk.attn.qknorm.qnorm,
                   blk.attn.qknorm.knorm):
            _p.scale.data = _p.scale.data.float()
        with torch.no_grad():
            blk.attn.wq.weight.copy_(W["wq"]); blk.attn.wk.weight.copy_(W["wk"])
            blk.attn.wv.weight.copy_(W["wv"]); blk.attn.gate.weight.copy_(W["gate"])
            blk.attn.wo.weight.copy_(W["wo"])
            blk.attn.qknorm.qnorm.scale.copy_(P["qnorm"])
            blk.attn.qknorm.knorm.scale.copy_(P["knorm"])
            blk.prenorm.scale.copy_(P["prenorm"]); blk.postnorm.scale.copy_(P["postnorm"])
            blk.mod.lin.copy_(P["mod_lin"])
            # SwiGLU's mlpdim is derived from `multiplier`; force the real 16384.
            blk.mlp.gate = torch.nn.Linear(FEATURES, MLPDIM, bias=False)
            blk.mlp.up = torch.nn.Linear(FEATURES, MLPDIM, bias=False)
            blk.mlp.down = torch.nn.Linear(MLPDIM, FEATURES, bias=False)
            blk.mlp.to(device=device, dtype=dt)
            blk.mlp.gate.weight.copy_(W["mlp_gate"]); blk.mlp.up.weight.copy_(W["mlp_up"])
            blk.mlp.down.weight.copy_(W["mlp_down"])

        # mmdit's ropeapply consumes a [b, n, Dh/2, 2, 2] rotation stack; build it
        # from OUR cos/sin so both sides rotate identically.
        L = x.shape[1]
        fr = torch.stack([cos, -sin, sin, cos], dim=-1).reshape(1, L, HALF, 2, 2)
        no_lora = {s: (None, None) for s in SLOTS}

        with torch.no_grad():
            # (a) uniform modulation: mmdit single-vec vs ours with both vecs == t
            ref_a = blk(x.detach(), blk_vec_t.detach().reshape(1, 1, 6 * FEATURES),
                        fr, None)
            mine_a, _ = omini_block_forward(
                x.detach(), blk_vec_t.detach(), blk_vec_t.detach(), W, P,
                no_lora, cos, sin, keymask, rowmask, COND_OFF, S_COND)
            # (b) PER-SEGMENT: mmdit tuple-vec (vec, refvec, split) vs ours
            ref_b = blk(x.detach(),
                        (blk_vec_t.detach().reshape(1, 1, 6 * FEATURES),
                         blk_vec_cond.detach().reshape(1, 1, 6 * FEATURES),
                         COND_OFF),
                        fr, None)
            mine_b, _ = omini_block_forward(
                x.detach(), blk_vec_t.detach(), blk_vec_cond.detach(), W, P,
                no_lora, cos, sin, keymask, rowmask, COND_OFF, S_COND)

        for tag, r, m in (("uniform-mod", ref_a, mine_a),
                          ("per-segment-mod", ref_b, mine_b)):
            c = cos_sim(r[:, :REAL_LEN], m[:, :REAL_LEN])
            e = rel_err(m[:, :REAL_LEN], r[:, :REAL_LEN])
            eq = torch.equal(r[:, :REAL_LEN], m[:, :REAL_LEN])
            print(f"  [fidelity vs REAL mmdit.SingleStreamBlock] {tag:16s} "
                  f"cos={c:.9f} rel={e:.3e} bit_equal={eq}")
            if c < 0.9999:
                raise SystemExit(
                    f"FATAL: hand forward diverges from the reference module "
                    f"({tag}, cos={c})")
        # the two mmdit modes must differ on the cond rows (proof the per-segment
        # branch is actually doing something)
        d = float((ref_a[:, COND_OFF:PAD_OFF] - ref_b[:, COND_OFF:PAD_OFF])
                  .float().abs().max())
        print(f"  [fidelity] reference cond-row max|mods(t) - mods(0)| = {d:.6f}")
        if d == 0.0:
            raise SystemExit("FATAL: per-segment modulation had no effect")
        del blk, ref_a, ref_b, mine_a, mine_b
        torch.cuda.empty_cache()
    finally:
        mmdit.attention = orig


def print_manifest(path):
    with safe_open(path, framework="pt") as f:
        md = f.metadata() or {}
        print(f"\n### {path}  ({os.path.getsize(path)/1e6:.1f} MB)")
        for k, v in sorted(md.items()):
            print(f"    [meta] {k} = {v}")
        for k in sorted(f.keys()):
            t = f.get_slice(k)
            print(f"    {k:34s} {str(t.get_dtype()):10s} {list(t.get_shape())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", action="store_true",
                    help="print the key list of the existing fixtures and exit")
    args = ap.parse_args()

    blocks = [int(b) for b in os.environ.get("KREA2_OMINI_BLOCKS", "0,27").split(",")]
    if args.manifest:
        for b in blocks:
            p = os.path.join(OUT_DIR, f"krea2_omini_block{b:02d}_oracle.safetensors")
            if os.path.exists(p):
                print_manifest(p)
        return

    device = require_cuda()
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(False)

    if not os.path.exists(CKPT):
        raise SystemExit(f"FATAL: krea2 checkpoint not found at {CKPT}")
    print(f"[ckpt] {CKPT}\n       -> {os.path.realpath(CKPT)}", flush=True)

    torch.cuda.reset_peak_memory_stats()
    written = []
    with safe_open(CKPT, framework="pt", device="cpu") as f:
        C = {
            "tmlp0_w": f.get_tensor("tmlp.0.weight").to(device),
            "tmlp0_b": f.get_tensor("tmlp.0.bias").to(device),
            "tmlp2_w": f.get_tensor("tmlp.2.weight").to(device),
            "tmlp2_b": f.get_tensor("tmlp.2.bias").to(device),
            "tproj_w": f.get_tensor("tproj.1.weight").to(device),
            "tproj_b": f.get_tensor("tproj.1.bias").to(device),
        }
        txt, img, cond, lt, info = build_real_inputs(f, device, torch.bfloat16)
        set_layout(lt)
        print(f"[input] REAL cache {info['cache']}")
        print(f"        target=clean.{TGT_SAMPLE} (noised at t={T_VALUE}) "
              f"cond=clean.{COND_SAMPLE} (clean)  crop {info['crop']}")
        print(f"        token std: txt={info['txt_std']:.4f} "
              f"img={info['img_std']:.4f} cond={info['cond_std']:.4f}")
        print(f"[layout] LTMAX={LTMAX} S_IMG={S_IMG} S_COND={S_COND} lt={LT} "
              f"LFULL={LFULL} real_len={REAL_LEN} cond_off={COND_OFF} "
              f"pad_off={PAD_OFF}")
        print(f"[lora] rank={RANK} alpha={ALPHA} scale={LSCALE} "
              f"(cond rows [{COND_OFF},{PAD_OFF}) only)")

        skip_env = os.environ.get("KREA2_OMINI_SKIP_F32ENV") == "1"
        for i, b in enumerate(blocks):
            p, _ = run_block(b, f, C, device, torch.bfloat16, SEED + b,
                             do_f32env=(not skip_env),
                             validate=True, real_in=(txt, img, cond))
            written.append(p)

    peak_a = torch.cuda.max_memory_allocated() / 1e9
    peak_r = torch.cuda.max_memory_reserved() / 1e9
    print(f"\n[vram] peak allocated = {peak_a:.3f} GB   peak reserved = {peak_r:.3f} GB")
    for p in written:
        print(f"[fixture] {p}  {os.path.getsize(p)/1e6:.1f} MB")
    print("OK — CUDA oracle complete.")


if __name__ == "__main__":
    main()

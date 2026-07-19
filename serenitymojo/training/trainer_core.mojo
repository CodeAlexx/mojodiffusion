# training/trainer_core.mojo — SHARED trainer-driver machinery (audit item 3).
#
# The model-AGNOSTIC skeleton every LoRA driver's main loop repeats: checkpoint
# path naming, rolling-retention prune, the FULL-vs-WARM resume probe + guard, and
# the micro-batch grad-accumulation window. Extracted VERBATIM from the krea2
# driver (models/krea2/train_krea2.mojo — the fleet's most complete driver, the
# reference shape) so the next drivers migrated in this wave consume ONE copy
# instead of re-drifting their own. Model-specific bits (the save-key prefixes,
# the `..._<model>_lora` stem, the resume-scope layout key, the log tag) enter as
# params; a driver keeps a thin wrapper that binds its values and forwards here, so
# its call sites and log output are byte-identical to the pre-extraction binary.
#
# Ownership boundary (skill: mojo-shared-code-import): this holds ONLY the shared
# driver skeleton. The optimizer arms (host AdamW / AdamW-device / Automagic3-
# device), the per-sample forward/backward, the save-key layout, fp8-resident, EMA,
# and perf records stay in the product driver — they are not shared machinery.
#
# The grad-accum window WRAPS training/grad_accum.mojo's SUM/MEAN/zero primitives
# (it does not duplicate them): SUM each micro-step, MEAN (÷window) at the boundary,
# recompute the grad norm from the meaned host grads. accum_steps==1 makes every
# step a boundary → byte-identical to the per-step path.
#
# Mojo 1.0.0b1.

from std.collections import List
from std.math import sqrt
from std.gpu.host import DeviceContext

from serenitymojo.training.train_config import TrainConfig
from serenitymojo.training.grad_accum import (
    accumulate_grad_group, scale_grad_group, zeros_like_group,
)
from serenitymojo.io.ffi import sys_remove
from serenitymojo.training.lora_save import (
    load_lora_train_state_meta, lora_train_state_has_moments,
)


# ── checkpoint path naming ────────────────────────────────────────────────────
# `<workspace>/checkpoints/<stem>_<step>.safetensors`, where stem = cfg.save_filename_prefix
# when set, else the model's `stem_fallback` (e.g. `<name>_krea2_lora`). Shared by
# the periodic save, the final save, and the retention prune (below).
def trainer_lora_save_path(
    cfg: TrainConfig, stem_fallback: String, step: Int
) raises -> String:
    var stem = cfg.save_filename_prefix if cfg.save_filename_prefix != String("") else stem_fallback
    return cfg.workspace_dir + String("/checkpoints/") + stem + String("_") + String(step) + String(".safetensors")


# ── resume-scope guard hash (cache identity, masked to 24 bits) ───────────────
# FNV-seeded rolling hash over the cache path, masked so it round-trips exactly
# through an F32 __meta__ field. Stable grouping key, not a cryptographic hash.
def trainer_cache_hash24(cache_path: String) -> Float32:
    var h = UInt64(2166136261) % UInt64(1000000007)
    var bytes = cache_path.as_bytes()
    for i in range(cache_path.byte_length()):
        h = ((h * UInt64(131)) + UInt64(bytes[i])) % UInt64(1000000007)
    return Float32(Int(h & UInt64(0xFFFFFF)))


# ── resume-scope guard meta vector (written into the `.state` __meta__) ───────
# meta = [step_t, seed_base, n_samples, layout_key, cache_hash24]. `layout_key` is
# the model's token-layout knob (e.g. krea2 LTMAX) — a change silently reshapes the
# padded sequence, so it is guarded on resume. F32 holds every field exactly for
# realistic values (integers < 2^24); the hash is 24-bit so it round-trips.
def trainer_state_meta(
    step_t: Int, seed_base: UInt64, n_samples: Int, layout_key: Int, cache_path: String
) raises -> List[Float32]:
    var m = List[Float32]()
    m.append(Float32(step_t))
    m.append(Float32(Int(seed_base)))   # exact for seeds < 2^24 (warn-only guard)
    m.append(Float32(n_samples))
    m.append(Float32(layout_key))
    m.append(trainer_cache_hash24(cache_path))
    return m^


# ── rolling checkpoint retention, pruned AFTER the new save ───────────────────
# Called once per periodic save, AFTER the new checkpoint is durable (a prior good
# checkpoint is never removed before its replacement exists). Deletes the pruned
# step's `.safetensors` AND its `.safetensors.state` / `.a3state.safetensors`
# sidecars together, so resume state never outlives (or orphans) its weights.
#
# keep count:
#   cfg.save_max_keep > 0  → config-driven: keep EXACTLY the newest N (no milestone
#                            exemption — this is the user-facing rolling-retention).
#   cfg.save_max_keep == 0 → fall back to `keep_default`, which additionally
#                            preserves every `milestone`-th checkpoint. 0 on BOTH =
#                            keep all.
# The keep-count DECISION alone (no path building, no I/O): returns the step
# whose checkpoint should now be pruned, or -1 if none. Extracted from the body
# below VERBATIM so the reference trainer-policy drivers — whose on-disk naming differs from
# krea2's `<workspace>/<stem>_<step>` scheme (they name `<base>_step<step>` via
# serenity_step_lora_path, with a `.state.safetensors` sidecar) — reuse the SAME
# retention math while building the pruned path with their OWN step-path helper
# (paired with trainer_prune_step_checkpoint). keep_default/milestone==0 (their
# wiring) means: prune ONLY when cfg.save_max_keep>0, i.e. keep-all is unchanged
# until the webui sets save_max_keep — audit item #4's exact intent.
def trainer_prune_target_step(
    cfg: TrainConfig, saved_step: Int, keep_default: Int, milestone: Int,
) -> Int:
    var config_driven = cfg.save_max_keep > 0
    var keep_n = cfg.save_max_keep
    if keep_default > 0:
        if not config_driven:
            keep_n = keep_default
    if keep_n <= 0 or cfg.save_every <= 0:
        return -1
    var old_step = saved_step - keep_n * cfg.save_every
    if old_step <= 0:
        return -1
    # milestone preservation applies ONLY to the default (non-config) path; an
    # explicit config keep-N keeps EXACTLY the newest N.
    if not config_driven:
        if milestone > 0:
            if old_step % milestone == 0:
                return -1
    return old_step


def trainer_prune_old_checkpoints(
    cfg: TrainConfig, saved_step: Int, stem_fallback: String,
    keep_default: Int, milestone: Int,
) raises:
    var old_step = trainer_prune_target_step(cfg, saved_step, keep_default, milestone)
    if old_step <= 0:
        return
    var op = trainer_lora_save_path(cfg, stem_fallback, old_step)
    if sys_remove(op) == 0:
        print("  [prune] removed old checkpoint ->", op)
    var op_state = op + String(".state")
    if sys_remove(op_state) == 0:
        print("  [prune] removed old checkpoint state ->", op_state)
    var op_a3 = op + String(".a3state.safetensors")
    if sys_remove(op_a3) == 0:
        print("  [prune] removed old A3 state ->", op_a3)


# ── prune a single step-numbered checkpoint by its FULLY-BUILT path ───────────
# The removal + the exact `[prune]` log strings, in ONE place (the webui parses
# these strings, so they MUST stay byte-identical to trainer_prune_old_checkpoints
# above). For the reference trainer-policy drivers: the caller pairs trainer_prune_target_step
# (the shared keep decision) with its OWN step-path helper to build `old_path`,
# then hands it here. Removes the checkpoint and its optimizer-state sidecar;
# a missing sidecar (the LoKr/LoHa/DoRA/OFT arms write none) is a silent no-op.
def trainer_prune_step_checkpoint(
    old_path: String, state_sidecar_suffix: String,
) raises:
    if sys_remove(old_path) == 0:
        print("  [prune] removed old checkpoint ->", old_path)
    var op_state = old_path + state_sidecar_suffix
    if sys_remove(op_state) == 0:
        print("  [prune] removed old checkpoint state ->", op_state)


# ── FULL-state resume path resolution ─────────────────────────────────────────
# The best FULL-state path to resume from. If `path` itself carries AdamW moments,
# use it. Else if the `<path><sidecar_suffix>` sibling carries moments, use that.
# Else return `path` unchanged (a genuine weights-only / warm resume). `probe_prefix`
# is the model's first-adapter key (moments are probed under it). `sidecar_suffix`
# names the model's save-state sibling: it defaults to krea2's `.state` (whose PEFT
# path already ends `.safetensors`, so the sibling is `<ckpt>.safetensors.state`); a
# driver whose sidecar naming differs passes its own (sd35: `.state.safetensors`).
def trainer_resolve_resume_path(
    path: String, probe_prefix: String, sidecar_suffix: String = String(".state")
) raises -> String:
    if lora_train_state_has_moments(path, probe_prefix):
        return path
    var sib = path + sidecar_suffix
    if lora_train_state_has_moments(sib, probe_prefix):
        return sib
    return path


# ── loud WARM-resume banner (AdamW moments zeroed) ────────────────────────────
# `full_resume_sidecar` names the FULL-state sibling in the closing advice line; it
# defaults to krea2's `<ckpt>.safetensors.state` naming. A driver whose save-state
# sidecar differs passes its own (sd35: `.state.safetensors`) so the banner stays
# byte-exact to that driver's pre-extraction output.
def trainer_warn_warm_resume(
    model_tag: String, path: String,
    full_resume_sidecar: String = String(".safetensors.state"),
):
    """LOUD multi-line warning: this resume is WARM (AdamW moments zeroed)."""
    print("")
    print("  ============================================================")
    print("  [" + model_tag + "-resume] !! WARM RESUME — AdamW moments RESTART at zero !!")
    print("  path:", path)
    print("  No FULL `.state` (A/B + adam_m/adam_v) was found for this checkpoint.")
    print("  The optimizer's first/second moments reset to zero, so training does")
    print("  NOT continue on the same trajectory as an uninterrupted run — the")
    print("  first resumed steps take large, under-damped AdamW updates.")
    print("  To FULL-resume, pass the `<ckpt>" + full_resume_sidecar + "` sidecar instead.")
    print("  ============================================================")
    print("")


# ── resume-scope guard (warn-only) ───────────────────────────────────────────
# Seed+step-derived noise and order[step % n] data order mean position == step BY
# CONSTRUCTION (no RNG/cursor snapshot to restore). But the KNOBS shaping those
# derived streams (seed, dataset sample count + identity, the layout key) must match
# the saved run or the resumed trajectory silently diverges. Stored in the `.state`
# __meta__ (trainer_state_meta); warn LOUDLY on any mismatch. `layout_label` names
# the model's layout knob in the log (e.g. "LTMAX").
def trainer_resume_meta_guard(
    model_tag: String, layout_label: String, live_layout_key: Int,
    resolved_path: String, is_full: Bool, start_step: Int,
    cfg_seed: UInt64, n_samples: Int, cache_path: String, ctx: DeviceContext,
) raises:
    if not is_full:
        return
    var meta = load_lora_train_state_meta(resolved_path, ctx)
    if len(meta) < 5:
        print("  [" + model_tag + "-resume] .state has no meta guard vector (older checkpoint) — skipping seed/dataset check")
        return
    var saved_step = Int(meta[0])
    var saved_seed = Int(meta[1])
    var saved_n = Int(meta[2])
    var saved_layout = Int(meta[3])
    var saved_hash = Int(meta[4])
    var live_hash = Int(trainer_cache_hash24(cache_path))
    var warned = False
    # MJ-1111: saved_seed came back through the F32 __meta__ vector (Int->F32->Int),
    # so it is rounded for seeds >= 2^24. Round the LIVE seed the SAME way before
    # comparing — otherwise a CORRECT same-seed resume spuriously warns. Do NOT
    # simplify to Int(cfg_seed); the F32 round-trip is load-bearing (the F32 storage
    # precision limit itself remains — this only fixes the false warn).
    if saved_seed != Int(Float32(Int(cfg_seed))):
        print("  [" + model_tag + "-resume][WARN] seed changed: .state=", saved_seed, " live=", Int(cfg_seed),
              "— the seed+step noise stream will DIVERGE from the saved run")
        warned = True
    if saved_n != n_samples:
        print("  [" + model_tag + "-resume][WARN] dataset size changed: .state n_samples=", saved_n, " live=", n_samples,
              "— the derived sample order (order[step mod n]) will DIVERGE")
        warned = True
    if saved_hash != live_hash:
        print("  [" + model_tag + "-resume][WARN] dataset identity changed: .state cache_hash=", saved_hash, " live=", live_hash,
              "— resuming onto a DIFFERENT cache than the one trained on")
        warned = True
    if saved_layout != live_layout_key:
        print("  [" + model_tag + "-resume][WARN] " + layout_label + " changed: .state=", saved_layout, " live=", live_layout_key,
              "— token padding/layout differs from the saved run")
        warned = True
    if saved_step != start_step:
        print("  [" + model_tag + "-resume][WARN] step mismatch: .state saved at step=", saved_step,
              " but resuming AT start_step=", start_step,
              "— noise+order are step-derived; pass that step as args[5] to continue exactly")
        warned = True
    if not warned:
        print("  [" + model_tag + "-resume] .state meta guard OK — seed/dataset/" + layout_label + "/step all match; derived noise+order continue exactly")


# ── micro-batch grad-accumulation window (wraps grad_accum.mojo primitives) ───
# SUM each micro-step's host LoRA grad groups into the window buffers; MEAN (÷the
# micro count) at the boundary, then recompute the grad norm from the MEANED host
# grads (the per-micro DEVICE norm is stale after the mean). accum_steps==1 → every
# step is a boundary and the sum-of-one/÷1 is byte-identical to the per-step path.
# The caller keeps the boundary control flow (progress print + `continue` on a
# non-boundary micro-step); this struct owns only the buffer math + norm.
struct GradAccumWindow(Movable):
    var acc_a: List[List[Float32]]
    var acc_b: List[List[Float32]]
    # second group-pair buffers (flux's stack LoRA groups). Unused by the 2-group
    # callers (krea2/chroma/sd35): the plain accumulate/finalize_mean never touch
    # them, so those callers stay byte-identical. Only the *_two_pairs methods
    # (flux) fill/mean them.
    var acc_a2: List[List[Float32]]
    var acc_b2: List[List[Float32]]
    var micro: Int
    var accum_steps: Int

    def __init__(out self, accum_steps: Int):
        self.acc_a = List[List[Float32]]()
        self.acc_b = List[List[Float32]]()
        self.acc_a2 = List[List[Float32]]()
        self.acc_b2 = List[List[Float32]]()
        self.micro = 0
        self.accum_steps = accum_steps

    # SUM this micro-step's grads into the window (zero-init on the first micro-step
    # of a window). Returns True on a boundary: the window is full OR `force_flush`
    # (the tail step) forces the partial window out.
    def accumulate(
        mut self, gl_a: List[List[Float32]], gl_b: List[List[Float32]],
        force_flush: Bool,
    ) raises -> Bool:
        if self.micro == 0:
            self.acc_a = zeros_like_group(gl_a)
            self.acc_b = zeros_like_group(gl_b)
        accumulate_grad_group(self.acc_a, gl_a)
        accumulate_grad_group(self.acc_b, gl_b)
        self.micro += 1
        return self.micro >= self.accum_steps or force_flush

    # MEAN the window (÷micro) back into gl, reset the window, and return the grad
    # norm RECOMPUTED (F64 accumulation) from the meaned host grads.
    def finalize_mean(
        mut self, mut gl_a: List[List[Float32]], mut gl_b: List[List[Float32]]
    ) raises -> Float32:
        var inv_micro = Float32(1.0) / Float32(self.micro)
        scale_grad_group(self.acc_a, inv_micro)
        scale_grad_group(self.acc_b, inv_micro)
        for i in range(len(gl_a)):
            gl_a[i] = self.acc_a[i].copy()
            gl_b[i] = self.acc_b[i].copy()
        self.micro = 0
        var _ss = Float64(0.0)
        for i in range(len(gl_a)):
            for j in range(len(gl_a[i])):
                var va = Float64(gl_a[i][j])
                _ss += va * va
            for j in range(len(gl_b[i])):
                var vb = Float64(gl_b[i][j])
                _ss += vb * vb
        return Float32(sqrt(_ss))

    # ── two group-pair variant (flux: block d_a/d_b + stack st_d_a/st_d_b) ────────
    # Flux accumulates FOUR host grad groups per micro-step: its block-proj LoRA pair
    # (gl_a/gl_b) plus the single/double-stream stack LoRA pair (gl_a2/gl_b2). SUM all
    # four into the window (zero-init on the first micro-step). The second pair is
    # empty when stack LoRA is disabled → its accumulate is a no-op, matching flux's
    # inline "empty stack groups accumulate as no-ops". Returns True on a boundary
    # exactly like `accumulate` (window full OR `force_flush` tail step).
    def accumulate_two_pairs(
        mut self,
        gl_a: List[List[Float32]], gl_b: List[List[Float32]],
        gl_a2: List[List[Float32]], gl_b2: List[List[Float32]],
        force_flush: Bool,
    ) raises -> Bool:
        if self.micro == 0:
            self.acc_a = zeros_like_group(gl_a)
            self.acc_b = zeros_like_group(gl_b)
            self.acc_a2 = zeros_like_group(gl_a2)
            self.acc_b2 = zeros_like_group(gl_b2)
        accumulate_grad_group(self.acc_a, gl_a)
        accumulate_grad_group(self.acc_b, gl_b)
        accumulate_grad_group(self.acc_a2, gl_a2)
        accumulate_grad_group(self.acc_b2, gl_b2)
        self.micro += 1
        return self.micro >= self.accum_steps or force_flush

    # MEAN all four groups (÷micro) back into gl_*, then reset the window. No grad
    # norm is returned — flux recomputes the (block+stack) global norm downstream
    # (its `_clip`), so the window owns only the SUM/MEAN, byte-op-identical to
    # flux's inline block (which likewise recomputed the norm after the mean).
    def finalize_mean_two_pairs(
        mut self,
        mut gl_a: List[List[Float32]], mut gl_b: List[List[Float32]],
        mut gl_a2: List[List[Float32]], mut gl_b2: List[List[Float32]],
    ) raises:
        var inv_micro = Float32(1.0) / Float32(self.micro)
        scale_grad_group(self.acc_a, inv_micro)
        scale_grad_group(self.acc_b, inv_micro)
        scale_grad_group(self.acc_a2, inv_micro)
        scale_grad_group(self.acc_b2, inv_micro)
        for i in range(len(gl_a)):
            gl_a[i] = self.acc_a[i].copy()
            gl_b[i] = self.acc_b[i].copy()
        for i in range(len(gl_a2)):
            gl_a2[i] = self.acc_a2[i].copy()
            gl_b2[i] = self.acc_b2[i].copy()
        self.micro = 0

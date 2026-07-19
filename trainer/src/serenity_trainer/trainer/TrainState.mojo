# TrainState.mojo — training-state checkpoint (save + restore), 1:1 port of
# Serenity's backup/INTERNAL-checkpoint contract, so a resumed run continues
# descent from EXACTLY where it stopped.
#
# SOURCE OF TRUTH (every rule below cites the Serenity .py line):
#
#   1. modules/util/TrainProgress.py                 (TrainProgress: epoch /
#        epoch_step / epoch_sample / global_step + next_step / next_epoch)
#   2. modules/modelSaver/mixin/InternalModelSaverMixin.py::_save_internal_data
#        (:14-42)  — WHAT a backup writes:
#          destination/optimizer/optimizer.pt   = optimizer.state_dict()   (:20-26)
#          destination/ema/ema.pt               = ema.state_dict()         (:29-31)
#          destination/meta.json                = train_progress 4 ints    (:34-42)
#   3. modules/modelLoader/mixin/InternalModelLoaderMixin.py::_load_internal_data
#        (:16-43)  — WHAT a resume reads back: meta.json → TrainProgress    (:23-30),
#          optimizer.pt → model.optimizer_state_dict                       (:33-35),
#          ema.pt → model.ema_state_dict                                   (:38-39).
#   4. modules/util/optimizer_util.py::init_model_parameters (:52-91) — HOW the
#        loaded optimizer_state_dict / ema_state_dict are re-applied:
#          create_optimizer(parameters, optimizer_state_dict, ...)         (:67)
#          create_ema(parameters.parameters(), ema_state_dict, ...)        (:76)
#        i.e. the moment state (m=exp_avg, v=exp_avg_sq) and EMA params are loaded
#        back INTO the freshly-built optimizer / EMA before the loop resumes.
#   5. modules/module/EMAModule.py::state_dict / load_state_dict (:77-86) — the EMA
#        state is just {"decay", "ema_parameters"}; load restores ema_parameters.
#   6. modules/trainer/GenericTrainer.py::train (:614,:635,:683) — resume USES the
#        restored progress: train_progress = self.model.train_progress (:614);
#        epochs = range(train_progress.epoch, self.config.epochs) (:635); the step
#        tqdm starts at train_progress.epoch_step (:683). Wiring note at EOF.
#
# WHY safetensors (not torch.save): Serenity torch.saves a Python pickle
# (optimizer.pt / ema.pt) — not pure-Mojo readable. The faithful port keeps the
# SAME logical contents (the per-param m/v moments for ALL LoRA adapters, the EMA
# params, the 4 progress counters) but serializes them with serenitymojo's
# verified safetensors writer/reader (the byte-exact analogue used by every other
# saver/loader in this port). The numbers restored are identical; only the
# container format differs (allowed by the port's "reuse serenitymojo io" rule).
#
# Move-only Tensor → each tensor boxed in ArcPointer[Tensor] (TArc). We never
# partial-move a Tensor field out of a struct; clone or move the whole slot.
#
# Dtype policy: m / v / EMA params are BF16 storage (zeros_like(p)=BF16 — see
# train_step.ParamSlot header, EMAModule.mojo header), written verbatim. The
# progress counters + the optimizer-step count are stored as a single F32 [5]
# tensor (their integer ranges round-trip through F32 exactly for any realistic
# step/epoch count).
#
# OPTIMIZER-STEP COUNT (opt_step) — added to the meta because Serenity's AdamW
# keeps a PER-PARAM state["step"] that increments once per optimizer.step()
# (adam_extensions.py:72 `step_t += 1`, used at :92/:124 `beta ** step` for bias
# correction) and is part of optimizer.state_dict() (InternalModelSaverMixin.py:21
# `model.optimizer.state_dict()`), restored on resume. This count equals the
# number of completed optimizer steps, which is `train_progress.global_step` ONLY
# when gradient_accumulation_steps == 1. At accum > 1, global_step is the MICRO
# (per-batch) counter (TrainProgress.next_step runs every batch — GenericTrainer.py
# :971) = opt_step * accum, so the two diverge. Persisting opt_step lets a resumed
# run restore the correct AdamW bias-correction step (beta ** opt_step) at any
# accum. Old 4-int checkpoints are still readable: opt_step is then derived from
# global_step (exact for accum == 1, the only regime old checkpoints were written
# in).

from std.memory import ArcPointer
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.io.sharded import ShardedSafeTensors
from serenity_trainer.trainer.train_step import ParamSlot


comptime TArc = ArcPointer[Tensor]


# ─────────────────────────────────────────────────────────────────────────────
# TrainProgress — 1:1 port of modules/util/TrainProgress.py.
#   epoch / epoch_step / epoch_sample / global_step                       (:9-12)
#   next_step(batch_size): epoch_step++, epoch_sample+=bs, global_step++  (:14-17)
#   next_epoch(): epoch_step=0, epoch_sample=0, epoch++                   (:19-22)
#   filename_string(): f"{global_step}-{epoch}-{epoch_step}"             (:24-25)
@fieldwise_init
struct TrainProgress(Copyable, Movable):
    var epoch: Int
    var epoch_step: Int
    var epoch_sample: Int
    var global_step: Int

    # Default-construct at the origin (TrainProgress.__init__ defaults, :2-12).
    def __init__(out self):
        self.epoch = 0
        self.epoch_step = 0
        self.epoch_sample = 0
        self.global_step = 0

    # next_step (TrainProgress.py:14-17).
    def next_step(mut self, batch_size: Int):
        self.epoch_step += 1
        self.epoch_sample += batch_size
        self.global_step += 1

    # next_epoch (TrainProgress.py:19-22).
    def next_epoch(mut self):
        self.epoch_step = 0
        self.epoch_sample = 0
        self.epoch += 1

    # filename_string (TrainProgress.py:24-25) — used by backup/save dir names.
    def filename_string(self) -> String:
        return (
            String(self.global_step)
            + "-"
            + String(self.epoch)
            + "-"
            + String(self.epoch_step)
        )


# ─────────────────────────────────────────────────────────────────────────────
# Stable key naming for the per-param moment state. Serenity keys optimizer
# state by the Parameter OBJECT (state[param] = {'step','exp_avg','exp_avg_sq'})
# and re-binds it to the SAME parameters at load via the param_group_mapping
# (create.py:1022-1060). Here the parameters are the driver's `slots` in a FIXED
# order (block-major / slot-minor, identical to ZImageLoraSet.ad and
# zimage_lora_target_prefixes — train_step.ParamSlot header). So we key the
# moments by slot INDEX, which is exactly that stable order. The save and the
# resume MUST iterate `slots` in the same order (they do — same builder).
#
#   slot.<i>.m  = exp_avg     (first moment)   [BF16, zeros_like(p)]
#   slot.<i>.v  = exp_avg_sq  (second moment)  [BF16, zeros_like(p)]
#
# `step` is NOT stored per-param: it is identical for every param (each param's
# state['step'] increments together, once per optimizer.step()), so a SINGLE
# scalar — `opt_step` in the meta below — restores the bias-correction step for
# ALL params at once. The port's AdamW reads its bias-correction step from this
# restored optimizer-step counter (train_step.mojo step_1based = global_step+1,
# where `global_step` is the OPTIMIZER-step counter seeded from opt_step), so
# beta ** step matches Serenity's per-param state['step'] at any accum.
def _m_key(i: Int) -> String:
    return String("slot.") + String(i) + ".m"


def _v_key(i: Int) -> String:
    return String("slot.") + String(i) + ".v"


def _ema_key(i: Int) -> String:
    return String("ema.") + String(i)


comptime _META_KEY = "train_progress"  # F32 [5]: epoch, epoch_step, epoch_sample, global_step, opt_step


# ─────────────────────────────────────────────────────────────────────────────
# save_train_state — write the resumable checkpoint to `dir`.
#
# Faithful to InternalModelSaverMixin._save_internal_data (:14-42): one file holds
#   * the optimizer moment state for ALL slots (m=exp_avg, v=exp_avg_sq)   (:20-26)
#   * the EMA parameters, if EMA is enabled                                (:29-31)
#   * the 4 TrainProgress counters                                         (:34-42)
#
# Unlike Serenity's three separate files (optimizer/, ema/, meta.json) we write
# ONE safetensors `<dir>/train_state.safetensors` (serenitymojo's writer takes a
# flat name→tensor map). The logical contents are identical.
#
# Args:
#   slots  : the driver's optimizer slots (p,m,v,accum) — the AdamW state for
#            every LoRA adapter. We persist m and v (the per-param moments).
#   ema    : the EMA buffers (one BF16 tensor per slot, same order). Empty list
#            ⇒ EMA disabled ⇒ no ema.* tensors written (matches `if model.ema:`,
#            InternalModelSaverMixin.py:29).
#   prog   : the live TrainProgress (epoch / epoch_step / epoch_sample / global_step).
#   opt_step: the optimizer-step count (= Serenity per-param state['step'] =
#            number of completed optimizer.step() calls). Persisted so resume
#            restores the AdamW bias-correction step at any accum (== prog.global_step
#            only when accum == 1). See module header.
#
# The accumulator (`slot.accum`, = torch .grad) is intentionally NOT saved:
# Serenity backs up AFTER optimizer.zero_grad on an update-step boundary, so
# .grad is None at backup time (GenericTrainer.py:695-723 enqueues backup only
# when `not has_gradient`). A resumed run re-seeds accum at micro_idx 0.
def save_train_state(
    dir: String,
    slots: List[ParamSlot],
    ema: List[TArc],
    prog: TrainProgress,
    opt_step: Int,
    ctx: DeviceContext,
) raises:
    var names = List[String]()
    var tensors = List[TArc]()

    # ── optimizer moments for every slot (InternalModelSaverMixin.py:20-26) ──
    for i in range(len(slots)):
        names.append(_m_key(i))
        tensors.append(TArc(slots[i].m[].clone(ctx)))   # exp_avg
        names.append(_v_key(i))
        tensors.append(TArc(slots[i].v[].clone(ctx)))   # exp_avg_sq

    # ── EMA params, if enabled (InternalModelSaverMixin.py:29-31) ────────────
    for i in range(len(ema)):
        names.append(_ema_key(i))
        tensors.append(TArc(ema[i][].clone(ctx)))

    # ── meta: the 4 TrainProgress counters (InternalModelSaverMixin.py:34-42)
    #    PLUS the optimizer-step count (= optimizer.state_dict()'s per-param
    #    state['step'], InternalModelSaverMixin.py:20-21). 5th slot. ─────────────
    var meta_vals = List[Float32]()
    meta_vals.append(Float32(prog.epoch))
    meta_vals.append(Float32(prog.epoch_step))
    meta_vals.append(Float32(prog.epoch_sample))
    meta_vals.append(Float32(prog.global_step))
    meta_vals.append(Float32(opt_step))
    var meta_shape = List[Int]()
    meta_shape.append(5)
    var meta_t = Tensor.from_host(meta_vals, meta_shape^, STDtype.F32, ctx)
    names.append(String(_META_KEY))
    tensors.append(TArc(meta_t^))

    save_safetensors(names, tensors, _state_path(dir), ctx)


# ─────────────────────────────────────────────────────────────────────────────
# LoadedTrainState — the restored counters + a handle to the opened checkpoint so
# the caller can pull each m/v/ema tensor by slot. Movable (owns the mmap'd file).
struct LoadedTrainState(Movable):
    var prog: TrainProgress
    var src: ShardedSafeTensors
    var n_slots: Int      # number of slot.<i>.m present in the file
    var n_ema: Int        # number of ema.<i> present (0 ⇒ EMA was disabled)
    var opt_step: Int     # restored optimizer-step count (= per-param AdamW step)

    def __init__(
        out self,
        var prog: TrainProgress,
        var src: ShardedSafeTensors,
        n_slots: Int,
        n_ema: Int,
        opt_step: Int,
    ):
        self.prog = prog^
        self.src = src^
        self.n_slots = n_slots
        self.n_ema = n_ema
        self.opt_step = opt_step

    # Load slot i's first moment (exp_avg) as an owned device Tensor (dtype
    # preserved = BF16). Mirrors create_optimizer rebinding state['exp_avg'].
    def m(self, i: Int, ctx: DeviceContext) raises -> Tensor:
        return Tensor.from_view(self.src.tensor_view(_m_key(i)), ctx)

    # Load slot i's second moment (exp_avg_sq).
    def v(self, i: Int, ctx: DeviceContext) raises -> Tensor:
        return Tensor.from_view(self.src.tensor_view(_v_key(i)), ctx)

    # Load EMA buffer i (only valid when i < n_ema).
    def ema(self, i: Int, ctx: DeviceContext) raises -> Tensor:
        return Tensor.from_view(self.src.tensor_view(_ema_key(i)), ctx)


# load_train_state — open `<dir>/train_state.safetensors`, read the progress
# counters, and return a handle the caller drains into its slots/EMA.
#
# Faithful to InternalModelLoaderMixin._load_internal_data (:16-43): meta →
# TrainProgress (:23-30); optimizer/ema tensors are restored into the freshly
# built state (the per-tensor m()/v()/ema() accessors are the moment-rebind seam
# = optimizer_util.init_model_parameters :67/:76).
def load_train_state(
    dir: String,
    ctx: DeviceContext,
) raises -> LoadedTrainState:
    var src = ShardedSafeTensors.open(_state_path(dir))

    # meta → TrainProgress (InternalModelLoaderMixin.py:23-30) + opt_step.
    # New checkpoints write 5 floats (… + opt_step); 4-float checkpoints are old
    # accum==1 backups, for which opt_step == global_step exactly (so deriving it
    # is byte-faithful). Anything else is malformed.
    var meta = Tensor.from_view(src.tensor_view(String(_META_KEY)), ctx).to_host(ctx)
    if len(meta) != 4 and len(meta) != 5:
        raise Error("load_train_state: malformed train_progress meta (want 4 or 5 ints)")
    var prog = TrainProgress(
        Int(meta[0]), Int(meta[1]), Int(meta[2]), Int(meta[3])
    )
    # opt_step: explicit 5th slot, else derived from global_step (accum==1 old ckpt).
    var opt_step = Int(meta[4]) if len(meta) == 5 else prog.global_step

    # Count the m/v slot pairs and the ema tensors present (the file is the
    # authority on counts — they must match the rebuilt slots/EMA at the callsite).
    var keys = src.names()
    var n_slots = 0
    var n_ema = 0
    for ref nm in keys:
        if nm.startswith("slot.") and nm.endswith(".m"):
            n_slots += 1
        elif nm.startswith("ema."):
            n_ema += 1

    return LoadedTrainState(prog^, src^, n_slots, n_ema, opt_step)


# Restore the moment state INTO an existing slot list, in slot order. This is the
# moment-rebind step Serenity performs inside create_optimizer (create.py:
# 1022-1060): the saved exp_avg / exp_avg_sq are copied back onto the SAME
# parameters (here: the same-ordered slots). After this the resumed AdamW step
# uses the restored m/v + the restored global_step bias correction → identical
# descent. The slot count MUST equal the checkpoint's (raises otherwise).
def restore_moments_into_slots(
    st: LoadedTrainState,
    mut slots: List[ParamSlot],
    ctx: DeviceContext,
) raises:
    if len(slots) != st.n_slots:
        raise Error(
            String("restore_moments_into_slots: slot count mismatch (have ")
            + String(len(slots))
            + ", checkpoint has "
            + String(st.n_slots)
            + ")"
        )
    for i in range(len(slots)):
        slots[i].m = TArc(st.m(i, ctx))
        slots[i].v = TArc(st.v(i, ctx))


# Restore the EMA buffers (create_ema with the saved ema_parameters,
# EMAModule.load_state_dict :77-80). Returns a fresh list of EMA tensors in slot
# order; caller adopts it as its EMA state. Empty if the checkpoint had no EMA.
def restore_ema(
    st: LoadedTrainState,
    ctx: DeviceContext,
) raises -> List[TArc]:
    var out = List[TArc]()
    for i in range(st.n_ema):
        out.append(TArc(st.ema(i, ctx)))
    return out^


# Single-file checkpoint path under `dir` (one safetensors, vs Serenity's
# optimizer/ + ema/ + meta.json under the backup dir).
def _state_path(dir: String) -> String:
    return dir + "/train_state.safetensors"

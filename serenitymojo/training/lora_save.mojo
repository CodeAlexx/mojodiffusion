# lora_save.mojo — save / load TRAINED LoRA adapters as model-keyed safetensors.
# The LoRA-WEIGHTS half of resume; the training-STATE half (BF16 master + AdamW
# m/v + step counter) already exists in training/loop.mojo (TrainState /
# save_checkpoint / load_checkpoint) and is reused unchanged.
#
# ── Why this file exists ─────────────────────────────────────────────────────
# training/loop.mojo persists the GENERIC optimizer state (param.<i>/adam_m.<i>/
# adam_v.<i>/__meta__) for an opaque parameter set. It does NOT know the LoRA
# key naming, so a loop checkpoint cannot be opened by an inference loader
# (lora.mojo) or by torchref. This module writes the trained A/B in the
# canonical PEFT key convention, plus explicit model parity variants, so:
#   * lora.mojo::LoraSet.load detects it as FMT_DIFFUSION_MODEL and merges it,
#   * the validation sampler (training/validation_sampler.mojo) can load it,
#   * external tools (torchref / diffusers PEFT) open it.
#
# ── Key convention (the EXACT inverse of how lora.mojo LOADS) ────────────────
# lora.mojo::_suffix_a / _suffix_b (lora.mojo:~200-215) for FMT_DIFFUSION_MODEL:
#       A suffix = ".lora_A.weight"
#       B suffix = ".lora_B.weight"
# and lora.mojo's header (lines ~30-45) documents EriDiffusion-v2 `train_klein`
# ships bare `<prefix>.lora_A.weight` (no `diffusion_model.` prefix), detected as
# DiffusionModel by the `.lora_A.weight` suffix match in `_detect_format`
# (lora.mojo:~120-160). We therefore write:
#       "<module>.lora_A.weight"   shape [rank, in]   (lora.mojo:118 "lora_A:[rank,in]")
#       "<module>.lora_B.weight"   shape [out, rank]  (lora.mojo:118 "lora_B:[out,rank]")
# A is the "down" projection, B the "up" — matching LoraAdapter.a/.b in
# training/train_step.mojo:120-125 ("a:[rank,in], b:[out,rank]"). This makes
# save_lora_peft the byte-exact inverse of LoraSet._compute_delta's load
# (lora.mojo:~480: load A [rank,in], B [out,rank], delta = scale*(B@A)).
#
# NOTE on scale/alpha for generic PEFT saves: save_lora_peft does NOT write a
# per-module `.alpha` scalar. The LoraAdapter carries `scale = alpha/rank`
# (train_step.mojo:153), but the canonical PEFT/train_klein file omits `.alpha`,
# and lora.mojo::_module_scale (lora.mojo:~300-320) then DEFAULTS alpha =
# module_rank → scale = multiplier. Qwen's SerenityTrainer parity path is explicit
# below in save_lora_serenity_trainer and does write BF16 scalar `.alpha` tensors.
#
# Mojo 1.0.0b1: `def` not `fn`; move-only Tensor → collections hold
# ArcPointer[Tensor]; A/B are BF16 model storage; AdamW moments remain F32.

from std.collections import List
from std.memory import ArcPointer
from max.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.io.safetensors_writer import (
    save_safetensors, save_safetensors_host, HostTensorDesc,
)
from serenitymojo.training.train_step import LoraAdapter, _f32_to_bf16_list


# ── A trained LoRA module paired with the base-weight prefix it adapts ───────
# `prefix` is the LoRA module name WITHOUT the lora_A/lora_B suffix, e.g.
# "double_blocks.0.img_attn.to_q" — exactly the `prefix` lora.mojo strips back
# off in LoraSet.load (lora.mojo:~370). The save appends ".lora_A.weight" /
# ".lora_B.weight" to it.
@fieldwise_init
struct NamedLora(Copyable, Movable):
    var prefix: String
    var adapter: LoraAdapter


# A host-F32 source adapter for trainers whose exact masters do not already
# live in LoraAdapter host lists. The external artifact is still canonical BF16
# PEFT; this representation only avoids staging a second full adapter set in
# device memory at save cadence.
struct F32NamedLora(Copyable, Movable):
    var prefix: String
    var a: List[Float32]
    var b: List[Float32]
    var rank: Int
    var in_f: Int
    var out_f: Int

    def __init__(
        out self, var prefix: String, var a: List[Float32], var b: List[Float32],
        rank: Int, in_f: Int, out_f: Int,
    ):
        self.prefix = prefix^
        self.a = a^
        self.b = b^
        self.rank = rank
        self.in_f = in_f
        self.out_f = out_f


# ── F32-EXACT trainer state for one adapter (resume-critical payload) ─────────
# The driver (train_ltx2_av.mojo) keeps its LoRA masters in F32 (struct F32Lora)
# because a per-step bf16 write-back absorbs 30-57% of the A-updates (MJ-1108).
# A resume that re-loads bf16-rounded A/B re-introduces exactly that loss on the
# resume boundary. `F32LoraState` carries the F32 masters + AdamW moments so the
# state sidecar can round-trip them BIT-EXACT (`save_lora_train_state_f32` /
# `load_lora_train_state_f32`). `a` is [rank,in], `b` is [out,rank] — same layout
# as LoraAdapter and F32Lora (a=down, b=up). rank/in_f/out_f let the loader
# validate against the trainer's comptime geometry.
struct F32LoraState(Copyable, Movable):
    var prefix: String
    var a: List[Float32]
    var b: List[Float32]
    var ma: List[Float32]
    var va: List[Float32]
    var mb: List[Float32]
    var vb: List[Float32]
    var rank: Int
    var in_f: Int
    var out_f: Int

    def __init__(
        out self, var prefix: String,
        var a: List[Float32], var b: List[Float32],
        var ma: List[Float32], var va: List[Float32],
        var mb: List[Float32], var vb: List[Float32],
        rank: Int, in_f: Int, out_f: Int,
    ):
        self.prefix = prefix^
        self.a = a^
        self.b = b^
        self.ma = ma^
        self.va = va^
        self.mb = mb^
        self.vb = vb^
        self.rank = rank
        self.in_f = in_f
        self.out_f = out_f


def _bf16_2d(var values: List[BFloat16], rows: Int, cols: Int, ctx: DeviceContext) raises -> Tensor:
    var sh = List[Int]()
    sh.append(rows)
    sh.append(cols)
    return Tensor.from_host_bf16(values^, sh^, ctx)


def _bf16_scalar(value: Float32, ctx: DeviceContext) raises -> Tensor:
    var values = List[BFloat16]()
    values.append(value.cast[DType.bfloat16]())
    var sh = List[Int]()
    return Tensor.from_host_bf16(values^, sh^, ctx)


# ── F32 device tensor from a host List[Float32] with a 2-D shape ─────────────
# Used for optimizer moments only. Trainable LoRA A/B use `_bf16_2d`.
def _f32_2d(var values: List[Float32], rows: Int, cols: Int, ctx: DeviceContext) raises -> Tensor:
    var sh = List[Int]()
    sh.append(rows)
    sh.append(cols)
    return Tensor.from_host(values^, sh^, STDtype.F32, ctx)


# ── F32 1-D device tensor (used for the optional resume-`__meta__` vector). ────
def _f32_1d(var values: List[Float32], n: Int, ctx: DeviceContext) raises -> Tensor:
    var sh = List[Int]()
    sh.append(n)
    return Tensor.from_host(values^, sh^, STDtype.F32, ctx)


# ── HOST-DIRECT byte packing (state files must never stage through VRAM) ─────
# Raw little-endian storage bytes for save_safetensors_host: the F32 list is
# bitcast in place (4 B/elem, native LE on x86-64); the bf16 compat copies are
# downcast host-side (2 B/elem). No DeviceContext anywhere on this path.
def _f32_le_bytes(vals: List[Float32]) -> List[UInt8]:
    var out = List[UInt8]()
    var p = vals.unsafe_ptr().bitcast[UInt8]()
    for i in range(len(vals) * 4):
        out.append(p[i])
    return out^


def _bf16_le_bytes_from_f32(vals: List[Float32]) -> List[UInt8]:
    var tmp = _f32_to_bf16_list(vals)
    var out = List[UInt8]()
    var p = tmp.unsafe_ptr().bitcast[UInt8]()
    for i in range(len(tmp) * 2):
        out.append(p[i])
    return out^


def _shape2(rows: Int, cols: Int) -> List[Int]:
    var s = List[Int]()
    s.append(rows)
    s.append(cols)
    return s^


def _shape1(n: Int) -> List[Int]:
    var s = List[Int]()
    s.append(n)
    return s^


# ─────────────────────────────────────────────────────────────────────────────
# SAVE: pack each adapter's A [rank,in] and B [out,rank] into a single
# safetensors via the proven byte-exact writer (io/safetensors_writer.mojo:186).
# Keys are PEFT/torchref: "<prefix>.lora_A.weight" / "<prefix>.lora_B.weight".
# ─────────────────────────────────────────────────────────────────────────────
def save_lora_peft(
    adapters: List[NamedLora], path: String, ctx: DeviceContext
) raises -> Int:
    """Write `adapters` to `path` as a PEFT-keyed LoRA safetensors. Returns the
    number of (A,B) PAIRS written. Tensors are BF16 model-storage tensors.

    For each NamedLora we emit two tensors, in A-then-B order per module:
        "<prefix>.lora_A.weight"  BF16 [rank, in]   (== LoraAdapter.a)
        "<prefix>.lora_B.weight"  BF16 [out, rank]  (== LoraAdapter.b)
    This is the exact inverse of lora.mojo::LoraSet._compute_delta's load. The
    writer lays tensors out in insertion order with contiguous data_offsets
    (safetensors_writer.mojo:111-127), and SafeTensors.open reads them back by
    name, so order is informational only — but A-before-B per module matches the
    `safetensors` Python canonical (insertion) order external tools expect."""
    if len(adapters) == 0:
        raise Error("save_lora_peft: refusing to write an empty LoRA file")

    var names = List[String]()
    var tensors = List[ArcPointer[Tensor]]()

    for ref nl in adapters:
        var a = nl.adapter.copy()
        # Shape sanity: a is [rank,in], b is [out,rank] (train_step.mojo:121-122).
        if len(a.a) != a.rank * a.in_f:
            raise Error(
                String("save_lora_peft: A numel ") + String(len(a.a))
                + " != rank*in " + String(a.rank * a.in_f)
                + " for '" + nl.prefix + "'"
            )
        if len(a.b) != a.out_f * a.rank:
            raise Error(
                String("save_lora_peft: B numel ") + String(len(a.b))
                + " != out*rank " + String(a.out_f * a.rank)
                + " for '" + nl.prefix + "'"
            )
        names.append(nl.prefix + ".lora_A.weight")
        tensors.append(ArcPointer(_bf16_2d(a.a.copy(), a.rank, a.in_f, ctx)))
        names.append(nl.prefix + ".lora_B.weight")
        tensors.append(ArcPointer(_bf16_2d(a.b.copy(), a.out_f, a.rank, ctx)))

    save_safetensors(names, tensors, path, ctx)
    return len(adapters)


def save_lora_peft_host_f32(
    adapters: List[F32NamedLora], path: String,
) raises -> Int:
    """Write host-F32 A/B sources as the same BF16 PEFT artifact emitted by
    `save_lora_peft`, without allocating output tensors on the GPU.

    This is for device-master trainers that download their exact F32 masters at
    save cadence. It emits only canonical `<prefix>.lora_A.weight` and
    `<prefix>.lora_B.weight` tensors; F32 masters and optimizer state belong in
    a separate resume sidecar."""
    if len(adapters) == 0:
        raise Error("save_lora_peft_host_f32: refusing to write an empty LoRA file")

    var names = List[String]()
    var descs = List[HostTensorDesc]()
    for ref nl in adapters:
        if len(nl.a) != nl.rank * nl.in_f:
            raise Error(
                String("save_lora_peft_host_f32: A numel ") + String(len(nl.a))
                + " != rank*in " + String(nl.rank * nl.in_f)
                + " for '" + nl.prefix + "'"
            )
        if len(nl.b) != nl.out_f * nl.rank:
            raise Error(
                String("save_lora_peft_host_f32: B numel ") + String(len(nl.b))
                + " != out*rank " + String(nl.out_f * nl.rank)
                + " for '" + nl.prefix + "'"
            )
        names.append(nl.prefix + ".lora_A.weight")
        descs.append(HostTensorDesc(
            STDtype.BF16, _shape2(nl.rank, nl.in_f),
            _bf16_le_bytes_from_f32(nl.a),
        ))
        names.append(nl.prefix + ".lora_B.weight")
        descs.append(HostTensorDesc(
            STDtype.BF16, _shape2(nl.out_f, nl.rank),
            _bf16_le_bytes_from_f32(nl.b),
        ))

    save_safetensors_host(names, descs, path)
    return len(adapters)


def save_lora_serenity_trainer(
    adapters: List[NamedLora], path: String, ctx: DeviceContext
) raises -> Int:
    """Write `adapters` using SerenityTrainer's raw LoRA state_dict convention.

    For each module prefix we emit:
        "<prefix>.alpha"             BF16 scalar [] (= alpha)
        "<prefix>.lora_down.weight"  BF16 [rank, in]   (== LoraAdapter.a)
        "<prefix>.lora_up.weight"    BF16 [out, rank]  (== LoraAdapter.b)

    This is separate from `save_lora_peft` so existing generic PEFT/torchref
    saves keep their lora_A/lora_B convention. Qwen uses this path for direct
    SerenityTrainer parity."""
    if len(adapters) == 0:
        raise Error("save_lora_serenity_trainer: refusing to write an empty LoRA file")

    var names = List[String]()
    var tensors = List[ArcPointer[Tensor]]()

    for ref nl in adapters:
        var a = nl.adapter.copy()
        if len(a.a) != a.rank * a.in_f:
            raise Error(
                String("save_lora_serenity_trainer: A numel ") + String(len(a.a))
                + " != rank*in " + String(a.rank * a.in_f)
                + " for '" + nl.prefix + "'"
            )
        if len(a.b) != a.out_f * a.rank:
            raise Error(
                String("save_lora_serenity_trainer: B numel ") + String(len(a.b))
                + " != out*rank " + String(a.out_f * a.rank)
                + " for '" + nl.prefix + "'"
            )

        names.append(nl.prefix + ".alpha")
        tensors.append(ArcPointer(_bf16_scalar(a.scale * Float32(a.rank), ctx)))
        names.append(nl.prefix + ".lora_down.weight")
        tensors.append(ArcPointer(_bf16_2d(a.a.copy(), a.rank, a.in_f, ctx)))
        names.append(nl.prefix + ".lora_up.weight")
        tensors.append(ArcPointer(_bf16_2d(a.b.copy(), a.out_f, a.rank, ctx)))

    save_safetensors(names, tensors, path, ctx)
    return len(adapters)


def save_lora_train_state(
    adapters: List[NamedLora], path: String, ctx: DeviceContext,
    var meta: List[Float32] = List[Float32](),
) raises -> Int:
    """Write trainer-only LoRA state: A/B plus AdamW moments.

    This is intentionally separate from the PEFT file. The PEFT file stays
    plain external-compatible LoRA, while this state file lets the pure-Mojo
    cadence supervisor resume without zeroing AdamW moments.

    `meta` (optional) is written as an F32 `__meta__` tensor — the resume-scope
    guard vector (step / seed / dataset-identity knobs). It is model-defined; the
    load-back path reads it via `load_lora_train_state_meta`. Empty = no meta
    tensor (byte-identical to the pre-meta writer for callers that omit it). The
    per-adapter loaders (`load_lora_train_state` / `load_lora_for_resume`) iterate
    the fixed prefix set and ignore `__meta__`, so this stays backward-compatible.
    """
    if len(adapters) == 0:
        raise Error("save_lora_train_state: refusing to write an empty state")

    var names = List[String]()
    var tensors = List[ArcPointer[Tensor]]()
    for ref nl in adapters:
        var a = nl.adapter.copy()
        if len(a.a) != a.rank * a.in_f or len(a.ma) != a.rank * a.in_f or len(a.va) != a.rank * a.in_f:
            raise Error(String("save_lora_train_state: A/m/v shape mismatch for ") + nl.prefix)
        if len(a.b) != a.out_f * a.rank or len(a.mb) != a.out_f * a.rank or len(a.vb) != a.out_f * a.rank:
            raise Error(String("save_lora_train_state: B/m/v shape mismatch for ") + nl.prefix)
        names.append(nl.prefix + ".lora_A.weight")
        tensors.append(ArcPointer(_bf16_2d(a.a.copy(), a.rank, a.in_f, ctx)))
        names.append(nl.prefix + ".lora_B.weight")
        tensors.append(ArcPointer(_bf16_2d(a.b.copy(), a.out_f, a.rank, ctx)))
        names.append(nl.prefix + ".lora_A.adam_m")
        tensors.append(ArcPointer(_f32_2d(a.ma.copy(), a.rank, a.in_f, ctx)))
        names.append(nl.prefix + ".lora_A.adam_v")
        tensors.append(ArcPointer(_f32_2d(a.va.copy(), a.rank, a.in_f, ctx)))
        names.append(nl.prefix + ".lora_B.adam_m")
        tensors.append(ArcPointer(_f32_2d(a.mb.copy(), a.out_f, a.rank, ctx)))
        names.append(nl.prefix + ".lora_B.adam_v")
        tensors.append(ArcPointer(_f32_2d(a.vb.copy(), a.out_f, a.rank, ctx)))

    var n_meta = len(meta)
    if n_meta > 0:
        names.append(String("__meta__"))
        tensors.append(ArcPointer(_f32_1d(meta^, n_meta, ctx)))

    save_safetensors(names, tensors, path, ctx)
    return len(adapters)


def lora_train_state_has_moments(path: String, probe_prefix: String) raises -> Bool:
    """True if `path` opens as a LoRA train_state carrying AdamW moments — probed
    via the `<probe_prefix>.lora_A.adam_m` key. False if the file is a plain PEFT
    LoRA (no moments) or cannot be opened. Lets a resume path distinguish a FULL
    `.state` from a weights-only checkpoint BEFORE choosing the load routine (so it
    can auto-find the `<ckpt>.state` sibling and only warm-fall-back when there is
    genuinely no moment state to restore)."""
    try:
        var st = SafeTensors.open(path)
        return (probe_prefix + ".lora_A.adam_m") in st.tensors
    except:
        return False


def load_lora_train_state_meta(path: String, ctx: DeviceContext) raises -> List[Float32]:
    """Read the optional `__meta__` resume-guard vector from a `.state` file.
    Returns an empty list when absent (older `.state` files carry no meta), so
    callers treat a missing meta as "nothing to check"."""
    var st = SafeTensors.open(path)
    if String("__meta__") not in st.tensors:
        return List[Float32]()
    return _read_f32(st, String("__meta__"), ctx)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD-BACK for resume: read A/B by PEFT key into fresh LoraAdapters. Optimizer
# state (ma/va/mb/vb) is ZEROED here — the AdamW moments for a *resumed* run come
# from the loop.mojo TrainState checkpoint (adam_m.<i>/adam_v.<i>), NOT from the
# LoRA-weights file. (PEFT LoRA files never carry optimizer state.) So a full
# resume is: load_checkpoint(...) for masters+m/v+t  AND  load_lora_for_resume
# for the A/B weights if the trainer keeps LoRA outside the TrainState param set.
# ─────────────────────────────────────────────────────────────────────────────
def _read_f32(st: SafeTensors, name: String, ctx: DeviceContext) raises -> List[Float32]:
    """Read one tensor by name to a host F32 list (upcasts via to_host). Uses the
    from_parts(info.dtype, info.shape, bytes) idiom documented in
    io/tensor_view.mojo:114-119 so the view's origin binds to `st`."""
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    if info.dtype == STDtype.F32:
        if info.size % 4 != 0:
            raise Error(String("_read_f32: bad F32 byte size for ") + name)
        var fp = bytes.unsafe_ptr().bitcast[Float32]()
        var out = List[Float32]()
        for i in range(info.size // 4):
            out.append(fp[i])
        return out^
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    var t = Tensor.from_view(tv, ctx)
    return t.to_host(ctx)


def load_lora_for_resume(
    prefixes: List[String], scale: Float32, path: String, ctx: DeviceContext
) raises -> List[NamedLora]:
    """Read a PEFT or SerenityTrainer raw LoRA file back into LoraAdapters, one per
    `prefix`. PEFT uses `.lora_A.weight` / `.lora_B.weight` and receives the
    caller supplied `scale`. SerenityTrainer raw saves use `.lora_down.weight` /
    `.lora_up.weight` plus `.alpha`; when alpha exists, the adapter scale is
    reconstructed as alpha/rank. AdamW moments are zeroed (resume them from a
    loop.mojo TrainState checkpoint instead).

    Shapes are read from the file header: A is [rank,in], B is [out,rank], so
    rank = A.shape[0], in = A.shape[1], out = B.shape[0]. The B.shape[1] is
    asserted to equal rank (the file's own self-consistency check)."""
    var st = SafeTensors.open(path)
    var out = List[NamedLora]()

    for ref pfx in prefixes:
        var key_a = pfx + ".lora_A.weight"
        var key_b = pfx + ".lora_B.weight"
        var key_alpha = pfx + ".alpha"
        if key_a not in st.tensors or key_b not in st.tensors:
            key_a = pfx + ".lora_down.weight"
            key_b = pfx + ".lora_up.weight"
        if key_a not in st.tensors or key_b not in st.tensors:
            # Some PEFT exports preserve the model-root prefix while the runtime
            # flat prefix order is model-local.
            var dm_pfx = String("diffusion_model.") + pfx
            key_a = dm_pfx + ".lora_A.weight"
            key_b = dm_pfx + ".lora_B.weight"
            key_alpha = dm_pfx + ".alpha"
        if key_a not in st.tensors or key_b not in st.tensors:
            var dm_pfx = String("diffusion_model.") + pfx
            key_a = dm_pfx + ".lora_down.weight"
            key_b = dm_pfx + ".lora_up.weight"
            key_alpha = dm_pfx + ".alpha"
        if key_a not in st.tensors:
            raise Error(String("load_lora_for_resume: missing ") + key_a)
        if key_b not in st.tensors:
            raise Error(String("load_lora_for_resume: missing ") + key_b)

        var a_info = st.tensor_info(key_a)
        var b_info = st.tensor_info(key_b)
        if len(a_info.shape) != 2 or len(b_info.shape) != 2:
            raise Error(String("load_lora_for_resume: A/B must be 2-D for ") + pfx)
        var rank = a_info.shape[0]
        var in_f = a_info.shape[1]
        var out_f = b_info.shape[0]
        if b_info.shape[1] != rank:
            raise Error(
                String("load_lora_for_resume: B.shape[1]=") + String(b_info.shape[1])
                + " != rank " + String(rank) + " for '" + pfx + "'"
            )

        var a_h = _read_f32(st, key_a, ctx)
        var b_h = _read_f32(st, key_b, ctx)
        var adapter_scale = scale
        if key_alpha in st.tensors:
            var alpha_h = _read_f32(st, key_alpha, ctx)
            if len(alpha_h) != 1:
                raise Error(String("load_lora_for_resume: .alpha must have one value for ") + pfx)
            adapter_scale = alpha_h[0] / Float32(rank)

        # Fresh zeroed AdamW moments (resumed from the loop checkpoint elsewhere).
        var ma = List[Float32]()
        var va = List[Float32]()
        for _ in range(rank * in_f):
            ma.append(Float32(0.0))
            va.append(Float32(0.0))
        var mb = List[Float32]()
        var vb = List[Float32]()
        for _ in range(out_f * rank):
            mb.append(Float32(0.0))
            vb.append(Float32(0.0))

        var ad = LoraAdapter(
            a_h^, b_h^, rank, in_f, out_f, adapter_scale, ma^, va^, mb^, vb^
        )
        out.append(NamedLora(pfx, ad^))

    return out^


def load_lora_train_state(
    prefixes: List[String], scale: Float32, path: String, ctx: DeviceContext
) raises -> List[NamedLora]:
    """Read the trainer-only state file written by save_lora_train_state."""
    var st = SafeTensors.open(path)
    var out = List[NamedLora]()

    for ref pfx in prefixes:
        var key_a = pfx + ".lora_A.weight"
        var key_b = pfx + ".lora_B.weight"
        var key_ma = pfx + ".lora_A.adam_m"
        var key_va = pfx + ".lora_A.adam_v"
        var key_mb = pfx + ".lora_B.adam_m"
        var key_vb = pfx + ".lora_B.adam_v"
        if key_a not in st.tensors:
            raise Error(String("load_lora_train_state: missing ") + key_a)
        if key_b not in st.tensors:
            raise Error(String("load_lora_train_state: missing ") + key_b)
        if key_ma not in st.tensors:
            raise Error(String("load_lora_train_state: missing ") + key_ma)
        if key_va not in st.tensors:
            raise Error(String("load_lora_train_state: missing ") + key_va)
        if key_mb not in st.tensors:
            raise Error(String("load_lora_train_state: missing ") + key_mb)
        if key_vb not in st.tensors:
            raise Error(String("load_lora_train_state: missing ") + key_vb)

        var a_info = st.tensor_info(key_a)
        var b_info = st.tensor_info(key_b)
        if len(a_info.shape) != 2 or len(b_info.shape) != 2:
            raise Error(String("load_lora_train_state: A/B must be 2-D for ") + pfx)
        var rank = a_info.shape[0]
        var in_f = a_info.shape[1]
        var out_f = b_info.shape[0]
        if b_info.shape[1] != rank:
            raise Error(String("load_lora_train_state: B rank mismatch for ") + pfx)

        var a_h = _read_f32(st, key_a, ctx)
        var b_h = _read_f32(st, key_b, ctx)
        var ma_h = _read_f32(st, key_ma, ctx)
        var va_h = _read_f32(st, key_va, ctx)
        var mb_h = _read_f32(st, key_mb, ctx)
        var vb_h = _read_f32(st, key_vb, ctx)
        if len(ma_h) != len(a_h) or len(va_h) != len(a_h):
            raise Error(String("load_lora_train_state: A moment len mismatch for ") + pfx)
        if len(mb_h) != len(b_h) or len(vb_h) != len(b_h):
            raise Error(String("load_lora_train_state: B moment len mismatch for ") + pfx)

        var ad = LoraAdapter(
            a_h^, b_h^, rank, in_f, out_f, scale,
            ma_h^, va_h^, mb_h^, vb_h^,
        )
        out.append(NamedLora(pfx, ad^))

    return out^


# ─────────────────────────────────────────────────────────────────────────────
# F32-EXACT trainer state (resume without bf16-rounding the masters).
# Layout per prefix, IN THIS ORDER:
#     "<prefix>.lora_A.weight"  BF16 [rank,in]   (downcast compat / warm-load probe)
#     "<prefix>.lora_B.weight"  BF16 [out,rank]
#     "<prefix>.lora_A.master"  F32  [rank,in]   (resume-critical F32 masters)
#     "<prefix>.lora_B.master"  F32  [out,rank]
#     "<prefix>.lora_A.adam_m"  F32
#     "<prefix>.lora_A.adam_v"  F32
#     "<prefix>.lora_B.adam_m"  F32
#     "<prefix>.lora_B.adam_v"  F32
# then "__meta__" if `meta` is non-empty. The BF16 `.weight` copies keep the file
# openable by the plain warm loader (load_lora_train_state) and by external probes
# that expect PEFT-style A/B; the `.master` F32 keys are the round-trip-exact copy.
# ─────────────────────────────────────────────────────────────────────────────
def save_lora_train_state_f32(
    states: List[F32LoraState], path: String, ctx: DeviceContext,
    var meta: List[Float32] = List[Float32](),
) raises -> Int:
    """Write F32-exact LoRA trainer state: bf16 A/B compat copies + F32 masters +
    AdamW moments. Returns the number of adapters written. See the header for the
    exact per-prefix key order. `meta` (optional) is the F32 `__meta__` resume-guard
    vector (step/seed), read back with `load_lora_train_state_meta`.

    HOST-DIRECT: the payload is packed from the host lists and pwritten via
    save_safetensors_host — NO device staging. The old device path transiently
    allocated the whole state on the GPU (~1.4 GB for the 384-adapter LTX2 F32
    state) on a trainer already peaking ~22.5/24 GiB — an OOM exposure at every
    save (Alex, 2026-07-16). `ctx` stays in the signature for call-site
    stability only; this function never touches the device."""
    if len(states) == 0:
        raise Error("save_lora_train_state_f32: refusing to write an empty state")
    _ = ctx

    var names = List[String]()
    var descs = List[HostTensorDesc]()
    for ref s in states:
        var na = s.rank * s.in_f
        var nb = s.out_f * s.rank
        if len(s.a) != na or len(s.ma) != na or len(s.va) != na:
            raise Error(String("save_lora_train_state_f32: A/m/v shape mismatch for ") + s.prefix)
        if len(s.b) != nb or len(s.mb) != nb or len(s.vb) != nb:
            raise Error(String("save_lora_train_state_f32: B/m/v shape mismatch for ") + s.prefix)
        # bf16 A/B compat copies (external probes / warm resume).
        names.append(s.prefix + ".lora_A.weight")
        descs.append(HostTensorDesc(
            STDtype.BF16, _shape2(s.rank, s.in_f), _bf16_le_bytes_from_f32(s.a)))
        names.append(s.prefix + ".lora_B.weight")
        descs.append(HostTensorDesc(
            STDtype.BF16, _shape2(s.out_f, s.rank), _bf16_le_bytes_from_f32(s.b)))
        # F32-exact masters (resume-critical).
        names.append(s.prefix + ".lora_A.master")
        descs.append(HostTensorDesc(
            STDtype.F32, _shape2(s.rank, s.in_f), _f32_le_bytes(s.a)))
        names.append(s.prefix + ".lora_B.master")
        descs.append(HostTensorDesc(
            STDtype.F32, _shape2(s.out_f, s.rank), _f32_le_bytes(s.b)))
        # AdamW moments (F32).
        names.append(s.prefix + ".lora_A.adam_m")
        descs.append(HostTensorDesc(
            STDtype.F32, _shape2(s.rank, s.in_f), _f32_le_bytes(s.ma)))
        names.append(s.prefix + ".lora_A.adam_v")
        descs.append(HostTensorDesc(
            STDtype.F32, _shape2(s.rank, s.in_f), _f32_le_bytes(s.va)))
        names.append(s.prefix + ".lora_B.adam_m")
        descs.append(HostTensorDesc(
            STDtype.F32, _shape2(s.out_f, s.rank), _f32_le_bytes(s.mb)))
        names.append(s.prefix + ".lora_B.adam_v")
        descs.append(HostTensorDesc(
            STDtype.F32, _shape2(s.out_f, s.rank), _f32_le_bytes(s.vb)))

    var n_meta = len(meta)
    if n_meta > 0:
        names.append(String("__meta__"))
        descs.append(HostTensorDesc(STDtype.F32, _shape1(n_meta), _f32_le_bytes(meta)))

    save_safetensors_host(names, descs, path)
    return len(states)


def lora_train_state_has_f32_masters(path: String, probe_prefix: String) raises -> Bool:
    """True if `path` carries F32-exact LoRA masters — probed via the
    `<probe_prefix>.lora_A.master` key. False if the file is a warm/old-era state
    (bf16 A/B + moments, no masters) or cannot be opened. Lets a resume path choose
    the F32-exact load (`load_lora_train_state_f32`) vs the bf16 warm fallback
    (`load_lora_train_state`)."""
    try:
        var st = SafeTensors.open(path)
        return (probe_prefix + ".lora_A.master") in st.tensors
    except:
        return False


def load_lora_train_state_f32(
    prefixes: List[String], path: String, ctx: DeviceContext
) raises -> List[F32LoraState]:
    """Read the F32-exact trainer state written by `save_lora_train_state_f32`.

    Requires, per prefix, the `.lora_A.master` / `.lora_B.master` F32 keys plus the
    four AdamW moment keys, and reads all six BIT-EXACT (the F32 branch of
    `_read_f32`). rank/in/out come from the master shapes (A [rank,in], B
    [out,rank]). A file missing `.master` is an old-era (bf16-only) state — this
    raises telling the caller to warm-resume via `load_lora_train_state` instead."""
    var st = SafeTensors.open(path)
    var out = List[F32LoraState]()

    for ref pfx in prefixes:
        var key_a = pfx + ".lora_A.master"
        var key_b = pfx + ".lora_B.master"
        var key_ma = pfx + ".lora_A.adam_m"
        var key_va = pfx + ".lora_A.adam_v"
        var key_mb = pfx + ".lora_B.adam_m"
        var key_vb = pfx + ".lora_B.adam_v"
        if key_a not in st.tensors or key_b not in st.tensors:
            raise Error(
                String("load_lora_train_state_f32: missing ") + key_a
                + " (old-era state; warm resume via load_lora_train_state)")
        if key_ma not in st.tensors:
            raise Error(String("load_lora_train_state_f32: missing ") + key_ma)
        if key_va not in st.tensors:
            raise Error(String("load_lora_train_state_f32: missing ") + key_va)
        if key_mb not in st.tensors:
            raise Error(String("load_lora_train_state_f32: missing ") + key_mb)
        if key_vb not in st.tensors:
            raise Error(String("load_lora_train_state_f32: missing ") + key_vb)

        var a_info = st.tensor_info(key_a)
        var b_info = st.tensor_info(key_b)
        if len(a_info.shape) != 2 or len(b_info.shape) != 2:
            raise Error(String("load_lora_train_state_f32: master A/B must be 2-D for ") + pfx)
        var rank = a_info.shape[0]
        var in_f = a_info.shape[1]
        var out_f = b_info.shape[0]
        if b_info.shape[1] != rank:
            raise Error(String("load_lora_train_state_f32: B master rank mismatch for ") + pfx)

        var a_h = _read_f32(st, key_a, ctx)
        var b_h = _read_f32(st, key_b, ctx)
        var ma_h = _read_f32(st, key_ma, ctx)
        var va_h = _read_f32(st, key_va, ctx)
        var mb_h = _read_f32(st, key_mb, ctx)
        var vb_h = _read_f32(st, key_vb, ctx)
        if len(a_h) != rank * in_f:
            raise Error(String("load_lora_train_state_f32: A master len mismatch for ") + pfx)
        if len(b_h) != out_f * rank:
            raise Error(String("load_lora_train_state_f32: B master len mismatch for ") + pfx)
        if len(ma_h) != len(a_h) or len(va_h) != len(a_h):
            raise Error(String("load_lora_train_state_f32: A moment len mismatch for ") + pfx)
        if len(mb_h) != len(b_h) or len(vb_h) != len(b_h):
            raise Error(String("load_lora_train_state_f32: B moment len mismatch for ") + pfx)

        out.append(F32LoraState(
            pfx.copy(), a_h^, b_h^, ma_h^, va_h^, mb_h^, vb_h^,
            rank, in_f, out_f,
        ))

    return out^

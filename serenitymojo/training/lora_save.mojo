# lora_save.mojo — save / load TRAINED LoRA adapters as a PEFT/ai-toolkit-keyed
# safetensors. The LoRA-WEIGHTS half of resume; the training-STATE half (F32
# master + AdamW m/v + step counter) already exists in training/loop.mojo
# (TrainState / save_checkpoint / load_checkpoint) and is reused unchanged.
#
# ── Why this file exists ─────────────────────────────────────────────────────
# training/loop.mojo persists the GENERIC optimizer state (param.<i>/adam_m.<i>/
# adam_v.<i>/__meta__) for an opaque parameter set. It does NOT know the LoRA
# key naming, so a loop checkpoint cannot be opened by an inference loader
# (lora.mojo) or by ai-toolkit. This module writes the trained A/B in the
# canonical PEFT key convention so:
#   * lora.mojo::LoraSet.load detects it as FMT_DIFFUSION_MODEL and merges it,
#   * the validation sampler (training/validation_sampler.mojo) can load it,
#   * external tools (ai-toolkit / diffusers PEFT) open it.
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
# NOTE on scale/alpha: we do NOT write a per-module `.alpha` scalar. The
# LoraAdapter carries `scale = alpha/rank` (train_step.mojo:153), but the
# canonical PEFT/train_klein file omits `.alpha`, and lora.mojo::_module_scale
# (lora.mojo:~300-320) then DEFAULTS alpha = module_rank → scale = multiplier.
# To reproduce the trained scale at inference the caller passes the SAME
# alpha/rank ratio as the `multiplier` to merge_into_indexed. (Writing a `.alpha`
# tensor is a 3-line extension flagged in the return notes if exact-alpha
# round-trip without a multiplier is wanted later.)
#
# Mojo 1.0.0b1: `def` not `fn`; move-only Tensor → collections hold
# ArcPointer[Tensor]; STDtype.F32 is a value; from_host(values, shape, dtype, ctx).

from std.collections import List
from std.memory import ArcPointer
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.training.train_step import LoraAdapter


# ── A trained LoRA module paired with the base-weight prefix it adapts ───────
# `prefix` is the LoRA module name WITHOUT the lora_A/lora_B suffix, e.g.
# "double_blocks.0.img_attn.to_q" — exactly the `prefix` lora.mojo strips back
# off in LoraSet.load (lora.mojo:~370). The save appends ".lora_A.weight" /
# ".lora_B.weight" to it.
@fieldwise_init
struct NamedLora(Copyable, Movable):
    var prefix: String
    var adapter: LoraAdapter


# ── F32 device tensor from a host List[Float32] with a 2-D shape ─────────────
# Mirrors train_step.mojo:169 (`Tensor.from_host(x_h.copy(), [M, in], F32, ctx)`).
# A/B are F32 in the LoraAdapter (host master precision per MOJO_CONVENTIONS §3
# "training masters are F32 throughout"), so the saved file is F32 — byte-exact
# on reload (no BF16 truncation), the same property loop.mojo relies on for the
# F32 masters.
def _f32_2d(var values: List[Float32], rows: Int, cols: Int, ctx: DeviceContext) raises -> Tensor:
    var sh = List[Int]()
    sh.append(rows)
    sh.append(cols)
    return Tensor.from_host(values^, sh^, STDtype.F32, ctx)


# ─────────────────────────────────────────────────────────────────────────────
# SAVE: pack each adapter's A [rank,in] and B [out,rank] into a single
# safetensors via the proven byte-exact writer (io/safetensors_writer.mojo:186).
# Keys are PEFT/ai-toolkit: "<prefix>.lora_A.weight" / "<prefix>.lora_B.weight".
# ─────────────────────────────────────────────────────────────────────────────
def save_lora_peft(
    adapters: List[NamedLora], path: String, ctx: DeviceContext
) raises -> Int:
    """Write `adapters` to `path` as a PEFT-keyed LoRA safetensors. Returns the
    number of (A,B) PAIRS written. Tensors are F32, byte-exact on reload.

    For each NamedLora we emit two tensors, in A-then-B order per module:
        "<prefix>.lora_A.weight"  F32 [rank, in]   (== LoraAdapter.a)
        "<prefix>.lora_B.weight"  F32 [out, rank]  (== LoraAdapter.b)
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
        tensors.append(ArcPointer(_f32_2d(a.a.copy(), a.rank, a.in_f, ctx)))
        names.append(nl.prefix + ".lora_B.weight")
        tensors.append(ArcPointer(_f32_2d(a.b.copy(), a.out_f, a.rank, ctx)))

    save_safetensors(names, tensors, path, ctx)
    return len(adapters)


def save_lora_train_state(
    adapters: List[NamedLora], path: String, ctx: DeviceContext
) raises -> Int:
    """Write trainer-only LoRA state: A/B plus AdamW moments.

    This is intentionally separate from the PEFT file. The PEFT file stays
    plain external-compatible LoRA, while this state file lets the pure-Mojo
    cadence supervisor resume without zeroing AdamW moments.
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
        tensors.append(ArcPointer(_f32_2d(a.a.copy(), a.rank, a.in_f, ctx)))
        names.append(nl.prefix + ".lora_B.weight")
        tensors.append(ArcPointer(_f32_2d(a.b.copy(), a.out_f, a.rank, ctx)))
        names.append(nl.prefix + ".lora_A.adam_m")
        tensors.append(ArcPointer(_f32_2d(a.ma.copy(), a.rank, a.in_f, ctx)))
        names.append(nl.prefix + ".lora_A.adam_v")
        tensors.append(ArcPointer(_f32_2d(a.va.copy(), a.rank, a.in_f, ctx)))
        names.append(nl.prefix + ".lora_B.adam_m")
        tensors.append(ArcPointer(_f32_2d(a.mb.copy(), a.out_f, a.rank, ctx)))
        names.append(nl.prefix + ".lora_B.adam_v")
        tensors.append(ArcPointer(_f32_2d(a.vb.copy(), a.out_f, a.rank, ctx)))

    save_safetensors(names, tensors, path, ctx)
    return len(adapters)


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
    """Read the LoRA file written by save_lora_peft back into LoraAdapters, one
    per `prefix`. `scale` (= alpha/rank) is re-supplied by the caller (it is not
    persisted, matching the no-`.alpha` PEFT convention). AdamW moments are
    zeroed (resume them from a loop.mojo TrainState checkpoint instead).

    Shapes are read from the file header: A is [rank,in], B is [out,rank], so
    rank = A.shape[0], in = A.shape[1], out = B.shape[0]. The B.shape[1] is
    asserted to equal rank (the file's own self-consistency check)."""
    var st = SafeTensors.open(path)
    var out = List[NamedLora]()

    for ref pfx in prefixes:
        var key_a = pfx + ".lora_A.weight"
        var key_b = pfx + ".lora_B.weight"
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
            a_h^, b_h^, rank, in_f, out_f, scale, ma^, va^, mb^, vb^
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

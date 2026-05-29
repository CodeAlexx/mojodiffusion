# lora.mojo — merge-at-load LoRA for inference. Pure-Mojo port of
# inference-flame/src/lora_merge.rs (the in-place fuse path) + the key-naming
# conventions from inference-flame/src/lora.rs and models/lora_loader.rs.
#
# Inference-only. This is the MINIMALLY-INVASIVE LoRA path: it never touches a
# model's forward. It loads a LoRA `.safetensors`, computes
#   delta_W = scale * (B @ A),   scale = (alpha / rank) * multiplier
# and merges it into the base model's already-loaded weight Dict in place:
#   W[target] += delta_W
# (with row/col/row-range slicing for the fused-QKV / single-block targets).
# The runtime `forward_lora` overlay path (lora.rs::LoraStack::apply) is NOT
# ported — that requires model-forward changes.
#
# ── LoRA math (matches lora_merge.rs:32-37, 520-550) ─────────────────────────
#   lora_A ("down"): [rank, in_features]   (PyTorch row-major)
#   lora_B ("up")  : [out_features, rank]
#   delta_W = scale * (B @ A)              → [out_features, in_features]
#   scale   = (alpha / rank) * multiplier
# B @ A is computed via `ops.linear(B, transpose(A))` because foundation
# `linear(x, w) = x @ wᵀ`; with w = Aᵀ = [in, rank] this gives B @ (Aᵀ)ᵀ = B @ A.
#
# ── Key conventions (verified against real on-disk headers, 2026-05-26) ───────
#   EriDiffusion-v2 train_klein  : bare `<prefix>.lora_A.weight` / `.lora_B.weight`
#                                  (no `diffusion_model.` prefix), split
#                                  to_q/to_k/to_v/proj + img_mlp.{0,2}. Detected
#                                  as DiffusionModel (suffix match), resolved by
#                                  map_prefix_diffusion_model → `<prefix>.weight`.
#   edv2-reference (FLUX/Klein)  : `diffusion_model.<key>.lora_A.weight`.
#   Z-Image trainer              : bare `<prefix>.lora_A` with split
#                                  attention.to_q/to_k/to_v → fused qkv RowRange.
#   kohya / sd-scripts (SDXL)    : `lora_unet_<path>.lora_down.weight` /
#                                  `.lora_up.weight` + per-module `.alpha`
#                                  (scalar, shape []). Confirmed F16 alpha in
#                                  my_sdxl_lora_v1.safetensors.
#
# Mojo 1.0.0b1, NVIDIA GPU. File I/O via io/safetensors only.

from std.memory import ArcPointer
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors, TensorRef
from serenitymojo.io.tensor_view import TensorView, from_parts
from serenitymojo.ops.tensor_algebra import transpose, concat, slice, add, mul_scalar
from serenitymojo.ops.linear import linear
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.models.dit.ltx2_dit import LTX2AVBlockWeights


# ── Slot kinds (mirror lora_merge.rs::Slot, lines 53-64) ─────────────────────
comptime SLOT_FULL = 0
"""Full overlay: base shape == delta shape. base += delta."""
comptime SLOT_ROWS = 1
"""Top `n` rows of base get delta. base[..n, :] += delta. (Klein-4B linear1.)"""
comptime SLOT_COLS = 2
"""Left `n` cols of base get delta. base[:, ..n] += delta. (Klein-4B linear2.)"""
comptime SLOT_ROWRANGE = 3
"""Row-range [start..start+len] of base gets delta. (Z-Image split-QKV → fused.)"""

# ── LoRA file formats (mirror lora_merge.rs::LoraFormat, lines 67-85) ────────
comptime FMT_KLEIN_TRAINER = 0
comptime FMT_ZIMAGE_TRAINER = 1
comptime FMT_DIFFUSION_MODEL = 2
comptime FMT_KOHYA_SDXL = 3
comptime FMT_LTX2_DISTILLED = 4
"""LTX-2.3 22B distilled rank-384 LoRA. Keys
`diffusion_model.transformer_blocks.{i}.<module>.lora_{A,B}.weight` for the six
attn families (attn1/attn2/audio_attn1/audio_attn2/audio_to_video_attn/
video_to_audio_attn) × {to_q,to_k,to_v,to_out.0,to_gate_logits} + ff/audio_ff
+ the global adaln/prompt/av_ca/patchify/proj_out families. The base weights are
the FP8-streamed-and-dequanted DiT block linears (no persistent resident W), so
the delta is ADDED at the dequanted block linear per stream (never a saved
fuse). Scale = `multiplier` (the LTX-2 path uses `strength*(B@A)`, NO alpha/rank
division — lora_loader.rs:110-116)."""

# Slot constants. Klein-4B single-block linear slicing (lora_merge.rs:46-47).
comptime SINGLE_QKV_ROWS = 9216
comptime SINGLE_OUT_COLS = 3072
# Z-Image fused QKV is [3*dim, dim], dim=3840 (lora_merge.rs:51).
comptime ZIMAGE_DIM = 3840


@fieldwise_init
struct LoraMapping(Copyable, Movable):
    """One resolved LoRA module: which prefix, which base weight key it merges
    into, and how (slot kind + slot params). Mirrors the `(base_key, slot)`
    pairs returned by the lora_merge.rs prefix mappers."""

    var prefix: String
    """LoRA prefix, e.g. `double_blocks.0.img_attn.to_q`."""
    var base_key: String
    """Base weight Dict key, e.g. `double_blocks.0.img_attn.to_q.weight`."""
    var slot_kind: Int
    """One of SLOT_FULL / SLOT_ROWS / SLOT_COLS / SLOT_ROWRANGE."""
    var slot_a: Int
    """Slot param: Rows→n, Cols→n, RowRange→start. (unused for Full.)"""
    var slot_b: Int
    """Slot param: RowRange→len. (unused otherwise.)"""


def _detect_format(names: List[String]) -> Int:
    """Detect the LoRA file format from its key shapes. Mirrors
    lora_merge.rs::detect_format (lines 91-111). Order matters: kohya first
    (lora_unet_/te + lora_down/up), then DiffusionModel (.lora_A.weight),
    then Z-Image trainer (split attention/feed_forward), else KleinTrainer."""
    var has_kohya_prefix = False
    var has_kohya_suffix = False
    var has_dm_suffix = False
    var has_zimage = False
    var has_ltx2 = False
    for ref n in names:
        if (
            n.startswith("lora_unet_")
            or n.startswith("lora_te1_")
            or n.startswith("lora_te2_")
        ):
            has_kohya_prefix = True
        if n.endswith(".lora_down.weight") or n.endswith(".lora_up.weight"):
            has_kohya_suffix = True
        if n.endswith(".lora_A.weight") or n.endswith(".lora_B.weight"):
            has_dm_suffix = True
        if (
            (".attention.to_q.lora_" in n)
            or (".feed_forward.w1.lora_" in n)
        ):
            has_zimage = True
        # LTX-2 distilled signature: the cross-modal AV attention family is
        # unique to the LTX-2 joint dual-stream DiT (no other format ships
        # `audio_to_video_attn` / `video_to_audio_attn` LoRA modules).
        if (
            (".audio_to_video_attn." in n)
            or (".video_to_audio_attn." in n)
            or (".audio_attn1." in n)
        ):
            has_ltx2 = True
    if has_kohya_prefix and has_kohya_suffix:
        return FMT_KOHYA_SDXL
    # LTX-2 must be checked BEFORE the generic DiffusionModel branch: it shares
    # the `.lora_A.weight` suffix but needs the LTX-2 base-key map + scale rule.
    if has_ltx2 and has_dm_suffix:
        return FMT_LTX2_DISTILLED
    if has_dm_suffix:
        return FMT_DIFFUSION_MODEL
    if has_zimage:
        return FMT_ZIMAGE_TRAINER
    return FMT_KLEIN_TRAINER


def _suffix_a(fmt: Int) -> String:
    """lora_A suffix for the format (lora_merge.rs:427-431)."""
    if fmt == FMT_DIFFUSION_MODEL or fmt == FMT_LTX2_DISTILLED:
        return ".lora_A.weight"
    if fmt == FMT_KOHYA_SDXL:
        return ".lora_down.weight"
    return ".lora_A"  # KleinTrainer / ZImageTrainer


def _suffix_b(fmt: Int) -> String:
    """lora_B suffix for the format (lora_merge.rs:427-431)."""
    if fmt == FMT_DIFFUSION_MODEL or fmt == FMT_LTX2_DISTILLED:
        return ".lora_B.weight"
    if fmt == FMT_KOHYA_SDXL:
        return ".lora_up.weight"
    return ".lora_B"  # KleinTrainer / ZImageTrainer


def _substr_bytes(s: String, start: Int, end: Int) -> String:
    """Build s[start:end] over BYTES. Tensor key names are ASCII, so per-byte
    chr() reconstruction is exact here (range-slice `s[byte=a:byte=b]` is not
    supported in 1.0.0b1; the codebase builds substrings byte-by-byte)."""
    var out = String("")
    var bytes = s.as_bytes()
    for i in range(start, end):
        out += chr(Int(bytes[i]))
    return out


def _strip_suffix(s: String, suf: String) -> String:
    """Return s without trailing `suf`, or "" if it doesn't end with suf.
    (Mojo String has no strip_suffix; "" signals no-match since real prefixes
    are never empty.)"""
    if s.endswith(suf) and s.byte_length() > suf.byte_length():
        return _substr_bytes(s, 0, s.byte_length() - suf.byte_length())
    return String("")


def _strip_prefix(s: String, pre: String) -> String:
    """Return s without leading `pre` if present, else s unchanged."""
    if s.startswith(pre):
        return _substr_bytes(s, pre.byte_length(), s.byte_length())
    return s


def _strip_block_prefix(base_key: String, block_idx: Int) -> String:
    """If `base_key` is `transformer_blocks.{block_idx}.<rest>`, return `<rest>`;
    otherwise "". Used by the LTX-2 runtime apply to map a full base key to the
    block-local canonical name the `LTX2AVBlockWeights` dict is keyed by."""
    var bp = String("transformer_blocks.") + String(block_idx) + "."
    if base_key.startswith(bp):
        return _substr_bytes(base_key, bp.byte_length(), base_key.byte_length())
    return String("")


def _is_block_key(base_key: String) -> Bool:
    return base_key.startswith("transformer_blocks.")


def _map_klein_trainer(prefix: String) -> LoraMapping:
    """Klein-trainer prefix → base key + slot. Mirrors
    lora_merge.rs::map_prefix_klein_trainer (lines 113-139). Returns a mapping
    with base_key=="" when the prefix doesn't map (caller skips)."""
    if prefix.startswith("input_bridges."):
        return LoraMapping(prefix, String(""), SLOT_FULL, 0, 0)  # trainer-only

    var rest = _strip_suffix(prefix, ".img_attn.qkv_proj")
    if rest.byte_length() > 0:
        return LoraMapping(prefix, rest + ".img_attn.qkv.weight", SLOT_FULL, 0, 0)
    rest = _strip_suffix(prefix, ".img_attn.out_proj")
    if rest.byte_length() > 0:
        return LoraMapping(prefix, rest + ".img_attn.proj.weight", SLOT_FULL, 0, 0)
    rest = _strip_suffix(prefix, ".txt_attn.qkv_proj")
    if rest.byte_length() > 0:
        return LoraMapping(prefix, rest + ".txt_attn.qkv.weight", SLOT_FULL, 0, 0)
    rest = _strip_suffix(prefix, ".txt_attn.out_proj")
    if rest.byte_length() > 0:
        return LoraMapping(prefix, rest + ".txt_attn.proj.weight", SLOT_FULL, 0, 0)
    if prefix.startswith("single_blocks."):
        rest = _strip_suffix(prefix, ".qkv_proj")
        if rest.byte_length() > 0:
            return LoraMapping(
                prefix, rest + ".linear1.weight", SLOT_ROWS, SINGLE_QKV_ROWS, 0
            )
        rest = _strip_suffix(prefix, ".out_proj")
        if rest.byte_length() > 0:
            return LoraMapping(
                prefix, rest + ".linear2.weight", SLOT_COLS, SINGLE_OUT_COLS, 0
            )
    return LoraMapping(prefix, String(""), SLOT_FULL, 0, 0)


def _map_zimage_trainer(prefix: String) -> LoraMapping:
    """Z-Image trainer prefix → base key + slot. Mirrors
    lora_merge.rs::map_prefix_zimage_trainer (lines 373-402). Split Q/K/V merge
    into the fused `attention.qkv.weight` row ranges; out + feed_forward full."""
    var rest = _strip_suffix(prefix, ".attention.to_q")
    if rest.byte_length() > 0:
        return LoraMapping(
            prefix, rest + ".attention.qkv.weight", SLOT_ROWRANGE, 0, ZIMAGE_DIM
        )
    rest = _strip_suffix(prefix, ".attention.to_k")
    if rest.byte_length() > 0:
        return LoraMapping(
            prefix, rest + ".attention.qkv.weight", SLOT_ROWRANGE, ZIMAGE_DIM, ZIMAGE_DIM
        )
    rest = _strip_suffix(prefix, ".attention.to_v")
    if rest.byte_length() > 0:
        return LoraMapping(
            prefix,
            rest + ".attention.qkv.weight",
            SLOT_ROWRANGE,
            2 * ZIMAGE_DIM,
            ZIMAGE_DIM,
        )
    rest = _strip_suffix(prefix, ".attention.out")
    if rest.byte_length() > 0:
        return LoraMapping(prefix, rest + ".attention.out.weight", SLOT_FULL, 0, 0)
    if (
        prefix.endswith(".feed_forward.w1")
        or prefix.endswith(".feed_forward.w2")
        or prefix.endswith(".feed_forward.w3")
    ):
        return LoraMapping(prefix, prefix + ".weight", SLOT_FULL, 0, 0)
    return LoraMapping(prefix, String(""), SLOT_FULL, 0, 0)


def _map_diffusion_model(prefix: String) -> LoraMapping:
    """edv2-reference / bare-PEFT prefix → base key. Strip leading
    `diffusion_model.` and trailing `.default`, then append `.weight` (unless
    the prefix already embeds `.weight`). Full overlay. Mirrors
    lora_merge.rs::map_prefix_diffusion_model (lines 143-149) AND the PEFT
    `.weight`-embedded fix from lora.rs:763-768 (avoids `.weight.weight`)."""
    var stripped = _strip_prefix(prefix, "diffusion_model.")
    var no_default = _strip_suffix(stripped, ".default")
    if no_default.byte_length() > 0:
        stripped = no_default
    var target: String
    if stripped.endswith(".weight"):
        target = stripped
    else:
        target = stripped + ".weight"
    return LoraMapping(prefix, target, SLOT_FULL, 0, 0)


def _map_klein_split_qkv(prefix: String, out_dim: Int) -> LoraMapping:
    """Klein DiffusionModel-format SPLIT Q/K/V → FUSED qkv RowRange mapper.

    The EriDiffusion-v2 `train_klein` LoRA ships SPLIT
    `double_blocks.<i>.{img,txt}_attn.to_q/to_k/to_v.lora_A.weight` (no
    `diffusion_model.` prefix → detected as DiffusionModel). The Klein9B base
    stores a FUSED `double_blocks.<i>.{img,txt}_attn.qkv.weight` of shape
    `[3*out, in]` (out = inner_dim, e.g. 4096 for 9B). The generic
    `_map_diffusion_model` would emit a nonexistent `...to_q.weight` key and the
    module would silently no-op at merge — so route Q/K/V into the fused
    qkv.weight row-ranges `0/out/2*out`, len `out`, exactly mirroring the
    Z-Image branch in `lora.rs::map_prefix_diffusion_model` (lines 730-750) but
    keyed on Klein's `.img_attn`/`.txt_attn` naming instead of `.attention`.

    `out_dim` is the per-module out features, read from the B tensor's shape[0]
    (B is `[out, rank]`) at load time — NOT hardcoded — so a base whose fused
    qkv first dim != 3*out_dim trips the `_apply_slot` RowRange bounds check
    (`start + len > bdims[0]`) and is skipped at merge (the shape gate the task
    asked for). Returns base_key=="" when `prefix` is not a Klein split-QKV name
    (caller falls back to the generic DiffusionModel mapper)."""
    var rest = _strip_suffix(prefix, ".img_attn.to_q")
    if rest.byte_length() > 0:
        return LoraMapping(prefix, rest + ".img_attn.qkv.weight", SLOT_ROWRANGE, 0, out_dim)
    rest = _strip_suffix(prefix, ".img_attn.to_k")
    if rest.byte_length() > 0:
        return LoraMapping(prefix, rest + ".img_attn.qkv.weight", SLOT_ROWRANGE, out_dim, out_dim)
    rest = _strip_suffix(prefix, ".img_attn.to_v")
    if rest.byte_length() > 0:
        return LoraMapping(prefix, rest + ".img_attn.qkv.weight", SLOT_ROWRANGE, 2 * out_dim, out_dim)
    rest = _strip_suffix(prefix, ".txt_attn.to_q")
    if rest.byte_length() > 0:
        return LoraMapping(prefix, rest + ".txt_attn.qkv.weight", SLOT_ROWRANGE, 0, out_dim)
    rest = _strip_suffix(prefix, ".txt_attn.to_k")
    if rest.byte_length() > 0:
        return LoraMapping(prefix, rest + ".txt_attn.qkv.weight", SLOT_ROWRANGE, out_dim, out_dim)
    rest = _strip_suffix(prefix, ".txt_attn.to_v")
    if rest.byte_length() > 0:
        return LoraMapping(prefix, rest + ".txt_attn.qkv.weight", SLOT_ROWRANGE, 2 * out_dim, out_dim)
    return LoraMapping(prefix, String(""), SLOT_FULL, 0, 0)


def _resolve_mapping(fmt: Int, prefix: String) -> LoraMapping:
    """Dispatch the prefix → (base_key, slot) mapping by format. kohya SDXL
    naming-rewrite is NOT ported (SDXL UNet not in this stack); kohya prefixes
    are resolved by direct dotted-name reconstruction only — see merge_into.

    NOTE: the Klein split→fused QKV case (`.img_attn`/`.txt_attn.to_q/k/v` →
    fused `qkv.weight` RowRange) is NOT handled here because it needs the
    per-module out-dim from the B tensor shape; `LoraSet.load` resolves it via
    `_map_klein_split_qkv` before falling back to this generic mapper."""
    if fmt == FMT_KLEIN_TRAINER:
        return _map_klein_trainer(prefix)
    if fmt == FMT_ZIMAGE_TRAINER:
        return _map_zimage_trainer(prefix)
    # FMT_DIFFUSION_MODEL, FMT_KOHYA_SDXL and FMT_LTX2_DISTILLED all strip the
    # `diffusion_model.` prefix and append `.weight` (SLOT_FULL). For LTX-2 the
    # base linears are SEPARATE to_q/to_k/to_v (no fused-QKV remap), so the
    # generic DiffusionModel mapper is exactly correct: it yields the full base
    # key `transformer_blocks.{i}.<mod>.weight` (the runtime AV-block apply hook
    # strips the per-block prefix to match the block-local weight dict).
    return _map_diffusion_model(prefix)


def _read_scalar_alpha(st: SafeTensors, key: String, ctx: DeviceContext) raises -> Float32:
    """Read a per-module `.alpha` scalar tensor (shape [], any compute dtype) as
    F32. Mirrors lora.rs:273-281 / lora_merge.rs:504-515. Returns the single
    value; `key` must exist (caller checks membership first)."""
    var info = st.tensor_info(key)
    var data = st.tensor_bytes(key)
    var tv = from_parts(info.dtype, info.shape.copy(), data)
    var t = Tensor.from_view(tv, ctx)
    var host = t.to_host(ctx)  # upcasts to F32
    if len(host) == 0:
        return 0.0
    return host[0]


struct LoraSet(Movable):
    """A loaded LoRA file ready to merge into a base weight Dict. Holds an open
    `SafeTensors` handle (mmap'd, not eagerly read), the detected format, and
    the resolved per-module mappings (prefix → base_key + slot).

    Movable-not-Copyable: owns the SafeTensors handle (which owns its mmap).
    Mirrors the load half of lora_merge.rs::merge_klein_lora (lines 440-498)."""

    var st: SafeTensors
    var format: Int
    var mappings: List[LoraMapping]
    var suffix_a: String
    var suffix_b: String

    def __init__(out self, var st: SafeTensors, format: Int, var mappings: List[LoraMapping], var suffix_a: String, var suffix_b: String):
        self.st = st^
        self.format = format
        self.mappings = mappings^
        self.suffix_a = suffix_a^
        self.suffix_b = suffix_b^

    @staticmethod
    def load(path: String) raises -> LoraSet:
        """Open a LoRA `.safetensors`, detect its format, and resolve every
        `<prefix>{suffix_a}` pair into a (base_key, slot) mapping. The data
        segment is mmap'd — tensors are loaded H2D lazily in `merge_into`.

        kohya SDXL: only direct-named (dotted→underscore) prefixes resolve;
        text-encoder (lora_te*) and diffusers-named UNet LoRAs are skipped (the
        diffusers→LDM rewriter is not ported — see module header)."""
        var st = SafeTensors.open(path)
        var names = st.names()
        var fmt = _detect_format(names)
        var sa = _suffix_a(fmt)
        var sb = _suffix_b(fmt)

        # Index prefixes from lora_A keys, resolve each to a mapping. A mapping
        # with base_key=="" is dropped (trainer-only / unknown / skipped).
        var mappings = List[LoraMapping]()
        for ref n in names:
            var prefix = _strip_suffix(n, sa)
            if prefix.byte_length() == 0:
                continue
            # kohya text-encoder LoRAs are not merged into the UNet base.
            if fmt == FMT_KOHYA_SDXL and (
                prefix.startswith("lora_te1_") or prefix.startswith("lora_te2_")
            ):
                continue
            # Klein DiffusionModel SPLIT Q/K/V → FUSED qkv RowRange. The base
            # stores fused `{img,txt}_attn.qkv.weight` but train_klein ships
            # split `to_q/k/v`; route them into row-ranges using the per-module
            # out-dim read from the B tensor (`[out, rank]`), NOT hardcoded.
            # Falls through to the generic DiffusionModel mapper otherwise.
            if fmt == FMT_DIFFUSION_MODEL:
                var key_b = prefix + sb
                if key_b in st.tensors:
                    var b_shape = st.tensor_info(key_b).shape.copy()
                    if len(b_shape) >= 1 and b_shape[0] > 0:
                        var ksplit = _map_klein_split_qkv(prefix, b_shape[0])
                        if ksplit.base_key.byte_length() > 0:
                            mappings.append(ksplit^)
                            continue
            var m = _resolve_mapping(fmt, prefix)
            if m.base_key.byte_length() == 0:
                continue
            mappings.append(m^)

        return LoraSet(st^, fmt, mappings^, sa^, sb^)

    def num_mappings(self) -> Int:
        """Number of resolved LoRA modules (post key-resolution, pre base-match)."""
        return len(self.mappings)

    def format_name(self) -> String:
        if self.format == FMT_KLEIN_TRAINER:
            return String("KleinTrainer")
        if self.format == FMT_ZIMAGE_TRAINER:
            return String("ZImageTrainer")
        if self.format == FMT_DIFFUSION_MODEL:
            return String("DiffusionModel")
        if self.format == FMT_LTX2_DISTILLED:
            return String("LTX2Distilled")
        return String("KohyaSdxl")

    def _load_lora_tensor(self, key: String, ctx: DeviceContext) raises -> Tensor:
        """H2D-load one LoRA tensor by full key from the mmap'd file."""
        var info = self.st.tensor_info(key)
        var data = self.st.tensor_bytes(key)
        var tv = from_parts(info.dtype, info.shape.copy(), data)
        return Tensor.from_view(tv, ctx)

    def _module_scale(
        self, m: LoraMapping, multiplier: Float32, ctx: DeviceContext
    ) raises -> Float32:
        """Resolve the scale for one module PER-MODULE (no file-level alpha/rank
        defaulting), exactly mirroring lora.rs:273-282:
            module_rank = A.shape[0]   (A is [rank, in])
            alpha       = <prefix>.alpha scalar if present, else module_rank
            scale       = (alpha / module_rank) * multiplier
        When `.alpha` is absent the alpha defaults to module_rank so
        alpha/rank == 1 and scale == multiplier — matching our trainers'
        default and the per-module derivation the binary
        klein_lora_infer.rs:253-261 relies on. This makes the scale robust to
        mixed-rank files and to callers that pass a mismatched file-level rank
        (the old fallback computed alpha/rank against caller args, diverging
        from canonical when caller rank != module rank)."""
        var key_a = m.prefix + self.suffix_a
        var a_info = self.st.tensor_info(key_a)
        var module_rank = a_info.shape[0]
        if module_rank <= 0:
            return multiplier
        var alpha_key = m.prefix + ".alpha"
        if alpha_key in self.st.tensors:
            var alpha_v = _read_scalar_alpha(self.st, alpha_key, ctx)
            return (alpha_v / Float32(module_rank)) * multiplier
        # No `.alpha`: alpha defaults to module_rank → scale = multiplier.
        return multiplier

    def _compute_delta(
        self, m: LoraMapping, scale: Float32, base_dtype: STDtype, ctx: DeviceContext
    ) raises -> Tensor:
        """Load A,B and return the scaled delta = scale*(B@A) cast to base_dtype.
        B @ A via linear(B, Aᵀ) = B @ (Aᵀ)ᵀ = B @ A (foundation linear = x@wᵀ)."""
        var key_a = m.prefix + self.suffix_a
        var key_b = m.prefix + self.suffix_b
        var a = self._load_lora_tensor(key_a, ctx)  # [rank, in]
        var b = self._load_lora_tensor(key_b, ctx)  # [out, rank]
        # linear needs matching dtypes; cast A to B's dtype first.
        var a_for_mm: Tensor
        if a.dtype() != b.dtype():
            a_for_mm = cast_tensor(a, b.dtype(), ctx)
        else:
            a_for_mm = a^
        var a_t = transpose(a_for_mm, 0, 1, ctx)  # [in, rank]
        var prod = linear(b, a_t, None, ctx)  # [out, in]
        var delta = mul_scalar(prod, scale, ctx)
        if delta.dtype() != base_dtype:
            return cast_tensor(delta, base_dtype, ctx)
        return delta^

    def _pair_present(self, m: LoraMapping) raises -> Bool:
        """True iff both lora_A and lora_B exist and neither is 4D (conv)."""
        var key_a = m.prefix + self.suffix_a
        var key_b = m.prefix + self.suffix_b
        if key_a not in self.st.tensors or key_b not in self.st.tensors:
            return False
        var a_info = self.st.tensor_info(key_a)
        var b_info = self.st.tensor_info(key_b)
        if len(a_info.shape) == 4 or len(b_info.shape) == 4:
            return False  # conv LoRA — not supported (lora.rs:289-294)
        return True

    def merge_into(
        self,
        mut base: Dict[String, ArcPointer[Tensor]],
        multiplier: Float32,
        ctx: DeviceContext,
    ) raises -> Int:
        """Merge every resolved LoRA module into `base` IN PLACE:
            base[target] += scale * (B @ A)
        with row/col/row-range slicing per the resolved slot. Returns the
        number of modules merged.

        Scale is PER-MODULE (lora.rs:273-282): scale = (alpha/module_rank)*
        multiplier, where module_rank = A.shape[0] and alpha is the per-module
        `<prefix>.alpha` scalar if present, else module_rank (→ scale =
        multiplier). There is NO file-level alpha/rank knob — `multiplier` is the
        only caller adjustment. This reads per-module alpha for ALL formats and
        is load-bearing: a musubi/kohya LoRA with alpha=3 rank=96 would
        otherwise be ~32× too strong (lora.rs:60-64), and a mixed-rank file is
        handled correctly per module.

        delta is cast to the base weight's dtype before the add
        (lora_merge.rs:560-564)."""
        var n_merged = 0

        for ref m in self.mappings:
            if not self._pair_present(m):
                continue
            if m.base_key not in base:
                continue  # base key missing — skip (lora_merge.rs:553-556)
            var scale = self._module_scale(m, multiplier, ctx)
            # Dict value is ArcPointer[Tensor] (Tensor is not Copyable) → deref
            # with `[]` to borrow the underlying Tensor.
            var base_dtype = base[m.base_key][].dtype()
            var delta = self._compute_delta(m, scale, base_dtype, ctx)
            var merged = _apply_slot(
                base[m.base_key][], delta, m.slot_kind, m.slot_a, m.slot_b, ctx
            )
            base[m.base_key] = ArcPointer[Tensor](merged^)
            n_merged += 1

        return n_merged

    def merge_into_indexed(
        self,
        mut weights: List[ArcPointer[Tensor]],
        name_to_idx: Dict[String, Int],
        multiplier: Float32,
        ctx: DeviceContext,
    ) raises -> Int:
        """Same as `merge_into` but for the List+name→idx weight layout used by
        the Klein/Z-Image DiT models (`Klein9BDiT.weights` + `.name_to_idx`,
        `NextDiT.weights` + `.name_to_idx`). Mutates `weights[idx]` in place for
        each matched target. This is the non-invasive Klein integration point:
        the caller passes `model.weights` and `model.name_to_idx` after
        `Klein9BDiT.load_full(...)` and before the denoise loop. (The offloaded
        path streams blocks lazily, so it cannot be merged this way — merge the
        all-resident `load_full` model, or fold the delta per block at stream
        time, which is out of scope here.)

        Scale is PER-MODULE (no file-level alpha/rank) — see `merge_into`."""
        var n_merged = 0

        for ref m in self.mappings:
            if not self._pair_present(m):
                continue
            if m.base_key not in name_to_idx:
                continue
            var idx = name_to_idx[m.base_key]
            var scale = self._module_scale(m, multiplier, ctx)
            var base_dtype = weights[idx][].dtype()
            var delta = self._compute_delta(m, scale, base_dtype, ctx)
            var merged = _apply_slot(
                weights[idx][], delta, m.slot_kind, m.slot_a, m.slot_b, ctx
            )
            weights[idx] = ArcPointer[Tensor](merged^)
            n_merged += 1

        return n_merged

    # ── LTX-2 distilled: runtime at-dequant additive apply ──────────────────
    def ltx2_block_mapping_count(self, block_idx: Int) -> Int:
        """How many resolved mappings target `transformer_blocks.{block_idx}.`."""
        var n = 0
        for ref m in self.mappings:
            if _strip_block_prefix(m.base_key, block_idx).byte_length() > 0:
                n += 1
        return n

    def ltx2_global_mapping_count(self) -> Int:
        """How many resolved mappings are NON-block (global) keys."""
        var n = 0
        for ref m in self.mappings:
            if not _is_block_key(m.base_key):
                n += 1
        return n

    def apply_to_av_block(
        self,
        block_idx: Int,
        mut block: LTX2AVBlockWeights,
        multiplier: Float32,
        ctx: DeviceContext,
    ) raises -> Int:
        """ADD every block-level LoRA delta for `block_idx` onto the resident
        dequanted weights of `block`, IN PLACE:
            W[name] += scale * (B @ A)
        This is the LTX-2 at-dequant application hook (HARD RULE): the FP8 block
        was streamed in and dequanted transiently for THIS step, so the delta is
        re-applied to the fresh dequant every time the block streams — never a
        one-time fuse, never written to disk. Returns the number of deltas added.

        FAIL-CLOSED: a mapping whose block-local name is NOT present in `block`
        raises (every LoRA key for the block MUST map to a base linear). Scale =
        `multiplier` (LTX-2 uses `strength*(B@A)`; per-module alpha absent so
        `_module_scale` returns `multiplier`). `to_gate_logits` (rank 32) and
        every attn/ff family are covered identically."""
        var n_applied = 0
        for ref m in self.mappings:
            var local = _strip_block_prefix(m.base_key, block_idx)
            if local.byte_length() == 0:
                continue
            if not self._pair_present(m):
                raise Error(
                    String("LTX2 apply: A/B pair missing or conv for ")
                    + m.prefix
                )
            if not block.has_weight(local):
                raise Error(
                    String("LTX2 apply: block ")
                    + String(block_idx)
                    + " has no base linear for LoRA key '"
                    + local
                    + "' (base_key=" + m.base_key + ") — fail-closed"
                )
            var scale = self._module_scale(m, multiplier, ctx)
            var base_dtype = block._w(local).dtype()
            var delta = self._compute_delta(m, scale, base_dtype, ctx)
            block.add_delta_to(local, delta^, ctx)
            n_applied += 1
        return n_applied

    def apply_to_globals(
        self,
        mut gw: Dict[String, ArcPointer[Tensor]],
        multiplier: Float32,
        ctx: DeviceContext,
    ) raises -> Int:
        """ADD every GLOBAL (non-block) LoRA delta onto the resident persistent
        weight tensors in `gw`, IN PLACE.  `gw` must map each global base_key
        (e.g. `patchify_proj.weight`, `adaln_single.linear.weight`) to an
        ArcPointer[Tensor].  These are PERSISTENT weights (not FP8-streamed per
        step), so ONE application at load time is correct and consistent with the
        HARD RULE: we add scale*(B@A) to the in-memory weight; the result is
        never written to disk.

        FAIL-CLOSED: every global LoRA mapping MUST have a matching key in `gw`;
        if any global base_key is absent the call raises immediately (this
        includes any future global LoRA key family that is not pre-loaded into
        `gw` — they cannot silently fall through the block `continue`).

        Returns the number of deltas applied."""
        var n_applied = 0
        for ref m in self.mappings:
            if _is_block_key(m.base_key):
                continue          # handled by apply_to_av_block per stream
            # FAIL-CLOSED: global key must be pre-loaded into gw
            if m.base_key not in gw:
                raise Error(
                    String("LTX2 global apply: base_key '")
                    + m.base_key
                    + "' not found in global weight dict — fail-closed"
                )
            if not self._pair_present(m):
                raise Error(
                    String("LTX2 global apply: A/B pair missing or conv for ")
                    + m.prefix
                )
            var scale = self._module_scale(m, multiplier, ctx)
            var base_dtype = gw[m.base_key][].dtype()
            var delta = self._compute_delta(m, scale, base_dtype, ctx)
            var merged = add(gw[m.base_key][], delta, ctx)
            gw[m.base_key] = ArcPointer[Tensor](merged^)
            n_applied += 1
        return n_applied

    def compute_delta_for_base(
        self, base_key: String, multiplier: Float32, base_dtype: STDtype,
        ctx: DeviceContext,
    ) raises -> Tensor:
        """Return scale*(B@A) cast to `base_dtype` for the mapping whose
        `base_key` matches (full path, e.g. `transformer_blocks.0.attn1.to_q.
        weight` or `patchify_proj.weight`). Raises if no such mapping. Used by
        the add-math gate to compare against a host F64 reference."""
        for ref m in self.mappings:
            if m.base_key == base_key:
                var scale = self._module_scale(m, multiplier, ctx)
                return self._compute_delta(m, scale, base_dtype, ctx)
        raise Error(String("compute_delta_for_base: no mapping for ") + base_key)

    def scale_for_base(
        self, base_key: String, multiplier: Float32, ctx: DeviceContext
    ) raises -> Float32:
        for ref m in self.mappings:
            if m.base_key == base_key:
                return self._module_scale(m, multiplier, ctx)
        raise Error(String("scale_for_base: no mapping for ") + base_key)

    def has_base(self, base_key: String) -> Bool:
        for ref m in self.mappings:
            if m.base_key == base_key:
                return True
        return False

    def load_ab_for_base(
        self, base_key: String, ctx: DeviceContext
    ) raises -> Tuple[Tensor, Tensor]:
        """H2D-load the (A [rank,in], B [out,rank]) tensors for `base_key` as
        device Tensors (verbatim dtype). For the add-math host-F64 reference."""
        for ref m in self.mappings:
            if m.base_key == base_key:
                var a = self._load_lora_tensor(m.prefix + self.suffix_a, ctx)
                var b = self._load_lora_tensor(m.prefix + self.suffix_b, ctx)
                return (a^, b^)
        raise Error(String("load_ab_for_base: no mapping for ") + base_key)

    def num_lora_pairs_in_file(self) raises -> Int:
        """Count A/B pairs PRESENT IN THE FILE (header), independent of mapping.
        A pair = a `<prefix>{suffix_a}` whose `<prefix>{suffix_b}` also exists.
        This is the ground-truth denominator for the key-coverage gate."""
        var n = 0
        for ref nm in self.st.names():
            if nm.endswith(self.suffix_a):
                var prefix = _strip_suffix(nm, self.suffix_a)
                if prefix.byte_length() == 0:
                    continue
                if (prefix + self.suffix_b) in self.st.tensors:
                    n += 1
        return n


def _apply_slot(
    base_w: Tensor,
    delta: Tensor,
    slot_kind: Int,
    slot_a: Int,
    slot_b: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    """Add `delta` into `base_w` per the slot kind, returning the merged weight.
    Mirrors the per-slot match in lora_merge.rs:566-638.
      FULL     : base + delta (shapes must match)
      ROWS(n)  : base[..n,:] += delta; rejoin with base[n..,:]
      COLS(n)  : base[:,..n] += delta; rejoin with base[:,n..]
      ROWRANGE : base[start..start+len,:] += delta; rejoin head|mid|tail."""
    var bdims = base_w.shape()

    if slot_kind == SLOT_FULL:
        var ddims = delta.shape()
        if len(bdims) != len(ddims):
            raise Error("lora Full: rank mismatch base vs delta")
        for i in range(len(bdims)):
            if bdims[i] != ddims[i]:
                raise Error("lora Full: shape mismatch base vs delta")
        return add(base_w, delta, ctx)

    if slot_kind == SLOT_ROWS:
        var n = slot_a
        if len(bdims) != 2 or bdims[0] < n:
            raise Error("lora Rows: base must be 2D with >= n rows")
        var top = slice(base_w, 0, 0, n, ctx)
        var bottom = slice(base_w, 0, n, bdims[0] - n, ctx)
        var top_merged = add(top, delta, ctx)
        return concat(0, ctx, top_merged, bottom)

    if slot_kind == SLOT_COLS:
        var n = slot_a
        if len(bdims) != 2 or bdims[1] < n:
            raise Error("lora Cols: base must be 2D with >= n cols")
        var left = slice(base_w, 1, 0, n, ctx)
        var right = slice(base_w, 1, n, bdims[1] - n, ctx)
        var left_merged = add(left, delta, ctx)
        return concat(1, ctx, left_merged, right)

    # SLOT_ROWRANGE
    var start = slot_a
    var length = slot_b
    if len(bdims) != 2 or start + length > bdims[0]:
        raise Error("lora RowRange: base must be 2D and range in bounds")
    var head_len = start
    var tail_len = bdims[0] - start - length
    var mid = slice(base_w, 0, start, length, ctx)
    var mid_merged = add(mid, delta, ctx)
    if head_len > 0 and tail_len > 0:
        var head = slice(base_w, 0, 0, head_len, ctx)
        var tail = slice(base_w, 0, start + length, tail_len, ctx)
        return concat(0, ctx, head, mid_merged, tail)
    elif head_len > 0:
        var head = slice(base_w, 0, 0, head_len, ctx)
        return concat(0, ctx, head, mid_merged)
    elif tail_len > 0:
        var tail = slice(base_w, 0, start + length, tail_len, ctx)
        return concat(0, ctx, mid_merged, tail)
    else:
        return mid_merged^

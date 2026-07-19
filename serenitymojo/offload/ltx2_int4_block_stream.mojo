# offload/ltx2_int4_block_stream.mojo — INT4 (SVDQuant class-A) DiT block streamer
# for LTX-2.
#
# Sibling of offload/ltx2_block_stream.mojo (the FP8 streamer). Same BlockLoader
# discipline (load → use → drop), same FP8Block return type, same prefix/block
# discovery — the ONLY difference is the on-use decode: where the FP8 stream
# dequants F8_E4M3 → BF16 with a per-tensor scale, this stream RECONSTRUCTS each
# quantized class-A linear's dense BF16 weight from its INT4 slab bundle
# (qweight/wscales/lora_down/lora_up/smooth/bias) via
# ops/svdquant.svdquant_reconstruct_weight (dequant4 + low-rank). Everything the
# quantizer left verbatim (norms, scale_shift_table, biases of quantized linears,
# any non-quantized weight) passes through as BF16.
#
# The returned FP8Block is prefix-stripped canonical keys → BF16 device Tensors,
# EXACTLY the layout LTX2BlockWeights.from_fp8_block already consumes, so the
# existing block forward runs UNCHANGED on top of an INT4-resident checkpoint.
#
# ── INT4 slab layout (per quantized class-A linear `<base>`) ──────────────────
#   <base>.qweight   I8   [out, in/2]   nibble-packed int4 (see ops/svdquant)
#   <base>.wscales   BF16 [in/64, out]  per-group × per-output scale
#   <base>.lora_down BF16 [in, rank]
#   <base>.lora_up   BF16 [out, rank]
#   <base>.smooth    BF16 [in]          activation smoothing (unused by dense
#                                       reconstruct; see svdquant_reconstruct_weight)
#   <base>.bias      BF16 [out]
# Everything else under the block prefix is verbatim BF16 (norms, biases of
# non-quantized layers, scale_shift_table, gate_logits, ...). Keys:
#   model.diffusion_model.transformer_blocks.<N>.<sub>
#
# Mojo current beta, NVIDIA GPU.

from std.memory import ArcPointer
from std.gpu.host import DeviceContext
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor
from serenitymojo.offload.ltx2_block_stream import FP8Block
from serenitymojo.ops.svdquant import (
    SvdquantLinearA,
    svdquant_reconstruct_weight,
)


def _substr(s: String, start: Int, end: Int) -> String:
    """Codepoint-indexed substring [start, end). Mojo-beta-safe (string slicing
    via `[a:b]` is unavailable in the current dialect). Mirrors
    ltx2_block_stream._substr."""
    var out = String("")
    var i = 0
    for ch in s.codepoint_slices():
        if i >= start and i < end:
            out += String(ch)
        i += 1
    return out^


def _first_dot(s: String) -> Int:
    """Index of the first '.' in `s` (codepoint index), or -1 if absent. Mirrors
    ltx2_block_stream._first_dot."""
    var i = 0
    for ch in s.codepoint_slices():
        if String(ch) == ".":
            return i
        i += 1
    return -1


struct LTX2Int4BlockStream(Movable):
    """Streams LTX-2 DiT transformer blocks one at a time from an INT4 slab.

    Holds the mmap-backed ShardedSafeTensors and the ComfyUI key prefix
    (`model.diffusion_model.transformer_blocks.`). `load_block_bf16(i, ctx)`
    returns one block as BF16 device Tensors — quantized class-A linears
    reconstructed on-use to dense BF16 (svdquant_reconstruct_weight), everything
    else copied verbatim/upcast to BF16.

    The returned FP8Block owns its VRAM; drop it (or let it fall out of scope) to
    evict before loading the next block — the single-resident window that keeps
    peak VRAM bounded. Mirror of LTX2BlockStream, INT4 decode instead of FP8."""

    var sharded: ShardedSafeTensors
    var prefix: String
    var n_blocks: Int

    @staticmethod
    def open(slab_path: String) raises -> LTX2Int4BlockStream:
        var st = ShardedSafeTensors.open(slab_path)
        # Detect prefix + block count from the names (mirror LTX2BlockStream).
        var pfx_comfy = String("model.diffusion_model.transformer_blocks.")
        var pfx_diff = String("transformer_blocks.")
        var prefix = pfx_comfy
        var found = False
        for ref nm in st.names():
            if nm.startswith(pfx_comfy):
                prefix = pfx_comfy
                found = True
                break
            if nm.startswith(pfx_diff):
                prefix = pfx_diff
                found = True
                break
        if not found:
            raise Error(
                "LTX2Int4BlockStream: no transformer_blocks.* keys found"
            )
        # Count blocks: max index + 1 over keys matching prefix.{N}.
        var max_idx = -1
        for ref nm in st.names():
            if nm.startswith(prefix):
                var rest = _substr(nm, len(prefix), len(nm))
                var dot = _first_dot(rest)
                if dot > 0:
                    try:
                        var idx = Int(_substr(rest, 0, dot))
                        if idx > max_idx:
                            max_idx = idx
                    except:
                        pass
        if max_idx < 0:
            raise Error(
                "LTX2Int4BlockStream: prefix found but no numbered blocks"
            )
        return LTX2Int4BlockStream(st^, prefix, max_idx + 1)

    def __init__(
        out self,
        var sharded: ShardedSafeTensors,
        prefix: String,
        n_blocks: Int,
    ):
        self.sharded = sharded^
        self.prefix = prefix
        self.n_blocks = n_blocks

    def block_count(self) -> Int:
        return self.n_blocks

    def load_block_bf16(
        self, block_idx: Int, ctx: DeviceContext
    ) raises -> FP8Block:
        """Load every tensor under `prefix{block_idx}.` to the GPU as BF16.

        - A quantized class-A linear is detected by the presence of `<base>.qweight`.
          Its 6 slab tensors are loaded (qweight raw int4, the other five BF16),
          bundled into an SvdquantLinearA, and reconstructed to a dense BF16 W
          [out, in] via svdquant_reconstruct_weight (dequant4 + low-rank). That W
          is placed under the prefix-stripped `<base>.weight`; `<base>.bias`
          (verbatim BF16) is placed alongside.
        - The int4 companion tensors (wscales/lora_down/lora_up/smooth) are
          consumed by the reconstruct and NOT placed in the block.
        - Every other tensor (norms, scale_shift_table, gate_logits, and the
          bias/weight of any NON-quantized layer) passes through as BF16 under its
          prefix-stripped name.

        Keys are stripped of the block prefix so the block dict is keyed by the
        canonical sub-name (e.g. `attn1.to_q.weight`) — the exact contract
        LTX2BlockWeights.from_fp8_block expects. Missing/mis-shaped tensors raise
        loud (svdquant_reconstruct_weight validates shapes; from_fp8_block
        validates the required key set)."""
        var block = FP8Block()
        var bp = self.prefix + String(block_idx) + "."
        for ref nm in self.sharded.names():
            if not nm.startswith(bp):
                continue
            var canon = _substr(nm, len(bp), len(nm))

            if canon.endswith(".qweight"):
                # ── Quantized class-A linear: reconstruct dense BF16 W. ──
                var qsuf = len(String(".qweight"))
                var base = _substr(canon, 0, len(canon) - qsuf)  # prefix-stripped
                var full = _substr(nm, 0, len(nm) - qsuf)        # full slab base
                var qtv = self.sharded.tensor_view(full + ".qweight")
                var qweight = Tensor.from_view_raw(qtv, ctx)     # I8 [out, in/2]
                var wtv = self.sharded.tensor_view(full + ".wscales")
                var wscales = Tensor.from_view(wtv, ctx)         # BF16 [in/64, out]
                var ldtv = self.sharded.tensor_view(full + ".lora_down")
                var lora_down = Tensor.from_view(ldtv, ctx)      # BF16 [in, rank]
                var lutv = self.sharded.tensor_view(full + ".lora_up")
                var lora_up = Tensor.from_view(lutv, ctx)        # BF16 [out, rank]
                var smtv = self.sharded.tensor_view(full + ".smooth")
                var smooth = Tensor.from_view(smtv, ctx)         # BF16 [in]
                var btv = self.sharded.tensor_view(full + ".bias")
                var bias = Tensor.from_view(btv, ctx)            # BF16 [out]

                var in_f = qweight.shape()[1] * 2
                var out_f = qweight.shape()[0]
                var rank = lora_down.shape()[1]

                # Emit bias verbatim; clone because the struct consumes `bias`.
                block[base + ".bias"] = ArcPointer(bias.clone(ctx))

                var w = SvdquantLinearA(
                    qweight^, wscales^, lora_down^, lora_up^, smooth^, bias^,
                    in_f, out_f, rank,
                )
                var W = svdquant_reconstruct_weight(w, ctx)      # BF16 [out, in]
                block[base + ".weight"] = ArcPointer(W^)

            elif (
                canon.endswith(".wscales")
                or canon.endswith(".lora_down")
                or canon.endswith(".lora_up")
                or canon.endswith(".smooth")
            ):
                # int4 companion — consumed by the reconstruct above.
                continue

            elif canon.endswith(".bias"):
                # A quantized class-A linear's bias is emitted by the .qweight
                # branch; skip it here to avoid a duplicate. All other biases
                # (non-quantized layers) pass through.
                var bsuf = len(String(".bias"))
                var bbase = _substr(nm, 0, len(nm) - bsuf)
                if (bbase + ".qweight") in self.sharded.name_to_shard:
                    continue
                var tv = self.sharded.tensor_view(nm)
                block[canon] = ArcPointer(Tensor.from_view_as_bf16(tv, ctx))

            else:
                # Verbatim BF16 passthrough: norms, scale_shift_table,
                # gate_logits / weights of any non-quantized layer.
                var tv = self.sharded.tensor_view(nm)
                block[canon] = ArcPointer(Tensor.from_view_as_bf16(tv, ctx))

        ctx.synchronize()
        return block^

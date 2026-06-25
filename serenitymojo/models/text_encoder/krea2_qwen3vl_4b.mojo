# models/text_encoder/krea2_qwen3vl_4b.mojo — Krea-2 Qwen3-VL-4B TEXT path (LM only).
#
# Krea-2's text encoder is Qwen3-VL-4B-Instruct (text-only; the vision tower is
# dropped — ai-toolkit krea2.py:213-216 sets `text_encoder.model.visual = None`).
# Krea-2 conditions on a STACK of 12 selected hidden-state layers, NOT a single
# layer, then drops the leading system-prompt rows. Reference:
#   ai-toolkit/extensions_built_in/diffusion_models/krea2/src/text_encoder.py
#     SELECT_LAYERS = (2,5,8,11,14,17,20,23,26,29,32,35)          # HF-indexed
#     PROMPT_TEMPLATE_ENCODE_START_IDX = 34                        # prefix drop
#     hiddens = stack([states.hidden_states[i] for i in SELECT_LAYERS], dim=2)
#     hiddens = hiddens[:, prefix_idx:]                            # drop prefix
#     return hiddens[0]                                            # (L', 12, 2560)
#
# Reuses Qwen3Encoder (the Qwen3-VL-4B text decoder layer is byte-identical to the
# Qwen3 text encoder Z-Image/Klein use: input_layernorm / q,k,v,o_proj + q_norm,
# k_norm / post_attention_layernorm / mlp.gate,up,down SwiGLU). Deltas vs the
# Boogu loader (boogu_qwen3vl.mojo, the template):
#   - config hidden = 2560 (NOT 4096): Qwen3-VL-4B text_config.hidden_size,
#   - output = a 12-LAYER STACK [1, L', 12, 2560] (not the single last layer),
#   - the leading 34 system-prefix rows are sliced off the L axis.
# Keys are `model.language_model.*` (identical to Boogu's prefix) -> remapped to
# the `model.*` keys Qwen3Encoder wants; dtype is BF16 (NOT fp8).
#
# Config truth (Qwen3-VL-4B-Instruct config.json text_config):
#   hidden 2560, intermediate 9728, layers 36, heads 32, kv_heads 8, head_dim 128,
#   eps 1e-6, vocab 151936, rope_theta 5e6, bf16. (intermediate is implicit in the
#   mlp.{gate,up,down}_proj weight shapes — Qwen3Config carries no intermediate.)
#
# LAYER MAP (HF hidden_states is [embeddings, layer0_out, ..., layer35_out], so
#   hidden_states[i] = output AFTER decoder layer (i-1)). Qwen3Encoder's
#   encode_layer_states returns [layer0_out, ..., layer35_out] (index i = layer i
#   output, NO embedding entry), so the Mojo index for HF SELECT_LAYER s is s-1:
#     HF  (2, 5, 8,11,14,17,20,23,26,29,32,35)
#     Mojo(1, 4, 7,10,13,16,19,22,25,28,31,34)
#   (Cross-checked vs Qwen3Encoder.encode_klein's documented HF[9,18,27]<->Mojo
#   [8,17,26] mapping, and vs the Boogu C7 gate: HF[-1]=hidden_states[36]=Mojo[35]
#   =num_layers-1, which passed cos>=0.999.)
#
# FLAG (do not mask): text_config.rope_scaling = {mrope_interleaved: true,
#   mrope_section: [24,20,20]}. The Qwen3-VL text RoPE is mROPE-interleaved, NOT
#   the plain half-split RoPE Qwen3Encoder implements. For TEXT-ONLY inputs the
#   three mrope sections share one 1D position sequence and collapse to identical
#   per-position angles, so plain RoPE is the standard text-only reduction — this
#   is the same reduction the Boogu C7 Qwen3-VL gate validated at cos>=0.999.
#   Surfaced here, not silently absorbed.
#
# Mojo 1.0.0b1, NVIDIA GPU. BF16 storage, F32 accumulation in foundation ops.
#
# VRAM (MEASURED 2026-06-24): the naive Tensor.from_view loader allocated a fresh
# PINNED host staging buffer per tensor (398 tensors, 8.0 GB) and let them pool
# alongside the 8.0 GB of resident device weights — external nvidia-smi peaked at
# 22.0 GB (cuMemGetInfo only saw the 7.6 GB device pool; the +14 GB pinned-host +
# CUDA/cuBLAS context is invisible to it but real, and OOMs a busy 24 GB card).
# load_krea2_qwen3vl_4b streams every tensor through ONE reusable pinned host
# staging buffer sized to the largest tensor (embed_tokens, ~778 MB), capping the
# pinned-host peak at one tensor instead of all 398 -> measured peak ~9.6 GB.
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.ffi import BytePtr, sys_memcpy
from serenitymojo.ops.tensor_algebra import slice, concat, reshape
from serenitymojo.models.text_encoder.qwen3_encoder import Qwen3Encoder, Qwen3Config


# Krea-2 prefix drop: PROMPT_TEMPLATE_ENCODE_START_IDX (text_encoder.py:34). The
# leading 34 system-prompt token rows are sliced off the L axis after stacking.
comptime KREA2_DROP_IDX = 34

# Qwen3 pad/bos id (== 151643). Krea-2 prompt tokens use 151644/151645, so padding
# with 151643 lets encode_layer_states' pad auto-detect recover real_len.
comptime _KREA2_PAD_ID = 151643


def _krea2_select_layers_mojo() -> List[Int]:
    """HF SELECT_LAYERS (2,5,8,11,14,17,20,23,26,29,32,35) mapped to Qwen3Encoder
    encode_layer_states indices (HF index - 1, since that list omits the embedding
    entry that HF's hidden_states[0] holds)."""
    var out = List[Int]()
    out.append(1)
    out.append(4)
    out.append(7)
    out.append(10)
    out.append(13)
    out.append(16)
    out.append(19)
    out.append(22)
    out.append(25)
    out.append(28)
    out.append(31)
    out.append(34)
    return out^


# One reusable pinned host staging buffer is sized to this ceiling (the largest
# tensor is embed_tokens = 151936*2560*2 B = 778 MB; 896 MB leaves margin). All
# 398 tensors stream through sub-buffers of this single allocation.
comptime _KREA2_STAGE_BYTES = 896 * 1024 * 1024


def _add_streamed(
    st: ShardedSafeTensors,
    stage: HostBuffer[DType.uint8],
    mut weights: List[ArcPointer[Tensor]],
    mut n2i: Dict[String, Int],
    dst: String, src: String, ctx: DeviceContext,
) raises:
    """Load one BF16 tensor `src` (raw H2D, dtype preserved) THROUGH the shared
    pinned `stage` buffer, and register it under the remapped name `dst` that
    Qwen3Encoder._layer/_embed/_w request. memcpy mmap pages -> stage[0:nbytes],
    then enqueue_copy stage_sub -> a fresh persistent device buffer. The stage is
    reused for every tensor, so the pinned-host peak is one tensor, not all 398."""
    var tv = st.tensor_view(src)
    var nbytes = tv.nbytes()
    if nbytes > _KREA2_STAGE_BYTES:
        raise Error(
            String("krea2 load: tensor ") + src + " is " + String(nbytes)
            + " B > stage ceiling " + String(_KREA2_STAGE_BYTES)
            + " B (raise _KREA2_STAGE_BYTES)"
        )
    # mmap host pages -> shared pinned stage (host->host, pageable->pinned).
    var hdst = BytePtr(unsafe_from_address=Int(stage.unsafe_ptr()))
    var hsrc = BytePtr(unsafe_from_address=Int(tv.data.unsafe_ptr()))
    _ = sys_memcpy(hdst, hsrc, nbytes)
    # H2D into a fresh persistent device buffer (the resident weight).
    var dev = ctx.enqueue_create_buffer[DType.uint8](nbytes)
    var ssub = stage.create_sub_buffer[DType.uint8](0, nbytes)
    ctx.enqueue_copy(dst_buf=dev, src_buf=ssub)
    ctx.synchronize()  # stage is reused next call, so the copy must complete first
    var t = Tensor(dev^, tv.shape.copy(), tv.dtype)
    n2i[dst] = len(weights)
    weights.append(ArcPointer(t^))


def load_krea2_qwen3vl_4b(te_dir: String, ctx: DeviceContext) raises -> Qwen3Encoder:
    """Load the Qwen3-VL-4B-Instruct language-model (text) stack into a
    Qwen3Encoder. Remaps `model.language_model.*` -> `model.*` and loads BF16. The
    VISION tower (`model.visual.*`) is NOT loaded (text-only; lm_head is tied and
    unused for conditioning). `te_dir` is the HF snapshot dir holding the sharded
    safetensors + index.json.

    All 398 tensors stream through ONE reusable pinned host staging buffer (see
    _add_streamed) so the pinned-host VRAM footprint is one tensor (~778 MB) rather
    than the full 8.0 GB — measured external peak ~9.6 GB (was 22.0 GB)."""
    var st = ShardedSafeTensors.open(te_dir)
    var stage = ctx.enqueue_create_host_buffer[DType.uint8](_KREA2_STAGE_BYTES)
    ctx.synchronize()
    var weights = List[ArcPointer[Tensor]]()
    var n2i = Dict[String, Int]()
    # Qwen3-VL-4B text_config: hidden 2560, layers 36, heads 32, kv 8, dh 128,
    # eps 1e-6, theta 5e6.
    var cfg = Qwen3Config(2560, 36, 32, 8, 128, Float32(1.0e-6), Float64(5000000.0))
    _add_streamed(st, stage, weights, n2i,
         "model.embed_tokens.weight", "model.language_model.embed_tokens.weight", ctx)
    for i in range(36):
        var ps = String("model.language_model.layers.") + String(i) + "."
        var pd = String("model.layers.") + String(i) + "."
        _add_streamed(st, stage, weights, n2i, pd + "input_layernorm.weight", ps + "input_layernorm.weight", ctx)
        _add_streamed(st, stage, weights, n2i, pd + "self_attn.q_proj.weight", ps + "self_attn.q_proj.weight", ctx)
        _add_streamed(st, stage, weights, n2i, pd + "self_attn.k_proj.weight", ps + "self_attn.k_proj.weight", ctx)
        _add_streamed(st, stage, weights, n2i, pd + "self_attn.v_proj.weight", ps + "self_attn.v_proj.weight", ctx)
        _add_streamed(st, stage, weights, n2i, pd + "self_attn.o_proj.weight", ps + "self_attn.o_proj.weight", ctx)
        _add_streamed(st, stage, weights, n2i, pd + "self_attn.q_norm.weight", ps + "self_attn.q_norm.weight", ctx)
        _add_streamed(st, stage, weights, n2i, pd + "self_attn.k_norm.weight", ps + "self_attn.k_norm.weight", ctx)
        _add_streamed(st, stage, weights, n2i, pd + "post_attention_layernorm.weight", ps + "post_attention_layernorm.weight", ctx)
        _add_streamed(st, stage, weights, n2i, pd + "mlp.gate_proj.weight", ps + "mlp.gate_proj.weight", ctx)
        _add_streamed(st, stage, weights, n2i, pd + "mlp.up_proj.weight", ps + "mlp.up_proj.weight", ctx)
        _add_streamed(st, stage, weights, n2i, pd + "mlp.down_proj.weight", ps + "mlp.down_proj.weight", ctx)
    _add_streamed(st, stage, weights, n2i, "model.norm.weight", "model.language_model.norm.weight", ctx)
    # `stage` (the 896 MB pinned buffer) drops here, before encode runs.
    return Qwen3Encoder(weights^, n2i^, cfg)


def _pad_ids(ids: List[Int]) raises -> List[Int]:
    """Pad the token list UP to the smallest comptime-supported SDPA seq
    (64/128/256/512/1024/2048) with _KREA2_PAD_ID, so encode_layer_states' pad
    auto-detect (first 151643) recovers real_len = the original L. The encoder is
    causal so pad columns are masked out — numerically identical to the unpadded L."""
    var L = len(ids)
    var pad: Int
    if L <= 64:
        pad = 64
    elif L <= 128:
        pad = 128
    elif L <= 256:
        pad = 256
    elif L <= 512:
        pad = 512
    elif L <= 1024:
        pad = 1024
    elif L <= 2048:
        pad = 2048
    else:
        raise Error(
            String("krea2 encode: L=") + String(L)
            + " exceeds 2048 (max supported encoder SDPA seq)"
        )
    var out = List[Int]()
    for i in range(L):
        out.append(ids[i])
    for _i in range(pad - L):
        out.append(_KREA2_PAD_ID)
    return out^


def encode_krea2_stack(
    enc: Qwen3Encoder, ids: List[Int], ctx: DeviceContext
) raises -> Tensor:
    """Krea-2 conditioning: the 12-layer hidden-state stack with the leading
    34 system-prefix rows dropped -> [1, L-34, 12, 2560].

    `ids` are the EXACT token ids of (system-prefix + prompt + assistant-suffix),
    i.e. the full sequence encode_krea_prompt feeds the model (text_encoder.py
    builds input_ids = cat[tokenizer(prefix+prompt), suffix_ids]). The stack pulls
    Qwen3Encoder layer outputs [1,4,...,34] (HF SELECT_LAYERS - 1), stacks them on
    a new layer axis, then drops rows [0,34) of the L axis (the system prefix).
    States are PRE-final-norm — matching HF's intermediate hidden_states (only the
    last entry, never selected here, gets model.norm)."""
    var L = len(ids)
    if L <= KREA2_DROP_IDX:
        raise Error(
            String("krea2 encode: L=") + String(L)
            + " <= DROP_IDX=" + String(KREA2_DROP_IDX)
            + " (prompt produced no post-prefix tokens)"
        )
    var padded = _pad_ids(ids)
    # All 36 layer states, each [1, L_pad, 2560], PRE-final-norm.
    var states = enc.encode_layer_states(padded, ctx)
    var sel = _krea2_select_layers_mojo()
    var H = enc.config.hidden_size

    # Stack the 12 selected layers on a new axis-2: each [1,L_pad,2560] is
    # reshaped to [1,L_pad,1,2560] and concatenated on dim 2 -> [1,L_pad,12,2560].
    # Then slice the L axis (dim 1) to [DROP_IDX, L) -> drop the system prefix AND
    # the SDPA padding in one narrow.
    var keep = L - KREA2_DROP_IDX
    var s4 = [1, len(padded), 1, H]
    var r0 = reshape(states[sel[0]][], s4.copy(), ctx)
    var r1 = reshape(states[sel[1]][], s4.copy(), ctx)
    var r2 = reshape(states[sel[2]][], s4.copy(), ctx)
    var r3 = reshape(states[sel[3]][], s4.copy(), ctx)
    var r4 = reshape(states[sel[4]][], s4.copy(), ctx)
    var r5 = reshape(states[sel[5]][], s4.copy(), ctx)
    var r6 = reshape(states[sel[6]][], s4.copy(), ctx)
    var r7 = reshape(states[sel[7]][], s4.copy(), ctx)
    var r8 = reshape(states[sel[8]][], s4.copy(), ctx)
    var r9 = reshape(states[sel[9]][], s4.copy(), ctx)
    var r10 = reshape(states[sel[10]][], s4.copy(), ctx)
    var r11 = reshape(states[sel[11]][], s4.copy(), ctx)
    var stacked = concat(2, ctx, r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11)
    # [1, L_pad, 12, 2560] -> drop prefix rows + padding: [1, keep, 12, 2560].
    return slice(stacked, 1, KREA2_DROP_IDX, keep, ctx)

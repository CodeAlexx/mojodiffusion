# models/nldiffusion/backbone_stack.mojo — CHUNK B: full 34-layer backbone.
#
# Pure-Mojo + MAX (GPU-only, INFERENCE) port of the NL-Diffusion-Image
# (Ministral3 diffusion-LM) backbone: the Mojo-native YaRN rope tables + the
# 34-layer decoder stack + the final `encoder.norm`. Reuses the CHUNK-A-proven
# `nldiff_decoder_layer` (cos 0.99999) unchanged.
#
#   serenitymojo/models/nldiffusion/PORT_SPEC.md  — layer math, gates
#   serenitymojo/models/nldiffusion/backbone.mojo — CHUNK A (reused here)
#
# Deltas vs CHUNK A: this file adds
#   1. `nldiff_yarn_inv_freq` — HF `_compute_yarn_parameters`(yarn) EXACT, 64 vals.
#   2. `nldiff_build_yarn_rope[S]` — full emb cos/sin [1,S,128] (gate vs oracle).
#   3. `nldiff_yarn_perhead[S]` — half-split per-head tables [S*heads, 64].
#   4. `nldiff_backbone[S]` — 34-layer loop that STREAMS weights per layer
#      (never resident all-at-once: the backbone is ~7.4B bf16 ~= 15GB) + final
#      RMSNorm `encoder.norm.weight`.
#
# YaRN params (config.json rope_parameters): rope_theta 1e6, factor 16,
# beta_fast 32, beta_slow 1, original_max_position_embeddings 16384,
# mscale 1.0, mscale_all_dim 1.0 -> attention_scaling = 1.0 (measured). truncate
# defaults True. head_dim 128 -> half 64.

from max.gpu.host import (
    DeviceContext, HostBuffer, DeviceBuffer, DeviceStream, DeviceEvent,
)
from std.math import cos as fcos, sin as fsin, log as flog
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.norm import rms_norm
from serenitymojo.models.nldiffusion.backbone import (
    NLDiffConfig,
    NLDiffLayerWeights,
    nldiff_decoder_layer,
)
from serenitymojo.models.text_encoder.qwen3_encoder import _clone
# Copy-stream H2D DMA (cuMemcpyHtoDAsync on an explicit stream) — the proven
# turbo_loader / krea2 double-buffer prefetch primitive (overlap next-layer
# H2D with current-layer compute).
from serenitymojo.offload.turbo_loader import _h2d_dma_copy
from serenitymojo.offload.vmm_cuda import cu_mempool_trim_current

# ── YaRN config constants (config.json rope_parameters) ───────────────────────
comptime YARN_THETA = Float64(1.0e6)
comptime YARN_FACTOR = Float64(16.0)
comptime YARN_BETA_FAST = Float64(32.0)
comptime YARN_BETA_SLOW = Float64(1.0)
comptime YARN_ORIG_MAX = Float64(16384.0)
comptime YARN_HEAD_DIM = 128
comptime YARN_HALF = 64  # head_dim // 2
comptime YARN_ATTN_SCALING = Float32(1.0)  # measured; mscale/mscale_all_dim -> 1
comptime TWO_PI = Float64(6.283185307179586476925)


def _floor_pos(x: Float64) -> Float64:
    """floor for x >= 0 (Int() truncates toward zero == floor for non-negative)."""
    return Float64(Int(x))


def _ceil_pos(x: Float64) -> Float64:
    """ceil for x >= 0."""
    var t = Int(x)
    if Float64(t) < x:
        t += 1
    return Float64(t)


def nldiff_yarn_inv_freq() raises -> List[Float32]:
    """HF `_compute_yarn_parameters` (rope_type='yarn'), EXACT, returns inv_freq[64].

    Reproduces transformers.modeling_rope_utils._compute_yarn_parameters:
        pos_freqs[j]              = base ** (2j / dim)          (j = 0..63)
        inv_freq_extrapolation[j] = 1 / pos_freqs[j]
        inv_freq_interpolation[j] = 1 / (factor * pos_freqs[j])
        low  = floor(find_correction_dim(beta_fast))           (truncate=True)
        high = ceil (find_correction_dim(beta_slow))
        low, high = max(low,0), min(high, dim-1)
        ramp[j]  = clamp((j - low) / (high - low), 0, 1)
        ext[j]   = 1 - ramp[j]
        inv_freq[j] = interp[j]*(1-ext[j]) + extrap[j]*ext[j]
    find_correction_dim(nr) = (dim*ln(orig/(nr*2*pi))) / (2*ln(base)).
    """
    var dim = Float64(YARN_HEAD_DIM)
    var base = YARN_THETA
    var ln_base = flog(base)

    # correction range (truncate=True: floor/ceil).
    var fcd_fast = (dim * flog(YARN_ORIG_MAX / (YARN_BETA_FAST * TWO_PI))) / (
        Float64(2.0) * ln_base
    )
    var fcd_slow = (dim * flog(YARN_ORIG_MAX / (YARN_BETA_SLOW * TWO_PI))) / (
        Float64(2.0) * ln_base
    )
    var low = _floor_pos(fcd_fast)
    var high = _ceil_pos(fcd_slow)
    if low < 0.0:
        low = 0.0
    if high > dim - 1.0:
        high = dim - 1.0
    var span = high - low
    if span == 0.0:
        span = 0.001  # linear_ramp_factor singularity guard

    var inv = List[Float32]()
    for j in range(YARN_HALF):
        var pos_freq = base ** (Float64(2 * j) / dim)
        var extrap = Float64(1.0) / pos_freq
        var interp = Float64(1.0) / (YARN_FACTOR * pos_freq)
        var ramp = (Float64(j) - low) / span
        if ramp < 0.0:
            ramp = 0.0
        elif ramp > 1.0:
            ramp = 1.0
        var ext = Float64(1.0) - ramp  # inv_freq_extrapolation_factor
        var val = interp * (Float64(1.0) - ext) + extrap * ext
        inv.append(Float32(val))
    return inv^


def nldiff_build_yarn_rope[
    S: Int
](ctx: DeviceContext) raises -> Tuple[Tensor, Tensor]:
    """Full emb cos/sin tables [1, S, 128] on GPU (F32), matching the reference
    `Ministral3RotaryEmbedding.forward`:
        freqs[pos, i] = pos * inv_freq[i]      (i = 0..63)
        emb           = cat(freqs, freqs)      (128 wide)
        cos = emb.cos() * attention_scaling ;  sin = emb.sin() * attention_scaling
    Used to GATE the Mojo YaRN build against `rope_oracle.safetensors`.
    """
    var inv = nldiff_yarn_inv_freq()
    var cos_vals = List[Float32]()
    var sin_vals = List[Float32]()
    for pos in range(S):
        # first-half columns (freqs)
        var c_first = List[Float32]()
        var s_first = List[Float32]()
        for i in range(YARN_HALF):
            var angle = Float32(pos) * inv[i]
            c_first.append(fcos(angle) * YARN_ATTN_SCALING)
            s_first.append(fsin(angle) * YARN_ATTN_SCALING)
        # emb = cat(freqs, freqs): cols 0..63 then an identical copy 64..127.
        for i in range(YARN_HALF):
            cos_vals.append(c_first[i])
            sin_vals.append(s_first[i])
        for i in range(YARN_HALF):
            cos_vals.append(c_first[i])
            sin_vals.append(s_first[i])
    var sh = [1, S, YARN_HEAD_DIM]
    var cos_t = Tensor.from_host(cos_vals, sh.copy(), STDtype.F32, ctx)
    var sin_t = Tensor.from_host(sin_vals, sh.copy(), STDtype.F32, ctx)
    return (cos_t^, sin_t^)


def nldiff_yarn_inv_freq_tensor(ctx: DeviceContext) raises -> Tensor:
    """inv_freq[64] as a GPU F32 tensor (for the optional inv_freq gate)."""
    var inv = nldiff_yarn_inv_freq()
    var sh = [YARN_HALF]
    return Tensor.from_host(inv, sh.copy(), STDtype.F32, ctx)


def nldiff_yarn_perhead[
    S: Int
](heads: Int, ctx: DeviceContext) raises -> Tuple[Tensor, Tensor]:
    """Half-split per-head rope tables [S*heads, 64] on GPU (F32) as required by
    `ops.rope.rope_halfsplit`: row = pos*heads + h, col i in [0,64) holds
    cos/sin(pos * inv_freq[i]) (angle depends only on position, repeated across
    heads). This is the first-half of the emb; identical to the CHUNK-A probe's
    `_build_rope_perhead` but generated natively from the Mojo inv_freq.
    """
    var inv = nldiff_yarn_inv_freq()
    var cos_vals = List[Float32]()
    var sin_vals = List[Float32]()
    for pos in range(S):
        for _h in range(heads):
            for i in range(YARN_HALF):
                var angle = Float32(pos) * inv[i]
                cos_vals.append(fcos(angle) * YARN_ATTN_SCALING)
                sin_vals.append(fsin(angle) * YARN_ATTN_SCALING)
    var sh = [S * heads, YARN_HALF]
    var cos_t = Tensor.from_host(cos_vals, sh.copy(), STDtype.F32, ctx)
    var sin_t = Tensor.from_host(sin_vals, sh.copy(), STDtype.F32, ctx)
    return (cos_t^, sin_t^)


def _load_bf16(
    st: ShardedSafeTensors, name: String, ctx: DeviceContext
) raises -> ArcPointer[Tensor]:
    var tv = st.tensor_view(name)
    var t = Tensor.from_view(tv, ctx)
    return ArcPointer(t^)


def _load_layer_weights(
    model: ShardedSafeTensors, layer_idx: Int, ctx: DeviceContext
) raises -> NLDiffLayerWeights:
    """Load the 9 bf16 weights of `encoder.layers.{i}` onto the GPU. The returned
    struct owns them; dropping it frees the GPU buffers (per-layer streaming)."""
    var p = String("encoder.layers.") + String(layer_idx) + String(".")
    return NLDiffLayerWeights(
        _load_bf16(model, p + "input_layernorm.weight", ctx),
        _load_bf16(model, p + "post_attention_layernorm.weight", ctx),
        _load_bf16(model, p + "self_attn.q_proj.weight", ctx),
        _load_bf16(model, p + "self_attn.k_proj.weight", ctx),
        _load_bf16(model, p + "self_attn.v_proj.weight", ctx),
        _load_bf16(model, p + "self_attn.o_proj.weight", ctx),
        _load_bf16(model, p + "mlp.gate_proj.weight", ctx),
        _load_bf16(model, p + "mlp.up_proj.weight", ctx),
        _load_bf16(model, p + "mlp.down_proj.weight", ctx),
    )


def nldiff_backbone[
    S: Int
](
    x0: Tensor,
    model: ShardedSafeTensors,
    cfg: NLDiffConfig,
    cos_q: Tensor,
    sin_q: Tensor,
    cos_k: Tensor,
    sin_k: Tensor,
    ctx: DeviceContext,
) raises -> Tensor:
    """Full NL-Diffusion-Image backbone: 34 Ministral3 decoder layers + final
    `encoder.norm` RMSNorm. `x0` is the post-embedding stack input [1, S, 4096]
    (bf16). Weights are STREAMED per layer from `model` — only one layer's
    ~435MB of bf16 weights is GPU-resident at a time (the full stack is ~15GB).
    cos/sin are the shared half-split YaRN tables from `nldiff_yarn_perhead`.
    """
    var hidden = _clone(x0, ctx)
    for i in range(cfg.num_layers):
        # Load this layer's weights; run; `w` (and its GPU buffers) drop at the
        # end of the iteration when it is re-bound next pass / goes out of scope.
        var w = _load_layer_weights(model, i, ctx)
        var new_hidden = nldiff_decoder_layer[S](
            hidden, w, cfg, cos_q, sin_q, cos_k, sin_k, ctx
        )
        hidden = new_hidden^
        _ = w^  # explicit drop of the layer weights before the next load
    # Final RMSNorm (encoder.norm.weight, eps 1e-5).
    var norm_w = _load_bf16(model, String("encoder.norm.weight"), ctx)
    return rms_norm(hidden, norm_w[], cfg.rms_norm_eps, ctx)


# ══════════════════════════════════════════════════════════════════════════════
# SPEED: persistent pinned-host bf16 weight store + double-buffered H2D prefetch.
#
# The streaming `nldiff_backbone` above re-parses all 34 layers (~15GB bf16) from
# the 6 shards on EVERY step (mmap H2D per layer) — the generation loop pays that
# 64x. This store parses the shards ONCE into pinned-host bf16 buffers (the exact
# `from_view` bytes, byte-identical) and streams per-layer H2D from host into 2
# device double-buffer slot-sets, overlapping the next layer's transfer behind
# the current layer's compute. Reuse: `nldiff_backbone_resident` per step. This
# is the krea2 `Krea2HostBf16` pattern (models/krea2/krea2_stack.mojo) specialized
# to the 9-weight Ministral3 decoder layer.
#
# BYTE-IDENTICAL: the slot device bytes equal the bytes `_load_layer_weights`
# would have produced (from_view -> D2H pinned host -> H2D slot is an identity
# round-trip on the raw bf16), so `nldiff_decoder_layer` sees identical inputs.
# ══════════════════════════════════════════════════════════════════════════════

comptime NLD_HArc = ArcPointer[HostBuffer[DType.uint8]]
comptime NLD_TArc = ArcPointer[Tensor]
comptime NLD_NW = 9  # weights per decoder layer (no biases)


def _nldiff_layer_tails() -> List[String]:
    """The 9 per-layer weight name tails, in the SAME order as the
    `NLDiffLayerWeights` constructor / `_load_layer_weights`."""
    var t = List[String]()
    t.append(String("input_layernorm.weight"))
    t.append(String("post_attention_layernorm.weight"))
    t.append(String("self_attn.q_proj.weight"))
    t.append(String("self_attn.k_proj.weight"))
    t.append(String("self_attn.v_proj.weight"))
    t.append(String("self_attn.o_proj.weight"))
    t.append(String("mlp.gate_proj.weight"))
    t.append(String("mlp.up_proj.weight"))
    t.append(String("mlp.down_proj.weight"))
    return t^


struct NLDiffHostLayer(Copyable, Movable):
    """One decoder layer's 9 bf16 weights as PINNED-HOST buffers (slot order)."""

    var w_h: List[NLD_HArc]
    var w_nbytes: List[Int]
    var w_shape: List[List[Int]]

    def __init__(
        out self,
        var w_h: List[NLD_HArc],
        var w_nbytes: List[Int],
        var w_shape: List[List[Int]],
    ):
        self.w_h = w_h^
        self.w_nbytes = w_nbytes^
        self.w_shape = w_shape^


struct NLDiffHostStore(Copyable, Movable):
    """All 34 layers pinned-host bf16 + a device-resident final norm weight, plus
    2 device double-buffer slot-sets (18 tensors), a copy stream, and 2 per-slot
    H2D-done events (per-block `ctx.synchronize()` makes slot reuse safe)."""

    var layers: List[NLDiffHostLayer]
    var norm: NLD_TArc                    # encoder.norm.weight (device-resident)
    var slot_w: List[NLD_TArc]            # 2*9 = 18 device bf16 slot tensors
    var copy_stream: List[ArcPointer[DeviceStream]]
    var ev: List[ArcPointer[DeviceEvent]]

    def __init__(
        out self,
        var layers: List[NLDiffHostLayer],
        var norm: NLD_TArc,
        var slot_w: List[NLD_TArc],
        var copy_stream: List[ArcPointer[DeviceStream]],
        var ev: List[ArcPointer[DeviceEvent]],
    ):
        self.layers = layers^
        self.norm = norm^
        self.slot_w = slot_w^
        self.copy_stream = copy_stream^
        self.ev = ev^


def build_nldiff_host_store(
    model: ShardedSafeTensors, cfg: NLDiffConfig, ctx: DeviceContext
) raises -> NLDiffHostStore:
    """Parse the 6 shards ONCE: pin all 34 layers' 9 bf16 weights to host
    (~15GB host RAM) as the EXACT `_load_bf16`/`from_view` bytes, then set up the
    2 device slot-sets + copy stream + events for double-buffered H2D."""
    var tails = _nldiff_layer_tails()
    var layers = List[NLDiffHostLayer]()
    for i in range(cfg.num_layers):
        var p = String("encoder.layers.") + String(i) + String(".")
        var w_h = List[NLD_HArc]()
        var w_nbytes = List[Int]()
        var w_shape = List[List[Int]]()
        for ki in range(NLD_NW):
            # from_view == the streaming path's per-layer load (raw bf16 bytes).
            var t = Tensor.from_view(model.tensor_view(p + tails[ki]), ctx)
            var bh = ctx.enqueue_create_host_buffer[DType.uint8](t.nbytes())
            ctx.enqueue_copy(bh, t.buf)      # D2H raw bytes
            ctx.synchronize()
            w_h.append(NLD_HArc(bh^))
            w_nbytes.append(t.nbytes())
            w_shape.append(t.shape().copy())
            # `t` drops here -> its ~100MB device buffer frees; trim the async
            # pool so 34x9 transient loads don't accumulate into an OOM.
        layers.append(NLDiffHostLayer(w_h^, w_nbytes^, w_shape^))
        cu_mempool_trim_current(0)
        if (i + 1) % 8 == 0 or i + 1 == cfg.num_layers:
            print("[store] pinned layer", i + 1, "/", cfg.num_layers)
    # Final norm stays device-resident (tiny [4096]).
    var norm_t = Tensor.from_view(
        model.tensor_view(String("encoder.norm.weight")), ctx)
    # 2 device slot-sets sized from layer-0 shapes (uniform across all 34 layers).
    var slot_w = List[NLD_TArc]()
    ref l0 = layers[0]
    for slot in range(2):
        _ = slot
        for i in range(NLD_NW):
            var dbuf = ctx.enqueue_create_buffer[DType.uint8](l0.w_nbytes[i])
            var t = Tensor(dbuf^, l0.w_shape[i].copy(), STDtype.BF16)
            slot_w.append(NLD_TArc(t^))
    ctx.synchronize()
    var copy_stream = List[ArcPointer[DeviceStream]]()
    copy_stream.append(ArcPointer(ctx.create_stream()))
    var ev = List[ArcPointer[DeviceEvent]]()
    ev.append(ArcPointer(ctx.create_event[disable_timing=True]()))
    ev.append(ArcPointer(ctx.create_event[disable_timing=True]()))
    cu_mempool_trim_current(0)
    print("[store] READY: 34 layers pinned-host bf16 + 2 device slot-sets")
    return NLDiffHostStore(
        layers^, NLD_TArc(norm_t^), slot_w^, copy_stream^, ev^)


def nldiff_host_prefetch(store: NLDiffHostStore, i: Int) raises:
    """Async H2D of layer i's 9 bf16 weights into slot i%2 on the copy stream."""
    if i < 0 or i >= len(store.layers):
        return
    var slot = i % 2
    ref L = store.layers[i]
    for k in range(NLD_NW):
        _h2d_dma_copy(
            UInt64(Int(store.slot_w[slot * NLD_NW + k][].buf.unsafe_ptr())),
            L.w_h[k][].unsafe_ptr(),
            L.w_nbytes[k],
            store.copy_stream[0][],
        )
    store.copy_stream[0][].record_event(store.ev[slot][])


def _load_nldiff_layer_slot(
    store: NLDiffHostStore, i: Int, ctx: DeviceContext
) raises -> NLDiffLayerWeights:
    """Fence slot i%2's H2D on the main stream, then build a `NLDiffLayerWeights`
    over the PERSISTENT slot tensors (refcount copies of the ArcPointers)."""
    var slot = i % 2
    ctx.stream().enqueue_wait_for(store.ev[slot][])
    var b = slot * NLD_NW
    return NLDiffLayerWeights(
        store.slot_w[b + 0].copy(),
        store.slot_w[b + 1].copy(),
        store.slot_w[b + 2].copy(),
        store.slot_w[b + 3].copy(),
        store.slot_w[b + 4].copy(),
        store.slot_w[b + 5].copy(),
        store.slot_w[b + 6].copy(),
        store.slot_w[b + 7].copy(),
        store.slot_w[b + 8].copy(),
    )


def nldiff_backbone_resident[
    S: Int
](
    x0: Tensor,
    store: NLDiffHostStore,
    cfg: NLDiffConfig,
    cos_q: Tensor,
    sin_q: Tensor,
    cos_k: Tensor,
    sin_k: Tensor,
    ctx: DeviceContext,
) raises -> Tensor:
    """Byte-identical drop-in for `nldiff_backbone` that streams the 34 layers
    from the persistent pinned-host store with double-buffered H2D prefetch —
    NO per-step shard re-parse. Same math as `nldiff_backbone`."""
    var hidden = _clone(x0, ctx)
    # Prime the double-buffer: start layer-0's H2D now (lands during the clone).
    nldiff_host_prefetch(store, 0)
    for i in range(cfg.num_layers):
        var w = _load_nldiff_layer_slot(store, i, ctx)   # fence slot i%2
        nldiff_host_prefetch(store, i + 1)                # stage next behind compute
        var new_hidden = nldiff_decoder_layer[S](
            hidden, w, cfg, cos_q, sin_q, cos_k, sin_k, ctx
        )
        hidden = new_hidden^
        # Per-block sync: guarantees compute-i (reading slot i%2) completes before
        # slot i%2 is reused by the layer-(i+2) prefetch issued next iteration.
        ctx.synchronize()
        _ = w^
    return rms_norm(hidden, store.norm[], cfg.rms_norm_eps, ctx)

# lingbot_qwen3vl_vision.mojo — pure-Mojo Qwen3-VL VISION tower for LingBot-Video.
#
# Ports transformers/models/qwen3_vl/modeling_qwen3_vl.py Qwen3VLVisionModel
# (config: depth 24, hidden 1024, heads 16 (head_dim 64), intermediate 4096,
#  patch 16, temporal_patch 2, spatial_merge 2, out_hidden 2560, pos_embed
#  48x48=2304 learned + bilinear interp, gelu_pytorch_tanh MLP, deepstack at
#  blocks [5,11,17], merger gelu-exact). Loads model.visual.* (315 BF16 tensors)
#  from the LingBot text_encoder shards.
#
# Pipeline (single image, seq = t*h*w patches, Nmerged = seq/4):
#   patch_embed(linear on [seq,1536]) + interpolated pos_embed
#   -> 24 pre-norm blocks (h=h+attn(norm1 h); h=h+mlp(norm2 h)), half-split 2D
#      rope on q/k, full non-causal MHA
#   -> capture hidden after blocks 5/11/17 -> 3 postshuffle deepstack mergers
#   -> final preshuffle merger -> pooler_output [Nmerged,2560]
#
# Parity: gated stage-by-stage cos>=0.999 vs torch (parity/qwen3vl_vision_oracle.py).
#
# Mojo 1.0.0b1, NVIDIA GPU. BF16 storage, F32 accumulation.

from std.math import cos as fcos, sin as fsin, exp as fexp, log as flog, sqrt as fsqrt
from std.memory import ArcPointer
from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.ffi import BytePtr, sys_memcpy
from serenitymojo.ops.linear import linear, linear_bias
from serenitymojo.ops.norm import layer_norm
from serenitymojo.ops.attention import sdpa_nomask
from serenitymojo.ops.activations import gelu, gelu_exact
from serenitymojo.ops.rope import rope_halfsplit
from serenitymojo.ops.tensor_algebra import add, reshape, reshape_owned


# ── Config (Qwen3-VL vision_config) ─────────────────────────────────────────
comptime V_DEPTH = 24
comptime V_HID = 1024
comptime V_HEADS = 16
comptime V_HEAD_DIM = 64          # V_HID // V_HEADS
comptime V_ROT_DIM = 32           # V_HEAD_DIM // 2 (rotary applied over head_dim)
comptime V_ROT_HALF = 16          # V_ROT_DIM // 2 (freqs per axis)
comptime V_INTER = 4096
comptime V_OUT = 2560
comptime V_PATCH_DIM = 1536       # in_ch*temporal*patch*patch = 3*2*16*16
comptime V_MERGE = 2
comptime V_MERGE_UNIT = 4         # merge*merge
comptime V_GRID_SIDE = 48         # sqrt(num_position_embeddings=2304)
comptime V_EPS = Float32(1.0e-6)
comptime V_THETA = Float64(10000.0)
comptime V_MERGED_HID = 4096      # V_HID * V_MERGE_UNIT
comptime DS0 = 5
comptime DS1 = 11
comptime DS2 = 17


# ── Per-stage forward taps (parity) ─────────────────────────────────────────
struct RopeTables(Movable):
    var cos_apply: Tensor   # [seq*heads, 32]
    var sin_apply: Tensor   # [seq*heads, 32]
    var cos64: Tensor       # [seq, 64]
    var sin64: Tensor       # [seq, 64]

    def __init__(
        out self, var cos_apply: Tensor, var sin_apply: Tensor,
        var cos64: Tensor, var sin64: Tensor,
    ):
        self.cos_apply = cos_apply^
        self.sin_apply = sin_apply^
        self.cos64 = cos64^
        self.sin64 = sin64^


struct VisionOutput(Movable):
    var s1_hidden: Tensor   # [seq, 1024] hidden after patch_embed + pos_embed
    var cos64: Tensor       # [seq, 64]   position_embeddings cos (gate)
    var sin64: Tensor       # [seq, 64]   position_embeddings sin (gate)
    var block0: Tensor      # [seq, 1024] output of blocks[0]
    var pooler: Tensor      # [Nmerged, 2560]
    var ds0: Tensor         # [Nmerged, 2560] deepstack feature (block 5)
    var ds1: Tensor         # [Nmerged, 2560] deepstack feature (block 11)
    var ds2: Tensor         # [Nmerged, 2560] deepstack feature (block 17)

    def __init__(
        out self,
        var s1_hidden: Tensor, var cos64: Tensor, var sin64: Tensor,
        var block0: Tensor, var pooler: Tensor,
        var ds0: Tensor, var ds1: Tensor, var ds2: Tensor,
    ):
        self.s1_hidden = s1_hidden^
        self.cos64 = cos64^
        self.sin64 = sin64^
        self.block0 = block0^
        self.pooler = pooler^
        self.ds0 = ds0^
        self.ds1 = ds1^
        self.ds2 = ds2^


# ── load helpers ────────────────────────────────────────────────────────────
def _slice_to_device(
    hbase: Int, byte_off: Int, nbytes: Int,
    var shape: List[Int], dtype: STDtype, ctx: DeviceContext,
) raises -> Tensor:
    """H2D-copy a contiguous byte sub-range of an mmap'd tensor into a fresh
    device buffer with the given shape/dtype (used to split fused qkv)."""
    var host = ctx.enqueue_create_host_buffer[DType.uint8](nbytes)
    var hdst = BytePtr(unsafe_from_address=Int(host.unsafe_ptr()))
    var hsrc = BytePtr(unsafe_from_address=hbase + byte_off)
    _ = sys_memcpy(hdst, hsrc, nbytes)
    var dev = ctx.enqueue_create_buffer[DType.uint8](nbytes)
    ctx.enqueue_copy(dst_buf=dev, src_buf=host)
    ctx.synchronize()
    return Tensor(dev^, shape^, dtype)


# ── Qwen3VLVisionModel ──────────────────────────────────────────────────────
struct Qwen3VLVisionModel(Movable):
    """Pure-Mojo Qwen3-VL vision tower. Owns all model.visual.* weights (keys
    stripped of the `model.visual.` prefix). Fused attn.qkv is pre-split into
    per-block q/k/v weight+bias at load."""

    var weights: List[ArcPointer[Tensor]]
    var name_to_idx: Dict[String, Int]

    def __init__(
        out self,
        var weights: List[ArcPointer[Tensor]],
        var name_to_idx: Dict[String, Int],
    ):
        self.weights = weights^
        self.name_to_idx = name_to_idx^

    def _w(self, name: String) raises -> ref [self.weights[0]] Tensor:
        if name not in self.name_to_idx:
            raise Error(String("missing visual weight: ") + name)
        return self.weights[self.name_to_idx[name]][]

    def _add(mut self, var name: String, var t: Tensor) raises:
        self.name_to_idx[name^] = len(self.weights)
        self.weights.append(ArcPointer(t^))

    @staticmethod
    def load(te_dir: String, ctx: DeviceContext) raises -> Qwen3VLVisionModel:
        """Load model.visual.* (315 BF16 tensors) from the LingBot text_encoder
        shards. patch_embed.proj.weight [1024,3,2,16,16] is reshaped to the
        rank-2 [1024,1536] linear weight; each block's fused attn.qkv
        weight[3072,1024]/bias[3072] is split into q/k/v [1024,1024]/[1024]."""
        var st = ShardedSafeTensors.open(te_dir)
        var weights = List[ArcPointer[Tensor]]()
        var n2i = Dict[String, Int]()
        var model = Qwen3VLVisionModel(weights^, n2i^)

        for ref nm in st.names():
            if not nm.startswith("model.visual."):
                continue
            var key = String(nm.removeprefix("model.visual."))
            var tv = st.tensor_view(nm)

            if key == "patch_embed.proj.weight":
                var t = Tensor.from_view(tv, ctx)
                var t2 = reshape_owned(t^, [V_HID, V_PATCH_DIM])
                model._add(key^, t2^)
                continue

            # Split fused qkv weight/bias into q/k/v.
            var wsuf = String(key.removesuffix("qkv.weight"))
            if wsuf.byte_length() != key.byte_length():
                var hbase = Int(tv.data.unsafe_ptr())
                var row_bytes = V_HID * tv.dtype.byte_size()   # 1024 cols
                var blk_bytes = V_HID * row_bytes              # 1024 rows
                var q = _slice_to_device(hbase, 0 * blk_bytes, blk_bytes, [V_HID, V_HID], tv.dtype, ctx)
                var k = _slice_to_device(hbase, 1 * blk_bytes, blk_bytes, [V_HID, V_HID], tv.dtype, ctx)
                var v = _slice_to_device(hbase, 2 * blk_bytes, blk_bytes, [V_HID, V_HID], tv.dtype, ctx)
                model._add(wsuf + "q.weight", q^)
                model._add(wsuf + "k.weight", k^)
                model._add(wsuf + "v.weight", v^)
                continue
            var bsuf = String(key.removesuffix("qkv.bias"))
            if bsuf.byte_length() != key.byte_length():
                var hbase = Int(tv.data.unsafe_ptr())
                var blk_bytes = V_HID * tv.dtype.byte_size()
                var q = _slice_to_device(hbase, 0 * blk_bytes, blk_bytes, [V_HID], tv.dtype, ctx)
                var k = _slice_to_device(hbase, 1 * blk_bytes, blk_bytes, [V_HID], tv.dtype, ctx)
                var v = _slice_to_device(hbase, 2 * blk_bytes, blk_bytes, [V_HID], tv.dtype, ctx)
                model._add(bsuf + "q.bias", q^)
                model._add(bsuf + "k.bias", k^)
                model._add(bsuf + "v.bias", v^)
                continue

            var t = Tensor.from_view(tv, ctx)
            model._add(key^, t^)

        return model^

    # ── S1: patch_embed + interpolated pos_embed ────────────────────────────
    def _build_pos_embeds(
        self, t: Int, h: Int, w: Int, ctx: DeviceContext
    ) raises -> Tensor:
        """Port fast_pos_embed_interpolate (modeling:702-763): bilinear interp
        of the 48x48 learned pos_embed to (h,w), then the 2x2-merge permute so
        the output token order matches the block-ordered patch stream. Returns
        [seq, 1024] BF16."""
        ref table = self._w(String("pos_embed.weight"))  # [2304, 1024]
        var tab = table.to_host(ctx)                     # F32 host copy
        var side = V_GRID_SIDE
        var seq = t * h * w

        # linspace(0, side-1, n) floor/ceil + fractional weights.
        var hf = List[Int]()
        var hc = List[Int]()
        var dh = List[Float32]()
        var step_h = Float32(side - 1) / Float32(h - 1) if h > 1 else Float32(0.0)
        for i in range(h):
            var v = Float32(i) * step_h
            var fl = Int(v)
            hf.append(fl)
            var cl = fl + 1
            if cl > side - 1: cl = side - 1
            hc.append(cl)
            dh.append(v - Float32(fl))
        var wf = List[Int]()
        var wc = List[Int]()
        var dw = List[Float32]()
        var step_w = Float32(side - 1) / Float32(w - 1) if w > 1 else Float32(0.0)
        for j in range(w):
            var v = Float32(j) * step_w
            var fl = Int(v)
            wf.append(fl)
            var cl = fl + 1
            if cl > side - 1: cl = side - 1
            wc.append(cl)
            dw.append(v - Float32(fl))

        # Raster-order (h-major) interpolated pos, then reorder into 2x2 blocks.
        var merged_h = h // V_MERGE
        var merged_w = w // V_MERGE
        var out = List[Float32]()
        out.resize(seq * V_HID, Float32(0.0))
        # iterate output tokens in block order == final stream order
        var tok = 0
        for _frame in range(t):
            for bi in range(merged_h):
                for bj in range(merged_w):
                    for mr in range(V_MERGE):
                        for mc in range(V_MERGE):
                            var i = bi * V_MERGE + mr
                            var j = bj * V_MERGE + mc
                            var w00 = (Float32(1.0) - dh[i]) * (Float32(1.0) - dw[j])
                            var w01 = (Float32(1.0) - dh[i]) * dw[j]
                            var w10 = dh[i] * (Float32(1.0) - dw[j])
                            var w11 = dh[i] * dw[j]
                            var i00 = (hf[i] * side + wf[j]) * V_HID
                            var i01 = (hf[i] * side + wc[j]) * V_HID
                            var i10 = (hc[i] * side + wf[j]) * V_HID
                            var i11 = (hc[i] * side + wc[j]) * V_HID
                            var ob = tok * V_HID
                            for c in range(V_HID):
                                out[ob + c] = (
                                    w00 * tab[i00 + c] + w01 * tab[i01 + c]
                                    + w10 * tab[i10 + c] + w11 * tab[i11 + c]
                                )
                            tok += 1
        # F32 activation stream (weights stay BF16): keeps the residual
        # accumulator from re-quantizing to BF16 every block, which otherwise
        # compounds to ~1.5e-3 cos deficit by block 23.
        return Tensor.from_host(out, [seq, V_HID], STDtype.F32, ctx)

    # ── S2: 2D rope tables (half-split) ─────────────────────────────────────
    def _build_rope(
        self, t: Int, h: Int, w: Int, ctx: DeviceContext,
    ) raises -> RopeTables:
        """Port rot_pos_emb (modeling:662-700): per-token (row,col) coords in
        2x2-merge block order, look up freqs, build cos/sin. Returns
        (cos_apply[seq*heads,32], sin_apply[seq*heads,32]) for rope on q/k, plus
        (cos64[seq,64], sin64[seq,64]) = cat(f,f) for the S2 gate."""
        var seq = t * h * w
        var merged_h = h // V_MERGE
        var merged_w = w // V_MERGE

        # inv_freq[k] = 1/theta^(2k/rot_dim), k=0..15  (rot_dim=32)
        var inv = List[Float32]()
        for k in range(V_ROT_HALF):
            var e = Float64(2 * k) / Float64(V_ROT_DIM)
            inv.append(Float32(fexp(-e * flog(V_THETA))))

        # f_cos/f_sin: per token, 32 values = [row*inv(16), col*inv(16)]
        var cos_apply = List[Float32]()
        var sin_apply = List[Float32]()
        cos_apply.resize(seq * V_HEADS * V_ROT_DIM, Float32(0.0))
        sin_apply.resize(seq * V_HEADS * V_ROT_DIM, Float32(0.0))
        var cos64 = List[Float32]()
        var sin64 = List[Float32]()
        cos64.resize(seq * V_HEAD_DIM, Float32(0.0))
        sin64.resize(seq * V_HEAD_DIM, Float32(0.0))

        var tok = 0
        for _frame in range(t):
            for bi in range(merged_h):
                for bj in range(merged_w):
                    for mr in range(V_MERGE):
                        for mc in range(V_MERGE):
                            var row = bi * V_MERGE + mr
                            var col = bj * V_MERGE + mc
                            # 32-vector f = [row*inv0..15, col*inv0..15]
                            for k in range(V_ROT_HALF):
                                var fr = Float32(row) * inv[k]
                                var fcv = Float32(col) * inv[k]
                                var cr = Float32(fcos(fr)); var sr = Float32(fsin(fr))
                                var cc = Float32(fcos(fcv)); var sc = Float32(fsin(fcv))
                                # apply-table index within the 32-vector
                                var a0 = k               # row half
                                var a1 = V_ROT_HALF + k  # col half
                                for head in range(V_HEADS):
                                    var base = (tok * V_HEADS + head) * V_ROT_DIM
                                    cos_apply[base + a0] = cr
                                    sin_apply[base + a0] = sr
                                    cos_apply[base + a1] = cc
                                    sin_apply[base + a1] = sc
                                # gate: cos64 = cat(f,f) -> [.., 32) and [32,64)
                                var g = tok * V_HEAD_DIM
                                cos64[g + a0] = cr;              sin64[g + a0] = sr
                                cos64[g + a1] = cc;              sin64[g + a1] = sc
                                cos64[g + V_ROT_DIM + a0] = cr;  sin64[g + V_ROT_DIM + a0] = sr
                                cos64[g + V_ROT_DIM + a1] = cc;  sin64[g + V_ROT_DIM + a1] = sc
                            tok += 1

        var ca = Tensor.from_host(cos_apply, [seq * V_HEADS, V_ROT_DIM], STDtype.F32, ctx)
        var sa = Tensor.from_host(sin_apply, [seq * V_HEADS, V_ROT_DIM], STDtype.F32, ctx)
        var c64 = Tensor.from_host(cos64, [seq, V_HEAD_DIM], STDtype.F32, ctx)
        var s64 = Tensor.from_host(sin64, [seq, V_HEAD_DIM], STDtype.F32, ctx)
        return RopeTables(ca^, sa^, c64^, s64^)

    # ── one vision block ────────────────────────────────────────────────────
    def _block[S: Int](
        self, blk: Int, var hidden: Tensor,
        cos_apply: Tensor, sin_apply: Tensor, ctx: DeviceContext,
    ) raises -> Tensor:
        var p = String("blocks.") + String(blk) + "."
        var scale = Float32(1.0) / fsqrt(Float32(V_HEAD_DIM))

        ref n1w = self._w(p + "norm1.weight")
        ref n1b = self._w(p + "norm1.bias")
        var normed = layer_norm(hidden, n1w, n1b, V_EPS, ctx)

        ref qw = self._w(p + "attn.q.weight"); ref qb = self._w(p + "attn.q.bias")
        ref kw = self._w(p + "attn.k.weight"); ref kb = self._w(p + "attn.k.bias")
        ref vw = self._w(p + "attn.v.weight"); ref vb = self._w(p + "attn.v.bias")
        var q = linear_bias(normed, qw, qb, ctx)   # [S, 1024]
        var k = linear_bias(normed, kw, kb, ctx)
        var v = linear_bias(normed, vw, vb, ctx)

        # rope on q,k: reshape [S,1024]->[S*heads,64] view (rows=S*heads,half=32)
        q = reshape(q, [S * V_HEADS, V_HEAD_DIM], ctx)
        k = reshape(k, [S * V_HEADS, V_HEAD_DIM], ctx)
        q = rope_halfsplit(q, cos_apply, sin_apply, ctx)
        k = rope_halfsplit(k, cos_apply, sin_apply, ctx)

        # BSHD [1,S,heads,64] for full non-causal MHA (single segment, no mask)
        q = reshape(q, [1, S, V_HEADS, V_HEAD_DIM], ctx)
        k = reshape(k, [1, S, V_HEADS, V_HEAD_DIM], ctx)
        v = reshape(v, [1, S, V_HEADS, V_HEAD_DIM], ctx)
        var attn = sdpa_nomask[1, S, V_HEADS, V_HEAD_DIM](q, k, v, scale, ctx)
        attn = reshape(attn, [S, V_HID], ctx)

        ref ow = self._w(p + "attn.proj.weight"); ref ob = self._w(p + "attn.proj.bias")
        var attn_out = linear_bias(attn, ow, ob, ctx)
        var h1 = add(hidden, attn_out, ctx)

        ref n2w = self._w(p + "norm2.weight"); ref n2b = self._w(p + "norm2.bias")
        var normed2 = layer_norm(h1, n2w, n2b, V_EPS, ctx)
        ref f1w = self._w(p + "mlp.linear_fc1.weight"); ref f1b = self._w(p + "mlp.linear_fc1.bias")
        ref f2w = self._w(p + "mlp.linear_fc2.weight"); ref f2b = self._w(p + "mlp.linear_fc2.bias")
        var mh = linear_bias(normed2, f1w, f1b, ctx)   # [S,4096]
        mh = gelu(mh, ctx)                              # gelu_pytorch_tanh
        var mlp_out = linear_bias(mh, f2w, f2b, ctx)    # [S,1024]
        return add(h1, mlp_out, ctx)

    # ── merger (preshuffle norm@1024) ───────────────────────────────────────
    def _merger(self, last_hidden: Tensor, ctx: DeviceContext) raises -> Tensor:
        var seq = last_hidden.shape()[0]
        var nm = seq // V_MERGE_UNIT
        ref nw = self._w(String("merger.norm.weight"))   # [1024]
        ref nb = self._w(String("merger.norm.bias"))
        var x = layer_norm(last_hidden, nw, nb, V_EPS, ctx)  # norm BEFORE shuffle
        x = reshape(x, [nm, V_MERGED_HID], ctx)              # group 4 tokens
        ref f1w = self._w(String("merger.linear_fc1.weight")); ref f1b = self._w(String("merger.linear_fc1.bias"))
        ref f2w = self._w(String("merger.linear_fc2.weight")); ref f2b = self._w(String("merger.linear_fc2.bias"))
        x = linear_bias(x, f1w, f1b, ctx)
        x = gelu_exact(x, ctx)                               # nn.GELU
        return linear_bias(x, f2w, f2b, ctx)                 # [nm, 2560]

    # ── deepstack merger (postshuffle norm@4096) ────────────────────────────
    def _deepstack(self, idx: Int, hidden: Tensor, ctx: DeviceContext) raises -> Tensor:
        var seq = hidden.shape()[0]
        var nm = seq // V_MERGE_UNIT
        var p = String("deepstack_merger_list.") + String(idx) + "."
        var x = reshape(hidden, [nm, V_MERGED_HID], ctx)     # shuffle FIRST
        ref nw = self._w(p + "norm.weight"); ref nb = self._w(p + "norm.bias")  # [4096]
        x = layer_norm(x, nw, nb, V_EPS, ctx)                # norm AFTER shuffle
        ref f1w = self._w(p + "linear_fc1.weight"); ref f1b = self._w(p + "linear_fc1.bias")
        ref f2w = self._w(p + "linear_fc2.weight"); ref f2b = self._w(p + "linear_fc2.bias")
        x = linear_bias(x, f1w, f1b, ctx)
        x = gelu_exact(x, ctx)
        return linear_bias(x, f2w, f2b, ctx)

    # ── full forward ────────────────────────────────────────────────────────
    def forward[S: Int](
        self, pixel_values: Tensor, t: Int, h: Int, w: Int, ctx: DeviceContext,
    ) raises -> VisionOutput:
        if t * h * w != S:
            raise Error(String("forward: t*h*w=") + String(t * h * w)
                        + " != comptime S=" + String(S))

        # patch embed (linear on [seq,1536]) + interpolated pos embed
        ref pw = self._w(String("patch_embed.proj.weight"))
        ref pb = self._w(String("patch_embed.proj.bias"))
        var hidden = linear_bias(pixel_values, pw, pb, ctx)  # [seq,1024]
        var pos = self._build_pos_embeds(t, h, w, ctx)
        hidden = add(hidden, pos, ctx)
        var s1_hidden = hidden.clone(ctx)

        var rope = self._build_rope(t, h, w, ctx)
        # clone the gate tables into the output; `rope` is consumed whole at
        # scope exit (avoids a partial-move of the RopeTables struct).
        var cos64 = rope.cos64.clone(ctx)
        var sin64 = rope.sin64.clone(ctx)

        var block0 = Tensor(ctx.enqueue_create_buffer[DType.uint8](1), [1], STDtype.BF16)
        var ds0 = Tensor(ctx.enqueue_create_buffer[DType.uint8](1), [1], STDtype.BF16)
        var ds1 = Tensor(ctx.enqueue_create_buffer[DType.uint8](1), [1], STDtype.BF16)
        var ds2 = Tensor(ctx.enqueue_create_buffer[DType.uint8](1), [1], STDtype.BF16)

        for b in range(V_DEPTH):
            hidden = self._block[S](b, hidden^, rope.cos_apply, rope.sin_apply, ctx)
            if b == 0:
                block0 = hidden.clone(ctx)
            if b == DS0:
                ds0 = self._deepstack(0, hidden, ctx)
            elif b == DS1:
                ds1 = self._deepstack(1, hidden, ctx)
            elif b == DS2:
                ds2 = self._deepstack(2, hidden, ctx)

        var pooler = self._merger(hidden, ctx)

        return VisionOutput(
            s1_hidden^, cos64^, sin64^, block0^, pooler^, ds0^, ds1^, ds2^,
        )

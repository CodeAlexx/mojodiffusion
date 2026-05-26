# models/dit/hidream_o1.mojo — HiDream-O1-Image DiT (Qwen3-VL 8B spine + 3 heads).
#
# HiDream-O1 is **Qwen3-VL 8B with three image-diffusion heads bolted on**,
# running ONE forward pass per denoise step. There is NO separate text encoder
# (the spine's own embed_tokens IS the conditioning path) and NO VAE
# (the model operates directly in 32x32x3 RGB patch space, latents in [-1,1]).
#
# Reference, read line-by-line (the Rust port is end-to-end):
#   /home/alex/EriDiffusion/inference-flame/src/models/hidream_o1/
#     model.rs              — HiDreamO1Model::forward_inner (the denoise step)
#     decoder.rs            — decoder_forward_with_weights_lora (one Qwen3-VL block)
#     mrope.rs              — interleaved_mrope_cos_sin + build_mrope_positions
#     bottleneck_patch_embed.rs / timestep_embedder.rs / final_layer.rs (the 3 heads)
#     pipeline.rs           — build_t2i_input, generate, gather_image_rows
#   config.json (on disk)   — Qwen3VLForConditionalGeneration text_config
#
# Forward (T2I, no ref-image, batch 1), model.rs:265-585:
#   1) text_emb = embed_tokens(input_ids)                       [1, S_text, H]
#   2) t_emb = TimestepEmbedder(timestep)  [1, H]; scatter into the tms slot
#   3) patch_emb = BottleneckPatchEmbed(noise_patches)          [1, L, H]
#   4) inputs = cat([text_emb_with_t, patch_emb], dim=1)        [1, S, H]
#   5) MRoPE (cos,sin) interleaved T/H/W half-table             [1, S, head_dim/2]
#   6) prefix-causal/full additive mask [1, Hh, S, S]
#   7) 36 decoder layers (GQA 32/8, head_dim 128, SwiGLU 12288)
#   8) final RMSNorm (model.norm)
#   9) FinalLayer linear -> [1, S, 3072]  (caller gathers the L image rows)
#
# mRoPE (mrope.rs, config rope_scaling.mrope_interleaved=True):
#   The rotation is HALFSPLIT (HF rotate_half); the "interleaved" part is only
#   how the T/H/W frequency bins are stride-3 woven into the head_dim/2 freqs.
#   slot d in [0,half): axis H if (d%3==1 and d<20*3); axis W if (d%3==2 and
#   d<20*3); else axis T. section=[24,20,20] sums to half=64. theta=5e6.
#   serenitymojo ops/rope.rope_halfsplit consumes cos/sin as [rows, half],
#   which is exactly the half-table this builder produces (Rust truncates the
#   (freqs,freqs) duplicate to [1,S,half] before upload — same thing).
#
# Reused foundation ops (NOT reimplemented): ops/linear.linear,
# ops/norm.rms_norm, ops/rope.rope_halfsplit, ops/activations.swiglu,
# ops/attention.sdpa (math-mode, Dh=128), ops/tensor_algebra.{add,reshape,...},
# ops/embeddings.timestep_embedding (cos-first sinusoid).
#
# The DiT struct is comptime-parameterized on S (full sequence length) so the
# comptime-shaped sdpa gets a static (B,S,H,Dh). The all-zeros / prefix mask is
# built ONCE per forward and passed in (per-call alloc is an OOM risk).
#
# Mojo 1.0.0b1, NVIDIA GPU. Weights BF16, F32 math accumulation.

from std.math import sqrt, exp, log, cos as fcos, sin as fsin
from std.memory import ArcPointer
from std.gpu.host import DeviceContext
from std.gpu import global_idx
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import rms_norm
from serenitymojo.ops.rope import rope_halfsplit
from serenitymojo.ops.activations import swiglu
from serenitymojo.ops.attention import sdpa
from serenitymojo.ops.tensor_algebra import (
    add,
    reshape,
    concat,
    gather_rows,
    slice as ts_slice,
)
from serenitymojo.ops.embeddings import timestep_embedding
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.activations import silu


# ── Config ──────────────────────────────────────────────────────────────────
@fieldwise_init
struct HiDreamO1Config(Copyable, Movable, ImplicitlyCopyable):
    """Mirrors HiDreamO1Config::dev_8b() (mod.rs:127) + config.json text_config.
    All values verified against the on-disk model.safetensors.index.json header
    and config.json (Qwen3VLForConditionalGeneration)."""

    var hidden_size: Int  # 4096
    var num_layers: Int  # 36
    var num_heads: Int  # 32
    var num_kv_heads: Int  # 8 (GQA 4:1)
    var head_dim: Int  # 128
    var intermediate_size: Int  # 12288 (SwiGLU)
    var rope_theta: Float64  # 5_000_000.0
    var rms_norm_eps: Float32  # 1e-6
    var vocab_size: Int  # 151936
    # mrope_section [24,20,20] baked as three Ints (sums to head_dim/2=64).
    var mrope_t: Int  # 24
    var mrope_h: Int  # 20
    var mrope_w: Int  # 20
    # HiDream heads.
    var patch_size: Int  # 32
    var patch_in_channels: Int  # 3
    var bottleneck_dim: Int  # hidden_size/4 = 1024
    var timestep_freq_dim: Int  # 256
    var fix_point: Int  # 4096
    # Special token ids (from config.json / tokenizer).
    var tms_token_id: Int  # 151673
    var image_token_id: Int  # 151655
    var video_token_id: Int  # 151656
    var vision_start_token_id: Int  # 151652

    @staticmethod
    def dev_8b() -> HiDreamO1Config:
        return HiDreamO1Config(
            4096,  # hidden_size
            36,  # num_layers
            32,  # num_heads
            8,  # num_kv_heads
            128,  # head_dim
            12288,  # intermediate_size
            Float64(5_000_000.0),  # rope_theta
            Float32(1.0e-6),  # rms_norm_eps
            151936,  # vocab_size
            24,  # mrope_t
            20,  # mrope_h
            20,  # mrope_w
            32,  # patch_size
            3,  # patch_in_channels
            1024,  # bottleneck_dim = 4096/4
            256,  # timestep_freq_dim
            4096,  # fix_point
            151673,  # tms_token_id
            151655,  # image_token_id
            151656,  # video_token_id
            151652,  # vision_start_token_id
        )


# ── MRoPE: interleaved T/H/W half-table (mrope.rs:306-442) ──────────────────
# Returns [cos_vals, sin_vals], each flat length seq*half, layout (position,
# pair) — the caller replicates across heads. axis(d): H if d%3==1 & d<3*mh; W
# if d%3==2 & d<3*mw; else T. inv_freq[d]=1/theta^(2d/head_dim). F32 trig.
def _build_mrope_tables(
    t_pos: List[Int],
    h_pos: List[Int],
    w_pos: List[Int],
    head_dim: Int,
    rope_theta: Float64,
    mrope_h: Int,
    mrope_w: Int,
) raises -> List[List[Float32]]:
    var s = len(t_pos)
    var half = head_dim // 2
    var len_h = mrope_h * 3
    var len_w = mrope_w * 3

    # Per-slot axis assignment (0=T, 1=H, 2=W).
    var axis = List[Int]()
    for d in range(half):
        var m = d % 3
        if m == 1 and d < len_h:
            axis.append(1)
        elif m == 2 and d < len_w:
            axis.append(2)
        else:
            axis.append(0)

    # inv_freq[d] = 1 / theta^(2d/head_dim).
    var inv_freq = List[Float32]()
    var log_theta = log(Float32(rope_theta))
    for d in range(half):
        var exponent = Float32(2 * d) / Float32(head_dim)
        inv_freq.append(exp(-log_theta * exponent))

    var cos_vals = List[Float32]()
    var sin_vals = List[Float32]()
    for si in range(s):
        for d in range(half):
            var pos: Int
            if axis[d] == 1:
                pos = h_pos[si]
            elif axis[d] == 2:
                pos = w_pos[si]
            else:
                pos = t_pos[si]
            var angle = Float32(pos) * inv_freq[d]
            cos_vals.append(fcos(angle))
            sin_vals.append(fsin(angle))
    var out = List[List[Float32]]()
    out.append(cos_vals^)
    out.append(sin_vals^)
    return out^


# Replicate a per-position [s, half] table to a per-head flat [s*heads*half]
# in (position, head, pair) order — the layout ops/rope.rope_halfsplit expects
# when the data tensor is [1, s, heads, head_dim] flattened to s*heads rows.
def _replicate_heads(
    table: List[Float32], s: Int, half: Int, heads: Int
) raises -> List[Float32]:
    var out = List[Float32]()
    for si in range(s):
        for _h in range(heads):
            for d in range(half):
                out.append(table[si * half + d])
    return out^


# ── MRoPE position-id builder (mrope.rs:113-273) ────────────────────────────
# Single gen-image, no-refs T2I. full_ids = text + [vision_start] + (L-1)*image.
# Text rows 0..text_len get T=H=W=row. Image rows get T=fix_point, H=fix_point+
# patch_row, W=fix_point+patch_col (skip=1 gen-image branch). text_len = ed -
# skip where ed = index of first image_token_id, skip=1.
def build_mrope_positions(
    full_ids: List[Int],
    image_token_id: Int,
    vision_start_token_id: Int,
    ph: Int,
    pw: Int,
    fix_point: Int,
) raises -> List[List[Int]]:
    var total = len(full_ids)
    var t_pos = List[Int]()
    var h_pos = List[Int]()
    var w_pos = List[Int]()
    for _ in range(total):
        t_pos.append(0)
        h_pos.append(0)
        w_pos.append(0)

    # Locate first image_token_id (= ed). text_len = ed - skip(=1).
    var ed = -1
    for i in range(total):
        if full_ids[i] == image_token_id:
            ed = i
            break
    var vs_idx = -1
    for i in range(total):
        if full_ids[i] == vision_start_token_id:
            vs_idx = i
            break
    if ed < 0:
        # No image tokens at all — pure-text fallback.
        for i in range(total):
            t_pos[i] = i
            h_pos[i] = i
            w_pos[i] = i
        var out = List[List[Int]]()
        out.append(t_pos^)
        out.append(h_pos^)
        out.append(w_pos^)
        return out^

    var skip = 1
    var text_len = ed - skip
    if text_len < 0:
        text_len = 0
    for i in range(text_len):
        t_pos[i] = i
        h_pos[i] = i
        w_pos[i] = i

    # gen-image branch (skip>0): fix_point shift, st_idx=0.
    var patch_start = text_len
    var fp = fix_point
    var k = 0
    for h in range(ph):
        for w in range(pw):
            var idx = patch_start + k
            if idx >= total:
                break
            t_pos[idx] = fp
            h_pos[idx] = h + fp
            w_pos[idx] = w + fp
            k += 1
    _ = vs_idx  # vision-start row keeps its zero stamp (matches Rust skip=1).

    var out = List[List[Int]]()
    out.append(t_pos^)
    out.append(h_pos^)
    out.append(w_pos^)
    return out^


# ── prefix-causal/full additive mask [1, H, S, S] ────────────────────────────
# decoder.rs hidream_o1_two_pass_attention: AR-prefix rows [0,ar_len) are
# causal (attend j<=i); gen rows [ar_len,S) attend to ALL keys. Additive: 0.0
# where attend, -1e4 where block. Built ONCE per forward (one mask alloc).
def _build_prefix_causal_mask(
    s: Int, heads: Int, ar_len: Int
) raises -> List[Float32]:
    var neg = Float32(-1.0e4)
    var data = List[Float32]()
    for _hh in range(heads):
        for i in range(s):
            var is_gen = i >= ar_len
            for j in range(s):
                if is_gen:
                    data.append(Float32(0.0))  # gen rows: full attention
                elif j <= i:
                    data.append(Float32(0.0))  # AR rows: causal
                else:
                    data.append(neg)
    return data^


# ── comptime SDPA dispatch (H=32, Dh=128, parameterized on S) ────────────────
def _sdpa_s[S: Int](
    q: Tensor, k: Tensor, v: Tensor, mask: Tensor, scale: Float32, ctx: DeviceContext
) raises -> Tensor:
    return sdpa[1, S, 32, 128](q, k, v, mask, scale, ctx)


# ── GQA repeat_kv (BSHD [1,S,H_kv,Dh] -> [1,S,H,Dh]) ─────────────────────────
# Grouped order: dst head `head` reads src kv-head `head // n_rep`
# (kv0,kv0,kv0,kv0,kv1,...) — PyTorch repeat_kv. Mirrors qwen3_encoder's
# _repeat_kv_kernel exactly: flat idx over [seq*h*dh],
# src_idx = (t*h_kv + head//n_rep)*dh + dh_i. New HiDream-local kernel
# (not modifying the shared encoder file).
comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256


def _repeat_kv_kernel_bf16(
    src: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    dst: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    seq: Int, h: Int, h_kv: Int, dh: Int, n_rep: Int,
):
    var idx = Int(global_idx.x)
    var total = seq * h * dh
    if idx < total:
        var dh_i = idx % dh
        var rest = idx // dh
        var head = rest % h
        var t = rest // h
        var kvh = head // n_rep
        var src_idx = (t * h_kv + kvh) * dh + dh_i
        dst[idx] = rebind[dst.element_type](src[src_idx])


def _repeat_kv_kernel_f32(
    src: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    dst: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    seq: Int, h: Int, h_kv: Int, dh: Int, n_rep: Int,
):
    var idx = Int(global_idx.x)
    var total = seq * h * dh
    if idx < total:
        var dh_i = idx % dh
        var rest = idx // dh
        var head = rest % h
        var t = rest // h
        var kvh = head // n_rep
        var src_idx = (t * h_kv + kvh) * dh + dh_i
        dst[idx] = rebind[dst.element_type](src[src_idx])


def _repeat_kv(
    x: Tensor, s: Int, h_kv: Int, n_rep: Int, dh: Int, ctx: DeviceContext
) raises -> Tensor:
    if n_rep == 1:
        var dev0 = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
        ctx.enqueue_copy(dst_buf=dev0, src_buf=x.buf)
        ctx.synchronize()
        return Tensor(dev0^, x.shape(), x.dtype())
    var h = h_kv * n_rep
    var dt = x.dtype().to_mojo_dtype()
    var out_n = s * h * dh
    var src_n = s * h_kv * dh
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](
        out_n * x.dtype().byte_size()
    )
    var src_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](src_n))
    var out_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](out_n))
    var grid = (out_n + _BLOCK - 1) // _BLOCK
    if dt == DType.bfloat16:
        var SRC = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), src_rl
        )
        var DST = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), out_rl
        )
        ctx.enqueue_function[_repeat_kv_kernel_bf16, _repeat_kv_kernel_bf16](
            SRC, DST, s, h, h_kv, dh, n_rep, grid_dim=grid, block_dim=_BLOCK
        )
    elif dt == DType.float32:
        var SRC = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float32](), src_rl
        )
        var DST = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), out_rl
        )
        ctx.enqueue_function[_repeat_kv_kernel_f32, _repeat_kv_kernel_f32](
            SRC, DST, s, h, h_kv, dh, n_rep, grid_dim=grid, block_dim=_BLOCK
        )
    else:
        raise Error("_repeat_kv: unsupported dtype (BF16/F32 only)")
    ctx.synchronize()
    var sh = List[Int]()
    sh.append(1); sh.append(s); sh.append(h); sh.append(dh)
    return Tensor(out_buf^, sh^, x.dtype())


# ── DiT ──────────────────────────────────────────────────────────────────────
struct HiDreamO1DiT[S: Int]:
    """HiDream-O1 DiT. S = full sequence length (S_text + L image patches),
    a comptime param so the comptime-shaped sdpa[1,S,32,128] gets a static
    shape. All 759-(vision+lm_head) DiT tensors resident; for a 24 GB box the
    36 BF16 layers (~8B params) should stream via BlockLoader (see report —
    this skeleton loads all-resident for compile validation; an offloaded
    variant mirrors models/dit/klein_dit.Klein9BOffloaded)."""

    var weights: List[ArcPointer[Tensor]]
    var name_to_idx: Dict[String, Int]
    var config: HiDreamO1Config

    def __init__(
        out self,
        var weights: List[ArcPointer[Tensor]],
        var name_to_idx: Dict[String, Int],
        config: HiDreamO1Config,
    ):
        self.weights = weights^
        self.name_to_idx = name_to_idx^
        self.config = config

    @staticmethod
    def load(dir: String, config: HiDreamO1Config, ctx: DeviceContext) raises -> HiDreamO1DiT[Self.S]:
        """Load DiT tensors from a sharded safetensors dir. Skips the vision
        tower (model.visual.*) and lm_head — not used for T2I generation
        (weight_loader.rs:60-62). Keys match the on-disk header exactly."""
        var sharded = ShardedSafeTensors.open(dir)
        var weights = List[ArcPointer[Tensor]]()
        var name_to_idx = Dict[String, Int]()
        for ref nm in sharded.names():
            if nm.startswith("model.visual.") or nm == String("lm_head.weight"):
                continue
            var tv = sharded.tensor_view(nm)
            var t = Tensor.from_view(tv, ctx)
            var idx = len(weights)
            weights.append(ArcPointer(t^))
            name_to_idx[nm] = idx
        return HiDreamO1DiT[Self.S](weights^, name_to_idx^, config)

    def _w(self, name: String) raises -> ref [self.weights] Tensor:
        if name not in self.name_to_idx:
            raise Error(String("HiDreamO1DiT: missing weight: ") + name)
        var idx = self.name_to_idx[name]
        return self.weights[idx][]

    def _has(self, name: String) -> Bool:
        return name in self.name_to_idx

    def _clone(self, x: Tensor, ctx: DeviceContext) raises -> Tensor:
        var dev = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
        ctx.enqueue_copy(dst_buf=dev, src_buf=x.buf)
        ctx.synchronize()
        return Tensor(dev^, x.shape(), x.dtype())

    # ── embedding gather: ids -> [1, seq, hidden] ────────────────────────────
    def _embed(self, ids: List[Int], ctx: DeviceContext) raises -> Tensor:
        # Reuse gather_rows from tensor_algebra (embedding lookup over [V,H]).
        ref table = self._w(String("model.language_model.embed_tokens.weight"))
        var rows = gather_rows(table, ids, ctx)  # [seq, hidden]
        var sh = List[Int]()
        sh.append(1)
        sh.append(len(ids))
        var ts = table.shape()
        sh.append(ts[len(ts) - 1])
        return reshape(rows, sh^, ctx)

    # ── 3 heads ──────────────────────────────────────────────────────────────
    def _patch_embed(self, patches: Tensor, ctx: DeviceContext) raises -> Tensor:
        # BottleneckPatchEmbed: proj1 (no bias) -> proj2 (bias). [1,L,3072]->[1,L,H]
        ref w1 = self._w(String("model.x_embedder.proj1.weight"))
        var h = linear(patches, w1, None, ctx)
        ref w2 = self._w(String("model.x_embedder.proj2.weight"))
        ref b2 = self._w(String("model.x_embedder.proj2.bias"))
        return linear(h, w2, Optional[Tensor](self._clone(b2, ctx)), ctx)

    def _t_embed(self, timestep: Float32, ctx: DeviceContext) raises -> Tensor:
        # TimestepEmbedder: sinusoid(t*1000, 256, cos-first) -> Linear -> SiLU
        # -> Linear. timestep_embedding (ops/embeddings) is cos-first and uses
        # max_period default 10000; we pre-scale t by 1000 here (the Rust port
        # folds t*1000 into the sinusoid; timestep_embedder.rs:25,180).
        var cfg = self.config
        var t_host = List[Float32]()
        t_host.append(timestep * Float32(1000.0))
        var dtype = self._w(String("model.t_embedder1.mlp.0.weight")).dtype()
        var t_sh = List[Int]()
        t_sh.append(1)
        var t_tensor = Tensor.from_host(t_host, t_sh^, STDtype.F32, ctx)
        # timestep_embedding wants F32 [N]; returns [N, dim] F32.
        var freq = timestep_embedding(t_tensor, cfg.timestep_freq_dim, ctx)
        # cast freq to the MLP dtype.
        var freq_c = cast_tensor(freq, dtype, ctx)
        ref w0 = self._w(String("model.t_embedder1.mlp.0.weight"))
        ref b0 = self._w(String("model.t_embedder1.mlp.0.bias"))
        var h = linear(freq_c, w0, Optional[Tensor](self._clone(b0, ctx)), ctx)  # [1, H]
        h = silu(h, ctx)
        ref w2 = self._w(String("model.t_embedder1.mlp.2.weight"))
        ref b2 = self._w(String("model.t_embedder1.mlp.2.bias"))
        return linear(h, w2, Optional[Tensor](self._clone(b2, ctx)), ctx)  # [1, H]

    def _final_layer(self, x: Tensor, ctx: DeviceContext) raises -> Tensor:
        # FinalLayer: Linear(H -> 3072) with bias. No adaLN (final_layer.rs).
        ref w = self._w(String("model.final_layer2.linear.weight"))
        ref b = self._w(String("model.final_layer2.linear.bias"))
        return linear(x, w, Optional[Tensor](self._clone(b, ctx)), ctx)

    # ── one Qwen3-VL decoder layer (decoder.rs:194-296 / 345-670) ────────────
    def _layer(
        self,
        layer_idx: Int,
        hidden: Tensor,
        cos_q: Tensor,
        sin_q: Tensor,
        cos_k: Tensor,
        sin_k: Tensor,
        mask: Tensor,
        ctx: DeviceContext,
    ) raises -> Tensor:
        var cfg = self.config
        var h = cfg.num_heads
        var h_kv = cfg.num_kv_heads
        var dh = cfg.head_dim
        var n_rep = h // h_kv
        var eps = cfg.rms_norm_eps
        var scale = Float32(1.0) / sqrt(Float32(dh))
        var p = String("model.language_model.layers.") + String(layer_idx)
        var seq = Self.S

        # 1) input_layernorm + attention
        ref in_ln = self._w(p + ".input_layernorm.weight")
        var normed = rms_norm(hidden, in_ln, eps, ctx)

        ref qw = self._w(p + ".self_attn.q_proj.weight")
        ref kw = self._w(p + ".self_attn.k_proj.weight")
        ref vw = self._w(p + ".self_attn.v_proj.weight")
        var q = linear(normed, qw, None, ctx)  # [1, S, H*Dh]
        var k = linear(normed, kw, None, ctx)  # [1, S, Hkv*Dh]
        var v = linear(normed, vw, None, ctx)  # [1, S, Hkv*Dh]

        var q_sh = List[Int]()
        q_sh.append(1); q_sh.append(seq); q_sh.append(h); q_sh.append(dh)
        q = reshape(q, q_sh^, ctx)
        var k_sh = List[Int]()
        k_sh.append(1); k_sh.append(seq); k_sh.append(h_kv); k_sh.append(dh)
        k = reshape(k, k_sh^, ctx)
        var v_sh = List[Int]()
        v_sh.append(1); v_sh.append(seq); v_sh.append(h_kv); v_sh.append(dh)
        v = reshape(v, v_sh^, ctx)

        # per-head q/k RMSNorm over Dh (decoder.rs:489-496).
        ref qn = self._w(p + ".self_attn.q_norm.weight")
        ref kn = self._w(p + ".self_attn.k_norm.weight")
        q = rms_norm(q, qn, eps, ctx)
        k = rms_norm(k, kn, eps, ctx)

        # MRoPE half-split (decoder.rs:503-505). cos_q/sin_q cover H heads;
        # cos_k/sin_k cover H_kv heads (per-position table replicated per head).
        q = rope_halfsplit(q, cos_q, sin_q, ctx)
        k = rope_halfsplit(k, cos_k, sin_k, ctx)

        # GQA repeat then SDPA in BSHD [1, S, H, Dh].
        var k_rep = _repeat_kv(k, seq, h_kv, n_rep, dh, ctx)
        var v_rep = _repeat_kv(v, seq, h_kv, n_rep, dh, ctx)
        var attn = _sdpa_s[Self.S](q, k_rep, v_rep, mask, scale, ctx)

        var attn_sh = List[Int]()
        attn_sh.append(1); attn_sh.append(seq); attn_sh.append(h * dh)
        attn = reshape(attn, attn_sh^, ctx)

        ref ow = self._w(p + ".self_attn.o_proj.weight")
        var attn_out = linear(attn, ow, None, ctx)
        var hidden2 = add(hidden, attn_out, ctx)

        # 2) post_attention_layernorm + SwiGLU MLP
        ref post_ln = self._w(p + ".post_attention_layernorm.weight")
        var normed2 = rms_norm(hidden2, post_ln, eps, ctx)
        ref gw = self._w(p + ".mlp.gate_proj.weight")
        ref uw = self._w(p + ".mlp.up_proj.weight")
        ref dw = self._w(p + ".mlp.down_proj.weight")
        var gate = linear(normed2, gw, None, ctx)
        var up = linear(normed2, uw, None, ctx)
        var act = swiglu(gate, up, ctx)
        var mlp_out = linear(act, dw, None, ctx)
        return add(hidden2, mlp_out, ctx)

    # ── single denoise-step forward (model.rs:265-585) ───────────────────────
    # input_ids: text tokens (incl. tms slot) length S_text.
    # noise_patches: [1, L, 3072] BF16 RGB patches (L = S - S_text).
    # t_pos/h_pos/w_pos: MRoPE positions over the FULL S stream (build via
    #   build_mrope_positions).
    # ar_len: AR-prefix length for the prefix-causal/full mask (= count of
    #   token_types_bin==0 prefix; includes everything before the tms row).
    # timestep: t_pixeldit in [0,1] (1=clean, 0=noisy); the t_embedder rescales.
    # Returns the FULL [1, S, 3072] velocity; caller gathers the L image rows.
    def forward(
        self,
        input_ids: List[Int],
        noise_patches: Tensor,
        t_pos: List[Int],
        h_pos: List[Int],
        w_pos: List[Int],
        ar_len: Int,
        timestep: Float32,
        ctx: DeviceContext,
    ) raises -> Tensor:
        var cfg = self.config
        var h = cfg.num_heads
        var h_kv = cfg.num_kv_heads
        var dh = cfg.head_dim
        var half = dh // 2
        var s_text = len(input_ids)
        var dtype = self._w(String("model.language_model.embed_tokens.weight")).dtype()

        # 1) text embeddings [1, S_text, H]
        var text_emb = self._embed(input_ids, ctx)

        # 2) t_emb [1, H] scattered into the tms slot. We build text_emb_with_t
        #    by replacing the tms row with t_emb. Find the tms slot host-side.
        var t_emb = self._t_embed(timestep, ctx)  # [1, H]
        var tms_idx = -1
        for i in range(s_text):
            if input_ids[i] == cfg.tms_token_id:
                tms_idx = i
        var text_emb_with_t = _scatter_row(text_emb, t_emb, tms_idx, s_text, cfg.hidden_size, ctx)

        # 3) patch embedding [1, L, H]
        var patch_emb = self._patch_embed(noise_patches, ctx)

        # 4) concat -> [1, S, H]
        var hidden = concat(1, ctx, text_emb_with_t, patch_emb)

        # 5) MRoPE tables (host trig F32), replicated per head, uploaded BF16.
        var tables = _build_mrope_tables(
            t_pos, h_pos, w_pos, dh, cfg.rope_theta, cfg.mrope_h, cfg.mrope_w
        )
        var seq = Self.S
        var cos_q_h = _replicate_heads(tables[0], seq, half, h)
        var sin_q_h = _replicate_heads(tables[1], seq, half, h)
        var cos_k_h = _replicate_heads(tables[0], seq, half, h_kv)
        var sin_k_h = _replicate_heads(tables[1], seq, half, h_kv)
        var cq_sh = List[Int](); cq_sh.append(seq * h * half)
        var ck_sh = List[Int](); ck_sh.append(seq * h_kv * half)
        var cos_q = Tensor.from_host(cos_q_h, cq_sh.copy(), dtype, ctx)
        var sin_q = Tensor.from_host(sin_q_h, cq_sh.copy(), dtype, ctx)
        var cos_k = Tensor.from_host(cos_k_h, ck_sh.copy(), dtype, ctx)
        var sin_k = Tensor.from_host(sin_k_h, ck_sh.copy(), dtype, ctx)

        # 6) prefix-causal/full additive mask [1, H, S, S], built ONCE.
        var mask_data = _build_prefix_causal_mask(seq, h, ar_len)
        var mask_sh = List[Int]()
        mask_sh.append(1); mask_sh.append(h); mask_sh.append(seq); mask_sh.append(seq)
        var mask = Tensor.from_host(mask_data, mask_sh^, dtype, ctx)

        # 7) 36 decoder layers
        for i in range(cfg.num_layers):
            hidden = self._layer(i, hidden, cos_q, sin_q, cos_k, sin_k, mask, ctx)

        # 8) final RMSNorm (model.language_model.norm)
        ref nw = self._w(String("model.language_model.norm.weight"))
        hidden = rms_norm(hidden, nw, cfg.rms_norm_eps, ctx)

        # 9) final layer head -> [1, S, 3072]
        return self._final_layer(hidden, ctx)


# Replace row `idx` of x [1, s, d] with row 0 of repl [1, d] (or [d]). Built by
# slicing the three spans and concatenating (no new kernel). If idx<0, no-op.
def _scatter_row(
    x: Tensor, repl: Tensor, idx: Int, s: Int, d: Int, ctx: DeviceContext
) raises -> Tensor:
    if idx < 0:
        var dev = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
        ctx.enqueue_copy(dst_buf=dev, src_buf=x.buf)
        ctx.synchronize()
        return Tensor(dev^, x.shape(), x.dtype())
    # x is [1, s, d]; slice along dim 1.
    var repl3 = reshape(repl, _row_shape(d), ctx)  # [1, 1, d]
    if idx == 0:
        if s == 1:
            return repl3^
        var tail = ts_slice(x, 1, 1, s - 1, ctx)
        return concat(1, ctx, repl3, tail)
    if idx == s - 1:
        var head = ts_slice(x, 1, 0, idx, ctx)
        return concat(1, ctx, head, repl3)
    var head = ts_slice(x, 1, 0, idx, ctx)
    var tail = ts_slice(x, 1, idx + 1, s - idx - 1, ctx)
    return concat(1, ctx, head, repl3, tail)


def _row_shape(d: Int) raises -> List[Int]:
    var sh = List[Int]()
    sh.append(1); sh.append(1); sh.append(d)
    return sh^

# serenitymojo/models/anima/anima_text_context.mojo
#
# Anima TEXT -> CONTEXT data path (Chunk C, TRAINING_PLAN_anima.md §C).
#
# Pure-Mojo, FORWARD-ONLY (the cross-attn context is a FROZEN trainer input —
# no backward needed). Ports net.llm_adapter (the diffusers AnimaTextConditioner /
# Anima-Standalone-Trainer LLMAdapter) which SerenityTrainer's AnimaModel.encode_text
# runs as the second stage after the Qwen3 encoder:
#
#   qwen_hidden = Qwen3(tokens).last_hidden_state          # [1, 512, 1024]
#   qwen_hidden *= tokens_mask.unsqueeze(-1)               # zero pad positions
#   context = llm_adapter(qwen_hidden, t5_ids, ...)        # [1, 512, 1024] FROZEN
#
# llm_adapter (AUTHORITATIVE: anima_models.py LLMAdapter, lines 1492-1695):
#   x = embed[t5_ids]                  # T5-id lookup -> queries [1, 512, 1024]
#   (in_proj = Identity since model_dim==target_dim==1024)
#   for each of 6 blocks:
#     x += self_attn ( RMSNorm(x) , RoPE q&k target-pos )
#     x += cross_attn( RMSNorm(x) q=target-pos , K/V=qwen_hidden k=ctx-pos )
#     x += mlp( RMSNorm(x) ) : Linear(1024->4096,bias) -> GELU(exact erf) ->
#                              Linear(4096->1024,bias)
#   context = RMSNorm( out_proj(x) )   # out_proj has bias
#
# Attention: per-head QK RMSNorm(head_dim=64,eps=1e-6); 1D half-split RoPE
# (theta=10000); SDPA scale=1/sqrt(64); 16 heads. NO attention mask in the gate
# (all-ones; padding-output-zeroing is a caller concern, anima_python_ref.py).
#
# F32 path throughout (matches the oracle; clean cos, no BF16 floor). Real dims.

from max.gpu.host import DeviceContext
from std.gpu import global_idx
from std.collections import List, Dict
from std.math import sin as fsin, cos as fcos, sqrt, erf
from std.memory import ArcPointer
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer
from serenitymojo.tokenizer.t5_tokenizer import T5Tokenizer
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import rms_norm
from serenitymojo.ops.rope import rope_halfsplit
from serenitymojo.ops.attention import sdpa_nomask, sdpa
from serenitymojo.ops.tensor_algebra import reshape, add
from serenitymojo.models.text_encoder.qwen3_encoder import Qwen3Encoder

comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256

# Adapter dims (real; ckpt-verified).
comptime ADP_DIM = 1024
comptime ADP_HEADS = 16
comptime ADP_HEAD_DIM = 64
comptime ADP_BLOCKS = 6
comptime ADP_VOCAB = 32128
comptime ADP_MLP = 4096
comptime ADP_THETA = Float64(10000.0)
comptime ADP_EPS = Float32(1e-6)
comptime ANIMA_TEXT_SEQ_LEN = 512
comptime ANIMA_QWEN3_PAD_ID = 151643
comptime ANIMA_T5_PAD_ID = 0


struct AnimaTokens(Movable):
    """The three padded token arrays consumed by the frozen Anima text path."""

    var qwen_ids: List[Int]
    var qwen_mask: List[Int]
    var t5_ids: List[Int]

    def __init__(
        out self, var qwen_ids: List[Int], var qwen_mask: List[Int],
        var t5_ids: List[Int],
    ):
        self.qwen_ids = qwen_ids^
        self.qwen_mask = qwen_mask^
        self.t5_ids = t5_ids^


def tokenize_anima_text(
    text: String, qtok: Qwen3Tokenizer, t5tok: T5Tokenizer
) raises -> AnimaTokens:
    """Reference-compatible Anima tokenization at the trained 512-token length."""
    var q_raw = qtok.encode(text)
    var qwen_ids = List[Int]()
    var qwen_mask = List[Int]()
    var q_keep = len(q_raw) if len(q_raw) < ANIMA_TEXT_SEQ_LEN else ANIMA_TEXT_SEQ_LEN
    for i in range(q_keep):
        qwen_ids.append(q_raw[i])
        qwen_mask.append(1)
    while len(qwen_ids) < ANIMA_TEXT_SEQ_LEN:
        qwen_ids.append(ANIMA_QWEN3_PAD_ID)
        qwen_mask.append(0)

    var t5_raw = t5tok.encode(text)
    var t5_ids = List[Int]()
    var t_keep = len(t5_raw) if len(t5_raw) < ANIMA_TEXT_SEQ_LEN else ANIMA_TEXT_SEQ_LEN
    for i in range(t_keep):
        t5_ids.append(t5_raw[i])
    while len(t5_ids) < ANIMA_TEXT_SEQ_LEN:
        t5_ids.append(ANIMA_T5_PAD_ID)

    return AnimaTokens(qwen_ids^, qwen_mask^, t5_ids^)


# ── exact-erf GELU (matches nn.GELU(), NOT the tanh-approx ops.gelu) ──────────
def _gelu_exact_kernel_f32(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    n_w: Int64,
):
    var n = Int(n_w)
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[DType.float32]](x[i])
        var c = v * Float32(0.5) * (Float32(1.0) + erf(v / sqrt(Float32(2.0))))
        o[i] = rebind[o.element_type](c)


def gelu_exact_f32(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    """y = 0.5*x*(1+erf(x/sqrt2)). F32 only (forward, inference)."""
    if x.dtype() != STDtype.F32:
        raise Error("gelu_exact_f32: F32 only")
    var n = x.numel()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var grid = (n + _BLOCK - 1) // _BLOCK
    var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(x.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=rl,
    )
    var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=rl,
    )
    ctx.enqueue_function[_gelu_exact_kernel_f32](
        X, O, Int64(n), grid_dim=grid, block_dim=_BLOCK
    )
    return Tensor(out_buf^, x.shape(), x.dtype())


# ── embedding lookup: table [V, D], ids [S] -> [1, S, D] ──────────────────────
def _embed_kernel_f32(
    table: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    ids: LayoutTensor[DType.int32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    seq_w: Int32,
    hidden_w: Int32,
):
    var seq = Int(seq_w)
    var hidden = Int(hidden_w)
    var idx = Int(global_idx.x)
    var total = seq * hidden
    if idx < total:
        var t = idx // hidden
        var j = idx % hidden
        var tok = Int(rebind[Scalar[DType.int32]](ids[t]))
        o[idx] = rebind[o.element_type](table[tok * hidden + j])


def embed_lookup_f32(
    table: Tensor, ids: List[Int], ctx: DeviceContext
) raises -> Tensor:
    """Gather rows of F32 table [V, D] by int ids -> [1, len(ids), D]."""
    var ts = table.shape()
    var hidden = ts[len(ts) - 1]
    var seq = len(ids)
    var id_host = ctx.enqueue_create_host_buffer[DType.uint8](seq * 4)
    var ip = id_host.unsafe_ptr().bitcast[Int32]()
    for i in range(seq):
        ip[i] = Int32(ids[i])
    var id_dev = ctx.enqueue_create_buffer[DType.uint8](seq * 4)
    ctx.enqueue_copy(dst_buf=id_dev, src_buf=id_host)
    ctx.synchronize()

    var out_buf = ctx.enqueue_create_buffer[DType.uint8](seq * hidden * 4)
    var tab_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](table.numel()))
    var id_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](seq))
    var out_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](seq * hidden))
    var grid = (seq * hidden + _BLOCK - 1) // _BLOCK
    var T = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(table.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=tab_rl,
    )
    var IDS = LayoutTensor[DType.int32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.int32], MutAnyOrigin](
            unsafe_from_address=Int(id_dev.unsafe_ptr().bitcast[Int32]())
        ),
        runtime_layout=id_rl,
    )
    var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=out_rl,
    )
    ctx.enqueue_function[_embed_kernel_f32](
        T, IDS, O, Int32(seq), Int32(hidden), grid_dim=grid, block_dim=_BLOCK
    )
    ctx.synchronize()
    var sh = List[Int]()
    sh.append(1)
    sh.append(seq)
    sh.append(hidden)
    return Tensor(out_buf^, sh^, STDtype.F32)


# ── 1D RoPE cos/sin table [S*H, D/2] (position-major, head-replicated) ───────
# Matches AdapterRotaryEmbedding: inv_freq=1/theta^(2i/D); emb=cat(freqs,freqs);
# applied as half-split rotate (rope_halfsplit consumes the first-half table).
struct _RopePair(Movable):
    var cos: Tensor
    var sin: Tensor

    def __init__(out self, var cos: Tensor, var sin: Tensor):
        self.cos = cos^
        self.sin = sin^


def _rope_pair(
    seq: Int, heads: Int, head_dim: Int, ctx: DeviceContext
) raises -> _RopePair:
    var half = head_dim // 2
    var cos_data = List[Float32]()
    var sin_data = List[Float32]()
    var freqs = List[Float64]()
    for i in range(half):
        freqs.append(1.0 / (ADP_THETA ** (Float64(2 * i) / Float64(head_dim))))
    for pos in range(seq):
        for _h in range(heads):
            for i in range(half):
                var angle = Float64(pos) * freqs[i]
                cos_data.append(Float32(fcos(angle)))
                sin_data.append(Float32(fsin(angle)))
    var sh = List[Int]()
    sh.append(seq * heads)
    sh.append(half)
    var cos_t = Tensor.from_host(cos_data^, sh.copy(), STDtype.F32, ctx)
    var sin_t = Tensor.from_host(sin_data^, sh^, STDtype.F32, ctx)
    return _RopePair(cos_t^, sin_t^)


# ── adapter weights holder ───────────────────────────────────────────────────
struct AnimaAdapterWeights(Movable):
    """All net.llm_adapter weights as F32 GPU Tensors, looked up by flat name
    (e.g. 'blocks.0.cross_attn.q_proj.weight', 'embed.weight',
    'out_proj.weight', 'out_proj.bias', 'norm.weight')."""

    var w: Dict[String, ArcPointer[Tensor]]

    def __init__(out self, var w: Dict[String, ArcPointer[Tensor]]):
        self.w = w^

    def get(self, name: String) raises -> ref [self.w[String("")]] Tensor:
        if name not in self.w:
            raise Error(String("adapter: missing weight ") + name)
        return self.w[name][]

    @staticmethod
    def load_checkpoint(path: String, ctx: DeviceContext) raises -> AnimaAdapterWeights:
        """Load all `net.llm_adapter.*` tensors from a single-file checkpoint as
        F32 GPU tensors, keyed by the flat sub-name (prefix stripped).

        This adapter is an F32 text-context oracle path; its output is a frozen
        context cache, not diffusion-model weight storage."""
        var st = ShardedSafeTensors.open(path)
        var w = Dict[String, ArcPointer[Tensor]]()
        var prefix = String("net.llm_adapter.")
        var found = 0
        var plen = prefix.byte_length()
        for ref nm in st.names():
            if nm.startswith(prefix):
                # strip the prefix (byte-wise; std slice API differs by build)
                var sub = String("")
                for i in range(plen, nm.byte_length()):
                    sub += String(nm[byte=i])
                var tv = st.tensor_view(nm)
                var t = Tensor.from_view_as_f32(tv, ctx)
                w[sub] = ArcPointer(t^)
                found += 1
        if found == 0:
            raise Error(String("no net.llm_adapter.* tensors in ") + path)
        return AnimaAdapterWeights(w^)


# ── attention sub-block (S_q == S_k == 512 for both self & cross) ─────────────
def _attn(
    x: Tensor,          # [1, S_q, 1024] (query source, RMSNormed)
    context: Tensor,    # [1, S_k, 1024] (k/v source)
    wts: AnimaAdapterWeights,
    prefix: String,     # e.g. 'blocks.0.self_attn'
    rope_q: _RopePair,
    rope_k: _RopePair,
    key_mask: Tensor,   # additive [1,H,S,S] (0 keep / -1e9 pad key) — reference trainer attn mask
    ctx: DeviceContext,
) raises -> Tensor:
    var H = ADP_HEADS
    var Dh = ADP_HEAD_DIM
    var scale = Float32(1.0) / sqrt(Float32(Dh))

    var xs = x.shape()
    var sq = xs[1]
    var cs = context.shape()
    var sk = cs[1]

    var q = linear(x, wts.get(prefix + ".q_proj.weight"), None, ctx)
    var k = linear(context, wts.get(prefix + ".k_proj.weight"), None, ctx)
    var v = linear(context, wts.get(prefix + ".v_proj.weight"), None, ctx)

    # reshape to BSHD
    var q_sh = List[Int]()
    q_sh.append(1); q_sh.append(sq); q_sh.append(H); q_sh.append(Dh)
    q = reshape(q, q_sh^, ctx)
    var k_sh = List[Int]()
    k_sh.append(1); k_sh.append(sk); k_sh.append(H); k_sh.append(Dh)
    k = reshape(k, k_sh^, ctx)
    var v_sh = List[Int]()
    v_sh.append(1); v_sh.append(sk); v_sh.append(H); v_sh.append(Dh)
    v = reshape(v, v_sh^, ctx)

    # per-head QK RMSNorm (weight [Dh]); rms_norm flattens leading dims.
    q = rms_norm(q, wts.get(prefix + ".q_norm.weight"), ADP_EPS, ctx)
    k = rms_norm(k, wts.get(prefix + ".k_norm.weight"), ADP_EPS, ctx)

    # half-split RoPE: cos/sin [S*H, Dh/2] ordered (pos, head).
    q = rope_halfsplit(q, rope_q.cos, rope_q.sin, ctx)
    k = rope_halfsplit(k, rope_k.cos, rope_k.sin, ctx)

    # Masked SDPA (BSHD, S_q==S_k==512). key_mask zeros padding keys so the
    # softmax never attends to Qwen3/T5 pad positions (reference trainer AnimaTextConditioner
    # source/target_attention_mask). Comptime B=1,S=512,H=16,Dh=64.
    var attn = sdpa[1, 512, ADP_HEADS, ADP_HEAD_DIM](
        q, k, v, key_mask, scale, ctx
    )
    var a_sh = List[Int]()
    a_sh.append(1); a_sh.append(sq); a_sh.append(H * Dh)
    attn = reshape(attn, a_sh^, ctx)
    return linear(attn, wts.get(prefix + ".o_proj.weight"), None, ctx)


# ── one adapter block ────────────────────────────────────────────────────────
def _block(
    x: Tensor,
    context: Tensor,
    wts: AnimaAdapterWeights,
    j: Int,
    rope_q: _RopePair,
    rope_k: _RopePair,
    self_mask: Tensor,   # additive [1,H,S,S] keyed by TARGET (T5) padding
    cross_mask: Tensor,  # additive [1,H,S,S] keyed by SOURCE (Qwen3) padding
    ctx: DeviceContext,
) raises -> Tensor:
    var bp = String("blocks.") + String(j)

    # self-attn (q,k both target positions) — mask T5 padding keys
    var n1 = rms_norm(x, wts.get(bp + ".norm_self_attn.weight"), ADP_EPS, ctx)
    var sa = _attn(n1, n1, wts, bp + ".self_attn", rope_q, rope_q, self_mask, ctx)
    var x1 = add(x, sa, ctx)

    # cross-attn (q target pos, k context pos) — mask Qwen3 padding keys
    var n2 = rms_norm(x1, wts.get(bp + ".norm_cross_attn.weight"), ADP_EPS, ctx)
    var ca = _attn(n2, context, wts, bp + ".cross_attn", rope_q, rope_k, cross_mask, ctx)
    var x2 = add(x1, ca, ctx)

    # MLP (bias, exact GELU)
    var n3 = rms_norm(x2, wts.get(bp + ".norm_mlp.weight"), ADP_EPS, ctx)
    var b0 = Optional[Tensor](wts.get(bp + ".mlp.0.bias").clone(ctx))
    var h = linear(n3, wts.get(bp + ".mlp.0.weight"), b0^, ctx)
    h = gelu_exact_f32(h, ctx)
    var b2 = Optional[Tensor](wts.get(bp + ".mlp.2.bias").clone(ctx))
    var mlp_out = linear(h, wts.get(bp + ".mlp.2.weight"), b2^, ctx)
    return add(x2, mlp_out, ctx)


# ── zero padding positions: hidden [1,S,D] *= mask[s] (mask 0/1) ─────────────
def _zero_pad_kernel_f32(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    mask: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    seq_w: Int32,
    hidden_w: Int32,
):
    var seq = Int(seq_w)
    var hidden = Int(hidden_w)
    var idx = Int(global_idx.x)
    var total = seq * hidden
    if idx < total:
        var s = idx // hidden
        var m = rebind[Scalar[DType.float32]](mask[s])
        o[idx] = rebind[o.element_type](rebind[Scalar[DType.float32]](x[idx]) * m)


def zero_pad_positions_f32(
    hidden: Tensor, attn_mask: List[Int], ctx: DeviceContext
) raises -> Tensor:
    """Zero out padding positions: out[0,s,:] = hidden[0,s,:] * mask[s].
    Mirrors AnimaModel.py:218 `qwen_hidden * tokens_mask.unsqueeze(-1)`."""
    if hidden.dtype() != STDtype.F32:
        raise Error("zero_pad_positions_f32: F32 only")
    var hs = hidden.shape()
    var seq = hs[1]
    var d = hs[2]
    if len(attn_mask) != seq:
        raise Error("zero_pad_positions_f32: mask len != seq")
    var mvals = List[Float32]()
    for i in range(seq):
        mvals.append(Float32(attn_mask[i]))
    var msh = List[Int]()
    msh.append(seq)
    var mask_t = Tensor.from_host(mvals^, msh^, STDtype.F32, ctx)

    var out_buf = ctx.enqueue_create_buffer[DType.uint8](hidden.nbytes())
    var x_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](seq * d))
    var m_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](seq))
    var grid = (seq * d + _BLOCK - 1) // _BLOCK
    var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(hidden.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=x_rl,
    )
    var M = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(mask_t.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=m_rl,
    )
    var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=x_rl,
    )
    ctx.enqueue_function[_zero_pad_kernel_f32](
        X, M, O, Int32(seq), Int32(d), grid_dim=grid, block_dim=_BLOCK
    )
    ctx.synchronize()
    return Tensor(out_buf^, hidden.shape(), STDtype.F32)


# ── additive key-mask [1,H,S,S] from a 0/1 mask list ─────────────────────────
# o[.,.,.,sk] = 0 if mask01[sk]!=0 else -1e9  (broadcast over query+head).
def _keymask_kernel_f32(
    m: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],   # [S] additive values
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],   # [H*S*S]
    S_w: Int32,
    total_w: Int64,
):
    var S = Int(S_w)
    var total = Int(total_w)
    var idx = Int(global_idx.x)
    if idx < total:
        var sk = idx % S
        o[idx] = m[sk]

def _build_keymask(mask01: List[Int], S: Int, ctx: DeviceContext) raises -> Tensor:
    """Additive attention mask [1,H,S,S] F32: 0 where mask01[sk]==1, -1e9 (pad)."""
    var H = ADP_HEADS
    var mvals = List[Float32]()
    for i in range(S):
        mvals.append(Float32(0.0) if mask01[i] != 0 else Float32(-1.0e9))
    var msh = List[Int]()
    msh.append(S)
    var m_t = Tensor.from_host(mvals^, msh^, STDtype.F32, ctx)

    var total = H * S * S
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](total * 4)
    var m_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](S))
    var o_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](total))
    var M = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(m_t.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=m_rl,
    )
    var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=o_rl,
    )
    var grid = (total + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_keymask_kernel_f32](
        M, O, Int32(S), Int64(total), grid_dim=grid, block_dim=_BLOCK
    )
    ctx.synchronize()
    var osh = List[Int]()
    osh.append(1); osh.append(H); osh.append(S); osh.append(S)
    return Tensor(out_buf^, osh^, STDtype.F32)


# ── full adapter forward ─────────────────────────────────────────────────────
def anima_llm_adapter_forward(
    t5_ids: List[Int],     # [S_TXT] T5 token ids (queries; pad id 0)
    qwen_hidden: Tensor,   # [1, S_LLM, 1024] F32 (Qwen3 last_hidden_state, pad-zeroed)
    qwen_mask: List[Int],  # [S_LLM] 0/1 Qwen3 attention mask (SOURCE keys)
    wts: AnimaAdapterWeights,
    ctx: DeviceContext,
) raises -> Tensor:
    """Run net.llm_adapter -> context [1, S_TXT, 1024] (F32). FORWARD ONLY.

    reference trainer AnimaTextConditioner: self-attn masks T5 padding (target), cross-attn masks
    Qwen3 padding (source), and padding context positions are zeroed at the output.
    """
    var s_txt = len(t5_ids)
    var qs = qwen_hidden.shape()
    var s_llm = qs[1]

    # target (T5) 0/1 mask: pad id is 0. Build from t5_ids.
    var t5_mask = List[Int]()
    for i in range(s_txt):
        t5_mask.append(1 if t5_ids[i] != 0 else 0)

    var self_mask = _build_keymask(t5_mask, s_txt, ctx)      # target keys (S_TXT)
    var cross_mask = _build_keymask(qwen_mask, s_llm, ctx)   # source keys (S_LLM)

    var x = embed_lookup_f32(wts.get(String("embed.weight")), t5_ids, ctx)
    var rope_q = _rope_pair(s_txt, ADP_HEADS, ADP_HEAD_DIM, ctx)
    var rope_k = _rope_pair(s_llm, ADP_HEADS, ADP_HEAD_DIM, ctx)

    for j in range(ADP_BLOCKS):
        x = _block(x, qwen_hidden, wts, j, rope_q, rope_k, self_mask, cross_mask, ctx)

    var ob = Optional[Tensor](wts.get(String("out_proj.bias")).clone(ctx))
    x = linear(x, wts.get(String("out_proj.weight")), ob^, ctx)
    x = rms_norm(x, wts.get(String("norm.weight")), ADP_EPS, ctx)
    # reference trainer zeros padding context positions (target padding) at the output.
    x = zero_pad_positions_f32(x, t5_mask, ctx)
    return x^


def encode_anima_context(
    tokens: AnimaTokens, enc: Qwen3Encoder, wts: AnimaAdapterWeights,
    ctx: DeviceContext,
) raises -> Tensor:
    """Qwen3 + Anima adapter -> frozen [1,512,1024] F32 context tensor."""
    var n_layers = enc.config.num_layers
    var pre_norm = enc.encode(tokens.qwen_ids, n_layers - 1, ctx)
    var last_hidden = enc.final_norm(pre_norm, ctx)
    var hidden_f32 = cast_tensor(last_hidden, STDtype.F32, ctx)
    var hidden_zeroed = zero_pad_positions_f32(hidden_f32, tokens.qwen_mask, ctx)
    return anima_llm_adapter_forward(
        tokens.t5_ids, hidden_zeroed, tokens.qwen_mask, wts, ctx
    )


def encode_anima_context_host(
    var tokens: AnimaTokens, enc: Qwen3Encoder, wts: AnimaAdapterWeights,
    ctx: DeviceContext,
) raises -> List[Float32]:
    """Host-list twin used by the inference backend's denoising state."""
    var context = encode_anima_context(tokens^, enc, wts, ctx)
    return context.to_host(ctx)

# models/text_encoder/mistral3b_encoder.mojo — ERNIE Mistral-3B text encoder.
#
# Pure-Mojo port of:
#   /home/alex/EriDiffusion/inference-flame/src/models/mistral3b_encoder.rs
#
# Contract:
# - tokenizer ids come from ERNIE tokenizer.json with TemplateProcessing `<s> A`;
#   helper `mistral3_tokenize` prepends token id 1.
# - encoder pads/truncates to max_len with token id 0, matching the Rust encoder.
# - returns output after layer 24 (hidden_states[-2]) as `[1,max_len,3072]`.

from std.collections import Dict, Optional
from std.gpu.host import DeviceContext
from std.utils.index import IndexList
from std.math import sqrt, cos as fcos, sin as fsin, exp as fexp, log as flog
from std.memory import ArcPointer
from layout import LayoutTensor
from layout.runtime_layout import RuntimeLayout

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.norm import rms_norm
from serenitymojo.ops.rope import rope_halfsplit
from serenitymojo.ops.attention import sdpa
from serenitymojo.ops.linear import linear
from serenitymojo.ops.activations import swiglu
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer
from serenitymojo.models.text_encoder.qwen3_encoder import (
    _add,
    _repeat_kv,
    _reshape,
    _clone,
    _embed_kernel_f32,
    _embed_kernel_bf16,
    _embed_kernel_f16,
    _DYN1,
    _BLOCK,
)


@fieldwise_init
struct Mistral3bConfig(Copyable, Movable, ImplicitlyCopyable):
    var vocab_size: Int
    var hidden_size: Int
    var num_layers: Int
    var intermediate_size: Int
    var num_heads: Int
    var num_kv_heads: Int
    var head_dim: Int
    var rms_norm_eps: Float32
    var rope_theta: Float64
    var rope_factor: Float64
    var rope_original_max_pos: Int
    var rope_beta_fast: Float64
    var rope_beta_slow: Float64
    var extract_layer: Int
    var max_seq_len: Int

    @staticmethod
    def ernie() -> Mistral3bConfig:
        return Mistral3bConfig(
            131072,
            3072,
            26,
            9216,
            32,
            8,
            128,
            Float32(1.0e-5),
            Float64(1000000.0),
            Float64(16.0),
            16384,
            Float64(32.0),
            Float64(1.0),
            24,
            512,
        )


def mistral3_tokenize(tok: Qwen3Tokenizer, text: String) raises -> List[Int]:
    """Tokenize ERNIE/Mistral text like tokenizers.encode(text, true).

    ERNIE tokenizer.json post_processor is TemplateProcessing("<s> $A"). The
    shared BPE tokenizer parser intentionally returns the model tokens only, so
    this helper prepends BOS id 1.
    """
    var ids = List[Int]()
    ids.append(1)
    var body = tok.encode(text)
    for i in range(len(body)):
        ids.append(body[i])
    return ids^


def mistral3_pad_or_trim(ids_in: List[Int], max_len: Int) -> Tuple[List[Int], Int]:
    var ids = List[Int]()
    var real_len = len(ids_in)
    if real_len > max_len:
        real_len = max_len
    for i in range(real_len):
        ids.append(ids_in[i])
    while len(ids) < max_len:
        ids.append(0)
    return (ids^, real_len)


def _build_yarn_rope_tables(
    seq: Int,
    heads: Int,
    head_dim: Int,
    theta: Float64,
    factor: Float64,
    original_max_pos: Int,
    beta_fast: Float64,
    beta_slow: Float64,
) raises -> List[List[Float32]]:
    var half = head_dim // 2
    var orig = Float64(original_max_pos)
    var two_pi = Float64(6.2831853071795864769)
    var low_freq_factor = (orig / (two_pi / beta_fast))
    var high_freq_factor = (orig / (two_pi / beta_slow))
    # Rust rounds these values. Mojo lacks Float64.round in this toolchain; the
    # factors are positive, so +0.5 then Int gives the same integer.
    low_freq_factor = Float64(Int(low_freq_factor + 0.5))
    high_freq_factor = Float64(Int(high_freq_factor + 0.5))
    if low_freq_factor < 1.0:
        low_freq_factor = 1.0
    if high_freq_factor < 1.0:
        high_freq_factor = 1.0
    var low_freq_wavelen = orig / low_freq_factor
    var high_freq_wavelen = orig / high_freq_factor

    var inv = List[Float32]()
    for i in range(half):
        var base_freq = Float64(1.0) / (theta ** (Float64(2 * i) / Float64(head_dim)))
        var wavelength = two_pi / base_freq
        var scaled: Float64
        if wavelength < high_freq_wavelen:
            scaled = base_freq
        elif wavelength > low_freq_wavelen:
            scaled = base_freq / factor
        else:
            var smooth = (orig / wavelength - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            scaled = (Float64(1.0) - smooth) * (base_freq / factor) + smooth * base_freq
        inv.append(Float32(scaled))

    var cos_vals = List[Float32]()
    var sin_vals = List[Float32]()
    for t in range(seq):
        for _h in range(heads):
            for i in range(half):
                var angle = Float32(t) * inv[i]
                cos_vals.append(fcos(angle))
                sin_vals.append(fsin(angle))
    var out = List[List[Float32]]()
    out.append(cos_vals^)
    out.append(sin_vals^)
    return out^


def _build_causal_mask(seq: Int, heads: Int, real_len: Int) raises -> List[Float32]:
    var neg = Float32(-1.0e4)
    var data = List[Float32]()
    for _hh in range(heads):
        for i in range(seq):
            for j in range(seq):
                if j <= i and j < real_len:
                    data.append(Float32(0.0))
                else:
                    data.append(neg)
    return data^


struct Mistral3bEncoder:
    var weights: List[ArcPointer[Tensor]]
    var name_to_idx: Dict[String, Int]
    var config: Mistral3bConfig

    def __init__(
        out self,
        var weights: List[ArcPointer[Tensor]],
        var name_to_idx: Dict[String, Int],
        config: Mistral3bConfig,
    ):
        self.weights = weights^
        self.name_to_idx = name_to_idx^
        self.config = config

    @staticmethod
    def load(dir_or_file: String, ctx: DeviceContext) raises -> Mistral3bEncoder:
        """Load the ERNIE text_encoder safetensors.

        `dir_or_file` may be `/home/alex/models/ERNIE-Image/text_encoder` or the
        concrete `model.safetensors` path.
        """
        var path = dir_or_file
        if path.endswith(String(".safetensors")):
            # ShardedSafeTensors expects a directory. Strip `/model.safetensors`.
            path = String(path.removesuffix(String("/model.safetensors")))
        var st = ShardedSafeTensors.open(path)
        var weights = List[ArcPointer[Tensor]]()
        var name_to_idx = Dict[String, Int]()
        for ref nm in st.names():
            # Only keep the language model; skip vision_tower and projector keys.
            if not nm.startswith(String("language_model.model.")):
                continue
            var key = String(nm.removeprefix(String("language_model.model.")))
            if key.startswith(String("vision_tower.")) or key.startswith(String("multi_modal_projector.")):
                continue
            var tv = st.tensor_view(nm)
            var t = Tensor.from_view(tv, ctx)
            var idx = len(weights)
            weights.append(ArcPointer(t^))
            name_to_idx[key] = idx
        if String("embed_tokens.weight") not in name_to_idx:
            raise Error("Mistral3bEncoder.load: missing embed_tokens.weight")
        if String("norm.weight") not in name_to_idx:
            raise Error("Mistral3bEncoder.load: missing norm.weight")
        return Mistral3bEncoder(weights^, name_to_idx^, Mistral3bConfig.ernie())

    def _w(self, name: String) raises -> ref [self.weights] Tensor:
        if name not in self.name_to_idx:
            raise Error(String("missing Mistral3B weight: ") + name)
        return self.weights[self.name_to_idx[name]][]

    def _embed(self, ids: List[Int], ctx: DeviceContext) raises -> Tensor:
        ref table = self._w(String("embed_tokens.weight"))
        var seq = len(ids)
        var hidden = self.config.hidden_size
        var dt = table.dtype().to_mojo_dtype()

        var id_host = ctx.enqueue_create_host_buffer[DType.uint8](seq * 4)
        var ip = id_host.unsafe_ptr().bitcast[Int32]()
        for i in range(seq):
            ip[i] = Int32(ids[i])
        var id_dev = ctx.enqueue_create_buffer[DType.uint8](seq * 4)
        ctx.enqueue_copy(dst_buf=id_dev, src_buf=id_host)
        ctx.synchronize()

        var out_buf = ctx.enqueue_create_buffer[DType.uint8](
            seq * hidden * table.dtype().byte_size()
        )
        var tab_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](table.numel()))
        var id_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](seq))
        var out_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](seq * hidden))
        var total = seq * hidden
        var grid = (total + _BLOCK - 1) // _BLOCK
        var IDS = LayoutTensor[DType.int32, _DYN1, MutAnyOrigin](
            id_dev.unsafe_ptr().bitcast[Int32](), id_rl
        )
        if dt == DType.float32:
            var T = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
                table.buf.unsafe_ptr().bitcast[Float32](), tab_rl
            )
            var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
                out_buf.unsafe_ptr().bitcast[Float32](), out_rl
            )
            ctx.enqueue_function[_embed_kernel_f32, _embed_kernel_f32](
                T, IDS, O, seq, hidden, grid_dim=grid, block_dim=_BLOCK
            )
        elif dt == DType.bfloat16:
            var T = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
                table.buf.unsafe_ptr().bitcast[BFloat16](), tab_rl
            )
            var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
                out_buf.unsafe_ptr().bitcast[BFloat16](), out_rl
            )
            ctx.enqueue_function[_embed_kernel_bf16, _embed_kernel_bf16](
                T, IDS, O, seq, hidden, grid_dim=grid, block_dim=_BLOCK
            )
        else:
            var T = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
                table.buf.unsafe_ptr().bitcast[Float16](), tab_rl
            )
            var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
                out_buf.unsafe_ptr().bitcast[Float16](), out_rl
            )
            ctx.enqueue_function[_embed_kernel_f16, _embed_kernel_f16](
                T, IDS, O, seq, hidden, grid_dim=grid, block_dim=_BLOCK
            )
        ctx.synchronize()
        var sh = List[Int]()
        sh.append(1)
        sh.append(seq)
        sh.append(hidden)
        return Tensor(out_buf^, sh^, table.dtype())

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
        var scale = Float32(1.0) / sqrt(Float32(dh))
        var p = String("layers.") + String(layer_idx)
        var hs = hidden.shape()
        var seq = hs[1]

        ref in_ln = self._w(p + String(".input_layernorm.weight"))
        var normed = rms_norm(hidden, in_ln, cfg.rms_norm_eps, ctx)

        var q = linear(normed, self._w(p + String(".self_attn.q_proj.weight")), None, ctx)
        var k = linear(normed, self._w(p + String(".self_attn.k_proj.weight")), None, ctx)
        var v = linear(normed, self._w(p + String(".self_attn.v_proj.weight")), None, ctx)

        var q_sh = List[Int]()
        q_sh.append(1)
        q_sh.append(seq)
        q_sh.append(h)
        q_sh.append(dh)
        q = _reshape(q, q_sh^, ctx)
        var k_sh = List[Int]()
        k_sh.append(1)
        k_sh.append(seq)
        k_sh.append(h_kv)
        k_sh.append(dh)
        k = _reshape(k, k_sh^, ctx)
        var v_sh = List[Int]()
        v_sh.append(1)
        v_sh.append(seq)
        v_sh.append(h_kv)
        v_sh.append(dh)
        v = _reshape(v, v_sh^, ctx)

        q = rope_halfsplit(q, cos_q, sin_q, ctx)
        k = rope_halfsplit(k, cos_k, sin_k, ctx)
        var k_rep = _repeat_kv(k^, h, h_kv, ctx)
        var v_rep = _repeat_kv(v^, h, h_kv, ctx)

        var attn = _sdpa_mistral_dispatch(q, k_rep, v_rep, mask, scale, seq, h, dh, ctx)
        var attn_sh = List[Int]()
        attn_sh.append(1)
        attn_sh.append(seq)
        attn_sh.append(h * dh)
        attn = _reshape(attn, attn_sh^, ctx)
        var attn_out = linear(
            attn,
            self._w(p + String(".self_attn.o_proj.weight")),
            None,
            ctx,
        )
        var hidden2 = _add(hidden, attn_out, ctx)

        var normed2 = rms_norm(
            hidden2,
            self._w(p + String(".post_attention_layernorm.weight")),
            cfg.rms_norm_eps,
            ctx,
        )
        var gate = linear(normed2, self._w(p + String(".mlp.gate_proj.weight")), None, ctx)
        var up = linear(normed2, self._w(p + String(".mlp.up_proj.weight")), None, ctx)
        var act = swiglu(gate, up, ctx)
        var mlp_out = linear(act, self._w(p + String(".mlp.down_proj.weight")), None, ctx)
        return _add(hidden2, mlp_out, ctx)

    def encode(
        self, token_ids: List[Int], max_len: Int, ctx: DeviceContext
    ) raises -> Tuple[Tensor, Int]:
        var cfg = self.config
        var padded = mistral3_pad_or_trim(token_ids, max_len)
        var ids = padded[0].copy()
        var real_len = padded[1]
        var seq = len(ids)

        var dtype = self._w(String("embed_tokens.weight")).dtype()
        var q_tables = _build_yarn_rope_tables(
            seq,
            cfg.num_heads,
            cfg.head_dim,
            cfg.rope_theta,
            cfg.rope_factor,
            cfg.rope_original_max_pos,
            cfg.rope_beta_fast,
            cfg.rope_beta_slow,
        )
        var k_tables = _build_yarn_rope_tables(
            seq,
            cfg.num_kv_heads,
            cfg.head_dim,
            cfg.rope_theta,
            cfg.rope_factor,
            cfg.rope_original_max_pos,
            cfg.rope_beta_fast,
            cfg.rope_beta_slow,
        )
        var half = cfg.head_dim // 2
        var q_tab_sh = List[Int]()
        q_tab_sh.append(seq * cfg.num_heads * half)
        var k_tab_sh = List[Int]()
        k_tab_sh.append(seq * cfg.num_kv_heads * half)
        var cos_q = Tensor.from_host(q_tables[0], q_tab_sh.copy(), dtype, ctx)
        var sin_q = Tensor.from_host(q_tables[1], q_tab_sh.copy(), dtype, ctx)
        var cos_k = Tensor.from_host(k_tables[0], k_tab_sh.copy(), dtype, ctx)
        var sin_k = Tensor.from_host(k_tables[1], k_tab_sh.copy(), dtype, ctx)

        var mask_data = _build_causal_mask(seq, cfg.num_heads, real_len)
        var mask_sh = List[Int]()
        mask_sh.append(1)
        mask_sh.append(cfg.num_heads)
        mask_sh.append(seq)
        mask_sh.append(seq)
        var mask = Tensor.from_host(mask_data, mask_sh^, dtype, ctx)

        var hidden = self._embed(ids, ctx)
        var result = _clone(hidden, ctx)
        var got_result = False
        for i in range(cfg.num_layers):
            hidden = self._layer(i, hidden, cos_q, sin_q, cos_k, sin_k, mask, ctx)
            if i == cfg.extract_layer:
                result = _clone(hidden, ctx)
                got_result = True
        if not got_result:
            raise Error("Mistral3bEncoder.encode: extract layer not reached")
        return (result^, real_len)


def _sdpa_mistral_dispatch(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor,
    scale: Float32,
    seq: Int,
    h: Int,
    dh: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    if h == 32 and dh == 128:
        if seq == 256:
            return sdpa[1, 256, 32, 128](q, k, v, mask, scale, ctx)
        if seq == 512:
            return sdpa[1, 512, 32, 128](q, k, v, mask, scale, ctx)
        if seq == 128:
            return sdpa[1, 128, 32, 128](q, k, v, mask, scale, ctx)
        if seq == 64:
            return sdpa[1, 64, 32, 128](q, k, v, mask, scale, ctx)
        if seq == 32:
            return sdpa[1, 32, 32, 128](q, k, v, mask, scale, ctx)
        if seq == 16:
            return sdpa[1, 16, 32, 128](q, k, v, mask, scale, ctx)
        if seq == 8:
            return sdpa[1, 8, 32, 128](q, k, v, mask, scale, ctx)
    raise Error(
        String("Mistral3b sdpa dispatch unsupported (seq,h,dh)=(")
        + String(seq) + String(",") + String(h) + String(",") + String(dh) + String(")")
    )

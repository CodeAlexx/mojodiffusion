# models/lance/lance_t2v.mojo — Lance 3B Video T2V spine, streamed.
#
# Source of truth:
#   /home/alex/Lance/modeling/lance/lance.py::validation_gen
#   /home/alex/Lance/modeling/lance/qwen2_navit.py
#   /home/alex/Lance/data/datasets_custom/validation_dataset.py::t2v_sample
#
# This is the first native Mojo Lance port slice: text_template=false T2V,
# visual_gen only, one sample, layer streaming from model.safetensors. Full-size
# Lance still needs a block-sparse/flex attention kernel before 480p/768p video
# is practical; this module intentionally starts with a tiny dense-mask smoke so
# real weights can execute on GPU while the streaming and MoE-gen path harden.

from std.gpu.host import DeviceContext
from std.gpu import global_idx
from std.math import sqrt, exp as fexp, log as flog, cos as fcos, sin as fsin
from std.memory import ArcPointer
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import rms_norm
from serenitymojo.ops.activations import swiglu, silu
from serenitymojo.ops.rope import rope_halfsplit_full
from serenitymojo.ops.attention import sdpa
from serenitymojo.ops.tensor_algebra import (
    add,
    concat,
    gather_rows,
    reshape,
    slice as ts_slice,
)
from serenitymojo.ops.embeddings import t_embedder
from serenitymojo.offload.block_loader import Block
from serenitymojo.offload.plan import OffloadConfig, build_lance_t2v_block_plan
from serenitymojo.offload.planned_loader import PlannedBlockLoader
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer


comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256


@fieldwise_init
struct LanceT2VConfig(Copyable, Movable, ImplicitlyCopyable):
    var hidden_size: Int
    var num_layers: Int
    var num_heads: Int
    var num_kv_heads: Int
    var head_dim: Int
    var intermediate_size: Int
    var rope_theta: Float64
    var rms_norm_eps: Float32
    var vocab_size: Int
    var timestep_dim: Int
    var patch_latent_dim: Int
    var max_latent_size: Int
    var latent_downsample_t: Int
    var latent_downsample_s: Int
    var mrope_t: Int
    var mrope_h: Int
    var mrope_w: Int
    var tokens_per_second: Int
    var bos_token_id: Int
    var eos_token_id: Int
    var start_of_image_id: Int
    var end_of_image_id: Int
    var video_token_id: Int

    @staticmethod
    def lance_3b_video() -> LanceT2VConfig:
        return LanceT2VConfig(
            2048, 36, 16, 2, 128, 11008,
            Float64(1_000_000.0), Float32(1.0e-6), 151936,
            256, 48, 64, 4, 16,
            16, 24, 24, 2,
            151644, 151645, 151652, 151653, 151656,
        )


@fieldwise_init
struct LanceT2VInput(Movable):
    var full_ids: List[Int]
    var latent_pos_ids: List[Int]
    var t_pos: List[Int]
    var h_pos: List[Int]
    var w_pos: List[Int]
    var text_split_len: Int
    var gen_start: Int
    var gen_len: Int


def _append_key(mut keys: List[String], name: String):
    keys.append(name)


def lance_shared_keys() -> List[String]:
    var keys = List[String]()
    _append_key(keys, String("language_model.model.embed_tokens.weight"))
    _append_key(keys, String("language_model.model.norm.weight"))
    _append_key(keys, String("language_model.model.norm_moe_gen.weight"))
    _append_key(keys, String("vae2llm.weight"))
    _append_key(keys, String("vae2llm.bias"))
    _append_key(keys, String("llm2vae.weight"))
    _append_key(keys, String("llm2vae.bias"))
    _append_key(keys, String("time_embedder.mlp.0.weight"))
    _append_key(keys, String("time_embedder.mlp.0.bias"))
    _append_key(keys, String("time_embedder.mlp.2.weight"))
    _append_key(keys, String("time_embedder.mlp.2.bias"))
    _append_key(keys, String("latent_pos_embed.pos_embed"))
    return keys^


def _clone(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var dev = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    ctx.enqueue_copy(dst_buf=dev, src_buf=x.buf)
    ctx.synchronize()
    return Tensor(dev^, x.shape(), x.dtype())


def _shape2(a: Int, b: Int) -> List[Int]:
    var sh = List[Int]()
    sh.append(a)
    sh.append(b)
    return sh^


def _shape3(a: Int, b: Int, c: Int) -> List[Int]:
    var sh = List[Int]()
    sh.append(a)
    sh.append(b)
    sh.append(c)
    return sh^


def _shape4(a: Int, b: Int, c: Int, d: Int) -> List[Int]:
    var sh = List[Int]()
    sh.append(a)
    sh.append(b)
    sh.append(c)
    sh.append(d)
    return sh^


def _replace_span(
    x: Tensor, repl: Tensor, start: Int, length: Int, ctx: DeviceContext
) raises -> Tensor:
    """Replace rows [start, start+length) along dim=1 in x [1,S,D]."""
    var xs = x.shape()
    var s = xs[1]
    if start < 0 or length < 0 or start + length > s:
        raise Error("_replace_span: span out of bounds")
    if start == 0:
        if length == s:
            return _clone(repl, ctx)
        var tail = ts_slice(x, 1, length, s - length, ctx)
        return concat(1, ctx, repl, tail)
    if start + length == s:
        var head = ts_slice(x, 1, 0, start, ctx)
        return concat(1, ctx, head, repl)
    var head = ts_slice(x, 1, 0, start, ctx)
    var tail = ts_slice(x, 1, start + length, s - start - length, ctx)
    return concat(1, ctx, head, repl, tail)


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
    if x.dtype() != STDtype.F32:
        raise Error("Lance _repeat_kv: F32 checkpoint path expected")
    var h = h_kv * n_rep
    var out_n = s * h * dh
    var src_n = s * h_kv * dh
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](out_n * 4)
    var src_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](src_n))
    var out_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](out_n))
    var grid = (out_n + _BLOCK - 1) // _BLOCK
    var SRC = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        x.buf.unsafe_ptr().bitcast[Float32](), src_rl
    )
    var DST = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[Float32](), out_rl
    )
    ctx.enqueue_function[_repeat_kv_kernel_f32, _repeat_kv_kernel_f32](
        SRC, DST, s, h, h_kv, dh, n_rep, grid_dim=grid, block_dim=_BLOCK
    )
    ctx.synchronize()
    return Tensor(out_buf^, _shape4(1, s, h, dh), STDtype.F32)


def _sdpa_lance[S: Int](
    q: Tensor, k: Tensor, v: Tensor, mask: Tensor, scale: Float32, ctx: DeviceContext
) raises -> Tensor:
    return sdpa[1, S, 16, 128](q, k, v, mask, scale, ctx)


def _build_t2v_mask(
    seq: Int, heads: Int, text_split_len: Int
) raises -> List[Float32]:
    # create_sparse_mask for split_lens=[text_split_len, L+2],
    # attn_modes=["causal","noise"]: text rows are causal; video/noise rows
    # attend to all text and all video rows.
    var neg = Float32(-1.0e4)
    var out = List[Float32]()
    for _h in range(heads):
        for q in range(seq):
            var q_noise = q >= text_split_len
            for k in range(seq):
                var k_noise = k >= text_split_len
                var allow = False
                if q_noise:
                    allow = True
                elif not k_noise and k <= q:
                    allow = True
                if allow:
                    out.append(Float32(0.0))
                else:
                    out.append(neg)
    return out^


def _axis_full(axis_vals: List[Float32], d: Int, half: Int) -> Float32:
    var j = d
    if j >= half:
        j -= half
    return axis_vals[j]


def _build_mrope_tables(
    t_pos: List[Int],
    h_pos: List[Int],
    w_pos: List[Int],
    heads: Int,
    head_dim: Int,
    theta: Float64,
    mrope_t: Int,
    mrope_h: Int,
    mrope_w: Int,
) raises -> List[List[Float32]]:
    """Qwen2.5-VL mRoPE full-width table after section selection."""
    var s = len(t_pos)
    var half = head_dim // 2
    var t_section = mrope_t * 2
    var h_section = mrope_h * 2
    var w_section = mrope_w * 2
    if t_section + h_section + w_section != head_dim:
        raise Error("_build_mrope_tables: sections must sum to head_dim")

    var inv = List[Float32]()
    var log_theta = flog(Float32(theta))
    for i in range(half):
        inv.append(fexp(-log_theta * Float32(i) / Float32(half)))

    var cos_vals = List[Float32]()
    var sin_vals = List[Float32]()
    for row in range(s):
        var t_axis = List[Float32]()
        var h_axis = List[Float32]()
        var w_axis = List[Float32]()
        for i in range(half):
            t_axis.append(Float32(t_pos[row]) * inv[i])
            h_axis.append(Float32(h_pos[row]) * inv[i])
            w_axis.append(Float32(w_pos[row]) * inv[i])
        for _head in range(heads):
            for d in range(head_dim):
                var angle: Float32
                if d < t_section:
                    angle = _axis_full(t_axis, d, half)
                elif d < t_section + h_section:
                    angle = _axis_full(h_axis, d, half)
                else:
                    angle = _axis_full(w_axis, d, half)
                cos_vals.append(fcos(angle))
                sin_vals.append(fsin(angle))
    var out = List[List[Float32]]()
    out.append(cos_vals^)
    out.append(sin_vals^)
    return out^


struct LanceWeights(Movable):
    var weights: List[ArcPointer[Tensor]]
    var name_to_idx: Dict[String, Int]
    var config: LanceT2VConfig

    def __init__(
        out self,
        var weights: List[ArcPointer[Tensor]],
        var name_to_idx: Dict[String, Int],
        config: LanceT2VConfig,
    ):
        self.weights = weights^
        self.name_to_idx = name_to_idx^
        self.config = config

    @staticmethod
    def load_shared(dir: String, ctx: DeviceContext) raises -> LanceWeights:
        var sharded = ShardedSafeTensors.open(dir)
        var keys = lance_shared_keys()
        var weights = List[ArcPointer[Tensor]]()
        var name_to_idx = Dict[String, Int]()
        for i in range(len(keys)):
            var name = keys[i]
            var tv = sharded.tensor_view(name)
            var t = Tensor.from_view(tv, ctx)
            name_to_idx[name] = len(weights)
            weights.append(ArcPointer(t^))
        return LanceWeights(weights^, name_to_idx^, LanceT2VConfig.lance_3b_video())

    def _w(self, name: String) raises -> ref [self.weights] Tensor:
        if name not in self.name_to_idx:
            raise Error(String("missing Lance weight: ") + name)
        var idx = self.name_to_idx[name]
        return self.weights[idx][]

    def _embed(self, ids: List[Int], ctx: DeviceContext) raises -> Tensor:
        ref table = self._w(String("language_model.model.embed_tokens.weight"))
        var rows = gather_rows(table, ids, ctx)
        return reshape(rows, _shape3(1, len(ids), self.config.hidden_size), ctx)

    def _time_embed(self, timestep: Tensor, ctx: DeviceContext) raises -> Tensor:
        return t_embedder(
            timestep,
            self.config.timestep_dim,
            self._w(String("time_embedder.mlp.0.weight")),
            Optional[Tensor](_clone(self._w(String("time_embedder.mlp.0.bias")), ctx)),
            self._w(String("time_embedder.mlp.2.weight")),
            Optional[Tensor](_clone(self._w(String("time_embedder.mlp.2.bias")), ctx)),
            ctx,
        )

    def _vae_embed(
        self, x_t: Tensor, timestep: Tensor, latent_pos_ids: List[Int], ctx: DeviceContext
    ) raises -> Tensor:
        # Keep x_t as [L,48] or [1,L,48]; linear flattens leading dims.
        var vae = linear(
            x_t,
            self._w(String("vae2llm.weight")),
            Optional[Tensor](_clone(self._w(String("vae2llm.bias")), ctx)),
            ctx,
        )
        var t_emb = self._time_embed(timestep, ctx)
        var pos = gather_rows(self._w(String("latent_pos_embed.pos_embed")), latent_pos_ids, ctx)
        var vae2 = add(vae, t_emb, ctx)
        return add(vae2, pos, ctx)

    def _mlp(self, x: Tensor, prefix: String, ctx: DeviceContext) raises -> Tensor:
        var gate = linear(x, self._w(prefix + ".gate_proj.weight"), None, ctx)
        var up = linear(x, self._w(prefix + ".up_proj.weight"), None, ctx)
        var act = swiglu(gate, up, ctx)
        return linear(act, self._w(prefix + ".down_proj.weight"), None, ctx)

    def _layer[S: Int](
        self,
        layer_idx: Int,
        hidden: Tensor,
        gen_start: Int,
        gen_len: Int,
        cos_q: Tensor,
        sin_q: Tensor,
        cos_k: Tensor,
        sin_k: Tensor,
        mask: Tensor,
        ctx: DeviceContext,
    ) raises -> Tensor:
        var cfg = self.config
        var p = String("language_model.model.layers.") + String(layer_idx)
        var seq = S
        var h = cfg.num_heads
        var h_kv = cfg.num_kv_heads
        var dh = cfg.head_dim
        var n_rep = h // h_kv
        var scale = Float32(1.0) / sqrt(Float32(dh))

        var norm_all = rms_norm(
            hidden, self._w(p + ".input_layernorm.weight"), cfg.rms_norm_eps, ctx
        )
        var gen = ts_slice(hidden, 1, gen_start, gen_len, ctx)
        var gen_norm = rms_norm(
            gen, self._w(p + ".input_layernorm_moe_gen.weight"), cfg.rms_norm_eps, ctx
        )
        var normed = _replace_span(norm_all, gen_norm, gen_start, gen_len, ctx)

        var q = linear(
            normed,
            self._w(p + ".self_attn.q_proj.weight"),
            Optional[Tensor](_clone(self._w(p + ".self_attn.q_proj.bias"), ctx)),
            ctx,
        )
        var k = linear(
            normed,
            self._w(p + ".self_attn.k_proj.weight"),
            Optional[Tensor](_clone(self._w(p + ".self_attn.k_proj.bias"), ctx)),
            ctx,
        )
        var v = linear(
            normed,
            self._w(p + ".self_attn.v_proj.weight"),
            Optional[Tensor](_clone(self._w(p + ".self_attn.v_proj.bias"), ctx)),
            ctx,
        )

        var gen_norm2 = ts_slice(normed, 1, gen_start, gen_len, ctx)
        var qg = linear(
            gen_norm2,
            self._w(p + ".self_attn.q_proj_moe_gen.weight"),
            Optional[Tensor](_clone(self._w(p + ".self_attn.q_proj_moe_gen.bias"), ctx)),
            ctx,
        )
        var kg = linear(
            gen_norm2,
            self._w(p + ".self_attn.k_proj_moe_gen.weight"),
            Optional[Tensor](_clone(self._w(p + ".self_attn.k_proj_moe_gen.bias"), ctx)),
            ctx,
        )
        var vg = linear(
            gen_norm2,
            self._w(p + ".self_attn.v_proj_moe_gen.weight"),
            Optional[Tensor](_clone(self._w(p + ".self_attn.v_proj_moe_gen.bias"), ctx)),
            ctx,
        )
        q = _replace_span(q, qg, gen_start, gen_len, ctx)
        k = _replace_span(k, kg, gen_start, gen_len, ctx)
        v = _replace_span(v, vg, gen_start, gen_len, ctx)

        q = reshape(q, _shape4(1, seq, h, dh), ctx)
        k = reshape(k, _shape4(1, seq, h_kv, dh), ctx)
        v = reshape(v, _shape4(1, seq, h_kv, dh), ctx)

        q = rms_norm(q, self._w(p + ".self_attn.q_norm.weight"), cfg.rms_norm_eps, ctx)
        k = rms_norm(k, self._w(p + ".self_attn.k_norm.weight"), cfg.rms_norm_eps, ctx)
        var qg4 = ts_slice(q, 1, gen_start, gen_len, ctx)
        var kg4 = ts_slice(k, 1, gen_start, gen_len, ctx)
        qg4 = rms_norm(
            qg4, self._w(p + ".self_attn.q_norm_moe_gen.weight"), cfg.rms_norm_eps, ctx
        )
        kg4 = rms_norm(
            kg4, self._w(p + ".self_attn.k_norm_moe_gen.weight"), cfg.rms_norm_eps, ctx
        )
        q = _replace_span(q, qg4, gen_start, gen_len, ctx)
        k = _replace_span(k, kg4, gen_start, gen_len, ctx)

        q = rope_halfsplit_full(q, cos_q, sin_q, ctx)
        k = rope_halfsplit_full(k, cos_k, sin_k, ctx)
        var k_rep = _repeat_kv(k, seq, h_kv, n_rep, dh, ctx)
        var v_rep = _repeat_kv(v, seq, h_kv, n_rep, dh, ctx)
        var attn = _sdpa_lance[S](q, k_rep, v_rep, mask, scale, ctx)
        attn = reshape(attn, _shape3(1, seq, h * dh), ctx)

        var attn_out = linear(attn, self._w(p + ".self_attn.o_proj.weight"), None, ctx)
        var attn_gen = ts_slice(attn, 1, gen_start, gen_len, ctx)
        var attn_gen_out = linear(
            attn_gen, self._w(p + ".self_attn.o_proj_moe_gen.weight"), None, ctx
        )
        attn_out = _replace_span(attn_out, attn_gen_out, gen_start, gen_len, ctx)
        var hidden2 = add(hidden, attn_out, ctx)

        var norm2 = rms_norm(
            hidden2, self._w(p + ".post_attention_layernorm.weight"), cfg.rms_norm_eps, ctx
        )
        var mlp_out = self._mlp(norm2, p + ".mlp", ctx)
        var gen2 = ts_slice(hidden2, 1, gen_start, gen_len, ctx)
        var gen2n = rms_norm(
            gen2, self._w(p + ".post_attention_layernorm_moe_gen.weight"), cfg.rms_norm_eps, ctx
        )
        var gen_mlp = self._mlp(gen2n, p + ".mlp_moe_gen", ctx)
        mlp_out = _replace_span(mlp_out, gen_mlp, gen_start, gen_len, ctx)
        return add(hidden2, mlp_out, ctx)


@fieldwise_init
struct LanceT2VOffloaded[S: Int](Movable):
    var shared: LanceWeights
    var loader: PlannedBlockLoader

    @staticmethod
    def load(dir: String, ctx: DeviceContext) raises -> LanceT2VOffloaded[Self.S]:
        var shared = LanceWeights.load_shared(dir, ctx)
        var plan = build_lance_t2v_block_plan()
        var loader = PlannedBlockLoader.open(
            dir, plan^, OffloadConfig.synchronous_single()
        )
        return LanceT2VOffloaded[Self.S](shared^, loader^)

    def _block_weights(self, block: Block) -> LanceWeights:
        var weights = self.shared.weights.copy()
        var name_to_idx = self.shared.name_to_idx.copy()
        for ref e in block.items():
            name_to_idx[e.key] = len(weights)
            weights.append(e.value)
        return LanceWeights(weights^, name_to_idx^, self.shared.config)

    def forward_velocity(
        mut self,
        input: LanceT2VInput,
        x_t: Tensor,
        timestep: Tensor,
        max_layers: Int,
        ctx: DeviceContext,
    ) raises -> Tensor:
        var cfg = self.shared.config
        if len(input.full_ids) != Self.S:
            raise Error("Lance forward_velocity: input length != static S")
        var hidden = self.shared._embed(input.full_ids, ctx)
        var vae = self.shared._vae_embed(x_t, timestep, input.latent_pos_ids, ctx)
        vae = reshape(vae, _shape3(1, input.gen_len, cfg.hidden_size), ctx)
        hidden = _replace_span(hidden, vae, input.gen_start, input.gen_len, ctx)

        var q_tables = _build_mrope_tables(
            input.t_pos,
            input.h_pos,
            input.w_pos,
            cfg.num_heads,
            cfg.head_dim,
            cfg.rope_theta,
            cfg.mrope_t,
            cfg.mrope_h,
            cfg.mrope_w,
        )
        var k_tables = _build_mrope_tables(
            input.t_pos,
            input.h_pos,
            input.w_pos,
            cfg.num_kv_heads,
            cfg.head_dim,
            cfg.rope_theta,
            cfg.mrope_t,
            cfg.mrope_h,
            cfg.mrope_w,
        )
        var q_sh = List[Int]()
        q_sh.append(Self.S * cfg.num_heads * cfg.head_dim)
        var k_sh = List[Int]()
        k_sh.append(Self.S * cfg.num_kv_heads * cfg.head_dim)
        var cos_q = Tensor.from_host(q_tables[0], q_sh.copy(), STDtype.F32, ctx)
        var sin_q = Tensor.from_host(q_tables[1], q_sh.copy(), STDtype.F32, ctx)
        var cos_k = Tensor.from_host(k_tables[0], k_sh.copy(), STDtype.F32, ctx)
        var sin_k = Tensor.from_host(k_tables[1], k_sh.copy(), STDtype.F32, ctx)

        var mask_data = _build_t2v_mask(Self.S, cfg.num_heads, input.text_split_len)
        var mask = Tensor.from_host(
            mask_data, _shape4(1, cfg.num_heads, Self.S, Self.S), STDtype.F32, ctx
        )

        var layers = max_layers
        if layers <= 0 or layers > cfg.num_layers:
            layers = cfg.num_layers
        self.loader.config = OffloadConfig.synchronous_single()
        self.loader.prefetch(0)
        for li in range(layers):
            self.loader.prefetch_next(li)
            var handle = self.loader.await_block(li, ctx)
            var bw = self._block_weights(handle.block)
            hidden = bw._layer[Self.S](
                li,
                hidden,
                input.gen_start,
                input.gen_len,
                cos_q,
                sin_q,
                cos_k,
                sin_k,
                mask,
                ctx,
            )

        var norm_all = rms_norm(
            hidden,
            self.shared._w(String("language_model.model.norm.weight")),
            cfg.rms_norm_eps,
            ctx,
        )
        var gen_h = ts_slice(hidden, 1, input.gen_start, input.gen_len, ctx)
        var gen_norm = rms_norm(
            gen_h,
            self.shared._w(String("language_model.model.norm_moe_gen.weight")),
            cfg.rms_norm_eps,
            ctx,
        )
        var final_h = _replace_span(norm_all, gen_norm, input.gen_start, input.gen_len, ctx)
        var gen_final = ts_slice(final_h, 1, input.gen_start, input.gen_len, ctx)
        return linear(
            gen_final,
            self.shared._w(String("llm2vae.weight")),
            Optional[Tensor](_clone(self.shared._w(String("llm2vae.bias")), ctx)),
            ctx,
        )


def build_lance_t2v_input(
    tok: Qwen3Tokenizer,
    prompt: String,
    latent_t: Int,
    latent_h: Int,
    latent_w: Int,
) raises -> LanceT2VInput:
    var text = tok.encode(prompt)
    return build_lance_t2v_input_from_text_ids(text, latent_t, latent_h, latent_w)


def build_lance_t2v_input_from_text_ids(
    text: List[Int],
    latent_t: Int,
    latent_h: Int,
    latent_w: Int,
) raises -> LanceT2VInput:
    var cfg = LanceT2VConfig.lance_3b_video()
    var full = List[Int]()
    full.append(cfg.bos_token_id)
    for i in range(len(text)):
        full.append(text[i])
    full.append(cfg.eos_token_id)
    var text_split_len = len(full)
    full.append(cfg.start_of_image_id)
    var gen_start = len(full)
    var gen_len = latent_t * latent_h * latent_w
    for _ in range(gen_len):
        full.append(cfg.video_token_id)
    full.append(cfg.end_of_image_id)

    var prefix_len = text_split_len + 1
    var t_pos = List[Int]()
    var h_pos = List[Int]()
    var w_pos = List[Int]()
    for i in range(prefix_len):
        t_pos.append(i)
        h_pos.append(i)
        w_pos.append(i)
    var latent_pos = List[Int]()
    var max_pos = prefix_len
    for tt in range(latent_t):
        for hh in range(latent_h):
            for ww in range(latent_w):
                var tp = prefix_len + tt * cfg.tokens_per_second
                var hp = prefix_len + hh
                var wp = prefix_len + ww
                t_pos.append(tp)
                h_pos.append(hp)
                w_pos.append(wp)
                if tp > max_pos:
                    max_pos = tp
                if hp > max_pos:
                    max_pos = hp
                if wp > max_pos:
                    max_pos = wp
                latent_pos.append(tt * cfg.max_latent_size * cfg.max_latent_size + hh * cfg.max_latent_size + ww)
    var end_pos = max_pos + 1
    t_pos.append(end_pos)
    h_pos.append(end_pos)
    w_pos.append(end_pos)

    return LanceT2VInput(
        full^, latent_pos^, t_pos^, h_pos^, w_pos^,
        text_split_len, gen_start, gen_len,
    )


def build_lance_t2v_padded_uncond_input(
    text_token_count: Int,
    latent_t: Int,
    latent_h: Int,
    latent_w: Int,
) raises -> LanceT2VInput:
    """Build a same-length empty-text input for dense CFG smokes.

    The production KV-cache path can use different cond/uncond text lengths.
    The current tiny dense forward has one comptime sequence length, so it pads
    the empty text side with end-of-text tokens to match the conditional prompt.
    """
    if text_token_count < 0:
        raise Error("build_lance_t2v_padded_uncond_input: negative text length")
    var ids = List[Int]()
    for _ in range(text_token_count):
        ids.append(151643)  # Qwen2 <|endoftext|>
    return build_lance_t2v_input_from_text_ids(ids, latent_t, latent_h, latent_w)

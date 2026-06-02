# klein_dit.mojo - FLUX.2 Klein DiT scaffolding.
#
# This file currently exposes a truncated Klein 9B smoke runner. It uses real
# checkpoint weights and executes the core 9B block math on a tiny token grid:
# input projections, timestep MLP, shared modulation, one double block, one
# single block, final AdaLN, and final projection. The full 8+24 block pipeline
# should extend these helpers rather than duplicating them.

from std.gpu.host import DeviceContext
from std.math import sqrt, cos as fcos, sin as fsin, exp as fexp, log as flog
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import rms_norm, layer_norm
from serenitymojo.ops.activations import silu, swiglu
from serenitymojo.ops.elementwise import modulate, residual_gate
from serenitymojo.ops.rope import rope_interleaved
from serenitymojo.ops.attention import sdpa_nomask
from serenitymojo.ops.embeddings import t_embedder
from serenitymojo.ops.tensor_algebra import reshape, slice, concat
from serenitymojo.offload.block_loader import Block
from serenitymojo.offload.plan import OffloadConfig, build_klein9b_block_plan
from serenitymojo.offload.planned_loader import PlannedBlockLoader
from serenitymojo.offload.turbo_planned_loader import TurboPlannedLoader
from serenitymojo.training.train_config import TrainConfig


@fieldwise_init
struct KleinConfig(Copyable, Movable, ImplicitlyCopyable):
    var inner_dim: Int
    var in_channels: Int
    var joint_attention_dim: Int
    var num_double: Int
    var num_single: Int
    var num_heads: Int
    var head_dim: Int
    var mlp_hidden: Int
    var timestep_dim: Int
    var rope_theta: Float64

    @staticmethod
    def klein_9b() -> KleinConfig:
        return KleinConfig(4096, 128, 12288, 8, 24, 32, 128, 12288, 256, 2000.0)

    @staticmethod
    def from_train_config(tc: TrainConfig) -> KleinConfig:
        """Build a KleinConfig from the run's config-file-driven TrainConfig.
        This is how the sampler gets its arch WITHOUT hardcoding a variant."""
        return KleinConfig(
            tc.d_model, tc.in_channels, tc.joint_attention_dim,
            tc.num_double, tc.num_single, tc.n_heads, tc.head_dim,
            tc.mlp_hidden, tc.timestep_dim, tc.rope_theta,
        )


def _load_weight(st: SafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    return Tensor.from_view(tv, ctx)


def _append_key(mut keys: List[String], name: String):
    keys.append(name)


def klein9b_truncated_keys() -> List[String]:
    var keys = klein9b_shared_keys()
    var dp = String("double_blocks.0")
    _append_key(keys, dp + ".img_attn.qkv.weight")
    _append_key(keys, dp + ".img_attn.proj.weight")
    _append_key(keys, dp + ".img_attn.norm.query_norm.scale")
    _append_key(keys, dp + ".img_attn.norm.key_norm.scale")
    _append_key(keys, dp + ".img_mlp.0.weight")
    _append_key(keys, dp + ".img_mlp.2.weight")
    _append_key(keys, dp + ".txt_attn.qkv.weight")
    _append_key(keys, dp + ".txt_attn.proj.weight")
    _append_key(keys, dp + ".txt_attn.norm.query_norm.scale")
    _append_key(keys, dp + ".txt_attn.norm.key_norm.scale")
    _append_key(keys, dp + ".txt_mlp.0.weight")
    _append_key(keys, dp + ".txt_mlp.2.weight")

    var sp = String("single_blocks.0")
    _append_key(keys, sp + ".linear1.weight")
    _append_key(keys, sp + ".linear2.weight")
    _append_key(keys, sp + ".norm.query_norm.scale")
    _append_key(keys, sp + ".norm.key_norm.scale")
    return keys^


def klein9b_shared_keys() -> List[String]:
    var keys = List[String]()
    _append_key(keys, String("img_in.weight"))
    _append_key(keys, String("txt_in.weight"))
    _append_key(keys, String("time_in.in_layer.weight"))
    _append_key(keys, String("time_in.out_layer.weight"))
    _append_key(keys, String("double_stream_modulation_img.lin.weight"))
    _append_key(keys, String("double_stream_modulation_txt.lin.weight"))
    _append_key(keys, String("single_stream_modulation.lin.weight"))
    _append_key(keys, String("final_layer.adaLN_modulation.1.weight"))
    _append_key(keys, String("final_layer.linear.weight"))
    return keys^


# Config-driven key list: generate the full weight-key set for a Klein variant
# with `num_double` double-stream + `num_single` single-stream blocks. The block
# counts come from the run config (NOT hardcoded) — this is the fix for the
# sampler "double_blocks.5 not found" bug on the 4B checkpoint.
def klein_all_keys(num_double: Int, num_single: Int) -> List[String]:
    var keys = List[String]()
    _append_key(keys, String("img_in.weight"))
    _append_key(keys, String("txt_in.weight"))
    _append_key(keys, String("time_in.in_layer.weight"))
    _append_key(keys, String("time_in.out_layer.weight"))
    _append_key(keys, String("double_stream_modulation_img.lin.weight"))
    _append_key(keys, String("double_stream_modulation_txt.lin.weight"))
    _append_key(keys, String("single_stream_modulation.lin.weight"))
    _append_key(keys, String("final_layer.adaLN_modulation.1.weight"))
    _append_key(keys, String("final_layer.linear.weight"))

    for bi in range(num_double):
        var dp = String("double_blocks.") + String(bi)
        _append_key(keys, dp + ".img_attn.qkv.weight")
        _append_key(keys, dp + ".img_attn.proj.weight")
        _append_key(keys, dp + ".img_attn.norm.query_norm.scale")
        _append_key(keys, dp + ".img_attn.norm.key_norm.scale")
        _append_key(keys, dp + ".img_mlp.0.weight")
        _append_key(keys, dp + ".img_mlp.2.weight")
        _append_key(keys, dp + ".txt_attn.qkv.weight")
        _append_key(keys, dp + ".txt_attn.proj.weight")
        _append_key(keys, dp + ".txt_attn.norm.query_norm.scale")
        _append_key(keys, dp + ".txt_attn.norm.key_norm.scale")
        _append_key(keys, dp + ".txt_mlp.0.weight")
        _append_key(keys, dp + ".txt_mlp.2.weight")

    for bi in range(num_single):
        var sp = String("single_blocks.") + String(bi)
        _append_key(keys, sp + ".linear1.weight")
        _append_key(keys, sp + ".linear2.weight")
        _append_key(keys, sp + ".norm.query_norm.scale")
        _append_key(keys, sp + ".norm.key_norm.scale")
    return keys^


# Back-compat: the 9B key list (8 double + 24 single) for the 9B inference smokes.
def klein9b_all_keys() -> List[String]:
    return klein_all_keys(8, 24)


@fieldwise_init
struct Klein9BDiT(Movable):
    var weights: List[ArcPointer[Tensor]]
    var name_to_idx: Dict[String, Int]
    var config: KleinConfig

    @staticmethod
    def load(path: String, ctx: DeviceContext) raises -> Klein9BDiT:
        var st = SafeTensors.open(path)
        var keys = klein9b_truncated_keys()
        return Klein9BDiT._load_keys(st^, keys^, KleinConfig.klein_9b(), ctx)

    @staticmethod
    def load_shared(path: String, ctx: DeviceContext) raises -> Klein9BDiT:
        var st = SafeTensors.open(path)
        var keys = klein9b_shared_keys()
        return Klein9BDiT._load_keys(st^, keys^, KleinConfig.klein_9b(), ctx)

    @staticmethod
    def load_full(path: String, ctx: DeviceContext) raises -> Klein9BDiT:
        # 9B back-compat loader for the 9B inference smokes.
        var st = SafeTensors.open(path)
        var keys = klein9b_all_keys()
        return Klein9BDiT._load_keys(st^, keys^, KleinConfig.klein_9b(), ctx)

    @staticmethod
    def load_with_config(
        path: String, cfg: KleinConfig, ctx: DeviceContext
    ) raises -> Klein9BDiT:
        """Config-driven full loader: block counts + arch come from `cfg` (built
        from the run's config file via KleinConfig.from_train_config). This is
        what the trainer's validation sampler uses so 4B vs 9B is data, not code."""
        var st = SafeTensors.open(path)
        var keys = klein_all_keys(cfg.num_double, cfg.num_single)
        return Klein9BDiT._load_keys(st^, keys^, cfg, ctx)

    @staticmethod
    def _load_keys(
        var st: SafeTensors, var keys: List[String], cfg: KleinConfig,
        ctx: DeviceContext,
    ) raises -> Klein9BDiT:
        var weights = List[ArcPointer[Tensor]]()
        var name_to_idx = Dict[String, Int]()
        for i in range(len(keys)):
            var name = keys[i]
            var t = _load_weight(st, name, ctx)
            name_to_idx[name] = len(weights)
            weights.append(ArcPointer(t^))
        return Klein9BDiT(weights^, name_to_idx^, cfg)

    def _w(self, name: String) raises -> ref [self.weights] Tensor:
        if name not in self.name_to_idx:
            raise Error(String("missing Klein weight: ") + name)
        var idx = self.name_to_idx[name]
        return self.weights[idx][]

    def _ones(self, d: Int, dtype: STDtype, ctx: DeviceContext) raises -> Tensor:
        var vals = List[Float32]()
        for _ in range(d):
            vals.append(1.0)
        var sh = List[Int]()
        sh.append(d)
        return Tensor.from_host(vals, sh^, dtype, ctx)

    def _zeros_vec(self, d: Int, dtype: STDtype, ctx: DeviceContext) raises -> Tensor:
        var vals = List[Float32]()
        for _ in range(d):
            vals.append(0.0)
        var sh = List[Int]()
        sh.append(d)
        return Tensor.from_host(vals, sh^, dtype, ctx)

    def _chunk_last(
        self, x: Tensor, idx: Int, chunk: Int, ctx: DeviceContext
    ) raises -> Tensor:
        var s = x.shape()
        var dim = len(s) - 1
        var part = slice(x, dim, idx * chunk, chunk, ctx)
        var out_shape = List[Int]()
        out_shape.append(chunk)
        return reshape(part, out_shape^, ctx)

    def _modulate_pre(
        self, x: Tensor, shift: Tensor, scale: Tensor, ctx: DeviceContext
    ) raises -> Tensor:
        var dtype = x.dtype()
        var d = x.shape()[len(x.shape()) - 1]
        var ones = self._ones(d, dtype, ctx)
        var zeros = self._zeros_vec(d, dtype, ctx)
        var normed = layer_norm(x, ones, zeros, 1.0e-6, ctx)
        return modulate(normed, scale, shift, ctx)

    def _qkv_part(self, qkv: Tensor, part: Int, ctx: DeviceContext) raises -> Tensor:
        var cfg = self.config
        var inner = cfg.inner_dim
        var qkv_part = slice(qkv, 2, part * inner, inner, ctx)
        var sh = qkv_part.shape()
        var n = sh[1]
        var out = List[Int]()
        out.append(1)
        out.append(n)
        out.append(cfg.num_heads)
        out.append(cfg.head_dim)
        return reshape(qkv_part, out^, ctx)

    def _attn_rope_only[S: Int](
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        cos: Tensor,
        sin: Tensor,
        ctx: DeviceContext,
    ) raises -> Tensor:
        var q_roped = rope_interleaved(q, cos, sin, ctx)
        var k_roped = rope_interleaved(k, cos, sin, ctx)
        return sdpa_nomask[1, S, 32, 128](
            q_roped, k_roped, v, Float32(1.0) / sqrt(Float32(128)), ctx
        )

    def _attn[S: Int](
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        q_scale: Tensor,
        k_scale: Tensor,
        cos: Tensor,
        sin: Tensor,
        ctx: DeviceContext,
    ) raises -> Tensor:
        var q_norm = rms_norm(q, q_scale, 1.0e-6, ctx)
        var k_norm = rms_norm(k, k_scale, 1.0e-6, ctx)
        return self._attn_rope_only[S](q_norm, k_norm, v, cos, sin, ctx)

    def _swiglu_linear(
        self, x: Tensor, gate_up_key: String, down_key: String, ctx: DeviceContext
    ) raises -> Tensor:
        ref guw = self._w(gate_up_key)
        ref down = self._w(down_key)
        var gu = linear(x, guw, None, ctx)
        var gate = slice(gu, 2, 0, self.config.mlp_hidden, ctx)
        var up = slice(gu, 2, self.config.mlp_hidden, self.config.mlp_hidden, ctx)
        var act = swiglu(gate, up, ctx)
        return linear(act, down, None, ctx)

    def _double_block[
        N_IMG: Int, N_TXT: Int, S: Int
    ](
        self,
        p: String,
        img: Tensor,
        txt: Tensor,
        img_mod: Tensor,
        txt_mod: Tensor,
        cos: Tensor,
        sin: Tensor,
        ctx: DeviceContext,
    ) raises -> Tensor:
        comptime assert S == N_IMG + N_TXT, "S must equal N_IMG + N_TXT"
        var d = self.config.inner_dim
        var img_shift1 = self._chunk_last(img_mod, 0, d, ctx)
        var img_scale1 = self._chunk_last(img_mod, 1, d, ctx)
        var img_gate1 = self._chunk_last(img_mod, 2, d, ctx)
        var img_shift2 = self._chunk_last(img_mod, 3, d, ctx)
        var img_scale2 = self._chunk_last(img_mod, 4, d, ctx)
        var img_gate2 = self._chunk_last(img_mod, 5, d, ctx)

        var txt_shift1 = self._chunk_last(txt_mod, 0, d, ctx)
        var txt_scale1 = self._chunk_last(txt_mod, 1, d, ctx)
        var txt_gate1 = self._chunk_last(txt_mod, 2, d, ctx)
        var txt_shift2 = self._chunk_last(txt_mod, 3, d, ctx)
        var txt_scale2 = self._chunk_last(txt_mod, 4, d, ctx)
        var txt_gate2 = self._chunk_last(txt_mod, 5, d, ctx)

        var img_norm = self._modulate_pre(img, img_shift1, img_scale1, ctx)
        var txt_norm = self._modulate_pre(txt, txt_shift1, txt_scale1, ctx)

        var img_qkv = linear(img_norm, self._w(p + ".img_attn.qkv.weight"), None, ctx)
        var txt_qkv = linear(txt_norm, self._w(p + ".txt_attn.qkv.weight"), None, ctx)
        var img_q = self._qkv_part(img_qkv, 0, ctx)
        var img_k = self._qkv_part(img_qkv, 1, ctx)
        var img_v = self._qkv_part(img_qkv, 2, ctx)
        var txt_q = self._qkv_part(txt_qkv, 0, ctx)
        var txt_k = self._qkv_part(txt_qkv, 1, ctx)
        var txt_v = self._qkv_part(txt_qkv, 2, ctx)
        img_q = rms_norm(
            img_q, self._w(p + ".img_attn.norm.query_norm.scale"), 1.0e-6, ctx
        )
        img_k = rms_norm(
            img_k, self._w(p + ".img_attn.norm.key_norm.scale"), 1.0e-6, ctx
        )
        txt_q = rms_norm(
            txt_q, self._w(p + ".txt_attn.norm.query_norm.scale"), 1.0e-6, ctx
        )
        txt_k = rms_norm(
            txt_k, self._w(p + ".txt_attn.norm.key_norm.scale"), 1.0e-6, ctx
        )
        var q = concat(1, ctx, txt_q, img_q)
        var k = concat(1, ctx, txt_k, img_k)
        var v = concat(1, ctx, txt_v, img_v)
        var att = self._attn_rope_only[S](q, k, v, cos, sin, ctx)

        var txt_att = slice(att, 1, 0, N_TXT, ctx)
        var img_att = slice(att, 1, N_TXT, N_IMG, ctx)
        var img_out_shape = List[Int]()
        img_out_shape.append(1)
        img_out_shape.append(N_IMG)
        img_out_shape.append(d)
        var txt_out_shape = List[Int]()
        txt_out_shape.append(1)
        txt_out_shape.append(N_TXT)
        txt_out_shape.append(d)
        img_att = reshape(img_att, img_out_shape^, ctx)
        txt_att = reshape(txt_att, txt_out_shape^, ctx)

        var img_out = linear(img_att, self._w(p + ".img_attn.proj.weight"), None, ctx)
        var txt_out = linear(txt_att, self._w(p + ".txt_attn.proj.weight"), None, ctx)
        var img_attn_res = residual_gate(img, img_gate1, img_out, ctx)
        var txt_attn_res = residual_gate(txt, txt_gate1, txt_out, ctx)

        var img_mlp_in = self._modulate_pre(img_attn_res, img_shift2, img_scale2, ctx)
        var txt_mlp_in = self._modulate_pre(txt_attn_res, txt_shift2, txt_scale2, ctx)
        var img_mlp = self._swiglu_linear(
            img_mlp_in, p + ".img_mlp.0.weight", p + ".img_mlp.2.weight", ctx
        )
        var txt_mlp = self._swiglu_linear(
            txt_mlp_in, p + ".txt_mlp.0.weight", p + ".txt_mlp.2.weight", ctx
        )
        var img_final = residual_gate(img_attn_res, img_gate2, img_mlp, ctx)
        var txt_final = residual_gate(txt_attn_res, txt_gate2, txt_mlp, ctx)
        return concat(1, ctx, txt_final, img_final)

    def _single_block[S: Int](
        self,
        p: String,
        x: Tensor,
        single_mod: Tensor,
        cos: Tensor,
        sin: Tensor,
        ctx: DeviceContext,
    ) raises -> Tensor:
        var d = self.config.inner_dim
        var shift = self._chunk_last(single_mod, 0, d, ctx)
        var scale = self._chunk_last(single_mod, 1, d, ctx)
        var gate = self._chunk_last(single_mod, 2, d, ctx)
        var x_norm = self._modulate_pre(x, shift, scale, ctx)
        var fused = linear(x_norm, self._w(p + ".linear1.weight"), None, ctx)
        var qkv = slice(fused, 2, 0, 3 * d, ctx)
        var gate_up = slice(fused, 2, 3 * d, 2 * self.config.mlp_hidden, ctx)
        var q = self._qkv_part(qkv, 0, ctx)
        var k = self._qkv_part(qkv, 1, ctx)
        var v = self._qkv_part(qkv, 2, ctx)
        var att = self._attn[S](
            q, k, v,
            self._w(p + ".norm.query_norm.scale"),
            self._w(p + ".norm.key_norm.scale"),
            cos, sin, ctx,
        )
        var att_shape = List[Int]()
        att_shape.append(1)
        att_shape.append(S)
        att_shape.append(d)
        var att_flat = reshape(att, att_shape^, ctx)
        var mlp_gate = slice(gate_up, 2, 0, self.config.mlp_hidden, ctx)
        var mlp_up = slice(gate_up, 2, self.config.mlp_hidden, self.config.mlp_hidden, ctx)
        var mlp = swiglu(mlp_gate, mlp_up, ctx)
        var out_in = concat(2, ctx, att_flat, mlp)
        var out = linear(out_in, self._w(p + ".linear2.weight"), None, ctx)
        return residual_gate(x, gate, out, ctx)

    def forward_truncated[
        N_IMG: Int, N_TXT: Int, S: Int
    ](
        self,
        img_tokens: Tensor,
        txt_tokens: Tensor,
        timestep: Tensor,
        cos: Tensor,
        sin: Tensor,
        ctx: DeviceContext,
    ) raises -> Tensor:
        comptime assert S == N_IMG + N_TXT, "S must equal N_IMG + N_TXT"
        var cfg = self.config
        var img = linear(img_tokens, self._w(String("img_in.weight")), None, ctx)
        var txt = linear(txt_tokens, self._w(String("txt_in.weight")), None, ctx)
        var vec = t_embedder(
            timestep,
            cfg.timestep_dim,
            self._w(String("time_in.in_layer.weight")),
            None,
            self._w(String("time_in.out_layer.weight")),
            None,
            ctx,
        )
        var vec_silu = silu(vec, ctx)
        var img_mod = linear(
            vec_silu,
            self._w(String("double_stream_modulation_img.lin.weight")),
            None,
            ctx,
        )
        var txt_mod = linear(
            vec_silu,
            self._w(String("double_stream_modulation_txt.lin.weight")),
            None,
            ctx,
        )
        var single_mod = linear(
            vec_silu,
            self._w(String("single_stream_modulation.lin.weight")),
            None,
            ctx,
        )
        var x = self._double_block[N_IMG, N_TXT, S](
            String("double_blocks.0"), img, txt, img_mod, txt_mod, cos, sin, ctx
        )
        x = self._single_block[S](String("single_blocks.0"), x, single_mod, cos, sin, ctx)
        var img_out = slice(x, 1, N_TXT, N_IMG, ctx)
        var final_mod = linear(
            vec_silu,
            self._w(String("final_layer.adaLN_modulation.1.weight")),
            None,
            ctx,
        )
        var shift = self._chunk_last(final_mod, 0, cfg.inner_dim, ctx)
        var scale = self._chunk_last(final_mod, 1, cfg.inner_dim, ctx)
        var normed = self._modulate_pre(img_out, shift, scale, ctx)
        return linear(normed, self._w(String("final_layer.linear.weight")), None, ctx)

    def forward_full[
        N_IMG: Int, N_TXT: Int, S: Int
    ](
        self,
        img_tokens: Tensor,
        txt_tokens: Tensor,
        timestep: Tensor,
        cos: Tensor,
        sin: Tensor,
        ctx: DeviceContext,
    ) raises -> Tensor:
        comptime assert S == N_IMG + N_TXT, "S must equal N_IMG + N_TXT"
        var cfg = self.config
        var img = linear(img_tokens, self._w(String("img_in.weight")), None, ctx)
        var txt = linear(txt_tokens, self._w(String("txt_in.weight")), None, ctx)
        var vec = t_embedder(
            timestep,
            cfg.timestep_dim,
            self._w(String("time_in.in_layer.weight")),
            None,
            self._w(String("time_in.out_layer.weight")),
            None,
            ctx,
        )
        var vec_silu = silu(vec, ctx)
        var img_mod = linear(
            vec_silu,
            self._w(String("double_stream_modulation_img.lin.weight")),
            None,
            ctx,
        )
        var txt_mod = linear(
            vec_silu,
            self._w(String("double_stream_modulation_txt.lin.weight")),
            None,
            ctx,
        )
        var single_mod = linear(
            vec_silu,
            self._w(String("single_stream_modulation.lin.weight")),
            None,
            ctx,
        )

        for bi in range(cfg.num_double):
            var x = self._double_block[N_IMG, N_TXT, S](
                String("double_blocks.") + String(bi),
                img, txt, img_mod, txt_mod, cos, sin, ctx,
            )
            txt = slice(x, 1, 0, N_TXT, ctx)
            img = slice(x, 1, N_TXT, N_IMG, ctx)

        var x = concat(1, ctx, txt, img)
        for bi in range(cfg.num_single):
            x = self._single_block[S](
                String("single_blocks.") + String(bi), x, single_mod, cos, sin, ctx
            )

        var img_out = slice(x, 1, N_TXT, N_IMG, ctx)
        var final_mod = linear(
            vec_silu,
            self._w(String("final_layer.adaLN_modulation.1.weight")),
            None,
            ctx,
        )
        var shift = self._chunk_last(final_mod, 0, cfg.inner_dim, ctx)
        var scale = self._chunk_last(final_mod, 1, cfg.inner_dim, ctx)
        var normed = self._modulate_pre(img_out, shift, scale, ctx)
        return linear(normed, self._w(String("final_layer.linear.weight")), None, ctx)


def build_klein_rope_tables[
    N_IMG: Int, N_TXT: Int, H: Int, DH: Int
](
    ctx: DeviceContext, dtype: STDtype
) raises -> Tuple[Tensor, Tensor]:
    comptime assert DH == 128, "Klein 9B head dim must be 128"
    comptime S = N_IMG + N_TXT
    comptime HALF = DH // 2
    var img_w = 1
    while img_w * img_w < N_IMG:
        img_w += 1
    if img_w * img_w != N_IMG:
        raise Error("build_klein_rope_tables: N_IMG must be a square grid")
    var cos_vals = List[Float32]()
    var sin_vals = List[Float32]()
    var log_theta = flog(Float32(2000.0))
    for tok in range(S):
        var p0 = 0
        var p1 = 0
        var p2 = 0
        var p3 = 0
        if tok >= N_TXT:
            var idx = tok - N_TXT
            p1 = idx // img_w
            p2 = idx % img_w
        else:
            # text-token RoPE = [0,0,0,k] (upstream Flux2 prepare_text_ids).
            # Axis-3 carries the L-axis rotary freqs → distinct phase per text
            # token. Was all-zero (the bug EDv2 klein.rs KLEIN_VERIFY §H2 fixed).
            p3 = tok
        for _h in range(H):
            for axis in range(4):
                var pos = p0
                if axis == 1:
                    pos = p1
                elif axis == 2:
                    pos = p2
                elif axis == 3:
                    pos = p3
                for i in range(16):
                    var inv_freq = fexp(-log_theta * Float32(2 * i) / Float32(32))
                    var angle = Float32(pos) * inv_freq
                    cos_vals.append(fcos(angle))
                    sin_vals.append(fsin(angle))
    var sh = List[Int]()
    sh.append(S * H)
    sh.append(HALF)
    return (
        Tensor.from_host(cos_vals, sh.copy(), dtype, ctx),
        Tensor.from_host(sin_vals, sh^, dtype, ctx),
    )


@fieldwise_init
struct KleinCfgPreds(Movable):
    var pos: Tensor
    var neg: Tensor


@fieldwise_init
struct Klein9BOffloaded(Movable):
    var shared: Klein9BDiT
    var loader: PlannedBlockLoader

    @staticmethod
    def load(path: String, ctx: DeviceContext) raises -> Klein9BOffloaded:
        var shared = Klein9BDiT.load_shared(path, ctx)
        var plan = build_klein9b_block_plan()
        var loader = PlannedBlockLoader.open(
            path, plan^, OffloadConfig.synchronous_cfg_paired()
        )
        return Klein9BOffloaded(shared^, loader^)

    def _block_model(self, block: Block) -> Klein9BDiT:
        var weights = self.shared.weights.copy()
        var name_to_idx = self.shared.name_to_idx.copy()
        for ref e in block.items():
            name_to_idx[e.key] = len(weights)
            weights.append(e.value)
        return Klein9BDiT(weights^, name_to_idx^, self.shared.config)

    def _run_double[
        N_IMG: Int, N_TXT: Int, S: Int
    ](
        self,
        block: Block,
        prefix: String,
        img: Tensor,
        txt: Tensor,
        img_mod: Tensor,
        txt_mod: Tensor,
        cos: Tensor,
        sin: Tensor,
        ctx: DeviceContext,
    ) raises -> Tensor:
        var tmp = self._block_model(block)
        return tmp._double_block[N_IMG, N_TXT, S](
            prefix, img, txt, img_mod, txt_mod, cos, sin, ctx
        )

    def _run_single[S: Int](
        self,
        block: Block,
        prefix: String,
        x: Tensor,
        single_mod: Tensor,
        cos: Tensor,
        sin: Tensor,
        ctx: DeviceContext,
    ) raises -> Tensor:
        var tmp = self._block_model(block)
        return tmp._single_block[S](prefix, x, single_mod, cos, sin, ctx)

    def forward_full[
        N_IMG: Int, N_TXT: Int, S: Int
    ](
        mut self,
        img_tokens: Tensor,
        txt_tokens: Tensor,
        timestep: Tensor,
        cos: Tensor,
        sin: Tensor,
        ctx: DeviceContext,
    ) raises -> Tensor:
        comptime assert S == N_IMG + N_TXT, "S must equal N_IMG + N_TXT"
        var cfg = self.shared.config
        var img = linear(img_tokens, self.shared._w(String("img_in.weight")), None, ctx)
        var txt = linear(txt_tokens, self.shared._w(String("txt_in.weight")), None, ctx)
        var vec = t_embedder(
            timestep,
            cfg.timestep_dim,
            self.shared._w(String("time_in.in_layer.weight")),
            None,
            self.shared._w(String("time_in.out_layer.weight")),
            None,
            ctx,
        )
        var vec_silu = silu(vec, ctx)
        var img_mod = linear(
            vec_silu,
            self.shared._w(String("double_stream_modulation_img.lin.weight")),
            None,
            ctx,
        )
        var txt_mod = linear(
            vec_silu,
            self.shared._w(String("double_stream_modulation_txt.lin.weight")),
            None,
            ctx,
        )
        var single_mod = linear(
            vec_silu,
            self.shared._w(String("single_stream_modulation.lin.weight")),
            None,
            ctx,
        )

        self.loader.config = OffloadConfig.synchronous_single()
        self.loader.prefetch(0)
        for bi in range(cfg.num_double):
            self.loader.prefetch_next(bi)
            var handle = self.loader.await_block(bi, ctx)
            var x = self._run_double[N_IMG, N_TXT, S](
                handle.block, handle.prefix, img, txt, img_mod, txt_mod, cos, sin, ctx
            )
            txt = slice(x, 1, 0, N_TXT, ctx)
            img = slice(x, 1, N_TXT, N_IMG, ctx)

        var x = concat(1, ctx, txt, img)
        for bi in range(cfg.num_single):
            var block_idx = cfg.num_double + bi
            self.loader.prefetch_next(block_idx)
            var handle = self.loader.await_block(block_idx, ctx)
            x = self._run_single[S](
                handle.block, handle.prefix, x, single_mod, cos, sin, ctx
            )

        var img_out = slice(x, 1, N_TXT, N_IMG, ctx)
        var final_mod = linear(
            vec_silu,
            self.shared._w(String("final_layer.adaLN_modulation.1.weight")),
            None,
            ctx,
        )
        var shift = self.shared._chunk_last(final_mod, 0, cfg.inner_dim, ctx)
        var scale = self.shared._chunk_last(final_mod, 1, cfg.inner_dim, ctx)
        var normed = self.shared._modulate_pre(img_out, shift, scale, ctx)
        return linear(
            normed, self.shared._w(String("final_layer.linear.weight")), None, ctx
        )

    def forward_full_cfg[
        N_IMG: Int, N_TXT: Int, S: Int
    ](
        mut self,
        img_tokens: Tensor,
        txt_pos_tokens: Tensor,
        txt_neg_tokens: Tensor,
        timestep: Tensor,
        cos: Tensor,
        sin: Tensor,
        ctx: DeviceContext,
    ) raises -> KleinCfgPreds:
        """Run positive and negative CFG branches through each streamed block
        before unloading it. This keeps the working set identical to
        forward_full() while roughly halving block H2D traffic for CFG."""
        comptime assert S == N_IMG + N_TXT, "S must equal N_IMG + N_TXT"
        var cfg = self.shared.config

        var img_pos = linear(
            img_tokens, self.shared._w(String("img_in.weight")), None, ctx
        )
        var img_neg = linear(
            img_tokens, self.shared._w(String("img_in.weight")), None, ctx
        )
        var txt_pos = linear(
            txt_pos_tokens, self.shared._w(String("txt_in.weight")), None, ctx
        )
        var txt_neg = linear(
            txt_neg_tokens, self.shared._w(String("txt_in.weight")), None, ctx
        )
        var vec = t_embedder(
            timestep,
            cfg.timestep_dim,
            self.shared._w(String("time_in.in_layer.weight")),
            None,
            self.shared._w(String("time_in.out_layer.weight")),
            None,
            ctx,
        )
        var vec_silu = silu(vec, ctx)
        var img_mod = linear(
            vec_silu,
            self.shared._w(String("double_stream_modulation_img.lin.weight")),
            None,
            ctx,
        )
        var txt_mod = linear(
            vec_silu,
            self.shared._w(String("double_stream_modulation_txt.lin.weight")),
            None,
            ctx,
        )
        var single_mod = linear(
            vec_silu,
            self.shared._w(String("single_stream_modulation.lin.weight")),
            None,
            ctx,
        )

        self.loader.config = OffloadConfig.synchronous_cfg_paired()
        self.loader.prefetch(0)
        for bi in range(cfg.num_double):
            self.loader.prefetch_next(bi)
            var handle = self.loader.await_block(bi, ctx)
            var x_pos = self._run_double[N_IMG, N_TXT, S](
                handle.block,
                handle.prefix,
                img_pos,
                txt_pos,
                img_mod,
                txt_mod,
                cos,
                sin,
                ctx,
            )
            var x_neg = self._run_double[N_IMG, N_TXT, S](
                handle.block,
                handle.prefix,
                img_neg,
                txt_neg,
                img_mod,
                txt_mod,
                cos,
                sin,
                ctx,
            )
            txt_pos = slice(x_pos, 1, 0, N_TXT, ctx)
            img_pos = slice(x_pos, 1, N_TXT, N_IMG, ctx)
            txt_neg = slice(x_neg, 1, 0, N_TXT, ctx)
            img_neg = slice(x_neg, 1, N_TXT, N_IMG, ctx)

        var x_pos = concat(1, ctx, txt_pos, img_pos)
        var x_neg = concat(1, ctx, txt_neg, img_neg)
        for bi in range(cfg.num_single):
            var block_idx = cfg.num_double + bi
            self.loader.prefetch_next(block_idx)
            var handle = self.loader.await_block(block_idx, ctx)
            x_pos = self._run_single[S](
                handle.block, handle.prefix, x_pos, single_mod, cos, sin, ctx
            )
            x_neg = self._run_single[S](
                handle.block, handle.prefix, x_neg, single_mod, cos, sin, ctx
            )

        var img_out_pos = slice(x_pos, 1, N_TXT, N_IMG, ctx)
        var img_out_neg = slice(x_neg, 1, N_TXT, N_IMG, ctx)
        var final_mod = linear(
            vec_silu,
            self.shared._w(String("final_layer.adaLN_modulation.1.weight")),
            None,
            ctx,
        )
        var shift = self.shared._chunk_last(final_mod, 0, cfg.inner_dim, ctx)
        var scale = self.shared._chunk_last(final_mod, 1, cfg.inner_dim, ctx)
        var normed_pos = self.shared._modulate_pre(img_out_pos, shift, scale, ctx)
        var normed_neg = self.shared._modulate_pre(img_out_neg, shift, scale, ctx)
        var pred_pos = linear(
            normed_pos, self.shared._w(String("final_layer.linear.weight")), None, ctx
        )
        var pred_neg = linear(
            normed_neg, self.shared._w(String("final_layer.linear.weight")), None, ctx
        )
        return KleinCfgPreds(pred_pos^, pred_neg^)


# ── Klein9BOffloadedTurbo ─────────────────────────────────────────────────────
# Phase 3: async-loader variant of Klein9BOffloaded.
#
# Identical API surface, uses TurboPlannedLoader instead of PlannedBlockLoader.
# The synchronous Klein9BOffloaded is UNCHANGED and remains the default.
# This struct exists solely for parity testing and future turbo production use.
#
# Minimal change: one new import (TurboPlannedLoader) + this struct.
# Klein's block math is NOT modified; _run_double/_run_single are forwarded
# from the shared field using the same pattern.

@fieldwise_init
struct Klein9BOffloadedTurbo(Movable):
    var shared: Klein9BDiT
    var loader: TurboPlannedLoader

    @staticmethod
    def load(path: String, ctx: DeviceContext) raises -> Klein9BOffloadedTurbo:
        var shared = Klein9BDiT.load_shared(path, ctx)
        var plan = build_klein9b_block_plan()
        var loader = TurboPlannedLoader.open(
            path, plan^, OffloadConfig.synchronous_single(), ctx
        )
        return Klein9BOffloadedTurbo(shared^, loader^)

    def _block_model(self, block: Block) -> Klein9BDiT:
        var weights = self.shared.weights.copy()
        var name_to_idx = self.shared.name_to_idx.copy()
        for ref e in block.items():
            name_to_idx[e.key] = len(weights)
            weights.append(e.value)
        return Klein9BDiT(weights^, name_to_idx^, self.shared.config)

    def _run_double[
        N_IMG: Int, N_TXT: Int, S: Int
    ](
        self,
        block: Block,
        prefix: String,
        img: Tensor,
        txt: Tensor,
        img_mod: Tensor,
        txt_mod: Tensor,
        cos: Tensor,
        sin: Tensor,
        ctx: DeviceContext,
    ) raises -> Tensor:
        var tmp = self._block_model(block)
        return tmp._double_block[N_IMG, N_TXT, S](
            prefix, img, txt, img_mod, txt_mod, cos, sin, ctx
        )

    def _run_single[S: Int](
        self,
        block: Block,
        prefix: String,
        x: Tensor,
        single_mod: Tensor,
        cos: Tensor,
        sin: Tensor,
        ctx: DeviceContext,
    ) raises -> Tensor:
        var tmp = self._block_model(block)
        return tmp._single_block[S](prefix, x, single_mod, cos, sin, ctx)

    def forward_full[
        N_IMG: Int, N_TXT: Int, S: Int
    ](
        mut self,
        img_tokens: Tensor,
        txt_tokens: Tensor,
        timestep: Tensor,
        cos: Tensor,
        sin: Tensor,
        ctx: DeviceContext,
    ) raises -> Tensor:
        comptime assert S == N_IMG + N_TXT, "S must equal N_IMG + N_TXT"
        var cfg = self.shared.config
        var img = linear(img_tokens, self.shared._w(String("img_in.weight")), None, ctx)
        var txt = linear(txt_tokens, self.shared._w(String("txt_in.weight")), None, ctx)
        var vec = t_embedder(
            timestep,
            cfg.timestep_dim,
            self.shared._w(String("time_in.in_layer.weight")),
            None,
            self.shared._w(String("time_in.out_layer.weight")),
            None,
            ctx,
        )
        var vec_silu = silu(vec, ctx)
        var img_mod = linear(
            vec_silu,
            self.shared._w(String("double_stream_modulation_img.lin.weight")),
            None,
            ctx,
        )
        var txt_mod = linear(
            vec_silu,
            self.shared._w(String("double_stream_modulation_txt.lin.weight")),
            None,
            ctx,
        )
        var single_mod = linear(
            vec_silu,
            self.shared._w(String("single_stream_modulation.lin.weight")),
            None,
            ctx,
        )

        self.loader.prefetch(0)
        for bi in range(cfg.num_double):
            self.loader.prefetch_next(bi)
            var handle = self.loader.await_block(bi, ctx)
            var x = self._run_double[N_IMG, N_TXT, S](
                handle.block, handle.prefix, img, txt, img_mod, txt_mod, cos, sin, ctx
            )
            txt = slice(x, 1, 0, N_TXT, ctx)
            img = slice(x, 1, N_TXT, N_IMG, ctx)

        var x = concat(1, ctx, txt, img)
        for bi in range(cfg.num_single):
            var block_idx = cfg.num_double + bi
            self.loader.prefetch_next(block_idx)
            var handle = self.loader.await_block(block_idx, ctx)
            x = self._run_single[S](
                handle.block, handle.prefix, x, single_mod, cos, sin, ctx
            )

        var img_out = slice(x, 1, N_TXT, N_IMG, ctx)
        var final_mod = linear(
            vec_silu,
            self.shared._w(String("final_layer.adaLN_modulation.1.weight")),
            None,
            ctx,
        )
        var shift = self.shared._chunk_last(final_mod, 0, cfg.inner_dim, ctx)
        var scale = self.shared._chunk_last(final_mod, 1, cfg.inner_dim, ctx)
        var normed = self.shared._modulate_pre(img_out, shift, scale, ctx)
        return linear(
            normed, self.shared._w(String("final_layer.linear.weight")), None, ctx
        )

    def forward_full_cfg[
        N_IMG: Int, N_TXT: Int, S: Int
    ](
        mut self,
        img_tokens: Tensor,
        txt_pos_tokens: Tensor,
        txt_neg_tokens: Tensor,
        timestep: Tensor,
        cos: Tensor,
        sin: Tensor,
        ctx: DeviceContext,
    ) raises -> KleinCfgPreds:
        """CFG-paired async forward: identical to Klein9BOffloaded.forward_full_cfg
        but uses TurboPlannedLoader for async H2D overlap."""
        comptime assert S == N_IMG + N_TXT, "S must equal N_IMG + N_TXT"
        var cfg = self.shared.config

        var img_pos = linear(
            img_tokens, self.shared._w(String("img_in.weight")), None, ctx
        )
        var img_neg = linear(
            img_tokens, self.shared._w(String("img_in.weight")), None, ctx
        )
        var txt_pos = linear(
            txt_pos_tokens, self.shared._w(String("txt_in.weight")), None, ctx
        )
        var txt_neg = linear(
            txt_neg_tokens, self.shared._w(String("txt_in.weight")), None, ctx
        )
        var vec = t_embedder(
            timestep,
            cfg.timestep_dim,
            self.shared._w(String("time_in.in_layer.weight")),
            None,
            self.shared._w(String("time_in.out_layer.weight")),
            None,
            ctx,
        )
        var vec_silu = silu(vec, ctx)
        var img_mod = linear(
            vec_silu,
            self.shared._w(String("double_stream_modulation_img.lin.weight")),
            None,
            ctx,
        )
        var txt_mod = linear(
            vec_silu,
            self.shared._w(String("double_stream_modulation_txt.lin.weight")),
            None,
            ctx,
        )
        var single_mod = linear(
            vec_silu,
            self.shared._w(String("single_stream_modulation.lin.weight")),
            None,
            ctx,
        )

        self.loader.prefetch(0)
        for bi in range(cfg.num_double):
            self.loader.prefetch_next(bi)
            var handle = self.loader.await_block(bi, ctx)
            var x_pos = self._run_double[N_IMG, N_TXT, S](
                handle.block,
                handle.prefix,
                img_pos,
                txt_pos,
                img_mod,
                txt_mod,
                cos,
                sin,
                ctx,
            )
            var x_neg = self._run_double[N_IMG, N_TXT, S](
                handle.block,
                handle.prefix,
                img_neg,
                txt_neg,
                img_mod,
                txt_mod,
                cos,
                sin,
                ctx,
            )
            txt_pos = slice(x_pos, 1, 0, N_TXT, ctx)
            img_pos = slice(x_pos, 1, N_TXT, N_IMG, ctx)
            txt_neg = slice(x_neg, 1, 0, N_TXT, ctx)
            img_neg = slice(x_neg, 1, N_TXT, N_IMG, ctx)

        var x_pos = concat(1, ctx, txt_pos, img_pos)
        var x_neg = concat(1, ctx, txt_neg, img_neg)
        for bi in range(cfg.num_single):
            var block_idx = cfg.num_double + bi
            self.loader.prefetch_next(block_idx)
            var handle = self.loader.await_block(block_idx, ctx)
            x_pos = self._run_single[S](
                handle.block, handle.prefix, x_pos, single_mod, cos, sin, ctx
            )
            x_neg = self._run_single[S](
                handle.block, handle.prefix, x_neg, single_mod, cos, sin, ctx
            )

        var img_out_pos = slice(x_pos, 1, N_TXT, N_IMG, ctx)
        var img_out_neg = slice(x_neg, 1, N_TXT, N_IMG, ctx)
        var final_mod = linear(
            vec_silu,
            self.shared._w(String("final_layer.adaLN_modulation.1.weight")),
            None,
            ctx,
        )
        var shift = self.shared._chunk_last(final_mod, 0, cfg.inner_dim, ctx)
        var scale = self.shared._chunk_last(final_mod, 1, cfg.inner_dim, ctx)
        var normed_pos = self.shared._modulate_pre(img_out_pos, shift, scale, ctx)
        var normed_neg = self.shared._modulate_pre(img_out_neg, shift, scale, ctx)
        var pred_pos = linear(
            normed_pos, self.shared._w(String("final_layer.linear.weight")), None, ctx
        )
        var pred_neg = linear(
            normed_neg, self.shared._w(String("final_layer.linear.weight")), None, ctx
        )
        return KleinCfgPreds(pred_pos^, pred_neg^)

# models/dit/flux1_dit.mojo — FLUX.1 Dev/Schnell DiT transformer (pure Mojo).
#
# Inference-only port of
#   /home/alex/EriDiffusion/inference-flame/src/models/flux1_dit.rs (read FULL, 1094 L)
# which is a verbatim port of black-forest-labs-flux src/flux/{model,modules/layers,math}.py.
#
# 12B model: 19 double-stream blocks + 38 single-stream blocks.
#
# Differences from Klein (flux2) — taken from flux1_dit.rs, NOT inferred:
#   - HAS biases EVERYWHERE (img_in/txt_in, all qkv/proj/mlp, mod lins, final).
#   - PER-BLOCK modulation (img_mod.lin + txt_mod.lin per double block;
#     modulation.lin per single block) — NOT a shared double/single modulation.
#   - GELU MLP (4x ratio, mlp.0 -> gelu -> mlp.2) instead of Klein's SwiGLU (6x).
#   - HAS guidance_in (Dev only, guidance-distilled) + vector_in (CLIP pooled
#     768 -> 3072). vec = time_mlp(t) + guidance_mlp(g) + vector_mlp(clip_pooled).
#   - in_channels=64 (img_in: 3072 x 64), joint_attention_dim=4096 (T5; txt_in: 3072 x 4096).
#   - 3-axis RoPE (16,56,56) interleaved-pair, theta 10000.
#   - timestep_embedding scales t by 1000 (time_factor) BEFORE the sinusoid.
#
# modulate_pre(x, shift, scale) = (1 + scale) * LayerNorm_no_affine(x, 1e-6) + shift.
# Double block: img_mod/txt_mod each chunk into (shift1,scale1,gate1,shift2,scale2,gate2).
#   modulate_pre -> qkv linear(+bias) -> split q/k/v -> q/k RMSNorm(1e-6) ->
#   cat([txt,img]) on seq -> RoPE on q,k -> SDPA -> split back -> proj(+bias) ->
#   gated residual1 -> modulate_pre2 -> mlp.0(+bias) -> gelu -> mlp.2(+bias) ->
#   gated residual2.
# Single block: modulation chunks (shift,scale,gate). modulate_pre -> linear1(+bias)
#   [QKV | MLP_up(4x)] -> split q/k/v -> q/k RMSNorm -> RoPE -> SDPA -> permute ->
#   gelu(mlp_up) -> linear2([attn|mlp_act], +bias) -> gated residual.
# Final: silu(vec) -> adaLN_modulation.1(+bias) -> (shift,scale) -> modulate_pre ->
#   final_layer.linear(+bias) -> velocity [B, N_img, 64].
#
# Foundation ops reused: linear (with bias), layer_norm (for modulate_pre affine-
# free path via ones/zeros), rms_norm, rope_interleaved, sdpa (zero additive mask),
# gelu (tanh-approx; BFL uses nn.GELU(approximate="tanh")), elementwise.modulate,
# elementwise.residual_gate, embeddings.t_embedder, tensor_algebra.{slice,reshape,
# concat,add}. No new foundation op required.
#
# Sequence lengths are comptime params on the forward (N_IMG, N_TXT, S) so the
# comptime-shaped sdpa monomorphizes (mirrors Klein's forward_full contract).
# Weights load via BlockLoader (double_blocks.i / single_blocks.i) + a resident
# shared set, mirroring Klein9BOffloaded.
#
# Mojo 1.0.0b1, NVIDIA GPU. BF16 storage, F32 accumulation.

from std.gpu.host import DeviceContext
from std.math import sqrt, cos as fcos, sin as fsin, exp as fexp, log as flog
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import rms_norm, layer_norm
from serenitymojo.ops.activations import silu, gelu
from serenitymojo.ops.elementwise import modulate, residual_gate
from serenitymojo.ops.rope import rope_interleaved
from serenitymojo.ops.attention import sdpa
from serenitymojo.ops.embeddings import t_embedder
from serenitymojo.ops.tensor_algebra import reshape, slice, concat, add
from serenitymojo.offload.block_loader import BlockLoader, Block, unload_block


@fieldwise_init
struct Flux1Config(Copyable, Movable, ImplicitlyCopyable):
    var num_double: Int
    var num_single: Int
    var inner_dim: Int
    var num_heads: Int
    var head_dim: Int
    var in_channels: Int
    var joint_attention_dim: Int
    var mlp_ratio: Int
    var timestep_dim: Int
    var has_guidance: Bool
    var vector_dim: Int
    var rope_theta: Float64

    @staticmethod
    def dev() -> Flux1Config:
        # 19 double, 38 single, inner 3072, 24 heads, head_dim 128,
        # in_channels 64, joint_attn_dim 4096, mlp_ratio 4, timestep_dim 256,
        # guidance distilled, vector_dim 768, rope_theta 10000.
        return Flux1Config(
            19, 38, 3072, 24, 128, 64, 4096, 4, 256, True, 768, 10000.0
        )

    @staticmethod
    def schnell() -> Flux1Config:
        return Flux1Config(
            19, 38, 3072, 24, 128, 64, 4096, 4, 256, False, 768, 10000.0
        )


def _load_weight(st: SafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    return Tensor.from_view(tv, ctx)


def flux1_shared_keys(has_guidance: Bool) -> List[String]:
    var keys = List[String]()
    keys.append(String("img_in.weight"))
    keys.append(String("img_in.bias"))
    keys.append(String("txt_in.weight"))
    keys.append(String("txt_in.bias"))
    keys.append(String("time_in.in_layer.weight"))
    keys.append(String("time_in.in_layer.bias"))
    keys.append(String("time_in.out_layer.weight"))
    keys.append(String("time_in.out_layer.bias"))
    if has_guidance:
        keys.append(String("guidance_in.in_layer.weight"))
        keys.append(String("guidance_in.in_layer.bias"))
        keys.append(String("guidance_in.out_layer.weight"))
        keys.append(String("guidance_in.out_layer.bias"))
    keys.append(String("vector_in.in_layer.weight"))
    keys.append(String("vector_in.in_layer.bias"))
    keys.append(String("vector_in.out_layer.weight"))
    keys.append(String("vector_in.out_layer.bias"))
    keys.append(String("final_layer.adaLN_modulation.1.weight"))
    keys.append(String("final_layer.adaLN_modulation.1.bias"))
    keys.append(String("final_layer.linear.weight"))
    keys.append(String("final_layer.linear.bias"))
    return keys^


@fieldwise_init
struct Flux1DiT(Movable):
    var weights: List[ArcPointer[Tensor]]
    var name_to_idx: Dict[String, Int]
    var config: Flux1Config

    @staticmethod
    def load_shared(path: String, config: Flux1Config, ctx: DeviceContext) raises -> Flux1DiT:
        var st = SafeTensors.open(path)
        var keys = flux1_shared_keys(config.has_guidance)
        var weights = List[ArcPointer[Tensor]]()
        var name_to_idx = Dict[String, Int]()
        for i in range(len(keys)):
            var name = keys[i]
            var t = _load_weight(st, name, ctx)
            name_to_idx[name] = len(weights)
            weights.append(ArcPointer(t^))
        return Flux1DiT(weights^, name_to_idx^, config)

    def _w(self, name: String) raises -> ref [self.weights] Tensor:
        if name not in self.name_to_idx:
            raise Error(String("missing FLUX.1 weight: ") + name)
        var idx = self.name_to_idx[name]
        return self.weights[idx][]

    # ── helpers ───────────────────────────────────────────────────────────
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

    def _zeros_mask[S: Int](self, dtype: STDtype, ctx: DeviceContext) raises -> Tensor:
        # Full-attention additive mask [1, num_heads, S, S] of zeros.
        comptime H = 24
        var n = H * S * S
        var dev = ctx.enqueue_create_buffer[DType.uint8](n * dtype.byte_size())
        ctx.enqueue_memset[DType.uint8](dev, 0)
        ctx.synchronize()
        var sh = List[Int]()
        sh.append(1)
        sh.append(H)
        sh.append(S)
        sh.append(S)
        return Tensor(dev^, sh^, dtype)

    # chunk #idx of `chunk` elements out of last dim, flatten to [chunk] (mod
    # vec is [1, 6*d] or [1, 2*d]/[1, 3*d]; we keep it [chunk] for the per-channel
    # modulate/residual_gate broadcast).
    def _chunk_last(
        self, x: Tensor, idx: Int, chunk: Int, ctx: DeviceContext
    ) raises -> Tensor:
        var s = x.shape()
        var dim = len(s) - 1
        var part = slice(x, dim, idx * chunk, chunk, ctx)
        var out_shape = List[Int]()
        out_shape.append(chunk)
        return reshape(part, out_shape^, ctx)

    # modulate_pre: (1 + scale) * LayerNorm_no_affine(x) + shift.
    # The foundation layer_norm applies an affine; we pass ones/zeros so it is
    # affine-free, then `modulate` applies (1+scale)*x + shift per channel.
    def _modulate_pre(
        self, x: Tensor, shift: Tensor, scale: Tensor, ctx: DeviceContext
    ) raises -> Tensor:
        var dtype = x.dtype()
        var d = x.shape()[len(x.shape()) - 1]
        var ones = self._ones(d, dtype, ctx)
        var zeros = self._zeros_vec(d, dtype, ctx)
        var normed = layer_norm(x, ones, zeros, 1.0e-6, ctx)
        return modulate(normed, scale, shift, ctx)

    # split one of q/k/v out of a packed [1, n, 3*inner] qkv -> BSHD [1,n,H,Dh].
    def _qkv_part(self, qkv: Tensor, part: Int, ctx: DeviceContext) raises -> Tensor:
        var cfg = self.config
        var inner = cfg.inner_dim
        var qkv_part = slice(qkv, 2, part * inner, inner, ctx)
        var n = qkv_part.shape()[1]
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
        mask: Tensor,
        ctx: DeviceContext,
    ) raises -> Tensor:
        # `mask` is the full-attention zero bias [1,24,S,S], built ONCE in forward
        # and borrowed here (the Rust reference passes None — a zeros additive mask
        # is a numeric no-op; sdpa mandates the [B,H,S,S] arg). Borrowing it avoids
        # the ~1 GiB alloc/memset per attention call (57 blocks x 20 steps).
        var q_roped = rope_interleaved(q, cos, sin, ctx)
        var k_roped = rope_interleaved(k, cos, sin, ctx)
        return sdpa[1, S, 24, 128](
            q_roped, k_roped, v, mask, Float32(1.0) / sqrt(Float32(128)), ctx
        )

    # ── double-stream block ─────────────────────────────────────────────────
    def _double_block[
        N_IMG: Int, N_TXT: Int, S: Int
    ](
        self,
        p: String,
        weights: Flux1DiT,
        img: Tensor,
        txt: Tensor,
        img_mod: Tensor,
        txt_mod: Tensor,
        cos: Tensor,
        sin: Tensor,
        mask: Tensor,
        ctx: DeviceContext,
    ) raises -> Tensor:
        # Returns cat([txt_final, img_final]) on the seq axis (txt first), so the
        # caller slices [0:N_TXT] -> txt, [N_TXT:N_TXT+N_IMG] -> img (mirrors the
        # Klein _double_block contract; avoids returning a tuple of Movable Tensors).
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

        var img_qkv = linear(
            img_norm,
            weights._w(p + ".img_attn.qkv.weight"),
            Optional[Tensor](_clone(weights._w(p + ".img_attn.qkv.bias"), ctx)),
            ctx,
        )
        var txt_qkv = linear(
            txt_norm,
            weights._w(p + ".txt_attn.qkv.weight"),
            Optional[Tensor](_clone(weights._w(p + ".txt_attn.qkv.bias"), ctx)),
            ctx,
        )
        var img_q = self._qkv_part(img_qkv, 0, ctx)
        var img_k = self._qkv_part(img_qkv, 1, ctx)
        var img_v = self._qkv_part(img_qkv, 2, ctx)
        var txt_q = self._qkv_part(txt_qkv, 0, ctx)
        var txt_k = self._qkv_part(txt_qkv, 1, ctx)
        var txt_v = self._qkv_part(txt_qkv, 2, ctx)

        img_q = rms_norm(img_q, weights._w(p + ".img_attn.norm.query_norm.scale"), 1.0e-6, ctx)
        img_k = rms_norm(img_k, weights._w(p + ".img_attn.norm.key_norm.scale"), 1.0e-6, ctx)
        txt_q = rms_norm(txt_q, weights._w(p + ".txt_attn.norm.query_norm.scale"), 1.0e-6, ctx)
        txt_k = rms_norm(txt_k, weights._w(p + ".txt_attn.norm.key_norm.scale"), 1.0e-6, ctx)

        # cat([txt, img]) on the seq axis (axis 1 of BSHD).
        var q = concat(1, ctx, txt_q, img_q)
        var k = concat(1, ctx, txt_k, img_k)
        var v = concat(1, ctx, txt_v, img_v)
        var att = self._attn_rope_only[S](q, k, v, cos, sin, mask, ctx)

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

        var img_attn = linear(
            img_att,
            weights._w(p + ".img_attn.proj.weight"),
            Optional[Tensor](_clone(weights._w(p + ".img_attn.proj.bias"), ctx)),
            ctx,
        )
        var txt_attn = linear(
            txt_att,
            weights._w(p + ".txt_attn.proj.weight"),
            Optional[Tensor](_clone(weights._w(p + ".txt_attn.proj.bias"), ctx)),
            ctx,
        )
        var img1 = residual_gate(img, img_gate1, img_attn, ctx)
        var txt1 = residual_gate(txt, txt_gate1, txt_attn, ctx)

        # MLP path: modulate2 -> mlp.0(+bias) -> gelu -> mlp.2(+bias).
        var img_mlp_in = self._modulate_pre(img1, img_shift2, img_scale2, ctx)
        var txt_mlp_in = self._modulate_pre(txt1, txt_shift2, txt_scale2, ctx)
        var img_mlp = linear(
            img_mlp_in,
            weights._w(p + ".img_mlp.0.weight"),
            Optional[Tensor](_clone(weights._w(p + ".img_mlp.0.bias"), ctx)),
            ctx,
        )
        img_mlp = gelu(img_mlp, ctx)
        img_mlp = linear(
            img_mlp,
            weights._w(p + ".img_mlp.2.weight"),
            Optional[Tensor](_clone(weights._w(p + ".img_mlp.2.bias"), ctx)),
            ctx,
        )
        var txt_mlp = linear(
            txt_mlp_in,
            weights._w(p + ".txt_mlp.0.weight"),
            Optional[Tensor](_clone(weights._w(p + ".txt_mlp.0.bias"), ctx)),
            ctx,
        )
        txt_mlp = gelu(txt_mlp, ctx)
        txt_mlp = linear(
            txt_mlp,
            weights._w(p + ".txt_mlp.2.weight"),
            Optional[Tensor](_clone(weights._w(p + ".txt_mlp.2.bias"), ctx)),
            ctx,
        )
        var img_final = residual_gate(img1, img_gate2, img_mlp, ctx)
        var txt_final = residual_gate(txt1, txt_gate2, txt_mlp, ctx)
        return concat(1, ctx, txt_final, img_final)

    # ── single-stream block ─────────────────────────────────────────────────
    def _single_block[S: Int](
        self,
        p: String,
        weights: Flux1DiT,
        x: Tensor,
        single_mod: Tensor,
        cos: Tensor,
        sin: Tensor,
        mask: Tensor,
        ctx: DeviceContext,
    ) raises -> Tensor:
        var d = self.config.inner_dim
        var mlp_hidden = d * self.config.mlp_ratio  # 4x = 12288

        var shift = self._chunk_last(single_mod, 0, d, ctx)
        var scale = self._chunk_last(single_mod, 1, d, ctx)
        var gate = self._chunk_last(single_mod, 2, d, ctx)

        var x_norm = self._modulate_pre(x, shift, scale, ctx)
        # linear1: [QKV (3*d) | MLP_up (mlp_hidden)] = 3*3072 + 12288 = 21504.
        var fused = linear(
            x_norm,
            weights._w(p + ".linear1.weight"),
            Optional[Tensor](_clone(weights._w(p + ".linear1.bias"), ctx)),
            ctx,
        )
        var qkv = slice(fused, 2, 0, 3 * d, ctx)
        var mlp_in = slice(fused, 2, 3 * d, mlp_hidden, ctx)

        var q = self._qkv_part(qkv, 0, ctx)
        var k = self._qkv_part(qkv, 1, ctx)
        var v = self._qkv_part(qkv, 2, ctx)
        q = rms_norm(q, weights._w(p + ".norm.query_norm.scale"), 1.0e-6, ctx)
        k = rms_norm(k, weights._w(p + ".norm.key_norm.scale"), 1.0e-6, ctx)

        var att = self._attn_rope_only[S](q, k, v, cos, sin, mask, ctx)
        var att_shape = List[Int]()
        att_shape.append(1)
        att_shape.append(S)
        att_shape.append(d)
        var att_flat = reshape(att, att_shape^, ctx)

        var mlp_act = gelu(mlp_in, ctx)
        # linear2: [attn (d) | mlp_act (mlp_hidden)] = 3072 + 12288 = 15360.
        var fused_in = concat(2, ctx, att_flat, mlp_act)
        var out = linear(
            fused_in,
            weights._w(p + ".linear2.weight"),
            Optional[Tensor](_clone(weights._w(p + ".linear2.bias"), ctx)),
            ctx,
        )
        return residual_gate(x, gate, out, ctx)

    # ── vec = time_mlp(t*1000) + guidance_mlp(g*1000) + vector_mlp(clip_pooled) ──
    # The foundation t_embedder does timestep_embedding(dim) -> Lin -> SiLU -> Lin.
    # BFL applies time_factor=1000 INSIDE timestep_embedding; the foundation does
    # NOT, so the caller passes pre-scaled t (t*1000) here. Returns [1, inner].
    def _embed_vec(
        self,
        timestep: Tensor,
        guidance: Optional[Tensor],
        vector: Tensor,
        ctx: DeviceContext,
    ) raises -> Tensor:
        var cfg = self.config
        var vec = t_embedder(
            timestep,
            cfg.timestep_dim,
            self._w(String("time_in.in_layer.weight")),
            Optional[Tensor](_clone(self._w(String("time_in.in_layer.bias")), ctx)),
            self._w(String("time_in.out_layer.weight")),
            Optional[Tensor](_clone(self._w(String("time_in.out_layer.bias")), ctx)),
            ctx,
        )
        if cfg.has_guidance and guidance:
            var g_vec = t_embedder(
                guidance.value(),
                cfg.timestep_dim,
                self._w(String("guidance_in.in_layer.weight")),
                Optional[Tensor](_clone(self._w(String("guidance_in.in_layer.bias")), ctx)),
                self._w(String("guidance_in.out_layer.weight")),
                Optional[Tensor](_clone(self._w(String("guidance_in.out_layer.bias")), ctx)),
                ctx,
            )
            vec = add(vec, g_vec, ctx)
        # vector_in is an MLPEmbedder over CLIP pooled (768 -> 3072): in_layer ->
        # silu -> out_layer. Reuse the t_embedder MLP shape by feeding the pooled
        # vector directly through linear/silu/linear (NOT a sinusoid).
        var v_in = linear(
            vector,
            self._w(String("vector_in.in_layer.weight")),
            Optional[Tensor](_clone(self._w(String("vector_in.in_layer.bias")), ctx)),
            ctx,
        )
        var v_act = silu(v_in, ctx)
        var v_vec = linear(
            v_act,
            self._w(String("vector_in.out_layer.weight")),
            Optional[Tensor](_clone(self._w(String("vector_in.out_layer.bias")), ctx)),
            ctx,
        )
        vec = add(vec, v_vec, ctx)
        return vec^

    def _final_layer(
        self, img_out: Tensor, vec_silu: Tensor, ctx: DeviceContext
    ) raises -> Tensor:
        var cfg = self.config
        var mods = linear(
            vec_silu,
            self._w(String("final_layer.adaLN_modulation.1.weight")),
            Optional[Tensor](_clone(self._w(String("final_layer.adaLN_modulation.1.bias")), ctx)),
            ctx,
        )
        var shift = self._chunk_last(mods, 0, cfg.inner_dim, ctx)
        var scale = self._chunk_last(mods, 1, cfg.inner_dim, ctx)
        var normed = self._modulate_pre(img_out, shift, scale, ctx)
        return linear(
            normed,
            self._w(String("final_layer.linear.weight")),
            Optional[Tensor](_clone(self._w(String("final_layer.linear.bias")), ctx)),
            ctx,
        )


# ── RoPE table build (FLUX.1 3-axis interleaved) ────────────────────────────
# BFL EmbedND: per axis with dim in axes_dims_rope=[16,56,56] (sum 128 = head_dim),
#   omega_i = 1 / theta^(2i/axis_dim)  for i in [0, axis_dim/2),
#   angle = pos_axis * omega_i,  cos/sin per (token, axis-half-i).
# The per-axis half-tables concatenate to [N, 64] = head_dim/2. ids layout:
#   txt tokens: ids = (0,0,0); img tokens: ids = (0, row, col) over an h2 x w2 grid.
# Foundation rope_interleaved consumes cos/sin [rows, head_dim/2] where rows ==
# the flattened leading dims of q [1, S, H, Dh] -> rows = S*H. So we TILE the
# per-token [S,64] table across the H heads (token-major, head-minor): row index
# = tok*H + head shares the token's angles. Mirrors Klein build_klein_rope_tables.
def build_flux1_rope_tables[
    N_IMG: Int, N_TXT: Int, H: Int, DH: Int
](
    img_h2: Int, img_w2: Int, ctx: DeviceContext, dtype: STDtype
) raises -> Tuple[Tensor, Tensor]:
    comptime assert DH == 128, "FLUX.1 head dim must be 128"
    comptime S = N_IMG + N_TXT
    comptime HALF = DH // 2  # 64
    if img_h2 * img_w2 != N_IMG:
        raise Error("build_flux1_rope_tables: img_h2*img_w2 must equal N_IMG")

    # axes_dims_rope = [16, 56, 56]; halves = [8, 28, 28] -> sum 64.
    var axes = List[Int]()
    axes.append(16)
    axes.append(56)
    axes.append(56)

    var cos_vals = List[Float32]()
    var sin_vals = List[Float32]()

    for tok in range(S):
        # position ids per FLUX 3-axis layout. txt: (0,0,0). img: (0, row, col).
        var pos0 = 0  # axis 0 always 0 for both txt and img
        var pos1 = 0
        var pos2 = 0
        if tok >= N_TXT:
            var idx = tok - N_TXT
            pos1 = idx // img_w2
            pos2 = idx % img_w2

        # Build the per-token [HALF] angle list once, then tile across H heads.
        var tok_cos = List[Float32]()
        var tok_sin = List[Float32]()
        for axis in range(3):
            var axis_dim = axes[axis]
            var half = axis_dim // 2
            var pos = pos0
            if axis == 1:
                pos = pos1
            elif axis == 2:
                pos = pos2
            var log_theta = flog(Float32(10000.0))
            for i in range(half):
                # omega = 1 / theta^(2i/axis_dim) = exp(-ln(theta)*2i/axis_dim)
                var inv_freq = fexp(-log_theta * Float32(2 * i) / Float32(axis_dim))
                var angle = Float32(pos) * inv_freq
                tok_cos.append(fcos(angle))
                tok_sin.append(fsin(angle))
        # tile across heads (rows = tok*H + head all share the token angles).
        for _h in range(H):
            for i in range(HALF):
                cos_vals.append(tok_cos[i])
                sin_vals.append(tok_sin[i])

    var sh = List[Int]()
    sh.append(S * H)
    sh.append(HALF)
    return (
        Tensor.from_host(cos_vals, sh.copy(), dtype, ctx),
        Tensor.from_host(sin_vals, sh^, dtype, ctx),
    )


# ── deep-copy a weight Tensor (Movable-not-Copyable -> clone for owned bias) ──
def _clone(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var dev = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    ctx.enqueue_copy(dst_buf=dev, src_buf=x.buf)
    ctx.synchronize()
    return Tensor(dev^, x.shape(), x.dtype())


# ── Offloaded FLUX.1 DiT (BlockLoader streams double/single blocks) ──────────
@fieldwise_init
struct Flux1Offloaded(Movable):
    var shared: Flux1DiT
    var loader: BlockLoader

    @staticmethod
    def load(path: String, config: Flux1Config, ctx: DeviceContext) raises -> Flux1Offloaded:
        var shared = Flux1DiT.load_shared(path, config, ctx)
        var loader = BlockLoader.open(path)
        return Flux1Offloaded(shared^, loader^)

    # Build a transient Flux1DiT whose weight table includes the streamed block.
    def _block_model(self, block: Block) -> Flux1DiT:
        var weights = self.shared.weights.copy()
        var name_to_idx = self.shared.name_to_idx.copy()
        for ref e in block.items():
            name_to_idx[e.key] = len(weights)
            weights.append(e.value)
        return Flux1DiT(weights^, name_to_idx^, self.shared.config)

    # forward: img [1,N_IMG,64], txt [1,N_TXT,4096], timestep [1] (already *1000
    # by the CALLER per BFL time_factor), guidance [1] (already *1000), vector
    # [1,768] CLIP pooled, cos/sin RoPE tables [S*H, 64]. Returns [1,N_IMG,64].
    def forward[
        N_IMG: Int, N_TXT: Int, S: Int
    ](
        self,
        img_tokens: Tensor,
        txt_tokens: Tensor,
        timestep: Tensor,
        guidance: Optional[Tensor],
        vector: Tensor,
        cos: Tensor,
        sin: Tensor,
        ctx: DeviceContext,
    ) raises -> Tensor:
        comptime assert S == N_IMG + N_TXT, "S must equal N_IMG + N_TXT"
        var cfg = self.shared.config

        # input projections (with bias).
        var img = linear(
            img_tokens,
            self.shared._w(String("img_in.weight")),
            Optional[Tensor](_clone(self.shared._w(String("img_in.bias")), ctx)),
            ctx,
        )
        var txt = linear(
            txt_tokens,
            self.shared._w(String("txt_in.weight")),
            Optional[Tensor](_clone(self.shared._w(String("txt_in.bias")), ctx)),
            ctx,
        )

        # vec = time + guidance + vector.
        var vec = self.shared._embed_vec(timestep, guidance, vector, ctx)

        # Full-attention zero mask [1,24,S,S], built ONCE and borrowed by every
        # block's attention (Rust passes None; zeros == no-op). Avoids the per-call
        # ~1 GiB alloc/memset across 57 blocks. q/k after RoPE keep `img`'s dtype,
        # so build the mask in that dtype to satisfy sdpa's q/mask dtype check.
        var mask = self.shared._zeros_mask[S](img.dtype(), ctx)

        # double blocks: per-block img_mod/txt_mod = silu(vec) -> img_mod.lin / txt_mod.lin.
        for bi in range(cfg.num_double):
            var prefix = String("double_blocks.") + String(bi)
            self.loader.prefetch_block(prefix)
            var block = self.loader.load_block(prefix, ctx)
            var bm = self._block_model(block)
            var vec_silu = silu(vec, ctx)
            var img_mod = linear(
                vec_silu,
                bm._w(prefix + ".img_mod.lin.weight"),
                Optional[Tensor](_clone(bm._w(prefix + ".img_mod.lin.bias"), ctx)),
                ctx,
            )
            var txt_mod = linear(
                vec_silu,
                bm._w(prefix + ".txt_mod.lin.weight"),
                Optional[Tensor](_clone(bm._w(prefix + ".txt_mod.lin.bias"), ctx)),
                ctx,
            )
            var merged = bm._double_block[N_IMG, N_TXT, S](
                prefix, bm, img, txt, img_mod, txt_mod, cos, sin, mask, ctx
            )
            txt = slice(merged, 1, 0, N_TXT, ctx)
            img = slice(merged, 1, N_TXT, N_IMG, ctx)
            unload_block(block^)

        # merge cat([txt, img]) for the single-stream blocks.
        var x = concat(1, ctx, txt, img)
        for bi in range(cfg.num_single):
            var prefix = String("single_blocks.") + String(bi)
            self.loader.prefetch_block(prefix)
            var block = self.loader.load_block(prefix, ctx)
            var bm = self._block_model(block)
            var vec_silu = silu(vec, ctx)
            var single_mod = linear(
                vec_silu,
                bm._w(prefix + ".modulation.lin.weight"),
                Optional[Tensor](_clone(bm._w(prefix + ".modulation.lin.bias"), ctx)),
                ctx,
            )
            x = bm._single_block[S](prefix, bm, x, single_mod, cos, sin, mask, ctx)
            unload_block(block^)

        # extract image tokens (txt first in the merged seq) and run final layer.
        var img_out = slice(x, 1, N_TXT, N_IMG, ctx)
        var vec_silu_final = silu(vec, ctx)
        return self.shared._final_layer(img_out, vec_silu_final, ctx)

# models/dit/mageflow_dit.mojo — MageFlow DiT double-stream block (GPU, inference-only).
#
# MageFlow's DiT (Mage-Flow-Edit-Turbo) is a Qwen-Image-family MMDiT: 12 double
# stream blocks, 0 single blocks. hidden 3072, 24 heads, head_dim 128,
# context_in_dim 2560, in/out 128, mlp_ratio 4.0 (FFN 12288), axes [16,56,56]
# theta 10000, qkv_bias, gelu-approximate MLP, LayerNorm-no-affine +
# 6-way AdaLN modulation (shift,scale,gate ×2), QK-RMSNorm(head_dim, eps 1e-6),
# joint attention with concat order [txt, img].
#
# ── The ONE delta vs serenitymojo's qwenimage_dit block ──────────────────────
# MageFlow config: `apply_text_rotary_emb=false`, `rope_type="msrope"`. TEXT
# TOKENS ARE NOT ROTATED — only image tokens get the 3-axis msrope. Everything
# else (modulation split order shift/scale/gate, QK-RMSNorm, txt-then-img
# concat, joint SDPA, split, output projections, gated residual, gelu-tanh FFN)
# is byte-for-byte the Qwen-Image double block. The weight KEY NAMES are also
# identical (transformer_blocks.N.{img_mod.1, attn.to_q/to_k/to_v,
# attn.add_{q,k,v}_proj, attn.norm_{q,k}/norm_added_{q,k}, attn.to_out.0,
# attn.to_add_out, img_mlp.net.{0.proj,2}, txt_mod.1, txt_mlp.net.{0.proj,2}}).
#
# Therefore this module REUSES qwenimage_dit.QwenImageDit._block_forward
# VERBATIM and only supplies a different RoPE table: image rows get the same
# msrope angles the qwenimage builder produces (p0=f_idx, p1=h_idx-H/2,
# p2=w_idx-W/2 — matches MageFlowEmbedRope scale_rope=True for even H,W), while
# text rows are set to cos=1, sin=0 (identity rotation) so the concat-then-rope
# path leaves the text q/k unchanged. rope_interleaved with cos=1/sin=0 is the
# identity in every convention (angle 0).
#
# QwenImageConfig.qwen_image() defaults already match MageFlow's block dims
# (inner_dim 3072, num_heads 24, head_dim 128, eps 1e-6); only num_layers (60 vs
# 12) differs and that field is unused by _block_forward.
#
# Mojo 1.0.0b1, NVIDIA GPU. BF16 storage, F32 accumulation in foundation ops.

from max.gpu.host import DeviceContext
from std.math import cos as fcos, sin as fsin, pow as fpow, exp as fexp, log as flog

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.activations import silu
from serenitymojo.ops.norm import rms_norm
from serenitymojo.ops.tensor_algebra import reshape
from serenitymojo.models.dit.qwenimage_dit import QwenImageDit, QwenImageConfig


# ── Config (documentation of the MageFlow block hyperparameters) ─────────────
@fieldwise_init
struct MageFlowConfig(Copyable, Movable, ImplicitlyCopyable):
    var depth: Int               # 12 double_stream / 0 single
    var inner_dim: Int           # 3072
    var num_heads: Int           # 24
    var head_dim: Int            # 128
    var in_channels: Int         # 128
    var out_channels: Int        # 128
    var context_in_dim: Int      # 2560
    var mlp_hidden: Int          # 12288 (mlp_ratio 4.0)
    var axis0: Int               # 16  (frame)
    var axis1: Int               # 56  (height)
    var axis2: Int               # 56  (width)
    var rope_theta: Float64      # 10000.0
    var timestep_dim: Int        # 256
    var eps: Float32             # 1e-6

    @staticmethod
    def mage_flow() -> MageFlowConfig:
        return MageFlowConfig(
            12, 3072, 24, 128, 128, 128, 2560, 12288,
            16, 56, 56, Float64(10000.0), 256, Float32(1e-6),
        )


# ── RoPE tables: image-only msrope, text rows identity (text NOT roped) ──────
# Token order is [txt (N_txt), img (frame*height*width)] to match the qwenimage
# block's txt-then-img concat. Output tables are [(N_txt+N_img)*heads, 64]
# (angle repeated across heads); rope_interleaved consumes them directly.
#
# Image angle per axis for token (f,h,w):  p0=f, p1=h-H/2, p2=w-W/2, freq_i =
# 1/theta^(2i/axis_dim). This reproduces MageFlowEmbedRope(scale_rope=True) for
# even H,W. Text rows: cos=1, sin=0 => identity (apply_text_rotary_emb=false).
def build_mageflow_rope_tables(
    frame: Int,
    height: Int,
    width: Int,
    txt_seq_len: Int,
    heads: Int,
    theta: Float64,
    dtype: STDtype,
    ctx: DeviceContext,
) raises -> Tuple[Tensor, Tensor]:
    var axes = [16, 56, 56]
    var total_half = 0
    for a in range(3):
        total_half += axes[a] // 2  # 8 + 28 + 28 = 64

    var n_img = frame * height * width
    var n_total = txt_seq_len + n_img

    # per-axis inverse-frequency tables
    var freq0 = List[Float32]()
    var freq1 = List[Float32]()
    var freq2 = List[Float32]()
    for axis in range(3):
        var axis_dim = axes[axis]
        var half = axis_dim // 2
        for i in range(half):
            var scale = Float64(2 * i) / Float64(axis_dim)
            var f = Float32(1.0 / fpow(theta, scale))
            if axis == 0:
                freq0.append(f)
            elif axis == 1:
                freq1.append(f)
            else:
                freq2.append(f)

    var cos_vals = List[Float32]()
    var sin_vals = List[Float32]()
    for tok in range(n_total):
        if tok < txt_seq_len:
            # text token: identity rotation (NOT roped)
            for _h in range(heads):
                for _i in range(total_half):
                    cos_vals.append(Float32(1.0))
                    sin_vals.append(Float32(0.0))
        else:
            var img_idx = tok - txt_seq_len
            var f_idx = img_idx // (height * width)
            var rem = img_idx % (height * width)
            var h_idx = rem // width
            var w_idx = rem % width
            var p0 = Float32(f_idx)
            var p1 = Float32(h_idx) - Float32(height) / Float32(2.0)
            var p2 = Float32(w_idx) - Float32(width) / Float32(2.0)
            for _h in range(heads):
                for i in range(len(freq0)):
                    var ang = p0 * freq0[i]
                    cos_vals.append(fcos(ang))
                    sin_vals.append(fsin(ang))
                for i in range(len(freq1)):
                    var ang = p1 * freq1[i]
                    cos_vals.append(fcos(ang))
                    sin_vals.append(fsin(ang))
                for i in range(len(freq2)):
                    var ang = p2 * freq2[i]
                    cos_vals.append(fcos(ang))
                    sin_vals.append(fsin(ang))

    var sh = List[Int]()
    sh.append(n_total * heads)
    sh.append(total_half)
    return (
        Tensor.from_host(cos_vals, sh.copy(), dtype, ctx),
        Tensor.from_host(sin_vals, sh^, dtype, ctx),
    )


# ── MageFlow double-stream block ─────────────────────────────────────────────
# Thin wrapper over QwenImageDit._block_forward (identical block math + keys);
# the caller supplies the MageFlow text-identity RoPE tables.
struct MageFlowDoubleBlock(Movable):
    var inner: QwenImageDit
    var config: MageFlowConfig

    def __init__(out self, var inner: QwenImageDit):
        self.inner = inner^
        self.config = MageFlowConfig.mage_flow()

    @staticmethod
    def load(dir: String, ctx: DeviceContext) raises -> MageFlowDoubleBlock:
        # Weight key names are identical to Qwen-Image; reuse its loader.
        return MageFlowDoubleBlock(QwenImageDit.load(dir, ctx))

    # Run one double block. Mutates img/txt in place.
    #   img:  [1, N_IMG, 3072]   txt: [1, N_TXT, 3072]   temb: [1, 3072]
    #   pe_cos/pe_sin: build_mageflow_rope_tables output ([(N_TXT+N_IMG)*H, 64]).
    def forward[
        N_IMG: Int, N_TXT: Int, S: Int
    ](
        self,
        block_idx: Int,
        mut img: Tensor,
        mut txt: Tensor,
        temb: Tensor,
        pe_cos: Tensor,
        pe_sin: Tensor,
        ctx: DeviceContext,
    ) raises:
        comptime assert S == N_IMG + N_TXT, "S must equal N_IMG + N_TXT"
        # real_txt_len == N_TXT => the qwen padmask is a no-op (all text real).
        self.inner._block_forward[N_IMG, N_TXT, S](
            block_idx, img, txt, temb, pe_cos, pe_sin, N_TXT, ctx
        )


# ── tiny shape helper (module-local) ─────────────────────────────────────────
def _mf_shape2(a: Int, b: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    s.append(b)
    return s^


def _mf_shape3(a: Int, b: Int, c: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    s.append(b)
    s.append(c)
    return s^


# ── Full MageFlow DiT (compose the double block into the transformer) ─────────
# MageFlow.forward (mage_flow.py:93-153) is structurally the Qwen-Image forward
# with three deltas:
#   1. txt_in in-dim 2560 (Qwen3-VL context) instead of 3584; num_layers 12.
#   2. RoPE = MageFlow msrope (build_mageflow_rope_tables): image tokens roped,
#      text tokens identity (apply_text_rotary_emb=false).
#   3. time embed = MageFlowTimestepProjEmbeddings (mage_layers.py:89-102): the
#      Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000)
#      sinusoid is built with the model's bf16-ROUNDED freq table
#      (mage_layers.py:45 `emb.to(timesteps.dtype)`) — see _mage_time_sinusoid.
#
# Everything else — img_in, txt_norm+txt_in, the 6-way AdaLN modulation split,
# the joint SDPA block, the AdaLayerNormContinuous norm_out (scale=first chunk,
# shift=second, mage_layers.py:713-716) and proj_out — is byte-for-byte the
# Qwen-Image path, so we hold a QwenImageDit and reuse its helpers + block.
struct MageFlowDiT(Movable):
    var inner: QwenImageDit
    var config: MageFlowConfig

    def __init__(out self, var inner: QwenImageDit):
        self.inner = inner^
        self.config = MageFlowConfig.mage_flow()

    @staticmethod
    def load(dir: String, ctx: DeviceContext) raises -> MageFlowDiT:
        # Weight key names are identical to Qwen-Image (verified: img_in,
        # txt_norm, txt_in, time_text_embed.timestep_embedder.linear_1/2,
        # norm_out.linear, proj_out, transformer_blocks.N.*); reuse its loader.
        return MageFlowDiT(QwenImageDit.load(dir, ctx))

    # ── mage bf16 timestep sinusoid ──────────────────────────────────────────
    # get_timestep_embedding (mage_layers.py:24-61) with num_channels=256,
    # flip_sin_to_cos=True (=> COS-first then SIN), downscale_freq_shift=0
    # (denom = half), scale=1000, max_period=10000. The freq table is downcast
    # to the timestep dtype (bf16) BEFORE the multiply (line 45), and the
    # timestep itself arrives bf16 (mage_flow.py:112 timesteps.to(img.dtype)).
    # We replicate BOTH bf16 roundings on host so the sinusoid is bit-faithful,
    # then store as `dtype` (bf16) to match the .to(bf16) before linear_1
    # (mage_layers.py:98). This is the piece the shared cos-first
    # ops.timestep_embedding (F32 freq table, no bf16 round) does NOT match.
    def _mage_time_sinusoid(
        self, timestep: Float32, dtype: STDtype, ctx: DeviceContext
    ) raises -> Tensor:
        comptime half = 128
        comptime dim = 256
        var t_bf = timestep.cast[DType.bfloat16]().cast[DType.float32]()
        var neg_ln = -flog(Float32(10000.0))
        var cos_part = List[Float32]()
        var sin_part = List[Float32]()
        for i in range(half):
            var exponent = neg_ln * (Float32(i) / Float32(half))
            var freq = fexp(exponent)
            var freq_bf = freq.cast[DType.bfloat16]().cast[DType.float32]()
            # emb = timesteps.float() * freq_bf ; then scale=1000 (line 46/49)
            var angle = Float32(1000.0) * (t_bf * freq_bf)
            cos_part.append(fcos(angle))
            sin_part.append(fsin(angle))
        # flip_sin_to_cos=True: final = cat([cos, sin]) (lines 52-56).
        var vals = List[Float32]()
        for i in range(half):
            vals.append(cos_part[i])
        for i in range(half):
            vals.append(sin_part[i])
        return Tensor.from_host(vals, _mf_shape3(1, 1, dim), dtype, ctx)

    # ── qwen_proj time embed: sinusoid -> linear_1 -> SiLU -> linear_2 ──
    def _temb(self, timestep: Float32, ctx: DeviceContext) raises -> Tensor:
        var dim = self.config.inner_dim
        var dtype = self.inner._w(String("img_in.weight")).dtype()
        var sinus = self._mage_time_sinusoid(timestep, dtype, ctx)  # [1,1,256]
        var h1 = self.inner._linear_b(
            sinus,
            "time_text_embed.timestep_embedder.linear_1.weight",
            "time_text_embed.timestep_embedder.linear_1.bias",
            ctx,
        )
        h1 = silu(h1, ctx)
        var temb3d = self.inner._linear_b(
            h1,
            "time_text_embed.timestep_embedder.linear_2.weight",
            "time_text_embed.timestep_embedder.linear_2.bias",
            ctx,
        )
        return reshape(temb3d, _mf_shape2(1, dim), ctx)  # [1, dim]

    # ── img_in: Linear(128 -> 3072) on the patchified latent tokens ──
    def _img_in(self, hidden_states: Tensor, ctx: DeviceContext) raises -> Tensor:
        return self.inner._linear_b(
            hidden_states, "img_in.weight", "img_in.bias", ctx
        )

    # ── txt_norm (RMSNorm 2560) -> txt_in Linear(2560 -> 3072) ──
    def _txt_in(
        self, encoder_hidden_states: Tensor, ctx: DeviceContext
    ) raises -> Tensor:
        ref txt_norm_w = self.inner._w(String("txt_norm.weight"))
        var normed = rms_norm(
            encoder_hidden_states, txt_norm_w, self.config.eps, ctx
        )
        return self.inner._linear_b(normed, "txt_in.weight", "txt_in.bias", ctx)

    # ── full forward: latent + context + timestep -> velocity ─────────────────
    # hidden_states:         [1, N_IMG, 128]  patchified latent tokens
    # encoder_hidden_states: [1, N_TXT, 2560] Qwen3-VL context (RAW, un-normed)
    # timestep:              scalar in [0, 1]  (time_proj folds scale=1000)
    # frame/h_latent/w_latent: msrope patch grid; N_IMG == frame*h*w.
    # Returns velocity [1, N_IMG, 128].
    def forward[
        N_IMG: Int, N_TXT: Int, S: Int
    ](
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        timestep: Float32,
        frame: Int,
        h_latent: Int,
        w_latent: Int,
        ctx: DeviceContext,
    ) raises -> Tensor:
        comptime assert S == N_IMG + N_TXT, "S must equal N_IMG + N_TXT"
        var cfg = self.config
        var dim = cfg.inner_dim

        # ── input projections + time embed (mage_flow.py:109-118) ──
        var img = self._img_in(hidden_states, ctx)          # [1, N_IMG, 3072]
        var txt = self._txt_in(encoder_hidden_states, ctx)  # [1, N_TXT, 3072]
        var temb = self._temb(timestep, ctx)                # [1, 3072]
        # temb = temb + txt_vec, txt_vec == zeros (mage_flow.py:116-118) => no-op.

        # ── MageFlow msrope: image roped, text identity ──
        var rope = build_mageflow_rope_tables(
            frame, h_latent, w_latent, N_TXT, cfg.num_heads,
            cfg.rope_theta, STDtype.F32, ctx,
        )

        # ── 12 double blocks (mage_flow.py:122-144) ──
        for i in range(cfg.depth):
            self.inner._block_forward[N_IMG, N_TXT, S](
                i, img, txt, temb, rope[0], rope[1], N_TXT, ctx
            )

        # ── norm_out AdaLayerNormContinuous (mage_flow.py:147-151;
        #    mage_layers.py:713-716): scale=first chunk, shift=second ──
        var temb_silu = silu(temb, ctx)
        var mods = self.inner._linear_b(
            temb_silu, "norm_out.linear.weight", "norm_out.linear.bias", ctx
        )  # [1, 2*dim]
        var out_scale = self.inner._mod_slice(mods, 0, dim, ctx)
        var out_shift = self.inner._mod_slice(mods, dim, dim, ctx)
        var normed = self.inner._layer_norm_no_affine(img, ctx)
        var modulated = self.inner._modulate(normed, out_scale, out_shift, ctx)

        # ── proj_out: Linear(3072 -> 128) (mage_flow.py:152) ──
        return self.inner._linear_b(modulated, "proj_out.weight", "proj_out.bias", ctx)

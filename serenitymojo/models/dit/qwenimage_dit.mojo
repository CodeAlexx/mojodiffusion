# models/dit/qwenimage_dit.mojo — Qwen-Image MMDiT (GPU, inference-only).
#
# Pure-Mojo port of inference-flame/src/models/qwenimage_dit.rs (the T2I
# `forward` + `block_forward` path). 60 identical DOUBLE-stream transformer
# blocks (no single blocks). Reference read line-by-line; the Rust file is the
# oracle. Reuses the Phase-A foundation ops VERBATIM — linear / layer_norm /
# rms_norm / rope_interleaved / sdpa / silu / gelu / add / mul / slice /
# reshape / permute / concat / timestep_embedding — plus tiny DiT-local glue
# (add_scalar already lives in tensor_algebra).
#
# ── Architecture (qwenimage_dit.rs:98-133 QwenImageConfig::default) ──────────
#   num_layers 60, inner_dim 3072, num_heads 24, head_dim 128, in_channels 64,
#   out_channels 16, patch_size 2, joint_attention_dim 3584, mlp_ratio 4.0
#   (FFN hidden = 12288), axes_dims_rope (16, 56, 56), rope_theta 10000.0,
#   timestep_dim 256, eps 1e-6.
#
# ── forward (qwenimage_dit.rs:607-700) ──────────────────────────────────────
#   img = Linear(img_in)(hidden_states)             [B, N_img, 64]  -> [B,N_img,3072]
#   txt = Linear(txt_in)(RMSNorm(txt_norm)(enc))    [B, N_txt, 3584]-> [B,N_txt,3072]
#   temb = time_text_embed(timestep)                [B, 3072]
#         = Linear2(SiLU(Linear1(timestep_embedding(t*1000, 256, cos-then-sin))))
#   (pe_cos, pe_sin) = build_rope_tables(frame,h,w, N_txt)  3-axis interleaved
#   for i in 0..60: (img, txt) = block_forward(...)
#   norm_out (AdaLayerNormContinuous):
#     mods = Linear(norm_out.linear)(SiLU(temb))     [B, 2*3072]
#     scale = mods[:, 0:3072]; shift = mods[:, 3072:6144]
#     out = LayerNorm_no_affine(img)*(1+scale)+shift
#   out = Linear(proj_out)(out)                      [B, N_img, 64]
#
# ── block_forward (qwenimage_dit.rs:1003-1197) ──────────────────────────────
#   img_mods = Linear(img_mod.1)(SiLU(temb))         [B, 6*3072]
#   txt_mods = Linear(txt_mod.1)(SiLU(temb))
#   img_mod1,img_mod2 = chunk(img_mods, 2)           each [B, 3*3072]
#   {shift1,scale1,gate1},{shift2,scale2,gate2} = split each into 3 [B,3072]
#   img_modulated = LayerNorm_no_affine(img)*(1+scale1)+shift1   (txt likewise)
#   img_q/k/v = to_q/to_k/to_v(img_modulated) -> [B,N,H,Dh]  (BSHD)
#   txt_q/k/v = add_q_proj/add_k_proj/add_v_proj(txt_modulated)
#   QK RMSNorm: norm_q/norm_k (img), norm_added_q/norm_added_k (txt), [128], eps 1e-6
#   q = cat([txt_q, img_q], seq);  k,v likewise           (TXT FIRST then IMG)
#   (q,k) = rope_interleaved(q/k, pe_cos, pe_sin)
#   attn = sdpa(q, k, v)                                  joint, full attention
#   txt_attn, img_attn = split(attn, [N_txt, N_img])
#   img_attn = to_out.0(img_attn);  txt_attn = to_add_out(txt_attn)
#   img = img + gate1 * img_attn;  txt = txt + gate1_txt * txt_attn
#   img FFN: LN_no_affine -> *(1+scale2)+shift2 -> net.0.proj -> GELU(tanh) -> net.2
#   img = img + gate2 * img_mlp;  (txt likewise with txt scale2/shift2/gate2)
#
# RoPE = INTERLEAVED (FLUX/Klein style, `rope_fused_bf16`). cos/sin tables are
# built [S*H, Dh/2] (rows = token*head, angle depends only on token; repeated
# across heads) so the foundation rope_interleaved consumes them directly with
# q/k flattened from [1,S,H,Dh].
#
# Token concat order is TXT THEN IMG (qwenimage_dit.rs:1155 cat([txt, img])).
#
# Mojo 1.0.0b1, NVIDIA GPU. BF16 storage, F32 accumulation in foundation ops.
# *** CODE-ONLY: compile-verified; NOT executed (GPU wedged). ***

from std.gpu.host import DeviceContext
from std.math import sqrt, cos as fcos, sin as fsin, exp as fexp, log as flog, pow as fpow
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import rms_norm, layer_norm
from serenitymojo.ops.activations import silu, gelu
from serenitymojo.ops.rope import rope_interleaved
from serenitymojo.ops.attention import sdpa_qwen_keymask
from serenitymojo.ops.embeddings import timestep_embedding
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import (
    reshape,
    slice,
    concat,
    add,
    mul,
    add_scalar,
)
from serenitymojo.offload.block_loader import Block
from serenitymojo.offload.turbo_planned_loader import TurboPlannedLoader
from serenitymojo.offload.plan import build_qwenimage_block_plan, OffloadConfig


# ── Config ──────────────────────────────────────────────────────────────────
@fieldwise_init
struct QwenImageConfig(Copyable, Movable, ImplicitlyCopyable):
    """Qwen-Image MMDiT hyperparameters (qwenimage_dit.rs:98-133)."""

    var num_layers: Int          # 60
    var inner_dim: Int           # 3072
    var num_heads: Int           # 24
    var head_dim: Int            # 128
    var in_channels: Int         # 64
    var out_channels: Int        # 16
    var patch_size: Int          # 2
    var joint_attention_dim: Int # 3584
    var mlp_hidden: Int          # 12288 (mlp_ratio 4.0 * inner_dim)
    var axis0: Int               # 16  (frame axis dim)
    var axis1: Int               # 56  (height axis dim)
    var axis2: Int               # 56  (width axis dim)
    var rope_theta: Float64      # 10000.0
    var timestep_dim: Int        # 256
    var eps: Float32             # 1e-6

    @staticmethod
    def qwen_image() -> QwenImageConfig:
        return QwenImageConfig(
            60, 3072, 24, 128, 64, 16, 2, 3584, 12288,
            16, 56, 56, Float64(10000.0), 256, Float32(1e-6),
        )


# ── RoPE 3-axis table builder ────────────────────────────────────────────────
# Mirrors QwenImageDit::build_rope_tables (qwenimage_dit.rs:321-420).
# axes = (16, 56, 56), total_half = 8 + 28 + 28 = 64 = head_dim/2.
# Text tokens get position max_vid_index + t; image tokens get
# (frame, h - height/2, w - width/2) per axis (scale_rope symmetric).
# Token order: txt[0..N_txt] then img[0..frame*height*width].
# Output tables are [S*H, total_half] (angle repeated across H heads).
def build_qwenimage_rope_tables(
    frame: Int,
    height: Int,
    width: Int,
    txt_seq_len: Int,
    heads: Int,
    config: QwenImageConfig,
    dtype: STDtype,
    ctx: DeviceContext,
) raises -> Tuple[Tensor, Tensor]:
    var axes = List[Int]()
    axes.append(config.axis0)
    axes.append(config.axis1)
    axes.append(config.axis2)
    var total_half = 0
    for a in range(3):
        total_half += axes[a] // 2

    var n_img = frame * height * width
    var n_total = txt_seq_len + n_img
    var theta = config.rope_theta

    # Per-axis inv-freq tables: freq[axis][i] = 1 / theta^(2i/axis_dim).
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

    # max_vid_index = max(height/2, width/2, 1) (scale_rope text offset).
    var hv = height // 2
    var wv = width // 2
    var max_vid_index = hv
    if wv > max_vid_index:
        max_vid_index = wv
    if max_vid_index < 1:
        max_vid_index = 1

    # Build per-token angle vector (length total_half), then repeat across heads.
    var cos_vals = List[Float32]()
    var sin_vals = List[Float32]()
    for tok in range(n_total):
        # compute the per-axis positions for this token
        var p0: Float32
        var p1: Float32
        var p2: Float32
        if tok < txt_seq_len:
            var pos = Float32(max_vid_index + tok)
            p0 = pos
            p1 = pos
            p2 = pos
        else:
            var img_idx = tok - txt_seq_len
            var f_idx = img_idx // (height * width)
            var rem = img_idx % (height * width)
            var h_idx = rem // width
            var w_idx = rem % width
            p0 = Float32(f_idx)
            p1 = Float32(h_idx) - Float32(height) / Float32(2.0)
            p2 = Float32(w_idx) - Float32(width) / Float32(2.0)
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


def build_qwenimage_edit_rope_tables(
    frame: Int,
    height: Int,
    width: Int,
    txt_seq_len: Int,
    heads: Int,
    config: QwenImageConfig,
    dtype: STDtype,
    ctx: DeviceContext,
) raises -> Tuple[Tensor, Tensor]:
    """Two-region Qwen-Image-Edit RoPE tables.

    Token order is txt, target image, reference image. Both image regions use
    the same spatial patch grid; the reference region's frame coordinate is
    offset by `frame` so target/reference position embeddings do not collide.
    """
    var axes = List[Int]()
    axes.append(config.axis0)
    axes.append(config.axis1)
    axes.append(config.axis2)
    var total_half = 0
    for a in range(3):
        total_half += axes[a] // 2

    var n_region = frame * height * width
    var n_img = n_region * 2
    var n_total = txt_seq_len + n_img
    var theta = config.rope_theta

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

    var hv = height // 2
    var wv = width // 2
    var max_vid_index = hv
    if wv > max_vid_index:
        max_vid_index = wv
    if max_vid_index < 1:
        max_vid_index = 1

    var cos_vals = List[Float32]()
    var sin_vals = List[Float32]()
    for tok in range(n_total):
        var p0: Float32
        var p1: Float32
        var p2: Float32
        if tok < txt_seq_len:
            var pos = Float32(max_vid_index + tok)
            p0 = pos
            p1 = pos
            p2 = pos
        else:
            var img_idx = tok - txt_seq_len
            var region = img_idx // n_region
            var local_idx = img_idx % n_region
            var f_idx = local_idx // (height * width)
            var rem = local_idx % (height * width)
            var h_idx = rem // width
            var w_idx = rem % width
            p0 = Float32(f_idx + region * frame)
            p1 = Float32(h_idx) - Float32(height) / Float32(2.0)
            p2 = Float32(w_idx) - Float32(width) / Float32(2.0)
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


# ── QwenImageDit ──────────────────────────────────────────────────────────────
struct QwenImageDit(Movable):
    """Qwen-Image MMDiT. All-resident weight load (60 double blocks). Owns
    weights as List[ArcPointer[Tensor]] (Tensor Movable-not-Copyable). Forward
    runs on GPU. comptime params N_IMG/N_TXT/S feed the comptime-shaped sdpa."""

    var weights: List[ArcPointer[Tensor]]
    var name_to_idx: Dict[String, Int]
    var config: QwenImageConfig

    def __init__(
        out self,
        var weights: List[ArcPointer[Tensor]],
        var name_to_idx: Dict[String, Int],
        config: QwenImageConfig,
    ):
        self.weights = weights^
        self.name_to_idx = name_to_idx^
        self.config = config

    @staticmethod
    def load(dir: String, ctx: DeviceContext) raises -> QwenImageDit:
        """Load all transformer tensors from a sharded diffusers-format dir via
        ShardedSafeTensors + Tensor.from_view (H2D copy). Looks up by name."""
        var sharded = ShardedSafeTensors.open(dir)
        var weights = List[ArcPointer[Tensor]]()
        var name_to_idx = Dict[String, Int]()
        for ref nm in sharded.names():
            var tv = sharded.tensor_view(nm)
            var t = Tensor.from_view(tv, ctx)
            var idx = len(weights)
            weights.append(ArcPointer(t^))
            name_to_idx[nm] = idx
        return QwenImageDit(weights^, name_to_idx^, QwenImageConfig.qwen_image())

    @staticmethod
    def load_shared(dir: String, ctx: DeviceContext) raises -> QwenImageDit:
        """Load only the non-block tensors needed around streamed Qwen blocks."""
        var sharded = ShardedSafeTensors.open(dir)
        var weights = List[ArcPointer[Tensor]]()
        var name_to_idx = Dict[String, Int]()
        for ref nm in sharded.names():
            var keep = (
                nm.startswith("img_in.")
                or nm.startswith("txt_norm.")
                or nm.startswith("txt_in.")
                or nm.startswith("time_text_embed.")
                or nm.startswith("norm_out.")
                or nm.startswith("proj_out.")
            )
            if not keep:
                continue
            var tv = sharded.tensor_view(nm)
            var t = Tensor.from_view(tv, ctx)
            var idx = len(weights)
            weights.append(ArcPointer(t^))
            name_to_idx[nm] = idx
        return QwenImageDit(weights^, name_to_idx^, QwenImageConfig.qwen_image())

    def _w(self, name: String) raises -> ref [self.weights] Tensor:
        if name not in self.name_to_idx:
            raise Error(String("missing weight: ") + name)
        var idx = self.name_to_idx[name]
        return self.weights[idx][]

    def _has(self, name: String) -> Bool:
        return name in self.name_to_idx

    # Clone a tiny weight (bias) into an owned Tensor so it can be passed as
    # Optional[Tensor] (Tensor not Copyable -> can't pass a borrow directly).
    def _clone(self, x: Tensor, ctx: DeviceContext) raises -> Tensor:
        var dev = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
        ctx.enqueue_copy(dst_buf=dev, src_buf=x.buf)
        ctx.synchronize()
        return Tensor(dev^, x.shape(), x.dtype())

    # Linear with bias borrowed from the weight store.
    def _linear_b(
        self, x: Tensor, w_key: String, b_key: String, ctx: DeviceContext
    ) raises -> Tensor:
        ref w = self._w(w_key)
        ref b = self._w(b_key)
        return linear(x, w, Optional[Tensor](self._clone(b, ctx)), ctx)

    def _linear_nb(
        self, x: Tensor, w_key: String, ctx: DeviceContext
    ) raises -> Tensor:
        ref w = self._w(w_key)
        return linear(x, w, None, ctx)

    # LayerNorm with NO affine (gamma=1, beta=0) over the last dim.
    def _layer_norm_no_affine(
        self, x: Tensor, ctx: DeviceContext
    ) raises -> Tensor:
        var dim = x.shape()[len(x.shape()) - 1]
        var dtype = x.dtype()
        var ones = List[Float32]()
        var zeros = List[Float32]()
        for _i in range(dim):
            ones.append(Float32(1.0))
            zeros.append(Float32(0.0))
        var osh = List[Int]()
        osh.append(dim)
        var g = Tensor.from_host(ones, osh.copy(), dtype, ctx)
        var z = Tensor.from_host(zeros, osh^, dtype, ctx)
        return layer_norm(x, g, z, self.config.eps, ctx)

    # AdaLN modulate over a [1, N, dim] tensor with [1, dim] scale/shift vectors:
    #   out = normed * (1 + scale) + shift
    # The foundation add/mul broadcast NumPy-style (rank<=6), so a [1, dim]
    # vector broadcasts against [1, N, dim].
    def _modulate(
        self, normed: Tensor, scale: Tensor, shift: Tensor, ctx: DeviceContext
    ) raises -> Tensor:
        var one_plus = add_scalar(scale, Float32(1.0), ctx)
        var scaled = mul(normed, one_plus, ctx)
        return add(scaled, shift, ctx)

    # Slice a [1, dim_total] mod vector along last dim -> [1, length].
    def _mod_slice(
        self, mods: Tensor, start: Int, length: Int, ctx: DeviceContext
    ) raises -> Tensor:
        return slice(mods, 1, start, length, ctx)

    def _zeros_tokens(
        self, seq_len: Int, dim: Int, dtype: STDtype, ctx: DeviceContext
    ) raises -> Tensor:
        var data = List[Float32]()
        data.resize(seq_len * dim, Float32(0.0))
        var sh = List[Int]()
        sh.append(1)
        sh.append(seq_len)
        sh.append(dim)
        return Tensor.from_host(data, sh^, dtype, ctx)

    def _mod_slice_pair(
        self,
        mods_target: Tensor,
        mods_ref: Tensor,
        start: Int,
        length: Int,
        target_seq_len: Int,
        ref_seq_len: Int,
        dtype: STDtype,
        ctx: DeviceContext,
    ) raises -> Tensor:
        var target = self._mod_slice(mods_target, start, length, ctx)
        var target_zeros = self._zeros_tokens(target_seq_len, length, dtype, ctx)
        target = add(target_zeros, target, ctx)
        if ref_seq_len == 0:
            return target^
        var ref_mod = self._mod_slice(mods_ref, start, length, ctx)
        var ref_zeros = self._zeros_tokens(ref_seq_len, length, dtype, ctx)
        ref_mod = add(ref_zeros, ref_mod, ctx)
        return concat(1, ctx, target, ref_mod)

    # Reshape [1, N, H*Dh] -> [1, N, H, Dh] (BSHD).
    def _to_bshd(
        self, x: Tensor, n: Int, h: Int, dh: Int, ctx: DeviceContext
    ) raises -> Tensor:
        var sh = List[Int]()
        sh.append(1)
        sh.append(n)
        sh.append(h)
        sh.append(dh)
        return reshape(x, sh^, ctx)

    # Reshape [1, N, H, Dh] -> [1, N, H*Dh].
    def _from_bshd(
        self, x: Tensor, n: Int, h: Int, dh: Int, ctx: DeviceContext
    ) raises -> Tensor:
        var sh = List[Int]()
        sh.append(1)
        sh.append(n)
        sh.append(h * dh)
        return reshape(x, sh^, ctx)

    # ── one double-stream block ───────────────────────────────────────────────
    def _block_forward[
        N_IMG: Int, N_TXT: Int, S: Int
    ](
        self,
        block_idx: Int,
        mut img: Tensor,
        mut txt: Tensor,
        temb: Tensor,
        pe_cos: Tensor,
        pe_sin: Tensor,
        real_txt_len: Int,
        ctx: DeviceContext,
    ) raises:
        comptime assert S == N_IMG + N_TXT, "S must equal N_IMG + N_TXT"
        var cfg = self.config
        var h = cfg.num_heads
        var d = cfg.head_dim
        var dim = cfg.inner_dim
        var scale = Float32(1.0) / sqrt(Float32(d))
        var p = String("transformer_blocks.") + String(block_idx)

        # ── modulation vectors ──
        var temb_silu = silu(temb, ctx)  # [1, dim]  (temb is [1, dim])
        var img_mods = self._linear_b(
            temb_silu, p + ".img_mod.1.weight", p + ".img_mod.1.bias", ctx
        )  # [1, 6*dim]
        var txt_mods = self._linear_b(
            temb_silu, p + ".txt_mod.1.weight", p + ".txt_mod.1.bias", ctx
        )

        # split: mod1 = [0,3*dim), mod2 = [3*dim, 6*dim); each -> shift,scale,gate
        var img_shift1 = self._mod_slice(img_mods, 0 * dim, dim, ctx)
        var img_scale1 = self._mod_slice(img_mods, 1 * dim, dim, ctx)
        var img_gate1 = self._mod_slice(img_mods, 2 * dim, dim, ctx)
        var img_shift2 = self._mod_slice(img_mods, 3 * dim, dim, ctx)
        var img_scale2 = self._mod_slice(img_mods, 4 * dim, dim, ctx)
        var img_gate2 = self._mod_slice(img_mods, 5 * dim, dim, ctx)

        var txt_shift1 = self._mod_slice(txt_mods, 0 * dim, dim, ctx)
        var txt_scale1 = self._mod_slice(txt_mods, 1 * dim, dim, ctx)
        var txt_gate1 = self._mod_slice(txt_mods, 2 * dim, dim, ctx)
        var txt_shift2 = self._mod_slice(txt_mods, 3 * dim, dim, ctx)
        var txt_scale2 = self._mod_slice(txt_mods, 4 * dim, dim, ctx)
        var txt_gate2 = self._mod_slice(txt_mods, 5 * dim, dim, ctx)

        # ── norm1 + modulate ──
        var img_normed = self._layer_norm_no_affine(img, ctx)
        var img_modulated = self._modulate(img_normed, img_scale1, img_shift1, ctx)
        var txt_normed = self._layer_norm_no_affine(txt, ctx)
        var txt_modulated = self._modulate(txt_normed, txt_scale1, txt_shift1, ctx)

        # ── Q/K/V projections (split) -> BSHD ──
        var img_q = self._to_bshd(
            self._linear_b(img_modulated, p + ".attn.to_q.weight", p + ".attn.to_q.bias", ctx),
            N_IMG, h, d, ctx,
        )
        var img_k = self._to_bshd(
            self._linear_b(img_modulated, p + ".attn.to_k.weight", p + ".attn.to_k.bias", ctx),
            N_IMG, h, d, ctx,
        )
        var img_v = self._to_bshd(
            self._linear_b(img_modulated, p + ".attn.to_v.weight", p + ".attn.to_v.bias", ctx),
            N_IMG, h, d, ctx,
        )
        var txt_q = self._to_bshd(
            self._linear_b(txt_modulated, p + ".attn.add_q_proj.weight", p + ".attn.add_q_proj.bias", ctx),
            N_TXT, h, d, ctx,
        )
        var txt_k = self._to_bshd(
            self._linear_b(txt_modulated, p + ".attn.add_k_proj.weight", p + ".attn.add_k_proj.bias", ctx),
            N_TXT, h, d, ctx,
        )
        var txt_v = self._to_bshd(
            self._linear_b(txt_modulated, p + ".attn.add_v_proj.weight", p + ".attn.add_v_proj.bias", ctx),
            N_TXT, h, d, ctx,
        )

        # ── QK RMSNorm (over head_dim, weight [128]) ──
        ref nq = self._w(p + ".attn.norm_q.weight")
        ref nk = self._w(p + ".attn.norm_k.weight")
        ref naq = self._w(p + ".attn.norm_added_q.weight")
        ref nak = self._w(p + ".attn.norm_added_k.weight")
        img_q = rms_norm(img_q, nq, cfg.eps, ctx)
        img_k = rms_norm(img_k, nk, cfg.eps, ctx)
        txt_q = rms_norm(txt_q, naq, cfg.eps, ctx)
        txt_k = rms_norm(txt_k, nak, cfg.eps, ctx)

        # ── concat TXT then IMG along seq (dim=1 in BSHD) ──
        var q = concat(1, ctx, txt_q, img_q)  # [1, S, H, Dh]
        var k = concat(1, ctx, txt_k, img_k)
        var v = concat(1, ctx, txt_v, img_v)

        # ── RoPE (interleaved) on q,k ──
        q = rope_interleaved(q, pe_cos, pe_sin, ctx)
        k = rope_interleaved(k, pe_cos, pe_sin, ctx)

        # ── joint attention with Qwen text-key padding mask ──
        # Qwen lays tokens out [TXT padded to N_TXT, IMG]. The product path used
        # to materialize [1,H,S,S] additive masks and call math SDPA; this
        # online key-mask path preserves the same -1e4 pad-column bias without
        # allocating a square mask or score slab.
        var attn = sdpa_qwen_keymask[1, S, 24, 128, N_TXT](
            q, k, v, real_txt_len, scale, ctx
        )  # [1, S, H, Dh]

        # ── split txt/img, flatten heads, output projections ──
        var txt_attn = slice(attn, 1, 0, N_TXT, ctx)
        var img_attn = slice(attn, 1, N_TXT, N_IMG, ctx)
        var txt_attn_2d = self._from_bshd(txt_attn, N_TXT, h, d, ctx)
        var img_attn_2d = self._from_bshd(img_attn, N_IMG, h, d, ctx)

        var img_o = self._linear_b(
            img_attn_2d, p + ".attn.to_out.0.weight", p + ".attn.to_out.0.bias", ctx
        )
        var txt_o = self._linear_b(
            txt_attn_2d, p + ".attn.to_add_out.weight", p + ".attn.to_add_out.bias", ctx
        )

        # ── gated residual (gate1) ──
        img = add(img, mul(img_gate1, img_o, ctx), ctx)
        txt = add(txt, mul(txt_gate1, txt_o, ctx), ctx)

        # ── FFN (img) ──
        var img_normed2 = self._layer_norm_no_affine(img, ctx)
        var img_mlp_in = self._modulate(img_normed2, img_scale2, img_shift2, ctx)
        var img_mlp = self._linear_b(
            img_mlp_in, p + ".img_mlp.net.0.proj.weight", p + ".img_mlp.net.0.proj.bias", ctx
        )
        img_mlp = gelu(img_mlp, ctx)
        img_mlp = self._linear_b(
            img_mlp, p + ".img_mlp.net.2.weight", p + ".img_mlp.net.2.bias", ctx
        )
        img = add(img, mul(img_gate2, img_mlp, ctx), ctx)

        # ── FFN (txt) ──
        var txt_normed2 = self._layer_norm_no_affine(txt, ctx)
        var txt_mlp_in = self._modulate(txt_normed2, txt_scale2, txt_shift2, ctx)
        var txt_mlp = self._linear_b(
            txt_mlp_in, p + ".txt_mlp.net.0.proj.weight", p + ".txt_mlp.net.0.proj.bias", ctx
        )
        txt_mlp = gelu(txt_mlp, ctx)
        txt_mlp = self._linear_b(
            txt_mlp, p + ".txt_mlp.net.2.weight", p + ".txt_mlp.net.2.bias", ctx
        )
        txt = add(txt, mul(txt_gate2, txt_mlp, ctx), ctx)

    # ── one edit block with target/reference image modulation ────────────────
    def _block_forward_edit[
        N_TARGET: Int, N_REF: Int, N_TXT: Int, S: Int
    ](
        self,
        block_idx: Int,
        mut img: Tensor,
        mut txt: Tensor,
        temb_target: Tensor,
        temb_ref: Tensor,
        pe_cos: Tensor,
        pe_sin: Tensor,
        real_txt_len: Int,
        ctx: DeviceContext,
    ) raises:
        comptime assert S == N_TARGET + N_REF + N_TXT, "S must equal target + ref + text"
        var cfg = self.config
        var h = cfg.num_heads
        var d = cfg.head_dim
        var dim = cfg.inner_dim
        var n_img = N_TARGET + N_REF
        var scale = Float32(1.0) / sqrt(Float32(d))
        var p = String("transformer_blocks.") + String(block_idx)
        var dtype = self._w(String("img_in.weight")).dtype()

        var temb_target_silu = silu(temb_target, ctx)
        var temb_ref_silu = silu(temb_ref, ctx)
        var img_mods_t = self._linear_b(
            temb_target_silu, p + ".img_mod.1.weight", p + ".img_mod.1.bias", ctx
        )
        var img_mods_r = self._linear_b(
            temb_ref_silu, p + ".img_mod.1.weight", p + ".img_mod.1.bias", ctx
        )
        var txt_mods = self._linear_b(
            temb_target_silu, p + ".txt_mod.1.weight", p + ".txt_mod.1.bias", ctx
        )

        var img_shift1 = self._mod_slice_pair(
            img_mods_t, img_mods_r, 0 * dim, dim, N_TARGET, N_REF, dtype, ctx
        )
        var img_scale1 = self._mod_slice_pair(
            img_mods_t, img_mods_r, 1 * dim, dim, N_TARGET, N_REF, dtype, ctx
        )
        var img_gate1 = self._mod_slice_pair(
            img_mods_t, img_mods_r, 2 * dim, dim, N_TARGET, N_REF, dtype, ctx
        )
        var img_shift2 = self._mod_slice_pair(
            img_mods_t, img_mods_r, 3 * dim, dim, N_TARGET, N_REF, dtype, ctx
        )
        var img_scale2 = self._mod_slice_pair(
            img_mods_t, img_mods_r, 4 * dim, dim, N_TARGET, N_REF, dtype, ctx
        )
        var img_gate2 = self._mod_slice_pair(
            img_mods_t, img_mods_r, 5 * dim, dim, N_TARGET, N_REF, dtype, ctx
        )

        var txt_shift1 = self._mod_slice(txt_mods, 0 * dim, dim, ctx)
        var txt_scale1 = self._mod_slice(txt_mods, 1 * dim, dim, ctx)
        var txt_gate1 = self._mod_slice(txt_mods, 2 * dim, dim, ctx)
        var txt_shift2 = self._mod_slice(txt_mods, 3 * dim, dim, ctx)
        var txt_scale2 = self._mod_slice(txt_mods, 4 * dim, dim, ctx)
        var txt_gate2 = self._mod_slice(txt_mods, 5 * dim, dim, ctx)

        var img_normed = self._layer_norm_no_affine(img, ctx)
        var img_modulated = self._modulate(img_normed, img_scale1, img_shift1, ctx)
        var txt_normed = self._layer_norm_no_affine(txt, ctx)
        var txt_modulated = self._modulate(txt_normed, txt_scale1, txt_shift1, ctx)

        var img_q = self._to_bshd(
            self._linear_b(img_modulated, p + ".attn.to_q.weight", p + ".attn.to_q.bias", ctx),
            n_img, h, d, ctx,
        )
        var img_k = self._to_bshd(
            self._linear_b(img_modulated, p + ".attn.to_k.weight", p + ".attn.to_k.bias", ctx),
            n_img, h, d, ctx,
        )
        var img_v = self._to_bshd(
            self._linear_b(img_modulated, p + ".attn.to_v.weight", p + ".attn.to_v.bias", ctx),
            n_img, h, d, ctx,
        )
        var txt_q = self._to_bshd(
            self._linear_b(txt_modulated, p + ".attn.add_q_proj.weight", p + ".attn.add_q_proj.bias", ctx),
            N_TXT, h, d, ctx,
        )
        var txt_k = self._to_bshd(
            self._linear_b(txt_modulated, p + ".attn.add_k_proj.weight", p + ".attn.add_k_proj.bias", ctx),
            N_TXT, h, d, ctx,
        )
        var txt_v = self._to_bshd(
            self._linear_b(txt_modulated, p + ".attn.add_v_proj.weight", p + ".attn.add_v_proj.bias", ctx),
            N_TXT, h, d, ctx,
        )

        ref nq = self._w(p + ".attn.norm_q.weight")
        ref nk = self._w(p + ".attn.norm_k.weight")
        ref naq = self._w(p + ".attn.norm_added_q.weight")
        ref nak = self._w(p + ".attn.norm_added_k.weight")
        img_q = rms_norm(img_q, nq, cfg.eps, ctx)
        img_k = rms_norm(img_k, nk, cfg.eps, ctx)
        txt_q = rms_norm(txt_q, naq, cfg.eps, ctx)
        txt_k = rms_norm(txt_k, nak, cfg.eps, ctx)

        var q = concat(1, ctx, txt_q, img_q)
        var k = concat(1, ctx, txt_k, img_k)
        var v = concat(1, ctx, txt_v, img_v)
        q = rope_interleaved(q, pe_cos, pe_sin, ctx)
        k = rope_interleaved(k, pe_cos, pe_sin, ctx)

        var attn = sdpa_qwen_keymask[1, S, 24, 128, N_TXT](
            q, k, v, real_txt_len, scale, ctx
        )
        var txt_attn = slice(attn, 1, 0, N_TXT, ctx)
        var img_attn = slice(attn, 1, N_TXT, n_img, ctx)
        var txt_attn_2d = self._from_bshd(txt_attn, N_TXT, h, d, ctx)
        var img_attn_2d = self._from_bshd(img_attn, n_img, h, d, ctx)

        var img_o = self._linear_b(
            img_attn_2d, p + ".attn.to_out.0.weight", p + ".attn.to_out.0.bias", ctx
        )
        var txt_o = self._linear_b(
            txt_attn_2d, p + ".attn.to_add_out.weight", p + ".attn.to_add_out.bias", ctx
        )
        img = add(img, mul(img_gate1, img_o, ctx), ctx)
        txt = add(txt, mul(txt_gate1, txt_o, ctx), ctx)

        var img_normed2 = self._layer_norm_no_affine(img, ctx)
        var img_mlp_in = self._modulate(img_normed2, img_scale2, img_shift2, ctx)
        var img_mlp = self._linear_b(
            img_mlp_in, p + ".img_mlp.net.0.proj.weight", p + ".img_mlp.net.0.proj.bias", ctx
        )
        img_mlp = gelu(img_mlp, ctx)
        img_mlp = self._linear_b(
            img_mlp, p + ".img_mlp.net.2.weight", p + ".img_mlp.net.2.bias", ctx
        )
        img = add(img, mul(img_gate2, img_mlp, ctx), ctx)

        var txt_normed2 = self._layer_norm_no_affine(txt, ctx)
        var txt_mlp_in = self._modulate(txt_normed2, txt_scale2, txt_shift2, ctx)
        var txt_mlp = self._linear_b(
            txt_mlp_in, p + ".txt_mlp.net.0.proj.weight", p + ".txt_mlp.net.0.proj.bias", ctx
        )
        txt_mlp = gelu(txt_mlp, ctx)
        txt_mlp = self._linear_b(
            txt_mlp, p + ".txt_mlp.net.2.weight", p + ".txt_mlp.net.2.bias", ctx
        )
        txt = add(txt, mul(txt_gate2, txt_mlp, ctx), ctx)

    # ── full forward ───────────────────────────────────────────────────────────
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
        """Single denoise step.
        hidden_states:          [1, N_IMG, 64] packed image latents
        encoder_hidden_states:  [1, N_TXT, 3584] Qwen2.5-VL text states
        timestep:               scalar in [0, 1]
        frame/h_latent/w_latent: RoPE patch-grid geometry; N_IMG == frame*h*w.
        Returns velocity [1, N_IMG, 64].
        """
        comptime assert S == N_IMG + N_TXT, "S must equal N_IMG + N_TXT"
        var cfg = self.config
        var h = cfg.num_heads
        var d = cfg.head_dim
        var dim = cfg.inner_dim

        var dtype = self._w(String("img_in.weight")).dtype()

        # ── img_in ──
        var img = self._linear_b(hidden_states, "img_in.weight", "img_in.bias", ctx)

        # ── txt_norm (RMSNorm) then txt_in ──
        ref txt_norm_w = self._w(String("txt_norm.weight"))
        var txt_normed = rms_norm(encoder_hidden_states, txt_norm_w, cfg.eps, ctx)
        var txt = self._linear_b(txt_normed, "txt_in.weight", "txt_in.bias", ctx)

        # ── time_text_embed: sinusoidal(t*1000, 256, cos-then-sin) -> MLP ──
        var t_host = List[Float32]()
        t_host.append(timestep * Float32(1000.0))
        var t_sh = List[Int]()
        t_sh.append(1)
        var t_tensor = Tensor.from_host(t_host, t_sh^, STDtype.F32, ctx)
        # timestep_embedding computes angle = t * exp(-ln(max_period)*i/half),
        # cos-first-then-sin (matches diffusers flip_sin_to_cos=True). max_period
        # 10000. We pre-scaled t by 1000 to fold in diffusers scale=1000.
        var t_embed = timestep_embedding(
            t_tensor, cfg.timestep_dim, ctx, Float32(10000.0), dtype
        )
        var t_embed_cast = reshape(t_embed, _shape3(1, 1, cfg.timestep_dim), ctx)
        var h1 = self._linear_b(
            t_embed_cast,
            "time_text_embed.timestep_embedder.linear_1.weight",
            "time_text_embed.timestep_embedder.linear_1.bias",
            ctx,
        )
        h1 = silu(h1, ctx)
        var temb_3d = self._linear_b(
            h1,
            "time_text_embed.timestep_embedder.linear_2.weight",
            "time_text_embed.timestep_embedder.linear_2.bias",
            ctx,
        )
        var temb = reshape(temb_3d, _shape2(1, dim), ctx)  # [1, dim]

        # ── RoPE tables for joint [txt, img] attention ──
        var rope = build_qwenimage_rope_tables(
            frame, h_latent, w_latent, N_TXT, h, cfg, dtype, ctx
        )
        # ── transformer blocks ── (block_forward mutates img/txt in place)
        for i in range(cfg.num_layers):
            self._block_forward[N_IMG, N_TXT, S](
                i, img, txt, temb, rope[0], rope[1], N_TXT, ctx
            )

        # ── norm_out: AdaLayerNormContinuous ──
        var temb_silu = silu(temb, ctx)
        var mods = self._linear_b(
            temb_silu, "norm_out.linear.weight", "norm_out.linear.bias", ctx
        )  # [1, 2*dim]
        var out_scale = self._mod_slice(mods, 0, dim, ctx)
        var out_shift = self._mod_slice(mods, dim, dim, ctx)
        var normed = self._layer_norm_no_affine(img, ctx)
        var modulated = self._modulate(normed, out_scale, out_shift, ctx)

        # ── proj_out ──
        return self._linear_b(modulated, "proj_out.weight", "proj_out.bias", ctx)


# ── tiny module-level helpers ────────────────────────────────────────────────
def _shape2(a: Int, b: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    s.append(b)
    return s^


def _shape3(a: Int, b: Int, c: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    s.append(b)
    s.append(c)
    return s^


# Cast a Tensor to a target dtype if it differs (uses ops/cast).
def _cast_like(x: Tensor, dtype: STDtype, ctx: DeviceContext) raises -> Tensor:
    if x.dtype() == dtype:
        var dev = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
        ctx.enqueue_copy(dst_buf=dev, src_buf=x.buf)
        ctx.synchronize()
        return Tensor(dev^, x.shape(), x.dtype())
    return cast_tensor(x, dtype, ctx)


@fieldwise_init
struct QwenImageCfgPreds(Movable):
    var pos: Tensor
    var neg: Tensor


@fieldwise_init
struct QwenImageDitOffloaded(Movable):
    var shared: QwenImageDit
    var loader: TurboPlannedLoader

    @staticmethod
    def load(dir: String, ctx: DeviceContext) raises -> QwenImageDitOffloaded:
        var shared = QwenImageDit.load_shared(dir, ctx)
        var plan = build_qwenimage_block_plan()
        var loader = TurboPlannedLoader.open(
            dir, plan^, OffloadConfig.synchronous_cfg_paired(), ctx
        )
        return QwenImageDitOffloaded(shared^, loader^)

    def _block_model(self, block: Block) -> QwenImageDit:
        var weights = self.shared.weights.copy()
        var name_to_idx = self.shared.name_to_idx.copy()
        for ref e in block.items():
            name_to_idx[e.key] = len(weights)
            weights.append(e.value)
        return QwenImageDit(weights^, name_to_idx^, self.shared.config)

    def _prepare_img(
        self,
        hidden_states: Tensor,
        ctx: DeviceContext,
    ) raises -> Tensor:
        return self.shared._linear_b(
            hidden_states, "img_in.weight", "img_in.bias", ctx
        )

    def _prepare_txt(
        self,
        encoder_hidden_states: Tensor,
        ctx: DeviceContext,
    ) raises -> Tensor:
        var cfg = self.shared.config
        ref txt_norm_w = self.shared._w(String("txt_norm.weight"))
        var txt_normed = rms_norm(encoder_hidden_states, txt_norm_w, cfg.eps, ctx)
        return self.shared._linear_b(
            txt_normed, "txt_in.weight", "txt_in.bias", ctx
        )

    def _prepare_temb(
        self,
        timestep: Float32,
        ctx: DeviceContext,
    ) raises -> Tensor:
        var cfg = self.shared.config
        var dim = cfg.inner_dim
        var dtype = self.shared._w(String("img_in.weight")).dtype()
        var t_host = List[Float32]()
        t_host.append(timestep * Float32(1000.0))
        var t_sh = List[Int]()
        t_sh.append(1)
        var t_tensor = Tensor.from_host(t_host, t_sh^, STDtype.F32, ctx)
        var t_embed = timestep_embedding(
            t_tensor, cfg.timestep_dim, ctx, Float32(10000.0), dtype
        )
        var t_embed_cast = reshape(t_embed, _shape3(1, 1, cfg.timestep_dim), ctx)
        var h1 = self.shared._linear_b(
            t_embed_cast,
            "time_text_embed.timestep_embedder.linear_1.weight",
            "time_text_embed.timestep_embedder.linear_1.bias",
            ctx,
        )
        h1 = silu(h1, ctx)
        var temb_3d = self.shared._linear_b(
            h1,
            "time_text_embed.timestep_embedder.linear_2.weight",
            "time_text_embed.timestep_embedder.linear_2.bias",
            ctx,
        )
        return reshape(temb_3d, _shape2(1, dim), ctx)

    def _finish[
        N_IMG: Int
    ](self, img: Tensor, temb: Tensor, ctx: DeviceContext) raises -> Tensor:
        var dim = self.shared.config.inner_dim
        var temb_silu = silu(temb, ctx)
        var mods = self.shared._linear_b(
            temb_silu, "norm_out.linear.weight", "norm_out.linear.bias", ctx
        )
        var out_scale = self.shared._mod_slice(mods, 0, dim, ctx)
        var out_shift = self.shared._mod_slice(mods, dim, dim, ctx)
        var normed = self.shared._layer_norm_no_affine(img, ctx)
        var modulated = self.shared._modulate(normed, out_scale, out_shift, ctx)
        return self.shared._linear_b(
            modulated, "proj_out.weight", "proj_out.bias", ctx
        )

    def forward[
        N_IMG: Int, N_TXT: Int, S: Int
    ](
        mut self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        timestep: Float32,
        frame: Int,
        h_latent: Int,
        w_latent: Int,
        ctx: DeviceContext,
    ) raises -> Tensor:
        comptime assert S == N_IMG + N_TXT, "S must equal N_IMG + N_TXT"
        var cfg = self.shared.config
        var img = self._prepare_img(hidden_states, ctx)
        var txt = self._prepare_txt(encoder_hidden_states, ctx)
        var temb = self._prepare_temb(timestep, ctx)
        var dtype = self.shared._w(String("img_in.weight")).dtype()
        var rope = build_qwenimage_rope_tables(
            frame, h_latent, w_latent, N_TXT, cfg.num_heads, cfg, dtype, ctx
        )
        self.loader.set_config(OffloadConfig.single_pass())
        self.loader.prefetch_with_ctx(0, ctx)
        for i in range(cfg.num_layers):
            var handle = self.loader.await_block(i, ctx)
            self.loader.prefetch_next_with_ctx(i, ctx)
            var tmp = self._block_model(handle.block)
            tmp._block_forward[N_IMG, N_TXT, S](
                i, img, txt, temb, rope[0], rope[1], N_TXT, ctx
            )
            self.loader.mark_active_block_done(ctx)

        return self._finish[N_IMG](img, temb, ctx)

    def forward_cfg[
        N_IMG: Int, N_TXT: Int, S: Int
    ](
        mut self,
        hidden_states: Tensor,
        encoder_pos: Tensor,
        encoder_neg: Tensor,
        timestep: Float32,
        real_txt_len: Int,
        frame: Int,
        h_latent: Int,
        w_latent: Int,
        ctx: DeviceContext,
    ) raises -> QwenImageCfgPreds:
        """Run positive and negative branches through each streamed block before
        unloading that block, avoiding duplicate block H2D loads for CFG.

        `real_txt_len` is the number of non-padded text tokens (after the
        34-token template drop). Text positions in [real_txt_len, N_TXT) are
        masked out of the joint attention via an additive -1e4 bias. This lets
        the pipeline pin N_TXT to a generous comptime max (e.g. 512) and accept
        arbitrary prompts at runtime without recompiling."""
        comptime assert S == N_IMG + N_TXT, "S must equal N_IMG + N_TXT"
        var cfg = self.shared.config
        var img_pos = self._prepare_img(hidden_states, ctx)
        var img_neg = self._prepare_img(hidden_states, ctx)
        var txt_pos = self._prepare_txt(encoder_pos, ctx)
        var txt_neg = self._prepare_txt(encoder_neg, ctx)
        var temb = self._prepare_temb(timestep, ctx)
        var dtype = self.shared._w(String("img_in.weight")).dtype()
        var rope = build_qwenimage_rope_tables(
            frame, h_latent, w_latent, N_TXT, cfg.num_heads, cfg, dtype, ctx
        )
        self.loader.set_config(OffloadConfig.synchronous_cfg_paired())
        self.loader.prefetch_with_ctx(0, ctx)
        for i in range(cfg.num_layers):
            var handle = self.loader.await_block(i, ctx)
            self.loader.prefetch_next_with_ctx(i, ctx)
            var tmp = self._block_model(handle.block)
            tmp._block_forward[N_IMG, N_TXT, S](
                i, img_pos, txt_pos, temb, rope[0], rope[1], real_txt_len, ctx
            )
            tmp._block_forward[N_IMG, N_TXT, S](
                i, img_neg, txt_neg, temb, rope[0], rope[1], real_txt_len, ctx
            )
            self.loader.mark_active_block_done(ctx)

        var pred_pos = self._finish[N_IMG](img_pos, temb, ctx)
        var pred_neg = self._finish[N_IMG](img_neg, temb, ctx)
        return QwenImageCfgPreds(pred_pos^, pred_neg^)

    def forward_cfg_mixed_text[
        N_IMG: Int, N_TXT_POS: Int, S_POS: Int, N_TXT_NEG: Int, S_NEG: Int
    ](
        mut self,
        hidden_states: Tensor,
        encoder_pos: Tensor,
        encoder_neg: Tensor,
        timestep: Float32,
        real_txt_len_pos: Int,
        real_txt_len_neg: Int,
        frame: Int,
        h_latent: Int,
        w_latent: Int,
        ctx: DeviceContext,
    ) raises -> QwenImageCfgPreds:
        """Run CFG with separate positive/negative text lengths.

        Qwen-Image's diffusers/Rust path drops the 34-token template prefix and
        keeps only non-padded text hidden states, so cond/uncond sequence lengths
        normally differ. This method still streams each block once, then runs the
        two branch-specific comptime attention shapes before unloading it.

        `real_txt_len_pos` / `real_txt_len_neg` are the non-padded text-token
        counts for each branch; positions in `[real_txt_len_*, N_TXT_*)` are
        masked out of joint attention with a -1e4 additive bias. The pipeline
        pads both branches to comptime-fixed N_TXT_POS / N_TXT_NEG (e.g. 512),
        decoupling prompt length from comptime shape.
        """
        comptime assert S_POS == N_IMG + N_TXT_POS, "S_POS must equal N_IMG + N_TXT_POS"
        comptime assert S_NEG == N_IMG + N_TXT_NEG, "S_NEG must equal N_IMG + N_TXT_NEG"
        var cfg = self.shared.config
        var img_pos = self._prepare_img(hidden_states, ctx)
        var img_neg = self._prepare_img(hidden_states, ctx)
        var txt_pos = self._prepare_txt(encoder_pos, ctx)
        var txt_neg = self._prepare_txt(encoder_neg, ctx)
        var temb = self._prepare_temb(timestep, ctx)
        var dtype = self.shared._w(String("img_in.weight")).dtype()
        var rope_pos = build_qwenimage_rope_tables(
            frame, h_latent, w_latent, N_TXT_POS, cfg.num_heads, cfg, dtype, ctx
        )
        var rope_neg = build_qwenimage_rope_tables(
            frame, h_latent, w_latent, N_TXT_NEG, cfg.num_heads, cfg, dtype, ctx
        )
        self.loader.set_config(OffloadConfig.synchronous_cfg_paired())
        self.loader.prefetch_with_ctx(0, ctx)
        for i in range(cfg.num_layers):
            var handle = self.loader.await_block(i, ctx)
            self.loader.prefetch_next_with_ctx(i, ctx)
            var tmp = self._block_model(handle.block)
            tmp._block_forward[N_IMG, N_TXT_POS, S_POS](
                i, img_pos, txt_pos, temb, rope_pos[0], rope_pos[1], real_txt_len_pos, ctx
            )
            tmp._block_forward[N_IMG, N_TXT_NEG, S_NEG](
                i, img_neg, txt_neg, temb, rope_neg[0], rope_neg[1], real_txt_len_neg, ctx
            )
            self.loader.mark_active_block_done(ctx)

        var pred_pos = self._finish[N_IMG](img_pos, temb, ctx)
        var pred_neg = self._finish[N_IMG](img_neg, temb, ctx)
        return QwenImageCfgPreds(pred_pos^, pred_neg^)

    def forward_edit_cfg[
        N_TARGET: Int, N_REF: Int, N_TXT: Int, S: Int
    ](
        mut self,
        target_hidden_states: Tensor,
        reference_hidden_states: Tensor,
        encoder_pos: Tensor,
        encoder_neg: Tensor,
        timestep: Float32,
        ref_timestep: Float32,
        frame: Int,
        h_latent: Int,
        w_latent: Int,
        ctx: DeviceContext,
    ) raises -> QwenImageCfgPreds:
        """Qwen-Image-Edit single-reference CFG forward.

        Concatenates `[target, reference]`, runs target tokens with `timestep`
        modulation and reference tokens with `ref_timestep` modulation
        (`zero_cond_t` path), then returns only the target-token predictions.
        """
        comptime assert S == N_TARGET + N_REF + N_TXT, "S must equal target + ref + text"
        var cfg = self.shared.config
        var hidden_states = concat(
            1, ctx, target_hidden_states, reference_hidden_states
        )
        var img_pos = self._prepare_img(hidden_states, ctx)
        var img_neg = self._prepare_img(hidden_states, ctx)
        var txt_pos = self._prepare_txt(encoder_pos, ctx)
        var txt_neg = self._prepare_txt(encoder_neg, ctx)
        var temb_target = self._prepare_temb(timestep, ctx)
        var temb_ref = self._prepare_temb(ref_timestep, ctx)
        var dtype = self.shared._w(String("img_in.weight")).dtype()
        var rope = build_qwenimage_edit_rope_tables(
            frame, h_latent, w_latent, N_TXT, cfg.num_heads, cfg, dtype, ctx
        )
        self.loader.set_config(OffloadConfig.synchronous_cfg_paired())
        self.loader.prefetch_with_ctx(0, ctx)
        for i in range(cfg.num_layers):
            var handle = self.loader.await_block(i, ctx)
            self.loader.prefetch_next_with_ctx(i, ctx)
            var tmp = self._block_model(handle.block)
            tmp._block_forward_edit[N_TARGET, N_REF, N_TXT, S](
                i, img_pos, txt_pos, temb_target, temb_ref, rope[0], rope[1], N_TXT, ctx
            )
            tmp._block_forward_edit[N_TARGET, N_REF, N_TXT, S](
                i, img_neg, txt_neg, temb_target, temb_ref, rope[0], rope[1], N_TXT, ctx
            )
            self.loader.mark_active_block_done(ctx)

        var pred_pos_full = self._finish[N_TARGET + N_REF](img_pos, temb_target, ctx)
        var pred_neg_full = self._finish[N_TARGET + N_REF](img_neg, temb_target, ctx)
        var pred_pos = slice(pred_pos_full, 1, 0, N_TARGET, ctx)
        var pred_neg = slice(pred_neg_full, 1, 0, N_TARGET, ctx)
        return QwenImageCfgPreds(pred_pos^, pred_neg^)

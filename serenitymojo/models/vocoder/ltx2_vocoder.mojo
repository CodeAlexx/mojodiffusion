# models/vocoder/ltx2_vocoder.mojo — LTX-2.3 BigVGAN-v2 vocoder + BWE.
#
# Pure-Mojo port of `LTX2Vocoder` / `LTX2VocoderWithBWE`
# (inference-flame/src/vae/ltx2_vocoder.rs). Builds entirely on the gated
# primitives: conv1d / conv_transpose1d / zero_insert1d / replicate_pad1d
# (P-conv), snake_beta (P-snake), activation1d (P1), compute_mel (P-stft),
# and tensor_algebra (slice / reshape / permute / concat / mul_scalar / add).
#
# === Vocoder (stereo 16 kHz) — ltx2_vocoder.rs:660-895 ===
#   mel [B,2,T,F] -> permute(0,1,3,2) -> reshape [B, 2*F, T]
#   conv_pre  Conv1d(128 -> 1536, k=7, pad=3)
#   for i in 0..6:
#     ConvTranspose1d(stride=upsample_rates[i], k=ups[i].K, pad=(K-stride)/2)
#     x = mean over 3 AMPBlock1(stage convs, dilations [1,3,5])
#   act_post  Activation1d(SnakeBeta(final_ch))
#   conv_post Conv1d(final_ch -> 2, k=7, pad=3)
#   no final tanh for LTX2.3 checkpoint metadata (`use_tanh_at_final=false`);
#   BWE generator also skips final activation.
#
# upsample_rates: 6 stages -> [5,2,2,2,2,2]; 5 stages (BWE) -> [6,5,2,2,2].
# resblock kernel sizes per stage position: [3,7,11]; dilations [1,3,5].
# get_padding(k,d) = (k*d - d)/2.
#
# === BWE (48 kHz) — ltx2_vocoder.rs:980-1080 ===
#   x16   = vocoder.forward(mel)                          # [B,2,L16]
#   pad x16 to a multiple of hop_length (=80) on the right (zeros)
#   mel_bwe = compute_mel(x16)            (P-stft; hop=80)  -> [B,2,64,Tf]
#   mel_for_bwe = mel_bwe.permute(0,1,3,2)                # [B,2,Tf,64]
#   residual = bwe_generator.forward(mel_for_bwe)         # [B,2,L48] (no tanh)
#   skip     = sinc_upsample(x16, ratio=3)                # [B,2,~L48]
#   out      = clamp(residual[:mix] + skip[:mix], -1, 1), trimmed to L16*3.
#
# Weights load as F32 (`_load_w`) and the whole chain computes F32 — the forward
# casts activations to F32 (bf16 compounds through 108 convs and fails spectral
# parity), so F32 weights are REQUIRED (bf16 weights + F32 acts = conv mismatch).
# Matches production (NAVA init_ltx_vae builds the audio VAE/vocoder in F32).
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.math import cos, sin, pi
from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.ops.conv1d import (
    conv1d,
    conv_transpose1d,
    precompute_conv_transpose_weight,
    zero_insert1d,
    replicate_pad1d,
)
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.snake import snake_beta, snake_beta_precompute
from serenitymojo.ops.activation1d import activation1d
from serenitymojo.ops.tensor_algebra import (
    reshape,
    permute,
    slice,
    mul_scalar,
    add,
    concat,
)
from serenitymojo.ops.unary import tanh_op
from serenitymojo.models.vocoder.ltx2_stft import compute_mel


comptime CONV_PRE_KERNEL = 7
comptime CONV_POST_KERNEL = 7
comptime NUM_KERNELS_PER_STAGE = 3
comptime SNAKE_EPS = Float32(1e-9)


# resblock kernel size per stage-position (Python resblock_kernel_sizes).
def _resblock_kernel_size(kernel_i: Int) -> Int:
    if kernel_i == 0:
        return 3
    elif kernel_i == 1:
        return 7
    else:
        return 11


# dilation per convs1 index (DILATIONS = [1,3,5]).
def _dilation(i: Int) -> Int:
    if i == 0:
        return 1
    elif i == 1:
        return 3
    else:
        return 5


# upsample rate per stage; 6 stages -> [5,2,2,2,2,2]; 5 stages -> [6,5,2,2,2].
def _upsample_rate(num_ups: Int, i: Int) raises -> Int:
    if num_ups == 6:
        if i == 0:
            return 5
        return 2
    elif num_ups == 5:
        if i == 0:
            return 6
        elif i == 1:
            return 5
        return 2
    raise Error("vocoder: unsupported ups count")


# get_padding(k,d) = (k*d - d)/2  (ltx2_vocoder.rs:64)
def _get_padding(k: Int, d: Int) -> Int:
    return (k * d - d) // 2


# ── weight loading helper ─────────────────────────────────────────────────────
# Runtime loader: F32. `LTX2VocoderWithBWE.forward` casts activations to F32 for
# the whole BigVGAN+BWE chain (bf16 compounds through 108 convs and fails spectral
# parity), so the WEIGHTS must be F32 too — bf16 weights + F32 activations is a
# conv dtype mismatch. This also matches production (NAVA's audio VAE/vocoder run
# in F32 via init_ltx_vae). Upcasts BF16 checkpoint storage on the host.
def _load_w(ref st: SafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var view = from_parts(info.dtype, info.shape.copy(), bytes)
    # dtype-contract: allow-f32-boundary - BigVGAN+BWE vocoder reference runs F32.
    return Tensor.from_view_as_f32(view, ctx)


def _put(
    mut tensors: List[ArcPointer[Tensor]],
    mut name_to_idx: Dict[String, Int],
    nm: String,
    var t: Tensor,
) raises:
    name_to_idx[nm] = len(tensors)
    tensors.append(ArcPointer(t^))


def _put_act(
    mut acts: List[ArcPointer[ActParams]],
    mut act_name_to_idx: Dict[String, Int],
    nm: String,
    var a: ActParams,
) raises:
    act_name_to_idx[nm] = len(acts)
    acts.append(ArcPointer(a^))


# Reshape a [C] param to [C,1,1] and exp it (alpha) / 1/(exp+eps) it (beta).
def _snake_params(
    ref st: SafeTensors, prefix: String, ctx: DeviceContext
) raises -> Tuple[Tensor, Tensor]:
    """Load act.alpha/act.beta [C], precompute (alpha_exp, inv_beta_eps) in
    [C,1,1] layout (matches ltx2_vocoder.rs:522-524)."""
    var alpha_bf = _load_w(st, prefix + ".act.alpha", ctx)
    var beta_bf = _load_w(st, prefix + ".act.beta", ctx)
    # dtype-contract: allow-f32-boundary
    # LTX2 computes SnakeBeta exp/reciprocal inside the vocoder F32 autocast block.
    var alpha = cast_tensor(alpha_bf^, STDtype.F32, ctx)
    var beta = cast_tensor(beta_bf^, STDtype.F32, ctx)
    var C = alpha.shape()[0]
    var alpha3 = reshape(alpha, [C, 1, 1], ctx)
    var beta3 = reshape(beta, [C, 1, 1], ctx)
    var pre = snake_beta_precompute(alpha3, beta3, ctx)
    var ae = reshape(pre[0], [C, 1, 1], ctx)
    var ibe = reshape(pre[1], [C, 1, 1], ctx)
    return (ae^, ibe^)


# ── ActParams ─────────────────────────────────────────────────────────────────
# Holds the precomputed SnakeBeta params + the [1,1,12] kaiser FIR filters for
# one Activation1d. Reusable for resblock acts and act_post.
struct ActParams(Movable):
    var alpha_exp: Tensor     # [C,1,1]
    var inv_beta_eps: Tensor  # [C,1,1]
    var up_filter: Tensor     # [1,1,12]
    var down_filter: Tensor   # [1,1,12]

    def __init__(
        out self,
        var alpha_exp: Tensor,
        var inv_beta_eps: Tensor,
        var up_filter: Tensor,
        var down_filter: Tensor,
    ):
        self.alpha_exp = alpha_exp^
        self.inv_beta_eps = inv_beta_eps^
        self.up_filter = up_filter^
        self.down_filter = down_filter^

    @staticmethod
    def load(
        ref st: SafeTensors, prefix: String, ctx: DeviceContext
    ) raises -> ActParams:
        var pre = _snake_params(st, prefix, ctx)
        var sh = pre[0].shape()
        var ae = reshape(pre[0], sh.copy(), ctx)
        var ibe = reshape(pre[1], sh.copy(), ctx)
        var up = _load_w(st, prefix + ".upsample.filter", ctx)
        var down = _load_w(st, prefix + ".downsample.lowpass.filter", ctx)
        return ActParams(ae^, ibe^, up^, down^)

    def apply(self, x: Tensor, ctx: DeviceContext) raises -> Tensor:
        if x.dtype() != self.alpha_exp.dtype():
            var ae = cast_tensor(self.alpha_exp, x.dtype(), ctx)
            var ibe = cast_tensor(self.inv_beta_eps, x.dtype(), ctx)
            return activation1d(
                x, ae, ibe,
                self.up_filter, self.down_filter, ctx,
            )
        return activation1d(
            x, self.alpha_exp, self.inv_beta_eps,
            self.up_filter, self.down_filter, ctx,
        )


# ── helpers to fetch from the weight store ────────────────────────────────────
struct VocoderWeights(Movable):
    """All weights for ONE vocoder (base or BWE), indexed by short name.

    conv_pre / conv_post / per-resblock conv weights are kept as-loaded (BF16,
    [Cout,Cin,K]); ups weights are PRE-TRANSFORMED into Conv1d layout (preconv)
    at load. SnakeBeta params + kaiser filters live in `acts`."""

    var tensors: List[ArcPointer[Tensor]]
    var name_to_idx: Dict[String, Int]
    var acts: List[ArcPointer[ActParams]]
    var act_name_to_idx: Dict[String, Int]
    var num_ups: Int
    var final_channels: Int
    var apply_final_tanh: Bool

    def __init__(
        out self,
        var tensors: List[ArcPointer[Tensor]],
        var name_to_idx: Dict[String, Int],
        var acts: List[ArcPointer[ActParams]],
        var act_name_to_idx: Dict[String, Int],
        num_ups: Int,
        final_channels: Int,
        apply_final_tanh: Bool,
    ):
        self.tensors = tensors^
        self.name_to_idx = name_to_idx^
        self.acts = acts^
        self.act_name_to_idx = act_name_to_idx^
        self.num_ups = num_ups
        self.final_channels = final_channels
        self.apply_final_tanh = apply_final_tanh

    def _t(self, name: String) raises -> ref [self.tensors] Tensor:
        if name not in self.name_to_idx:
            raise Error(String("vocoder: missing weight: ") + name)
        return self.tensors[self.name_to_idx[name]][]

    def _a(self, name: String) raises -> ref [self.acts] ActParams:
        if name not in self.act_name_to_idx:
            raise Error(String("vocoder: missing act params: ") + name)
        return self.acts[self.act_name_to_idx[name]][]

    # Fresh D2D copy of a stored bias, for building an Optional[Tensor] arg
    # (conv1d's bias param needs an owned value, not a borrowed ref).
    def _bias(self, name: String, ctx: DeviceContext) raises -> Tensor:
        ref b = self._t(name)
        return reshape(b, b.shape(), ctx)

    @staticmethod
    def load(
        ref st: SafeTensors,
        prefix: String,
        apply_final_tanh: Bool,
        ctx: DeviceContext,
    ) raises -> VocoderWeights:
        """Load `{prefix}.*` (prefix like 'vocoder.vocoder' or
        'vocoder.bwe_generator')."""
        var tensors = List[ArcPointer[Tensor]]()
        var name_to_idx = Dict[String, Int]()
        var acts = List[ArcPointer[ActParams]]()
        var act_name_to_idx = Dict[String, Int]()

        # conv_pre
        _put(tensors, name_to_idx, prefix + ".conv_pre.weight",
            _load_w(st, prefix + ".conv_pre.weight", ctx))
        _put(tensors, name_to_idx, prefix + ".conv_pre.bias",
            _load_w(st, prefix + ".conv_pre.bias", ctx))

        # count ups
        var num_ups = 0
        while (prefix + ".ups." + String(num_ups) + ".weight") in st.tensors:
            num_ups += 1
        if num_ups == 0:
            raise Error(String("vocoder: no ups for prefix ") + prefix)

        var channels_per_stage = List[Int]()
        for i in range(num_ups):
            var wname = prefix + ".ups." + String(i) + ".weight"
            var w_raw = _load_w(st, wname, ctx)
            # ConvTranspose1d weight [Cin, Cout, K]; precompute -> Conv1d layout.
            var c_out = w_raw.shape()[1]
            channels_per_stage.append(c_out)
            var w_pre = precompute_conv_transpose_weight(w_raw, 1, ctx)
            _put(tensors, name_to_idx, prefix + ".ups." + String(i) + ".preconv", w_pre^)
            _put(tensors, name_to_idx, prefix + ".ups." + String(i) + ".bias",
                _load_w(st, prefix + ".ups." + String(i) + ".bias", ctx))

        var final_channels = channels_per_stage[num_ups - 1]

        # resblocks: num_ups stages x 3 AMPBlock1
        for stage_i in range(num_ups):
            for kernel_i in range(NUM_KERNELS_PER_STAGE):
                var block_idx = stage_i * NUM_KERNELS_PER_STAGE + kernel_i
                var bp = prefix + ".resblocks." + String(block_idx)
                for j in range(3):
                    var c1 = bp + ".convs1." + String(j)
                    var c2 = bp + ".convs2." + String(j)
                    _put(tensors, name_to_idx, c1 + ".weight", _load_w(st, c1 + ".weight", ctx))
                    _put(tensors, name_to_idx, c1 + ".bias", _load_w(st, c1 + ".bias", ctx))
                    _put(tensors, name_to_idx, c2 + ".weight", _load_w(st, c2 + ".weight", ctx))
                    _put(tensors, name_to_idx, c2 + ".bias", _load_w(st, c2 + ".bias", ctx))
                    _put_act(acts, act_name_to_idx, bp + ".acts1." + String(j),
                            ActParams.load(st, bp + ".acts1." + String(j), ctx))
                    _put_act(acts, act_name_to_idx, bp + ".acts2." + String(j),
                            ActParams.load(st, bp + ".acts2." + String(j), ctx))

        # act_post
        _put_act(acts, act_name_to_idx, prefix + ".act_post",
                 ActParams.load(st, prefix + ".act_post", ctx))

        # conv_post (bias optional)
        _put(tensors, name_to_idx, prefix + ".conv_post.weight",
            _load_w(st, prefix + ".conv_post.weight", ctx))
        if (prefix + ".conv_post.bias") in st.tensors:
            _put(tensors, name_to_idx, prefix + ".conv_post.bias",
                _load_w(st, prefix + ".conv_post.bias", ctx))

        return VocoderWeights(
            tensors^, name_to_idx^, acts^, act_name_to_idx^,
            num_ups, final_channels, apply_final_tanh,
        )


# One AMPBlock1 forward (ltx2_vocoder.rs:598-653).
#   for i in 0..3:
#     xt = acts1[i].apply(x)
#     xt = conv1d(xt, convs1[i], bias, stride=1, pad=get_padding(k,DIL[i]), dil=DIL[i])
#     xt = acts2[i].apply(xt)
#     xt = conv1d(xt, convs2[i], bias, stride=1, pad=get_padding(k,1), dil=1)
#     x  = x + xt
def _ampblock_forward(
    ref w: VocoderWeights,
    bp: String,
    k: Int,
    var x: Tensor,
    ctx: DeviceContext,
) raises -> Tensor:
    for i in range(3):
        var d = _dilation(i)
        var pad1 = _get_padding(k, d)
        var xt = w._a(bp + ".acts1." + String(i)).apply(x, ctx)
        xt = conv1d(
            xt, w._t(bp + ".convs1." + String(i) + ".weight"),
            Optional[Tensor](w._bias(bp + ".convs1." + String(i) + ".bias", ctx)),
            1, pad1, d, 1, ctx,
        )
        xt = w._a(bp + ".acts2." + String(i)).apply(xt, ctx)
        var pad2 = _get_padding(k, 1)
        xt = conv1d(
            xt, w._t(bp + ".convs2." + String(i) + ".weight"),
            Optional[Tensor](w._bias(bp + ".convs2." + String(i) + ".bias", ctx)),
            1, pad2, 1, 1, ctx,
        )
        x = add(x, xt, ctx)
    return x^


# Vocoder forward: mel [B,2,T,F] -> waveform [B,2,L].
def vocoder_forward(
    ref w: VocoderWeights, prefix: String, mel: Tensor, ctx: DeviceContext
) raises -> Tensor:
    var d = mel.shape()
    if len(d) != 4 or d[1] != 2:
        raise Error("vocoder_forward: mel must be [B,2,T,F]")
    var B = d[0]
    var T = d[2]
    var F = d[3]
    # [B,2,T,F] -> [B,2,F,T] -> [B,2*F,T]
    var x = permute(mel, [0, 1, 3, 2], ctx)
    x = reshape(x, [B, 2 * F, T], ctx)

    # conv_pre k=7 pad=3
    x = conv1d(
        x, w._t(prefix + ".conv_pre.weight"),
        Optional[Tensor](w._bias(prefix + ".conv_pre.bias", ctx)),
        1, CONV_PRE_KERNEL // 2, 1, 1, ctx,
    )

    for i in range(w.num_ups):
        var stride = _upsample_rate(w.num_ups, i)
        var K = w._t(prefix + ".ups." + String(i) + ".preconv").shape()[2]
        var padding = (K - stride) // 2
        # ConvTranspose1d via preconv weight: zero-insert(stride) + side-pad
        # (side = (K-1) - padding for dil=1) + conv1d(stride=1).
        var x_zi = zero_insert1d(x, stride, ctx)
        var side = (K - 1) - padding
        if side < 0:
            raise Error("vocoder_forward: negative ConvTranspose side pad")
        x = conv1d(
            x_zi, w._t(prefix + ".ups." + String(i) + ".preconv"),
            Optional[Tensor](w._bias(prefix + ".ups." + String(i) + ".bias", ctx)),
            1, side, 1, 1, ctx,
        )

        # 3 resblocks averaged
        var start = i * NUM_KERNELS_PER_STAGE
        var x0 = reshape(x, x.shape(), ctx)
        var sum_t = _ampblock_forward(
            w, prefix + ".resblocks." + String(start), _resblock_kernel_size(0),
            x0^, ctx,
        )
        for jj in range(1, NUM_KERNELS_PER_STAGE):
            var block_idx = start + jj
            var k = _resblock_kernel_size(jj)
            var bp = prefix + ".resblocks." + String(block_idx)
            var x_in = reshape(x, x.shape(), ctx)  # fresh copy per resblock
            var out_j = _ampblock_forward(w, bp, k, x_in^, ctx)
            sum_t = add(sum_t, out_j, ctx)
        x = mul_scalar(sum_t, Float32(1.0) / Float32(NUM_KERNELS_PER_STAGE), ctx)

    # act_post + conv_post (+ tanh)
    x = w._a(prefix + ".act_post").apply(x, ctx)
    var has_post_bias = (prefix + ".conv_post.bias") in w.name_to_idx
    if has_post_bias:
        x = conv1d(
            x, w._t(prefix + ".conv_post.weight"),
            Optional[Tensor](w._bias(prefix + ".conv_post.bias", ctx)),
            1, CONV_POST_KERNEL // 2, 1, 1, ctx,
        )
    else:
        x = conv1d(
            x, w._t(prefix + ".conv_post.weight"), None,
            1, CONV_POST_KERNEL // 2, 1, 1, ctx,
        )
    if w.apply_final_tanh:
        x = tanh_op(x, ctx)
    return x^


# ── hann-windowed sinc resample filter (ltx2_vocoder.rs:1083-1116) ────────────
# Returns (filter[1,1,kernel_size] BF16, pad, pad_left, pad_right) for `ratio`.
def hann_sinc_resample_filter(
    ratio: Int, ctx: DeviceContext
) raises -> Tuple[Tensor, Int, Int, Int]:
    var rolloff = Float64(0.99)
    var lowpass_filter_width = Float64(6.0)
    var width = Int((lowpass_filter_width / rolloff).__ceil__())
    var kernel_size = 2 * width * ratio + 1
    var pad = width
    var pad_left = 2 * width * ratio
    var pad_right = kernel_size - ratio
    var data = List[Float32]()
    for i in range(kernel_size):
        var time_axis = (Float64(i) / Float64(ratio) - Float64(width)) * rolloff
        var time_clamped = time_axis
        if time_clamped < -lowpass_filter_width:
            time_clamped = -lowpass_filter_width
        elif time_clamped > lowpass_filter_width:
            time_clamped = lowpass_filter_width
        var c = cos(time_clamped * pi / lowpass_filter_width / Float64(2.0))
        var window = c * c
        var sinc: Float64
        if time_axis.__abs__() < Float64(1.0e-8):
            sinc = Float64(1.0)
        else:
            var xx = pi * time_axis
            sinc = sin(xx) / xx
        data.append(Float32(sinc * window * rolloff / Float64(ratio)))
    var fsh = List[Int]()
    fsh.append(1); fsh.append(1); fsh.append(kernel_size)
    var filt = Tensor.from_host(data, fsh^, STDtype.F32, ctx)  # F32: convolved with F32 vocoder output
    return (filt^, pad, pad_left, pad_right)


# ── LTX2VocoderWithBWE ────────────────────────────────────────────────────────
struct LTX2VocoderWithBWE(Movable):
    var voc: VocoderWeights
    var bwe: VocoderWeights
    var mel_basis: Tensor          # [64,257]
    var forward_basis: Tensor      # [514,1,512]
    var resample_preconv: Tensor   # Conv1d-layout resample filter
    var resample_ratio: Int
    var resample_pad: Int
    var resample_pad_left: Int
    var resample_pad_right: Int
    var hop_length: Int
    var input_sr: Int
    var output_sr: Int

    def __init__(
        out self,
        var voc: VocoderWeights,
        var bwe: VocoderWeights,
        var mel_basis: Tensor,
        var forward_basis: Tensor,
        var resample_preconv: Tensor,
        resample_ratio: Int,
        resample_pad: Int,
        resample_pad_left: Int,
        resample_pad_right: Int,
        hop_length: Int,
        input_sr: Int,
        output_sr: Int,
    ):
        self.voc = voc^
        self.bwe = bwe^
        self.mel_basis = mel_basis^
        self.forward_basis = forward_basis^
        self.resample_preconv = resample_preconv^
        self.resample_ratio = resample_ratio
        self.resample_pad = resample_pad
        self.resample_pad_left = resample_pad_left
        self.resample_pad_right = resample_pad_right
        self.hop_length = hop_length
        self.input_sr = input_sr
        self.output_sr = output_sr

    @staticmethod
    def from_file(path: String, ctx: DeviceContext) raises -> LTX2VocoderWithBWE:
        var st = SafeTensors.open(path)
        var voc = VocoderWeights.load(st, String("vocoder.vocoder"), False, ctx)
        var bwe = VocoderWeights.load(st, String("vocoder.bwe_generator"), False, ctx)
        var mel_basis = _load_w(st, String("vocoder.mel_stft.mel_basis"), ctx)
        var forward_basis = _load_w(
            st, String("vocoder.mel_stft.stft_fn.forward_basis"), ctx
        )
        var input_sr = 16000
        var output_sr = 48000
        var hop_length = 80
        var ratio = output_sr // input_sr
        var rf = hann_sinc_resample_filter(ratio, ctx)
        var preconv = precompute_conv_transpose_weight(rf[0], 1, ctx)
        return LTX2VocoderWithBWE(
            voc^, bwe^, mel_basis^, forward_basis^, preconv^,
            ratio, rf[1], rf[2], rf[3], hop_length, input_sr, output_sr,
        )

    def output_sample_rate(self) -> Int:
        return self.output_sr

    # sinc_upsample (ltx2_vocoder.rs:1049-1080)
    def _sinc_upsample(self, x: Tensor, ctx: DeviceContext) raises -> Tensor:
        var d = x.shape()
        var B = d[0]
        var C = d[1]
        var L = d[2]
        var x_bc = reshape(x, [B * C, 1, L], ctx)
        var x_pad = replicate_pad1d(
            x_bc, self.resample_pad, self.resample_pad, ctx
        )
        var K = self.resample_preconv.shape()[2]
        # conv_transpose1d_prepare(x_pad, ratio, K-1, K-1): zero-insert(ratio) +
        # side-pad (K-1) each side, then conv1d(stride=1) with the preconv weight.
        var x_zi = zero_insert1d(x_pad, self.resample_ratio, ctx)
        var y = conv1d(x_zi, self.resample_preconv, None, 1, K - 1, 1, 1, ctx)
        y = mul_scalar(y, Float32(self.resample_ratio), ctx)
        var y_len = y.shape()[2]
        var crop_len = y_len - self.resample_pad_left - self.resample_pad_right
        if crop_len <= 0:
            raise Error("sinc_upsample: non-positive crop length")
        var y_cropped = slice(y, 2, self.resample_pad_left, crop_len, ctx)
        return reshape(y_cropped, [B, C, crop_len], ctx)

    # compute_mel wrapper (ltx2_vocoder.rs:1018-1047): audio [B,C,T] -> [B,C,64,Tf]
    def _compute_mel(self, audio: Tensor, ctx: DeviceContext) raises -> Tensor:
        return compute_mel(
            audio, self.forward_basis, self.mel_basis, self.hop_length, ctx
        )

    def forward(self, mel: Tensor, ctx: DeviceContext) raises -> Tensor:
        var output_dtype = mel.dtype()
        # dtype-contract: allow-f32-boundary
        # LTX2 VocoderWithBWE.forward autocasts the whole BigVGAN+BWE chain to F32
        # compute, then returns to the input dtype; BF16 intermediates compound
        # through 108 convolutions and fail spectral parity.
        var mel_compute = cast_tensor(mel, STDtype.F32, ctx)
        var x = vocoder_forward(self.voc, String("vocoder.vocoder"), mel_compute^, ctx)
        var length_low_rate = x.shape()[2]
        var output_length = length_low_rate * self.output_sr // self.input_sr

        var remainder = length_low_rate % self.hop_length
        if remainder != 0:
            # right zero-pad to a multiple of hop_length: build zeros, concat.
            var pad_n = self.hop_length - remainder
            var d = x.shape()
            var zeros_h = List[Float32]()
            zeros_h.resize(d[0] * d[1] * pad_n, Float32(0.0))
            var zsh = List[Int]()
            zsh.append(d[0]); zsh.append(d[1]); zsh.append(pad_n)
            var zeros = Tensor.from_host(zeros_h, zsh^, x.dtype(), ctx)
            x = concat(2, ctx, x, zeros)

        var mel_bwe = self._compute_mel(x, ctx)           # [B,2,64,Tf]
        var mel_for_bwe = permute(mel_bwe, [0, 1, 3, 2], ctx)  # [B,2,Tf,64]
        var residual = vocoder_forward(
            self.bwe, String("vocoder.bwe_generator"), mel_for_bwe, ctx
        )
        var skip = self._sinc_upsample(x, ctx)

        var residual_len = residual.shape()[2]
        var skip_len = skip.shape()[2]
        var mix_len = residual_len if residual_len < skip_len else skip_len
        if residual_len != mix_len:
            residual = slice(residual, 2, 0, mix_len, ctx)
        if skip_len != mix_len:
            skip = slice(skip, 2, 0, mix_len, ctx)

        var out = add(residual, skip, ctx)
        out = _clamp_pm1(out, ctx)
        var out_len = out.shape()[2]
        if out_len > output_length:
            out = slice(out, 2, 0, output_length, ctx)
        if output_dtype != STDtype.F32:
            return cast_tensor(out, output_dtype, ctx)
        return out^


# clamp(x, -1, 1) reuses the stft clamp_log without log — do it via two scalar
# ops would change values; instead a dedicated tiny clamp. We reuse the
# already-gated pattern by clamping in a fused kernel-free manner: min then max.
# Simplest correct path: clamp via tensor ops is unavailable, so use a small
# elementwise helper here.
from std.gpu import global_idx
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout

comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256


def _clamp_pm1_kernel_bf16(
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[DType.bfloat16]](x[i]).cast[DType.float32]()
        if v < Float32(-1.0):
            v = Float32(-1.0)
        elif v > Float32(1.0):
            v = Float32(1.0)
        o[i] = rebind[o.element_type](v.cast[DType.bfloat16]())


def _clamp_pm1_kernel_f32(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[DType.float32]](x[i])
        if v < Float32(-1.0):
            v = Float32(-1.0)
        elif v > Float32(1.0):
            v = Float32(1.0)
        o[i] = rebind[o.element_type](v)


def _clamp_pm1(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var dt = x.dtype().to_mojo_dtype()
    var n = x.numel()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var grid = (n + _BLOCK - 1) // _BLOCK
    if dt == DType.float32:
        var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[Float32](), rl
        )
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), rl
        )
        ctx.enqueue_function[_clamp_pm1_kernel_f32, _clamp_pm1_kernel_f32](
            X, O, n, grid_dim=grid, block_dim=_BLOCK
        )
    else:
        var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            x.buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), rl
        )
        ctx.enqueue_function[_clamp_pm1_kernel_bf16, _clamp_pm1_kernel_bf16](
            X, O, n, grid_dim=grid, block_dim=_BLOCK
        )
    ctx.synchronize()
    return Tensor(out_buf^, x.shape(), x.dtype())

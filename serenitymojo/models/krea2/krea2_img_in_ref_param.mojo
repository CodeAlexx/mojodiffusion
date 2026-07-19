# serenitymojo/models/krea2/krea2_img_in_ref_param.mojo
#
# PORT of models/klein/klein_img_in_ref_param.mojo to KREA2 (dims are the only
# difference: krea2 `first` is [D=6144, in_ch=64], see config.mojo / the real
# krea2_raw_bf16.safetensors header `first.weight F32 [6144, 64]`).
#
# The img_in_ref reference-conditioning projection as a STANDALONE TRAINED
# PARAMETER that lives alongside the krea2 LoRA set. krea2 is text-to-image only;
# ai-toolkit's "krea2 o-edit" ADDS image-EDIT reference conditioning. This is the
# same parallel additive input projection Klein uses:
#     img = first(noise_tokens) + linear(ref_tokens, img_in_ref)
# where img_in_ref is a full [D, in_ch] matrix (NOT low-rank).
#
# STORAGE (mirrors klein's / LoraAdapter): BF16 weight + F32 AdamW moments (m/v).
# Zero-init the weight -> the flags-off / img_in_ref=0 identity (the C13 contract)
# holds: linear(ref_tokens, 0) == 0 -> the summed input projection is byte-
# identical to today's text-to-image krea2 path.
#
# OPTIMIZER: reuses the EXACT per-adapter AdamW helper the (klein-shared) LoRA set
# uses (models/klein/lora_adapter.mojo `_adamw_host_list_precomputed` — SerenityTrainer
# bias-corrected AdamW with stochastic BF16 rounding). krea2's LoRA path already
# reuses klein's lora infrastructure (krea2_stack_parity imports
# klein.lora_block.LoraAdapterDevice), so this is the SAME math, not a reimpl.
#
# SAVE/RESUME (byte-exact): a small safetensors file written next to the krea2 LoRA
# checkpoint via io/safetensors_writer.save_safetensors. The BF16 weight round-trips
# byte-exact (bf16->f32 widen on read is exact; f32->bf16 requantize on load is
# exact for a value that is already a bf16). Moments are F32, exact.
#
# Mojo 1.0.0b1: `def` not `fn`; BF16 storage; F32 moments; host List carriers.

from std.collections import List
from std.builtin.dtype import DType
from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from std.math import sqrt
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.training.lora_save import _read_f32
# REUSE the exact per-adapter AdamW leg the (klein-shared) krea2 LoRA set uses.
from serenitymojo.models.klein.lora_adapter import _adamw_host_list_precomputed


comptime KREA2_IMG_IN_REF_W_KEY = "img_in_ref.weight"
comptime KREA2_IMG_IN_REF_MA_KEY = "img_in_ref.adam_m"
comptime KREA2_IMG_IN_REF_VA_KEY = "img_in_ref.adam_v"


# ── the trained param: [D, in_ch] BF16 weight + F32 AdamW moments ─────────────
struct Krea2ImgInRefParam(Copyable, Movable):
    var w: List[BFloat16]   # [D, in_ch]  BF16 storage (zero-init => flags-off id)
    var m: List[Float32]    # [D*in_ch]   AdamW first moment
    var v: List[Float32]    # [D*in_ch]   AdamW second moment
    var D: Int
    var in_ch: Int

    def __init__(
        out self, var w: List[BFloat16],
        var m: List[Float32], var v: List[Float32], D: Int, in_ch: Int,
    ):
        self.w = w^
        self.m = m^
        self.v = v^
        self.D = D
        self.in_ch = in_ch


# ── ZERO-INIT (the C13 flags-off contract): weight 0 -> ref projection 0 ──────
def make_krea2_img_in_ref_param(D: Int, in_ch: Int) -> Krea2ImgInRefParam:
    var n = D * in_ch
    var zbf = Float32(0.0).cast[DType.bfloat16]()
    var w = List[BFloat16]()
    var m = List[Float32]()
    var v = List[Float32]()
    for _ in range(n):
        w.append(zbf)
        m.append(Float32(0.0))
        v.append(Float32(0.0))
    return Krea2ImgInRefParam(w^, m^, v^, D, in_ch)


# ── materialize the F32 [D, in_ch] weight the stack/compute-unit forward wants ─
def krea2_img_in_ref_w_f32(param: Krea2ImgInRefParam) -> List[Float32]:
    var out = List[Float32]()
    for i in range(len(param.w)):
        out.append(param.w[i].cast[DType.float32]())
    return out^


# ── materialize the device [D, in_ch] weight the stack forward consumes ───────
# The krea2 stack forward/backward take DEVICE tensors (TArc), unlike Klein's
# host-list stack. Upload the F32 widened weight so linear() / linear_backward_dw
# run against the trained param directly.
def krea2_img_in_ref_w_device(
    param: Krea2ImgInRefParam, ctx: DeviceContext
) raises -> Tensor:
    var shp = List[Int]()
    shp.append(param.D)
    shp.append(param.in_ch)
    return Tensor.from_host(krea2_img_in_ref_w_f32(param), shp.copy(), STDtype.F32, ctx)


# ── AdamW step on the single img_in_ref leg (mirrors _lora_adamw on one leg) ──
# d_w: [D, in_ch] grad from the krea2 stack backward's d_img_in_ref (or the
# standalone img_in_ref_backward). `t` is the 1-based step. Same scalars/SR
# stream as the LoRA set.
def krea2_img_in_ref_adamw_step(
    mut param: Krea2ImgInRefParam, d_w: List[Float32], t: Int, lr: Float32,
    ctx: DeviceContext,
    beta1: Float32 = Float32(0.9), beta2: Float32 = Float32(0.999),
    eps: Float32 = Float32(1.0e-8), weight_decay: Float32 = Float32(0.01),
    stochastic_rounding: Bool = True,
) raises:
    _ = ctx
    if t < 1:
        raise Error("krea2_img_in_ref_adamw_step: t must be >= 1")
    var b1p = Float32(1.0)
    var b2p = Float32(1.0)
    for _ in range(t):
        b1p *= beta1
        b2p *= beta2
    var bc1 = Float32(1.0) - b1p
    var bc2 = Float32(1.0) - b2p
    _adamw_host_list_precomputed(
        param.w, d_w, param.m, param.v, lr / bc1, sqrt(bc2),
        Float32(1.0) - lr * weight_decay,
        Float32(1.0) - beta1, beta2, Float32(1.0) - beta2,
        eps, UInt32(t), stochastic_rounding,
    )


# ── SAVE (byte-exact): weight (BF16) + moments (F32) next to the LoRA ckpt ────
def save_krea2_img_in_ref(
    param: Krea2ImgInRefParam, path: String, ctx: DeviceContext
) raises -> Int:
    var names = List[String]()
    var tensors = List[ArcPointer[Tensor]]()
    names.append(String(KREA2_IMG_IN_REF_W_KEY))
    tensors.append(ArcPointer(
        Tensor.from_host_bf16(param.w.copy(), [param.D, param.in_ch], ctx)
    ))
    names.append(String(KREA2_IMG_IN_REF_MA_KEY))
    tensors.append(ArcPointer(
        Tensor.from_host(param.m.copy(), [param.D, param.in_ch], STDtype.F32, ctx)
    ))
    names.append(String(KREA2_IMG_IN_REF_VA_KEY))
    tensors.append(ArcPointer(
        Tensor.from_host(param.v.copy(), [param.D, param.in_ch], STDtype.F32, ctx)
    ))
    save_safetensors(names, tensors, path, ctx)
    return 3


# ── RESUME (byte-exact): read weight + moments back from a save file ─────────
def load_krea2_img_in_ref_resume(
    D: Int, in_ch: Int, path: String, ctx: DeviceContext
) raises -> Krea2ImgInRefParam:
    var st = SafeTensors.open(path)
    if String(KREA2_IMG_IN_REF_W_KEY) not in st.tensors:
        raise Error(
            String("load_krea2_img_in_ref_resume: missing ") + String(KREA2_IMG_IN_REF_W_KEY)
        )
    var w_f32 = _read_f32(st, String(KREA2_IMG_IN_REF_W_KEY), ctx)   # bf16 widened, exact
    var n = D * in_ch
    if len(w_f32) != n:
        raise Error("load_krea2_img_in_ref_resume: weight len mismatch")
    var w = List[BFloat16]()
    for i in range(n):
        w.append(w_f32[i].cast[DType.bfloat16]())               # exact requantize
    var m = List[Float32]()
    var v = List[Float32]()
    if String(KREA2_IMG_IN_REF_MA_KEY) in st.tensors:
        m = _read_f32(st, String(KREA2_IMG_IN_REF_MA_KEY), ctx)
        v = _read_f32(st, String(KREA2_IMG_IN_REF_VA_KEY), ctx)
    else:
        for _ in range(n):
            m.append(Float32(0.0))
            v.append(Float32(0.0))
    return Krea2ImgInRefParam(w^, m^, v^, D, in_ch)

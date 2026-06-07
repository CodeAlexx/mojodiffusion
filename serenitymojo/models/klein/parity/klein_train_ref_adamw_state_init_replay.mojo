# Klein train-ref AdamW state-init replay over the real OneTrainer adapter dump.
#
# This opens `/home/alex/onetrainer-mojo/parity/klein_train_ref_step000_adapters.safetensors`,
# loads all 288 adapter_post_clip LoRA tensors plus their post-clip gradients in
# the exact KleinLoraSet flat order, and calls the real model-level
# `klein_lora_adamw_step` entry used by train_klein_real.
#
# CONTRACT MARKERS:
# real OneTrainer adapter dump
# all 288 train-ref adapters
# adapter_post_clip_grad
# zero-lr state-init support evidence only
# does not execute Klein predict/backward_lora
# does not compare optimizer moment tensors against OneTrainer payloads
# does not prove full Mojo predict/backward/AdamW parity
# does not prove nonzero OneTrainer update parity
# does not prove low-memory offload/checkpoint backward parity

from std.collections import List
from std.builtin.dtype import DType
from std.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.models.klein.klein_stack_lora import (
    DBL_SLOTS,
    SGL_SLOTS,
    KleinLoraGrads,
    KleinLoraSet,
    klein_lora_adamw_step,
    klein_lora_prefixes,
)
from serenitymojo.ops.torch_bf16 import torch_bf16_rne_value
from serenitymojo.training.train_step import LoraAdapter


comptime ADAPTER_DUMP = "/home/alex/onetrainer-mojo/parity/klein_train_ref_step000_adapters.safetensors"
comptime NUM_DOUBLE = 8
comptime NUM_SINGLE = 24
comptime RANK = 16
comptime ALPHA = Float32(16.0)


struct LoraPair(Copyable, Movable):
    var down: List[Float32]
    var up: List[Float32]
    var rank: Int
    var in_f: Int
    var out_f: Int

    def __init__(
        out self, var down: List[Float32], var up: List[Float32],
        rank: Int, in_f: Int, out_f: Int,
    ):
        self.down = down^
        self.up = up^
        self.rank = rank
        self.in_f = in_f
        self.out_f = out_f


struct MomentCheck(Copyable, Movable):
    var max_err: Float32
    var nonzero: Int

    def __init__(out self, max_err: Float32, nonzero: Int):
        self.max_err = max_err
        self.nonzero = nonzero


struct LoadedReplay(Movable):
    var lora: KleinLoraSet
    var grads: KleinLoraGrads
    var total_elems: Int
    var after_equal_elems: Int

    def __init__(
        out self, var lora: KleinLoraSet, var grads: KleinLoraGrads,
        total_elems: Int, after_equal_elems: Int,
    ):
        self.lora = lora^
        self.grads = grads^
        self.total_elems = total_elems
        self.after_equal_elems = after_equal_elems


def _require(ok: Bool, msg: String) raises:
    if not ok:
        raise Error(msg)


def _abs(x: Float32) -> Float32:
    if x < Float32(0.0):
        return -x
    return x


def _zeros(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(Float32(0.0))
    return out^


def _strip_transformer_prefix(name: String) -> String:
    var prefix = String("transformer.")
    if not name.startswith(prefix):
        return name
    var out = String("")
    var ptr = name.unsafe_ptr()
    for i in range(prefix.byte_length(), name.byte_length()):
        out += String(chr(Int(ptr[i])))
    return out^


def _numel(shape: List[Int]) -> Int:
    var n = 1
    for i in range(len(shape)):
        n *= shape[i]
    return n


def _shape_text(shape: List[Int]) -> String:
    var out = String("[")
    for i in range(len(shape)):
        if i > 0:
            out += String(",")
        out += String(shape[i])
    out += String("]")
    return out^


def _assert_shape2(info_shape: List[Int], d0: Int, d1: Int, key: String) raises:
    _require(len(info_shape) == 2, key + String(" expected rank-2 shape, got ") + _shape_text(info_shape))
    _require(
        info_shape[0] == d0 and info_shape[1] == d1,
        key + String(" shape mismatch got ") + _shape_text(info_shape)
        + String(" expected [") + String(d0) + String(",") + String(d1) + String("]"),
    )


def _read_f32(st: SafeTensors, key: String) raises -> List[Float32]:
    var info = st.tensor_info(key)
    _require(info.dtype == STDtype.F32, key + String(" dtype is not F32"))
    _require(info.size % 4 == 0, key + String(" byte size is not divisible by 4"))
    _require(info.size == _numel(info.shape) * 4, key + String(" byte size does not match shape"))
    var bytes = st.tensor_bytes(key)
    var fp = bytes.unsafe_ptr().bitcast[Float32]()
    var out = List[Float32]()
    for i in range(info.size // 4):
        out.append(fp[i])
    return out^


def _read_lora_pair(
    st: SafeTensors, phase: String, bare_prefix: String,
) raises -> LoraPair:
    var down_key = phase + String(".") + bare_prefix + String(".lora_down.weight")
    var up_key = phase + String(".") + bare_prefix + String(".lora_up.weight")
    var down_info = st.tensor_info(down_key)
    var up_info = st.tensor_info(up_key)
    _require(down_info.dtype == STDtype.F32, down_key + String(" dtype is not F32"))
    _require(up_info.dtype == STDtype.F32, up_key + String(" dtype is not F32"))
    _require(len(down_info.shape) == 2, down_key + String(" must be rank-2"))
    _require(len(up_info.shape) == 2, up_key + String(" must be rank-2"))
    var rank = down_info.shape[0]
    var in_f = down_info.shape[1]
    var out_f = up_info.shape[0]
    _assert_shape2(up_info.shape, out_f, rank, up_key)
    var down = _read_f32(st, down_key)
    var up = _read_f32(st, up_key)
    return LoraPair(down^, up^, rank, in_f, out_f)


def _assert_f32_same(actual: List[Float32], expected: List[Float32], label: String) raises:
    _require(len(actual) == len(expected), label + String(" length mismatch"))
    for i in range(len(actual)):
        _require(actual[i] == expected[i], label + String(" mismatch at ") + String(i))


def _snapshot_params(adapters: List[LoraAdapter], use_a: Bool) -> List[List[BFloat16]]:
    var out = List[List[BFloat16]]()
    for i in range(len(adapters)):
        if use_a:
            out.append(adapters[i].a.copy())
        else:
            out.append(adapters[i].b.copy())
    return out^


def _assert_same_bf16(actual: List[BFloat16], expected: List[BFloat16], label: String) raises:
    _require(len(actual) == len(expected), label + String(" length mismatch"))
    for i in range(len(actual)):
        var av = actual[i].cast[DType.float32]()
        var ev = expected[i].cast[DType.float32]()
        _require(av == ev, label + String(" BF16 payload changed at ") + String(i))


def _assert_snapshots_unchanged(
    adapters: List[LoraAdapter],
    before_a: List[List[BFloat16]],
    before_b: List[List[BFloat16]],
    label: String,
) raises:
    _require(len(adapters) == len(before_a), label + String(" A snapshot count mismatch"))
    _require(len(adapters) == len(before_b), label + String(" B snapshot count mismatch"))
    for i in range(len(adapters)):
        _assert_same_bf16(adapters[i].a, before_a[i], label + String(".a[") + String(i) + String("]"))
        _assert_same_bf16(adapters[i].b, before_b[i], label + String(".b[") + String(i) + String("]"))


def _count_changed(
    adapters: List[LoraAdapter],
    before_a: List[List[BFloat16]],
    before_b: List[List[BFloat16]],
) -> Int:
    var changed = 0
    for i in range(len(adapters)):
        for j in range(len(adapters[i].a)):
            if adapters[i].a[j].cast[DType.float32]() != before_a[i][j].cast[DType.float32]():
                changed += 1
        for j in range(len(adapters[i].b)):
            if adapters[i].b[j].cast[DType.float32]() != before_b[i][j].cast[DType.float32]():
                changed += 1
    return changed


def _assert_moments_for_adapter(
    lo: LoraAdapter,
    g_a: List[Float32],
    g_b: List[Float32],
    beta1: Float32,
    beta2: Float32,
    label: String,
) raises -> MomentCheck:
    _require(len(lo.ma) == len(g_a), label + String(" ma length mismatch"))
    _require(len(lo.va) == len(g_a), label + String(" va length mismatch"))
    _require(len(lo.mb) == len(g_b), label + String(" mb length mismatch"))
    _require(len(lo.vb) == len(g_b), label + String(" vb length mismatch"))
    var max_err = Float32(0.0)
    var nonzero = 0
    for i in range(len(g_a)):
        var expected_m = torch_bf16_rne_value((Float32(1.0) - beta1) * g_a[i]).cast[DType.float32]()
        var expected_v = torch_bf16_rne_value((Float32(1.0) - beta2) * g_a[i] * g_a[i]).cast[DType.float32]()
        var err_m = _abs(lo.ma[i] - expected_m)
        var err_v = _abs(lo.va[i] - expected_v)
        if err_m > max_err:
            max_err = err_m
        if err_v > max_err:
            max_err = err_v
        _require(err_m == Float32(0.0), label + String(" ma mismatch at ") + String(i))
        _require(err_v == Float32(0.0), label + String(" va mismatch at ") + String(i))
        if lo.ma[i] != Float32(0.0):
            nonzero += 1
    for i in range(len(g_b)):
        var expected_m = torch_bf16_rne_value((Float32(1.0) - beta1) * g_b[i]).cast[DType.float32]()
        var expected_v = torch_bf16_rne_value((Float32(1.0) - beta2) * g_b[i] * g_b[i]).cast[DType.float32]()
        var err_m = _abs(lo.mb[i] - expected_m)
        var err_v = _abs(lo.vb[i] - expected_v)
        if err_m > max_err:
            max_err = err_m
        if err_v > max_err:
            max_err = err_v
        _require(err_m == Float32(0.0), label + String(" mb mismatch at ") + String(i))
        _require(err_v == Float32(0.0), label + String(" vb mismatch at ") + String(i))
        if lo.mb[i] != Float32(0.0):
            nonzero += 1
    return MomentCheck(max_err, nonzero)


def _assert_all_moments(
    lora: KleinLoraSet, grads: KleinLoraGrads, beta1: Float32, beta2: Float32
) raises -> MomentCheck:
    var max_err = Float32(0.0)
    var nonzero = 0
    for i in range(len(lora.dbl)):
        var r = _assert_moments_for_adapter(
            lora.dbl[i], grads.dbl_d_a[i], grads.dbl_d_b[i], beta1, beta2,
            String("dbl[") + String(i) + String("]"),
        )
        if r.max_err > max_err:
            max_err = r.max_err
        nonzero += r.nonzero
    for i in range(len(lora.sgl)):
        var r = _assert_moments_for_adapter(
            lora.sgl[i], grads.sgl_d_a[i], grads.sgl_d_b[i], beta1, beta2,
            String("sgl[") + String(i) + String("]"),
        )
        if r.max_err > max_err:
            max_err = r.max_err
        nonzero += r.nonzero
    return MomentCheck(max_err, nonzero)


def _load_train_ref_lora(st: SafeTensors) raises -> LoadedReplay:
    var prefixes = klein_lora_prefixes(NUM_DOUBLE, NUM_SINGLE)
    _require(len(prefixes) == NUM_DOUBLE * DBL_SLOTS + NUM_SINGLE * SGL_SLOTS, String("prefix count mismatch"))
    var dbl = List[LoraAdapter]()
    var sgl = List[LoraAdapter]()
    var dbl_d_a = List[List[Float32]]()
    var dbl_d_b = List[List[Float32]]()
    var sgl_d_a = List[List[Float32]]()
    var sgl_d_b = List[List[Float32]]()
    var total_elems = 0
    var after_equal_elems = 0

    for i in range(len(prefixes)):
        var bare = _strip_transformer_prefix(prefixes[i])
        var post = _read_lora_pair(st, String("adapter_post_clip"), bare)
        var after = _read_lora_pair(st, String("adapter_after"), bare)
        _assert_f32_same(post.down, after.down, bare + String(".down adapter_after != adapter_post_clip"))
        _assert_f32_same(post.up, after.up, bare + String(".up adapter_after != adapter_post_clip"))
        after_equal_elems += len(post.down) + len(post.up)

        var grad = _read_lora_pair(st, String("adapter_post_clip_grad"), bare)
        _require(post.rank == RANK, bare + String(" rank mismatch"))
        _require(
            grad.rank == post.rank and grad.in_f == post.in_f and grad.out_f == post.out_f,
            bare + String(" grad shape mismatch"),
        )
        var post_down_len = len(post.down)
        var post_up_len = len(post.up)
        var zeros_a = _zeros(post_down_len)
        var zeros_b = _zeros(post_up_len)
        var adapter = LoraAdapter(
            post.down.copy(), post.up.copy(), post.rank, post.in_f, post.out_f,
            ALPHA / Float32(post.rank),
            zeros_a.copy(), zeros_a^, zeros_b.copy(), zeros_b^,
        )
        if i < NUM_DOUBLE * DBL_SLOTS:
            dbl.append(adapter^)
            dbl_d_a.append(grad.down.copy())
            dbl_d_b.append(grad.up.copy())
        else:
            sgl.append(adapter^)
            sgl_d_a.append(grad.down.copy())
            sgl_d_b.append(grad.up.copy())
        total_elems += post_down_len + post_up_len
    var lora = KleinLoraSet(dbl^, sgl^, NUM_DOUBLE, NUM_SINGLE, RANK)
    var grads = KleinLoraGrads(
        dbl_d_a^, dbl_d_b^, sgl_d_a^, sgl_d_b^,
        _zeros(0), _zeros(0), _zeros(0), _zeros(0), _zeros(0),
        _zeros(0), _zeros(0), _zeros(0), _zeros(0), _zeros(0),
    )
    return LoadedReplay(lora^, grads^, total_elems, after_equal_elems)


def main() raises:
    var ctx = DeviceContext()
    var st = SafeTensors.open(String(ADAPTER_DUMP))
    _require(st.count() == 1728, String("expected 1728 Klein adapter tensors"))
    var loaded = _load_train_ref_lora(st)
    var total_elems = loaded.total_elems
    var after_equal_elems = loaded.after_equal_elems
    _require(len(loaded.lora.dbl) == NUM_DOUBLE * DBL_SLOTS, String("double adapter count mismatch"))
    _require(len(loaded.lora.sgl) == NUM_SINGLE * SGL_SLOTS, String("single adapter count mismatch"))
    _require(total_elems == 43515904, String("train-ref LoRA numel mismatch"))
    _require(after_equal_elems == total_elems, String("adapter_after equality count mismatch"))

    var beta1 = Float32(0.9)
    var beta2 = Float32(0.999)
    var eps = Float32(1.0e-8)
    var weight_decay = Float32(0.01)

    var dbl_a_before = _snapshot_params(loaded.lora.dbl, True)
    var dbl_b_before = _snapshot_params(loaded.lora.dbl, False)
    var sgl_a_before = _snapshot_params(loaded.lora.sgl, True)
    var sgl_b_before = _snapshot_params(loaded.lora.sgl, False)
    klein_lora_adamw_step(loaded.lora, loaded.grads, 1, Float32(0.0), ctx, beta1, beta2, eps, weight_decay)
    _assert_snapshots_unchanged(loaded.lora.dbl, dbl_a_before, dbl_b_before, String("dbl zero-lr"))
    _assert_snapshots_unchanged(loaded.lora.sgl, sgl_a_before, sgl_b_before, String("sgl zero-lr"))
    var moment = _assert_all_moments(loaded.lora, loaded.grads, beta1, beta2)
    _require(moment.nonzero > 0, String("all optimizer moments are zero after nonzero gradients"))

    var dbl_a_step1 = _snapshot_params(loaded.lora.dbl, True)
    var dbl_b_step1 = _snapshot_params(loaded.lora.dbl, False)
    var sgl_a_step1 = _snapshot_params(loaded.lora.sgl, True)
    var sgl_b_step1 = _snapshot_params(loaded.lora.sgl, False)
    klein_lora_adamw_step(loaded.lora, loaded.grads, 2, Float32(2.9999999999999997e-06), ctx, beta1, beta2, eps, weight_decay)
    var changed = _count_changed(loaded.lora.dbl, dbl_a_step1, dbl_b_step1)
    changed += _count_changed(loaded.lora.sgl, sgl_a_step1, sgl_b_step1)
    _require(changed > 0, String("synthetic positive-lr follow-up changed no BF16 adapter values"))

    print(
        "[klein-train-ref-adamw-state-init] PASS",
        " adapters=", len(loaded.lora.dbl) + len(loaded.lora.sgl),
        " elems=", total_elems,
        " after_equal=", after_equal_elems,
        " max_moment_err=", moment.max_err,
        " nonzero_moments=", moment.nonzero,
        " synthetic_positive_lr_changed=", changed,
    )

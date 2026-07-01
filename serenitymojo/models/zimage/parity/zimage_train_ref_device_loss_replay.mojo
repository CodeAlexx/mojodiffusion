# ZImage OneTrainer train-ref device loss-root replay.
#
# Opens the real OneTrainer ZImage step-0 dump, patchifies the dumped BF16
# predicted/target flow tensors into the product [rows, OUT_CH] layout, and
# replays the v5 device-native flow MSE root. This validates the device loss and
# d_patches seed that feeds ZImage backward; it does not execute the transformer
# forward/backward, adapter gradients, optimizer, save/resume, or product loop.

from std.builtin.dtype import DType
from std.gpu.host import DeviceContext
from std.math import sqrt
from std.pathlib import Path

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.tensor import Tensor
from serenitymojo.models.zimage.zimage_stack_lora import (
    zimage_step_io_init,
    zimage_step_io_write_flow_mse_d_patches,
)


comptime META_JSON = "/home/alex/serenity-trainer/parity/zimage_train_ref_meta.json"
comptime STEP_DUMP = "/home/alex/serenity-trainer/parity/zimage_train_ref_step000.safetensors"
comptime BATCH = 2
comptime CHANNELS = 16
comptime HEIGHT = 64
comptime WIDTH = 64
comptime PATCH = 2
comptime OUT_CH = CHANNELS * PATCH * PATCH
comptime ROWS_PER_SAMPLE = (HEIGHT // PATCH) * (WIDTH // PATCH)
comptime REAL_ROWS = BATCH * ROWS_PER_SAMPLE
comptime TOTAL_NUMEL = REAL_ROWS * OUT_CH
comptime EXPECTED_LOSS = Float32(0.40854018926620483)
comptime LOSS_TOL = Float32(1.0e-5)
comptime GRAD_TOL = Float32(1.0e-8)


@fieldwise_init
struct GradStats(Copyable, Movable):
    var max_abs: Float32
    var l2_sumsq: Float64
    var nonzero_error: Int
    var nonzero_grad: Int


def _require(ok: Bool, msg: String) raises:
    if not ok:
        raise Error(msg)


def _abs(x: Float32) -> Float32:
    if x < Float32(0.0):
        return -x
    return x


def _shape4(a: Int, b: Int, c: Int, d: Int) -> List[Int]:
    var out = List[Int]()
    out.append(a)
    out.append(b)
    out.append(c)
    out.append(d)
    return out^


def _shape1(a: Int) -> List[Int]:
    var out = List[Int]()
    out.append(a)
    return out^


def _same_shape(a: List[Int], b: List[Int]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


def _numel(shape: List[Int]) -> Int:
    var n = 1
    for i in range(len(shape)):
        n *= shape[i]
    return n


def _require_tensor(
    st: SafeTensors, key: String, dtype: STDtype, expected_shape: List[Int]
) raises:
    _require(key in st.tensors, String("missing ZImage OT tensor ") + key)
    var info = st.tensor_info(key)
    _require(
        info.dtype == dtype,
        String("dtype mismatch for ")
        + key
        + String(" got=")
        + info.dtype.name()
        + String(" expected=")
        + dtype.name(),
    )
    _require(_same_shape(info.shape, expected_shape), String("shape mismatch for ") + key)
    _require(info.size == _numel(expected_shape) * dtype.byte_size(), String("byte-size mismatch for ") + key)


def _read_f32_value(st: SafeTensors, key: String, index: Int) raises -> Float32:
    var info = st.tensor_info(key)
    _require(info.dtype == STDtype.F32, String("expected F32 payload for ") + key)
    _require(index >= 0 and index < info.size // 4, String("F32 index out of bounds for ") + key)
    var bytes = st.tensor_bytes(key)
    var fp = bytes.unsafe_ptr().bitcast[Float32]()
    return fp[index]


def _read_bf16_value(st: SafeTensors, key: String, index: Int) raises -> Float32:
    var info = st.tensor_info(key)
    _require(info.dtype == STDtype.BF16, String("expected BF16 payload for ") + key)
    _require(index >= 0 and index < info.size // 2, String("BF16 index out of bounds for ") + key)
    var bytes = st.tensor_bytes(key)
    var bp = bytes.unsafe_ptr().bitcast[BFloat16]()
    return bp[index].cast[DType.float32]()


def _patchified_value(
    st: SafeTensors, key: String, sample: Int, row: Int, col: Int
) raises -> Float32:
    var ih = row // (WIDTH // PATCH)
    var iw = row - ih * (WIDTH // PATCH)
    var phpw = col // CHANNELS
    var c = col - phpw * CHANNELS
    var ph = phpw // PATCH
    var pw = phpw - ph * PATCH
    var h = ih * PATCH + ph
    var w = iw * PATCH + pw
    var idx = (
        sample * CHANNELS * HEIGHT * WIDTH
        + c * HEIGHT * WIDTH
        + h * WIDTH
        + w
    )
    return _read_bf16_value(st, key, idx)


def _patchify_raw_and_neg_target(st: SafeTensors) raises -> List[List[Float32]]:
    var raw = List[Float32]()
    var neg_target = List[Float32]()
    for b in range(BATCH):
        for r in range(ROWS_PER_SAMPLE):
            for c in range(OUT_CH):
                raw.append(-_patchified_value(st, String("predicted_flow"), b, r, c))
                neg_target.append(-_patchified_value(st, String("flow_target"), b, r, c))
    var out = List[List[Float32]]()
    out.append(raw^)
    out.append(neg_target^)
    return out^


def _zeros(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(Float32(0.0))
    return out^


def _host_loss_and_grad_stats(
    raw: List[Float32], neg_target: List[Float32], got_grad: List[Float32]
) raises -> List[Float64]:
    _require(len(raw) == TOTAL_NUMEL, String("raw patch numel mismatch"))
    _require(len(neg_target) == TOTAL_NUMEL, String("target patch numel mismatch"))
    _require(len(got_grad) == TOTAL_NUMEL, String("device d_patches numel mismatch"))
    var loss_sum = Float64(0.0)
    var stats = GradStats(Float32(0.0), Float64(0.0), 0, 0)
    var grad_scale = Float32(2.0) / Float32(TOTAL_NUMEL)
    for i in range(TOTAL_NUMEL):
        var diff = raw[i] - neg_target[i]
        loss_sum += Float64(diff) * Float64(diff)
        var expected_grad = diff * grad_scale
        if expected_grad != Float32(0.0):
            stats.nonzero_grad += 1
        var err = got_grad[i] - expected_grad
        if err != Float32(0.0):
            stats.nonzero_error += 1
        var ae = _abs(err)
        if ae > stats.max_abs:
            stats.max_abs = ae
        var e64 = Float64(err)
        stats.l2_sumsq += e64 * e64
    var out = List[Float64]()
    out.append(loss_sum / Float64(TOTAL_NUMEL))
    out.append(Float64(stats.max_abs))
    out.append(stats.l2_sumsq)
    out.append(Float64(stats.nonzero_error))
    out.append(Float64(stats.nonzero_grad))
    return out^


def _check_meta_json() raises:
    var meta = Path(String(META_JSON)).read_text()
    _require(meta.byte_length() > 0, String("empty ZImage OneTrainer metadata JSON"))
    _require(meta.find(String("\"producer\": \"scripts/zimage_dump_train_ref.py\"")) >= 0, String("metadata producer mismatch"))
    _require(meta.find(String("\"model_type\": \"Z_IMAGE\"")) >= 0, String("metadata model_type mismatch"))
    _require(meta.find(String("\"loss_for_backward\": 0.40854018926620483")) >= 0, String("metadata loss mismatch"))
    _require(meta.find(String(STEP_DUMP)) >= 0, String("metadata step path mismatch"))


def main() raises:
    _check_meta_json()
    var ctx = DeviceContext()
    var st = SafeTensors.open(String(STEP_DUMP))
    _require_tensor(st, String("predicted_flow"), STDtype.BF16, _shape4(BATCH, CHANNELS, HEIGHT, WIDTH))
    _require_tensor(st, String("flow_target"), STDtype.BF16, _shape4(BATCH, CHANNELS, HEIGHT, WIDTH))
    _require_tensor(st, String("batch.loss_weight"), STDtype.F32, _shape1(BATCH))
    _require(_read_f32_value(st, String("batch.loss_weight"), 0) == Float32(1.0), String("sample 0 loss weight is not 1.0"))
    _require(_read_f32_value(st, String("batch.loss_weight"), 1) == Float32(1.0), String("sample 1 loss weight is not 1.0"))

    var patched = _patchify_raw_and_neg_target(st)
    var raw_t = Tensor.from_host(patched[0].copy(), [REAL_ROWS, OUT_CH], STDtype.F32, ctx)
    var neg_tgt_t = Tensor.from_host(patched[1].copy(), [REAL_ROWS, OUT_CH], STDtype.F32, ctx)
    var final_bias = Tensor.from_host(_zeros(OUT_CH), [OUT_CH], STDtype.F32, ctx)
    var io = zimage_step_io_init(
        REAL_ROWS, 0, 1, OUT_CH, 0, 0, 1, 2, final_bias, ctx
    )
    var dev_loss = zimage_step_io_write_flow_mse_d_patches(
        io, raw_t, neg_tgt_t, ctx
    )
    _require(dev_loss.full_tensor_readback_count == 0, String("device loss root performed full tensor readback"))
    _require(dev_loss.scalar_readback_count == 1, String("device loss root scalar readback count changed"))
    _require(dev_loss.sync_count == 1, String("device loss root sync count changed"))

    var got_grad = io.d_patches[].to_host(ctx)
    var host_stats = _host_loss_and_grad_stats(patched[0], patched[1], got_grad)
    var host_loss = Float32(host_stats[0])
    var grad_max_abs = Float32(host_stats[1])
    var grad_l2 = sqrt(host_stats[2])
    var nonzero_error = Int(host_stats[3])
    var nonzero_grad = Int(host_stats[4])
    _require(_abs(host_loss - EXPECTED_LOSS) <= LOSS_TOL, String("host patchified loss mismatch"))
    _require(_abs(dev_loss.loss - EXPECTED_LOSS) <= LOSS_TOL, String("device loss mismatch"))
    _require(grad_max_abs <= GRAD_TOL, String("device d_patches mismatch"))

    print(
        "[zimage-train-ref-device-loss] PASS loss=",
        dev_loss.loss,
        " host_loss=",
        host_loss,
        " rows=",
        REAL_ROWS,
        " out_ch=",
        OUT_CH,
        " numel=",
        TOTAL_NUMEL,
        " nonzero_grad=",
        nonzero_grad,
        " nonzero_error=",
        nonzero_error,
        " grad_max_abs=",
        grad_max_abs,
        " grad_l2=",
        grad_l2,
        " backend=",
        dev_loss.backend,
        " full_readbacks=",
        dev_loss.full_tensor_readback_count,
        " scalar_readbacks=",
        dev_loss.scalar_readback_count,
        " syncs=",
        dev_loss.sync_count,
    )
    print(
        "[zimage-train-ref-device-loss] scope=real OneTrainer step0 dump through v5 device flow-MSE d_patches root; not transformer forward/backward parity"
    )

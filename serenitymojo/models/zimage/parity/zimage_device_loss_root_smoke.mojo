# zimage_device_loss_root_smoke.mojo — v5 device MSE root for ZImage step I/O.
#
# Run:
#   pixi run mojo run -I . serenitymojo/models/zimage/parity/zimage_device_loss_root_smoke.mojo

from std.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor
from serenitymojo.models.zimage.zimage_stack_lora import (
    zimage_step_io_init,
    zimage_step_io_write_mse_d_patches,
    zimage_step_io_write_flow_mse_d_patches,
)


def _fabs(x: Float32) -> Float32:
    if x < Float32(0.0):
        return -x
    return x


def _close(a: Float32, b: Float32, tol: Float32) -> Bool:
    return _fabs(a - b) <= tol


def _check(cond: Bool, msg: String) raises:
    if not cond:
        raise Error(String("zimage_device_loss_root_smoke FAILED: ") + msg)


def _constant_values(n: Int, value: Float32) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(value)
    return out^


def _prod_raw_value(row: Int, col: Int) -> Float32:
    var a = Float32((row % 17) - 8) * Float32(0.03125)
    var b = Float32((col % 11) - 5) * Float32(0.015625)
    return a + b


def _prod_target_value(row: Int, col: Int) -> Float32:
    var a = Float32((row % 13) - 6) * Float32(0.02734375)
    var b = Float32((col % 7) - 3) * Float32(0.01953125)
    return a - b


def _production_bucket_patches(
    n_img: Int, n_txt: Int, out_ch: Int, real_rows: Int,
) -> List[Float32]:
    var out = List[Float32]()
    var total_rows = n_img + n_txt
    for r in range(total_rows):
        for c in range(out_ch):
            if r < real_rows:
                out.append(_prod_raw_value(r, c))
            else:
                # Padded image rows and caption rows must not affect loss or
                # receive d_patches when the target only covers real rows.
                out.append(Float32(99.0))
    return out^


def _production_bucket_neg_targets(real_rows: Int, out_ch: Int) -> List[Float32]:
    var out = List[Float32]()
    for r in range(real_rows):
        for c in range(out_ch):
            out.append(-_prod_target_value(r, c))
    return out^


def _host_flow_loss_ref(real_rows: Int, out_ch: Int) -> Float32:
    var ss = Float64(0.0)
    for r in range(real_rows):
        for c in range(out_ch):
            var d = Float64(_prod_raw_value(r, c) + _prod_target_value(r, c))
            ss += d * d
    return Float32(ss / Float64(real_rows * out_ch))


def _host_flow_grad_ref(row: Int, col: Int, real_rows: Int, out_ch: Int) -> Float32:
    if row >= real_rows:
        return Float32(0.0)
    var diff = _prod_raw_value(row, col) + _prod_target_value(row, col)
    var grad_scale = Float32(2.0) / Float32(real_rows * out_ch)
    return diff * grad_scale


def main() raises:
    var ctx = DeviceContext()
    var n_img = 2
    var n_txt = 1
    var out_ch = 3
    var d_model = 4
    var heads = 1
    var dh = 2
    var final_bias = Tensor.from_host(
        [Float32(0.0), Float32(0.0), Float32(0.0)],
        [out_ch],
        STDtype.F32,
        ctx,
    )
    var io = zimage_step_io_init(
        n_img, n_txt, d_model, out_ch,
        0, 0, heads, dh, final_bias, ctx,
    )

    # First two rows are image predictions; the final row is cap output and must
    # not contribute to loss or d_patches.
    var patches = Tensor.from_host(
        [
            Float32(1.0), Float32(2.0), Float32(3.0),
            Float32(-1.0), Float32(0.5), Float32(4.0),
            Float32(9.0), Float32(8.0), Float32(7.0),
        ],
        [n_img + n_txt, out_ch],
        STDtype.F32,
        ctx,
    )
    var target = Tensor.from_host(
        [
            Float32(0.0), Float32(1.0), Float32(1.0),
            Float32(-2.0), Float32(0.5), Float32(1.0),
        ],
        [n_img, out_ch],
        STDtype.F32,
        ctx,
    )

    var res = zimage_step_io_write_mse_d_patches(io, patches, target, ctx)
    _check(_close(res.loss, Float32(16.0 / 6.0), Float32(1.0e-6)), "loss")
    _check(res.scalar_readback_count == 1, "scalar readback count")
    _check(res.full_tensor_readback_count == 0, "no full tensor readback")
    _check(res.sync_count == 1, "sync count should be loss scalar only")
    _check(res.backend == String("device-mse-block-reduce-into-scratch"), "backend label")

    var got = io.d_patches[].to_host(ctx)
    var want: List[Float32] = [
        Float32(1.0 / 3.0), Float32(1.0 / 3.0), Float32(2.0 / 3.0),
        Float32(1.0 / 3.0), Float32(0.0), Float32(1.0),
        Float32(0.0), Float32(0.0), Float32(0.0),
    ]
    for i in range(len(want)):
        if not _close(got[i], want[i], Float32(1.0e-6)):
            raise Error(
                String("zimage_device_loss_root_smoke FAILED: d_patches[")
                + String(i)
                + String("] got=")
                + String(got[i])
                + String(" want=")
                + String(want[i])
            )

    var final_bias2 = Tensor.from_host(
        [Float32(0.0), Float32(0.0), Float32(0.0)],
        [out_ch],
        STDtype.F32,
        ctx,
    )
    var io2 = zimage_step_io_init(
        3, n_txt, d_model, out_ch,
        0, 0, heads, dh, final_bias2, ctx,
    )
    var raw_patches = Tensor.from_host(
        [
            Float32(1.0), Float32(-2.0), Float32(0.0),
            Float32(4.0), Float32(1.0), Float32(-3.0),
            Float32(99.0), Float32(88.0), Float32(77.0),
            Float32(9.0), Float32(8.0), Float32(7.0),
        ],
        [4, out_ch],
        STDtype.F32,
        ctx,
    )
    # Host reference is MSE(-raw, target). Passing -target into the device root
    # must produce d_raw = 2 / N * (raw + target) over real rows only. The
    # padded image rows stay at the StepIO zero root.
    var neg_target = Tensor.from_host(
        [
            Float32(-2.0), Float32(1.0), Float32(-1.0),
            Float32(0.0), Float32(-2.0), Float32(-1.0),
        ],
        [2, out_ch],
        STDtype.F32,
        ctx,
    )
    var flow = zimage_step_io_write_flow_mse_d_patches(
        io2, raw_patches, neg_target, ctx
    )
    _check(_close(flow.loss, Float32(48.0 / 6.0), Float32(1.0e-6)), "flow loss")
    _check(flow.scalar_readback_count == 1, "flow scalar readback count")
    _check(flow.full_tensor_readback_count == 0, "flow no full tensor readback")
    _check(flow.sync_count == 1, "flow sync count should be loss scalar only")

    var got_flow = io2.d_patches[].to_host(ctx)
    var want_flow: List[Float32] = [
        Float32(1.0), Float32(-1.0), Float32(1.0 / 3.0),
        Float32(4.0 / 3.0), Float32(1.0), Float32(-2.0 / 3.0),
        Float32(0.0), Float32(0.0), Float32(0.0),
        Float32(0.0), Float32(0.0), Float32(0.0),
    ]
    for i in range(len(want_flow)):
        if not _close(got_flow[i], want_flow[i], Float32(1.0e-6)):
            raise Error(
                String("zimage_device_loss_root_smoke FAILED: flow d_patches[")
                + String(i)
                + String("] got=")
                + String(got_flow[i])
                + String(" want=")
                + String(want_flow[i])
            )

    # Real 512px production bucket: latent 56x72 gives 1008 image tokens, padded
    # to 1024, with the first caption bucket at 224 and patchified OUT_CH=64.
    # This proves the v5 device flow loss handles production row counts without
    # prediction readback and keeps padded image/caption rows at zero.
    var prod_n_img = 1024
    var prod_real_rows = 1008
    var prod_n_txt = 224
    var prod_out_ch = 64
    var prod_d_model = 4
    var prod_heads = 1
    var prod_dh = 2
    var prod_final_bias = Tensor.from_host(
        _constant_values(prod_out_ch, Float32(0.0)),
        [prod_out_ch],
        STDtype.F32,
        ctx,
    )
    var prod_io = zimage_step_io_init(
        prod_n_img, prod_n_txt, prod_d_model, prod_out_ch,
        0, 0, prod_heads, prod_dh, prod_final_bias, ctx,
    )
    var prod_patches = Tensor.from_host(
        _production_bucket_patches(
            prod_n_img, prod_n_txt, prod_out_ch, prod_real_rows
        ),
        [prod_n_img + prod_n_txt, prod_out_ch],
        STDtype.F32,
        ctx,
    )
    var prod_neg_target = Tensor.from_host(
        _production_bucket_neg_targets(prod_real_rows, prod_out_ch),
        [prod_real_rows, prod_out_ch],
        STDtype.F32,
        ctx,
    )
    var prod_flow = zimage_step_io_write_flow_mse_d_patches(
        prod_io, prod_patches, prod_neg_target, ctx
    )
    _check(
        _close(prod_flow.loss, _host_flow_loss_ref(prod_real_rows, prod_out_ch), Float32(1.0e-5)),
        "production bucket flow loss vs host reference",
    )
    _check(prod_flow.scalar_readback_count == 1, "production scalar readback count")
    _check(prod_flow.full_tensor_readback_count == 0, "production no full tensor readback")
    _check(prod_flow.sync_count == 1, "production sync count should be loss scalar only")
    _check(
        prod_flow.backend == String("device-mse-block-reduce-into-scratch"),
        "production backend label",
    )

    var prod_got = prod_io.d_patches[].to_host(ctx)
    # The parity smoke reads d_patches back for inspection, so compare every
    # real, padded-image, and caption row instead of sampling boundary probes.
    for row in range(prod_n_img + prod_n_txt):
        for c in range(prod_out_ch):
            var got_val = prod_got[row * prod_out_ch + c]
            var expected = _host_flow_grad_ref(row, c, prod_real_rows, prod_out_ch)
            if not _close(got_val, expected, Float32(1.0e-8)):
                raise Error(
                    String("zimage_device_loss_root_smoke FAILED: production d_patches row=")
                    + String(row)
                    + String(" col=")
                    + String(c)
                    + String(" got=")
                    + String(got_val)
                    + String(" want=")
                    + String(expected)
                )

    print("PASS: ZImage v5 device loss writes d_patches root without full tensor readback")

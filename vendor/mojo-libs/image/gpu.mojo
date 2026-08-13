# image/gpu.mojo — GPU-accelerated image ops (MAX), parity-checked vs the CPU
# reference in color.mojo / filter.mojo (which are themselves PIL-verified).
#
# The CPU ops are the correctness oracle; these kernels must match them. Per-pixel
# color ops are bit-exact (same integer / F64 math). Guarded by has_accelerator();
# callers should fall back to the CPU op when no GPU is present.

from std.sys import has_accelerator
from std.math import ceildiv, floor
from std.gpu import global_idx
from max.gpu.host import DeviceContext
from std.memory import UnsafePointer

from image.buffer import Image

comptime DevPtr = UnsafePointer[UInt8, MutAnyOrigin]
comptime BLOCK = 256


@always_inline
def _clamp255(v: Int) -> Int:
    if v < 0:
        return 0
    if v > 255:
        return 255
    return v


# ─── per-pixel kernels (unified signature so one launcher serves all) ─────────
# op selects the operation; fparam/iparam carry op-specific params.
#   op 0 = invert, 1 = brightness(+iparam), 2 = grayscale, 3 = contrast(*fparam@128)
def _k_pixel(
    inp: DevPtr,
    dst: DevPtr,
    npix_w: Int32,
    channels_w: Int32,
    op_w: Int32,
    fparam: Float64,
    iparam_w: Int32,
):
    var npix = Int(npix_w)
    var channels = Int(channels_w)
    var op = Int(op_w)
    var iparam = Int(iparam_w)
    var tid = Int(global_idx.x)
    if tid >= npix:
        return
    var base = tid * channels
    var ncolor = 3 if channels == 4 else channels

    if op == 2:  # grayscale (luma to color channels, alpha passthrough)
        if channels == 1:
            dst[base] = inp[base]
        else:
            var r = Int(inp[base])
            var g = Int(inp[base + 1])
            var b = Int(inp[base + 2])
            var l = UInt8((r * 19595 + g * 38470 + b * 7471 + 0x8000) >> 16)
            dst[base] = l
            dst[base + 1] = l
            dst[base + 2] = l
    else:
        for c in range(ncolor):
            var v = Int(inp[base + c])
            if op == 0:        # invert
                dst[base + c] = UInt8(255 - v)
            elif op == 1:      # brightness
                dst[base + c] = UInt8(_clamp255(v + iparam))
            else:              # contrast (op == 3), F64 to match CPU exactly
                var nv = (Float64(v) - 128.0) * fparam + 128.0
                var r = Int(floor(nv + 0.5))
                dst[base + c] = UInt8(_clamp255(r))

    if channels == 4:
        dst[base + 3] = inp[base + 3]


def _run_pixel(img: Image, op: Int, fparam: Float64, iparam: Int) raises -> Image:
    var ctx = DeviceContext()
    var n = img.byte_len()
    var npix = img.width * img.height

    var hin = ctx.enqueue_create_host_buffer[DType.uint8](n)
    for i in range(n):
        hin[i] = img.data[i]
    var din = ctx.enqueue_create_buffer[DType.uint8](n)
    var dout = ctx.enqueue_create_buffer[DType.uint8](n)
    ctx.enqueue_copy(din, hin)

    ctx.enqueue_function[_k_pixel](
        din.unsafe_ptr(), dout.unsafe_ptr(), Int32(npix), Int32(img.channels), Int32(op), fparam, Int32(iparam),
        grid_dim=ceildiv(npix, BLOCK), block_dim=BLOCK,
    )

    var hout = ctx.enqueue_create_host_buffer[DType.uint8](n)
    ctx.enqueue_copy(hout, dout)
    ctx.synchronize()

    var res = Image.new(img.width, img.height, img.channels)
    for i in range(n):
        res.data[i] = hout[i]
    return res^


def gpu_invert(img: Image) raises -> Image:
    return _run_pixel(img, 0, 0.0, 0)

def gpu_brightness(img: Image, delta: Int) raises -> Image:
    return _run_pixel(img, 1, 0.0, delta)

def gpu_grayscale(img: Image) raises -> Image:
    return _run_pixel(img, 2, 0.0, 0)

def gpu_contrast(img: Image, factor: Float64) raises -> Image:
    return _run_pixel(img, 3, factor, 0)


# ─── convolution (covers box/sharpen/sobel/gaussian) ──────────────────────────
comptime FloatPtr = UnsafePointer[Float64, MutAnyOrigin]


@always_inline
def _clampi(v: Int, hi: Int) -> Int:
    if v < 0:
        return 0
    if v > hi:
        return hi
    return v


def _k_convolve(
    inp: DevPtr,
    dst: DevPtr,
    kbuf: FloatPtr,
    w_w: Int32,
    h_w: Int32,
    channels_w: Int32,
    kw_w: Int32,
    kh_w: Int32,
    div: Float64,
    bias: Float64,
):
    var w = Int(w_w)
    var h = Int(h_w)
    var channels = Int(channels_w)
    var kw = Int(kw_w)
    var kh = Int(kh_w)
    var tid = Int(global_idx.x)
    if tid >= w * h:
        return
    var x = tid % w
    var y = tid // w
    var base = tid * channels
    var ncolor = 3 if channels == 4 else channels
    var ax = kw // 2
    var ay = kh // 2
    for c in range(ncolor):
        var acc = 0.0
        for ky in range(kh):
            for kx in range(kw):
                var k = kbuf[ky * kw + kx]
                if k != 0.0:
                    var sx = _clampi(x + kx - ax, w - 1)
                    var sy = _clampi(y + ky - ay, h - 1)
                    acc += k * Float64(Int(inp[(sy * w + sx) * channels + c]))
        var r = Int(floor(acc / div + bias + 0.5))
        dst[base + c] = UInt8(_clamp255(r))
    if channels == 4:
        dst[base + 3] = inp[base + 3]


def gpu_convolve(
    img: Image, kernel: List[Float64], kw: Int, kh: Int, divisor: Float64, bias: Float64
) raises -> Image:
    if kw <= 0 or kh <= 0 or len(kernel) != kw * kh:
        raise Error("gpu_convolve: kernel size mismatch")
    var div = divisor
    if div == 0.0:
        div = 1.0
    var ctx = DeviceContext()
    var n = img.byte_len()
    var npix = img.width * img.height
    var nk = kw * kh

    var hin = ctx.enqueue_create_host_buffer[DType.uint8](n)
    for i in range(n):
        hin[i] = img.data[i]
    var din = ctx.enqueue_create_buffer[DType.uint8](n)
    var dout = ctx.enqueue_create_buffer[DType.uint8](n)
    ctx.enqueue_copy(din, hin)

    var hk = ctx.enqueue_create_host_buffer[DType.float64](nk)
    for i in range(nk):
        hk[i] = kernel[i]
    var dk = ctx.enqueue_create_buffer[DType.float64](nk)
    ctx.enqueue_copy(dk, hk)

    ctx.enqueue_function[_k_convolve](
        din.unsafe_ptr(), dout.unsafe_ptr(), dk.unsafe_ptr(),
        Int32(img.width), Int32(img.height), Int32(img.channels), Int32(kw), Int32(kh), div, bias,
        grid_dim=ceildiv(npix, BLOCK), block_dim=BLOCK,
    )

    var hout = ctx.enqueue_create_host_buffer[DType.uint8](n)
    ctx.enqueue_copy(hout, dout)
    ctx.synchronize()

    var res = Image.new(img.width, img.height, img.channels)
    for i in range(n):
        res.data[i] = hout[i]
    return res^

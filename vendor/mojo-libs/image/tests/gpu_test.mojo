from image.buffer import Image
from image.color import invert, brightness, grayscale, contrast
from image.filter import convolve
from image.gpu import gpu_invert, gpu_brightness, gpu_grayscale, gpu_contrast, gpu_convolve
from std.sys import has_accelerator

def check(mut p: Int, mut f: Int, cond: Bool, name: String):
    if cond: p += 1
    else:
        f += 1
        print("  FAIL:", name)

def _identical(a: Image, b: Image) raises -> Bool:
    if a.byte_len() != b.byte_len(): return False
    for i in range(a.byte_len()):
        if a.data[i] != b.data[i]: return False
    return True

def _fixture() raises -> Image:
    var im = Image.new(40, 24, 4)
    for y in range(24):
        for x in range(40):
            im.set(x, y, 0, UInt8((x * 6) & 0xFF))
            im.set(x, y, 1, UInt8((y * 10) & 0xFF))
            im.set(x, y, 2, UInt8(((x + y) * 3) & 0xFF))
            im.set(x, y, 3, UInt8((x * y) & 0xFF))
    return im^

def main() raises:
    var p = 0
    var f = 0
    if not has_accelerator():
        print("NO GPU — skipping"); return
    var im = _fixture()

    check(p, f, _identical(gpu_invert(im), invert(im)), "gpu_invert bit-exact")
    check(p, f, _identical(gpu_brightness(im, 50), brightness(im, 50)), "gpu_brightness(+50) bit-exact")
    check(p, f, _identical(gpu_brightness(im, -30), brightness(im, -30)), "gpu_brightness(-30) bit-exact")
    check(p, f, _identical(gpu_grayscale(im), grayscale(im)), "gpu_grayscale bit-exact")
    check(p, f, _identical(gpu_contrast(im, 1.5), contrast(im, 1.5)), "gpu_contrast(1.5) bit-exact")
    check(p, f, _identical(gpu_contrast(im, 0.6), contrast(im, 0.6)), "gpu_contrast(0.6) bit-exact")

    # convolution: 3x3 box blur (div 9)
    var box = List[Float64]()
    for _ in range(9): box.append(1.0)
    check(p, f, _identical(gpu_convolve(im, box, 3, 3, 9.0, 0.0), convolve(im, box, 3, 3, 9.0, 0.0)), "gpu_convolve box3x3 bit-exact")

    # sharpen kernel
    var sh = List[Float64]()
    for v in [Float64(0), -1, 0, -1, 5, -1, 0, -1, 0]: sh.append(v)
    check(p, f, _identical(gpu_convolve(im, sh, 3, 3, 1.0, 0.0), convolve(im, sh, 3, 3, 1.0, 0.0)), "gpu_convolve sharpen bit-exact")

    # 5x5 gaussian-ish + bias
    var g5 = List[Float64]()
    for v in [Float64(1),4,6,4,1, 4,16,24,16,4, 6,24,36,24,6, 4,16,24,16,4, 1,4,6,4,1]: g5.append(v)
    check(p, f, _identical(gpu_convolve(im, g5, 5, 5, 256.0, 2.0), convolve(im, g5, 5, 5, 256.0, 2.0)), "gpu_convolve gauss5x5+bias bit-exact")

    print("passed:", p, " failed:", f)
    if f == 0: print("ALL GPU TESTS PASSED")

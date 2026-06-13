from std.python import Python

from serenitymojo.serve.image_io import (
    decode_comfy_mask,
    resize_mask_bilinear,
    resize_mask_nearest_exact,
    binarize_lanpaint_denoise_mask,
    comfy_mask_to_preserve_mask,
    load_comfy_latent_preserve_mask,
    load_lanpaint_latent_preserve_mask,
    mask_active_count,
    mask_mean,
)


def _abs(v: Float32) -> Float32:
    return v if v >= 0.0 else -v


def _check_close(name: String, got: Float32, expected: Float32, tol: Float32) raises:
    var d = _abs(got - expected)
    if d > tol:
        raise Error(
            name + String(" got=") + String(got) + String(" expected=")
            + String(expected) + String(" diff=") + String(d)
        )


def _check_int(name: String, got: Int, expected: Int) raises:
    if got != expected:
        raise Error(
            name + String(" got=") + String(got) + String(" expected=")
            + String(expected)
        )


def _make_fixture(path: String) raises:
    var maker = Python.evaluate(
        (
            "def make(path):\n"
            "    from PIL import Image\n"
            "    w, h = 4, 3\n"
            "    alpha = [0, 64, 128, 255, 32, 96, 160, 224, 16, 80, 144, 208]\n"
            "    pixels = []\n"
            "    for y in range(h):\n"
            "        for x in range(w):\n"
            "            r = y * 40 + x * 10\n"
            "            g = 255 - r\n"
            "            b = (x * 60 + y * 20) % 256\n"
            "            pixels.append((r, g, b, alpha[y * w + x]))\n"
            "    img = Image.new('RGBA', (w, h))\n"
            "    img.putdata(pixels)\n"
            "    img.save(path)\n"
        ),
        file=True,
    )
    maker.make(path)


def main() raises:
    var path = String("/tmp/serenity_comfy_mask_fixture.png")
    _make_fixture(path)

    var load_mask = decode_comfy_mask(path, String("load_image_mask"))
    _check_int("load mask width", load_mask.width, 4)
    _check_int("load mask height", load_mask.height, 3)
    _check_close("LoadImage MASK alpha=0", load_mask.values[0], 1.0, 1.0e-6)
    _check_close("LoadImage MASK alpha=255", load_mask.values[3], 0.0, 1.0e-6)
    _check_close("LoadImage MASK alpha=128", load_mask.values[2], 1.0 - Float32(128.0 / 255.0), 1.0e-6)

    var red = decode_comfy_mask(path, String("red"))
    _check_close("ImageToMask red raw[0,0]", red.values[0], 0.0, 1.0e-6)
    _check_close("ImageToMask red raw[3,2]", red.values[11], Float32(110.0 / 255.0), 1.0e-6)
    var alpha = decode_comfy_mask(path, String("alpha"))
    _check_close("ImageToMask alpha raw[0,0]", alpha.values[0], 0.0, 1.0e-6)
    _check_close("ImageToMask alpha raw[3,0]", alpha.values[3], 1.0, 1.0e-6)

    # PyTorch nearest-exact for 3x4 -> 2x2 selects y=[0,2], x=[1,3].
    var red_2x2 = resize_mask_nearest_exact(red, 2, 2)
    _check_int("resized width", red_2x2.width, 2)
    _check_int("resized height", red_2x2.height, 2)
    _check_close("nearest-exact[0,0]", red_2x2.values[0], Float32(10.0 / 255.0), 1.0e-6)
    _check_close("nearest-exact[1,0]", red_2x2.values[1], Float32(30.0 / 255.0), 1.0e-6)
    _check_close("nearest-exact[0,1]", red_2x2.values[2], Float32(90.0 / 255.0), 1.0e-6)
    _check_close("nearest-exact[1,1]", red_2x2.values[3], Float32(110.0 / 255.0), 1.0e-6)

    var red_bilinear = resize_mask_bilinear(red, 2, 2)
    _check_close("bilinear[0,0]", red_bilinear.values[0], Float32(15.0 / 255.0), 1.0e-6)
    _check_close("bilinear[1,0]", red_bilinear.values[1], Float32(35.0 / 255.0), 1.0e-6)
    _check_close("bilinear[0,1]", red_bilinear.values[2], Float32(75.0 / 255.0), 1.0e-6)
    _check_close("bilinear[1,1]", red_bilinear.values[3], Float32(95.0 / 255.0), 1.0e-6)

    var hard = binarize_lanpaint_denoise_mask(red_2x2)
    _check_close("LanPaint >0.5 hard[0]", hard.values[0], 0.0, 1.0e-6)
    _check_close("LanPaint >0.5 hard[3]", hard.values[3], 0.0, 1.0e-6)
    var preserve = comfy_mask_to_preserve_mask(hard)
    _check_close("preserve inverse[0]", preserve.values[0], 1.0, 1.0e-6)
    _check_close("preserve inverse[3]", preserve.values[3], 1.0, 1.0e-6)
    _check_int("preserve active count", mask_active_count(preserve), 4)
    _check_close("preserve mean", mask_mean(preserve), 1.0, 1.0e-6)

    var latent_preserve = load_lanpaint_latent_preserve_mask(path, String("load_image_mask"), 2, 2)
    _check_close("full pipeline preserve[0]", latent_preserve.values[0], 0.0, 1.0e-6)
    _check_close("full pipeline preserve[1]", latent_preserve.values[1], 1.0, 1.0e-6)
    var comfy_preserve = load_comfy_latent_preserve_mask(path, String("load_image_mask"), 2, 2)
    _check_close("Comfy soft preserve[0]", comfy_preserve.values[0], Float32(40.0 / 255.0), 1.0e-5)
    _check_close("Comfy soft preserve[1]", comfy_preserve.values[1], Float32(191.625 / 255.0), 1.0e-5)
    print("image_io_mask_smoke: pass")

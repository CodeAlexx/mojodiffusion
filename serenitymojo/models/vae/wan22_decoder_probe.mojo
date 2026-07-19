# Wan2.2 VAE decoder metadata probe for Lance T2V.
#
# This is intentionally metadata-only: it mmap-opens the Lance Wan2.2 VAE
# safetensors file and checks the decode-side key/shape contract without
# loading the 2.7 GB checkpoint into GPU memory.
#
# Run:
#   pixi run mojo run -I . serenitymojo/models/vae/wan22_decoder_probe.mojo

from serenitymojo.io.safetensors import SafeTensors


comptime WAN22_VAE_PATH = "/home/alex/.serenity/models/vaes/wan2.2_vae.safetensors"


def _shape_to_string(shape: List[Int]) -> String:
    var out = String("[")
    for i in range(len(shape)):
        if i > 0:
            out += ", "
        out += String(shape[i])
    out += "]"
    return out


def _expect_shape(ref st: SafeTensors, name: String, expected: List[Int]) raises:
    var info = st.tensor_info(name)
    if len(info.shape) != len(expected):
        raise Error(
            String("Wan22 VAE shape rank mismatch for ")
            + name
            + String(": got ")
            + _shape_to_string(info.shape)
            + String(" expected ")
            + _shape_to_string(expected)
        )
    for i in range(len(expected)):
        if info.shape[i] != expected[i]:
            raise Error(
                String("Wan22 VAE shape mismatch for ")
                + name
                + String(": got ")
                + _shape_to_string(info.shape)
                + String(" expected ")
                + _shape_to_string(expected)
            )
    print("  ok", name, "dtype=", info.dtype.name(), "shape=", _shape_to_string(info.shape))


def _shape5(a: Int, b: Int, c: Int, d: Int, e: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    s.append(b)
    s.append(c)
    s.append(d)
    s.append(e)
    return s^


def _shape4(a: Int, b: Int, c: Int, d: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    s.append(b)
    s.append(c)
    s.append(d)
    return s^


def _shape3(a: Int, b: Int, c: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    s.append(b)
    s.append(c)
    return s^


def _shape1(a: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    return s^


def main() raises:
    var st = SafeTensors.open(String(WAN22_VAE_PATH))
    print("Wan2.2 VAE safetensors opened:", WAN22_VAE_PATH)
    print("  tensors:", st.count())
    print("  data bytes:", st.data_size())

    # Top-level post-quant conv and decoder input.
    _expect_shape(st, String("conv2.weight"), _shape5(48, 48, 1, 1, 1))
    _expect_shape(st, String("conv2.bias"), _shape1(48))
    _expect_shape(st, String("decoder.conv1.weight"), _shape5(1024, 48, 3, 3, 3))
    _expect_shape(st, String("decoder.conv1.bias"), _shape1(1024))

    # Middle attention and residual structure.
    _expect_shape(st, String("decoder.middle.1.norm.gamma"), _shape3(1024, 1, 1))
    _expect_shape(st, String("decoder.middle.1.to_qkv.weight"), _shape4(3072, 1024, 1, 1))
    _expect_shape(st, String("decoder.middle.1.to_qkv.bias"), _shape1(3072))
    _expect_shape(st, String("decoder.middle.1.proj.weight"), _shape4(1024, 1024, 1, 1))
    _expect_shape(st, String("decoder.middle.1.proj.bias"), _shape1(1024))

    # Nested Wan2.2 upsample paths. These names differ from Wan2.1/QwenImage.
    _expect_shape(
        st,
        String("decoder.upsamples.0.upsamples.3.time_conv.weight"),
        _shape5(2048, 1024, 3, 1, 1),
    )
    _expect_shape(
        st,
        String("decoder.upsamples.1.upsamples.3.time_conv.weight"),
        _shape5(2048, 1024, 3, 1, 1),
    )
    _expect_shape(
        st,
        String("decoder.upsamples.2.upsamples.3.resample.1.weight"),
        _shape4(512, 512, 3, 3),
    )

    # Head outputs 12 channels, then unpatchify(2) -> RGB.
    _expect_shape(st, String("decoder.head.0.gamma"), _shape4(256, 1, 1, 1))
    _expect_shape(st, String("decoder.head.2.weight"), _shape5(12, 256, 3, 3, 3))
    _expect_shape(st, String("decoder.head.2.bias"), _shape1(12))

    # Encoder-side key confirms this is the full Wan2.2 VAE file, not a decoder-only slice.
    _expect_shape(st, String("encoder.conv1.weight"), _shape5(160, 12, 3, 3, 3))

    print("Wan2.2 VAE metadata gate: passed")

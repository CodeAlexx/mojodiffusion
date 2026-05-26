# smoke_safetensors.mojo — opens the Z-Image VAE single-file safetensors via
# the pure-Mojo reader and prints the tensor count + 3 sample tensors.
#
# Run: pixi run mojo run -I . serenitymojo/io/smoke_safetensors.mojo
#
# Path resolved 2026-05-25 via:
#   ls ~/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image/snapshots/*/vae/...
# (there is exactly one snapshot dir) -> hardcoded below.

from serenitymojo.io.safetensors import SafeTensors, TensorRef


comptime VAE_PATH = (
    "/home/alex/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image/"
    "snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021/vae/"
    "diffusion_pytorch_model.safetensors"
)


def _print_tensor(ref st: SafeTensors, name: String) raises:
    var info = st.tensor_info(name)
    # Origin-bound view: the Span ties the mmap'd bytes to `st`'s lifetime, so
    # the compiler keeps the region alive while `bytes` is in use (no footgun).
    var bytes = st.tensor_bytes(name)
    var shape_str = String("[")
    for i in range(len(info.shape)):
        if i > 0:
            shape_str += ", "
        shape_str += String(info.shape[i])
    shape_str += "]"
    # Dereference the mmap'd data segment (proves the pages are actually
    # readable, not just that the metadata parsed).
    var first_byte = Int(bytes[0])
    print(
        "  ",
        name,
        "| dtype=",
        info.dtype.name(),
        "| shape=",
        shape_str,
        "| offset=",
        info.offset,
        "| size=",
        info.size,
        "| first_byte=",
        first_byte,
    )


def main() raises:
    var st = SafeTensors.open(String(VAE_PATH))
    print("Z-Image VAE safetensors opened.")
    print("total tensor count:", st.count())
    print("data segment bytes:", st.data_size())

    # Print 3 specific tensors verified against the Python oracle.
    print("sample tensors:")
    _print_tensor(st, String("decoder.conv_in.bias"))
    _print_tensor(st, String("decoder.conv_in.weight"))
    _print_tensor(st, String("decoder.conv_norm_out.bias"))

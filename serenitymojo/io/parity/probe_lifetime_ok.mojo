# probe_lifetime_ok.mojo — control: keep SafeTensors alive, deref succeeds.
from serenitymojo.io.safetensors import SafeTensors

comptime VAE_PATH = (
    "/home/alex/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image/"
    "snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021/vae/"
    "diffusion_pytorch_model.safetensors"
)


def main() raises:
    var st = SafeTensors.open(String(VAE_PATH))
    # Safe accessor: the Span borrows st, so st is kept alive while p is used.
    var p = st.tensor_bytes(String("decoder.conv_in.bias"))
    var b = Int(p[0])
    print("deref while st alive =", b, " (no crash)")
    # keep st alive to here
    _ = st.count()

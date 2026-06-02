from serenitymojo.io.safetensors import SafeTensors
comptime VAE_PATH = (
    "/home/alex/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image/"
    "snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021/vae/"
    "diffusion_pytorch_model.safetensors"
)
def main() raises:
    var st = SafeTensors.open(String(VAE_PATH))
    st.prefetch_tensor(String("decoder.conv_in.weight"))
    st.prefetch_tensor(String("does.not.exist"))   # guarded no-op
    var p = st.tensor_bytes(String("decoder.conv_in.weight"))
    print("post-prefetch deref:", Int(p[0]), Int(p[100]))
    st.release_to_os()
    # re-access after release: page re-read from disk transparently
    var p2 = st.tensor_bytes(String("decoder.conv_in.weight"))
    print("post-release deref:", Int(p2[0]), Int(p2[100]))
    print("data_size:", st.data_size())

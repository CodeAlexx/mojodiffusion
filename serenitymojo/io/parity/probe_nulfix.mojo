from std.ffi import external_call
from std.memory import alloc
from serenitymojo.io.ffi import sys_open, sys_close, O_RDONLY, BytePtr
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.sharded import _join
comptime VAE_DIR = ("/home/alex/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image/snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021/vae")
def open_nulterm(p: String) -> Int:
    var n = p.byte_length()
    var buf = alloc[UInt8](n + 1)
    var sp = p.as_bytes()
    for i in range(n):
        buf[i] = sp[i]
    buf[n] = 0
    var bp = BytePtr(unsafe_from_address=Int(buf))
    var fd = Int(external_call["open", Int32](bp, O_RDONLY))
    buf.free()
    return fd
def main() raises:
    var dir = String(VAE_DIR)
    var st = SafeTensors.open(_join(dir, String("diffusion_pytorch_model.safetensors")))
    _ = st.count()
    for i in range(3):
        var p = _join(dir, String("diffusion_pytorch_model.safetensors"))
        var fd_std = sys_open(p, O_RDONLY)
        var fd_nul = open_nulterm(p)
        print("  iter", i, " sys_open(unsafe_ptr)=", fd_std, "  open(NUL-term copy)=", fd_nul)
        if fd_std >= 0: _ = sys_close(fd_std)
        if fd_nul >= 0: _ = sys_close(fd_nul)

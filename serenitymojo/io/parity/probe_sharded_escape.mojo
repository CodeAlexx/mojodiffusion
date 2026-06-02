# probe_sharded_escape.mojo — SKEPTIC chunk-2 (4b): the ESCAPE case.
# Try to keep a Span alive past the explicit destruction of the
# ShardedSafeTensors it borrows. This SHOULD be a compile error
# (the documented "rejects letting it escape past self" protection). Build only.

from serenitymojo.io.sharded import ShardedSafeTensors

comptime VAE_DIR = (
    "/home/alex/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image/"
    "snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021/vae"
)


def main() raises:
    var sh = ShardedSafeTensors.open(String(VAE_DIR))
    var s = sh.tensor_bytes(String("decoder.conv_in.bias"))
    # Explicitly end sh's lifetime, then use s. Must NOT compile if the borrow
    # is enforced against the consuming destroy.
    sh^.__del__()
    print(Int(s[0]))

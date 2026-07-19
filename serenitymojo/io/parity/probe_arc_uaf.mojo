# probe_arc_uaf.mojo — SKEPTIC chunk-2 (4a) DECISIVE use-after-free demo.
# Correct first 4 bytes of VAE decoder.conv_in.bias = [105, ...] (Python truth).
# Read the span BEFORE reassign (should be 105) and AFTER reassign (if the
# borrow were enforced this would not compile; if the Arc kept the mmap alive
# it would still be 105; if it is a UAF it will differ / be garbage).

from serenitymojo.io.sharded import ShardedSafeTensors

comptime VAE_DIR = (
    "/home/alex/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image/"
    "snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021/vae"
)
comptime TF_DIR = (
    "/home/alex/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image/"
    "snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021/transformer"
)


def main() raises:
    var sh = ShardedSafeTensors.open(String(VAE_DIR))
    var s = sh.tensor_bytes(String("decoder.conv_in.bias"))
    print("BEFORE reassign: len =", len(s), " first4 =",
          Int(s[0]), Int(s[1]), Int(s[2]), Int(s[3]), " (truth: 105 ...)")
    sh = ShardedSafeTensors.open(String(TF_DIR))
    print("AFTER reassign:  len =", len(s), " first4 =",
          Int(s[0]), Int(s[1]), Int(s[2]), Int(s[3]),
          " (105 ⇒ Arc kept it; anything else ⇒ USE-AFTER-FREE)")
    _ = sh.num_shards()

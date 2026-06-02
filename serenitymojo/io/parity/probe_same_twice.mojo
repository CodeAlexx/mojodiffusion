from serenitymojo.io.sharded import ShardedSafeTensors
def t(dir: String, label: String):
    try:
        var sh = ShardedSafeTensors.open(dir)
        print(label, "OK shards", sh.num_shards(), "tensors", sh.num_tensors())
    except e:
        print(label, "RAISED:", String(e))
def main():
    # Same dir twice (e.g. caller re-opens VAE)
    t(String("/home/alex/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image/snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021/vae"), "[vae #1]")
    t(String("/home/alex/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image/snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021/vae"), "[vae #2 SAME dir]")
    # Same transformer twice in a row
    t(String("/home/alex/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image/snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021/transformer"), "[tf #1]")
    t(String("/home/alex/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image/snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021/transformer"), "[tf #2 SAME dir]")

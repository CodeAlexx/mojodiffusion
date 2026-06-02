# probe_edges.mojo — SKEPTIC chunk-2 (5): edge-case behavior of
# ShardedSafeTensors.open. For each fixture dir, report whether open raises
# cleanly (caught) or crashes the process. We catch raises; a SIGSEGV would
# abort the whole run (so reaching the end == no crash).

from serenitymojo.io.sharded import ShardedSafeTensors


def try_open(dir: String, label: String):
    print("---", label, "---")
    try:
        var sh = ShardedSafeTensors.open(dir)
        print("  OPENED OK: num_shards =", sh.num_shards(),
              " num_tensors =", sh.num_tensors())
    except e:
        print("  RAISED:", String(e))


def main():
    try_open(String("/tmp/edge/missing_shard"), "1 missing shard file")
    try_open(String("/tmp/edge/no_wmap"), "2 no weight_map key")
    try_open(String("/tmp/edge/empty_wmap"), "3 empty weight_map")
    try_open(String("/tmp/edge/dup_names"), "4 duplicate tensor names")
    try_open(String("/tmp/edge/multi_noidx"), "5 multiple .safetensors no index")
    try_open(String("/tmp/edge/empty_dir"), "6 empty dir (no index, no st)")

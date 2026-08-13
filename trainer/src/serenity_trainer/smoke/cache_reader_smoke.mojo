# cache_reader_smoke.mojo — verify gate for dataLoader/CacheReader.mojo.
#
# Reads parity/zi_realclean.safetensors (the single-sample parity cache:
#   latent [1,16,72,56] f32, cap [224,2560] f32) through CacheReader and checks:
#   * open() finds exactly 1 sample (scheme B single-sample fallback)
#   * sample(0) yields latent squeezed to [16,72,56] and cap [224,2560]
#   * the deterministic shuffle is reproducible and a valid permutation
#   * batch() assembles N samples (here N=1, the whole cache)
#
# Built+run by the MAIN LOOP (agents do not build). Plug-in: this mirrors the
# trainer wiring — open the cache, get an epoch order, pull a Sample, hand its
# latent+cap to ModelSpec.predict.

from max.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenity_trainer.dataLoader.CacheReader import CacheReader, Sample, Batch


comptime CACHE_PATH = "trainer/parity/zi_realclean.safetensors"


def _expect(name: String, got: Int, want: Int) raises:
    if got != want:
        raise Error(
            name + String(" MISMATCH got=") + String(got)
            + String(" want=") + String(want)
        )
    print("  OK ", name, "=", got)


def main() raises:
    var ctx = DeviceContext()
    print("=== CacheReader smoke (parity/zi_realclean.safetensors) ===")

    var cache = CacheReader.open(String(CACHE_PATH), ctx)
    _expect("num_samples", cache.len(), 1)

    # ── sample 0: shapes ──────────────────────────────────────────────────────
    var s = cache.sample(0, ctx)
    var lsh = s.latent[].shape()
    var csh = s.cap[].shape()
    print("  latent shape rank =", len(lsh))
    _expect("latent.rank", len(lsh), 3)          # [16,72,56] (leading 1 squeezed)
    _expect("latent.C", lsh[0], 16)
    _expect("latent.H", lsh[1], 72)
    _expect("latent.W", lsh[2], 56)
    _expect("cap.rank", len(csh), 2)
    _expect("cap.L", csh[0], 224)
    _expect("cap.D", csh[1], 2560)

    # ── deterministic shuffle: reproducible + valid permutation ───────────────
    var o1 = cache.shuffle_order(UInt64(1234))
    var o2 = cache.shuffle_order(UInt64(1234))
    _expect("shuffle.len", len(o1), cache.len())
    for i in range(len(o1)):
        if o1[i] != o2[i]:
            raise Error("shuffle NOT reproducible at i=" + String(i))
    # validity: every index 0..len-1 appears exactly once
    var seen = List[Bool]()
    for _i in range(cache.len()):
        seen.append(False)
    for i in range(len(o1)):
        var idx = o1[i]
        if idx < 0 or idx >= cache.len() or seen[idx]:
            raise Error("shuffle invalid permutation at i=" + String(i))
        seen[idx] = True
    print("  OK  shuffle reproducible + valid permutation")

    # ── batch assembly ────────────────────────────────────────────────────────
    var order = cache.sequential_order()
    var b = cache.batch(order, 0, 4, ctx)        # request 4; cache has 1 → 1
    _expect("batch.size", len(b), 1)
    var b0 = b.get(0)
    _expect("batch[0].latent.C", b0.latent[].shape()[0], 16)

    print("=== CacheReader smoke PASSED ===")

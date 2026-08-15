# minimax_h3_train_cache_parity — gate the Mojo mmh3 cache reader against
# files written by TORCHREF'S OWN save functions (h3_train_cache_fixture.py,
# run first with the torchref venv). Checks: item discovery + pairing, every
# logical key, shapes, dtype handling (bf16 / int64 / bool), and f64 value
# sums vs the torch-computed sidecar (rel 1e-9 — same f64 accumulation, only
# summation order differs).
from std.collections import Dict
from std.ffi import external_call
from std.memory import alloc
from max.gpu.host import DeviceContext

from serenitymojo.io.ffi import (
    sys_open, sys_pread, sys_close, BytePtr, O_RDONLY,
)
from serenitymojo.tensor import Tensor
from serenitymojo.models.minimax_h3.h3_train_cache import (
    h3_discover_cache_items, h3_read_latent_cache, h3_read_text_cache,
)

comptime FIXTURE_DIR = "/home/alex/mojodiffusion/output/checks/h3_cache_fixture"


def _read_text(path: String) raises -> List[UInt8]:
    var fd = sys_open(path, O_RDONLY, Int32(0))
    if fd < 0:
        raise Error(String("cannot open ") + path)
    var bytes = List[UInt8]()
    comptime CHUNK = 65536
    var buf = alloc[UInt8](CHUNK)
    var offset = 0
    while True:
        var n = sys_pread(fd, BytePtr(unsafe_from_address=Int(buf)), CHUNK, offset)
        if n < 0:
            buf.free()
            _ = sys_close(fd)
            raise Error("read error")
        if n == 0:
            break
        for i in range(n):
            bytes.append(buf[i])
        offset += n
        if n < CHUNK:
            break
    buf.free()
    _ = sys_close(fd)
    return bytes^


def _atof(s: String) -> Float64:
    var n = s.byte_length()
    var buf = alloc[UInt8](n + 1)
    var src = s.as_bytes()
    for i in range(n):
        buf[i] = src[i]
    buf[n] = 0
    var v = external_call["atof", Float64](
        BytePtr(unsafe_from_address=Int(buf))
    )
    buf.free()
    return v


def _load_expected(path: String) raises -> Dict[String, String]:
    var bytes = _read_text(path)
    var table = Dict[String, String]()
    var line = List[UInt8]()
    var i = 0
    while i <= len(bytes):
        var at_end = i == len(bytes)
        if at_end or bytes[i] == 10:
            if len(line) > 0:
                var eq = -1
                for j in range(len(line)):
                    if line[j] == 61:  # '='
                        eq = j
                        break
                if eq < 0:
                    raise Error("expected.txt: malformed line")
                var k = List[UInt8]()
                var v = List[UInt8]()
                for j in range(eq):
                    k.append(line[j])
                for j in range(eq + 1, len(line)):
                    v.append(line[j])
                table[String(unsafe_from_utf8=k)] = String(unsafe_from_utf8=v)
                line = List[UInt8]()
        else:
            line.append(bytes[i])
        i += 1
    return table^


def _f64_sum(t: Tensor, ctx: DeviceContext) raises -> Float64:
    var h = t.to_host(ctx)
    var s = Float64(0)
    for i in range(len(h)):
        s += Float64(h[i])
    return s


struct _Gate(Movable):
    var expected: Dict[String, String]
    var ok: Bool

    def __init__(out self, var expected: Dict[String, String]):
        self.expected = expected^
        self.ok = True

    def want_int(mut self, key: String, got: Int) raises:
        if key not in self.expected:
            raise Error(String("expected.txt missing ") + key)
        var want = Int(_atof(self.expected[key]))
        var good = want == got
        print(("PASS " if good else "FAIL ") + key + " want", want, "got", got)
        if not good:
            self.ok = False

    def want_f64(mut self, key: String, got: Float64) raises:
        if key not in self.expected:
            raise Error(String("expected.txt missing ") + key)
        var want = _atof(self.expected[key])
        var denom = abs(want)
        if denom < 1.0:
            denom = 1.0
        var rel = abs(want - got) / denom
        var good = rel <= 1.0e-9
        print(
            ("PASS " if good else "FAIL ") + key + " want", want,
            "got", got, "rel", rel,
        )
        if not good:
            self.ok = False


def main() raises:
    var ctx = DeviceContext()
    var g = _Gate(_load_expected(String(FIXTURE_DIR) + "/expected.txt"))

    var items = h3_discover_cache_items(String(FIXTURE_DIR))
    if len(items) != 2:
        raise Error("discovery: expected 2 paired items")
    # order by name for stable comparison
    var want_names: List[String] = [String("clip_alpha"), String("clip_beta")]
    var order = List[Int]()
    for w in range(len(want_names)):
        var found = -1
        for i in range(len(items)):
            if items[i].item_key == want_names[w]:
                found = i
        if found < 0:
            raise Error(String("discovery: missing item ") + want_names[w])
        order.append(found)
    print("PASS discovery: 2 items paired (clip_alpha, clip_beta)")

    for oi in range(len(order)):
        var it = items[order[oi]].copy()
        var name = it.item_key
        var lat = h3_read_latent_cache(it.latent_path, ctx)
        if not lat.has_video or not lat.has_audio or not lat.has_keyframe_rows:
            raise Error("latent cache missing video/audio/keyframes")
        g.want_int(name + ".lat_f", lat.lat_f)
        g.want_int(name + ".lat_h", lat.lat_h)
        g.want_int(name + ".lat_w", lat.lat_w)
        g.want_int(name + ".audio_t", lat.audio_t)
        g.want_f64(name + ".video_sum", _f64_sum(lat.video[], ctx))
        g.want_f64(name + ".audio_sum", _f64_sum(lat.audio[], ctx))
        var true_count = 0
        for i in range(len(lat.audio_loss_mask)):
            if lat.audio_loss_mask[i]:
                true_count += 1
        g.want_int(name + ".mask_true", true_count)
        var kf_sh = lat.keyframe_rows[].shape()
        g.want_int(name + ".kf_rows", kf_sh[0])
        g.want_int(name + ".kf_width", kf_sh[1])
        g.want_f64(name + ".kf_sum", _f64_sum(lat.keyframe_rows[], ctx))
        g.want_int(
            name + ".has_video_loss_mask", 1 if lat.has_video_loss_mask else 0
        )

        var te = h3_read_text_cache(it.te_path, ctx)
        g.want_int(name + ".tokens", te.tokens)
        g.want_f64(name + ".hidden_sum", _f64_sum(te.hidden[], ctx))
        var tag_sum = 0
        for i in range(len(te.tags)):
            tag_sum += te.tags[i]
        g.want_int(name + ".tags_sum", tag_sum)
        g.want_int(name + ".task_id", te.task_id)
        g.want_int(name + ".has_empty", 1 if te.has_empty else 0)
        if te.has_empty:
            g.want_int(name + ".empty_tokens", te.empty_hidden[].shape()[0])
            g.want_f64(name + ".empty_hidden_sum", _f64_sum(te.empty_hidden[], ctx))
            var esum = 0
            for i in range(len(te.empty_tags)):
                esum += te.empty_tags[i]
            g.want_int(name + ".empty_tags_sum", esum)

    if g.ok:
        print("PASS: mmh3 cache reader matches torchref-written fixture")
    else:
        print("FAIL: mmh3 cache reader parity below bar")
        raise Error("minimax_h3_train_cache_parity failed")

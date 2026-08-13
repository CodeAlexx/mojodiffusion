# Real-weight SCAIL-2 block-0 + shared image-projection parity gate.
# Fixture is produced only by scail2_real_block_oracle.py from the pinned source.

from std.collections import List
from max.gpu.host import DeviceContext
from std.sys import argv

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.models.scail2.scail2_block import scail2_block_forward
from serenitymojo.models.scail2.scail2_config import Scail2Config
from serenitymojo.models.scail2.scail2_streamed_dit import Scail2StreamedDiT
from serenitymojo.models.dit.wan22_dit import _expand_rope_per_head
from serenitymojo.tensor import Tensor


comptime S = 9
comptime TXT = 512
comptime IMG = 257
comptime D = 5120
comptime H = 40
comptime DH = 128


def _shape_matches(a: List[Int], b: List[Int]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


def _load(
    st: ShardedSafeTensors, name: String, shape: List[Int], dtype: STDtype,
    ctx: DeviceContext,
) raises -> Tensor:
    var tv = st.tensor_view(name)
    if tv.dtype != dtype or not _shape_matches(tv.shape, shape):
        raise Error(String("SCAIL-2 real block fixture mismatch: ") + name)
    return Tensor.from_view(tv, ctx)


def _metrics(a: Tensor, b: Tensor, ctx: DeviceContext) raises -> Tuple[Float64, Float64]:
    var ah = a.to_host(ctx)
    var bh = b.to_host(ctx)
    if len(ah) != len(bh):
        raise Error("SCAIL-2 real block metric length mismatch")
    var dot = 0.0
    var aa = 0.0
    var bb = 0.0
    var max_abs = 0.0
    for i in range(len(ah)):
        var av = Float64(ah[i])
        var bv = Float64(bh[i])
        dot += av * bv
        aa += av * av
        bb += bv * bv
        var delta = av - bv
        if delta < 0.0:
            delta = -delta
        if delta > max_abs:
            max_abs = delta
    return (dot / (aa * bb) ** 0.5, max_abs)


def main() raises:
    var args = argv()
    if len(args) != 3:
        raise Error("usage: scail2_real_block_parity <fixture> <fp8-cache-dir>")
    var ctx = DeviceContext()
    var fixture = ShardedSafeTensors.open(args[1])
    var x = _load(fixture, "x", [1, S, D], STDtype.BF16, ctx)
    var e0 = _load(fixture, "e0", [1, 1, 6, D], STDtype.F32, ctx)
    var text = _load(fixture, "text_context", [1, TXT, D], STDtype.BF16, ctx)
    var image = _load(fixture, "image_context", [1, IMG, D], STDtype.BF16, ctx)
    var cos = _load(fixture, "cos", [S, DH // 2], STDtype.BF16, ctx)
    var sin = _load(fixture, "sin", [S, DH // 2], STDtype.BF16, ctx)
    var cos_h = _expand_rope_per_head(cos, S, H, DH // 2, ctx)
    var sin_h = _expand_rope_per_head(sin, S, H, DH // 2, ctx)
    var canonical = _load(
        fixture, "canonical_output", [1, S, D], STDtype.F32, ctx
    )
    var python_fp8 = _load(fixture, "fp8_output", [1, S, D], STDtype.F32, ctx)

    var model = Scail2StreamedDiT.open(String(args[2]), ctx)
    var block = model.stream.load_block_bf16(0, ctx)
    var actual = scail2_block_forward[S, TXT, IMG, H, DH](
        x, e0, text, image, cos_h, sin_h, block, Scail2Config.scail2_14b(), ctx
    )
    var vs_python_fp8 = _metrics(actual, python_fp8, ctx)
    var vs_canonical = _metrics(actual, canonical, ctx)
    var native_block = model.stream.load_block_fp8(0, ctx)
    var native_actual = scail2_block_forward[S, TXT, IMG, H, DH](
        x, e0, text, image, cos_h, sin_h, native_block,
        Scail2Config.scail2_14b(), ctx
    )
    var native_vs_fp8 = _metrics(native_actual, actual, ctx)
    var native_vs_canonical = _metrics(native_actual, canonical, ctx)
    print(
        "SCAIL-2 Mojo FP8 vs Python FP8: cos=", vs_python_fp8[0],
        " maxabs=", vs_python_fp8[1],
    )
    print(
        "SCAIL-2 Mojo FP8 vs canonical: cos=", vs_canonical[0],
        " maxabs=", vs_canonical[1],
    )
    print(
        "SCAIL-2 native FP8 vs Mojo FP8: cos=", native_vs_fp8[0],
        " maxabs=", native_vs_fp8[1],
    )
    print(
        "SCAIL-2 native FP8 vs canonical: cos=", native_vs_canonical[0],
        " maxabs=", native_vs_canonical[1],
    )

    var raw_clip = _load(
        fixture, "raw_clip", [1, IMG, 1280], STDtype.F16, ctx
    )
    var image_expected = _load(
        fixture, "image_expected", [1, IMG, D], STDtype.F32, ctx
    )
    var image_actual = model._image[IMG](raw_clip, ctx)
    var image_metrics = _metrics(image_actual, image_expected, ctx)
    print(
        "SCAIL-2 shared image MLP: cos=", image_metrics[0],
        " maxabs=", image_metrics[1],
    )

    var video20 = _load(
        fixture, "video20", [20, 1, 4, 4], STDtype.F32, ctx
    )
    var primary_ref20 = _load(
        fixture, "primary_ref20", [20, 1, 4, 4], STDtype.F32, ctx
    )
    var pose20 = _load(
        fixture, "pose20", [20, 1, 2, 2], STDtype.F32, ctx
    )
    var primary_video_masks28 = _load(
        fixture, "primary_video_masks28", [28, 2, 4, 4], STDtype.F32, ctx
    )
    var driving_masks28 = _load(
        fixture, "driving_masks28", [28, 1, 2, 2], STDtype.F32, ctx
    )
    var sequence_expected = _load(
        fixture, "sequence_expected", [1, S, D], STDtype.BF16, ctx
    )
    var sequence_actual = model._embed_base_sequence[1, 2, 2, S](
        video20, primary_ref20, pose20, primary_video_masks28,
        driving_masks28, ctx,
    )
    var sequence_metrics = _metrics(sequence_actual, sequence_expected, ctx)
    print(
        "SCAIL-2 shared patch/mask embedding: cos=", sequence_metrics[0],
        " maxabs=", sequence_metrics[1],
    )

    var head_hidden = _load(
        fixture, "head_hidden", [1, S, D], STDtype.F32, ctx
    )
    var e_head = _load(fixture, "e_head", [1, 1, D], STDtype.F32, ctx)
    var head_expected = _load(
        fixture, "head_expected", [16, 1, 4, 4], STDtype.F32, ctx
    )
    var head_actual = model._head_video[1, 2, 2, 0, S](
        head_hidden, e_head, STDtype.BF16, ctx
    )
    var head_metrics = _metrics(head_actual, head_expected, ctx)
    print(
        "SCAIL-2 shared head video slice: cos=", head_metrics[0],
        " maxabs=", head_metrics[1],
    )

    if (
        vs_python_fp8[0] < 0.999 or vs_canonical[0] < 0.999
        or native_vs_fp8[0] < 0.999 or native_vs_canonical[0] < 0.999
        or image_metrics[0] < 0.999 or sequence_metrics[0] < 0.999
        or head_metrics[0] < 0.999
    ):
        raise Error("SCAIL-2 real block/shared microgate cosine below 0.999")
    print("GATE PASS SCAIL-2 BF16/native-FP8 block0 + shared cos>=0.999")

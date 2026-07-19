# SDXL ADM pooled+size conditioning parity against conditioning_oracle.py and
# the frozen local Serenity sampler helper time-id contract it consumes.

from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.models.sdxl.conditioning import sdxl_adm_y
from serenitymojo.parity import ParityHarness


comptime ORACLE = "serenitymojo/models/sdxl/parity/conditioning_oracle.safetensors"


def main() raises:
    var ctx = DeviceContext()
    var st = ShardedSafeTensors.open(String(ORACLE))
    var pooled = Tensor.from_view(st.tensor_view(String("pooled")), ctx)
    var expected_ref = Tensor.from_view(
        st.tensor_view(String("expected_sampler_ref")), ctx
    ).to_host(ctx)
    var actual_ref = sdxl_adm_y(pooled, 1024, 1152, ctx)
    var result_ref = ParityHarness(0.99999).compare(actual_ref, expected_ref, ctx)
    print("SDXL ADM frozen Serenity sampler-ref 1152x1024:", result_ref)

    var expected_square = Tensor.from_view(
        st.tensor_view(String("expected_square")), ctx
    ).to_host(ctx)
    var actual_square = sdxl_adm_y(pooled, 1024, 1024, ctx)
    var result_square = ParityHarness(0.99999).compare(
        actual_square, expected_square, ctx
    )
    print("SDXL ADM product square 1024x1024:", result_square)

    var expected_1152x896 = Tensor.from_view(
        st.tensor_view(String("expected_1152x896")), ctx
    ).to_host(ctx)
    var actual_1152x896 = sdxl_adm_y(pooled, 896, 1152, ctx)
    var result_1152x896 = ParityHarness(0.99999).compare(
        actual_1152x896, expected_1152x896, ctx
    )
    print("SDXL ADM product 1152x896:", result_1152x896)

    var expected_896x1152 = Tensor.from_view(
        st.tensor_view(String("expected_896x1152")), ctx
    ).to_host(ctx)
    var actual_896x1152 = sdxl_adm_y(pooled, 1152, 896, ctx)
    var result_896x1152 = ParityHarness(0.99999).compare(
        actual_896x1152, expected_896x1152, ctx
    )
    print("SDXL ADM product 896x1152:", result_896x1152)

    var expected_landscape = Tensor.from_view(
        st.tensor_view(String("expected_landscape")), ctx
    ).to_host(ctx)
    var actual_landscape = sdxl_adm_y(pooled, 768, 1344, ctx)
    var result_landscape = ParityHarness(0.99999).compare(
        actual_landscape, expected_landscape, ctx
    )
    print("SDXL ADM product landscape 1344x768:", result_landscape)

    var expected_portrait = Tensor.from_view(
        st.tensor_view(String("expected_portrait")), ctx
    ).to_host(ctx)
    var actual_portrait = sdxl_adm_y(pooled, 1344, 768, ctx)
    var result_portrait = ParityHarness(0.99999).compare(
        actual_portrait, expected_portrait, ctx
    )
    print("SDXL ADM product portrait 768x1344:", result_portrait)

    var expected_1280x832 = Tensor.from_view(
        st.tensor_view(String("expected_1280x832")), ctx
    ).to_host(ctx)
    var actual_1280x832 = sdxl_adm_y(pooled, 832, 1280, ctx)
    var result_1280x832 = ParityHarness(0.99999).compare(
        actual_1280x832, expected_1280x832, ctx
    )
    print("SDXL ADM product 1280x832:", result_1280x832)

    var expected_832x1280 = Tensor.from_view(
        st.tensor_view(String("expected_832x1280")), ctx
    ).to_host(ctx)
    var actual_832x1280 = sdxl_adm_y(pooled, 1280, 832, ctx)
    var result_832x1280 = ParityHarness(0.99999).compare(
        actual_832x1280, expected_832x1280, ctx
    )
    print("SDXL ADM product 832x1280:", result_832x1280)

    # Torch and the Mojo GPU kernel both evaluate in F32 then store BF16; their
    # libm frequency paths can land one BF16 ULP apart near unit magnitude.
    if (
        result_ref.cos < 0.99999 or result_ref.max_abs > 0.00390625
        or result_square.cos < 0.99999
        or result_square.max_abs > 0.00390625
        or result_1152x896.cos < 0.99999
        or result_1152x896.max_abs > 0.00390625
        or result_896x1152.cos < 0.99999
        or result_896x1152.max_abs > 0.00390625
        or result_landscape.cos < 0.99999
        or result_landscape.max_abs > 0.00390625
        or result_portrait.cos < 0.99999
        or result_portrait.max_abs > 0.00390625
        or result_1280x832.cos < 0.99999
        or result_1280x832.max_abs > 0.00390625
        or result_832x1280.cos < 0.99999
        or result_832x1280.max_abs > 0.00390625
    ):
        raise Error("SDXL ADM conditioning parity failed")
    print(
        "PASS: SDXL ADM matches frozen local time-id contract and all seven"
        " product shapes"
    )

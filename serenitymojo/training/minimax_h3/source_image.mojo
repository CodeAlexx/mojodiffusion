# Native source-image decode seam for MiniMax-H3 training caches.
#
# Musubi preserves RGB/RGBA source pixels and `_prepare_pixels` later selects
# the first three channels.  Grayscale-family inputs are converted to RGB.

from serenitymojo.pipeline.minimax_h3_keyframe_image import MiniMaxH3RgbImage
from serenitymojo.image.decode import decode_image


def minimax_h3_decode_source_rgb8(path: String) raises -> MiniMaxH3RgbImage:
    """Decode JPEG/PNG/WebP in native Mojo and apply Musubi RGB semantics."""
    # The trainer decoder uses system libturbojpeg/libpng/libwebp.  The
    # `drop_alpha=True` flag means preserve the stored RGB channels and ignore
    # alpha, exactly as Musubi's later `frames[..., :3]` boundary does.
    var decoded = decode_image(path, drop_alpha=True)
    if decoded.width < 1 or decoded.height < 1:
        raise Error("MiniMax H3 source image has invalid geometry")
    if len(decoded.rgb) != decoded.width * decoded.height * 3:
        raise Error("MiniMax H3 source decoder did not return RGB8 HWC")
    return MiniMaxH3RgbImage(decoded.rgb.copy(), decoded.height, decoded.width)

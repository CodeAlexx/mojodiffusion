# image — codecs + manipulation for Mojo (100% Mojo)

Decode/encode **PNG, JPEG, and WebP**, plus a full **manipulation** suite
(resize/rotate/color/filters) and **studio** features (16-bit/HDR, ICC profiles,
EXIF, CMYK, lanczos, unsharp). Pure Mojo — the only "dependency" is the sibling
[`graphics`](../graphics/) lib for its `Canvas`, DEFLATE (`deflate`) and INFLATE
(`inflate`). No image/codec C library is linked.

Every codec and op is verified against **Pillow** (and `ImageCms`/littleCMS for
ICC) as an independent oracle — lossless paths are checked **pixel-exact**, lossy
paths by **PSNR**. Fixture generators live in `image/tests/*.py` (committed), so
the suite is reproducible.

## Modules

| Module | What it is |
|---|---|
| `buffer.mojo` | `Image` — row-major pixel container (Gray/RGB/RGBA, 8-bit core; carries `bit_depth`/`sample_format`/`colorspace`/`icc`/`exif` for the studio path). `get/set`, `get_pixel/set_pixel(Color)`, `clone`, and a `to_canvas`/`from_canvas` bridge to `graphics.Canvas`. |
| `png.mojo` | **PNG** decode+encode: all color types (gray/RGB/palette/gray+α/RGBA), bit depths 1/2/4/8/16, **Adam7** interlace, tRNS, CRC-validated chunks; encode via zlib-wrapped DEFLATE. Decode is **pixel-exact vs PIL**. |
| `jpeg.mojo` | **JPEG** baseline (SOF0) decode + encode. Decode: Huffman, dequant, 8×8 IDCT, 4:4:4/4:2:2/4:2:0 with fancy chroma upsample, YCbCr→RGB, restart markers (progressive → clear error). Encode: 4:4:4, standard quant+Huffman tables, quality param. |
| `webp.mojo` | **WebP VP8L (lossless)** decode + encode: predictor/color/subtract-green/color-indexing transforms, color-cache, meta-Huffman, LZ77 back-refs. Decode is **pixel-exact vs PIL**. (lossy VP8 → clear error.) |
| `transform.mojo` | resize (nearest/**bilinear**/**bicubic** — bit-exact vs PIL's resampler), rotate90/180/270 + arbitrary, flip_h/v, crop, transpose. |
| `color.mojo` | grayscale (601 luma), invert, brightness, contrast, gamma, saturation, levels, threshold. |
| `filter.mojo` | convolve, box/gaussian blur, sharpen, sobel/edge, median (clamp-to-edge). |
| `depth.mojo` | 16-bit (`get16/set16`) + F32 (`getf/setf`) accessors, U8↔U16↔F32 conversions, Reinhard HDR tonemap. |
| `icc.mojo` | ICC profile parse (header, tag table, XYZ primaries, TRC `curv`/`para`) + matrix/TRC **profile→sRGB** transform (D50-consistent). |
| `exif.mojo` | EXIF/TIFF-IFD parse (II/MM, ExifIFD sub-IFD, common tags incl. rationals) + `build_exif` (PIL-readable). |
| `cmyk.mojo` | CMYK↔RGB (naive, **bit-exact vs PIL**) + Adobe-inverted CMYK helper. |
| `studio_ops.mojo` | **lanczos** resize (upscale bit-exact vs PIL) + **unsharp mask**. |
| `gpu.mojo` | **GPU-accelerated ops** (MAX): `gpu_invert`/`gpu_brightness`/`gpu_grayscale`/`gpu_contrast` + generic `gpu_convolve` (box/sharpen/gaussian/sobel). **Bit-exact vs the CPU reference**; `has_accelerator()`-guarded (fall back to CPU when no GPU). **~18× faster** than the CPU path on a 1024² gaussian (incl. transfer), measured on an RTX 3090 Ti. |

## Example

```mojo
from image.png import decode_png, encode_png
from image.jpeg import encode_jpeg
from image.transform import resize_bilinear
from image.color import grayscale

var img = decode_png("in.png")             # -> Image
var small = resize_bilinear(img, 256, 256)
var gray = grayscale(small)
encode_jpeg(gray, "out.jpg", 90)
```

## Verified (PIL/oracle, measured — not asserted)

11 test files, all passing. Decode pixel-exact / encode PIL-openable / ops vs PIL:

```bash
# regenerate fixtures, then run (from the repo root, -I .)
python3 image/tests/png_fixtures.py   && pixi run mojo run -I . image/tests/png_test.mojo
python3 image/tests/jpeg_fixtures.py  && pixi run mojo run -I . image/tests/jpeg_test.mojo
python3 image/tests/webp_fixtures.py  && pixi run mojo run -I . image/tests/webp_test.mojo
pixi run mojo run -I . image/tests/ops_test.mojo      && python3 image/tests/ops_oracle.py
pixi run mojo run -I . image/tests/depth_test.mojo
pixi run mojo run -I . image/tests/icc_test.mojo
pixi run mojo run -I . image/tests/exif_test.mojo
pixi run mojo run -I . image/tests/cmyk_test.mojo
pixi run mojo run -I . image/tests/studio_ops_test.mojo && python3 image/tests/studio_ops_oracle.py
```

Measured highlights: PNG decode/encode **pixel-exact** (RGB/RGBA/gray/palette/Adam7/16-bit);
JPEG decode **53–65 dB** vs PIL, encode q90 **35–40 dB** (PIL opens it); WebP VP8L
**pixel-exact** decode+round-trip; resize bilinear/bicubic/lanczos-upscale **bit-exact**;
CMYK **bit-exact** vs PIL; ICC sRGB→sRGB identity **max-diff 0**; EXIF round-trips through PIL.

## Scope / limitations (honest)

- **GPU acceleration** (`gpu.mojo`) covers the per-pixel color ops + generic convolution,
  bit-exact vs the CPU reference and ~18× faster on a 1024² gaussian (RTX 3090 Ti, measured).
  Still CPU-only (planned GPU follow-ups): resize/rotate resampling, JPEG IDCT, and the codecs
  (PNG/JPEG/WebP decode are sequential — they stay CPU).
- **JPEG**: baseline only (progressive → clear error); encode is 4:4:4 (no chroma subsampling)
  with standard (non-optimized) Huffman tables.
- **WebP**: lossless VP8L only (lossy VP8 → clear error); the encoder is minimal-but-valid
  (all-literal Huffman, no transforms/LZ77) — files are libwebp/PIL-decodable.
- **Studio**: `depth`/`icc`/`cmyk` provide the buffer machinery + conversions; wiring codecs to
  *emit* 16-bit / CMYK Images (PNG-16 decode path, iCCP chunk, CMYK-JPEG) are follow-ups.
  ICC transforms matrix/TRC RGB profiles (LUT-based & CMYK profiles parsed, not transformed);
  unsharp uses a true gaussian (PSNR-close to PIL's box approximation, not bit-exact).
- 16-bit accessors store little-endian; AVIF / JPEG-XL are out of scope (would require a full
  AV1 / JXL codec).

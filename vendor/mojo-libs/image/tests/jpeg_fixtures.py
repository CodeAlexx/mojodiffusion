import numpy as np
from PIL import Image
import os

base = "/tmp/jpg_fix"
os.makedirs(base, exist_ok=True)

W, H = 96, 64
# gradient + texture/noise so the DCT is exercised
xx, yy = np.meshgrid(np.arange(W), np.arange(H))
r = (xx * 255 // W).astype(np.float64)
g = (yy * 255 // H).astype(np.float64)
b = ((xx + yy) * 255 // (W + H)).astype(np.float64)
# texture: checker + sine ripples
checker = (((xx // 8) + (yy // 8)) % 2) * 40
ripple = (np.sin(xx / 3.0) * 30 + np.cos(yy / 4.0) * 30)
rng = np.random.RandomState(1234)
noise = rng.randint(-12, 13, size=(H, W))
r = np.clip(r + checker + noise, 0, 255)
g = np.clip(g + ripple + noise, 0, 255)
b = np.clip(b + checker - ripple + noise, 0, 255)
rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
img = Image.fromarray(rgb, "RGB")

def save_pix(name, pil_img):
    # decode PIL's own JPEG and dump pixels
    arr = np.asarray(pil_img)
    if arr.ndim == 2:
        c = 1
        h, w = arr.shape
        flat = arr.reshape(-1)
    else:
        h, w, c = arr.shape
        flat = arr.reshape(-1)
    with open(f"{base}/{name}.pix", "w") as fh:
        fh.write(f"{w} {h} {c}\n")
        fh.write(" ".join(str(int(v)) for v in flat))

# 4:4:4
img.save(f"{base}/q90_444.jpg", "JPEG", quality=90, subsampling=0)
save_pix("q90_444", Image.open(f"{base}/q90_444.jpg"))
# 4:2:2
img.save(f"{base}/q90_422.jpg", "JPEG", quality=90, subsampling=1)
save_pix("q90_422", Image.open(f"{base}/q90_422.jpg"))
# 4:2:0
img.save(f"{base}/q90_420.jpg", "JPEG", quality=90, subsampling=2)
save_pix("q90_420", Image.open(f"{base}/q90_420.jpg"))
# grayscale
gray = img.convert("L")
gray.save(f"{base}/q90_gray.jpg", "JPEG", quality=90)
save_pix("q90_gray", Image.open(f"{base}/q90_gray.jpg"))
# restart interval (4:2:0). restart_marker_blocks param sets MCU rows interval.
try:
    img.save(f"{base}/q90_rst.jpg", "JPEG", quality=90, subsampling=2, restart_marker_blocks=4)
    save_pix("q90_rst", Image.open(f"{base}/q90_rst.jpg"))
    print("restart fixture: OK")
except Exception as e:
    # fallback: try restart_marker_rows
    try:
        img.save(f"{base}/q90_rst.jpg", "JPEG", quality=90, subsampling=2, restart_marker_rows=1)
        save_pix("q90_rst", Image.open(f"{base}/q90_rst.jpg"))
        print("restart fixture: OK (rows)")
    except Exception as e2:
        print("restart fixture FAILED:", e, e2)

print("fixtures written to", base)

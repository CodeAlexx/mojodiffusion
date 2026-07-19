#!/usr/bin/env python3
"""Quantitative audio-quality diagnostic for LTX2 outputs.

Reports, per file (wav or mp4 audio track):
  rms, peak, spectral centroid (Hz), % energy above 1 kHz / 4 kHz,
  spectral flatness (geometric/arithmetic mean of the power spectrum —
  ~1.0 = white noise, <<1 = tonal/structured), and a 4-band energy split.
Writes an optional spectrogram PNG for eye verification.

The 2026-06-10 "audio still noise" class measured rms 0.03-0.05 with <=1 kHz
domination; real speech/music shows structured spectra (low flatness) and
meaningful >1 kHz energy.
"""
import subprocess
import sys
import tempfile

import numpy as np


def load_audio(path: str, sr: int = 48000) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".f32", delete=False) as t:
        raw = t.name
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", path, "-f", "f32le",
         "-ac", "1", "-ar", str(sr), raw],
        check=True,
    )
    return np.fromfile(raw, dtype=np.float32)


def diag(path: str, png: str | None = None) -> None:
    sr = 48000
    x = load_audio(path, sr)
    if len(x) < sr // 2:
        print(f"{path}: too short ({len(x)} samples)")
        return
    rms = float(np.sqrt(np.mean(x ** 2)))
    peak = float(np.max(np.abs(x)))

    n = 4096
    hop = 2048
    n_frames = (len(x) - n) // hop
    win = np.hanning(n)
    spec = np.stack([
        np.abs(np.fft.rfft(x[i * hop:i * hop + n] * win)) ** 2
        for i in range(n_frames)
    ])  # [T, F]
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    pmean = spec.mean(axis=0) + 1e-20
    centroid = float((freqs * pmean).sum() / pmean.sum())
    e_total = pmean.sum()
    above1k = float(pmean[freqs > 1000].sum() / e_total)
    above4k = float(pmean[freqs > 4000].sum() / e_total)
    flat = float(np.exp(np.mean(np.log(pmean))) / np.mean(pmean))
    bands = []
    for lo, hi in ((0, 300), (300, 1000), (1000, 4000), (4000, 24000)):
        m = (freqs >= lo) & (freqs < hi)
        bands.append(float(pmean[m].sum() / e_total))
    # temporal structure: std over time of per-frame energy (dB) — speech/music
    # modulates; stationary noise doesn't.
    fe = 10 * np.log10(spec.sum(axis=1) + 1e-20)
    tmod = float(np.std(fe))

    print(f"== {path}")
    print(f"   dur={len(x)/sr:.2f}s rms={rms:.4f} peak={peak:.3f}")
    print(f"   centroid={centroid:.0f}Hz  E>1kHz={above1k*100:.1f}%  "
          f"E>4kHz={above4k*100:.1f}%  flatness={flat:.4f}")
    print(f"   bands 0-300/300-1k/1k-4k/4k+ = "
          f"{bands[0]*100:.1f}/{bands[1]*100:.1f}/{bands[2]*100:.1f}/{bands[3]*100:.1f}%")
    print(f"   temporal-mod std={tmod:.2f} dB")

    if png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        db = 10 * np.log10(spec.T + 1e-12)
        plt.figure(figsize=(12, 4))
        plt.imshow(db, aspect="auto", origin="lower",
                   extent=[0, n_frames * hop / sr, 0, sr / 2 / 1000],
                   vmin=db.max() - 80, vmax=db.max(), cmap="magma")
        plt.ylabel("kHz"); plt.xlabel("s"); plt.title(path.split("/")[-1])
        plt.colorbar(label="dB")
        plt.tight_layout()
        plt.savefig(png, dpi=80)
        plt.close()
        print(f"   spectrogram -> {png}")


if __name__ == "__main__":
    for i, p in enumerate(sys.argv[1:]):
        png = None
        if p.endswith(".png"):
            continue
        diag(p, png=p.rsplit(".", 1)[0] + "_spec.png")

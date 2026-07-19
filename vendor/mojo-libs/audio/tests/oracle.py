#!/usr/bin/env python3
# Independent oracle for the WAV lib: cross-check Mojo decode/encode vs
# scipy/soundfile. Requires the echo CLI built + run first:
#   pixi run mojo build -I . audio/tests/echo_cli.mojo -o /tmp/wav_echo
#   /tmp/wav_echo <real_s16.wav> /tmp/echo32.wav     # mojo decode -> float32 wav
#   pixi run mojo run -I . audio/tests/wav_test.mojo  # also writes /tmp/sine16/32.wav
import sys
import numpy as np
import soundfile as sf
from scipy.io import wavfile

SRC = sys.argv[1] if len(sys.argv) > 1 else "/home/alex/Wan2.2/examples/zero_shot_prompt.wav"

sr0, d0 = wavfile.read(SRC)
d0f = d0.astype(np.float64) / 32768.0
d1, sr1 = sf.read("/tmp/echo32.wav", dtype="float64")
print("scipy:", sr0, d0.shape, " mojo-echo:", sr1, d1.shape)
n = min(len(d0f), len(d1))
err = float(np.max(np.abs(d0f[:n] - d1[:n]))) if n else 1.0
ok_decode = sr0 == sr1 and len(d0f) == len(d1) and err < 2e-5
print(f"max |scipy_decode - mojo_decode| over {n} = {err:.2e} ->", "PASS" if ok_decode else "FAIL")

s16, _ = sf.read("/tmp/sine16.wav")
s32, _ = sf.read("/tmp/sine32.wav")
ok_write = len(s16) == 16000 and len(s32) == 16000
print("mojo-written sine16/sine32 valid ->", "PASS" if ok_write else "FAIL")

raise SystemExit(0 if (ok_decode and ok_write) else 1)

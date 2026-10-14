"""Bikin 'humming' sintetis untuk ngetes pipeline tanpa harus merekam.

    python examples/make_synthetic_hum.py examples/twinkle.wav
    neofunk transcribe examples/twinkle.wav --bpm 100
"""

import sys

import soundfile as sf

from neofunk.synth import TWINKLE, synth_hum

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "examples/twinkle.wav"
    y, sr = synth_hum(TWINKLE)
    sf.write(out, y, sr)
    print(f"Tersimpan: {out}")

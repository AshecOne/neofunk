"""Humming sintetis (vibrato, scoop, jeda napas, noise) untuk tes & demo."""

from __future__ import annotations

import numpy as np

# Twinkle Twinkle (C D ...) : (pitch MIDI, durasi dalam ketukan)
TWINKLE = [(60, 1), (60, 1), (67, 1), (67, 1), (69, 1), (69, 1), (67, 2),
           (65, 1), (65, 1), (64, 1), (64, 1), (62, 1), (62, 1), (60, 2)]


def synth_hum(melody, bpm=100.0, sr=16000, detune_cents=-30.0, vibrato_cents=25.0,
              noise=0.01, gap_s=0.06, seed=0):
    rng = np.random.default_rng(seed)
    beat = 60.0 / bpm
    chunks = [np.zeros(int(0.3 * sr))]
    for pitch, beats in melody:
        dur = beats * beat
        n = int(dur * sr)
        voiced = max(1, n - int(gap_s * sr))
        t = np.arange(voiced) / sr
        # scoop kecil di awal not (penyanyi biasanya "naik" ke nada)
        scoop = -60.0 * np.exp(-t / 0.03)
        vib = vibrato_cents * np.sin(2 * np.pi * 5.5 * t) * np.clip(t / 0.3, 0, 1)
        cents = detune_cents + scoop + vib
        f0 = 440.0 * 2 ** ((pitch - 69 + cents / 100) / 12)
        phase = 2 * np.pi * np.cumsum(f0) / sr
        tone = sum((0.6 / k) * np.sin(k * phase) for k in range(1, 6))
        env = np.minimum(1, np.minimum(t / 0.02, (voiced / sr - t) / 0.03))
        chunks.append(tone * env * rng.uniform(0.6, 1.0))
        chunks.append(np.zeros(n - voiced))
    chunks.append(np.zeros(int(0.3 * sr)))
    y = np.concatenate(chunks)
    y = y + noise * rng.standard_normal(len(y))
    return (0.8 * y / np.max(np.abs(y))).astype(np.float32), sr


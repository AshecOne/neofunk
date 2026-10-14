"""Pitch tracking (f0 per frame) dengan beberapa backend.

- "pyin"  : librosa.pyin. CPU saja, tanpa download model, cukup akurat untuk humming bersih.
- "crepe" : torchcrepe (CNN). Pakai GPU (RTX 4060 lebih dari cukup), lebih tahan noise & napas.

Backend "basic-pitch" langsung menghasilkan not, jadi ditangani di pipeline.py, bukan di sini.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PitchTrack:
    times: np.ndarray  # detik, per frame
    f0: np.ndarray  # Hz, NaN untuk frame tak bersuara
    confidence: np.ndarray  # 0..1
    hop_s: float


def hz_to_midi(f0: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return 69.0 + 12.0 * np.log2(f0 / 440.0)


def track_pyin(y: np.ndarray, sr: int, fmin: float, fmax: float, hop: int = 256) -> PitchTrack:
    import librosa

    # 64 ms @16 kHz: cukup untuk nada rendah (~65 Hz) tapi tidak terlalu "melebar" di onset
    frame_length = 1024 if sr <= 16000 else 2048
    f0, voiced, prob = librosa.pyin(
        y,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=frame_length,
        hop_length=hop,
        center=True,
    )
    f0 = np.where(voiced, f0, np.nan)
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop)
    return PitchTrack(times=times, f0=f0, confidence=np.nan_to_num(prob), hop_s=hop / sr)


def track_crepe(
    y: np.ndarray,
    sr: int,
    fmin: float,
    fmax: float,
    hop_s: float = 0.01,
    model: str = "full",
    device: str | None = None,
    voicing_threshold: float = 0.5,
) -> PitchTrack:
    try:
        import torch
        import torchcrepe
    except ImportError as e:  # pragma: no cover - dependensi opsional
        raise ImportError("Backend 'crepe' butuh: pip install -e '.[gpu]'") from e

    if sr != 16000:
        import librosa

        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    hop = int(round(hop_s * sr))
    audio = torch.tensor(y, dtype=torch.float32, device=device)[None]
    f0, periodicity = torchcrepe.predict(
        audio,
        sr,
        hop,
        fmin,
        fmax,
        model=model,
        decoder=torchcrepe.decode.viterbi,
        return_periodicity=True,
        batch_size=1024,
        device=device,
    )
    periodicity = torchcrepe.filter.median(periodicity, 3)
    f0 = torchcrepe.filter.mean(f0, 3)
    f0 = f0[0].cpu().numpy().astype(np.float64)
    conf = periodicity[0].cpu().numpy().astype(np.float64)
    f0 = np.where(conf >= voicing_threshold, f0, np.nan)
    times = np.arange(len(f0)) * (hop / sr)
    return PitchTrack(times=times, f0=f0, confidence=conf, hop_s=hop / sr)


def track_pitch(y: np.ndarray, sr: int, backend: str = "pyin", fmin: float = 65.0, fmax: float = 1050.0) -> PitchTrack:
    if backend == "pyin":
        return track_pyin(y, sr, fmin, fmax)
    if backend == "crepe":
        return track_crepe(y, sr, fmin, fmax)
    raise ValueError(f"Backend pitch tidak dikenal: {backend!r} (pilihan: pyin, crepe)")

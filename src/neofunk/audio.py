"""Loading & pra-pemrosesan audio."""

from __future__ import annotations

import numpy as np
import scipy.signal


def load_audio(path: str, sr: int = 16000) -> tuple[np.ndarray, int]:
    """Load file audio apa pun (wav/mp3/m4a/flac...) jadi mono float32 di sample rate `sr`."""
    import librosa

    y, _ = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32), sr


def preprocess(y: np.ndarray, sr: int, highpass_hz: float = 60.0) -> np.ndarray:
    """High-pass (buang dengung/rumble mic), hilangkan DC, normalisasi peak."""
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return y
    y = y - float(np.mean(y))
    if highpass_hz and highpass_hz > 0:
        sos = scipy.signal.butter(4, highpass_hz, btype="highpass", fs=sr, output="sos")
        y = scipy.signal.sosfiltfilt(sos, y).astype(np.float32)
    peak = float(np.max(np.abs(y)))
    if peak > 1e-6:
        y = 0.95 * y / peak
    return y


def frame_rms(y: np.ndarray, hop: int, frame: int, n_frames: int) -> np.ndarray:
    """RMS per frame, panjangnya dipaskan ke `n_frames` (sejajar dengan pitch track)."""
    import librosa

    rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop, center=True)[0]
    if rms.size < n_frames:
        rms = np.pad(rms, (0, n_frames - rms.size), mode="edge" if rms.size else "constant")
    return rms[:n_frames]

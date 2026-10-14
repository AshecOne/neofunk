"""Pipeline utama: audio -> pitch -> not -> (tuning, key, tempo, kuantisasi)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from neofunk import music
from neofunk.audio import frame_rms, load_audio, preprocess
from neofunk.notes import Note, Transcription
from neofunk.pitch import track_pitch
from neofunk.segment import SegmentConfig, segment_notes


@dataclass
class TranscribeConfig:
    backend: str = "pyin"  # pyin | crepe | basic-pitch
    sr: int = 16000
    fmin: float = 65.0  # ~C2, suara cowok rendah
    fmax: float = 1050.0  # ~C6, siulan / falsetto tinggi
    bpm: float | None = None  # None = estimasi otomatis
    grid: int = 16  # 16 = kuantisasi ke not 1/16; 0 = tanpa kuantisasi
    key: str | None = "auto"  # "auto", None (tanpa key), atau mis. "A minor"
    snap_to_scale: bool = False
    transpose: int = 0  # semitone, mis. -12 untuk turun satu oktaf
    beats_per_bar: int = 4
    segment: SegmentConfig = field(default_factory=SegmentConfig)


def _basic_pitch_notes(path: str) -> list[Note]:
    try:
        from basic_pitch.inference import predict
    except ImportError as e:  # pragma: no cover
        raise ImportError("Backend 'basic-pitch' butuh: pip install -e '.[basicpitch]'") from e

    _, _, events = predict(path)
    notes = [
        Note(start=float(s), end=float(e), pitch=int(p), velocity=int(np.clip(a * 127, 1, 127)))
        for s, e, p, a, _bends in events
    ]
    # Suara mulut itu monofonik: kalau ada not tumpang tindih, ambil yang paling keras.
    notes.sort(key=lambda n: n.start)
    mono: list[Note] = []
    for n in notes:
        if mono and n.start < mono[-1].end:
            if n.velocity > mono[-1].velocity:
                mono[-1].end = n.start
                if mono[-1].duration <= 0.03:
                    mono.pop()
                mono.append(n)
            continue
        mono.append(n)
    return mono


def transcribe(
    source: str | np.ndarray,
    config: TranscribeConfig | None = None,
    sr: int | None = None,
) -> Transcription:
    """Transkripsi suara jadi not.

    `source` bisa path file audio, atau numpy array (wajib isi `sr`)."""
    cfg = config or TranscribeConfig()
    if isinstance(source, str):
        y, sr = load_audio(source, cfg.sr)
    else:
        if sr is None:
            raise ValueError("Kalau source berupa array, sr wajib diisi")
        y = np.asarray(source, dtype=np.float32)
        if sr != cfg.sr:
            import librosa

            y = librosa.resample(y, orig_sr=sr, target_sr=cfg.sr)
            sr = cfg.sr
    y = preprocess(y, sr)

    tuning = 0.0
    if cfg.backend == "basic-pitch":
        if not isinstance(source, str):
            raise ValueError("Backend basic-pitch butuh path file audio")
        notes = _basic_pitch_notes(source)
    else:
        import librosa

        track = track_pitch(y, sr, cfg.backend, cfg.fmin, cfg.fmax)
        hop = int(round(track.hop_s * sr))
        rms = frame_rms(y, hop, 2 * hop if hop >= 256 else 512, len(track.f0))
        onsets = None
        if cfg.segment.onset_split:
            onsets = librosa.onset.onset_detect(
                y=y, sr=sr, hop_length=hop, units="time", backtrack=False, delta=0.2
            )
        notes, tuning = segment_notes(track, rms, onsets, cfg.segment)

    for n in notes:
        n.pitch = int(np.clip(n.pitch + cfg.transpose, 0, 127))

    key = None
    if cfg.key == "auto":
        key = music.detect_key(notes)
    elif cfg.key:
        key = music.parse_key(cfg.key)
    if key and cfg.snap_to_scale:
        notes = music.snap_to_scale(notes, key)

    tempo = cfg.bpm or music.estimate_tempo(y, sr)
    if cfg.grid:
        notes = music.quantize(notes, tempo, cfg.grid)

    return Transcription(
        notes=notes,
        tempo=float(tempo),
        key=music.key_name(key),
        tuning_cents=round(tuning * 100, 1),
        beats_per_bar=cfg.beats_per_bar,
        meta={"backend": cfg.backend, "grid": cfg.grid, "duration_s": round(len(y) / sr, 3)},
    )

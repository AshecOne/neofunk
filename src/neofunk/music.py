"""Teori musik praktis: estimasi tempo, deteksi key, kuantisasi ke grid, snap ke tangga nada."""

from __future__ import annotations

import numpy as np

from neofunk.notes import NOTE_NAMES, Note

# Profil Krumhansl-Kessler (bobot kemunculan tiap pitch class dalam key mayor/minor)
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_SCALE_STEPS = {"major": [0, 2, 4, 5, 7, 9, 11], "minor": [0, 2, 3, 5, 7, 8, 10]}


def estimate_tempo(y: np.ndarray, sr: int, lo: float = 70.0, hi: float = 160.0) -> float:
    """Estimasi BPM dari onset envelope. Untuk humming hasilnya kadang ganda/setengah,
    jadi dilipat ke rentang [lo, hi]. Lebih baik pakai --bpm kalau kamu tahu tempo-nya
    (misalnya rekam sambil dengar metronom)."""
    import librosa

    env = librosa.onset.onset_strength(y=y, sr=sr)
    if not np.any(env > 0):
        return 120.0
    tempo = float(np.atleast_1d(librosa.feature.tempo(onset_envelope=env, sr=sr))[0])
    if not np.isfinite(tempo) or tempo <= 0:
        return 120.0
    while tempo < lo:
        tempo *= 2
    while tempo > hi:
        tempo /= 2
    return round(tempo, 1)


def detect_key(notes: list[Note]) -> tuple[int, str] | None:
    """Algoritma Krumhansl-Schmuckler, dibobot durasi. Return (tonic_pc, 'major'|'minor')."""
    if not notes:
        return None
    hist = np.zeros(12)
    for n in notes:
        hist[n.pitch % 12] += n.duration
    if hist.sum() == 0:
        return None
    best, best_r = None, -np.inf
    for mode, prof in (("major", _MAJOR), ("minor", _MINOR)):
        for tonic in range(12):
            r = np.corrcoef(hist, np.roll(prof, tonic))[0, 1]
            if np.isfinite(r) and r > best_r:
                best, best_r = (tonic, mode), r
    return best


def key_name(key: tuple[int, str] | None) -> str | None:
    return None if key is None else f"{NOTE_NAMES[key[0]]} {key[1]}"


def parse_key(text: str) -> tuple[int, str]:
    """'A minor', 'Am', 'C', 'F# major' -> (pc, mode)."""
    t = text.strip().replace("♯", "#")
    mode = "major"
    low = t.lower()
    for suffix, m in ((" minor", "minor"), (" major", "major"), ("min", "minor"), ("maj", "major"), ("m", "minor")):
        if low.endswith(suffix):
            t, mode = t[: -len(suffix)].strip(), m
            break
    t = t[:1].upper() + t[1:]
    flats = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}
    t = flats.get(t, t)
    if t not in NOTE_NAMES:
        raise ValueError(f"Key tidak dikenal: {text!r}")
    return NOTE_NAMES.index(t), mode


def snap_to_scale(notes: list[Note], key: tuple[int, str]) -> list[Note]:
    """Geser not di luar tangga nada ke nada terdekat di dalam tangga nada.
    Arah geser mengikuti intonasi aslinya (cents > 0 -> naik)."""
    tonic, mode = key
    allowed = {(tonic + s) % 12 for s in _SCALE_STEPS[mode]}
    for n in notes:
        if n.pitch % 12 in allowed:
            continue
        step = 1 if n.cents > 0 else -1
        if (n.pitch + step) % 12 in allowed:
            n.pitch += step
        elif (n.pitch - step) % 12 in allowed:
            n.pitch -= step
    return notes


def best_grid_offset(onsets: list[float], step_s: float, n_candidates: int = 21) -> float:
    """Cari offset (dekat onset pertama) yang meminimalkan jarak onset ke grid."""
    t = np.asarray(sorted(onsets))
    shifts = np.linspace(-0.5, 0.5, n_candidates) * step_s
    errs = []
    for sh in shifts:
        r = (t - (t[0] + sh)) / step_s
        errs.append(np.sum(np.abs(r - np.round(r))) + 1e-3 * abs(sh) / step_s)
    return float(t[0] + shifts[int(np.argmin(errs))])


def quantize(
    notes: list[Note],
    tempo: float,
    grid: int = 16,
    offset: float | None = None,
) -> list[Note]:
    """Kuantisasi ke grid (grid=16 -> not 1/16, grid=8 -> 1/8, dst).

    `offset` = waktu (detik) yang dianggap beat 0. Default: dicari otomatis di sekitar
    awal not pertama, yaitu fase grid yang paling pas dengan semua onset not (jadi kalau
    not pertama agak telat/kecepetan, sisa lagu tidak ikut bergeser).
    Mengisi Note.start_beat / end_beat (dalam satuan ketukan / quarter note)."""
    if not notes:
        return notes
    beat_s = 60.0 / tempo
    steps_per_beat = grid / 4
    step_s = beat_s / steps_per_beat
    if offset is None:
        offset = best_grid_offset([n.start for n in notes], step_s)

    out: list[Note] = []
    for n in sorted(notes, key=lambda x: x.start):
        s = int(round((n.start - offset) / step_s))
        e = int(round((n.end - offset) / step_s))
        e = max(e, s + 1)
        if out:
            ps = int(round(out[-1].start_beat * steps_per_beat))
            pe = int(round(out[-1].end_beat * steps_per_beat))
            if s <= ps:  # dua not jatuh di step yang sama
                s = ps + 1
                if s >= e:
                    continue
                pe = s
            if s < pe:
                pe = s
            out[-1].end_beat = pe / steps_per_beat
        n.start_beat = s / steps_per_beat
        n.end_beat = e / steps_per_beat
        out.append(n)
    return out

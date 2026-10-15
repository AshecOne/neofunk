"""Segmentasi not: kurva pitch per frame -> daftar Note diskrit.

Ide utamanya (versi heuristik, tanpa training):
1. Konversi f0 ke skala MIDI (semitone), buang frame yang terlalu pelan.
2. Estimasi offset tuning global penyanyi (kebanyakan orang tidak pas A=440).
3. Tambal celah unvoiced yang sangat pendek (napas/konsonan kecil di tengah not).
4. Haluskan dengan median filter supaya vibrato & jitter tidak memecah not.
5. Potong jadi not baru kalau pitch bergeser > ~0.7 semitone dan bertahan beberapa frame,
   atau kalau ada onset (serangan "ta/da/na") di tengah not yang pitch-nya sama.
6. Pitch not = median pitch setelah bagian attack (scoop di awal not diabaikan).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.ndimage

from neofunk.notes import Note
from neofunk.pitch import PitchTrack, hz_to_midi


@dataclass
class SegmentConfig:
    min_note_s: float = 0.07  # not lebih pendek dari ini dibuang
    split_semitones: float = 0.7  # pergeseran pitch untuk mulai not baru
    min_change_s: float = 0.04  # pergeseran harus bertahan selama ini
    gap_fill_s: float = 0.05  # celah unvoiced <= ini ditambal kalau pitch kiri/kanan mirip
    # not bersebelahan dgn pitch sama & celah <= ini digabung. Default 0: hanya potongan yang
    # menempel (hasil jitter pitch). Celah sungguhan sudah diputuskan oleh _fill_gaps.
    merge_gap_s: float = 0.0
    smooth_s: float = 0.05  # lebar median filter
    min_confidence: float = 0.1  # frame dengan confidence pitch di bawah ini = unvoiced
    silence_db: float = -45.0  # frame di bawah (max_rms + silence_db) dianggap hening
    min_note_db: float = -30.0  # not yang rata-rata lebih pelan dari ini dibuang (noise/napas)
    attack_skip: float = 0.15  # porsi awal not yang diabaikan saat hitung pitch
    correct_tuning: bool = True
    onset_split: bool = True
    onset_guard_s: float = 0.12  # onset sedekat ini dari awal/akhir not diabaikan (masih attack)


def estimate_tuning(midi: np.ndarray) -> float:
    """Offset tuning global (semitone, -0.5..0.5) via rata-rata sirkular bagian pecahan."""
    v = midi[np.isfinite(midi)]
    if v.size == 0:
        return 0.0
    ang = 2 * np.pi * (v - np.round(v))
    return float(np.angle(np.mean(np.exp(1j * ang))) / (2 * np.pi))


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Daftar (start, end_exclusive) untuk run True berturut-turut."""
    if mask.size == 0:
        return []
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _fill_gaps(
    midi: np.ndarray,
    max_gap: int,
    max_jump: float = 1.0,
    db: np.ndarray | None = None,
    dip_db: float = 10.0,
) -> np.ndarray:
    """Tambal celah unvoiced pendek. Celah dengan penurunan energi > `dip_db` tidak
    ditambal: itu jeda sungguhan antar not (mis. "da . da"), bukan pitch tracker yang
    sesaat kehilangan nada."""
    midi = midi.copy()
    voiced = np.isfinite(midi)
    for s, e in _runs(~voiced):
        if s == 0 or e == len(midi) or e - s > max_gap:
            continue
        if db is not None:
            w = max_gap  # bandingkan dengan level not di kiri & kanan (bukan frame tepi yang sedang fade)
            around = min(np.max(db[max(0, s - w):s]), np.max(db[e:e + w]))
            if np.min(db[s:e]) < around - dip_db:
                continue
        left, right = midi[s - 1], midi[e]
        if abs(left - right) <= max_jump:
            midi[s:e] = np.linspace(left, right, e - s + 2)[1:-1]
    return midi


def _split_run(m: np.ndarray, thresh: float, min_change: int) -> list[tuple[int, int]]:
    """Pecah satu run voiced (sudah dihaluskan) berdasarkan perubahan pitch."""
    segs = []
    seg_start = 0
    dev_start = None
    for i in range(1, len(m)):
        center = np.median(m[seg_start:dev_start if dev_start is not None else i])
        if abs(m[i] - center) > thresh:
            if dev_start is None:
                dev_start = i
            elif i - dev_start + 1 >= min_change:
                segs.append((seg_start, dev_start))
                seg_start, dev_start = dev_start, None
        else:
            dev_start = None
    segs.append((seg_start, len(m)))
    return segs


def segment_notes(
    track: PitchTrack,
    rms: np.ndarray | None = None,
    onsets: np.ndarray | None = None,
    cfg: SegmentConfig | None = None,
) -> tuple[list[Note], float]:
    """Kembalikan (notes, tuning_semitones)."""
    cfg = cfg or SegmentConfig()
    hop = track.hop_s
    midi = hz_to_midi(track.f0).astype(np.float64)
    midi[~np.isfinite(midi)] = np.nan
    midi[track.confidence < cfg.min_confidence] = np.nan

    if rms is not None and rms.size == midi.size and np.max(rms) > 0:
        db = 20 * np.log10(np.maximum(rms, 1e-10) / np.max(rms))
        midi[db < cfg.silence_db] = np.nan
    else:
        db = None

    tuning = estimate_tuning(midi) if cfg.correct_tuning else 0.0
    midi = midi - tuning

    midi = _fill_gaps(midi, max(1, int(round(cfg.gap_fill_s / hop))), db=db)

    k = max(1, int(round(cfg.smooth_s / hop)))
    k += 1 - k % 2  # harus ganjil
    min_change = max(1, int(round(cfg.min_change_s / hop)))
    min_frames = max(1, int(round(cfg.min_note_s / hop)))
    guard = max(min_frames, int(round(cfg.onset_guard_s / hop)))
    onset_frames = (
        np.unique(np.round(np.asarray(onsets) / hop).astype(int))
        if (cfg.onset_split and onsets is not None and len(onsets))
        else np.array([], dtype=int)
    )

    # (start, end, forced_split) — forced_split=True artinya dipotong oleh onset,
    # jadi jangan digabung lagi walau pitch-nya sama (mis. "da-da-da" di nada yang sama).
    segments: list[tuple[int, int, bool]] = []
    for rs, re_ in _runs(np.isfinite(midi)):
        run = midi[rs:re_]
        smooth = scipy.ndimage.median_filter(run, size=min(k, len(run) | 1), mode="nearest")
        for a, b in _split_run(smooth, cfg.split_semitones, min_change):
            a, b = a + rs, b + rs
            cuts = [o for o in onset_frames if a + guard <= o <= b - guard]
            bounds = [a, *cuts, b]
            for j in range(len(bounds) - 1):
                segments.append((bounds[j], bounds[j + 1], j > 0))

    notes: list[Note] = []
    forced: list[bool] = []
    for a, b, f in segments:
        vals = midi[a:b]
        skip = int((b - a) * cfg.attack_skip)
        core = vals[skip:] if b - a - skip >= 2 else vals
        center = float(np.nanmedian(core))
        pitch = int(round(center))
        if db is not None:
            level = float(np.mean(db[a:b]))  # dB relatif ke frame terkeras
            if level < cfg.min_note_db:
                continue
            vel = int(np.clip(110 + level * 1.5, 30, 127))
        else:
            vel = 90
        notes.append(
            Note(
                start=float(track.times[a]),
                end=float(track.times[b - 1] + hop),
                pitch=pitch,
                velocity=vel,
                cents=round((center - pitch) * 100, 1),
            )
        )
        forced.append(f)

    # Gabung not bersebelahan dengan pitch sama (hasil glide/jitter), kecuali dipotong onset.
    merged: list[Note] = []
    for n, f in zip(notes, forced):
        if (
            merged
            and not f
            and n.pitch == merged[-1].pitch
            and n.start - merged[-1].end <= cfg.merge_gap_s + hop / 2
        ):
            merged[-1].end = n.end
        else:
            merged.append(n)

    # Buang not terlalu pendek; kalau menempel ke not sebelumnya, perpanjang not itu
    # (biasanya ini nada transisi/portamento antara dua not).
    out: list[Note] = []
    for n in merged:
        if n.duration >= cfg.min_note_s:
            out.append(n)
        elif out and n.start - out[-1].end <= hop * 1.5:
            out[-1].end = n.end
    return out, tuning

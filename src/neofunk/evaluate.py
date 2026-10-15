"""Benchmark transkripsi pakai mir_eval (metrik standar MIREX untuk transkripsi not).

Metrik per file:
- note_f1        : not benar kalau onset meleset <= 50 ms DAN pitch meleset <= 50 cents
                   (offset diabaikan). Ini metrik utama ("COn" di paper SVT).
- note_f1_offset : seperti di atas + offset juga harus pas (<= 20% durasi / 50 ms). Lebih ketat.
- onset_f1       : hanya cek onset (pitch diabaikan). Mengukur kualitas "pemotongan" not.
- pitch_acc      : dari not yang onset-nya cocok, berapa persen pitch-nya benar.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from neofunk.datasets import Pair
from neofunk.notes import Note


def _arrays(notes: list[Note]) -> tuple[np.ndarray, np.ndarray]:
    if not notes:
        return np.zeros((0, 2)), np.zeros(0)
    itv = np.array([[n.start, max(n.end, n.start + 1e-3)] for n in notes], dtype=float)
    hz = 440.0 * 2 ** ((np.array([n.pitch for n in notes], dtype=float) - 69) / 12)
    return itv, hz


def read_midi_notes(path: str | Path) -> list[Note]:
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(str(path))
    notes = [
        Note(float(n.start), float(n.end), int(n.pitch), int(n.velocity))
        for inst in pm.instruments
        if not inst.is_drum
        for n in inst.notes
    ]
    return sorted(notes, key=lambda n: n.start)


def best_octave_shift(ref: list[Note], est: list[Note]) -> int:
    """Banyak orang humming 1 oktaf lebih rendah/tinggi dari label. Cari geseran oktaf
    (kelipatan 12) yang membuat median pitch paling dekat."""
    if not ref or not est:
        return 0
    diff = np.median([n.pitch for n in ref]) - np.median([n.pitch for n in est])
    return int(12 * round(diff / 12))


def note_metrics(
    ref: list[Note],
    est: list[Note],
    onset_tol: float = 0.05,
    pitch_tol_cents: float = 50.0,
    octave_invariant: bool = False,
) -> dict[str, float]:
    import mir_eval

    if octave_invariant:
        sh = best_octave_shift(ref, est)
        est = [Note(n.start, n.end, n.pitch + sh, n.velocity) for n in est]
    ri, rp = _arrays(ref)
    ei, ep = _arrays(est)
    if len(ref) == 0 or len(est) == 0:
        hit = float(len(ref) == len(est))
        return {"note_f1": hit, "note_f1_offset": hit, "onset_f1": hit, "pitch_acc": hit,
                "n_ref": len(ref), "n_est": len(est)}
    t = mir_eval.transcription
    _, _, f_on, _ = t.precision_recall_f1_overlap(ri, rp, ei, ep, onset_tolerance=onset_tol,
                                                  pitch_tolerance=pitch_tol_cents, offset_ratio=None)
    _, _, f_off, _ = t.precision_recall_f1_overlap(ri, rp, ei, ep, onset_tolerance=onset_tol,
                                                   pitch_tolerance=pitch_tol_cents)
    _, _, f_onset = t.onset_precision_recall_f1(ri, ei, onset_tolerance=onset_tol)
    n_onset = len(t.match_note_onsets(ri, ei, onset_tolerance=onset_tol))
    n_pitch = len(t.match_notes(ri, rp, ei, ep, onset_tolerance=onset_tol,
                                pitch_tolerance=pitch_tol_cents, offset_ratio=None))
    return {
        "note_f1": float(f_on),
        "note_f1_offset": float(f_off),
        "onset_f1": float(f_onset),
        "pitch_acc": n_pitch / n_onset if n_onset else 0.0,
        "n_ref": len(ref),
        "n_est": len(est),
    }


@dataclass
class BackendResult:
    backend: str
    rows: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    seconds: float = 0.0

    def mean(self, key: str) -> float:
        vals = [r[key] for r in self.rows]
        return float(np.mean(vals)) if vals else float("nan")


def evaluate(
    pairs: list[Pair],
    backends: list[str],
    octave_invariant: bool = False,
    onset_tol: float = 0.05,
    progress: bool = True,
) -> list[BackendResult]:
    from neofunk.pipeline import TranscribeConfig, transcribe

    results = []
    for backend in backends:
        res = BackendResult(backend)
        # tanpa kuantisasi & tanpa koreksi key: kita bandingkan timing asli dengan label
        cfg = TranscribeConfig(backend=backend, grid=0, key=None, bpm=120)
        t0 = time.perf_counter()
        for i, p in enumerate(pairs):
            try:
                est = transcribe(str(p.audio), cfg).notes
                m = note_metrics(read_midi_notes(p.midi), est, onset_tol=onset_tol,
                                 octave_invariant=octave_invariant)
                res.rows.append({"name": p.name, **m})
            except Exception as e:  # satu file rusak jangan menghentikan benchmark
                res.errors.append(f"{p.name}: {e}")
            if progress:
                print(f"\r[{backend}] {i + 1}/{len(pairs)}", end="", flush=True)
        res.seconds = time.perf_counter() - t0
        if progress:
            print()
        results.append(res)
    return results


def format_table(results: list[BackendResult]) -> str:
    head = f"{'backend':<12} {'note F1':>8} {'F1+offset':>10} {'onset F1':>9} {'pitch acc':>10} {'file':>5} {'waktu':>8}"
    lines = [head, "-" * len(head)]
    for r in results:
        lines.append(
            f"{r.backend:<12} {r.mean('note_f1') * 100:7.1f}% {r.mean('note_f1_offset') * 100:9.1f}% "
            f"{r.mean('onset_f1') * 100:8.1f}% {r.mean('pitch_acc') * 100:9.1f}% {len(r.rows):>5} {r.seconds:7.1f}s"
        )
        for e in r.errors[:3]:
            lines.append(f"    ! {e}")
        if len(r.errors) > 3:
            lines.append(f"    ! ... {len(r.errors) - 3} error lain")
    return "\n".join(lines)


def write_csv(results: list[BackendResult], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = ["backend", "name", "note_f1", "note_f1_offset", "onset_f1", "pitch_acc", "n_ref", "n_est"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in results:
            for row in r.rows:
                w.writerow({"backend": r.backend, **row})
    return path

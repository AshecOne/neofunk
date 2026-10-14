"""Ekspor hasil transkripsi ke MIDI, MusicXML (partitur), JSON, dan WAV preview."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from neofunk.notes import Transcription


def _note_times(tr: Transcription, quantized: bool):
    beat_s = 60.0 / tr.tempo
    for n in tr.notes:
        if quantized and n.start_beat is not None:
            yield n, n.start_beat * beat_s, n.end_beat * beat_s
        else:
            yield n, n.start, n.end


def to_pretty_midi(tr: Transcription, program: int = 0, quantized: bool = True):
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(initial_tempo=tr.tempo)
    pm.time_signature_changes.append(pretty_midi.TimeSignature(tr.beats_per_bar, 4, 0.0))
    inst = pretty_midi.Instrument(program=program, name="Melody")
    for n, s, e in _note_times(tr, quantized):
        if e > s:
            inst.notes.append(pretty_midi.Note(velocity=n.velocity, pitch=n.pitch, start=max(0.0, s), end=e))
    pm.instruments.append(inst)
    return pm


def write_midi(tr: Transcription, path: str | Path, program: int = 0, quantized: bool = True) -> Path:
    path = Path(path)
    to_pretty_midi(tr, program, quantized).write(str(path))
    return path


def write_json(tr: Transcription, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(tr.to_dict(), indent=2, ensure_ascii=False))
    return path


def write_musicxml(tr: Transcription, path: str | Path, title: str = "neofunk melody") -> Path:
    """Tulis partitur MusicXML (buka di MuseScore / Sibelius / Finale / Dorico)."""
    try:
        from music21 import key as m21key
        from music21 import metadata, meter, note, stream, tempo
    except ImportError as e:  # pragma: no cover
        raise ImportError("Ekspor MusicXML butuh: pip install -e '.[score]'") from e

    path = Path(path)
    score = stream.Score()
    score.metadata = metadata.Metadata(title=title)
    part = stream.Part()
    part.insert(0, meter.TimeSignature(f"{tr.beats_per_bar}/4"))
    part.insert(0, tempo.MetronomeMark(number=round(tr.tempo)))
    if tr.key:
        tonic, mode = tr.key.split()
        part.insert(0, m21key.Key(tonic, mode))

    beat_s = 60.0 / tr.tempo
    for n in tr.notes:
        if n.start_beat is not None:
            off, ql = n.start_beat, n.end_beat - n.start_beat
        else:  # belum dikuantisasi: bulatkan kasar ke 1/16
            off = round(n.start / beat_s * 4) / 4
            ql = max(0.25, round(n.duration / beat_s * 4) / 4)
        if ql <= 0:
            continue
        m = note.Note(n.pitch, quarterLength=ql)
        m.volume.velocity = n.velocity
        part.insert(max(0.0, off), m)
    score.insert(0, part)
    score.write("musicxml", fp=str(path))
    return path


def write_preview_wav(tr: Transcription, path: str | Path, sr: int = 22050, quantized: bool = True) -> Path:
    """Render cepat (gelombang sinus) supaya kamu bisa langsung dengar hasilnya."""
    import soundfile as sf

    path = Path(path)
    audio = to_pretty_midi(tr, 0, quantized).synthesize(fs=sr)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0:
        audio = 0.9 * audio / peak
    sf.write(str(path), audio.astype(np.float32), sr)
    return path

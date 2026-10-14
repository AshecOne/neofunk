"""Struktur data dasar: Note dan hasil Transcription."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_to_name(pitch: int) -> str:
    """60 -> 'C4'."""
    return f"{NOTE_NAMES[pitch % 12]}{pitch // 12 - 1}"


@dataclass
class Note:
    start: float  # detik
    end: float  # detik
    pitch: int  # nomor MIDI (60 = C4)
    velocity: int = 90
    cents: float = 0.0  # deviasi rata-rata dari pitch bulat (info intonasi)
    start_beat: float | None = None  # terisi setelah kuantisasi
    end_beat: float | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def name(self) -> str:
        return midi_to_name(self.pitch)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["name"] = self.name
        return d


@dataclass
class Transcription:
    notes: list[Note]
    tempo: float  # BPM
    key: str | None = None  # contoh: "A minor"
    tuning_cents: float = 0.0  # offset tuning global penyanyi
    beats_per_bar: int = 4
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "tempo": self.tempo,
            "key": self.key,
            "tuning_cents": self.tuning_cents,
            "beats_per_bar": self.beats_per_bar,
            "meta": self.meta,
            "notes": [n.to_dict() for n in self.notes],
        }

    def pretty(self) -> str:
        lines = [
            f"Tempo: {self.tempo:.1f} BPM | Key: {self.key or '-'} | "
            f"Tuning offset: {self.tuning_cents:+.0f} cents | {len(self.notes)} not",
        ]
        for n in self.notes:
            beat = f"  beat {n.start_beat:6.2f}-{n.end_beat:6.2f}" if n.start_beat is not None else ""
            lines.append(f"  {n.name:<4} {n.start:7.3f}s - {n.end:7.3f}s  vel {n.velocity:3d}{beat}")
        return "\n".join(lines)

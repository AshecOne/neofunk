import numpy as np
import pytest

from neofunk import TranscribeConfig, transcribe
from neofunk.export import write_json, write_midi, write_musicxml
from neofunk.music import detect_key, parse_key, quantize, snap_to_scale
from neofunk.notes import Note
from neofunk.synth import TWINKLE, synth_hum


@pytest.fixture(scope="module")
def twinkle():
    return synth_hum(TWINKLE, bpm=100)


@pytest.fixture(scope="module")
def result(twinkle):
    y, sr = twinkle
    return transcribe(y, TranscribeConfig(bpm=100), sr=sr)


def test_pitches_match_melody(result):
    assert [n.pitch for n in result.notes] == [p for p, _ in TWINKLE]


def test_rhythm_quantized_to_beats(result):
    expected, t = [], 0
    for _, beats in TWINKLE:
        expected.append(t)
        t += beats
    assert [n.start_beat for n in result.notes] == expected


def test_tuning_and_key(result):
    assert -45 < result.tuning_cents < -20  # synth di-detune -30 cents
    assert result.key == "C major"


def test_auto_tempo_close(twinkle):
    y, sr = twinkle
    tr = transcribe(y, TranscribeConfig(), sr=sr)
    assert abs(tr.tempo - 100) < 6


def test_transpose(twinkle):
    y, sr = twinkle
    tr = transcribe(y, TranscribeConfig(bpm=100, transpose=-12, grid=0), sr=sr)
    assert tr.notes[0].pitch == 48 and tr.notes[0].start_beat is None


def test_silence_gives_no_notes():
    sr = 16000
    y = 0.001 * np.random.default_rng(1).standard_normal(sr * 2).astype(np.float32)
    tr = transcribe(y, TranscribeConfig(bpm=120), sr=sr)
    assert tr.notes == []


def test_exports(result, tmp_path):
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(str(write_midi(result, tmp_path / "a.mid")))
    assert len(pm.instruments[0].notes) == len(TWINKLE)
    write_json(result, tmp_path / "a.json")
    assert (tmp_path / "a.json").read_text().count('"pitch"') == len(TWINKLE)
    pytest.importorskip("music21")
    xml = write_musicxml(result, tmp_path / "a.musicxml").read_text()
    assert xml.count("<pitch>") == len(TWINKLE)


def test_parse_key():
    assert parse_key("A minor") == (9, "minor")
    assert parse_key("F#m") == (6, "minor")
    assert parse_key("Bb") == (10, "major")
    assert parse_key("c major") == (0, "major")


def test_detect_key_minor():
    a_minor = [57, 59, 60, 62, 64, 65, 67, 69, 64, 57, 60, 57]
    notes = [Note(i * 0.5, i * 0.5 + 0.45, p) for i, p in enumerate(a_minor)]
    assert detect_key(notes) == (9, "minor")


def test_snap_to_scale():
    notes = [Note(0, 1, 61, cents=30), Note(1, 2, 61, cents=-30), Note(2, 3, 60)]
    snap_to_scale(notes, (0, "major"))
    assert [n.pitch for n in notes] == [62, 60, 60]


def test_quantize_resolves_overlaps():
    notes = [Note(0.0, 0.52, 60), Note(0.49, 1.0, 62), Note(0.51, 0.6, 64)]
    q = quantize(notes, tempo=120, grid=16)  # 1 step = 0.125 s
    for a, b in zip(q, q[1:]):
        assert a.end_beat <= b.start_beat
        assert a.start_beat < a.end_beat


def test_quantize_finds_grid_phase():
    # not pertama telat 40 ms, sisanya tepat di grid 120 BPM
    starts = [0.54, 1.0, 1.5, 2.0, 2.25]
    notes = [Note(s, s + 0.2, 60 + i) for i, s in enumerate(starts)]
    q = quantize(notes, tempo=120, grid=16)
    assert [n.start_beat for n in q] == [0.0, 1.0, 2.0, 3.0, 3.5]


def test_legato_scale_without_gaps():
    mel = [(64, .5), (66, .5), (67, .5), (69, .5), (71, 1), (69, .5), (67, .5), (66, 1), (64, 2)]
    y, sr = synth_hum(mel, bpm=120, gap_s=0.0, vibrato_cents=40, seed=3)
    tr = transcribe(y, TranscribeConfig(bpm=120), sr=sr)
    assert [n.pitch for n in tr.notes] == [p for p, _ in mel]
    assert tr.key == "E minor"


def test_cli_end_to_end(tmp_path):
    import soundfile as sf

    from neofunk.cli import main

    y, sr = synth_hum(TWINKLE[:4], bpm=100)
    wav = tmp_path / "hum.wav"
    sf.write(wav, y, sr)
    assert main(["transcribe", str(wav), "-o", str(tmp_path / "out"), "--bpm", "100", "--formats", "mid,json,wav"]) == 0
    for name in ("hum.mid", "hum.raw.mid", "hum.json", "hum.preview.wav"):
        assert (tmp_path / "out" / name).exists()


def test_crepe_backend(twinkle):
    pytest.importorskip("torchcrepe")
    y, sr = twinkle
    tr = transcribe(y, TranscribeConfig(bpm=100, backend="crepe"), sr=sr)
    assert [n.pitch for n in tr.notes] == [p for p, _ in TWINKLE]

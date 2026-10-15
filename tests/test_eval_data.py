import json
import os
import sys
import textwrap
import zipfile

import pytest
import soundfile as sf

from neofunk.datasets import load_humtrans, load_pairs
from neofunk.notes import Note, Transcription
from neofunk.synth import TWINKLE, melody_to_notes, synth_hum

mir_eval = pytest.importorskip("mir_eval")


def _write_midi(notes, path):
    from neofunk.export import write_midi

    write_midi(Transcription(notes=notes, tempo=120), path, quantized=False)


def test_note_metrics_perfect_and_octave():
    from neofunk.evaluate import note_metrics

    ref = melody_to_notes(TWINKLE)
    assert note_metrics(ref, ref)["note_f1"] == pytest.approx(1.0)
    low = [Note(n.start, n.end, n.pitch - 12) for n in ref]
    assert note_metrics(ref, low)["note_f1"] == 0.0
    assert note_metrics(ref, low, octave_invariant=True)["note_f1"] == pytest.approx(1.0)
    m = note_metrics(ref, [])
    assert m["note_f1"] == 0.0 and m["n_ref"] == len(ref)


def test_evaluate_on_synthetic_pairs(tmp_path):
    from neofunk.evaluate import evaluate, format_table, write_csv

    for i, seed in enumerate([0, 1]):
        y, sr = synth_hum(TWINKLE, bpm=100, seed=seed)
        sf.write(tmp_path / f"take{i}.wav", y, sr)
        _write_midi(melody_to_notes(TWINKLE, bpm=100), tmp_path / f"take{i}.mid")
    (tmp_path / "take0.raw.mid").write_bytes(b"")  # keluaran neofunk, harus diabaikan

    pairs = load_pairs(tmp_path)
    assert [p.name for p in pairs] == ["take0", "take1"]
    (res,) = evaluate(pairs, ["pyin"], progress=False)
    assert not res.errors
    assert res.mean("note_f1") > 0.9
    assert res.mean("pitch_acc") > 0.95
    assert "pyin" in format_table([res])
    assert write_csv([res], tmp_path / "r.csv").read_text().count("\n") == 3


def test_load_humtrans_layout(tmp_path):
    names = ["P1_M1_S1_R1", "P1_M1_S2_R1", "P2_M3_S1_R2_U"]
    (tmp_path / "wav" / "all_wav").mkdir(parents=True)
    (tmp_path / "midi" / "all_midi").mkdir(parents=True)
    for n in names:
        (tmp_path / "wav" / "all_wav" / f"{n}.wav").write_bytes(b"")
        (tmp_path / "midi" / "all_midi" / f"{n}.mid").write_bytes(b"")
    split = {"TRAIN": [names[0], names[1]], "VALID": [], "TEST": [names[2] + ".wav"]}
    (tmp_path / "train_valid_test_keys.json").write_text(json.dumps(split))
    assert [p.name for p in load_humtrans(tmp_path, "TEST")] == [names[2]]
    assert len(load_humtrans(tmp_path, "train")) == 2
    assert len(load_humtrans(tmp_path, None)) == 3


FAKE_ROSVOT = textwrap.dedent('''
    import argparse, pretty_midi, soundfile as sf
    from pathlib import Path
    p = argparse.ArgumentParser()
    p.add_argument("-o"); p.add_argument("-p"); p.add_argument("--thr", type=float)
    p.add_argument("--no_save_every_npy", action="store_true")
    p.add_argument("--no_save_final_npy", action="store_true")
    a = p.parse_args()
    y, sr = sf.read(a.p)
    assert sr == 24000, sr
    pm = pretty_midi.PrettyMIDI(); inst = pretty_midi.Instrument(0)
    inst.notes.append(pretty_midi.Note(120, 64, 0.5, 1.0))
    inst.notes.append(pretty_midi.Note(120, 60, 0.1, 0.5))
    pm.instruments.append(inst)
    Path(a.o, "midi").mkdir(parents=True)
    pm.write(str(Path(a.o, "midi", "output.mid")))
''')


@pytest.fixture
def fake_rosvot(tmp_path, monkeypatch):
    d = tmp_path / "ROSVOT"
    (d / "inference").mkdir(parents=True)
    (d / "inference" / "rosvot.py").write_text(FAKE_ROSVOT)
    monkeypatch.setenv("NEOFUNK_ROSVOT_DIR", str(d))
    return d


def test_rosvot_setup_from_zip_and_transcribe(fake_rosvot, tmp_path):
    from neofunk import TranscribeConfig, transcribe
    from neofunk.rosvot import missing_files, setup_rosvot

    assert "checkpoints/rosvot/model.pt" in missing_files()
    zpath = tmp_path / "checkpoints.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        for m in ("rosvot", "rwbd", "rmvpe"):
            z.writestr(f"checkpoints/{m}/model.pt", b"x")
    setup_rosvot(ckpt_zip=str(zpath))
    assert missing_files() == []

    y, sr = synth_hum(TWINKLE[:2])
    tr = transcribe(y, TranscribeConfig(backend="rosvot", bpm=120, grid=0, key=None), sr=sr)
    assert [(n.pitch, round(n.start, 2)) for n in tr.notes] == [(60, 0.1), (64, 0.5)]


def test_rosvot_not_setup_message(fake_rosvot):
    from neofunk.rosvot import transcribe_rosvot

    with pytest.raises(RuntimeError, match="neofunk setup rosvot"):
        transcribe_rosvot("x.wav")

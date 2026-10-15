"""Command line: `neofunk transcribe suara.wav` / `neofunk record`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from neofunk.export import write_json, write_midi, write_musicxml, write_preview_wav
from neofunk.pipeline import TranscribeConfig, transcribe
from neofunk.segment import SegmentConfig


def _cmd_transcribe(args: argparse.Namespace) -> int:
    src = Path(args.input)
    if not src.exists():
        print(f"File tidak ditemukan: {src}", file=sys.stderr)
        return 1
    out_dir = Path(args.output or src.parent / "output")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = src.stem

    cfg = TranscribeConfig(
        backend=args.backend,
        fmin=args.fmin,
        fmax=args.fmax,
        bpm=args.bpm,
        grid=args.grid,
        key=None if args.key == "none" else args.key,
        snap_to_scale=args.snap_scale,
        transpose=args.transpose,
        beats_per_bar=args.beats_per_bar,
        segment=SegmentConfig(min_note_s=args.min_note, onset_split=not args.no_onset_split),
        rosvot_threshold=args.rosvot_thr,
    )
    tr = transcribe(str(src), cfg)
    print(tr.pretty())

    formats = {f.strip() for f in args.formats.split(",") if f.strip()}
    written = []
    if "mid" in formats:
        written.append(write_midi(tr, out_dir / f"{stem}.mid", program=args.program))
        if args.grid:
            written.append(write_midi(tr, out_dir / f"{stem}.raw.mid", program=args.program, quantized=False))
    if "json" in formats:
        written.append(write_json(tr, out_dir / f"{stem}.json"))
    if "musicxml" in formats:
        try:
            written.append(write_musicxml(tr, out_dir / f"{stem}.musicxml", title=stem))
        except ImportError as e:
            print(f"[skip musicxml] {e}", file=sys.stderr)
    if "wav" in formats:
        written.append(write_preview_wav(tr, out_dir / f"{stem}.preview.wav"))
    print("\nFile hasil:")
    for p in written:
        print(f"  {p}")
    return 0


def _cmd_record(args: argparse.Namespace) -> int:
    try:
        import sounddevice as sd
        import soundfile as sf
    except ImportError:
        print("Perlu: pip install sounddevice", file=sys.stderr)
        return 1
    sr = 44100
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.bpm:
        print(f"Tip: nyalakan metronom {args.bpm} BPM di HP sambil merekam (pakai headset).")
    input("Tekan Enter untuk mulai merekam...")
    print(f"Merekam {args.seconds} detik... mulai bersenandung!")
    audio = sd.rec(int(args.seconds * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    sf.write(str(out), audio, sr)
    print(f"Tersimpan: {out}\nLanjut: neofunk transcribe {out}" + (f" --bpm {args.bpm}" if args.bpm else ""))
    return 0


def _cmd_setup(args: argparse.Namespace) -> int:
    from neofunk.rosvot import setup_rosvot

    setup_rosvot(force=args.force, ckpt_zip=args.ckpt_zip)
    return 0


def _cmd_data(args: argparse.Namespace) -> int:
    from neofunk.datasets import download_humtrans, load_humtrans

    root = download_humtrans(
        args.dir, with_audio=not args.no_audio, keep_zip=args.keep_zip, with_corrections=not args.no_corrections
    )
    if not args.no_audio:
        for sp in ("TRAIN", "VALID", "TEST"):
            print(f"  {sp:<5}: {len(load_humtrans(root, sp))} pasangan")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from neofunk.datasets import load_humtrans, load_pairs
    from neofunk.evaluate import evaluate, format_table, write_csv

    if args.humtrans:
        pairs = load_humtrans(args.humtrans, None if args.split == "ALL" else args.split)
    elif args.data:
        pairs = load_pairs(args.data)
    else:
        print("Isi --data FOLDER (pasangan nama.wav + nama.mid) atau --humtrans ROOT", file=sys.stderr)
        return 1
    if args.limit:
        pairs = pairs[: args.limit]
    if not pairs:
        print("Tidak ada pasangan audio+MIDI yang ditemukan.", file=sys.stderr)
        return 1
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    print(f"Benchmark {len(pairs)} file x {len(backends)} backend")
    results = evaluate(pairs, backends, octave_invariant=args.octave_invariant, onset_tol=args.onset_tol)
    print()
    print(format_table(results))
    if args.csv:
        print(f"\nDetail per file: {write_csv(results, args.csv)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="neofunk", description="Suara mulut -> not musik (MIDI / MusicXML)")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("transcribe", help="Transkripsi file audio jadi not")
    t.add_argument("input", help="file audio (wav/mp3/m4a/flac)")
    t.add_argument("-o", "--output", help="folder output (default: <folder input>/output)")
    t.add_argument("-b", "--backend", default="pyin", choices=["pyin", "crepe", "basic-pitch", "rosvot"])
    t.add_argument("--bpm", type=float, help="tempo; kalau kosong diestimasi otomatis")
    t.add_argument("--grid", type=int, default=16, help="kuantisasi: 4/8/16/32, 0 = mati (default 16)")
    t.add_argument("--key", default="auto", help="'auto', 'none', atau mis. 'A minor', 'C', 'F#m'")
    t.add_argument("--snap-scale", action="store_true", help="paksa not masuk tangga nada key")
    t.add_argument("--transpose", type=int, default=0, help="geser semitone, mis. 12 = naik 1 oktaf")
    t.add_argument("--beats-per-bar", type=int, default=4)
    t.add_argument("--min-note", type=float, default=0.07, help="durasi not minimum (detik)")
    t.add_argument("--no-onset-split", action="store_true", help="jangan pecah not berdasarkan onset")
    t.add_argument("--fmin", type=float, default=65.0)
    t.add_argument("--fmax", type=float, default=1050.0)
    t.add_argument("--program", type=int, default=0, help="General MIDI program (0 piano, 29 overdrive gtr, 81 saw lead)")
    t.add_argument("--formats", default="mid,json,musicxml,wav", help="daftar dipisah koma: mid,json,musicxml,wav")
    t.add_argument("--rosvot-thr", type=float, default=0.85, help="ambang batas not ROSVOT (kecil = lebih banyak not)")
    t.set_defaults(func=_cmd_transcribe)

    r = sub.add_parser("record", help="Rekam dari mikrofon ke WAV")
    r.add_argument("-o", "--output", default="recordings/take.wav")
    r.add_argument("-s", "--seconds", type=float, default=15.0)
    r.add_argument("--bpm", type=float)
    r.set_defaults(func=_cmd_record)

    su = sub.add_parser("setup", help="Pasang model eksternal (ROSVOT)")
    su.add_argument("model", choices=["rosvot"])
    su.add_argument("--force", action="store_true", help="hapus & pasang ulang")
    su.add_argument("--ckpt-zip", help="path checkpoints.zip kalau download otomatis gagal")
    su.set_defaults(func=_cmd_setup)

    d = sub.add_parser("data", help="Download dataset untuk evaluasi / fine-tuning")
    d.add_argument("name", choices=["humtrans"])
    d.add_argument("--dir", default="data/humtrans")
    d.add_argument("--no-audio", action="store_true", help="hanya label MIDI + split (±6 MB), tanpa audio 14.7 GB")
    d.add_argument("--keep-zip", action="store_true", help="jangan hapus zip setelah ekstrak")
    d.add_argument("--no-corrections", action="store_true", help="jangan clone label koreksi Dynamic HumTrans")
    d.set_defaults(func=_cmd_data)

    e = sub.add_parser("eval", help="Benchmark akurasi backend (mir_eval)")
    e.add_argument("--data", help="folder pasangan nama.wav + nama.mid (rekamanmu sendiri)")
    e.add_argument("--humtrans", help="root dataset HumTrans (hasil `neofunk data humtrans`)")
    e.add_argument("--split", default="TEST", choices=["TRAIN", "VALID", "TEST", "ALL"])
    e.add_argument("--limit", type=int, help="ambil N file pertama saja (untuk tes cepat)")
    e.add_argument("--backends", default="pyin,crepe", help="mis. pyin,crepe,rosvot,basic-pitch")
    e.add_argument("--octave-invariant", action="store_true", help="abaikan salah oktaf (umum di humming)")
    e.add_argument("--onset-tol", type=float, default=0.05, help="toleransi onset (detik)")
    e.add_argument("--csv", help="simpan hasil per file ke CSV")
    e.set_defaults(func=_cmd_eval)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

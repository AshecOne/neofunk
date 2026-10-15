"""Dataset pasangan (audio, MIDI) untuk evaluasi dan fine-tuning.

- HumTrans (Liu et al., 2023): ±56 jam humming + label MIDI, split resmi TRAIN/VALID/TEST.
  Lisensi CC-BY-NC-4.0 (non-komersial). Sumber: huggingface.co/datasets/dadinghh2/HumTrans
  Ukuran: all_wav.zip ±14.7 GB (+ ±15 GB setelah diekstrak), all_midi.zip ±5.5 MB.
- Dynamic HumTrans (Gupta et al., 2024): koreksi label onset/offset HumTrans.
  Sumber: github.com/shubham-gupta-30/humming_transcription
- Folder pasangan milikmu sendiri: `nama.wav` + `nama.mid` dengan nama yang sama.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path

HUMTRANS_REPO = "dadinghh2/HumTrans"
HUMTRANS_FILES = {"midi": "all_midi.zip", "wav": "all_wav.zip", "split": "train_valid_test_keys.json"}
DYNAMIC_HUMTRANS_REPO = "https://github.com/shubham-gupta-30/humming_transcription.git"
AUDIO_EXTS = (".wav", ".flac", ".mp3", ".m4a", ".ogg")


@dataclass
class Pair:
    name: str
    audio: Path
    midi: Path
    split: str | None = None


def _hf_download(filename: str, root: Path) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError("Butuh huggingface_hub: pip install -e '.[data]'") from e
    return Path(hf_hub_download(HUMTRANS_REPO, filename, repo_type="dataset", local_dir=str(root)))


def _extract(zpath: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[humtrans] ekstrak {zpath.name} -> {dest}")
    with zipfile.ZipFile(zpath) as z:
        z.extractall(dest)


def download_humtrans(
    root: str | Path = "data/humtrans",
    with_audio: bool = True,
    keep_zip: bool = False,
    with_corrections: bool = True,
) -> Path:
    """Download HumTrans ke `root`. Hasil: root/midi, root/wav, root/train_valid_test_keys.json."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    _hf_download(HUMTRANS_FILES["split"], root)
    if not any((root / "midi").rglob("*.mid")):
        z = _hf_download(HUMTRANS_FILES["midi"], root)
        _extract(z, root / "midi")
        if not keep_zip:
            z.unlink()

    if with_audio and not any((root / "wav").rglob("*.wav")):
        need = 32 * 1024**3
        free = shutil.disk_usage(root).free
        if free < need:
            raise RuntimeError(
                f"Ruang disk kurang: butuh ±32 GB (zip + ekstrak), tersedia {free / 1024**3:.1f} GB. "
                "Pakai --no-audio dulu kalau hanya mau lihat label MIDI."
            )
        print("[humtrans] download all_wav.zip (±14.7 GB), ini akan lama ...")
        z = _hf_download(HUMTRANS_FILES["wav"], root)
        _extract(z, root / "wav")
        if not keep_zip:
            z.unlink()

    if with_corrections and not (root / "dynamic_humtrans").exists():
        print("[humtrans] clone label koreksi Dynamic HumTrans")
        subprocess.run(["git", "clone", "--depth", "1", DYNAMIC_HUMTRANS_REPO, str(root / "dynamic_humtrans")], check=True)

    print(f"[humtrans] selesai: {root}")
    return root


def _index(folder: Path, exts: tuple[str, ...]) -> dict[str, Path]:
    return {p.stem: p for p in folder.rglob("*") if p.suffix.lower() in exts}


def load_humtrans(root: str | Path = "data/humtrans", split: str | None = "TEST") -> list[Pair]:
    """Daftar pasangan HumTrans. `split` = TRAIN / VALID / TEST / None (semua)."""
    root = Path(root)
    audio = _index(root / "wav", AUDIO_EXTS)
    midi = _index(root / "midi", (".mid", ".midi"))
    split_of: dict[str, str] = {}
    split_file = root / HUMTRANS_FILES["split"]
    if split_file.exists():
        for sp, keys in json.loads(split_file.read_text()).items():
            for k in keys:
                split_of[Path(str(k)).stem] = sp.upper()
    pairs = [
        Pair(name, audio[name], midi[name], split_of.get(name))
        for name in sorted(audio.keys() & midi.keys())
    ]
    if split:
        pairs = [p for p in pairs if p.split == split.upper()]
    return pairs


def load_pairs(folder: str | Path) -> list[Pair]:
    """Folder berisi `nama.wav` + `nama.mid` (boleh di subfolder)."""
    folder = Path(folder)
    audio = _index(folder, AUDIO_EXTS)
    midi = _index(folder, (".mid", ".midi"))
    # abaikan file keluaran neofunk sendiri (nama.raw.mid / nama.preview.wav)
    midi = {k: v for k, v in midi.items() if not k.endswith(".raw")}
    return [Pair(n, audio[n], midi[n]) for n in sorted(audio.keys() & midi.keys())]

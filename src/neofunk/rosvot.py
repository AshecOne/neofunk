"""Backend ROSVOT (ACL 2024): model note-level khusus suara nyanyi.

ROSVOT adalah codebase riset (bukan package pip), jadi neofunk:
1. meng-clone repo-nya ke `third_party/ROSVOT` (commit di-pin), dan
2. men-download checkpoint resminya (Google Drive) ke `third_party/ROSVOT/checkpoints`,
lalu memanggil `inference/rosvot.py` sebagai subprocess dan membaca MIDI hasilnya.

Setup sekali:   neofunk setup rosvot
Pakai:          neofunk transcribe suara.wav --backend rosvot

Catatan: inference ROSVOT di-hardcode ke CUDA, jadi butuh GPU NVIDIA.
Lokasi bisa diubah via env NEOFUNK_ROSVOT_DIR, interpreter via NEOFUNK_ROSVOT_PYTHON
(berguna kalau ROSVOT dipasang di conda env terpisah).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from neofunk.notes import Note

ROSVOT_REPO = "https://github.com/RickyL-2000/ROSVOT.git"
ROSVOT_COMMIT = "3c8332bf43adae35f6e4d64971862f2f6139b310"
ROSVOT_CKPT_GDRIVE_ID = "1JNtNT37KiLq9uFQqHk7JFs-3trxd3bRh"
ROSVOT_SR = 24000  # sample rate model (configs/rosvot.yaml)
REQUIRED_CKPTS = ("checkpoints/rosvot/model.pt", "checkpoints/rwbd/model.pt", "checkpoints/rmvpe/model.pt")


def project_root() -> Path:
    # src/neofunk/rosvot.py -> root repo (untuk instalasi `pip install -e .`)
    return Path(__file__).resolve().parents[2]


def rosvot_dir() -> Path:
    env = os.environ.get("NEOFUNK_ROSVOT_DIR")
    if env:
        return Path(env).expanduser()
    root = project_root()
    if (root / "pyproject.toml").exists():
        return root / "third_party" / "ROSVOT"
    return Path.home() / ".cache" / "neofunk" / "ROSVOT"


def missing_files(d: Path | None = None) -> list[str]:
    d = d or rosvot_dir()
    files = ["inference/rosvot.py", *REQUIRED_CKPTS]
    return [f for f in files if not (d / f).exists()]


def _find_ckpt_root(extracted: Path) -> Path | None:
    for p in extracted.rglob("rosvot"):
        if p.is_dir() and (p / "model.pt").exists() and p.parent.name == "checkpoints":
            return p.parent
    return None


def setup_rosvot(force: bool = False, ckpt_zip: str | None = None) -> Path:
    """Clone ROSVOT + download checkpoint. `ckpt_zip` = path checkpoints.zip yang sudah
    kamu download manual (kalau download otomatis dari Google Drive gagal)."""
    d = rosvot_dir()
    if force and d.exists():
        shutil.rmtree(d)

    if not (d / "inference" / "rosvot.py").exists():
        d.parent.mkdir(parents=True, exist_ok=True)
        print(f"[rosvot] clone {ROSVOT_REPO} -> {d}")
        subprocess.run(["git", "clone", ROSVOT_REPO, str(d)], check=True)
        subprocess.run(["git", "-C", str(d), "checkout", "-q", ROSVOT_COMMIT], check=True)

    if any(not (d / f).exists() for f in REQUIRED_CKPTS):
        with tempfile.TemporaryDirectory() as tmp:
            zpath = Path(ckpt_zip) if ckpt_zip else Path(tmp) / "checkpoints.zip"
            if not ckpt_zip:
                try:
                    import gdown
                except ImportError as e:
                    raise ImportError("Butuh gdown: pip install -e '.[rosvot]'") from e
                print("[rosvot] download checkpoint dari Google Drive ...")
                out = gdown.download(id=ROSVOT_CKPT_GDRIVE_ID, output=str(zpath), quiet=False)
                if not out:
                    raise RuntimeError(
                        "Download gagal. Download manual checkpoints.zip dari\n"
                        f"  https://drive.google.com/file/d/{ROSVOT_CKPT_GDRIVE_ID}/view\n"
                        "lalu jalankan: neofunk setup rosvot --ckpt-zip path/ke/checkpoints.zip"
                    )
            extract = Path(tmp) / "x"
            with zipfile.ZipFile(zpath) as z:
                z.extractall(extract)
            src = _find_ckpt_root(extract)
            if src is None:
                raise RuntimeError("Isi zip tidak dikenali (folder checkpoints/rosvot/model.pt tidak ada)")
            shutil.copytree(src, d / "checkpoints", dirs_exist_ok=True)

    miss = missing_files(d)
    if miss:
        raise RuntimeError(f"Setup ROSVOT belum lengkap, file hilang: {miss}")
    print(f"[rosvot] siap di {d}")
    return d


def transcribe_rosvot(
    source: str | np.ndarray,
    sr: int | None = None,
    threshold: float = 0.85,
    python: str | None = None,
    verbose: bool = False,
) -> list[Note]:
    """Jalankan ROSVOT pada satu file/array audio dan kembalikan daftar Note."""
    import pretty_midi
    import soundfile as sf

    d = rosvot_dir()
    miss = missing_files(d)
    if miss:
        raise RuntimeError(f"ROSVOT belum di-setup ({d}). Jalankan dulu: neofunk setup rosvot")

    import librosa

    if isinstance(source, str):
        y, _ = librosa.load(source, sr=ROSVOT_SR, mono=True)
    else:
        if sr is None:
            raise ValueError("sr wajib diisi kalau source berupa array")
        y = librosa.resample(np.asarray(source, dtype=np.float32), orig_sr=sr, target_sr=ROSVOT_SR)

    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "input.wav"
        sf.write(wav, y, ROSVOT_SR, subtype="PCM_16")
        out_dir = Path(tmp) / "out"
        cmd = [
            python or os.environ.get("NEOFUNK_ROSVOT_PYTHON") or sys.executable,
            "inference/rosvot.py",
            "-o", str(out_dir),
            "-p", str(wav),
            "--thr", str(threshold),
            "--no_save_every_npy",
            "--no_save_final_npy",
        ]
        env = {**os.environ, "PYTHONPATH": str(d)}
        proc = subprocess.run(cmd, cwd=d, env=env, capture_output=True, text=True)
        if verbose:
            print(proc.stdout, proc.stderr, file=sys.stderr)
        if proc.returncode != 0:
            tail = "\n".join((proc.stderr or proc.stdout).strip().splitlines()[-15:])
            raise RuntimeError(f"ROSVOT gagal (exit {proc.returncode}):\n{tail}")
        midi_path = out_dir / "midi" / "output.mid"
        if not midi_path.exists():  # ROSVOT tidak menulis file kalau tidak ada not
            return []
        pm = pretty_midi.PrettyMIDI(str(midi_path))

    notes = [
        Note(start=float(n.start), end=float(n.end), pitch=int(n.pitch), velocity=100)
        for inst in pm.instruments
        for n in inst.notes
    ]
    notes.sort(key=lambda n: n.start)
    return notes

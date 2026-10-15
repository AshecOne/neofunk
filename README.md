# neofunk 🎤 → 🎼

**Ubah suara mulut jadi not musik.** Senandungkan melodi, nyanyikan "na-na-na", atau
"solo gitar" pakai mulut, lalu neofunk mengubahnya jadi **MIDI** (buat DAW) dan
**MusicXML** (partitur, bisa dibuka di MuseScore). Dari MIDI itu kamu bisa lanjut bikin
drum, chord, bass, dan isian lain, entah manual atau pakai AI.

Jalan **100% lokal**. Backend default (pYIN) cukup pakai CPU. Backend `crepe` memakai
GPU, dan RTX 4060 sudah lebih dari cukup: modelnya cuma beberapa MB dan prosesnya jauh
lebih cepat dari real-time.

```
suara.wav ──► pra-proses ──► pitch tracking ──► segmentasi not ──► tuning/key/tempo ──► kuantisasi ──► MIDI / MusicXML / JSON / preview.wav
              (high-pass,    (pYIN / CREPE /    (vibrato & napas   (Krumhansl,          (grid 1/16,
               normalisasi)   Basic Pitch)        tidak memecah not) offset tuning)       fase otomatis)
```

## Instalasi

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[score,dev]"      # inti + ekspor partitur

# opsional: backend neural pakai GPU (pasang PyTorch CUDA dulu sesuai pytorch.org)
pip install -e ".[gpu]"
# opsional: ROSVOT (model khusus suara nyanyi, butuh GPU NVIDIA) — lihat bagian ROSVOT
pip install -e ".[rosvot]" && neofunk setup rosvot
# opsional: Spotify Basic Pitch
pip install -e ".[basicpitch]"
# opsional: download dataset & benchmark
pip install -e ".[data,eval]"
# opsional: rekam langsung dari mic
pip install -e ".[record]"

# atau semuanya sekaligus
pip install -e ".[all,dev]"
```

## Pemakaian cepat

```bash
# 1. rekam (atau pakai rekaman HP: wav/mp3/m4a semua bisa)
neofunk record -o recordings/ide1.wav -s 20 --bpm 100

# 2. transkripsi
neofunk transcribe recordings/ide1.wav --bpm 100
neofunk transcribe recordings/ide1.wav --backend crepe --key "A minor" --snap-scale
```

Contoh output:

```
Tempo: 100.0 BPM | Key: C major | Tuning offset: -34 cents | 14 not
  C4     0.288s -   0.960s  vel 102  beat   0.00-  1.00
  C4     0.960s -   1.472s  vel 103  beat   1.00-  2.00
  G4     1.504s -   2.144s  vel  99  beat   2.00-  3.00
  ...
File hasil:
  recordings/output/ide1.mid          <- sudah dikuantisasi, siap drag ke DAW
  recordings/output/ide1.raw.mid      <- timing asli (lebih "manusiawi")
  recordings/output/ide1.json
  recordings/output/ide1.musicxml     <- buka di MuseScore untuk lihat partitur
  recordings/output/ide1.preview.wav  <- dengar cepat hasilnya
```

Belum punya rekaman? Bikin humming sintetis dulu buat coba-coba:

```bash
python examples/make_synthetic_hum.py examples/twinkle.wav
neofunk transcribe examples/twinkle.wav --bpm 100
```

Dari Python:

```python
from neofunk import transcribe, TranscribeConfig
from neofunk.export import write_midi

tr = transcribe("ide1.wav", TranscribeConfig(backend="crepe", bpm=100, key="auto"))
print(tr.pretty())
write_midi(tr, "ide1.mid", program=29)  # 29 = overdrive guitar
```

### Opsi penting

| Opsi | Fungsi |
|---|---|
| `--backend pyin\|crepe\|basic-pitch\|rosvot` | pYIN (CPU, default), CREPE (GPU, tahan noise), Basic Pitch (Spotify), ROSVOT (GPU, model note-level khusus nyanyi) |
| `--bpm 100` | Tempo. Kalau dikosongkan akan diestimasi, tapi **lebih akurat kalau kamu rekam pakai metronom** |
| `--grid 16` | Kuantisasi ke not 1/16 (`8` = 1/8, `32` = 1/32, `0` = mati) |
| `--key auto` / `--key "F#m"` / `--key none` | Deteksi atau tetapkan key |
| `--snap-scale` | Paksa not "fals" masuk ke tangga nada |
| `--transpose 12` | Geser oktaf. Berguna kalau suaramu rendah tapi mau jadi lead gitar |
| `--min-note 0.07` | Not lebih pendek dari ini dibuang (naikkan kalau terlalu banyak not kecil) |
| `--no-onset-split` | Jangan pecah not berdasarkan serangan konsonan |
| `--program 29` | Instrumen General MIDI di file MIDI |

## ROSVOT (model note-level, ACL 2024)

[ROSVOT](https://github.com/RickyL-2000/ROSVOT) adalah model yang langsung memprediksi
**batas not + pitch** dari suara nyanyi. Jadi bagian yang paling susah (memotong suara jadi
not) dikerjakan model terlatih, bukan aturan buatan. Ukurannya kecil (±12 M parameter
untuk model not + ±10 M untuk model batas kata, plus RMVPE untuk pitch), jadi ringan di RTX 4060.

```bash
pip install -e ".[rosvot]"
neofunk setup rosvot              # clone repo ROSVOT ke third_party/ + download checkpoint
neofunk transcribe ide1.wav --backend rosvot --bpm 100
neofunk transcribe ide1.wav --backend rosvot --rosvot-thr 0.7   # lebih banyak not kecil
```

Catatan:
- Inference ROSVOT **butuh GPU NVIDIA (CUDA)**; tidak bisa jalan di CPU.
- Checkpoint resminya dilatih dengan dataset nyanyi Mandarin (M4Singer). Penulisnya bilang
  modelnya tetap bagus untuk bahasa lain, tapi untuk humming/"solo gitar" mulut, **cek dengan
  benchmark** (lihat di bawah) dan bandingkan dengan `crepe`.
- Kalau `gdown` gagal download dari Google Drive, download manual
  [checkpoints.zip](https://drive.google.com/file/d/1JNtNT37KiLq9uFQqHk7JFs-3trxd3bRh/view)
  lalu `neofunk setup rosvot --ckpt-zip ~/Downloads/checkpoints.zip`.
- Kalau mau ROSVOT di conda env terpisah: set `NEOFUNK_ROSVOT_PYTHON=/path/ke/python`.

## Benchmark: seberapa akurat untuk suaramu?

`neofunk eval` membandingkan hasil transkripsi dengan "kunci jawaban" (MIDI yang benar)
memakai [`mir_eval`](https://github.com/mir-evaluation/mir_eval), metrik standar riset.

```bash
# A. rekamanmu sendiri: taruh pasangan file dengan nama sama di satu folder
#    my_eval/lagu1.wav + my_eval/lagu1.mid, my_eval/lagu2.wav + my_eval/lagu2.mid, ...
neofunk eval --data my_eval --backends pyin,crepe,rosvot --csv hasil/eval_saya.csv

# B. dataset publik HumTrans (lihat docs/DATASETS.md)
neofunk eval --humtrans data/humtrans --split TEST --limit 100 \
    --backends pyin,crepe,rosvot --octave-invariant
```

Contoh output:

```
backend       note F1  F1+offset  onset F1  pitch acc  file    waktu
--------------------------------------------------------------------
pyin            ...%       ...%      ...%       ...%    100    ...s
crepe           ...%       ...%      ...%       ...%    100    ...s
rosvot          ...%       ...%      ...%       ...%    100    ...s
```

- **note F1**: metrik utama. Not dihitung benar kalau awalnya meleset ≤ 50 ms dan nadanya ≤ ½ semitone.
- **F1+offset**: sama, tapi akhir not juga harus pas (lebih ketat).
- **onset F1**: hanya cek "pemotongan" not (nada diabaikan).
- **pitch acc**: dari not yang awalnya cocok, berapa persen nadanya benar.
- `--octave-invariant`: abaikan salah oktaf (wajar kalau kamu humming 1 oktaf di bawah lagu aslinya).

Cara gampang bikin kunci jawaban untuk rekamanmu: jalankan `neofunk transcribe` → buka
`.raw.mid` di DAW/MuseScore → betulkan not yang salah → simpan sebagai `nama.mid`
di sebelah `nama.wav`.

## Tips merekam supaya hasilnya bagus

1. **Pakai suku kata yang jelas**: "da-da-da", "na-na", "ta-ta" lebih mudah dipisah jadi
   not daripada "mmmmm" yang menyambung terus (terutama untuk nada yang diulang).
2. **Rekam pakai metronom** (lewat headset supaya klik tidak ikut terekam), lalu isi `--bpm`.
3. Ruangan tenang, mic jangan terlalu dekat (hindari "pop" dan clipping).
4. Untuk "solo gitar" yang cepat, coba `--backend crepe --grid 32 --min-note 0.05`.
5. Bend & vibrato tidak masalah: pitch dihaluskan dan offset tuning global dikoreksi otomatis.

## Struktur proyek

```
src/neofunk/
  audio.py     load & pra-proses audio
  pitch.py     pitch tracking: pYIN (librosa) & CREPE (torchcrepe)
  segment.py   kurva pitch -> not (inti heuristiknya di sini)
  music.py     tempo, deteksi key, snap tangga nada, kuantisasi
  export.py    MIDI, MusicXML, JSON, preview WAV
  pipeline.py  merangkai semuanya + backend Basic Pitch
  rosvot.py    setup & wrapper ROSVOT
  datasets.py  download HumTrans + loader pasangan (audio, MIDI)
  evaluate.py  benchmark mir_eval
  cli.py       command line `neofunk`
  synth.py     humming sintetis untuk tes
tests/         pytest (pakai audio sintetis, tanpa file rekaman)
docs/RESEARCH.md  referensi, perbandingan model, dan roadmap
docs/DATASETS.md  dataset untuk benchmark & fine-tuning
```

Jalankan tes: `pytest`

## Roadmap

Lihat [docs/RESEARCH.md](docs/RESEARCH.md) untuk detail dan referensinya.

- [x] **Fase 1**: pipeline tanpa training (pitch tracker pretrained + segmentasi heuristik)
- [x] **Fase 2**: integrasi model note-level khusus suara (ROSVOT, sudah termasuk RMVPE)
- [x] **Fase 3**: tooling evaluasi (`neofunk eval`, `mir_eval`) + download HumTrans
- [ ] **Fase 3b**: jalankan benchmark di HumTrans & rekamanmu, pilih backend default terbaik
- [ ] SOME sebagai backend tambahan
- [ ] **Fase 4**: fine-tune model kecil dengan suaramu sendiri (muat di 16 GB VRAM)
- [ ] **Fase 5**: dari melodi ke lagu: auto-chord, bassline, drum (MIDI → AI/DAW)
- [ ] UI sederhana (Gradio): rekam → lihat piano roll → edit → download
